"""LPSE inference: CLI REPL, one-shot --query, or HTTP server with --serve."""

import argparse
import json
import logging
import os
import threading
import time
from collections import deque

import torch

from lpse.collectors import get_node_ips, get_num_cores_from_node_exporter
from lpse.lpse import (
    ANSWER_DIM,
    FUSION_DIM,
    LATENT_DIM,
    LPSE,
)
from lpse.prometheus_ingest import (
    CLUSTER_TOKEN_DIM,
    RunningNormalizer,
    cluster_snapshot_to_token,
    collect_full_cluster_snapshot,
    compute_semantic_features,
)
from lpse.qa_generator import (
    ANSWER_TEMPLATES,
    fill_answer_template,
)
from lpse.question_encoder import (
    QUESTION_DIM,
    build_question_encoder,
    encode_questions,
    parse_question,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("infer_lpse")

CHECKPOINT_PATH = os.path.abspath(
    os.getenv("CHECKPOINT_PATH", "artifacts/checkpoints/lpse2.pt")
)
MODEL_VARIANT = "answer_jepa_v1"
SEQ_LEN = int(os.getenv("SEQ_LEN", "16"))
SCRAPE_INTERVAL = float(os.getenv("SCRAPE_INTERVAL", "1.0"))
OOD_THRESHOLD = float(os.getenv("OOD_THRESHOLD", "0.3"))
Q_ENCODER_BACKEND = os.getenv("Q_ENCODER_BACKEND", "")
CODEBOOK_TAU = float(os.getenv("CODEBOOK_TAU", "0.07"))
UPDATE_NORMALIZER = os.getenv("UPDATE_NORMALIZER", "0").lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def load_model(checkpoint_path: str):
    """Load LPSE model and metadata from checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    stored_node_ips = ckpt.get("node_ips", [])
    stored_node_cores = ckpt.get("node_cores", [])
    stored_answers = ckpt.get("answer_templates", ANSWER_TEMPLATES)
    model_config = ckpt.get("model_config", {})
    ckpt_variant = ckpt.get("model_variant", model_config.get("model_variant"))
    if ckpt_variant not in (MODEL_VARIANT, "answer_jepa_offline_v1"):
        raise ValueError(
            f"Checkpoint model_variant={ckpt_variant!r} does not match current "
            f"model_variant={MODEL_VARIANT!r}. Retrain before inference."
        )
    codebook_tau = float(
        model_config.get(
            "codebook_temperature", ckpt.get("codebook_temperature", CODEBOOK_TAU)
        )
    )
    q_encoder_backend = Q_ENCODER_BACKEND or ckpt.get(
        "q_encoder_backend",
        model_config.get("q_encoder_backend", "sentence_transformer"),
    )

    num_nodes = len(stored_node_ips)
    input_dim = int(
        model_config.get("input_dim", ckpt.get("input_dim", CLUSTER_TOKEN_DIM))
    )
    if input_dim != CLUSTER_TOKEN_DIM:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} does not match current token_dim={CLUSTER_TOKEN_DIM}. "
            "Retrain with the identity-preserving node token before inference."
        )
    latent_dim = int(model_config.get("latent_dim", LATENT_DIM))
    fusion_dim = int(model_config.get("fusion_dim", FUSION_DIM))
    answer_dim = int(model_config.get("answer_dim", ANSWER_DIM))
    use_structured = bool(model_config.get("use_structured", True))
    use_token_sequence = bool(model_config.get("use_token_sequence", False))

    ckpt_model_sd = ckpt.get("model", {})
    proj_key = "question_encoder.proj.0.weight"
    if proj_key in ckpt_model_sd and "use_structured" not in model_config:
        proj_in = int(ckpt_model_sd[proj_key].shape[1])

        if proj_in == 416:
            use_structured = False
            logger.info(
                "Inferred use_structured=False from checkpoint proj dim %d", proj_in
            )
        elif proj_in == 496:
            use_structured = True
            logger.info(
                "Inferred use_structured=True from checkpoint proj dim %d", proj_in
            )
    if (
        "token_proj.weight" in ckpt_model_sd
        and "use_token_sequence" not in model_config
    ):
        use_token_sequence = True
        logger.info("Inferred use_token_sequence=True from checkpoint token_proj")

    question_encoder = build_question_encoder(
        backend=q_encoder_backend,
        out_dim=QUESTION_DIM,
        use_structured=use_structured,
    ).to(DEVICE)

    model = LPSE(
        input_dim=input_dim,
        num_answers=len(stored_answers),
        question_encoder=question_encoder,
        latent_dim=latent_dim,
        fusion_dim=fusion_dim,
        answer_dim=answer_dim,
        encoder_heads=int(model_config.get("encoder_heads", 8)),
        encoder_layers=int(model_config.get("encoder_layers", 4)),
        fusion_heads=int(model_config.get("fusion_heads", 16)),
        fusion_layers=int(model_config.get("fusion_layers", 4)),
        dropout=float(model_config.get("dropout", 0.1)),
        codebook_temperature=codebook_tau,
        use_token_sequence=use_token_sequence,
    ).to(DEVICE)

    current_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in ckpt["model"].items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    current_state.update(compatible)
    model.load_state_dict(current_state)
    if skipped:
        logger.warning(
            "Skipping %d incompatible checkpoint keys (model may be corrupted / randomly "
            "initialized): %s",
            len(skipped),
            skipped[:10],
        )
    model.eval()

    normalizer = RunningNormalizer(input_dim)
    if "normalizer" in ckpt:
        try:
            normalizer.load_state_dict(ckpt["normalizer"])
        except Exception as e:
            logger.info("Skipping incompatible normalizer state: %s", e)

    logger.info(
        "Loaded model from %s (step %d, %d checkpoint nodes, %d answers, token_dim=%d, q_encoder=%s)",
        checkpoint_path,
        ckpt.get("step", 0),
        num_nodes,
        len(stored_answers),
        input_dim,
        q_encoder_backend,
    )

    return (
        model,
        question_encoder,
        normalizer,
        stored_node_ips,
        stored_node_cores,
        stored_answers,
    )


class MetricContext:
    """
    Continuously scrapes cluster metrics in a background thread,
    maintaining a rolling temporal window for the model.
    """

    def __init__(
        self,
        node_ips: list[str],
        normalizer: RunningNormalizer,
        node_cores: list[int] | None = None,
        seq_len: int = SEQ_LEN,
        interval: float = SCRAPE_INTERVAL,
        update_normalizer: bool = UPDATE_NORMALIZER,
    ):
        self.node_ips = node_ips
        self.normalizer = normalizer
        self.seq_len = seq_len
        self.interval = interval
        self.update_normalizer = update_normalizer
        self.buffer = deque(maxlen=seq_len)
        self.latest_semantic = {}
        self.latest_raw = {}
        self.node_cores: list[int] = list(node_cores or [])
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._scrape_loop, daemon=True)
        self._thread.start()
        logger.info("Background scraper started (interval=%.1fs)", self.interval)

    def stop(self):
        self._running = False

    def _scrape_loop(self):
        while self._running:
            try:
                live_node_ips = get_node_ips()
                if not live_node_ips:
                    time.sleep(self.interval)
                    continue

                live_node_cores: list[int] = []
                for ip in live_node_ips:
                    cores = get_num_cores_from_node_exporter(ip)
                    if cores <= 0:
                        cores = 10
                    live_node_cores.append(cores)

                node_tensors, raw_parsed = collect_full_cluster_snapshot(live_node_ips)
                cluster_token = cluster_snapshot_to_token(node_tensors, live_node_ips)
                if self.update_normalizer:
                    self.normalizer.update(cluster_token)
                token_norm = self.normalizer.normalize(cluster_token)

                semantic = compute_semantic_features(raw_parsed, live_node_ips)

                with self._lock:
                    self.node_ips = live_node_ips
                    self.node_cores = live_node_cores
                    self.buffer.append(token_norm)
                    self.latest_semantic = semantic
                    self.latest_raw = raw_parsed
            except Exception as e:
                logger.warning("Scrape failed: %s", e)

            time.sleep(self.interval)

    def get_context(self) -> torch.Tensor | None:
        """Return [1, T, F] context tensor if buffer is full."""
        with self._lock:
            if len(self.buffer) < self.seq_len:
                return None
            seq = torch.stack(list(self.buffer), dim=0)
            return seq.unsqueeze(0)

    def get_semantic(self) -> dict:
        with self._lock:
            return dict(self.latest_semantic)

    def get_nodes(self) -> tuple[list[str], list[int]]:
        with self._lock:
            return list(self.node_ips), list(self.node_cores)

    @property
    def ready(self) -> bool:
        with self._lock:
            return len(self.buffer) >= self.seq_len


class LPSEInference:
    """
    High-level inference interface.

    Usage:
        engine = LPSEInference.from_checkpoint("checkpoints/lpse2.pt")
        engine.start()
        result = engine.ask("Is any node CPU-bound?")
        print(result["answer"])
    """

    def __init__(
        self,
        model: LPSE,
        question_encoder,
        normalizer: RunningNormalizer,
        node_ips: list[str],
        node_cores: list[int],
        answer_templates: list[str],
    ):
        self.model = model
        self.question_encoder = question_encoder
        self.normalizer = normalizer
        self.node_ips = node_ips
        self.node_cores = node_cores
        self.answer_templates = answer_templates
        self.context = MetricContext(node_ips, normalizer, node_cores=node_cores)

    @classmethod
    def from_checkpoint(
        cls,
        path: str = CHECKPOINT_PATH,
        update_normalizer: bool = UPDATE_NORMALIZER,
    ) -> "LPSEInference":
        model, q_enc, norm, ips, cores, answers = load_model(path)
        engine = cls(model, q_enc, norm, ips, cores, answers)
        engine.context.update_normalizer = update_normalizer
        return engine

    def start(self):
        """Start background metric scraping."""
        self.context.start()
        logger.info("Waiting for context buffer to fill (%d samples)...", SEQ_LEN)
        while not self.context.ready:
            time.sleep(0.5)
        logger.info("Context ready. You can now ask questions.")

    def stop(self):
        self.context.stop()

    def ask(
        self,
        question: str,
        top_k: int = 3,
    ) -> dict:
        """
        Ask a question about the current cluster state.

        Returns:
            {
                "question": str,
                "answer": str,
                "confidence": float,
                "is_ood": bool,
                "top_k": [{"answer": str, "score": float}, ...],
                "parsed_question": dict,
                "latency_ms": float,
            }
        """
        t0 = time.monotonic()

        ctx = self.context.get_context()
        if ctx is None:
            return {
                "question": question,
                "answer": "Context buffer not ready yet. Please wait.",
                "confidence": 0.0,
                "is_ood": True,
                "top_k": [],
                "parsed_question": {},
                "latency_ms": 0.0,
            }

        ctx = ctx.to(DEVICE)

        node_ips, node_cores = self.context.get_nodes()
        if not node_ips:
            node_ips, node_cores = self.node_ips, self.node_cores

        parsed = parse_question(question, node_ips)
        q_emb = encode_questions(self.question_encoder, [parsed], node_ips)

        with torch.no_grad():
            idx, conf, all_logits = self.model.infer(ctx, q_emb)
            topk_results = self.model.infer_topk(ctx, q_emb, k=top_k)

        is_ood = self.model.ood_cosine(ctx, q_emb) < OOD_THRESHOLD

        semantic = self.context.get_semantic()
        answer = fill_answer_template(idx, node_ips, semantic)

        answer = self._enrich_answer(answer, parsed, semantic, node_ips, node_cores)

        latency_ms = (time.monotonic() - t0) * 1000

        return {
            "question": question,
            "answer": answer,
            "confidence": float(conf),
            "is_ood": is_ood,
            "top_k": [
                {
                    "answer": self.answer_templates[i]
                    if i < len(self.answer_templates)
                    else "?",
                    "score": float(s),
                }
                for i, s in topk_results
            ],
            "parsed_question": parsed,
            "latency_ms": latency_ms,
        }

    def _enrich_answer(
        self,
        answer: str,
        parsed: dict,
        semantic: dict,
        node_ips: list[str],
        node_cores: list[int],
    ) -> str:
        """Add specific values to the answer based on context."""
        metric = parsed.get("metric", "")
        node_ref = parsed.get("node_ref")

        if node_ref and node_ref in semantic:
            sf = semantic[node_ref]

            if "cpu" in metric.lower() or "cpu" in answer.lower():
                cpu_util = sf.get("cpu_utilization_fraction")
                load1 = sf.get("node_load1")
                node_idx = node_ips.index(node_ref) if node_ref in node_ips else -1
                cores = node_cores[node_idx] if 0 <= node_idx < len(node_cores) else 1
                if cpu_util is not None:
                    detail = f"utilization: {cpu_util * 100:.0f}%"
                    if load1 is not None:
                        detail += f", load avg: {load1:.1f}"
                    answer += f" ({detail})"
                elif load1 is not None:
                    pct = min(100.0, load1 / max(1, cores) * 100)
                    answer += (
                        f" (load avg: {load1:.1f}, ~{pct:.0f}% across {cores} cores)"
                    )

            elif "mem" in metric.lower() or "memory" in answer.lower():
                mem_frac = sf.get("mem_used_fraction")
                mem_total = sf.get("mem_total_bytes")
                if mem_frac is not None:
                    answer += f" ({mem_frac * 100:.1f}% used"
                    if mem_total:
                        answer += f" of {mem_total / 1e9:.1f} GB"
                    answer += ")"

            elif "temp" in metric.lower() or "temp" in answer.lower():
                temp = sf.get("max_temp_celsius")
                if temp is not None:
                    answer += f" ({temp:.1f}°C)"

        return answer


def run_repl(engine: LPSEInference):
    """Interactive question-answering loop."""
    print("LPSE interactive REPL. Type 'nodes' to list nodes, 'quit' to exit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if question.lower() == "nodes":
            for i, ip in enumerate(engine.node_ips):
                cores = engine.node_cores[i] if i < len(engine.node_cores) else "?"
                print(f"  [{i}] {ip} ({cores} cores)")
            continue
        if question.lower() == "answers":
            for i, tmpl in enumerate(engine.answer_templates):
                print(f"  [{i:3d}] {tmpl}")
            continue

        result = engine.ask(question)

        ood_marker = " ⚠️  LOW CONFIDENCE (possible OOD)" if result["is_ood"] else ""
        print(f"\n💡 {result['answer']}{ood_marker}")
        print(
            f"   confidence: {result['confidence']:.3f}  |  latency: {result['latency_ms']:.1f}ms"
        )

        if result["top_k"]:
            print("   top alternatives:")
            for alt in result["top_k"]:
                marker = " ←" if alt["answer"] == result["answer"] else ""
                print(f"     [{alt['score']:.3f}] {alt['answer']}{marker}")

        print()


def run_http_server(engine: LPSEInference, port: int = 8080):
    """Minimal HTTP server for programmatic access."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/ask":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                try:
                    req = json.loads(body)
                    question = req.get("question", "")
                    top_k_raw = req.get("top_k", 3)
                    try:
                        top_k = max(1, min(10, int(top_k_raw)))
                    except (TypeError, ValueError):
                        top_k = 3
                except (json.JSONDecodeError, AttributeError):
                    self.send_error(400, "Invalid JSON")
                    return

                result = engine.ask(question, top_k=top_k)

                result.pop("parsed_question", None)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())
            else:
                self.send_error(404)

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "context_ready": engine.context.ready,
                            "nodes": len(engine.node_ips),
                            "num_answers": len(engine.answer_templates),
                        }
                    ).encode()
                )
            elif self.path == "/nodes":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                nodes = [
                    {"ip": ip, "cores": cores}
                    for ip, cores in zip(engine.node_ips, engine.node_cores)
                ]
                self.wfile.write(json.dumps(nodes, indent=2).encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            logger.debug(format, *args)

    server = HTTPServer(("0.0.0.0", port), Handler)
    logger.info("HTTP server listening on port %d", port)
    logger.info('  POST /ask    — {"question": "..."}')
    logger.info("  GET  /health — system status")
    logger.info("  GET  /nodes  — list nodes")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="LPSE Inference")
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Run HTTP server instead of REPL"
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    parser.add_argument(
        "--query", type=str, default=None, help="One-shot query (print result and exit)"
    )
    parser.add_argument(
        "--update-normalizer",
        action=argparse.BooleanOptionalAction,
        default=UPDATE_NORMALIZER,
        help="Update the checkpoint normalizer with live inference samples.",
    )
    args = parser.parse_args()

    engine = LPSEInference.from_checkpoint(
        args.checkpoint, update_normalizer=args.update_normalizer
    )
    engine.start()

    if args.query:
        result = engine.ask(args.query)
        print(json.dumps(result, indent=2, default=str))
        engine.stop()
    elif args.serve:
        run_http_server(engine, port=args.port)
        engine.stop()
    else:
        run_repl(engine)
        engine.stop()


if __name__ == "__main__":
    main()
