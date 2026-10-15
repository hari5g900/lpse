#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

KUBECTL="$(command -v kubectl || true)"; KUBECTL="${KUBECTL:-/tmp/opencode/kubectl}"
BASE_PORT=39101
MAP_FILE="artifacts/node_scrape_map.json"
PF_LOG_DIR="/tmp/opencode"

stop() {
  pkill -f "kubectl port-forward -n monitoring pod/node-exporter" 2>/dev/null || true
  rm -f "$MAP_FILE"
  echo "stopped node-exporter tunnel"
}
[ "${1:-}" = "--stop" ] && { stop; exit 0; }

mkdir -p artifacts "$PF_LOG_DIR"
stop || true

while read -r pod node; do
  [ -n "${pod:-}" ] || continue
  ip="$("$KUBECTL" get node "$node" -o jsonpath='{.status.addresses[?(@.type=="InternalIP")].address}')"
  echo "forwarding $node ($ip) -> 127.0.0.1:$BASE_PORT  (pod $pod)"
  "$KUBECTL" port-forward -n monitoring "pod/$pod" "$BASE_PORT:9100" >"$PF_LOG_DIR/pf_${ip}.log" 2>&1 &
  ENTRY_JSON="${ENTRY_JSON:-}${ENTRY_JSON:+,}\"${ip}\": \"http://127.0.0.1:${BASE_PORT}\""
  PIDS="${PIDS:-}${PIDS:+,}$!"
  BASE_PORT=$((BASE_PORT + 1))
done < <("$KUBECTL" get pods -n monitoring -l app=node-exporter \
  -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName --no-headers \
  | grep '^node-exporter-')

sleep 4

printf '{\n  "map": {%s},\n  "pids": [%s]\n}\n' "$ENTRY_JSON" "$PIDS" > "$MAP_FILE"
echo ""
echo "Set this in your environment before running the benchmark:"
echo "export NODE_SCRAPE_URL_MAP=\"$(python3 -c "import json;print(json.dumps(json.load(open('$MAP_FILE'))['map']))")\""
echo ""
echo "Tunnel running with pids: $PIDS  (config: $MAP_FILE)"
