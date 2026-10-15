#!/usr/bin/env python3
"""Remove duplicate-step datapoints from LPSE summary and training logs.

The first record seen for each step is kept, then outputs are written sorted by step.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CleanStats:
    input_count: int
    output_count: int
    duplicate_count: int
    output_path: Path


def _cleaned_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _step_key(step: Any) -> str:
    if isinstance(step, float) and step.is_integer():
        return str(int(step))
    return str(step).strip()


def _step_sort_key(step: str) -> tuple[int, float, str]:
    try:
        return (0, float(step), step)
    except ValueError:
        return (1, 0.0, step)


def clean_summary(summary_path: Path, output_path: Path) -> CleanStats:
    seen_steps: set[str] = set()
    records_by_step: dict[str, str] = {}
    input_count = 0
    duplicate_count = 0

    with summary_path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            record_text = line.strip()
            if not record_text:
                continue

            try:
                record = json.loads(record_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{summary_path}:{line_number} is not valid JSON: {exc}"
                ) from exc

            if "step" not in record:
                raise ValueError(
                    f"{summary_path}:{line_number} is missing a step field"
                )

            input_count += 1
            step = _step_key(record["step"])
            if step in seen_steps:
                duplicate_count += 1
                continue

            seen_steps.add(step)
            records_by_step[step] = line if line.endswith("\n") else f"{line}\n"

    with output_path.open("w", encoding="utf-8") as dst:
        for step in sorted(records_by_step, key=_step_sort_key):
            dst.write(records_by_step[step])

    return CleanStats(input_count, len(records_by_step), duplicate_count, output_path)


def clean_csv(csv_path: Path, output_path: Path, step_column: str) -> CleanStats:
    seen_steps: set[str] = set()
    rows_by_step: dict[str, list[str]] = {}
    input_count = 0
    duplicate_count = 0

    with csv_path.open("r", newline="", encoding="utf-8") as src:
        reader = csv.reader(src)

        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{csv_path} is empty") from exc

        if step_column not in header:
            raise ValueError(f"{csv_path} is missing a {step_column!r} column")

        step_index = header.index(step_column)

        for row_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if step_index >= len(row):
                raise ValueError(f"{csv_path}:{row_number} is missing the step value")

            input_count += 1
            step = _step_key(row[step_index])
            if step in seen_steps:
                duplicate_count += 1
                continue

            seen_steps.add(step)
            rows_by_step[step] = row

    with output_path.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst, lineterminator="\n")
        writer.writerow(header)
        for step in sorted(rows_by_step, key=_step_sort_key):
            writer.writerow(rows_by_step[step])

    return CleanStats(input_count, len(rows_by_step), duplicate_count, output_path)


def _print_stats(label: str, stats: CleanStats) -> None:
    print(
        f"{label}: read {stats.input_count}, wrote {stats.output_count}, "
        f"dropped {stats.duplicate_count} duplicate-step records"
    )
    print(f"Saved: {stats.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean LPSE summary and training logs by keeping only the first "
            "datapoint observed for each step and writing the output sorted by step."
        )
    )
    parser.add_argument(
        "--summary",
        default="lpse_summary.txt",
        help="Path to newline-delimited JSON summary file.",
    )
    parser.add_argument(
        "--csv",
        default="lpse_training_log.csv",
        help="Path to CSV training log file.",
    )
    parser.add_argument(
        "--step-column",
        default="step",
        help="CSV column name containing the training step.",
    )
    parser.add_argument(
        "--suffix",
        default="_cleaned",
        help="Suffix added before each output file extension.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary).expanduser()
    csv_path = Path(args.csv).expanduser()

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    summary_stats = clean_summary(
        summary_path, _cleaned_path(summary_path, args.suffix)
    )
    csv_stats = clean_csv(
        csv_path, _cleaned_path(csv_path, args.suffix), args.step_column
    )

    _print_stats("Summary", summary_stats)
    _print_stats("CSV", csv_stats)


if __name__ == "__main__":
    main()
