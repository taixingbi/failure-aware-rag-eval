#!/usr/bin/env python3
"""Summarize RQ5 latency/cost records into paper tables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "results" / "latency_s42" / "latency_records.jsonl",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=None,
        help="Optional evaluated.jsonl to join correctness for CPCA",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.records)
    if not rows:
        print(f"No records in {args.records}")
        return

    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model_alias"]].append(r)

    print("## Table A: Latency and Cost")
    print(
        "| Model | N | E2E p50 | E2E p90 | E2E p95 | E2E p99 | E2E mean | "
        "TTFT p50 | TTFT p95 | OTPS mean | Cost/1K QA | Error rate |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for slug in sorted(by_model):
        all_m = by_model[slug]
        ok = [r for r in all_m if not r.get("error")]
        e2e = [float(r["e2e_latency_ms"]) for r in ok if r.get("e2e_latency_ms") is not None]
        ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
        otps = [
            float(r["output_tokens_per_second"])
            for r in ok
            if r.get("output_tokens_per_second") is not None
        ]
        costs = [float(r["model_cost_usd"]) for r in ok if r.get("model_cost_usd") is not None]
        cost_1k = (mean(costs) * 1000.0) if costs else None
        err = 1.0 - (len(ok) / len(all_m)) if all_m else 1.0
        print(
            f"| {slug} | {len(ok)} | {fmt(percentile(e2e, 50))} | {fmt(percentile(e2e, 90))} | "
            f"{fmt(percentile(e2e, 95))} | {fmt(percentile(e2e, 99))} | {fmt(mean(e2e))} | "
            f"{fmt(percentile(ttft, 50))} | {fmt(percentile(ttft, 95))} | {fmt(mean(otps), 2)} | "
            f"{fmt(cost_1k, 4)} | {err:.3f} |"
        )

    print("\n## Failure-specific overhead")
    print("| Failure | N | Input tok mean | Output tok mean | E2E p50 | Cost/req mean |")
    print("|---|---:|---:|---:|---:|---:|")
    by_fail: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        key = r.get("failure_type") or r.get("condition") or "clean"
        by_fail[key].append(r)
    for fail in sorted(by_fail):
        rs = by_fail[fail]
        tin = [float(r["input_tokens"]) for r in rs if r.get("input_tokens") is not None]
        tout = [float(r["output_tokens"]) for r in rs if r.get("output_tokens") is not None]
        e2e = [float(r["e2e_latency_ms"]) for r in rs if r.get("e2e_latency_ms") is not None]
        costs = [float(r["model_cost_usd"]) for r in rs if r.get("model_cost_usd") is not None]
        print(
            f"| {fail} | {len(rs)} | {fmt(mean(tin), 1)} | {fmt(mean(tout), 1)} | "
            f"{fmt(percentile(e2e, 50))} | {fmt(mean(costs), 6)} |"
        )

    if args.eval and args.eval.exists():
        eval_rows = read_jsonl(args.eval)
        correct: dict[tuple[str, str, str], int] = {}
        for er in eval_rows:
            key = (
                str(er.get("model_slug") or er.get("model")),
                str(er.get("sample_id")),
                str(er.get("condition")),
            )
            acc = er.get("accuracy")
            if acc is not None:
                correct[key] = int(acc)

        print("\n## Table C: Cost per Correct Answer (joined with eval)")
        print("| Model | Total serving cost | Correct answers | Cost/correct |")
        print("|---|---:|---:|---:|")
        for slug in sorted(by_model):
            # Use first rep only to avoid double-counting correctness.
            subset = [r for r in by_model[slug] if not r.get("error") and int(r.get("rep") or 1) == 1]
            total_cost = 0.0
            n_correct = 0
            n = 0
            for r in subset:
                c = r.get("model_cost_usd")
                if c is None:
                    continue
                n += 1
                total_cost += float(c)
                k = (slug, str(r.get("sample_id")), str(r.get("condition")))
                # evidence_position: eval keys may not include gold_position; best-effort join
                if correct.get(k) == 1:
                    n_correct += 1
            cpca = (total_cost / n_correct) if n_correct else None
            print(f"| {slug} | {fmt(total_cost, 6)} | {n_correct}/{n} | {fmt(cpca, 6)} |")

    print("\n_Note: OTPS includes prefill wait (non-streaming-equivalent wall time / output tokens)._")
    print("_bedrock_latency_ms is null until Lambda returns it; TTFT/E2E are client-measured on stream._")


if __name__ == "__main__":
    main()
