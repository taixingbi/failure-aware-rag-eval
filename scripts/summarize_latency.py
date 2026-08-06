#!/usr/bin/env python3
"""Summarize RQ5 latency/cost records into paper tables + latency_summary.json.

Primary latency metrics for the paper are percentiles over successful calls:
  E2E p50 / p95, TTFT p50 / p95
(not per-row single-call values, and not mean-alone).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
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


def round_or_none(v: float | None, digits: int = 3) -> float | None:
    return None if v is None else round(v, digits)


def fmt(v: float | None, digits: int = 1) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def model_stats(rows: list[dict]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    e2e = [float(r["e2e_latency_ms"]) for r in ok if r.get("e2e_latency_ms") is not None]
    ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
    otps = [
        float(r["output_tokens_per_second"])
        for r in ok
        if r.get("output_tokens_per_second") is not None
    ]
    costs = [float(r["model_cost_usd"]) for r in ok if r.get("model_cost_usd") is not None]
    err = 1.0 - (len(ok) / len(rows)) if rows else 1.0
    return {
        "n_total": len(rows),
        "n_success": len(ok),
        "error_rate": round(err, 4),
        "e2e_latency_ms": {
            "p50": round_or_none(percentile(e2e, 50), 1),
            "p90": round_or_none(percentile(e2e, 90), 1),
            "p95": round_or_none(percentile(e2e, 95), 1),
            "p99": round_or_none(percentile(e2e, 99), 1),
            "mean": round_or_none(mean(e2e), 1),
        },
        "ttft_ms": {
            "p50": round_or_none(percentile(ttft, 50), 1),
            "p95": round_or_none(percentile(ttft, 95), 1),
            "mean": round_or_none(mean(ttft), 1),
        },
        "otps_mean": round_or_none(mean(otps), 2),
        "cost_per_request_usd_mean": round_or_none(mean(costs), 6),
        "cost_per_1k_usd": round_or_none((mean(costs) * 1000.0) if costs else None, 4),
    }


def build_summary(rows: list[dict]) -> dict[str, Any]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model_alias"]].append(r)

    models = {slug: model_stats(rs) for slug, rs in sorted(by_model.items())}

    by_fail: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        key = r.get("failure_type") or r.get("condition") or "clean"
        by_fail[str(key)].append(r)

    failures: dict[str, Any] = {}
    for fail, rs in sorted(by_fail.items()):
        e2e = [float(r["e2e_latency_ms"]) for r in rs if r.get("e2e_latency_ms") is not None]
        tin = [float(r["input_tokens"]) for r in rs if r.get("input_tokens") is not None]
        tout = [float(r["output_tokens"]) for r in rs if r.get("output_tokens") is not None]
        costs = [float(r["model_cost_usd"]) for r in rs if r.get("model_cost_usd") is not None]
        failures[fail] = {
            "n": len(rs),
            "e2e_p50_ms": round_or_none(percentile(e2e, 50), 1),
            "e2e_p95_ms": round_or_none(percentile(e2e, 95), 1),
            "input_tokens_mean": round_or_none(mean(tin), 1),
            "output_tokens_mean": round_or_none(mean(tout), 1),
            "cost_per_request_usd_mean": round_or_none(mean(costs), 6),
        }

    return {
        "metric_note": (
            "p50/p95 are percentiles over successful stream calls only "
            "(rows with error excluded). Each jsonl row is one call; "
            "these aggregates are the paper latency metrics."
        ),
        "n_records": len(rows),
        "n_success": sum(1 for r in rows if not r.get("error")),
        "models": models,
        "by_failure": failures,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("## Table A: Latency p50 / p95 (primary)")
    lines.append("| Model | N | E2E p50 | E2E p95 | TTFT p50 | TTFT p95 | Cost/1K QA | Error rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for slug, st in summary["models"].items():
        e2e = st["e2e_latency_ms"]
        ttft = st["ttft_ms"]
        lines.append(
            f"| {slug} | {st['n_success']} | {fmt(e2e['p50'])} | {fmt(e2e['p95'])} | "
            f"{fmt(ttft['p50'])} | {fmt(ttft['p95'])} | {fmt(st['cost_per_1k_usd'], 4)} | "
            f"{st['error_rate']:.3f} |"
        )

    lines.append("")
    lines.append("## Extended percentiles")
    lines.append("| Model | E2E p90 | E2E p99 | E2E mean | TTFT mean | OTPS mean |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for slug, st in summary["models"].items():
        e2e = st["e2e_latency_ms"]
        ttft = st["ttft_ms"]
        lines.append(
            f"| {slug} | {fmt(e2e['p90'])} | {fmt(e2e['p99'])} | {fmt(e2e['mean'])} | "
            f"{fmt(ttft['mean'])} | {fmt(st['otps_mean'], 2)} |"
        )

    lines.append("")
    lines.append("## Failure-specific overhead")
    lines.append("| Failure | N | E2E p50 | E2E p95 | Input tok mean | Output tok mean | Cost/req |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fail, st in summary["by_failure"].items():
        lines.append(
            f"| {fail} | {st['n']} | {fmt(st['e2e_p50_ms'])} | {fmt(st['e2e_p95_ms'])} | "
            f"{fmt(st['input_tokens_mean'], 1)} | {fmt(st['output_tokens_mean'], 1)} | "
            f"{fmt(st['cost_per_request_usd_mean'], 6)} |"
        )

    lines.append("")
    lines.append(f"_{summary['metric_note']}_")
    lines.append(
        "_OTPS includes prefill wait. bedrock_latency_ms is null until Lambda returns it._"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "results" / "latency_s42" / "latency_records.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write latency_summary.json + latency_report.md here (default: records parent)",
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

    summary = build_summary(rows)
    out_dir = args.out_dir or args.records.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "latency_summary.json"
    report_path = out_dir / "latency_report.md"
    md = render_markdown(summary)

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

        cpca_rows = []
        md += "\n## Table C: Cost per Correct Answer\n"
        md += "| Model | Total serving cost | Correct answers | Cost/correct |\n"
        md += "|---|---:|---:|---:|\n"
        by_model: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_model[r["model_alias"]].append(r)
        for slug in sorted(by_model):
            subset = [
                r for r in by_model[slug] if not r.get("error") and int(r.get("rep") or 1) == 1
            ]
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
                if correct.get(k) == 1:
                    n_correct += 1
            cpca = (total_cost / n_correct) if n_correct else None
            cpca_rows.append(
                {
                    "model": slug,
                    "total_cost_usd": round(total_cost, 6),
                    "n_correct": n_correct,
                    "n": n,
                    "cost_per_correct_usd": round(cpca, 6) if cpca is not None else None,
                }
            )
            md += (
                f"| {slug} | {fmt(total_cost, 6)} | {n_correct}/{n} | "
                f"{fmt(cpca, 6)} |\n"
            )
        summary["cost_per_correct"] = cpca_rows

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
