#!/usr/bin/env python3
"""Print pilot / paper table stubs from evaluation summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def fmt(v, digits=3):
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "results" / "eval" / "pilot_s42" / "summary.json",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "configs" / "enterprise_weights.yaml",
    )
    args = parser.parse_args()

    with args.summary.open(encoding="utf-8") as f:
        summary = json.load(f)
    with args.weights.open(encoding="utf-8") as f:
        weights = yaml.safe_load(f)

    models = summary.get("models") or {}

    print("## Table 1: Clean Quality")
    print("| Model | Accuracy | Relevance | Faithfulness | Citation Accuracy |")
    print("|---|---:|---:|---:|---:|")
    for m, d in models.items():
        c = d.get("clean") or {}
        print(
            f"| {m} | {fmt(c.get('accuracy'))} | {fmt(c.get('relevance'))} | "
            f"{fmt(c.get('faithfulness'))} | {fmt(c.get('citation_accuracy'))} |"
        )

    print("\n## Table 2: Failure Accuracy")
    print("| Model | Missing | Conflict | Hard Negative | Boundary | Noise | Position |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m, d in models.items():
        f = d.get("failure_accuracy") or {}
        print(
            f"| {m} | {fmt(f.get('missing_evidence'))} | {fmt(f.get('conflict'))} | "
            f"{fmt(f.get('hard_negative'))} | {fmt(f.get('boundary'))} | "
            f"{fmt(f.get('noise'))} | {fmt(f.get('evidence_position'))} |"
        )

    print("\n## Table 3: Failure-specific Behavior")
    print("| Model | MAR | CRS | HNR | NRS | BRS | Position Drop |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m, d in models.items():
        b = d.get("failure_behavior") or {}
        print(
            f"| {m} | {fmt(b.get('MAR'))} | {fmt(b.get('CRS'))} | {fmt(b.get('HNR'))} | "
            f"{fmt(b.get('NRS'))} | {fmt(b.get('BRS'))} | {fmt(b.get('PositionDrop'))} |"
        )

    print("\n## Table 4: Stability (FRS stub)")
    print("| Model | Mean FRS | Worst Failure | Best Failure |")
    print("|---|---:|---|---|")
    fail_keys = [
        "missing_evidence",
        "conflict",
        "hard_negative",
        "boundary",
        "noise",
        "evidence_position",
    ]
    for m, d in models.items():
        f = d.get("failure_accuracy") or {}
        scores = {k: f.get(k) for k in fail_keys if f.get(k) is not None}
        if not scores:
            print(f"| {m} | — | — | — |")
            continue
        mean_frs = sum(scores.values()) / len(scores)
        worst = min(scores, key=scores.get)
        best = max(scores, key=scores.get)
        print(f"| {m} | {fmt(mean_frs)} | {worst} ({fmt(scores[worst])}) | {best} ({fmt(scores[best])}) |")

    print("\n## Table 5: Cost-aware Selection (stub — costs not instrumented yet)")
    print("| Model | EnterpriseScore | Cost/1K | Latency | FRS per Dollar |")
    print("|---|---:|---:|---:|---:|")
    w = {k: float(v) for k, v in weights.items() if k in fail_keys}
    for m, d in models.items():
        f = d.get("failure_accuracy") or {}
        # EnterpriseScore uses behavior metrics where available else accuracy
        b = d.get("failure_behavior") or {}
        score_map = {
            "missing_evidence": b.get("MAR") if b.get("MAR") is not None else f.get("missing_evidence"),
            "conflict": b.get("CRS") if b.get("CRS") is not None else f.get("conflict"),
            "hard_negative": b.get("HNR") if b.get("HNR") is not None else f.get("hard_negative"),
            "noise": b.get("NRS") if b.get("NRS") is not None else f.get("noise"),
            "boundary": b.get("BRS") if b.get("BRS") is not None else f.get("boundary"),
            "evidence_position": f.get("evidence_position"),
        }
        ent = 0.0
        wsum = 0.0
        for k, wt in w.items():
            if score_map.get(k) is not None:
                ent += wt * float(score_map[k])
                wsum += wt
        ent = ent / wsum if wsum else None
        print(f"| {m} | {fmt(ent)} | — | — | — |")

    print("\n## Pilot gates")
    print("Parse rates:", json.dumps(summary.get("parse_rates"), indent=2))
    print("Error rates:", json.dumps(summary.get("error_rates"), indent=2))
    print("Judge stats:", json.dumps(summary.get("judge_stats"), indent=2))


if __name__ == "__main__":
    main()
