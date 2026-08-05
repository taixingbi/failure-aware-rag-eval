#!/usr/bin/env python3
"""Summarize ablation metrics by severity and failure-condition variation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.3f}"


def load_evals(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        if path.is_dir():
            path = path / "evaluated.jsonl"
        rows.extend(read_jsonl(path))
    return rows


def summarize_by_key(rows: list[dict], model: str, condition: str, key: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("model_slug") != model:
            continue
        if row.get("condition") != condition:
            continue
        label = row.get(key)
        if label is None:
            continue
        values[str(label)].append(float(row.get("accuracy")) if row.get("accuracy") is not None else 0.0)
    return {k: mean(v) for k, v in values.items()}


def summarize_noise(rows: list[dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    models = sorted({r.get("model_slug") for r in rows if r.get("model_slug")})
    for model in models:
        for severity in ["low", "medium", "high"]:
            subset = [r for r in rows if r.get("model_slug") == model and r.get("condition") == "noise" and r.get("severity") == severity]
            acc = mean([float(r["accuracy"]) for r in subset if r.get("accuracy") is not None])
            fa = mean([float(r["faithfulness"]) for r in subset if r.get("faithfulness") is not None])
            ca = mean([float(r["citation_accuracy"]) for r in subset if r.get("citation_accuracy") is not None])
            result.setdefault(model, {})[severity] = acc if acc is not None else None
            result.setdefault(model, {})[f"{severity}_faithfulness"] = fa
            result.setdefault(model, {})[f"{severity}_citation_accuracy"] = ca
    return result


def summarize_hard_negative(rows: list[dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    models = sorted({r.get("model_slug") for r in rows if r.get("model_slug")})
    for model in models:
        for severity in ["low", "medium", "high"]:
            subset = [r for r in rows if r.get("model_slug") == model and r.get("condition") == "hard_negative" and r.get("severity") == severity]
            acc = mean([float(r["accuracy"]) for r in subset if r.get("accuracy") is not None])
            fa = mean([float(r["faithfulness"]) for r in subset if r.get("faithfulness") is not None])
            ca = mean([float(r["citation_accuracy"]) for r in subset if r.get("citation_accuracy") is not None])
            num_ctx = mean([float(r["num_contexts"]) for r in subset if r.get("num_contexts") is not None])
            mean_lex = mean([float(r["mean_lexical_overlap"]) for r in subset if r.get("mean_lexical_overlap") is not None])
            result.setdefault(model, {})[severity] = acc
            result.setdefault(model, {})[f"{severity}_faithfulness"] = fa
            result.setdefault(model, {})[f"{severity}_citation_accuracy"] = ca
            result.setdefault(model, {})[f"{severity}_num_contexts"] = num_ctx
            result.setdefault(model, {})[f"{severity}_mean_lex"] = mean_lex
    return result


def summarize_position(rows: list[dict]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    models = sorted({r.get("model_slug") for r in rows if r.get("model_slug")})
    for model in models:
        for position in [0, 4, 7]:
            subset = [r for r in rows if r.get("model_slug") == model and r.get("condition") == "evidence_position" and r.get("gold_position") == position]
            label = {0: "first", 4: "middle", 7: "last"}[position]
            result.setdefault(model, {})[label] = mean([float(r["accuracy"]) for r in subset if r.get("accuracy") is not None])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evals",
        nargs="+",
        type=Path,
        default=[ROOT / "results" / "eval" / "s42"],
        help="Path(s) to evaluated.jsonl or parent directories containing evaluated.jsonl.",
    )
    args = parser.parse_args()

    # filter missing eval paths with a warning
    existing_paths: list[Path] = []
    for p in args.evals:
        if not p.exists():
            print(f"Warning: eval path not found, skipping: {p}")
            continue
        existing_paths.append(p)
    rows = load_evals(existing_paths)
    if not rows:
        raise SystemExit("No evaluated rows found.")

    models = sorted({r.get("model_slug") for r in rows if r.get("model_slug")})

    print("## Noise severity ablation")
    print("| Model | Low Acc | Mid Acc | High Acc | Low Faith | Mid Faith | High Faith | Low CA | Mid CA | High CA |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    noise = summarize_noise(rows)
    for model in models:
        data = noise.get(model, {})
        print(
            f"| {model} | {fmt(data.get('low'))} | {fmt(data.get('medium'))} | {fmt(data.get('high'))} | "
            f"{fmt(data.get('low_faithfulness'))} | {fmt(data.get('medium_faithfulness'))} | {fmt(data.get('high_faithfulness'))} | "
            f"{fmt(data.get('low_citation_accuracy'))} | {fmt(data.get('medium_citation_accuracy'))} | {fmt(data.get('high_citation_accuracy'))} |"
        )

    print("\n## Hard-negative severity ablation")
    print("| Model | Low Acc | Mid Acc | High Acc | Low Ctx | Mid Ctx | High Ctx | Low Lex | Mid Lex | High Lex |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    hard = summarize_hard_negative(rows)
    for model in models:
        data = hard.get(model, {})
        print(
            f"| {model} | {fmt(data.get('low'))} | {fmt(data.get('medium'))} | {fmt(data.get('high'))} | "
            f"{fmt(data.get('low_num_contexts'))} | {fmt(data.get('medium_num_contexts'))} | {fmt(data.get('high_num_contexts'))} | "
            f"{fmt(data.get('low_mean_lex'))} | {fmt(data.get('medium_mean_lex'))} | {fmt(data.get('high_mean_lex'))} |"
        )

    print("\n## Evidence position sensitivity")
    print("| Model | First Acc | Middle Acc | Last Acc |")
    print("|---|---:|---:|---:|")
    position = summarize_position(rows)
    for model in models:
        data = position.get(model, {})
        print(
            f"| {model} | {fmt(data.get('first'))} | {fmt(data.get('middle'))} | {fmt(data.get('last'))} |"
        )


if __name__ == "__main__":
    main()
