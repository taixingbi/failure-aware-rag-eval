#!/usr/bin/env python3
"""Aggregate multiple evaluation summaries into mean±std tables and decision metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

FAILURE_KEYS = [
    "missing_evidence",
    "conflict",
    "hard_negative",
    "boundary",
    "noise",
    "evidence_position",
]

BASE_METRICS = ["accuracy", "relevance", "faithfulness", "citation_accuracy"]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def load_summary(path: Path) -> dict:
    if path.is_dir():
        path = path / "summary.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_costs(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("models", {})


def format_stat(xs: list[float]) -> str:
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{mean(xs):.3f} ± {stdev(xs):.3f}"


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def summary_label(path: Path) -> str:
    if path.is_dir():
        return path.name
    return path.stem


def summarize_models(summaries: list[dict]) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for summary in summaries:
        models = summary.get("models", {})
        for model, metrics in models.items():
            result.setdefault(model, {})
            for field, value in metrics.get("clean", {}).items():
                key = f"clean.{field}"
                if value is not None:
                    result[model].setdefault(key, []).append(float(value))
            for field, value in metrics.get("failure_accuracy", {}).items():
                key = f"failure.{field}"
                if value is not None:
                    result[model].setdefault(key, []).append(float(value))
            for field, value in metrics.get("failure_behavior", {}).items():
                if isinstance(value, dict):
                    for subfield, subvalue in value.items():
                        key = f"behavior.{field}.{subfield}"
                        if subvalue is not None:
                            result[model].setdefault(key, []).append(float(subvalue))
                elif value is not None:
                    key = f"behavior.{field}"
                    result[model].setdefault(key, []).append(float(value))
            for cond, values in metrics.get("failure_metrics", {}).items():
                for field, value in values.items():
                    key = f"failure.{cond}.{field}"
                    if value is not None:
                        result[model].setdefault(key, []).append(float(value))
            performance = metrics.get("performance", {})
            for field in ("latency_ms", "input_tokens", "output_tokens"):
                val = performance.get(field)
                if val is not None:
                    key = f"performance.{field}"
                    result[model].setdefault(key, []).append(float(val))
    return result


def enterprise_score(weights: dict[str, float], model_values: dict[str, float]) -> float | None:
    total = 0.0
    wsum = 0.0
    for key, wt in weights.items():
        value = model_values.get(key)
        if value is None:
            continue
        total += wt * value
        wsum += wt
    return total / wsum if wsum else None


def cost_per_request(costs: dict[str, dict[str, float]], model: str, input_tokens: float, output_tokens: float) -> float | None:
    rates = costs.get(model)
    if not rates:
        return None
    prompt_price = rates.get("prompt")
    completion_price = rates.get("completion")
    if prompt_price is None or completion_price is None:
        return None
    return (input_tokens / 1000.0) * prompt_price + (output_tokens / 1000.0) * completion_price


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summaries",
        nargs="+",
        type=Path,
        default=[ROOT / "results" / "eval" / "s42"],
        help="One or more summary.json file paths or parent directories.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "configs" / "enterprise_weights.yaml",
    )
    parser.add_argument(
        "--costs",
        type=Path,
        default=ROOT / "configs" / "costs.yaml",
    )
    args = parser.parse_args()

    summaries = [load_summary(p) for p in args.summaries]
    costs = load_costs(args.costs)
    weights_cfg = yaml.safe_load(args.weights.open(encoding="utf-8"))
    weights = {k: float(v) for k, v in weights_cfg.items() if k != "cost_lambda"}
    cost_lambda = float(weights_cfg.get("cost_lambda", 1.0))

    model_stats = summarize_models(summaries)
    model_names = sorted(model_stats)

    print("## Table 1: Clean Quality (mean ± std across summaries)")
    print("| Model | Accuracy | Relevance | Faithfulness | Citation Accuracy |")
    print("|---|---:|---:|---:|---:|")
    for model in model_names:
        stats = model_stats[model]
        print(
            f"| {model} | {format_stat(stats.get('clean.accuracy', []))} | "
            f"{format_stat(stats.get('clean.relevance', []))} | "
            f"{format_stat(stats.get('clean.faithfulness', []))} | "
            f"{format_stat(stats.get('clean.citation_accuracy', []))} |"
        )

    print("\n## Table 2: Failure Accuracy (mean ± std)")
    print("| Model | Missing | Conflict | Hard Negative | Boundary | Noise | Position |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model in model_names:
        stats = model_stats[model]
        print(
            f"| {model} | {format_stat(stats.get('failure.missing_evidence', []))} | "
            f"{format_stat(stats.get('failure.conflict', []))} | "
            f"{format_stat(stats.get('failure.hard_negative', []))} | "
            f"{format_stat(stats.get('failure.boundary', []))} | "
            f"{format_stat(stats.get('failure.noise', []))} | "
            f"{format_stat(stats.get('failure.evidence_position', []))} |"
        )

    print("\n## Table 2b: Failure Quality per Condition")
    print("| Model | Condition | Accuracy | Faithfulness | Citation Accuracy |")
    print("|---|---:|---:|---:|---:|")
    for model in model_names:
        stats = model_stats[model]
        for cond in FAILURE_KEYS:
            print(
                f"| {model} | {cond} | {format_stat(stats.get(f'failure.{cond}.accuracy', []))} | "
                f"{format_stat(stats.get(f'failure.{cond}.faithfulness', []))} | "
                f"{format_stat(stats.get(f'failure.{cond}.citation_accuracy', []))} |"
            )

    print("\n## Table 3: Failure Behavior (mean ± std)")
    print("| Model | MAR | CRS | HNR | NRS | BRS | Position Drop |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model in model_names:
        stats = model_stats[model]
        print(
            f"| {model} | {format_stat(stats.get('behavior.MAR', []))} | "
            f"{format_stat(stats.get('behavior.CRS', []))} | "
            f"{format_stat(stats.get('behavior.HNR', []))} | "
            f"{format_stat(stats.get('behavior.NRS', []))} | "
            f"{format_stat(stats.get('behavior.BRS', []))} | "
            f"{format_stat(stats.get('behavior.PositionDrop', []))} |"
        )

    print("\n## Table 4: Performance and Cost")
    print("| Model | EnterpriseScore | Latency (ms) | Input tokens | Output tokens | Cost / request | Normalized cost | Mean FRS | FRS / $ | Utility |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    costs_by_model = {
        model: cost_per_request(costs, model, mean(model_stats[model].get('performance.input_tokens', []) or [0.0]), mean(model_stats[model].get('performance.output_tokens', []) or [0.0]))
        for model in model_names
    }
    max_cost = max([c for c in costs_by_model.values() if c is not None], default=None)
    for model in model_names:
        stats = model_stats[model]
        avg_latency = mean(stats.get('performance.latency_ms', [])) if stats.get('performance.latency_ms') else None
        avg_input = mean(stats.get('performance.input_tokens', [])) if stats.get('performance.input_tokens') else None
        avg_output = mean(stats.get('performance.output_tokens', [])) if stats.get('performance.output_tokens') else None
        failure_keys = [f'failure.{key}' for key in FAILURE_KEYS if stats.get(f'failure.{key}')]
        mean_frs = mean([mean(stats[key]) for key in failure_keys]) if failure_keys else None
        cost = costs_by_model.get(model)
        ent_score = enterprise_score(weights, {
            'missing_evidence': mean(stats.get('behavior.MAR', [])) if stats.get('behavior.MAR') else None,
            'conflict': mean(stats.get('behavior.CRS', [])) if stats.get('behavior.CRS') else None,
            'hard_negative': mean(stats.get('behavior.HNR', [])) if stats.get('behavior.HNR') else None,
            'noise': mean(stats.get('behavior.NRS', [])) if stats.get('behavior.NRS') else None,
            'boundary': mean(stats.get('behavior.BRS', [])) if stats.get('behavior.BRS') else None,
            'evidence_position': mean(stats.get('failure.evidence_position', [])) if stats.get('failure.evidence_position') else None,
        })
        normalized_cost = cost / max_cost if cost is not None and max_cost is not None and max_cost > 0 else None
        utility = None
        if ent_score is not None and normalized_cost is not None:
            utility = ent_score - cost_lambda * normalized_cost
        frs_per_dollar = None
        if cost is not None and cost > 0 and mean_frs is not None:
            frs_per_dollar = mean_frs / cost
        print(
            f"| {model} | {fmt(ent_score)} | {fmt(avg_latency)} | {fmt(avg_input)} | {fmt(avg_output)} | "
            f"{fmt(cost)} | {fmt(normalized_cost)} | {fmt(mean_frs)} | {fmt(frs_per_dollar)} | {fmt(utility)} |"
        )

    print("\n## Decision matrix notes")
    print("- EnterpriseScore weights are read from configs/enterprise_weights.yaml.")
    print("- Cost estimates require configs/costs.yaml with per-model prompt/completion price rates.")
    print("- Use mean FRS versus normalized cost to select models for high-stakes enterprise deployments.")

    print("\n## Summaries aggregated")
    for summary_path in args.summaries:
        print(f"- {summary_label(summary_path)}")


if __name__ == "__main__":
    main()
