#!/usr/bin/env python3
"""Evaluate benchmark results: rule metrics + dual LLM judge + failure scores."""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.judge import judge_prediction
from src.metrics import (
    conflict_resolution_ok,
    correct_abstention,
    citation_completeness,
    citation_support_rate,
    hard_negative_resistant,
    mean,
    rule_accuracy,
    safe_ratio,
)

CONDITIONS = [
    "clean",
    "missing_evidence",
    "conflict",
    "hard_negative",
    "boundary",
    "noise",
    "evidence_position",
]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for cond in CONDITIONS:
        d = results_dir / cond
        if not d.exists():
            continue
        for path in sorted(d.glob("*.jsonl")):
            for row in read_jsonl(path):
                row.setdefault("condition", cond)
                row.setdefault("model_slug", path.stem)
                rows.append(row)
    return rows


def finalize_accuracy(rule: dict, judge: dict | None) -> int | None:
    if rule.get("rule_confident") and rule.get("accuracy") is not None:
        return int(rule["accuracy"])
    if judge and judge.get("accuracy") is not None:
        return int(judge["accuracy"])
    return rule.get("accuracy")


def cache_key_for(row: dict) -> str:
    return (
        f"{row.get('model_slug')}|{row.get('sample_id')}|{row.get('condition')}"
        f"|pos{row.get('gold_position')}"
    )


def evaluate_row(
    row: dict,
    *,
    judges: dict,
    decode: dict,
    skip_judge: bool,
    judge_cache: dict[str, dict],
    cache_lock: threading.Lock,
) -> dict:
    expected = row.get("expected_behavior") or "answer"
    rule = rule_accuracy(
        answer=row.get("answer"),
        gold_answer=row.get("gold_answer") or "",
        expected_behavior=expected,
        abstained=row.get("abstained"),
        alternate_answer=row.get("alternate_answer"),
    )
    cit_comp = citation_completeness(row.get("citations"))
    cit_prog = citation_support_rate(row.get("citations"), row.get("gold_chunk_ids"))

    judge = None
    if not skip_judge:
        cache_key = cache_key_for(row)
        with cache_lock:
            cached = judge_cache.get(cache_key)
        if cached is not None:
            judge = cached
        else:
            judge = judge_prediction(
                question=row.get("question") or "",
                gold_answer=row.get("gold_answer") or "",
                model_answer=row.get("answer") or "",
                citations=row.get("citations"),
                contexts=row.get("contexts") or [],
                expected_behavior=expected,
                answer_available=bool(row.get("answer_available", True)),
                primary_model=judges["primary"],
                secondary_model=judges["secondary"],
                temperature=float(decode.get("temperature", 0)),
                top_p=float(decode.get("top_p", 1.0)),
                max_tokens=int(decode.get("judge_max_tokens") or decode.get("max_tokens", 256)),
            )
            with cache_lock:
                judge_cache[cache_key] = judge

    accuracy = finalize_accuracy(rule, judge)

    out = {
        **{k: row.get(k) for k in (
            "run_seed", "sample_id", "condition", "model", "model_slug",
            "question", "gold_answer", "answer", "citations", "abstained",
            "parse_ok", "error", "expected_behavior", "answer_available",
            "gold_chunk_ids", "conflict_chunk_ids", "alternate_answer",
            "gold_position", "latency_ms", "input_tokens", "output_tokens",
        )},
        "rule": rule,
        "accuracy": accuracy,
        "relevance": judge.get("relevance") if judge else None,
        "faithfulness": judge.get("faithfulness") if judge else None,
        "citation_accuracy": (
            judge.get("citation_accuracy") if judge and judge.get("citation_accuracy") is not None
            else cit_prog
        ),
        "citation_completeness": cit_comp,
        "judge_agreement": judge.get("judge_agreement") if judge else None,
        "needs_human_review": judge.get("needs_human_review") if judge else None,
        "judge": judge,
    }

    cond = row.get("condition")
    if cond == "missing_evidence":
        out["correct_abstention"] = correct_abstention(row.get("answer"), row.get("abstained"))
    if cond == "conflict":
        out["conflict_resolved"] = conflict_resolution_ok(
            answer=row.get("answer"),
            gold_answer=row.get("gold_answer") or "",
            alternate_answer=row.get("alternate_answer"),
            citations=row.get("citations"),
            gold_chunk_ids=row.get("gold_chunk_ids"),
            abstained=row.get("abstained"),
        )
    if cond == "hard_negative":
        out["hard_negative_resistant"] = hard_negative_resistant(
            answer=row.get("answer"),
            expected_behavior=expected,
            abstained=row.get("abstained"),
        )
    return out


def aggregate(evals: list[dict]) -> dict:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for e in evals:
        by_model[e.get("model_slug") or e.get("model")].append(e)

    summary: dict = {"models": {}, "parse_rates": {}, "error_rates": {}, "judge_stats": {}}

    agreements = [e["judge_agreement"] for e in evals if e.get("judge_agreement") is not None]
    reviews = [e["needs_human_review"] for e in evals if e.get("needs_human_review") is not None]
    summary["judge_stats"] = {
        "n_judged": len(agreements),
        "agreement_rate": mean([1.0 if a else 0.0 for a in agreements]),
        "human_review_rate": mean([1.0 if r else 0.0 for r in reviews]),
    }

    for model, rows in sorted(by_model.items()):
        parse_ok = [1.0 if r.get("parse_ok") else 0.0 for r in rows]
        errors = [1.0 if r.get("error") else 0.0 for r in rows]
        summary["parse_rates"][model] = mean(parse_ok)
        summary["error_rates"][model] = mean(errors)

        by_cond: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_cond[r["condition"]].append(r)

        def acc(cond: str) -> float | None:
            xs = [float(r["accuracy"]) for r in by_cond.get(cond, []) if r.get("accuracy") is not None]
            return mean(xs)

        clean_acc = acc("clean")
        noise_acc = acc("noise")
        boundary_acc = acc("boundary")

        pos_rows = by_cond.get("evidence_position", [])
        pos_groups: dict[str, list[float]] = {"first": [], "middle": [], "last": []}
        for r in pos_rows:
            if r.get("accuracy") is None:
                continue
            gp = r.get("gold_position")
            if gp == 0:
                pos_groups["first"].append(float(r["accuracy"]))
            elif gp == 4:
                pos_groups["middle"].append(float(r["accuracy"]))
            elif gp == 7:
                pos_groups["last"].append(float(r["accuracy"]))

        first_acc = mean(pos_groups["first"])
        last_acc = mean(pos_groups["last"])

        mar_rows = by_cond.get("missing_evidence", [])
        mar = mean([1.0 if r.get("correct_abstention") else 0.0 for r in mar_rows]) if mar_rows else None

        crs_rows = by_cond.get("conflict", [])
        crs = mean([1.0 if r.get("conflict_resolved") else 0.0 for r in crs_rows]) if crs_rows else None

        hnr_rows = by_cond.get("hard_negative", [])
        hnr = mean([1.0 if r.get("hard_negative_resistant") else 0.0 for r in hnr_rows]) if hnr_rows else None

        clean_rows = by_cond.get("clean", [])
        summary["models"][model] = {
            "clean": {
                "accuracy": clean_acc,
                "relevance": mean([float(r["relevance"]) for r in clean_rows if r.get("relevance") is not None]),
                "faithfulness": mean(
                    [float(r["faithfulness"]) for r in clean_rows if r.get("faithfulness") is not None]
                ),
                "citation_accuracy": mean(
                    [
                        float(r["citation_accuracy"])
                        for r in clean_rows
                        if r.get("citation_accuracy") is not None
                    ]
                ),
            },
            "failure_accuracy": {
                "missing_evidence": acc("missing_evidence"),
                "conflict": acc("conflict"),
                "hard_negative": acc("hard_negative"),
                "boundary": boundary_acc,
                "noise": noise_acc,
                "evidence_position": acc("evidence_position"),
            },
            "failure_behavior": {
                "MAR": mar,
                "CRS": crs,
                "HNR": hnr,
                "NRS": safe_ratio(noise_acc, clean_acc),
                "BRS": safe_ratio(boundary_acc, clean_acc),
                "NoiseDrop": (clean_acc - noise_acc) if clean_acc is not None and noise_acc is not None else None,
                "PositionDrop": (first_acc - last_acc) if first_acc is not None and last_acc is not None else None,
                "position_accuracy": {
                    "first": first_acc,
                    "middle": mean(pos_groups["middle"]),
                    "last": last_acc,
                },
            },
            "n": {c: len(by_cond.get(c, [])) for c in CONDITIONS},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "pilot_s42")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "eval" / "pilot_s42")
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--human-sample-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    with args.models_config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    judges = cfg.get("judges") or {"primary": "gpt-oss", "secondary": "deepseek"}
    decode = cfg.get("decode") or {}
    decode = {
        **decode,
        "judge_max_tokens": cfg.get("judge_max_tokens", decode.get("max_tokens", 256)),
    }

    rows = load_results(args.results)
    if not rows:
        print(f"No results found under {args.results}")
        sys.exit(1)

    enabled_slugs = {
        m["slug"] for m in (cfg.get("models") or []) if m.get("enabled", True)
    }
    if enabled_slugs:
        before = len(rows)
        rows = [r for r in rows if (r.get("model_slug") or "") in enabled_slugs]
        print(f"Using enabled models {sorted(enabled_slugs)}: {len(rows)}/{before} rows")

    judge_cache: dict[str, dict] = {}
    cache_lock = threading.Lock()
    cache_path = args.out / "judge_cache.jsonl"
    for prev in read_jsonl(cache_path):
        judge_cache[prev["cache_key"]] = prev["judge"]

    evals: list[dict | None] = [None] * len(rows)
    concurrency = 1 if args.skip_judge else max(1, args.concurrency)

    def _job(idx: int, row: dict) -> tuple[int, dict]:
        return idx, evaluate_row(
            row,
            judges=judges,
            decode=decode,
            skip_judge=args.skip_judge,
            judge_cache=judge_cache,
            cache_lock=cache_lock,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_job, i, row) for i, row in enumerate(rows)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="evaluate"):
            idx, e = fut.result()
            evals[idx] = e

    assert all(e is not None for e in evals)
    evals_final: list[dict] = evals  # type: ignore

    write_jsonl(
        cache_path,
        [{"cache_key": k, "judge": v} for k, v in sorted(judge_cache.items())],
    )
    write_jsonl(args.out / "evaluated.jsonl", evals_final)
    summary = aggregate(evals_final)
    write_json(args.out / "summary.json", summary)

    rng = random.Random(args.seed)
    n = max(1, int(len(evals_final) * args.human_sample_rate))
    disagreed = [e for e in evals_final if e.get("needs_human_review")]
    sample = disagreed[:n]
    if len(sample) < n:
        rest = [e for e in evals_final if e not in sample]
        sample.extend(rng.sample(rest, min(n - len(sample), len(rest))))
    write_jsonl(args.out / "human_review_sample.jsonl", sample)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
