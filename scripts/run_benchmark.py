#!/usr/bin/env python3
"""Run unified multi-model RAG benchmark (clean + failure conditions)."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client import InferenceClient
from src.parse import parse_answer_payload
from src.prompts import build_answer_messages, load_prompts

FAILURE_TYPES = [
    "missing_evidence",
    "conflict",
    "hard_negative",
    "boundary",
    "noise",
    "evidence_position",
]

_write_lock = threading.Lock()


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


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def load_models(cfg_path: Path) -> tuple[list[dict], dict, int]:
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = [m for m in cfg["models"] if m.get("enabled", True)]
    return models, cfg["decode"], int(cfg.get("concurrency_per_model", 1))


def result_key(sample: dict, cond: str) -> str:
    sid = sample.get("seed_id") or sample.get("sample_id")
    if cond == "evidence_position":
        return f"{sid}__pos{sample.get('gold_position')}"
    return str(sid)


def run_one(
    *,
    model_id: str,
    model_slug: str,
    sample: dict,
    condition: str,
    contexts: list[str],
    decode: dict,
    prompts: dict,
) -> dict:
    # One client per call keeps requests thread-safe under concurrency.
    client = InferenceClient()
    auth_ids = None
    if condition == "conflict":
        auth_ids = sample.get("gold_chunk_ids") or []

    messages = build_answer_messages(
        sample["question"],
        contexts,
        authoritative_chunk_ids=auth_ids,
        prompts=prompts,
    )
    result = client.chat_completions(
        model_id,
        messages,
        temperature=float(decode.get("temperature", 0)),
        top_p=float(decode.get("top_p", 1.0)),
        max_tokens=int(decode.get("max_tokens", 256)),
    )
    parsed = parse_answer_payload(result.content)
    return {
        "run_seed": sample.get("run_seed"),
        "sample_id": sample.get("seed_id"),
        "condition": condition,
        "model": model_id,
        "model_slug": model_slug,
        "failure_type": sample.get("failure_type") if condition != "clean" else None,
        "question": sample.get("question"),
        "gold_answer": sample.get("gold_answer"),
        "contexts": contexts,
        "gold_chunk_ids": sample.get("gold_chunk_ids"),
        "conflict_chunk_ids": sample.get("conflict_chunk_ids"),
        "alternate_answer": sample.get("alternate_answer"),
        "expected_behavior": (
            "answer" if condition == "clean" else sample.get("expected_behavior")
        ),
        "answer_available": (
            True if condition == "clean" else sample.get("answer_available")
        ),
        "gold_position": sample.get("gold_position"),
        "severity": sample.get("severity"),
        "num_contexts": sample.get("num_contexts"),
        "mean_lexical_overlap": sample.get("mean_lexical_overlap"),
        "max_lexical_overlap": sample.get("max_lexical_overlap"),
        "same_category_only": sample.get("same_category_only"),
        "position_variant": sample.get("position_variant"),
        "source_failure_id": sample.get("source_failure_id") if condition != "clean" else None,
        "answer": parsed.get("answer"),
        "citations": parsed.get("citations"),
        "abstained": parsed.get("abstained"),
        "parse_ok": parsed.get("parse_ok"),
        "raw_text": parsed.get("raw_text"),
        "latency_ms": result.latency_ms,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "error": result.error,
    }


def run_model_tasks(
    model: dict,
    pending: list[tuple[dict, str, list[str]]],
    *,
    out_dir: Path,
    decode: dict,
    prompts: dict,
    concurrency: int,
) -> None:
    """Run one model's pending tasks with up to `concurrency` in-flight requests."""
    if not pending:
        return

    def _job(sample: dict, cond: str, contexts: list[str]) -> tuple[Path, str, dict]:
        record = run_one(
            model_id=model["model_id"],
            model_slug=model["slug"],
            sample=sample,
            condition=cond,
            contexts=contexts,
            decode=decode,
            prompts=prompts,
        )
        out_path = out_dir / cond / f"{model['slug']}.jsonl"
        key = result_key(sample, cond)
        return out_path, key, record

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_job, sample, cond, contexts) for sample, cond, contexts in pending]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{model['slug']}",
            leave=True,
        ):
            out_path, key, record = fut.result()
            append_jsonl(out_path, record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "benchmark" / "pilot_s42_n20.jsonl",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pilot_s42")
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--models", default="", help="Comma-separated slugs to run (default: enabled)")
    parser.add_argument("--conditions", default="", help="Comma-separated conditions (default: all)")
    parser.add_argument("--limit", type=int, default=0, help="Limit unique seeds (0=all)")
    parser.add_argument("--concurrency", type=int, default=0, help="Override concurrency_per_model")
    parser.add_argument("--smoke", action="store_true", help="One short hello per model only")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    models, decode, cfg_concurrency = load_models(args.models_config)
    concurrency = args.concurrency if args.concurrency > 0 else cfg_concurrency
    if args.models:
        wanted = {s.strip() for s in args.models.split(",") if s.strip()}
        # allow explicit --models to include disabled entries
        with args.models_config.open(encoding="utf-8") as f:
            all_models = yaml.safe_load(f)["models"]
        models = [m for m in all_models if m["slug"] in wanted]

    prompts = load_prompts()

    if args.smoke:
        client = InferenceClient()
        for m in models:
            r = client.chat_completions(
                m["model_id"],
                [{"role": "user", "content": "Say hello in one short sentence."}],
                temperature=0,
                top_p=1.0,
                max_tokens=64,
            )
            print(
                json.dumps(
                    {
                        "slug": m["slug"],
                        "model": m["model_id"],
                        "error": r.error,
                        "answer": r.content,
                        "latency_ms": r.latency_ms,
                        "usage": {
                            "input_tokens": r.input_tokens,
                            "output_tokens": r.output_tokens,
                        },
                    },
                    ensure_ascii=False,
                )
            )
        return

    rows = read_jsonl(args.benchmark)
    seed_ids: list[str] = []
    seen: set[str] = set()
    for r in rows:
        sid = r["seed_id"]
        if sid not in seen:
            seen.add(sid)
            seed_ids.append(sid)
    if args.limit and args.limit > 0:
        seed_ids = seed_ids[: args.limit]
    seed_set = set(seed_ids)
    rows = [r for r in rows if r["seed_id"] in seed_set]

    by_seed_failure: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_seed_failure.setdefault((r["seed_id"], r["failure_type"]), []).append(r)

    conditions = ["clean"] + FAILURE_TYPES
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print(
        f"Models: {[m['slug'] for m in models]} | concurrency_per_model={concurrency} | "
        f"seeds={len(seed_ids)}"
    )

    for m in models:
        pending: list[tuple[dict, str, list[str]]] = []
        for sid in seed_ids:
            sample = next(r for r in rows if r["seed_id"] == sid)
            for cond in conditions:
                out_path = args.out / cond / f"{m['slug']}.jsonl"
                done = {
                    result_key(prev, prev.get("condition") or cond)
                    for prev in read_jsonl(out_path)
                }
                if cond == "clean":
                    key = result_key(sample, "clean")
                    if key not in done:
                        pending.append((sample, "clean", list(sample["clean_context"])))
                else:
                    for fr in by_seed_failure.get((sid, cond), []):
                        key = result_key(fr, cond)
                        if key not in done:
                            pending.append((fr, cond, list(fr["failure_context"])))

        print(f"{m['slug']}: {len(pending)} pending")
        run_model_tasks(
            m,
            pending,
            out_dir=args.out,
            decode=decode,
            prompts=prompts,
            concurrency=concurrency,
        )


if __name__ == "__main__":
    main()
