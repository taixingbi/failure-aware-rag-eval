#!/usr/bin/env python3
"""RQ5 latency/cost run on the same RAG benchmark workload (streaming).

Measures client-visible:
  - e2e_latency_ms: request start → stream complete ([DONE])
  - ttft_ms: request start → first non-empty delta.content

Uses concurrency=1, optional warm-up, round-robin across models, and N reps.
Does not count judge calls (serving cost only).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
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
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_models(cfg_path: Path, slugs: str) -> tuple[list[dict], dict]:
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    models = list(cfg["models"])
    if slugs:
        wanted = {s.strip() for s in slugs.split(",") if s.strip()}
        models = [m for m in models if m["slug"] in wanted]
    else:
        models = [m for m in models if m.get("enabled", True)]
    return models, cfg["decode"]


def load_costs(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_cost_usd(
    costs_cfg: dict,
    slug: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    models = costs_cfg.get("models") or {}
    rates = models.get(slug)
    if not rates:
        return None
    # Prefer per-1M if set; else per-1K prompt/completion (legacy stub).
    if "input_price_per_million" in rates or "output_price_per_million" in rates:
        pin = float(rates.get("input_price_per_million") or 0.0)
        pout = float(rates.get("output_price_per_million") or 0.0)
        tin = float(input_tokens or 0)
        tout = float(output_tokens or 0)
        return (tin / 1_000_000.0) * pin + (tout / 1_000_000.0) * pout
    pin = float(rates.get("prompt") or 0.0)
    pout = float(rates.get("completion") or 0.0)
    tin = float(input_tokens or 0)
    tout = float(output_tokens or 0)
    return (tin / 1000.0) * pin + (tout / 1000.0) * pout


def result_key(sample: dict, cond: str, model_slug: str, rep: int) -> str:
    sid = sample.get("seed_id") or sample.get("sample_id")
    if cond == "evidence_position":
        return f"{sid}__pos{sample.get('gold_position')}__{model_slug}__r{rep}"
    return f"{sid}__{cond}__{model_slug}__r{rep}"


def build_tasks(
    rows: list[dict],
    seed_ids: list[str],
    conditions: list[str],
) -> list[tuple[dict, str, list[str]]]:
    by_seed_failure: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_seed_failure.setdefault((r["seed_id"], r["failure_type"]), []).append(r)

    tasks: list[tuple[dict, str, list[str]]] = []
    for sid in seed_ids:
        sample = next(r for r in rows if r["seed_id"] == sid)
        for cond in conditions:
            if cond == "clean":
                tasks.append((sample, "clean", list(sample["clean_context"])))
            else:
                for fr in by_seed_failure.get((sid, cond), []):
                    tasks.append((fr, cond, list(fr["failure_context"])))
    return tasks


def warm_up(client: InferenceClient, models: list[dict], n: int, decode: dict) -> None:
    if n <= 0:
        return
    print(f"Warm-up: {n} request(s) per model (not recorded)")
    for m in models:
        for _ in range(n):
            client.chat_completions_stream(
                m["model_id"],
                [{"role": "user", "content": "Say hello in one short sentence."}],
                temperature=float(decode.get("temperature", 0)),
                top_p=float(decode.get("top_p", 1.0)),
                max_tokens=min(64, int(decode.get("max_tokens", 256))),
            )


def run_one(
    *,
    client: InferenceClient,
    model: dict,
    sample: dict,
    condition: str,
    contexts: list[str],
    decode: dict,
    prompts: dict,
    costs_cfg: dict,
    run_id: str,
    rep: int,
    region: str,
    service_tier: str,
) -> dict:
    auth_ids = None
    if condition == "conflict":
        auth_ids = sample.get("gold_chunk_ids") or []

    messages = build_answer_messages(
        sample["question"],
        contexts,
        authoritative_chunk_ids=auth_ids,
        prompts=prompts,
    )
    request_start = datetime.now(timezone.utc).isoformat()
    result = client.chat_completions_stream(
        model["model_id"],
        messages,
        temperature=float(decode.get("temperature", 0)),
        top_p=float(decode.get("top_p", 1.0)),
        max_tokens=int(decode.get("max_tokens", 256)),
    )
    parsed = parse_answer_payload(result.content)
    e2e = float(result.latency_ms)
    ttft = float(result.ttft_ms) if result.ttft_ms is not None else None
    out_tok = result.output_tokens
    otps = None
    if out_tok is not None and e2e > 0:
        # Approximate; includes prefill wait (not pure decode speed).
        otps = out_tok / (e2e / 1000.0)
    cost = model_cost_usd(costs_cfg, model["slug"], result.input_tokens, result.output_tokens)

    return {
        "run_id": run_id,
        "benchmark_seed": sample.get("run_seed"),
        "sample_id": sample.get("seed_id"),
        "condition": condition,
        "failure_type": sample.get("failure_type") if condition != "clean" else None,
        "severity": sample.get("severity"),
        "gold_position": sample.get("gold_position"),
        "model_alias": model["slug"],
        "exact_model_id": result.model or model["model_id"],
        "region": region,
        "service_tier": service_tier,
        "rep": rep,
        "request_start": request_start,
        "e2e_latency_ms": round(e2e, 3),
        "ttft_ms": round(ttft, 3) if ttft is not None else None,
        "bedrock_latency_ms": None,  # optional; requires Lambda instrumentation
        "gateway_overhead_ms": None,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "output_tokens_per_second": round(otps, 3) if otps is not None else None,
        "model_cost_usd": cost,
        "lambda_cost_usd": None,
        "total_cost_usd": cost,
        "http_status": result.http_status,
        "retry_count": result.retry_count,
        "throttled": bool(result.error and "throttl" in str(result.error).lower()),
        "chunk_count": result.chunk_count,
        "parse_ok": parsed.get("parse_ok"),
        "answer": parsed.get("answer"),
        "citations": parsed.get("citations"),
        "abstained": parsed.get("abstained"),
        "error": result.error,
        "streamed": True,
    }


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


def print_quick_summary(rows: list[dict]) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("error"):
            continue
        by_model.setdefault(r["model_alias"], []).append(r)

    print("\n## Quick latency summary (successful stream calls)")
    print("| Model | N | E2E p50 | E2E p95 | TTFT p50 | TTFT p95 | Err rate |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    all_rows_by_model: dict[str, list[dict]] = {}
    for r in rows:
        all_rows_by_model.setdefault(r["model_alias"], []).append(r)
    for slug in sorted(all_rows_by_model):
        all_m = all_rows_by_model[slug]
        ok = by_model.get(slug, [])
        e2e = [float(r["e2e_latency_ms"]) for r in ok if r.get("e2e_latency_ms") is not None]
        ttft = [float(r["ttft_ms"]) for r in ok if r.get("ttft_ms") is not None]
        err_rate = 1.0 - (len(ok) / len(all_m)) if all_m else 1.0

        def fmt(v: float | None) -> str:
            return f"{v:.1f}" if v is not None else "—"

        print(
            f"| {slug} | {len(ok)} | {fmt(percentile(e2e, 50))} | {fmt(percentile(e2e, 95))} | "
            f"{fmt(percentile(ttft, 50))} | {fmt(percentile(ttft, 95))} | {err_rate:.3f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "benchmark" / "s42" / "medium" / "paired.jsonl",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "latency_s42")
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--costs", type=Path, default=ROOT / "configs" / "costs.yaml")
    parser.add_argument("--models", default="", help="Comma-separated slugs (default: enabled)")
    parser.add_argument("--conditions", default="", help="Comma-separated (default: clean+6 failures)")
    parser.add_argument("--limit", type=int, default=30, help="Unique seeds (0=all; default 30)")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per case×model")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up requests per model")
    parser.add_argument("--run-id", default="", help="Run id tag (default: auto)")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--service-tier", default="standard")
    parser.add_argument("--smoke", action="store_true", help="One streamed hello per model")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    models, decode = load_models(args.models_config, args.models)
    costs_cfg = load_costs(args.costs)
    prompts = load_prompts()
    client = InferenceClient()

    run_id = args.run_id or f"latency_cost_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_path = args.out / "latency_records.jsonl"
    meta_path = args.out / "run_meta.json"

    if args.smoke:
        for m in models:
            r = client.chat_completions_stream(
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
                        "e2e_latency_ms": r.latency_ms,
                        "ttft_ms": r.ttft_ms,
                        "chunk_count": r.chunk_count,
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

    conditions = ["clean"] + FAILURE_TYPES
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    tasks = build_tasks(rows, seed_ids, conditions)
    done = {result_key_from_row(prev) for prev in read_jsonl(out_path)}

    meta = {
        "run_id": run_id,
        "benchmark": str(args.benchmark),
        "seeds": len(seed_ids),
        "conditions": conditions,
        "models": [m["slug"] for m in models],
        "reps": args.reps,
        "warmup": args.warmup,
        "concurrency": 1,
        "stream": True,
        "region": args.region,
        "service_tier": args.service_tier,
        "decode": decode,
        "pricing_meta": {
            "pricing_date": costs_cfg.get("pricing_date"),
            "region": costs_cfg.get("region", args.region),
            "service_tier": costs_cfg.get("service_tier", args.service_tier),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"RQ5 latency run {run_id} | models={[m['slug'] for m in models]} | "
        f"seeds={len(seed_ids)} | tasks/case-set={len(tasks)} | reps={args.reps} | concurrency=1"
    )
    warm_up(client, models, args.warmup, decode)

    # Round-robin: for each task, cycle model order by task index; then reps.
    recorded: list[dict] = []
    pending_jobs: list[tuple[dict, str, list[str], dict, int]] = []
    for ti, (sample, cond, contexts) in enumerate(tasks):
        order = models[ti % len(models) :] + models[: ti % len(models)]
        for rep in range(1, args.reps + 1):
            for m in order:
                key = result_key(sample, cond, m["slug"], rep)
                if key in done:
                    continue
                pending_jobs.append((sample, cond, contexts, m, rep))

    for sample, cond, contexts, m, rep in tqdm(pending_jobs, desc="latency", leave=True):
        rec = run_one(
            client=client,
            model=m,
            sample=sample,
            condition=cond,
            contexts=contexts,
            decode=decode,
            prompts=prompts,
            costs_cfg=costs_cfg,
            run_id=run_id,
            rep=rep,
            region=args.region,
            service_tier=args.service_tier,
        )
        append_jsonl(out_path, rec)
        recorded.append(rec)
        # Be gentle on shared quotas even at concurrency=1.
        time.sleep(0.05)

    all_rows = read_jsonl(out_path)
    print_quick_summary(all_rows)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["n_records"] = len(all_rows)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(all_rows)} records)")


def result_key_from_row(row: dict) -> str:
    sid = row.get("sample_id")
    cond = row.get("condition") or "clean"
    slug = row.get("model_alias")
    rep = row.get("rep", 1)
    if cond == "evidence_position":
        return f"{sid}__pos{row.get('gold_position')}__{slug}__r{rep}"
    return f"{sid}__{cond}__{slug}__r{rep}"


if __name__ == "__main__":
    main()
