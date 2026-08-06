#!/usr/bin/env python3
"""Recompute model_cost_usd on latency records using costs.yaml + token sources.

Stream responses from bedrock-inference-mvp often omit usage. Prefer tokens from
the non-stream accuracy run (same sample_id/condition/model), which match billing
tokenizers. Falls back to tokens already on the latency row when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_costs(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_cost_usd(
    costs_cfg: dict,
    slug: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    rates = (costs_cfg.get("models") or {}).get(slug)
    if not rates:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    tin = float(input_tokens or 0)
    tout = float(output_tokens or 0)
    if "input_price_per_million" in rates or "output_price_per_million" in rates:
        pin = float(rates.get("input_price_per_million") or 0.0)
        pout = float(rates.get("output_price_per_million") or 0.0)
        return (tin / 1_000_000.0) * pin + (tout / 1_000_000.0) * pout
    pin = float(rates.get("prompt") or 0.0)
    pout = float(rates.get("completion") or 0.0)
    return (tin / 1000.0) * pin + (tout / 1000.0) * pout


def accuracy_token_index(results_dir: Path) -> dict[tuple[str, str, str, str | None], dict]:
    """Key: (model_slug, sample_id, condition, gold_position_str_or_None)."""
    idx: dict[tuple[str, str, str, str | None], dict] = {}
    if not results_dir.exists():
        return idx
    for path in results_dir.rglob("*.jsonl"):
        for row in read_jsonl(path):
            slug = row.get("model_slug") or path.stem
            sid = row.get("sample_id")
            cond = row.get("condition") or path.parent.name
            if not sid:
                continue
            pos = row.get("gold_position")
            pos_key = str(pos) if pos is not None and cond == "evidence_position" else None
            key = (str(slug), str(sid), str(cond), pos_key)
            if row.get("input_tokens") is None and row.get("output_tokens") is None:
                continue
            idx[key] = {
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            }
    return idx


def lookup_tokens(
    idx: dict[tuple[str, str, str, str | None], dict],
    row: dict,
) -> dict | None:
    slug = str(row.get("model_alias") or "")
    sid = str(row.get("sample_id") or "")
    cond = str(row.get("condition") or "")
    pos = row.get("gold_position")
    pos_key = str(pos) if pos is not None and cond == "evidence_position" else None
    hit = idx.get((slug, sid, cond, pos_key))
    if hit:
        return hit
    # fallback without position
    return idx.get((slug, sid, cond, None))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=ROOT / "results" / "latency_s42" / "latency_records.jsonl",
    )
    parser.add_argument(
        "--accuracy-results",
        type=Path,
        default=ROOT / "results" / "s42",
        help="Non-stream accuracy run dir to backfill tokens from",
    )
    parser.add_argument("--costs", type=Path, default=ROOT / "configs" / "costs.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: overwrite --records)",
    )
    args = parser.parse_args()

    costs_cfg = load_costs(args.costs)
    rows = read_jsonl(args.records)
    idx = accuracy_token_index(args.accuracy_results)

    n_backfill = 0
    n_costed = 0
    for row in rows:
        usage_source = row.get("usage_source")
        if row.get("input_tokens") is None or row.get("output_tokens") is None:
            hit = lookup_tokens(idx, row)
            if hit:
                row["input_tokens"] = hit["input_tokens"]
                row["output_tokens"] = hit["output_tokens"]
                row["usage_source"] = "accuracy_nonstream"
                row["usage_source_path"] = hit["source"]
                n_backfill += 1
                usage_source = "accuracy_nonstream"
            elif row.get("input_tokens") is not None or row.get("output_tokens") is not None:
                usage_source = usage_source or "latency_partial"
            else:
                usage_source = usage_source or None

        cost = model_cost_usd(
            costs_cfg,
            str(row.get("model_alias") or ""),
            row.get("input_tokens"),
            row.get("output_tokens"),
        )
        row["model_cost_usd"] = cost
        row["total_cost_usd"] = cost
        row["pricing_date"] = costs_cfg.get("pricing_date")
        row["pricing_region"] = costs_cfg.get("region")
        row["service_tier"] = costs_cfg.get("service_tier") or row.get("service_tier")
        if cost is not None:
            n_costed += 1

        e2e = row.get("e2e_latency_ms")
        out_tok = row.get("output_tokens")
        if e2e and out_tok is not None and float(e2e) > 0:
            row["output_tokens_per_second"] = round(float(out_tok) / (float(e2e) / 1000.0), 3)

        if usage_source:
            row["usage_source"] = usage_source

    out = args.out or args.records
    write_jsonl(out, rows)
    print(
        f"Updated {len(rows)} rows → {out}\n"
        f"  token backfill from accuracy: {n_backfill}\n"
        f"  rows with model_cost_usd: {n_costed}\n"
        f"  pricing_date={costs_cfg.get('pricing_date')} region={costs_cfg.get('region')} "
        f"tier={costs_cfg.get('service_tier')}"
    )


if __name__ == "__main__":
    main()
