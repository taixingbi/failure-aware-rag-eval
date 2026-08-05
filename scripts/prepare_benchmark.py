#!/usr/bin/env python3
"""Download RAGFailBench runs and emit paired medium-severity benchmark JSONL."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW_BASE = "https://raw.githubusercontent.com/taixingbi/RAGFailBench/main/data/runs"

# source filename -> canonical failure type name used in this paper
FAILURE_MAP = {
    "missing_evidence": "missing_evidence",
    "conflict": "conflict",
    "hard_negative": "hard_negative",
    "chunk_boundary": "boundary",
    "context_noise": "noise",
    "evidence_position": "evidence_position",
}

FAILURE_TYPES = list(FAILURE_MAP.values())


def download(url: str, dest: Path, *, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not force and dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as f:
        f.write(resp.read())


def read_jsonl(path: Path) -> list[dict]:
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


def chunk_ids(n: int) -> list[str]:
    return [f"chunk_{i}" for i in range(1, n + 1)]


def positions_to_chunk_ids(positions: list[int] | None, n: int) -> list[str]:
    ids = chunk_ids(n)
    out = []
    for p in positions or []:
        if isinstance(p, int) and 0 <= p < n:
            out.append(ids[p])
    return out


def derive_boundary_gold_chunk_ids(
    contexts: list[str],
    *,
    gold_answer: str | None,
    split_sentence: str | None,
) -> list[str]:
    """RAGFailBench chunk_boundary has no gold_position; recover supporting chunk ids.

    Prefer chunks that contain the gold answer; otherwise chunks that overlap the
    split supporting sentence (case-insensitive substring).
    """
    if not contexts:
        return []
    gold = (gold_answer or "").strip().lower()
    split = (split_sentence or "").strip().lower()

    hits: list[str] = []
    if gold:
        for i, ctx in enumerate(contexts, start=1):
            if gold in (ctx or "").lower():
                hits.append(f"chunk_{i}")
    if hits:
        return hits

    # Fallback: any chunk overlapping a substantial piece of the split sentence
    if split and len(split) >= 12:
        # use mid-length window so short split fragments still match
        needle = split[: max(40, min(80, len(split)))]
        for i, ctx in enumerate(contexts, start=1):
            text = (ctx or "").lower()
            if needle in text or (len(text) >= 20 and text in split):
                hits.append(f"chunk_{i}")
    if hits:
        return hits

    # Last resort for boundary: all split pieces jointly carry the evidence
    return chunk_ids(len(contexts))


def unify_failure(run_seed: int, clean: dict, failure: dict, canonical_type: str) -> dict:
    contexts = failure.get("contexts") or []
    n = len(contexts)
    params = failure.get("parameters") or {}
    gold_positions = params.get("gold_positions")
    if gold_positions is None and params.get("gold_position") is not None:
        gold_positions = [params["gold_position"]]
    conflict_positions = params.get("conflict_positions") or []

    gold_chunk_ids = positions_to_chunk_ids(gold_positions, n)
    conflict_chunk_ids = positions_to_chunk_ids(conflict_positions, n)

    # for types without explicit gold_positions, if answer is available assume chunk_1
    # holds supporting evidence when only one supporting span is known (noise/position)
    if not gold_chunk_ids and failure.get("answer_available") and n > 0:
        gp = params.get("gold_position")
        if gp is not None and 0 <= int(gp) < n:
            gold_chunk_ids = [f"chunk_{int(gp) + 1}"]

    # boundary: original operator has no gold_position — derive from contexts
    if canonical_type == "boundary" and not gold_chunk_ids:
        gold_chunk_ids = derive_boundary_gold_chunk_ids(
            contexts,
            gold_answer=failure.get("gold_answer") or clean.get("gold_answer"),
            split_sentence=params.get("split_sentence"),
        )

    return {
        "run_seed": run_seed,
        "seed_id": failure.get("parent_seed_id") or clean.get("sample_id"),
        "failure_type": canonical_type,
        "question": failure.get("question") or clean.get("question"),
        "gold_answer": failure.get("gold_answer") or clean.get("gold_answer"),
        "clean_context": list(clean.get("clean_contexts") or []),
        "failure_context": list(contexts),
        "supporting_evidence": failure.get("supporting_sentence")
        or clean.get("supporting_sentence"),
        "expected_behavior": failure.get("expected_behavior") or "answer",
        "answer_available": bool(failure.get("answer_available")),
        "gold_chunk_ids": gold_chunk_ids,
        "conflict_chunk_ids": conflict_chunk_ids,
        "alternate_answer": params.get("alternate_answer"),
        "gold_position": params.get("gold_position"),
        "num_contexts": params.get("num_contexts", n),
        "severity": failure.get("severity"),
        "source_failure_id": failure.get("failure_id"),
        "category_group": failure.get("category_group") or clean.get("category_group"),
        "answer_type": clean.get("answer_type"),
    }


def prepare_seed(
    run_seed: int,
    raw_dir: Path,
    out_dir: Path,
    severity: str,
    *,
    force: bool = False,
) -> list[dict]:
    run_name = f"pilot_stability_s{run_seed}"
    base = f"{RAW_BASE}/{run_name}/6_final"
    local = raw_dir / run_name / "6_final"

    clean_path = local / "clean_seeds.jsonl"
    download(f"{base}/clean_seeds.jsonl", clean_path, force=force)
    cleans = {r["sample_id"]: r for r in read_jsonl(clean_path)}

    # Most failures: one row per parent at the chosen severity.
    # evidence_position: severity encodes position (low=first, medium=middle, high=last),
    # so keep all three for PositionDrop analysis, keyed by parent for intersection.
    failures_by_type: dict[str, dict[str, dict]] = {}
    position_by_parent: dict[str, list[dict]] = {}

    for src_name, canon in FAILURE_MAP.items():
        path = local / "failures" / f"{src_name}.jsonl"
        download(f"{base}/failures/{src_name}.jsonl", path, force=force)
        all_rows = read_jsonl(path)

        if canon == "evidence_position":
            by_parent: dict[str, list[dict]] = {}
            for r in all_rows:
                by_parent.setdefault(r["parent_seed_id"], []).append(r)
            # require all three positions for a parent to count
            position_by_parent = {
                pid: sorted(
                    rs,
                    key=lambda x: (
                        (x.get("parameters") or {}).get("gold_position") is None,
                        (x.get("parameters") or {}).get("gold_position", -1),
                    ),
                )
                for pid, rs in by_parent.items()
                if len({(x.get("parameters") or {}).get("gold_position") for x in rs}) >= 3
            }
            failures_by_type[canon] = {pid: rs[0] for pid, rs in position_by_parent.items()}
        else:
            rows = [r for r in all_rows if r.get("severity") == severity]
            by_parent_one: dict[str, dict] = {}
            for r in rows:
                pid = r["parent_seed_id"]
                if pid not in by_parent_one:
                    by_parent_one[pid] = r
            failures_by_type[canon] = by_parent_one

    parent_sets = [set(m.keys()) for m in failures_by_type.values()]
    intersection = set.intersection(*parent_sets) if parent_sets else set()
    intersection &= set(cleans.keys())
    ordered = sorted(intersection)

    paired: list[dict] = []
    for seed_id in ordered:
        clean = cleans[seed_id]
        for canon in FAILURE_TYPES:
            if canon == "evidence_position":
                for fr in position_by_parent[seed_id]:
                    rec = unify_failure(run_seed, clean, fr, canon)
                    # disambiguate sample for runner resume keys
                    rec["position_variant"] = rec.get("gold_position")
                    paired.append(rec)
            else:
                paired.append(
                    unify_failure(run_seed, clean, failures_by_type[canon][seed_id], canon)
                )

    out_path = out_dir / f"s{run_seed}" / "paired.jsonl"
    write_jsonl(out_path, paired)
    print(
        f"s{run_seed}: {len(ordered)} paired seeds × {len(FAILURE_TYPES)} failures "
        f"= {len(paired)} rows → {out_path}"
    )
    return paired


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--severity", default="medium")
    parser.add_argument("--pilot-n", type=int, default=20)
    parser.add_argument("--pilot-seed", type=int, default=42)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "benchmark")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download RAGFailBench raw jsonl even if local files exist",
    )
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    all_paired: dict[int, list[dict]] = {}
    for seed in seeds:
        all_paired[seed] = prepare_seed(
            seed, args.raw_dir, args.out_dir, args.severity, force=args.force
        )

    # pilot slice for pilot_seed
    pilot_rows = all_paired.get(args.pilot_seed) or prepare_seed(
        args.pilot_seed, args.raw_dir, args.out_dir, args.severity, force=args.force
    )
    pilot_seed_ids = sorted({r["seed_id"] for r in pilot_rows})[: args.pilot_n]
    pilot_set = set(pilot_seed_ids)
    pilot = [r for r in pilot_rows if r["seed_id"] in pilot_set]
    pilot_path = args.out_dir / f"pilot_s{args.pilot_seed}_n{args.pilot_n}.jsonl"
    write_jsonl(pilot_path, pilot)
    print(
        f"Pilot: {len(pilot_seed_ids)} seeds × "
        f"{len(pilot) // max(1, len(pilot_seed_ids))} rows/seed = {len(pilot)} → {pilot_path}"
    )


if __name__ == "__main__":
    main()
