# RAGFailBench data load

## Upstream sources

- https://github.com/taixingbi/RAGFailBench/tree/main/data/runs/pilot_stability_s42/6_final/failures
- https://github.com/taixingbi/RAGFailBench/tree/main/data/runs/pilot_stability_s123/6_final/failures
- https://github.com/taixingbi/RAGFailBench/tree/main/data/runs/pilot_stability_s2026/6_final/failures

Also downloads sibling `clean_seeds.jsonl` from each run’s `6_final/` directory.

Raw GitHub base:

```text
https://raw.githubusercontent.com/taixingbi/RAGFailBench/main/data/runs/pilot_stability_s{SEED}/6_final/
```

Files per run:

| Local path | Upstream |
|---|---|
| `data/raw/.../clean_seeds.jsonl` | `6_final/clean_seeds.jsonl` |
| `data/raw/.../failures/missing_evidence.jsonl` | failures/ |
| `data/raw/.../failures/conflict.jsonl` | failures/ |
| `data/raw/.../failures/hard_negative.jsonl` | failures/ |
| `data/raw/.../failures/chunk_boundary.jsonl` | → paired as `boundary` |
| `data/raw/.../failures/context_noise.jsonl` | → paired as `noise` |
| `data/raw/.../failures/evidence_position.jsonl` | failures/ |

## Reload command

```bash
python scripts/prepare_benchmark.py \
  --seeds 42,123,2026 \
  --severity medium \
  --pilot-n 20 \
  --force
```

`--force` re-downloads all raw JSONL even if local copies exist, then regenerates paired benchmark files.

## Last successful load

| Field | Value |
|---|---|
| Timestamp | 2026-08-04T18:32:18-0400 |
| Severity filter | medium (except `evidence_position`: keep low/medium/high for first/mid/last) |
| Command | `prepare_benchmark.py --seeds 42,123,2026 --severity medium --pilot-n 20 --force` |

### Output counts

| Output | Seeds | Rows |
|---|---:|---:|
| `data/benchmark/s42/paired.jsonl` | 75 | 600 |
| `data/benchmark/s123/paired.jsonl` | 67 | 536 |
| `data/benchmark/s2026/paired.jsonl` | 64 | 512 |
| `data/benchmark/pilot_s42_n20.jsonl` | 20 | 160 |

Row math: each seed contributes 5 medium failures + 3 evidence-position variants (= 8 rows/seed).

### Verification (this reload)

| Check | s42 | s123 | s2026 | pilot |
|---|---|---|---|---|
| Pairing integrity | 0 deviations | — | — | — |
| Boundary `gold_chunk_ids` empty | **0/75** | **0/67** | **0/64** | **0/20** |
| Boundary gold answer in labeled chunk | **75/75** | **66/67** | **64/64** | **20/20** |
| Conflict gold/conflict/alternate labels | 75/75 | — | — | — |
| Missing/hard-neg gold leak | 0/75 | — | — | — |

### Boundary `gold_chunk_ids` status

Upstream `chunk_boundary` now includes `gold_positions` (plus `split_pieces`, `distractor_positions`, `piece_to_position`).

Prepare maps `parameters.gold_positions` → `gold_chunk_ids`. If missing, falls back to deriving IDs from gold-answer / `split_sentence` overlap.

After this load: **0 empty** across all paired files. One s123 case has labels that do not contain an exact gold-answer substring (phrasing mismatch); evidence is still present in the split contexts.

## Name mapping

| Upstream failure file | Paired `failure_type` |
|---|---|
| `missing_evidence` | `missing_evidence` |
| `conflict` | `conflict` |
| `hard_negative` | `hard_negative` |
| `chunk_boundary` | `boundary` |
| `context_noise` | `noise` |
| `evidence_position` | `evidence_position` |

## Related docs

- [s42 paired analysis & paper impact](docs/s42_paired_analysis.md)
