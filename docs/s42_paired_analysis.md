# Analysis: `data/benchmark/s42/paired.jsonl`

Status check of the seed-42 paired benchmark used for failure-aware RAG evaluation, plus implications for the paper.

Source construction: medium-severity cases from [RAGFailBench](https://github.com/taixingbi/RAGFailBench) `pilot_stability_s42`, joined so each `seed_id` keeps the same question / gold answer across clean and all six failure conditions.

---

## 1. Dataset snapshot

| Metric | Value |
|---|---|
| File | `data/benchmark/s42/paired.jsonl` |
| Rows | **600** |
| Unique seeds | **75** |
| Run seed | 42 |
| Pairing integrity | **0 deviations** (every seed has all six failure types) |

Row accounting:

```text
75 seeds × (5 single-severity failures + 3 evidence_position variants) = 600
```

`evidence_position` contributes 225 rows because each seed keeps three gold locations (first / middle / last).

---

## 2. Failure-type breakdown

| Failure type | Rows | Expected behavior | Failure contexts (typical) | Key labels present |
|---|---:|---|---:|---|
| `missing_evidence` | 75 | abstain | 1–2 | no gold in context (0/75 leaks) |
| `hard_negative` | 75 | abstain | 4 | distractors only; 0/75 gold leaks |
| `conflict` | 75 | answer | 4 | `gold_chunk_ids`, `conflict_chunk_ids`, `alternate_answer` |
| `boundary` | 75 | answer | 3 | split supporting span across chunks |
| `noise` | 75 | answer | 5 | `gold_position` ∈ {0..4}, `gold_chunk_ids` set |
| `evidence_position` | 225 | answer | 8 | `gold_position` ∈ {0, 4, 7} |

### Severity encoding

- Non-position failures in this file: **medium only**.
- `evidence_position`: severity encodes location, not an independent difficulty axis:
  - `low` → gold at index **0** (first)
  - `medium` → gold at index **4** (middle)
  - `high` → gold at index **7** (last)

All 75 seeds have the full `{0, 4, 7}` position set.

### Content mix (by seed)

Categories are roughly balanced (person / location / organization_product / historical_event / science_technology).  
Answer types skew toward **date** and **other**, with location / numeric / person / organization less frequent.

---

## 3. Quality checks

| Check | Result |
|---|---|
| Same `clean_context` shared across all failure rows of a seed | **75/75** |
| Conflict: alternate answer appears in conflict chunk | **75/75** |
| Conflict: gold answer substring appears in gold chunk | **73/75** (2 phrasing mismatches, evidence still present) |
| Missing / hard-negative: gold answer leaked into failure context | **0/75** each |
| Boundary: gold answer appears in some failure chunk | **75/75** |
| Boundary: `gold_chunk_ids` populated | **75/75** (from upstream `gold_positions`) |

### Known data issue (resolved)

Upstream `chunk_boundary` now emits `gold_positions` (reload 2026-08-04). Prepare maps these to `gold_chunk_ids` (with local overlap fallback if absent).

After reload: **0 empty** on s42 / s123 / s2026 / pilot. See [`data_load.md`](../data_load.md).

---

## 4. Schema fields used for evaluation

Each row links clean ↔ failure for paired comparison:

```json
{
  "run_seed": 42,
  "seed_id": "seed_000001",
  "failure_type": "conflict",
  "question": "...",
  "gold_answer": "...",
  "clean_context": ["..."],
  "failure_context": ["..."],
  "supporting_evidence": "...",
  "expected_behavior": "answer",
  "answer_available": true,
  "gold_chunk_ids": ["chunk_2"],
  "conflict_chunk_ids": ["chunk_1"],
  "alternate_answer": "...",
  "gold_position": null,
  "severity": "medium"
}
```

This enables **Clean → Failure** drops on the *same* questions, which is required for robustness metrics (NRS, BRS, PositionDrop, MAR, CRS, HNR).

---

## 5. Impact for the paper

### What this file strengthens

1. **Paired causal comparison**  
   Same `seed_id` / question / gold across conditions supports claims like “accuracy drops under noise” without confounding by different question sets.

2. **Failure-specific behavioral metrics**  
   - Missing / hard-negative → **MAR / HNR** (correct abstention)  
   - Conflict → **CRS** (prefer authoritative / gold chunk over alternate)  
   - Noise / boundary → **NRS / BRS** (Acc_failure / Acc_clean)  
   - Position → **PositionDrop** = Acc(first) − Acc(last), plus middle

3. **Multi-seed stability story**  
   s42 (75), s123 (67), s2026 (64) paired seeds allow reporting mean ± std across benchmark seeds and arguing that rankings are not an artifact of one draw.

4. **Enterprise model selection narrative**  
   Per-failure scores feed weighted `EnterpriseScore(M)` so the paper is not only a leaderboard but a risk-profile selector (e.g. conflict-heavy vs abstention-heavy deployments).

### What must be stated carefully in the paper

1. **Clean context is multi-chunk (8)** in the paired file, not a single gold passage.  
   Clean baseline is already a mild multi-context setting. Report this explicitly so reviewers do not assume “oracle single-chunk clean.”

2. **Position severity ≠ difficulty.**  
   In Methods, define low/medium/high for `evidence_position` as first/middle/last placement.

3. **Boundary citation labels**  
   Upstream now provides `gold_positions` for chunk-boundary splits (supporting pieces may span multiple chunk indices). Map these to `gold_chunk_ids` in prepare.

4. **Conflict string-match edge cases (2/75)**  
   Prefer normalized / judge-backed accuracy over brittle exact substring checks when validating conflict constructions.

5. **Pilot vs full scale**  
   Pilot used 20 of 75 s42 seeds. Full paper tables should use all paired seeds (and all three run seeds) before claiming significance / stability.

### Recommended Methods wording (draft)

> We evaluate models on paired clean and failure instances derived from RAGFailBench stability runs (`pilot_stability_s{42,123,2026}`). For each seed question we retain the gold answer and compare performance under six controlled context corruptions. Medium severity is used for all operators except evidence position, where three placements (first, middle, last) are retained to measure position sensitivity. Chunk-boundary cases label supporting chunk IDs from upstream `gold_positions` over the split pieces.

### Recommended Results tables this file supports

| Table | Uses from s42 paired |
|---|---|
| Table 1 Clean Quality | clean condition on paired seeds |
| Table 2 Failure Accuracy | six failure conditions |
| Table 3 Failure-specific Behavior | MAR, CRS, HNR, NRS, BRS, PositionDrop |
| Table 4 Stability | mean FRS / worst failure across seeds (with s123, s2026) |
| Table 5 Cost-aware Selection | EnterpriseScore weights over failure scores |

---

## 6. Action items

1. ~~Fix `boundary.gold_chunk_ids`~~ — done via derived remap on reload (see `data_load.md`).  
2. Re-run citation metrics for boundary after the fix (if eval was produced on older paired files).  
3. In the paper, document clean multi-chunk context and position↔severity mapping.  
4. Scale from pilot (n=20) to full paired sets before final claims.

---

## 7. Related artifacts

- Full paired sets: `data/benchmark/s{42,123,2026}/paired.jsonl`
- Pilot slice: `data/benchmark/pilot_s42_n20.jsonl`
- Pilot gate report: `results/pilot_gate_report.md`
- Prepare script: `scripts/prepare_benchmark.py`
