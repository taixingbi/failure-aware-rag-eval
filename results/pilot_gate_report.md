# Pilot Gate Report (seed_42, n=20 paired seeds)

## Setup

| Item | Value |
|---|---|
| Benchmark | `data/benchmark/pilot_s42_n20.jsonl` (medium; position keeps first/mid/last) |
| Models | qwen, llama, gpt_oss, deepseek, nova |
| Claude | **skipped** (`enabled: false` until Bedrock/Anthropic approval) |
| Concurrency | **4 per model** (`concurrency_per_model`); Nova retry used 2 after throttle |
| Decode | temperature=0, top_p=1.0, max_tokens=256 (judge_max_tokens=512) |
| Inference calls | 5 × 180 = **900** |
| Judges | gpt-oss + deepseek (dual), agreement **75.8%** |

## Pilot gates

| Gate | Status | Notes |
|---|---|---|
| JSON parse ≥ ~90% | **PASS** | deepseek/nova 100%, qwen 99.4%, gpt_oss 96.1%, llama 93.9% |
| Citations parseable | **PASS** | structured `citations` field |
| Abstention unified | **PASS** | `INSUFFICIENT_EVIDENCE` / `abstained` |
| Lambda timeouts | **PASS** | no systemic timeouts |
| Model aliases | **PASS (5/6)** | Claude unavailable on this account |
| Judge accuracy vs faithfulness | **PASS** | dual judge separates fields; agreement 75.8% after `judge_max_tokens=512` |
| Small-model prompt | **WATCH** | Qwen CRS=0.15 — often follows conflicting evidence |

## Concurrency / throttling

- Default: 4 concurrent requests **per model** (models still run sequentially).
- Nova first pass hit `Too many requests` (~36%); client now backs off on throttle; Nova re-run at concurrency=2 → 180/180 clean.
- Recommendation: keep concurrency=4 globally; rely on retry/backoff for Nova, or cap Nova at 2.

## Skip / re-enable Claude

```yaml
# configs/models.yaml
- slug: claude
  model_id: claude-sonnet-5
  enabled: false   # set true after Bedrock Anthropic approval
```

```bash
python scripts/run_benchmark.py --models claude --concurrency 4 \
  --benchmark data/benchmark/pilot_s42_n20.jsonl --out results/pilot_s42
```

## Key pilot metrics (final)

Clean: all models ≥0.95 accuracy; deepseek/gpt_oss/nova ≈1.0 on clean quality.

Worst failure modes:

| Model | Weakest | Score |
|---|---|---|
| qwen | conflict | 0.20 |
| llama | hard_negative | 0.40 |
| nova | conflict | 0.70 |
| gpt_oss | boundary | 0.80 |
| deepseek | missing_evidence | 0.90 |

EnterpriseScore (config weights): deepseek 0.927 > gpt_oss 0.882 > nova 0.858 > llama 0.806 > qwen 0.656

## Blockers before full 3-seed run

1. Approve Claude on Bedrock; set `claude.enabled: true`
2. Optionally restore Claude as primary judge once available
3. Human-review `results/eval/pilot_s42/human_review_sample.jsonl` (90 rows; ~24% flagged disagreement)
4. Full scale: ~64–75 paired seeds × 3 run seeds × 5–6 models
5. Add cost/latency into Table 5 from recorded `latency_ms` / token usage

## Artifacts

- Results: `results/pilot_s42/{condition}/{model}.jsonl`
- Eval: `results/eval/pilot_s42/{summary.json,evaluated.jsonl,judge_cache.jsonl}`
- Tables: `results/pilot_report.md`
