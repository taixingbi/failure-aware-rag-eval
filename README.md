# Failure-Aware Evaluation of Retrieval-Augmented Generation Models

Pilot pipeline for clean vs failure-conditioned RAG evaluation across six Bedrock-hosted models.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set FUNCTION_URL and INFERENCE_API_KEY in .env
```

```bash
export FUNCTION_URL=$(aws cloudformation describe-stacks \
  --region us-east-1 \
  --stack-name bedrock-inference-mvp \
  --query "Stacks[0].Outputs[?OutputKey=='InferenceFunctionUrl'].OutputValue" \
  --output text)
export INFERENCE_API_KEY=1234
```

## Pipeline

```bash
# 1) Download + pair clean↔failure cases
python scripts/prepare_benchmark.py --seeds 42,123,2026 --severity medium --pilot-n 20

# 1a) Optional ablation datasets over all severity levels
python scripts/prepare_benchmark.py --seeds 42,123,2026 --severity low,medium,high

# 2) Smoke-test model aliases
python scripts/run_benchmark.py --smoke

# 3) Full seed run examples
# s42 is the currently documented completed seed; s123 and s2026 are included here as explicit pending runs.
python scripts/run_benchmark.py \
  --benchmark data/benchmark/s42/medium/paired.jsonl \
  --out results/s42

# Pending full runs for s123 and s2026
python scripts/run_benchmark.py \
  --benchmark data/benchmark/s123/medium/paired.jsonl \
  --out results/s123

python scripts/run_benchmark.py \
  --benchmark data/benchmark/s2026/medium/paired.jsonl \
  --out results/s2026

# 4) Evaluate (rule metrics + dual LLM judge)
# Run these once the corresponding seed outputs exist; s123 and s2026 are pending until the commands above finish.
python scripts/evaluate_results.py \
  --results results/s42 \
  --out results/eval/s42

python scripts/evaluate_results.py \
  --results results/s123 \
  --out results/eval/s123

python scripts/evaluate_results.py \
  --results results/s2026 \
  --out results/eval/s2026

# 5) Aggregate across run seeds with mean ± std
python scripts/aggregate_summaries.py --summaries results/eval/s42 results/eval/s123 results/eval/s2026

# 6) Run ablation analyses for noise/hard-negative/context-length effects
python scripts/ablation_analysis.py --evals results/eval/s42 results/eval/s123 results/eval/s2026

# 7) RQ5 latency/cost on the same RAG workload (streaming; concurrency=1)
# Smoke stream TTFT/E2E for each enabled model
python scripts/run_latency_cost.py --smoke

# Default: 30 seeds × conditions × 5 models × 3 reps (budget-friendly)
python scripts/run_latency_cost.py \
  --benchmark data/benchmark/s42/medium/paired.jsonl \
  --out results/latency_s42 \
  --limit 30 --reps 3 --warmup 5

python scripts/summarize_latency.py --records results/latency_s42/latency_records.jsonl
# → results/latency_s42/latency_summary.json  (E2E/TTFT p50/p95 per model)
# → results/latency_s42/latency_report.md

# After a latency run (or if stream omitted usage), backfill tokens from accuracy + apply costs.yaml
python scripts/recompute_latency_costs.py \
  --records results/latency_s42/latency_records.jsonl \
  --accuracy-results results/s42 \
  --costs configs/costs.yaml
```

Paper latency metrics are **percentiles over successful calls** (`e2e_latency_ms.p50/p95`, `ttft_ms.p50/p95`), not single-row values and not mean-alone.

Client-measured stream metrics:

- `ttft_ms`: request start → first non-empty `delta.content`
- `e2e_latency_ms`: request start → stream `[DONE]`
- `bedrock_latency_ms`: optional (null until Lambda returns it)
- Cost uses Bedrock Standard rates in `configs/costs.yaml`; when stream omits `usage`, tokens are joined from the non-stream accuracy run (`usage_source: accuracy_nonstream`)

Serving cost only (judge / evaluation cost excluded). Fill/verify `configs/costs.yaml` from the [Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) (`pricing_date`, `us-east-1`, Standard tier).

## Models

| Slug | API model id |
|---|---|
| qwen | qwen3-next-80b-a3b |
| llama | llama |
| gpt_oss | gpt-oss |
| claude | claude-sonnet-5 |
| deepseek | deepseek |
| nova | nova-pro |

Shared decode: `temperature=0`, `top_p=1.0`, `max_tokens=256`.  
Concurrency: **4 in-flight requests per model** (`concurrency_per_model` in `configs/models.yaml`).

Claude Sonnet is temporarily `enabled: false` until Bedrock/Anthropic marketplace access is approved.

## Pricing and enterprise selection

- Fill `configs/costs.yaml` with Bedrock per-model rates (`input_price_per_million` / `output_price_per_million`, plus `pricing_date`).
- Use `scripts/aggregate_summaries.py` to compute enterprise-weighted decision scores, cost per request, and FRS-per-dollar.
- Use `scripts/run_latency_cost.py` + `scripts/summarize_latency.py` for RQ5 quality–latency–cost tables on the **same** RAG prompts (not toy hellos).

## Benchmark source

[RAGFailBench](https://github.com/taixingbi/RAGFailBench) runs:

- `pilot_stability_s42`
- `pilot_stability_s123`
- `pilot_stability_s2026`

Severity filter: **low/medium/high** options are available. Failure name mapping: `chunk_boundary→boundary`, `context_noise→noise`.

## Docs

- [s42 paired data analysis & paper impact](docs/s42_paired_analysis.md)
- [Enterprise model selection decision matrix](docs/enterprise_decision_matrix.md)
