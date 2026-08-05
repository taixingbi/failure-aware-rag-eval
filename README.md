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
# 1) Download + pair medium-severity clean↔failure cases
python scripts/prepare_benchmark.py --seeds 42,123,2026 --severity medium --pilot-n 20

# 2) Smoke-test model aliases
python scripts/run_benchmark.py --smoke

# 3) Pilot run (seed 42, 20 paired seeds × clean+6 failures × 6 models ≈ 840 calls)
python scripts/run_benchmark.py \
  --benchmark data/benchmark/pilot_s42_n20.jsonl \
  --out results/pilot_s42

# 4) Evaluate (rule metrics + dual LLM judge)
python scripts/evaluate_results.py \
  --results results/pilot_s42 \
  --out results/eval/pilot_s42

# 5) Print tables
python scripts/summarize_tables.py
```

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

## Benchmark source

[RAGFailBench](https://github.com/taixingbi/RAGFailBench) runs:

- `pilot_stability_s42`
- `pilot_stability_s123`
- `pilot_stability_s2026`

Severity filter: **medium** only. Failure name mapping: `chunk_boundary→boundary`, `context_noise→noise`.

## Docs

- [s42 paired data analysis & paper impact](docs/s42_paired_analysis.md)
