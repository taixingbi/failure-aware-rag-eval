# Enterprise Model Selection Decision Matrix

This document explains how to choose a Bedrock-hosted RAG model for an enterprise application using failure-aware quality metrics, latency, and cost.

## Inputs

- `results/eval/<seed>/summary.json`: per-model clean/failure metrics and performance statistics.
- `configs/enterprise_weights.yaml`: weights for failure-specific enterprise priorities.
- `configs/costs.yaml`: Bedrock prompt and completion token pricing.

## Decision matrix logic

1. Compute the core enterprise score for each model
   - Use `MAR` for `missing_evidence`
   - Use `CRS` for `conflict`
   - Use `HNR` for `hard_negative`
   - Use `NRS` for `noise`
   - Use `BRS` for `boundary`
   - Use `failure_accuracy.evidence_position` for `evidence_position`
   - Weight each axis using `enterprise_weights.yaml`

2. Compute cost and latency
   - Average latency from `performance.latency_ms`
   - Average token usage from `performance.input_tokens` / `performance.output_tokens`
   - Bedrock cost per request = prompt tokens × prompt rate + completion tokens × completion rate

3. Normalize cost and compute utility
   - Normalize cost by the maximum observed cost across candidate models
   - Utility = EnterpriseScore − `cost_lambda` × NormalizedCost

4. Rank models by enterprise utility or FRS per dollar
   - Use `FRS / $` for strict cost-sensitive procurement.
   - Use `Utility` to compare enterprise risk-adjusted quality.

## Practical usage

```bash
python scripts/aggregate_summaries.py --summaries results/eval/s42 results/eval/s123 results/eval/s2026
```

If you want a quick ablation of severity effects, run:

```bash
python scripts/ablation_analysis.py --evals results/eval/s42 results/eval/s123 results/eval/s2026
```

## When to use this matrix

- Choose a model for a deployment where failure mode sensitivity matters more than raw clean accuracy.
- Compare models on high-risk conditions such as missing evidence or conflict rather than only average accuracy.
- Produce a shortlist for enterprise buyers with cost and latency transparently reported.
