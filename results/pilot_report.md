## Table 1: Clean Quality
| Model | Accuracy | Relevance | Faithfulness | Citation Accuracy |
|---|---:|---:|---:|---:|
| deepseek | 1.000 | 1.000 | 1.000 | 1.000 |
| gpt_oss | 1.000 | 1.000 | 1.000 | 1.000 |
| llama | 0.950 | 1.000 | 0.950 | 0.947 |
| nova | 1.000 | 1.000 | 1.000 | 1.000 |
| qwen | 0.950 | 0.947 | 0.947 | 1.000 |

## Table 2: Failure Accuracy
| Model | Missing | Conflict | Hard Negative | Boundary | Noise | Position |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 0.900 | 1.000 | 0.900 | 1.000 | 0.900 | 0.950 |
| gpt_oss | 0.850 | 0.950 | 0.900 | 0.800 | 1.000 | 1.000 |
| llama | 0.750 | 0.950 | 0.400 | 1.000 | 1.000 | 0.983 |
| nova | 0.850 | 0.700 | 0.900 | 0.950 | 1.000 | 0.983 |
| qwen | 0.900 | 0.200 | 0.750 | 0.800 | 0.650 | 0.833 |

## Table 3: Failure-specific Behavior
| Model | MAR | CRS | HNR | NRS | BRS | Position Drop |
|---|---:|---:|---:|---:|---:|---:|
| deepseek | 0.900 | 0.950 | 0.900 | 0.900 | 1.000 | 0.050 |
| gpt_oss | 0.850 | 0.850 | 0.900 | 1.000 | 0.800 | 0.000 |
| llama | 0.750 | 0.850 | 0.400 | 1.053 | 1.053 | 0.050 |
| nova | 0.850 | 0.700 | 0.900 | 1.000 | 0.950 | 0.050 |
| qwen | 0.900 | 0.150 | 0.750 | 0.684 | 0.842 | 0.150 |

## Table 4: Stability (FRS stub)
| Model | Mean FRS | Worst Failure | Best Failure |
|---|---:|---|---|
| deepseek | 0.942 | missing_evidence (0.900) | conflict (1.000) |
| gpt_oss | 0.917 | boundary (0.800) | noise (1.000) |
| llama | 0.847 | hard_negative (0.400) | boundary (1.000) |
| nova | 0.897 | conflict (0.700) | noise (1.000) |
| qwen | 0.689 | conflict (0.200) | missing_evidence (0.900) |

## Table 5: Cost-aware Selection (stub — costs not instrumented yet)
| Model | EnterpriseScore | Cost/1K | Latency | FRS per Dollar |
|---|---:|---:|---:|---:|
| deepseek | 0.927 | — | — | — |
| gpt_oss | 0.882 | — | — | — |
| llama | 0.806 | — | — | — |
| nova | 0.858 | — | — | — |
| qwen | 0.656 | — | — | — |

## Pilot gates
Parse rates: {
  "deepseek": 1.0,
  "gpt_oss": 0.9611111111111111,
  "llama": 0.9388888888888889,
  "nova": 1.0,
  "qwen": 0.9944444444444445
}
Error rates: {
  "deepseek": 0.0,
  "gpt_oss": 0.0,
  "llama": 0.0,
  "nova": 0.0,
  "qwen": 0.005555555555555556
}
Judge stats: {
  "n_judged": 900,
  "agreement_rate": 0.7577777777777778,
  "human_review_rate": 0.24222222222222223
}
