## Table A: Latency p50 / p95 (primary)
| Model | N | E2E p50 | E2E p95 | TTFT p50 | TTFT p95 | Cost/1K QA | Error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| deepseek | 809 | 938.8 | 2105.3 | 661.4 | 1280.8 | 0.7424 | 0.001 |
| gpt_oss | 809 | 1169.2 | 2823.5 | 1110.9 | 2369.9 | 0.2610 | 0.001 |
| llama | 810 | 589.2 | 1339.9 | 377.3 | 755.8 | 0.8343 | 0.000 |
| nova | 810 | 675.4 | 1075.8 | 469.2 | 817.9 | 0.9944 | 0.000 |
| qwen | 809 | 831.6 | 1386.1 | 630.6 | 863.5 | 0.2159 | 0.001 |

## Extended percentiles
| Model | E2E p90 | E2E p99 | E2E mean | TTFT mean | OTPS mean |
|---|---:|---:|---:|---:|---:|
| deepseek | 1738.2 | 3927.8 | 1168.3 | 824.6 | 38.39 |
| gpt_oss | 2199.0 | 3651.6 | 1386.6 | 1286.9 | 130.63 |
| llama | 1027.1 | 2684.3 | 718.1 | 449.3 | 59.02 |
| nova | 971.3 | 1338.1 | 705.0 | 521.1 | 53.18 |
| qwen | 1226.9 | 1950.1 | 895.4 | 655.9 | 43.60 |

## Failure-specific overhead
| Failure | N | E2E p50 | E2E p95 | Input tok mean | Output tok mean | Cost/req |
|---|---:|---:|---:|---:|---:|---:|
| boundary | 450 | 805.7 | 2037.7 | 529.7 | 62.7 | 0.000326 |
| clean | 447 | 796.8 | 1873.4 | 1542.4 | 56.9 | 0.000817 |
| conflict | 450 | 874.9 | 2238.8 | 779.1 | 69.9 | 0.000451 |
| evidence_position | 1350 | 803.7 | 1750.2 | 1546.1 | 58.3 | 0.000817 |
| hard_negative | 450 | 826.7 | 2069.1 | 1202.2 | 54.9 | 0.000646 |
| missing_evidence | 450 | 794.7 | 2089.2 | 381.5 | 60.6 | 0.000250 |
| noise | 450 | 782.5 | 1842.0 | 996.1 | 56.3 | 0.000548 |

_p50/p95 are percentiles over successful stream calls only (rows with error excluded). Each jsonl row is one call; these aggregates are the paper latency metrics._
_OTPS includes prefill wait. bedrock_latency_ms is null until Lambda returns it._
