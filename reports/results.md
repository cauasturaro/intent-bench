| Model | Macro-F1 (95% CI) | Accuracy | Auto-routed | Error when auto-routed | Cost / 1k msgs | Latency median / p95 |
|---|---|---|---|---|---|---|
| tfidf_logreg | 0.887 (0.876–0.897) | 0.887 | 86.1% | 4.9% | $0.000 | 0.14 / 0.16 ms |
| distilbert | 0.920 (0.910–0.928) | 0.920 | 93.9% | 4.5% | $0.000 | 5.82 / 9.50 ms |
| llm_zero_shot | 0.756 (0.740–0.768) | 0.765 | 100.0% | 23.5% * | $0.052 | 1140.24 / 1450.45 ms |
| llm_few_shot | 0.856 (0.843–0.867) | 0.862 | 88.7% | 8.6% * | $0.634 | 1479.37 / 1690.57 ms |

\* No threshold reached the 5% error target on validation; the strictest one is used.
