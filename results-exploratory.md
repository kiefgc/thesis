# Exploratory analyses (protocol v1.0; no confirmatory claim)

266 tasks, 20 repetitions of stratified 5-fold, seed 42, 10000 bootstrap resamples. Handcrafted features. My arithmetic.

## 1. Hold flag: tasks neither model solves (protocol section 8)

Base rate of 'neither model solves': 29.3% (78 of 266 tasks).

AUC for predicting 'neither': Advisor-form classifier 0.632, repository-only 0.494, difference +0.138, DeLong p 0.004.

| Cutoff | Tasks flagged | Precision | 95% interval | Lift over base rate | Recall | 95% interval | Spend avoided (cheaper-first cost of flagged tasks) | 95% interval | Solves lost | 95% interval |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 112.1 | 0.383 | 0.299 to 0.470 | 1.31 | 0.551 | 0.452 to 0.647 | $80.03 | 63.27 to 97.92 | 69.2 | 56.8 to 81.9 |
| 0.5 | 15.2 | 0.525 | 0.324 to 0.720 | 1.79 | 0.103 | 0.055 to 0.155 | $12.93 | 7.22 to 19.55 | 7.2 | 3.6 to 11.5 |
| 0.7 | 0.2 | 0.333 | 0.000 to 1.000 | 1.14 | 0.001 | 0.000 to 0.002 | $0.10 | 0.00 to 0.27 | 0.1 | 0.0 to 0.2 |

Spend avoided and solves lost are reported separately and are not combined. Flagged tasks are assumed to be handled by the developer, whose time is not counted. A cutoff at which no task is flagged shows a precision of nan.

*This file was produced by a single run of `analysis_exploratory.py` on 2026-11-07 and has not been modified since.*

## 2. Price-ratio sensitivity (post hoc)

Actual average cost ratio (cheaper ÷ frontier) on these tasks: 0.140. For each ratio r the Kimi costs are scaled by r ÷ 0.140 and the routing threshold is r (send a task straight to the frontier model if the predicted chance of cheaper-model success is below r).

| Ratio r | Advisor: share routed to frontier | Advisor: cost per solved vs cheaper-first | 95% interval | Solve-rate diff (pp) | Scrouting head: share routed | Head: cost per solved vs cheaper-first | 95% interval | Oracle vs cheaper-first |
|---|---|---|---|---|---|---|---|---|
| 0.14 (actual) | 0.0% | -0.000 | -0.000 to +0.000 | +0.00 | 3.0% | +0.013 | +0.011 to +0.015 | -0.079 |
| 0.2 | 0.0% | +0.000 | -0.000 to +0.001 | -0.02 | 9.0% | +0.034 | +0.027 to +0.042 | -0.112 |
| 0.3 | 1.5% | +0.008 | -0.002 to +0.021 | -0.47 | 24.8% | +0.073 | +0.052 to +0.095 | -0.168 |
| 0.4 | 6.3% | +0.027 | -0.002 to +0.061 | -1.28 | 36.8% | +0.076 | +0.042 to +0.111 | -0.224 |
| 0.5 | 23.1% | +0.033 | -0.026 to +0.097 | -3.63 | 53.8% | +0.061 | +0.007 to +0.118 | -0.281 |
| 0.6 | 68.3% | +0.003 | -0.098 to +0.107 | -8.38 | 68.4% | +0.011 | -0.064 to +0.089 | -0.337 |
| 0.8 | 98.7% | -0.195 | -0.319 to -0.066 | -11.20 | 96.2% | -0.188 | -0.308 to -0.065 | -0.449 |
| 1.0 | 100.0% | -0.411 | -0.552 to -0.273 | -11.28 | 100.0% | -0.411 | -0.552 to -0.273 | -0.561 |

Lowest ratio tried at which the Advisor's interval lies wholly below zero (a saving over cheaper-first): **0.8**.
This table is post hoc and exploratory. It changes the prices, not the predictions, and it is not a claim about any real price.
