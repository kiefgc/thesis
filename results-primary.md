# Primary analysis: handcrafted Advisor (protocol v1.0)

266 tasks, 20 repetitions of stratified 5-fold, seed 42, 10000 bootstrap resamples. f = 0 (no false-solve correction). My arithmetic.

## 1. Policies (primary analysis)

| Policy | Spend | Solved | Cost per solved | 95% interval | Difference vs C | 95% interval | Solve-rate diff vs C (pp) | 95% interval |
|---|---|---|---|---|---|---|---|---|
| A Frontier for all | $201.25 | 158.0 | $1.274 | 1.110 to 1.471 | +0.509 | +0.403 to +0.631 | -11.28 | -15.04 to -7.52 |
| B Cheaper for all | $28.23 | 149.0 | $0.189 | 0.165 to 0.219 | -0.575 | -0.708 to -0.461 | -14.66 | -19.17 to -10.53 |
| C Cheaper first | $143.77 | 188.0 | $0.765 | 0.630 to 0.922 | +0.000 | +0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| D Advisor | $143.76 | 188.0 | $0.765 | 0.630 to 0.922 | -0.000 | -0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| E Repository-only | $143.77 | 188.0 | $0.765 | 0.630 to 0.922 | +0.000 | +0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| F Existing-practice rule | $175.91 | 166.4 | $1.057 | 0.897 to 1.247 | +0.292 | +0.201 to +0.397 | -8.12 | -11.50 to -4.96 |
| G Scrouting text head | $145.06 | 188.0 | $0.772 | 0.638 to 0.929 | +0.007 | -0.001 to +0.016 | +0.00 | +0.00 to +0.00 |
| H Oracle (ceiling) | $128.97 | 188.0 | $0.686 | 0.566 to 0.826 | -0.079 | -0.097 to -0.063 | +0.00 | +0.00 to +0.00 |

**Primary claim, Advisor against cheaper-first:** cost per solved task difference -0.000 (95% interval -0.000 to +0.000); interval wholly below zero: False. Solve-rate difference +0.00 pp (lower 95% bound +0.00); non-inferior (bound above -2 pp): True. Verdict under the protocol rule: **not supported**.

Share of tasks sent straight to the frontier model: Advisor 0.0%, repository-only 0.0%, rule F 55.0%, Scrouting head 3.0%. Chosen C values (counts): 0.01: 37, 0.1: 61, 1.0: 2, 10.0: 0.

## 2. Threshold sensitivity (Advisor)

| θ | Spend | Solved | Cost per solved | Difference vs C | 95% interval | Solve-rate diff (pp) | lower bound |
|---|---|---|---|---|---|---|---|
| 0.3 | $144.77 | 186.8 | $0.775 | +0.010 | +0.000 to +0.024 | -0.47 | -1.09 |
| 0.5 | $160.07 | 178.3 | $0.897 | +0.133 | +0.077 to +0.199 | -3.63 | -5.70 |
| 0.7 | $196.12 | 161.0 | $1.218 | +0.453 | +0.354 to +0.567 | -10.15 | -13.78 |

The primary θ is the training-fold cost ratio (about 0.14).

## 3. Prediction of Kimi success (out-of-fold, scores averaged over repetitions)

| Comparison | AUC (first) | AUC (second) | Difference | DeLong p |
|---|---|---|---|---|
| Advisor vs repository-only | 0.600 | 0.489 | +0.112 | 0.019 |
| Advisor vs Scrouting text head | 0.600 | 0.523 | +0.078 | 0.117 |

Brier score (lower is better): Advisor 0.2413, repository-only 0.2459, base rate 0.2464.
With 266 tasks only a substantial gain in AUC is detectable (protocol section 9); a non-significant difference is not evidence of no effect.

## 4. Leave-one-repository-out (secondary scheme)

| Policy | Spend | Solved | Cost per solved | 95% interval | Difference vs C | 95% interval | Solve-rate diff vs C (pp) | 95% interval |
|---|---|---|---|---|---|---|---|---|
| A Frontier for all | $201.25 | 158.0 | $1.274 | 1.110 to 1.471 | +0.509 | +0.403 to +0.631 | -11.28 | -15.04 to -7.52 |
| B Cheaper for all | $28.23 | 149.0 | $0.189 | 0.165 to 0.219 | -0.575 | -0.708 to -0.461 | -14.66 | -19.17 to -10.53 |
| C Cheaper first | $143.77 | 188.0 | $0.765 | 0.630 to 0.922 | +0.000 | +0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| D Advisor | $143.77 | 188.0 | $0.765 | 0.630 to 0.922 | +0.000 | +0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| E Repository-only | $143.77 | 188.0 | $0.765 | 0.630 to 0.922 | +0.000 | +0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| H Oracle (ceiling) | $128.97 | 188.0 | $0.686 | 0.566 to 0.826 | -0.079 | -0.097 to -0.063 | +0.00 | +0.00 to +0.00 |

AUC pooled over held-out repositories: Advisor 0.517. Repository-only is undefined for an unseen repository and falls back to the training base rate.

## 5. Label 100 (a separate check, never pooled; text form differs, so the shift is large)

| Policy | Spend | Solved | Cost per solved | 95% interval | Difference vs C | 95% interval | Solve-rate diff vs C (pp) | 95% interval |
|---|---|---|---|---|---|---|---|---|
| A Frontier for all | $78.34 | 48.0 | $1.632 | 1.199 to 2.266 | +0.619 | +0.307 to +1.034 | -12.00 | -19.00 to -6.00 |
| B Cheaper for all | $20.78 | 52.0 | $0.400 | 0.271 to 0.578 | -0.614 | -0.955 to -0.375 | -8.00 | -14.00 to -3.00 |
| C Cheaper first | $60.81 | 60.0 | $1.013 | 0.670 to 1.509 | +0.000 | +0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| D Advisor | $60.81 | 60.0 | $1.013 | 0.670 to 1.509 | +0.000 | +0.000 to +0.000 | +0.00 | +0.00 to +0.00 |
| H Oracle (ceiling) | $47.37 | 60.0 | $0.789 | 0.534 to 1.155 | -0.224 | -0.364 to -0.125 | +0.00 | +0.00 to +0.00 |

AUC against Kimi success on the 100 tasks: 0.567.
