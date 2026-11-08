# Hold flag: leave-one-repository-out and 100-task check (post hoc, exploratory)

266 tasks, features ['n_chars', 'n_code_blocks', 'has_trace_or_error', 'n_code_refs', 'n_headings', 'mentions_tests', 'n_list_items'], seed 42, 10000 bootstrap resamples. Same model, label and cutoffs as the 6 November exploratory run. My arithmetic.

## 1. Leave-one-repository-out (266 tasks)

Pooled AUC for predicting 'neither': **0.540** (95% interval 0.463 to 0.615). For reference, the 6 November run with mixed-repository cross-validation gave 0.632.

| Held-out repository | Tasks | Base rate of 'neither' | AUC within that repository |
|---|---|---|---|
| ansible | 96 | 38.5% | 0.535 |
| openlibrary | 91 | 27.5% | 0.646 |
| qutebrowser | 79 | 20.3% | 0.528 |

The within-repository AUCs have wide uncertainty because each repository has 79 to 96 tasks.

| Cutoff | Tasks flagged | Precision | 95% interval | Lift over base rate | Recall | 95% interval | Spend avoided | 95% interval | Solves lost | 95% interval |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 118 | 0.305 | 0.221 to 0.391 | 1.04 | 0.462 | 0.351 to 0.573 | $66.66 | 51.19 to 83.28 | 82 | 68 to 97 |
| 0.5 | 4 | 0.500 | 0.000 to 1.000 | 1.71 | 0.026 | 0.000 to 0.067 | $5.57 | 0.21 to 12.66 | 2 | 0 to 5 |
| 0.7 | 0 | nan | nan to nan | nan | 0.000 | 0.000 to 0.000 | $0.00 | 0.00 to 0.00 | 0 | 0 to 0 |

Base rate of 'neither model solves': 29.3% (78 of 266 tasks). A cutoff that flags no task shows precision nan.

## 2. Trained on the 266 tasks, applied to the 100-task set

AUC for predicting 'neither': **0.618** (95% interval 0.504 to 0.724), 100 tasks.

| Cutoff | Tasks flagged | Precision | 95% interval | Lift over base rate | Recall | 95% interval | Spend avoided | 95% interval | Solves lost | 95% interval |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 10 | 0.400 | 0.100 to 0.750 | 1.00 | 0.100 | 0.023 to 0.205 | $4.13 | 1.07 to 8.06 | 6 | 2 to 11 |
| 0.5 | 3 | 0.333 | 0.000 to 1.000 | 0.83 | 0.025 | 0.000 to 0.083 | $2.02 | 0.00 to 5.40 | 2 | 0 to 5 |
| 0.7 | 0 | nan | nan to nan | nan | 0.000 | 0.000 to 0.000 | $0.00 | 0.00 to 0.00 | 0 | 0 to 0 |

Base rate of 'neither model solves': 40.0% (40 of 100 tasks). A cutoff that flags no task shows precision nan.

## 3. Criteria written before the run (section 12 entry, 7 November 2026)

1. Pooled leave-one-repository-out AUC interval lies wholly above 0.5: **not met** (lower end 0.463).
2. At cutoff 0.5 the precision interval's lower end is above the base rate: **not met** (lower end 0.000 against 0.293).
3. The 100-task AUC is above 0.5: **met** (0.618).

All three met: no. Any other outcome is reported as it is. Spend avoided and solves lost are never combined. This is exploratory and changes nothing about the primary claim.
