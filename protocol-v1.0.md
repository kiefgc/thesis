# Analysis Protocol v1.0

29 October 2026 · K · **Status: ready to freeze.** All choices in section 11 are settled. Commit this file to git on 30 October 2026, before any predictor is fitted. After that commit, changes are allowed only in section 12. The commit hash and date are the proof of order.

This protocol fixes how the Delegation Advisor will be built and judged on the recorded runs. It makes no claim about results. Evidence tags: **[READ]** opened and read; **[UNVERIFIED]** not checked. "(my arithmetic)" marks figures I computed from the released files.

## Plain-English summary

Before we look at whether the Advisor works, we write down exactly what we will do and how we will judge it, so that we cannot adjust the rules after seeing the results. If the Advisor turns out to save little or nothing, we report that.

## 1. Question and claims

**Question.** Can information known before a task is run predict whether a cheaper model will solve it, and does acting on that prediction lower the cost per solved task, with failures and wrong patches counted?

**Primary claim (confirmatory).** Under the policy in section 4, the Delegation Advisor changes cost per solved task compared with "cheaper first, escalate on failure". Either direction, or no difference, is reported.

**Secondary claims (confirmatory but lower priority).** The Advisor predicts the cheaper model's success better than the repository-only rule, and it compares to Scrouting's text-only head and the existing-practice rule.

**Exploratory (no confirmatory claim).** The hold flag for tasks neither model solves (section 8), and any other analysis not listed here.

## 2. Data

| Set | File | Tasks | Use |
|---|---|---|---|
| Pro 266 | `tasks_pro266.csv` | 266 (ansible 96, openlibrary 91, qutebrowser 79) | Development and evaluation, by cross-validation |
| Label 100 | `tasks_label100.csv` | 100 from 80 repositories | A separate check, never pooled with Pro 266 |

**Column rule.** Only `pre_*` columns may be predictor inputs. `out_*` (outcomes, costs) are labels and cost accounting only. `post_*` (steps) are never inputs. Gold patches and test-derived fields are not in the tables.

**Caveats fixed in advance.** Label 100 appears to come from a different benchmark family than Pro 266 (its image names point to SWE-rebench), and its cap and grader are not confirmed to match [UNVERIFIED]. Whether the Scrouting router used Label 100 to fit or calibrate (its threshold was calibrated on a 99-task label run [READ]) is unconfirmed; the question is posted in the issue to the authors. Until answered, Label 100 is used only for the predictor we train, not for judging Scrouting's head. **The text differs in form:** Pro 266 issue text is a structured description (title, description, requirements, interface), while Label 100 is plain issue text. The medians differ accordingly (3,160 against 1,928 characters, and 10 against 2 list items, from the feature values), so a predictor trained on Pro 266 faces a distribution shift on Label 100 and that check is weak.

## 3. Models and labels

- **Cheaper model:** Kimi K2.5. **Frontier model:** Claude Opus 4.6. Costs are `cost_usd_pinned`.
- **Label for prediction:** `y = out_kimi_solved`, defined as "solved within the benchmark's capped tier" (50 model calls, a $2.00 per-attempt limit, working diff auto-submitted at the cap **[READ]**). It is called that everywhere.
- **Step cap.** The label is not redefined. A re-run of capped Kimi failures with a higher cap is a later sensitivity check, not part of this protocol.
- **Label for the hold flag:** `neither = (Kimi unsolved) and (Opus unsolved)`.

## 4. Policy and cost accounting

**Policy.** One attempt on the chosen model. If the cheaper model fails, escalate once to the frontier model and count both attempts. No repeated retries.

| Policy | Rule | Cost of a task |
|---|---|---|
| A. Frontier for all | Opus on every task | Opus cost |
| B. Cheaper for all | Kimi only, no escalation | Kimi cost |
| C. Cheaper first | Kimi, then Opus if Kimi fails | Kimi cost, plus Opus cost if Kimi failed |
| D. Advisor | Opus only if predicted success is below the threshold θ; otherwise as C | As A or C |
| E. Repository-only | As D, with a repository-only predictor | As A or C |
| F. Existing-practice rule | As D, with the rule in section 7 | As A or C |
| G. Scrouting text head | As D, with its released text-only probability for Kimi | As A or C |
| H. Oracle | Cheapest model that actually solves; ceiling only | Per outcome |

**Threshold θ for the primary analysis.** θ = (mean Kimi cost) ÷ (mean Opus cost), computed on the training folds only. On Pro 266 this is about 0.14 (my arithmetic: $28.23 ÷ $201.25), so a task goes straight to Opus only if the predicted chance of Kimi succeeding is below about 14%. **Expect the Advisor to behave almost like policy C on dollars.** The most a perfect identifier could save over C is about $14.80 of $143.77 (my arithmetic). Sensitivity: θ in {0.14, 0.30, 0.50, 0.70}, all reported, only the cost-ratio value is primary.

**Rework adjustment.** A solved task is a false solve with rate f for each model. A false solve is assumed to be discovered and re-run once on the frontier model at its recorded cost for that task, and the task counts as solved after the rework (an optimistic simplification, stated as a limitation). Until measured rates exist, f = 0 is reported alongside f in {5%, 10%, 20%}, labelled "assumed". The paper's headline uses measured rates from the audit, and says so; without them the rework claim is dropped.

## 5. Features (pre-run only)

Primary Advisor: a regularized logistic regression on handcrafted features. They are defined exactly by the code in `features.py` (committed with this protocol), which reads only the issue text. Count features enter the model as ln(1 + count).

| # | Feature | Exact definition |
|---|---|---|
| F1 | Text length | Number of characters |
| F2 | Code blocks | Number of fenced code blocks (pairs of three backticks) |
| F3 | Trace or error | 1 if the text contains a Python traceback line, a `File "...", line N` line, or an exception class name ending in Error or Exception; else 0 |
| F4 | Code references | Number of distinct inline-code spans (up to 80 characters) plus distinct file-like names (.py, .yml, .json and similar) |
| F5 | Headings | Number of lines starting with one to six `#` |
| F6 | URLs | Number of `http://` or `https://` occurrences |
| F7 | Tests mentioned | 1 if the text contains test, tests, pytest, unittest, assert or assertion; else 0 |
| F8 | List items | Number of numbered or bulleted lines |

**Outcome-blind screening rule, fixed before any model is fitted.** A feature that is nonzero in fewer than 10% of the 266 tasks is dropped, because with 117 cheaper-model failures it cannot be estimated. I checked this on the feature values only, without looking at any outcome: F6 (URLs) is nonzero in 14 of 266 tasks (5%) and is dropped. F2 (code blocks) is nonzero in 30 of 266 (11%) and stays. That leaves seven features, within the rule of thumb of about ten.

Secondary Advisor: logistic regression on a text embedding of the issue text (Qwen3-Embedding-0.6B, as in Scrouting's text head, with L2 normalization), with a word-frequency (TF-IDF) fallback if the embedding model cannot be downloaded. The fallback is declared before use.

Repository-only baseline: the repository as a categorical variable. Handcrafted-feature extraction code is committed with this protocol.

## 6. Validation

- **Primary scheme:** repeated stratified 5-fold cross-validation, 20 repetitions, seed 42, on Pro 266. Folds are stratified by the label. Hyperparameters (regularization strength C in {0.01, 0.1, 1, 10}) are chosen by inner 5-fold cross-validation inside each training fold.
- **Secondary scheme:** leave-one-repository-out (three folds), which shows whether the Advisor carries to an unseen repository. The repository-only rule is undefined there and falls back to the training base rate.
- **Label 100:** the trained predictor is applied as a check and reported separately.

## 7. Existing-practice rule (reconstruction)

A keyword-and-length heuristic of the kind routing tools describe, built from their public descriptions, which I read only in part **[PARTIAL]**. It is a reconstruction, not the tools' code: route straight to the frontier model if the issue text is longer than the training-fold median length, or if it contains any of: *refactor, redesign, architecture, migrate, migration, security, concurrency, race condition, rewrite*. Otherwise use policy C. The word list and the rule are fixed here.

## 8. Hold flag (exploratory)

Run only after the primary analysis. A classifier of the same form predicts `neither`. Report precision and recall at three fixed probability cutoffs (0.3, 0.5, 0.7), against the base rate (78 of 266 tasks, 29%, my arithmetic). Report **spend avoided and solves lost separately**, without combining them. No confirmatory claim is made.

## 9. Metrics and decision rules

- **Primary metric:** cost per solved task (total spend ÷ solved tasks) for the Advisor minus policy C, as a paired bootstrap over tasks (10,000 resamples, seed 42), 95% interval. The primary claim is "supported" only if the interval excludes zero in the Advisor's favour at f = 0. It is "not supported" otherwise, and the interval is reported either way.
- **Solve rate:** reported next to cost. Because the Advisor can send a task straight to Opus, which may lose Kimi-only solves, the Advisor's solves must be within 2 percentage points of policy C (about 5 of 266 tasks, my arithmetic) to count as non-inferior. The margin was set earlier in the thesis discussions.
- **Prediction:** AUC of each predictor against the repository-only rule, with the DeLong test, and the Brier score. With 266 tasks, only a substantial gain in AUC is detectable (about 44% power for a gain of 0.05, from an optimistic simulation in the limitations paper), and the smallest detectable gain is stated in the results.
- **Inconclusive results** are reported as inconclusive, not as support.

## 10. Already looked at before this protocol

Solved counts and costs per model on both sets; the cost of policies A, B, C and the oracle on Pro 266 (my arithmetic); the repository-only in-sample AUC (0.575 for Kimi, 0.616 for Opus); cap statistics; a simulation of power for AUC differences; and the router's released probabilities as files (not analysed). The distribution of each handcrafted feature on both sets (medians, ranges and the share of tasks where it is nonzero), used only for the screening rule in section 5. No text-based predictor has been fitted and no feature has been compared with an outcome.

## 11. Choices made (29 October 2026)

| # | Choice | Decision | Reason |
|---|---|---|---|
| 1 | Primary Advisor | Handcrafted features (section 5). The embedding version is a declared secondary, run only if the model downloads, with the word-frequency fallback | We control and can explain the rules; it does not depend on a download |
| 2 | Threshold θ | Mean Kimi cost ÷ mean Opus cost, from the training folds | Kept as is, to stay consistent with the sources used so far; sensitivity values are also reported |
| 3 | Primary validation | Repeated stratified 5-fold, 20 repetitions; leave-one-repository-out as secondary | Default |
| 4 | Non-inferiority margin | 2 percentage points of solve rate | Set earlier in the thesis discussions |
| 5 | False-solve convention | One extra frontier attempt, counted as solved after rework | Default |
| 6 | Hold-flag cutoffs | 0.3, 0.5, 0.7 | Default |

## 12. Changes after freezing

Any change after the commit is written below with its date and reason, and the analysis it affects is marked as post hoc.

| Date | Change | Reason | Analyses affected |
|---|---|---|---|
| 1 November 2026 | Procedural record, not an analysis change. The secondary variant's embeddings were computed once by `embed_all.py` on CPU in float32, with Qwen3-Embedding-0.6B at revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, maximum length 4096, L2-normalised, plain issue text, no prefix, batch size 2. Versions: Python 3.13.7, torch 2.15.0+cpu, transformers 5.22.1, sentence-transformers 6.2.0, numpy 2.5.4. Input hashes matched the frozen tables. Output hashes: Pro 266 `20CE8147FA596276B7A6501A042234C9536BC989ED14908A1051FC5D9F91C89E`, Label 100 `CA655F47D17BAC6C3D8991BC06CB8D334842D2789915FD34342DE1059C0843EF`. Run time 883 s and 385 s. An earlier smoke test on 10 and 3 texts used bfloat16 and then float32; its outputs were not kept or used. | The protocol named the embedding model and settings but not the precision, device or library versions, which affect the numbers slightly. Recording them makes the run repeatable. | Secondary embedding variant only. No feature, label, threshold, metric or baseline changed. |

## Sources

**Opened [READ]:** the Scrouting repository (commit af038e1): `data/receipts/outcomes_per_task.json`, `data/labelrun/*`, `code/router/router_spec.json`, and the paper sources `paper/sections/06_experimental_setup.tex` and `07_results.tex`. **Partly read [PARTIAL]:** the pages describing existing routing tools, used only for the general kind of heuristic in section 7.
