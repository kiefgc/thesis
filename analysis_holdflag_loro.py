#!/usr/bin/env python3
"""Post hoc exploratory check of the hold flag (protocol v1.0, section 12 entry dated 7 November 2026).

Question: does the hold-flag signal ('neither model solves the task') carry to a repository the model has not seen,
and to the 100-task set? The 6 November exploratory run used repeated 5-fold cross-validation on the 266 tasks, which
mixes repositories between training and test. The primary Advisor lost its signal under leave-one-repository-out, so
the same check is needed here. Nothing about the model, features, label or cutoffs differs from the exploratory run.

  1. Leave-one-repository-out on the 266 tasks: train on two repositories, predict the third. One prediction per task.
  2. Train on all 266, predict the 100-task set (plain issue text, different benchmark family).

Reported: AUC (pooled and per held-out repository) with bootstrap 95% intervals; at cutoffs 0.3, 0.5, 0.7 the tasks
flagged, precision against the base rate, recall, spend avoided and solves lost (kept separate, never combined).
Spend avoided = the cheaper-first cost of flagged tasks. Solves lost = flagged tasks that cheaper-first would have solved.

DEFAULT MODE is a leakage check: the 'neither' labels are shuffled, so results are noise and nothing real is shown.
REAL MODE needs --real --confirm-reviewed --out <file> --exploratory-results results-exploratory.md and runs once.

Usage:
  python analysis_holdflag_loro.py <tasks_pro266.csv> <tasks_label100.csv>                 (shuffled-label check)
  python analysis_holdflag_loro.py <tasks_pro266.csv> <tasks_label100.csv> --real --confirm-reviewed \
        --exploratory-results results-exploratory.md --out results-holdflag-loro.md
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import advisor_common as AC

CUTS = (0.3, 0.5, 0.7)          # same cutoffs as protocol section 8 and the 6 November run


def boot_ci(v):
    v = np.sort(v[~np.isnan(v)])
    return (float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])) if len(v) else (float("nan"), float("nan"))


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def auc_boot(y, s, idx):
    out = np.full(len(idx), np.nan)
    for i, b in enumerate(idx):
        if 0 < y[b].sum() < len(b):
            out[i] = roc_auc_score(y[b], s[b])
    return boot_ci(out)


def cutoff_table(real, p, d, idx):
    N = len(real)
    Ccost = d["ck"] + (1 - d["ks"]) * d["co"]
    Csolved = ((d["ks"] + d["os"]) > 0).astype(float)
    base = real.mean()
    L = ["| Cutoff | Tasks flagged | Precision | 95% interval | Lift over base rate | Recall | 95% interval | Spend avoided | 95% interval | Solves lost | 95% interval |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for cut in CUTS:
        f = (p >= cut).astype(float)
        nf, tp = f.sum(), (f * real).sum()
        prec = tp / nf if nf > 0 else float("nan")
        fb, rb = f[idx], real[idx]
        nfb = fb.sum(1)
        pb = np.where(nfb > 0, (fb * rb).sum(1) / np.where(nfb > 0, nfb, 1), np.nan)
        rcb = (fb * rb).sum(1) / np.maximum(rb.sum(1), 1)
        ab, lb = (fb * Ccost[idx]).sum(1), (fb * Csolved[idx]).sum(1)
        pc, rc, ac, lc = boot_ci(pb), boot_ci(rcb), boot_ci(ab), boot_ci(lb)
        rows[cut] = dict(n=nf, prec=prec, prec_lo=pc[0])
        L.append(f"| {cut} | {nf:.0f} | {prec:.3f} | {pc[0]:.3f} to {pc[1]:.3f} | {prec / base if nf > 0 else float('nan'):.2f} | "
                 f"{tp / max(real.sum(), 1):.3f} | {rc[0]:.3f} to {rc[1]:.3f} | ${(f * Ccost).sum():.2f} | {ac[0]:.2f} to {ac[1]:.2f} | "
                 f"{(f * Csolved).sum():.0f} | {lc[0]:.0f} to {lc[1]:.0f} |")
    L.append(f"\nBase rate of 'neither model solves': {base:.1%} ({int(real.sum())} of {N} tasks). A cutoff that flags no task shows precision nan.")
    return L, rows, base


def main():
    flags, rest, it = {}, [], iter(sys.argv[1:])
    for a in it:
        if a in ("--out", "--exploratory-results"):
            flags[a] = next(it)
        elif a.startswith("--"):
            flags[a] = True
        else:
            rest.append(a)
    if len(rest) != 2:
        sys.exit("Expected 2 file arguments.\n" + __doc__)
    real_mode = "--real" in flags
    if real_mode:
        if "--confirm-reviewed" not in flags or "--out" not in flags or "--exploratory-results" not in flags:
            sys.exit("Real mode needs --confirm-reviewed, --out and --exploratory-results.")
        if not Path(flags["--exploratory-results"]).is_file():
            sys.exit("The exploratory results file does not exist yet.")
        if Path(flags["--out"]).exists():
            sys.exit(f"{flags['--out']} already exists. Run once; move it away deliberately to repeat.")
    d, d100 = AC.load_table(rest[0]), AC.load_table(rest[1])
    Xp, keep, _ = AC.feature_matrix(d["text"])
    Xl, _, _ = AC.feature_matrix(d100["text"], keep=keep)         # screening decided on the 266 tasks only
    real = ((d["ks"] == 0) & (d["os"] == 0)).astype(int)
    real100 = ((d100["ks"] == 0) & (d100["os"] == 0)).astype(int)
    if not real_mode:
        rng = np.random.default_rng(AC.SEED)
        real, real100 = rng.permutation(real), rng.permutation(real100)
    N, N100 = len(real), len(real100)
    idx = np.random.default_rng(AC.SEED).integers(0, N, size=(AC.B, N))
    idx100 = np.random.default_rng(AC.SEED).integers(0, N100, size=(AC.B, N100))

    # 1. leave-one-repository-out
    p = np.zeros(N)
    per_repo = []
    for r in sorted(set(d["repo"])):
        te = d["repo"] == r
        p[te] = AC.fit_predict(Xp[~te], real[~te], Xp[te], True)[0]
        per_repo.append((r, int(te.sum()), real[te].mean(), safe_auc(real[te], p[te])))
    pooled = safe_auc(real, p)
    lo, hi = auc_boot(real, p, idx)
    # 2. train on all 266, predict the 100-task set
    p100 = AC.fit_predict(Xp, real, Xl, True)[0]
    a100 = safe_auc(real100, p100)
    lo100, hi100 = auc_boot(real100, p100, idx100)

    L = ["# Hold flag: leave-one-repository-out and 100-task check (post hoc, exploratory)\n"]
    if not real_mode:
        L.append("**SHUFFLED-LABEL CHECK. The labels were shuffled; these numbers are noise and mean nothing.**\n")
    L.append(f"{N} tasks, features {keep}, seed {AC.SEED}, {AC.B} bootstrap resamples. Same model, label and cutoffs as the 6 November exploratory run. My arithmetic.\n")
    L.append("## 1. Leave-one-repository-out (266 tasks)\n")
    L.append(f"Pooled AUC for predicting 'neither': **{pooled:.3f}** (95% interval {lo:.3f} to {hi:.3f}). "
             "For reference, the 6 November run with mixed-repository cross-validation gave 0.632.\n")
    L.append("| Held-out repository | Tasks | Base rate of 'neither' | AUC within that repository |")
    L.append("|---|---|---|---|")
    for r, n, b, a in per_repo:
        L.append(f"| {r} | {n} | {b:.1%} | {a:.3f} |")
    L.append("\nThe within-repository AUCs have wide uncertainty because each repository has 79 to 96 tasks.\n")
    tab, rows, base = cutoff_table(real, p, d, idx)
    L += tab
    L.append("\n## 2. Trained on the 266 tasks, applied to the 100-task set\n")
    L.append(f"AUC for predicting 'neither': **{a100:.3f}** (95% interval {lo100:.3f} to {hi100:.3f}), {N100} tasks.\n")
    tab100, rows100, base100 = cutoff_table(real100, p100, d100, idx100)
    L += tab100
    L.append("\n## 3. Criteria written before the run (section 12 entry, 7 November 2026)\n")
    c1 = lo > 0.5
    c2 = rows[0.5]["n"] > 0 and rows[0.5]["prec_lo"] > base
    c3 = a100 > 0.5
    L.append(f"1. Pooled leave-one-repository-out AUC interval lies wholly above 0.5: **{'met' if c1 else 'not met'}** (lower end {lo:.3f}).")
    L.append(f"2. At cutoff 0.5 the precision interval's lower end is above the base rate: **{'met' if c2 else 'not met'}** "
             f"(lower end {rows[0.5]['prec_lo']:.3f} against {base:.3f}).")
    L.append(f"3. The 100-task AUC is above 0.5: **{'met' if c3 else 'not met'}** ({a100:.3f}).")
    L.append(f"\nAll three met: {'yes' if (c1 and c2 and c3) else 'no'}. Any other outcome is reported as it is. "
             "Spend avoided and solves lost are never combined. This is exploratory and changes nothing about the primary claim.")
    text = "\n".join(L) + "\n"
    if real_mode:
        Path(flags["--out"]).write_text(text, encoding="utf-8")
        print(text)
        print(f"written to {flags['--out']}")
    else:
        print(text)
        print("(shuffled-label check; nothing written)")


if __name__ == "__main__":
    main()
