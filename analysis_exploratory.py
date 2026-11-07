#!/usr/bin/env python3
"""Exploratory analyses under protocol v1.0 (no confirmatory claim is made from anything here).

  1. The hold flag (protocol section 8): a classifier of the same form as the primary Advisor predicts
     'neither model solves the task'. Precision and recall at cutoffs 0.3, 0.5, 0.7 against the base rate, with
     spend avoided and solves lost reported separately and never combined.
  2. Price-ratio sensitivity (post hoc, planned 2 November after the primary result): at what cost ratio between
     the cheaper and the frontier model would routing by the primary Advisor change the cost per solved task?
     The recorded Kimi costs are scaled so that the average cost ratio equals each value tried; Opus costs are kept.

Run only after results-primary.md exists. Usage:
    python analysis_exploratory.py <tasks_pro266.csv> <tasks_label100.csv> <superscout-clone> --primary-results results-primary.md --out results-exploratory.md --confirm-reviewed
Refuses to start if the output file already exists.
"""
import json
import sys
from pathlib import Path

import numpy as np

import advisor_common as AC

CUTS = (0.3, 0.5, 0.7)                                   # protocol section 8
RATIOS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)             # post hoc grid; the actual ratio is added


def _ci(v):
    v = np.sort(v[~np.isnan(v)])
    return (float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])) if len(v) else (float("nan"), float("nan"))


def hold_flag(d, Xp, label=None, reps=AC.REPS):
    N = len(d["ks"])
    real = ((d["ks"] == 0) & (d["os"] == 0)).astype(int)
    lab = real if label is None else label
    P, _, _ = AC.repeated_oof(Xp, lab, d, True, reps)
    PE, _, _ = AC.repeated_oof(AC.repo_matrix(d["repo"], sorted(set(d["repo"]))), lab, d, False, reps)
    Ccost = d["ck"] + (1 - d["ks"]) * d["co"]                 # cost of the cheaper-first policy per task
    Csolved = ((d["ks"] + d["os"]) > 0).astype(float)
    idx = np.random.default_rng(AC.SEED).integers(0, N, size=(AC.B, N))
    L = []
    w = L.append
    base = real.mean()
    w(f"Base rate of 'neither model solves': {base:.1%} ({real.sum()} of {N} tasks).\n")
    a1, a2, dd, p = AC.delong(real, P.mean(0), PE.mean(0))
    w(f"AUC for predicting 'neither': Advisor-form classifier {a1:.3f}, repository-only {a2:.3f}, difference {dd:+.3f}, DeLong p {p:.3f}.\n")
    w("| Cutoff | Tasks flagged | Precision | 95% interval | Lift over base rate | Recall | 95% interval | Spend avoided (cheaper-first cost of flagged tasks) | 95% interval | Solves lost | 95% interval |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for cut in CUTS:
        f = (P >= cut).mean(0)                                 # share of repetitions in which the task is flagged
        nf, tp = f.sum(), (f * real).sum()
        prec = tp / nf if nf > 0 else float("nan")
        fb, rb = f[idx], real[idx]
        nfb = fb.sum(1)
        precb = np.where(nfb > 0, (fb * rb).sum(1) / np.where(nfb > 0, nfb, 1), np.nan)
        recb = (fb * rb).sum(1) / np.maximum(rb.sum(1), 1)
        avoidb = (fb * Ccost[idx]).sum(1)
        lostb = (fb * Csolved[idx]).sum(1)
        pc, rc, ac_, lc = _ci(precb), _ci(recb), _ci(avoidb), _ci(lostb)
        w(f"| {cut} | {nf:.1f} | {prec:.3f} | {pc[0]:.3f} to {pc[1]:.3f} | {prec / base if nf > 0 else float('nan'):.2f} | {tp / real.sum():.3f} | {rc[0]:.3f} to {rc[1]:.3f} | "
          f"${(f * Ccost).sum():.2f} | {ac_[0]:.2f} to {ac_[1]:.2f} | {(f * Csolved).sum():.1f} | {lc[0]:.1f} to {lc[1]:.1f} |")
    w("\nSpend avoided and solves lost are reported separately and are not combined. Flagged tasks are assumed to be handled by the developer, whose time is not counted. "
      "A cutoff at which no task is flagged shows a precision of nan.")
    return L


def ratio_table(d, Xp, root, reps=AC.REPS):
    ks = d["ks"]
    P, _, _ = AC.repeated_oof(Xp, ks, d, True, reps)
    pv = json.load(open(Path(root) / "data/receipts/routing_scenarios.json", encoding="utf-8"))["probability_vectors"]
    pt = np.array([pv[t]["kimi-k2-5"]["p_text"] for t in d["ids"]])
    actual = d["ck"].sum() / d["co"].sum()
    L = []
    w = L.append
    w(f"Actual average cost ratio (cheaper ÷ frontier) on these tasks: {actual:.3f}. For each ratio r the Kimi costs are scaled by r ÷ {actual:.3f} and the routing threshold is r "
      "(send a task straight to the frontier model if the predicted chance of cheaper-model success is below r).\n")
    w("| Ratio r | Advisor: share routed to frontier | Advisor: cost per solved vs cheaper-first | 95% interval | Solve-rate diff (pp) | Scrouting head: share routed | Head: cost per solved vs cheaper-first | 95% interval | Oracle vs cheaper-first |")
    w("|---|---|---|---|---|---|---|---|---|")
    first_win = None
    for r in sorted(set(RATIOS) | {round(actual, 3)}):
        d2 = dict(d)
        d2["ck"] = d["ck"] * (r / actual)
        shD, shG = (P < r).mean(0), (pt < r).mean(0)
        res = AC.bootstrap(AC.policies(d2, {"D": shD, "G": shG}))
        D, G, H = res["D"], res["G"], res["H"]
        if first_win is None and D["d_ci"][1] < 0:
            first_win = r
        tag = " (actual)" if abs(r - actual) < 5e-4 else ""
        w(f"| {r}{tag} | {shD.mean():.1%} | {D['d']:+.3f} | {D['d_ci'][0]:+.3f} to {D['d_ci'][1]:+.3f} | {D['dsolve_pp']:+.2f} | {shG.mean():.1%} | "
          f"{G['d']:+.3f} | {G['d_ci'][0]:+.3f} to {G['d_ci'][1]:+.3f} | {H['d']:+.3f} |")
    w("\nLowest ratio tried at which the Advisor's interval lies wholly below zero (a saving over cheaper-first): "
      + (f"**{first_win}**." if first_win is not None else "**none of the ratios tried.**"))
    w("This table is post hoc and exploratory. It changes the prices, not the predictions, and it is not a claim about any real price.")
    return L


if __name__ == "__main__":
    flags, rest, it = {}, [], iter(sys.argv[1:])
    for a in it:
        if a in ("--primary-results", "--out"):
            flags[a] = next(it)
        elif a.startswith("--"):
            flags[a] = True
        else:
            rest.append(a)
    if len(rest) != 3 or "--primary-results" not in flags or "--out" not in flags or "--confirm-reviewed" not in flags:
        sys.exit("Expected 3 arguments and --primary-results, --out, --confirm-reviewed.\n" + __doc__)
    if not Path(flags["--primary-results"]).is_file():
        sys.exit("The primary results file does not exist yet.")
    out_path = Path(flags["--out"])
    if out_path.exists():
        sys.exit(f"{out_path} already exists. Run once; move it away deliberately to repeat.")
    tasks, label, root = rest
    d = AC.load_table(tasks)
    Xp, keep, _ = AC.feature_matrix(d["text"])
    lines = ["# Exploratory analyses (protocol v1.0; no confirmatory claim)\n",
             f"{len(d['ks'])} tasks, {AC.REPS} repetitions of stratified 5-fold, seed {AC.SEED}, {AC.B} bootstrap resamples. Handcrafted features. My arithmetic.\n",
             "## 1. Hold flag: tasks neither model solves (protocol section 8)\n"]
    lines += hold_flag(d, Xp)
    lines += ["\n## 2. Price-ratio sensitivity (post hoc)\n"]
    lines += ratio_table(d, Xp, root)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out_path}")
