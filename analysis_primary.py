#!/usr/bin/env python3
"""Primary analysis under protocol v1.0: the handcrafted-feature Delegation Advisor.

Usage (safe default, shuffled-label leakage check; touches no real signal):
    python analysis_primary.py <tasks_pro266.csv> <tasks_label100.csv> <superscout-clone>

The real run (once, after you have reviewed the code and logged the two clarifications in protocol section 12):
    python analysis_primary.py <tasks_pro266.csv> <tasks_label100.csv> <superscout-clone> --real --confirm-reviewed --out results-primary.md

The real run refuses to start if the output file already exists. Nothing else is written.
"""
import sys
from pathlib import Path

import numpy as np

import advisor_common as AC

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
if "--out" in sys.argv:
    out_path = Path(sys.argv[sys.argv.index("--out") + 1])
    args = [a for a in args if a != str(out_path)]
else:
    out_path = None
if len(args) != 3:
    sys.exit("Expected 3 arguments.\n" + __doc__)
tasks, label, root = args
real = "--real" in flags

d, d100 = AC.load_table(tasks), AC.load_table(label)
Xp, keep, share = AC.feature_matrix(d["text"])
Xl, _, _ = AC.feature_matrix(d100["text"], keep)
print("features kept:", keep)
print("share of tasks with the feature nonzero (outcome-blind):", {k: round(v, 3) for k, v in share.items()})

if not real:
    print("\nSHUFFLED-LABEL CHECK: the labels are permuted for BOTH training and judging, which destroys any real link")
    print("between text and outcome. A leak-free pipeline gives AUC near 0.5 (standard error about 0.036; out-of-fold")
    print("AUC on pure noise tends to sit slightly below 0.5). A mean far above 0.5 means label information is leaking.")
    aucs = []
    for seed in range(5):
        y_perm = np.random.default_rng(1000 + seed).permutation(d["ks"])
        P, TH, _ = AC.repeated_oof(Xp, y_perm, d, True, reps=3)
        aucs.append(AC.auc(y_perm, P.mean(0)))
    print("AUC over 5 permutations:", [round(a, 3) for a in aucs], "mean", round(float(np.mean(aucs)), 3))
    print("PASS" if abs(np.mean(aucs) - 0.5) < 0.06 else "CHECK: the mean is far from 0.5; look for a leak before any real run")
    sys.exit(0)

if "--confirm-reviewed" not in flags or out_path is None:
    sys.exit("The real run needs --confirm-reviewed and --out <file>. Read the code first.")
if out_path.exists():
    sys.exit(f"{out_path} already exists. The real run happens once; move it away deliberately to repeat.")
lines = AC.run_analysis(d, d100, Xp, Xl, True, "Primary analysis: handcrafted Advisor (protocol v1.0)", root)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nwritten to {out_path}")
