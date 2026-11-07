#!/usr/bin/env python3
"""Secondary analysis under protocol v1.0: the embedding variant of the Advisor.

Run only AFTER the primary results file exists (the script checks this).
Usage:
    python analysis_secondary.py <tasks_pro266.csv> <tasks_label100.csv> <superscout-clone> <embeddings-dir> --primary-results results-primary.md --out results-secondary.md --confirm-reviewed

Refuses to start if the output file already exists.
"""
import sys
from pathlib import Path

import numpy as np

import advisor_common as AC

flags = {}
rest = []
it = iter(sys.argv[1:])
for a in it:
    if a in ("--primary-results", "--out"):
        flags[a] = next(it)
    elif a.startswith("--"):
        flags[a] = True
    else:
        rest.append(a)
if len(rest) != 4 or "--primary-results" not in flags or "--out" not in flags or "--confirm-reviewed" not in flags:
    sys.exit("Expected 4 arguments and --primary-results, --out, --confirm-reviewed.\n" + __doc__)
tasks, label, root, emb = rest
if not Path(flags["--primary-results"]).is_file():
    sys.exit("The primary results file does not exist yet. The secondary analysis runs only after the primary one.")
out_path = Path(flags["--out"])
if out_path.exists():
    sys.exit(f"{out_path} already exists. Run once; move it away deliberately to repeat.")

d, d100 = AC.load_table(tasks), AC.load_table(label)
emb = Path(emb)
for name, dd in (("pro266", d), ("label100", d100)):
    ids = (emb / f"task_ids_{name}.txt").read_text(encoding="utf-8").split()
    if ids != dd["ids"]:
        sys.exit(f"{name}: embedding task order does not match the table.")
Xp = np.load(emb / "embeddings_pro266.npy").astype(float)
Xl = np.load(emb / "embeddings_label100.npy").astype(float)
if Xp.shape != (len(d["ids"]), 1024) or Xl.shape != (len(d100["ids"]), 1024):
    sys.exit("Unexpected embedding shapes.")
lines = AC.run_analysis(d, d100, Xp, Xl, False, "Secondary analysis: embedding Advisor (protocol v1.0)", root)
out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nwritten to {out_path}")
