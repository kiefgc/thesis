#!/usr/bin/env python3
"""Build the task table for the Delegation Advisor study from the Scrouting release.

Usage:  python3 -I build_task_table.py <path-to-superscout-clone> <output-dir>

Reads only files in the Scrouting repository (standard library only) and writes:
  tasks_pro266.csv    266 tasks: pre-run columns, outcomes and costs, post-run columns
  tasks_label100.csv  100 tasks: pre-run columns, solo outcomes and costs
  table_check.txt     counts and totals to compare with the expected values

Column naming rule (leakage guard):
  pre_*   known before any model runs (repository, issue text). Allowed as predictor inputs.
  out_*   outcomes and costs. Labels and cost accounting only, never inputs.
  post_*  known only after the run (steps, auto-submit). Never inputs.
Gold-patch and test-derived fields in the 100-task manifest (derived_difficulty, n_files_patch,
n_lines_patch, gold_patch_bytes, FAIL_TO_PASS, ...) are deliberately NOT copied.
"""
import csv, json, sys, re
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
data = root / "data"

# ---------- 266-task set ----------
solo = json.load(open(data / "receipts/outcomes_per_task.json", encoding="utf-8"))["solo"]

text = {}
with open(data / "trajectories/pro266/episodes.jsonl", encoding="utf-8") as f:
    for line in f:
        e = json.loads(line)
        c = e["messages"][1]["content"]
        c = c if isinstance(c, str) else json.dumps(c)
        m = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", c, re.S)
        text[e["instance_id"]] = (e["repo_family"], m.group(1) if m else "")

models = ["kimi", "opus", "gpt"]
cols = ["task_id", "pre_repo", "pre_issue_text"]
for k in models:
    cols += [f"out_{k}_solved", f"out_{k}_cost_usd"]
for k in models:
    cols += [f"post_{k}_steps"]
n_missing_text = 0
with open(out / "tasks_pro266.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for tid, rec in solo.items():
        fam, txt = text.get(tid, (rec.get("family", ""), ""))
        n_missing_text += (txt == "")
        row = [tid, fam, txt]
        for k in models:
            row += [int(bool(rec[k]["resolved"])), rec[k]["cost_usd_pinned"]]
        for k in models:
            row += [rec[k]["steps"]]
        w.writerow(row)

# ---------- 100-task set (solo arm only) ----------
inputs = {}
with open(data / "labelrun/task_inputs.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        inputs[r["instance_id"]] = r["problem_statement"]
repo = {}
with open(data / "labelrun/labelrun_manifest.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        repo[r["instance_id"]] = r["repo"]
fx = {"Kimi K2.5": "kimi", "Claude Opus 4.6": "opus", "GPT-5.2": "gpt", "Gemini 3 Flash": "gemini"}
res = {}
with open(data / "labelrun/outcomes.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["arm"] == "solo":
            res.setdefault(r["task_id"], {})[fx[r["fixer"]]] = (int(bool(r["solved"])), r["cost_usd_pinned"])
m4 = ["kimi", "opus", "gpt", "gemini"]
cols2 = ["task_id", "pre_repo", "pre_issue_text"]
for k in m4:
    cols2 += [f"out_{k}_solved", f"out_{k}_cost_usd"]
with open(out / "tasks_label100.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(cols2)
    for tid in inputs:
        row = [tid, repo.get(tid, ""), inputs[tid]]
        for k in m4:
            row += list(res[tid][k])
        w.writerow(row)

# ---------- checks ----------
def stats(path, a, b, n_expected):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ks = sum(int(r[f"out_{a}_solved"]) for r in rows)
    os_ = sum(int(r[f"out_{b}_solved"]) for r in rows)
    union = sum(1 for r in rows if int(r[f"out_{a}_solved"]) or int(r[f"out_{b}_solved"]))
    neither = len(rows) - union
    cost_a = sum(float(r[f"out_{a}_cost_usd"]) for r in rows)
    cost_b = sum(float(r[f"out_{b}_cost_usd"]) for r in rows)
    esc = sum(float(r[f"out_{a}_cost_usd"]) + (0 if int(r[f"out_{a}_solved"]) else float(r[f"out_{b}_cost_usd"])) for r in rows)
    empty = sum(1 for r in rows if not r["pre_issue_text"].strip())
    return (f"{path.name}: rows={len(rows)} (expected {n_expected}) empty_issue_text={empty}\n"
            f"  {a} solved={ks}  {b} solved={os_}  either={union}  neither={neither}\n"
            f"  spend {a}-all=${cost_a:.2f}  {b}-all=${cost_b:.2f}  {a}-first-then-{b}=${esc:.2f}\n")

report = stats(out / "tasks_pro266.csv", "kimi", "opus", 266)
report += "  expected for pro266: kimi 149, opus 158, either 188, neither 78, spend 28.23 / 201.25 / 143.77\n"
report += stats(out / "tasks_label100.csv", "kimi", "opus", 100)
report += f"pro266 tasks with no extracted issue text: {n_missing_text}\n"
(out / "table_check.txt").write_text(report, encoding="utf-8")
print(report)
