#!/usr/bin/env python3
"""Baseline analysis under protocol v1.0 (no fitted predictor).

Usage:  python3 -I analysis_baselines.py <tasks_pro266.csv> <superscout-clone> <output-md>

Policies (protocol section 4): A frontier-all, B cheaper-all, C cheaper-first, H oracle,
F existing-practice rule, G Scrouting text-only head (released probabilities).
Cost per solved task = total spend / solved tasks. Intervals: paired bootstrap over tasks,
10,000 resamples, seed 42, 95% percentile interval, differences taken against policy C.
Standard library only. Reads cost and outcome columns, which is allowed for baselines.
"""
import csv, json, random, re, sys, statistics as st

tasks_csv, repo, out_md = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(tasks_csv, encoding="utf-8")))
N = len(rows)
ck = [float(r["out_kimi_cost_usd"]) for r in rows]
co = [float(r["out_opus_cost_usd"]) for r in rows]
ks = [int(r["out_kimi_solved"]) for r in rows]
os_ = [int(r["out_opus_solved"]) for r in rows]
text = [r["pre_issue_text"] for r in rows]
tid = [r["task_id"] for r in rows]
SEED, B = 42, 10000

# ---- per-task cost and solved vectors for each policy ----
def pol_A(): return co[:], os_[:], [os_[i] for i in range(N)]  # cost, solved, final-solver-is-frontier flag unused
def vec(cost, solved): return cost, solved

A = (co[:], os_[:])
Bp = (ck[:], ks[:])
C = ([ck[i] + (0 if ks[i] else co[i]) for i in range(N)], [1 if (ks[i] or os_[i]) else 0 for i in range(N)])
H = ([ck[i] if ks[i] else co[i] for i in range(N)], [1 if (ks[i] or os_[i]) else 0 for i in range(N)])


def route_policy(to_frontier):
    """to_frontier[i] in [0,1]: share of the time the rule sends task i straight to the frontier model."""
    cost = [to_frontier[i] * co[i] + (1 - to_frontier[i]) * C[0][i] for i in range(N)]
    solved = [to_frontier[i] * os_[i] + (1 - to_frontier[i]) * C[1][i] for i in range(N)]
    return cost, solved


def strat_folds(rep, k=5):
    rnd = random.Random(SEED + rep)
    folds = [[] for _ in range(k)]
    for lab in (0, 1):
        idx = [i for i in range(N) if ks[i] == lab]
        rnd.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return folds


# ---- F: existing-practice rule (protocol section 7) ----
KW = re.compile(r"\b(refactor|redesign|architecture|migrate|migration|security|concurrency|race condition|rewrite)\b", re.I)
lens = [len(t) for t in text]
has_kw = [bool(KW.search(t)) for t in text]
REPS = 20
share_F = [0.0] * N
for rep in range(REPS):
    for f in strat_folds(rep):
        test = set(f)
        med = st.median(lens[i] for i in range(N) if i not in test)
        for i in f:
            share_F[i] += (1.0 if (lens[i] > med or has_kw[i]) else 0.0) / REPS
F = route_policy(share_F)

# ---- G: Scrouting text-only head (released probabilities for Kimi) ----
pv = json.load(open(f"{repo}/data/receipts/routing_scenarios.json", encoding="utf-8"))["probability_vectors"]
pt = [pv[t]["kimi-k2-5"]["p_text"] for t in tid]
def G_policy(theta_fn):
    share = [0.0] * N
    for rep in range(REPS):
        for f in strat_folds(rep):
            test = set(f)
            train = [i for i in range(N) if i not in test]
            th = theta_fn(train)
            for i in f:
                share[i] += (1.0 if pt[i] < th else 0.0) / REPS
    return route_policy(share)

ratio = lambda train: sum(ck[i] for i in train) / sum(co[i] for i in train)
G_primary = G_policy(ratio)
G_grid = {th: G_policy(lambda tr, th=th: th) for th in (0.14, 0.30, 0.425, 0.50, 0.70)}

# ---- AUC of released text head against Kimi success ----
def auc(score, y):
    pos = [s for s, v in zip(score, y) if v]
    neg = [s for s, v in zip(score, y) if not v]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))
auc_G = auc(pt, ks)

# ---- bootstrap ----
def with_rework(pol, f):
    cost, solved = pol
    return [cost[i] + f * co[i] * solved[i] for i in range(N)], solved

def boot(pols, f=0.0):
    """pols: dict name -> (cost, solved). Returns point estimates and CIs of cost per solved and difference vs C."""
    P = {k: with_rework(v, f) for k, v in pols.items()}
    rnd = random.Random(SEED)
    res = {k: [] for k in P}
    diff = {k: [] for k in P}
    for _ in range(B):
        idx = rnd.choices(range(N), k=N)
        cps = {}
        for k, (cost, solved) in P.items():
            s = sum(solved[i] for i in idx)
            cps[k] = sum(cost[i] for i in idx) / s if s else float("nan")
        for k in P:
            res[k].append(cps[k])
            diff[k].append(cps[k] - cps["C"])
    def ci(v):
        v = sorted(x for x in v if x == x)
        return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
    out = {}
    for k, (cost, solved) in P.items():
        tot, sol = sum(cost), sum(solved)
        out[k] = dict(spend=tot, solved=sol, cps=tot / sol, cps_ci=ci(res[k]), d=tot / sol - sum(P["C"][0]) / sum(P["C"][1]), d_ci=ci(diff[k]))
    return out

main = {"A Frontier for all": A, "B Cheaper for all": Bp, "C Cheaper first": C, "H Oracle": H,
        "F Existing-practice rule": F, "G Scrouting text head": G_primary}
# keep a short key for diff vs C
keyed = {k.split()[0]: v for k, v in main.items()}
names = {k.split()[0]: k for k in main}
r0 = boot(keyed, 0.0)

lines = []
w = lines.append
w("# Baseline results (protocol v1.0, no fitted predictor)\n")
w(f"Generated by `analysis_baselines.py` on {N} tasks, seed {SEED}, {B} bootstrap resamples. Costs are `cost_usd_pinned`. Solved means solved within the capped tier. No audited false-solve correction (f = 0). My arithmetic from the released files.\n")
w("## 1. Policies at f = 0\n")
w("| Policy | Spend | Solved | Cost per solved | 95% interval | Difference vs C | 95% interval |")
w("|---|---|---|---|---|---|---|")
for k in keyed:
    o = r0[k]
    w(f"| {names[k]} | ${o['spend']:.2f} | {o['solved']:.1f} | ${o['cps']:.3f} | {o['cps_ci'][0]:.3f} to {o['cps_ci'][1]:.3f} | {o['d']:+.3f} | {o['d_ci'][0]:+.3f} to {o['d_ci'][1]:+.3f} |")
w("\nF and G use 20 repetitions of stratified 5-fold cross-validation; their spend and solved counts are averages over repetitions. G uses the cost-ratio threshold from the training folds (protocol section 4).\n")
w("## 2. Scrouting text head: threshold sensitivity (f = 0)\n")
w("| Threshold θ | Spend | Solved | Cost per solved | Difference vs C | 95% interval |")
w("|---|---|---|---|---|---|")
for th, pol in G_grid.items():
    o = boot({"C": C, "G": pol}, 0.0)["G"]
    w(f"| {th} | ${o['spend']:.2f} | {o['solved']:.1f} | ${o['cps']:.3f} | {o['d']:+.3f} | {o['d_ci'][0]:+.3f} to {o['d_ci'][1]:+.3f} |")
w(f"\nAUC of the released text-only head against Kimi success on {N} tasks: **{auc_G:.3f}** (0.5 is chance). Share of tasks the head sends straight to the frontier model at the primary threshold: {sum(share_F) and 0:.0f}", )
lines[-1] = lines[-1].split("Share of tasks")[0].rstrip()
w("")
w("## 3. Assumed false-solve rates (labelled assumed, not measured)\n")
w("A false solve is re-run once on the frontier model at its recorded cost and counts as solved after rework (protocol section 4). The same assumed rate f applies to both models.\n")
w("| f | A Frontier | B Cheaper | C Cheaper first | H Oracle | C minus A |")
w("|---|---|---|---|---|---|")
for f in (0.0, 0.05, 0.10, 0.20):
    o = boot({"A": A, "B": Bp, "C": C, "H": H}, f)
    w(f"| {f:.0%} | ${o['A']['cps']:.3f} | ${o['B']['cps']:.3f} | ${o['C']['cps']:.3f} | ${o['H']['cps']:.3f} | {o['C']['cps'] - o['A']['cps']:+.3f} |")
w("\nThese are cost per solved task in dollars. The rows are an illustration of sensitivity. The paper's headline rework figure must come from the audit.\n")
w("## 4. What the rule F sends where\n")
w(f"Under rule F the share of tasks sent straight to the frontier model, averaged over repetitions: {sum(share_F)/N:.1%}. Tasks matching a keyword: {sum(has_kw)} of {N}.")
open(out_md, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
