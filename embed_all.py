#!/usr/bin/env python3
"""Embed the issue text of every task once, for the secondary Advisor variant (protocol section 5).

Usage:  python embed_all.py <tasks_pro266.csv> <tasks_label100.csv> <output-dir>

Settings recorded in Scrouting's router_spec.json: Qwen/Qwen3-Embedding-0.6B at revision
97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, max sequence length 4096, L2-normalised, plain issue text,
no instruction prefix. Run on CPU in float32 (the precision chosen after the 1 November timing test).

Reads only task_id and pre_issue_text. Fits nothing and reads no outcome columns.
Writes, per input file: embeddings_<name>.npy (float32, one row per task, in CSV order) and
task_ids_<name>.txt; plus embed_log.json with versions, settings, timings and SHA-256 hashes.
Run once. Do not re-run with different settings without a dated entry in protocol section 12.
"""
import csv, hashlib, json, platform, sys, time
from pathlib import Path

REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
USAGE = "Usage: python embed_all.py <tasks_pro266.csv> <tasks_label100.csv> <output-dir>"
# SHA-256 of the task tables recorded at the protocol freeze (DATA_HASHES.txt).
EXPECTED_SHA = {
    "tasks_pro266.csv": "9D4E96BA3375FA08A5BCD3E6187E1F63411BED459AF9F87FD716469B57D654D5",
    "tasks_label100.csv": "62990EDEB7EFC45A267110B24B902566CE07677D1972790F4BD31E1EE404AD4C",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ---- validate everything cheap BEFORE loading the model ----
if len(sys.argv) != 4:
    sys.exit(f"Expected 3 arguments, got {len(sys.argv) - 1}.\n{USAGE}")
files = [Path(sys.argv[1]), Path(sys.argv[2])]
out = Path(sys.argv[3])
problems = []
for p in files:
    if p.name not in EXPECTED_SHA:
        problems.append(f"{p.name}: not one of the frozen task tables {sorted(EXPECTED_SHA)}")
        continue
    if not p.is_file():
        problems.append(f"missing file: {p}")
        continue
    if sha(p) != EXPECTED_SHA[p.name]:
        problems.append(f"{p.name}: hash differs from the frozen table (line endings changed, or the file was edited)")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows or not {"task_id", "pre_issue_text"} <= set(rows[0]):
        problems.append(f"{p.name}: needs the columns task_id and pre_issue_text and at least one row")
        continue
    ids = [r["task_id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append(f"{p.name}: duplicate task_id values")
    empty = sum(1 for r in rows if not r["pre_issue_text"].strip())
    if empty:
        problems.append(f"{p.name}: {empty} empty issue texts")
if len({p.name for p in files}) != 2:
    problems.append("the two input files must be tasks_pro266.csv and tasks_label100.csv, one each")
if (out / "embed_log.json").exists():
    problems.append(f"{out} already holds a finished run (embed_log.json). The run is meant to happen once; "
                    "move it away deliberately if you really need to repeat it.")
if problems:
    sys.exit("Not starting:\n- " + "\n- ".join(problems) + f"\n{USAGE}")
out.mkdir(parents=True, exist_ok=True)
print("input checks passed (files, columns, unique ids, no empty texts, hashes match the freeze)")

import numpy as np
import sentence_transformers, torch, transformers
from sentence_transformers import SentenceTransformer


log = {
    "model": "Qwen/Qwen3-Embedding-0.6B", "revision": REVISION, "max_seq_length": 4096,
    "device": "cpu", "dtype": "float32", "normalize": True, "batch_size": 2, "prompt": None,
    "versions": {"python": platform.python_version(), "torch": torch.__version__,
                 "transformers": transformers.__version__, "sentence_transformers": sentence_transformers.__version__,
                 "numpy": np.__version__},
    "files": {},
}
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", revision=REVISION, device="cpu",
                            model_kwargs={"dtype": torch.float32})
model.max_seq_length = 4096

for path in files:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    ids = [r["task_id"] for r in rows]
    texts = [r["pre_issue_text"] for r in rows]
    t0 = time.time()
    emb = model.encode(texts, batch_size=2, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    dt = time.time() - t0
    assert emb.shape == (len(rows), 1024) and not np.isnan(emb).any()
    name = path.stem.replace("tasks_", "")
    np.save(out / f"embeddings_{name}.npy", emb)
    (out / f"task_ids_{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    log["files"][path.name] = {"input_sha256": sha(path), "n": len(rows), "seconds": round(dt, 1),
                               "embeddings_sha256": sha(out / f"embeddings_{name}.npy")}
    print(f"{path.name}: {len(rows)} tasks in {dt / 60:.1f} min")

(out / "embed_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
print("done; wrote embed_log.json")
