#!/usr/bin/env python3
"""Smoke test for the secondary Advisor variant: can Qwen3-Embedding-0.6B run on this machine (CPU)?

Usage:  python embed_test.py <tasks_pro266.csv> [n_texts]

Uses the exact settings recorded in Scrouting's router_spec.json: model Qwen/Qwen3-Embedding-0.6B at
revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3, max sequence length 4096, L2-normalised, plain issue text
with no instruction prefix. Reads only task_id and pre_issue_text. Fits nothing and reads no outcomes.
"""
import csv, sys, time

REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10

rows = list(csv.DictReader(open(path, encoding="utf-8")))
texts = [r["pre_issue_text"] for r in rows[:n]]
print(f"{len(texts)} texts, mean length {sum(map(len, texts)) / len(texts):.0f} characters")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "(CPU is fine)")
t0 = time.time()
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", revision=REVISION, device="cpu",
                            model_kwargs={"dtype": torch.float32})  # float32 is usually much faster than bfloat16 on CPU
model.max_seq_length = 4096
print(f"model loaded in {time.time() - t0:.1f} s")

t0 = time.time()
emb = model.encode(texts, batch_size=2, normalize_embeddings=True, show_progress_bar=False)
dt = time.time() - t0
print(f"encoded in {dt:.1f} s = {dt / len(texts):.2f} s per text")
print("estimated time for all 266 tasks:", f"{dt / len(texts) * 266 / 60:.1f} minutes")

print("shape:", emb.shape, "(expected (n, 1024))")
print("norms min/max:", float(np.linalg.norm(emb, axis=1).min()), float(np.linalg.norm(emb, axis=1).max()), "(expected about 1.0)")
print("any NaN:", bool(np.isnan(emb).any()))
sim = emb @ emb.T
off = sim[~np.eye(len(emb), dtype=bool)]
print(f"cosine similarity between different tasks: min {off.min():.3f}, mean {off.mean():.3f}, max {off.max():.3f}")
again = model.encode(texts[:1], normalize_embeddings=True, show_progress_bar=False)
print("same text encoded twice, cosine:", float((again[0] * emb[0]).sum()), "(expected about 1.0; small differences from batching are normal)")
print("PASS" if emb.shape[1] == 1024 and not np.isnan(emb).any() else "CHECK THE OUTPUT ABOVE")
