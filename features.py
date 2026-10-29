#!/usr/bin/env python3
"""Pre-run features for the Delegation Advisor (protocol v1.0 candidate, section 5).

Usage:  python3 -I features.py <tasks_csv> <features_csv>

Reads only the `task_id` and `pre_issue_text` columns of a table made by build_task_table.py.
It never reads out_* or post_* columns. Standard library only.

All eight features are computed from the issue text alone. Count features enter the model as
ln(1 + count); the file stores the raw counts and the transformed values side by side.
"""
import csv, math, re, sys

FENCE = "```"
PATH_RE = re.compile(r"[\w./\\-]+\.(?:py|yml|yaml|json|cfg|ini|toml|txt|md|rst|js|ts|html|css|sh)\b")
INLINE_RE = re.compile(r"`([^`\n]{1,80})`")
TRACE_RE = re.compile(r"Traceback \(most recent call last\)|File \".+\", line \d+|\b[A-Z]\w*(?:Error|Exception)\b")
URL_RE = re.compile(r"https?://")
TEST_RE = re.compile(r"\b(?:tests?|pytest|unittest|assert(?:ion)?s?)\b", re.I)
LIST_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+\S", re.M)
HEAD_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S", re.M)


def features(text: str) -> dict:
    code_refs = set(INLINE_RE.findall(text)) | set(PATH_RE.findall(text))
    raw = {
        "n_chars": len(text),                              # F1 (enters as ln(1 + n_chars))
        "n_code_blocks": text.count(FENCE) // 2,           # F2 fenced code blocks
        "has_trace_or_error": int(bool(TRACE_RE.search(text))),  # F3
        "n_code_refs": len(code_refs),                     # F4 distinct inline-code spans and file-like names
        "n_headings": len(HEAD_RE.findall(text)),          # F5 markdown headings
        "n_urls": len(URL_RE.findall(text)),               # F6
        "mentions_tests": int(bool(TEST_RE.search(text))), # F7
        "n_list_items": len(LIST_RE.findall(text)),        # F8 numbered or bulleted lines
    }
    return raw


COUNTS = ["n_chars", "n_code_blocks", "n_code_refs", "n_headings", "n_urls", "n_list_items"]


def main(src, dst):
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    out_cols = ["task_id"]
    sample = features("")
    out_cols += list(sample) + [f"ln1p_{c}" for c in COUNTS]
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_cols)
        for r in rows:
            ft = features(r["pre_issue_text"])
            w.writerow([r["task_id"]] + [ft[k] for k in sample] + [round(math.log1p(ft[c]), 6) for c in COUNTS])
    print(f"wrote {len(rows)} rows to {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
