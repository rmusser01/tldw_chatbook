"""Compare geometry dumps from geom_capture_plugin between base and head.

Usage: geom_diff.py <base-dir> <head-dir> [file-filter-substring ...]
Reports, per test node captured in BOTH trees, controls that are newly
zero-area, newly horizontally clipped, newly overlapping a sibling, or lost
(present at base, absent at head), plus which nodes changed layout at all.
"""
import collections
import glob
import json
import os
import sys
from typing import Any

#: One capture: ``node``, ``n``, ``screen`` and its widget ``rows``.
Record = dict[str, Any]
#: A widget row's identity: ``(kind, type, id, path)``.
RowKey = tuple[str, str, str | None, str]


def load(d: str) -> dict[tuple[str, int], Record]:
    """Read every capture geom_capture_plugin wrote under ``d``.

    Args:
        d: A directory of ``*.jsonl`` capture files.

    Returns:
        Each record keyed by ``(node, n)``: its test node id and the
        capture's ordinal within that test.
    """
    out = {}
    for path in glob.glob(os.path.join(d, "*.jsonl")):
        with open(path, encoding="utf-8") as capture:
            for line in capture:
                rec = json.loads(line)
                out[(rec["node"], rec["n"])] = rec
    return out


def key(row: dict[str, Any]) -> RowKey:
    """Return the identity a widget row keeps between base and head.

    Args:
        row: One entry of a record's ``rows``.

    Returns:
        ``(kind, type, id, path)``; ``id`` is ``None`` for an unnamed widget.
    """
    return (row["kind"], row["type"], row["id"], row["path"])


def index(rec: Record) -> dict[tuple[str, str, str | None, str, int], dict[str, Any]]:
    """Key a record's rows so repeated identities stay distinct.

    Args:
        rec: One capture record.

    Returns:
        Each row keyed by its ``key()`` plus its 1-based occurrence count.
    """
    counts: collections.Counter[RowKey] = collections.Counter()
    rows = {}
    for row in rec["rows"]:
        k = key(row)
        counts[k] += 1
        rows[k + (counts[k],)] = row
    return rows


base, head = load(sys.argv[1]), load(sys.argv[2])
filters = sys.argv[3:]
common = sorted(k for k in base.keys() & head.keys() if not filters or any(f in k[0] for f in filters))
print(f"captures: base={len(base)} head={len(head)} common={len(common)}")
changed_files = collections.Counter()
issues = []
for k in common:
    b, h = index(base[k]), index(head[k])
    if base[k]["screen"] != head[k]["screen"]:
        issues.append((k, "screen differs", base[k]["screen"], head[k]["screen"]))
        continue
    changed = False
    for rk, hr in h.items():
        br = b.get(rk)
        if br is None:
            changed = True
            continue
        if br["region"] != hr["region"]:
            changed = True
        if hr["zero"] and not br["zero"]:
            issues.append((k, "NEW zero-area", rk, br["region"], hr["region"]))
        if hr["hclip"] and not br["hclip"]:
            issues.append((k, "NEW h-clip", rk, br["region"], hr["region"]))
        new_ov = set(hr.get("overlaps", [])) - set(br.get("overlaps", []))
        if new_ov:
            issues.append((k, "NEW overlap", rk, sorted(new_ov)))
    for rk in b.keys() - h.keys():
        changed = True
        if rk[0] != "focusable" or not b[rk]["zero"]:
            issues.append((k, "LOST (shown at base, not at head)", rk, b[rk]["region"]))
    if changed:
        changed_files[k[0].split("::")[0]] += 1
print("nodes with any layout change, by file:")
for f, n in sorted(changed_files.items()):
    print(f"  {n:4d}  {f}")
print(f"issues: {len(issues)}")
for issue in issues:
    print("  ", issue)
