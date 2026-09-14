"""Pin current literal/style counts as the ratchet baseline (spec 3.7/3.10).

Usage (from the repo root)::

    .venv/bin/python Tests/UI/pin_pattern_ratchets.py

Rewrites ``Tests/UI/pattern_ratchet_baseline.json`` with the current
comment-stripped counts, then prints the per-kind totals.  The governance
test (``Tests/UI/test_component_pattern_governance.py``) fails on any count
above this baseline; the floor is zero.  Run this deliberately when an
intentional migration step lowers counts and you want to re-pin, and commit
the resulting JSON with the change that earned it (ADR-161).

The counting regexes here are byte-identical to the ones in the governance
test -- if you change one, change both (spec 3.7 pins them as one contract).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# This script has NO argument parser ON PURPOSE: any invocation re-pins the
# baseline (an accidental `--help` run executed the pin during ADR-161 task 11
# -- luckily onto identical counts). Refuse stray argv instead of guessing.
if len(sys.argv) != 1:
    sys.exit("refusing: this script takes no arguments (it re-pins the ratchet baseline); pass none to pin")

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "tldw_chatbook/css"
PKG = ROOT / "tldw_chatbook"

_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_DIM = re.compile(r"\b(?:padding|margin|width|height)(?:-(?:top|right|bottom|left))?\s*:\s*[0-9]")
_PYSTYLE = re.compile(r"\.styles\.(?:background|color|border\w*|width|height|padding\w*|margin\w*|opacity\w*)\s*=\s*[^=]")


def sheets() -> list[Path]:
    out: list[Path] = []
    for sub in ("core", "layout", "components", "features", "utilities"):
        out += sorted((CSS / sub).glob("*.tcss"))
    return out


baseline: dict[str, dict[str, int]] = {"dimensions": {}, "python_styles": {}}
for s in sheets():
    n = len(_DIM.findall(_COMMENT.sub("", s.read_text(encoding="utf-8"))))
    if n:
        baseline["dimensions"][str(s.relative_to(CSS))] = n
for p in sorted(PKG.rglob("*.py")):
    n = len(_PYSTYLE.findall(p.read_text(encoding="utf-8", errors="ignore")))
    if n:
        baseline["python_styles"][str(p.relative_to(PKG))] = n
Path(__file__).parent.joinpath("pattern_ratchet_baseline.json").write_text(
    json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({k: sum(v.values()) for k, v in baseline.items()}))
