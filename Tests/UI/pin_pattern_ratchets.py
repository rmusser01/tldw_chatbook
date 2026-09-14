"""Verify both hard-zero floors before writing the empty baseline artifact.

Usage: .venv/bin/python Tests/UI/pin_pattern_ratchets.py

Neither floor can be raised by pinning. The shared AST inventory covers all
Python write forms and preserves explicitly documented runtime exceptions.
The dimension regex matches the governance test; token definitions alone are
exempt from the sheet scan. No arguments are accepted to avoid accidental pins.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from python_style_inventory import inventory_styles

# This script has NO argument parser ON PURPOSE: any invocation re-pins the
# baseline (an accidental `--help` run executed the pin during ADR-161 task 11
# -- luckily onto identical counts). Refuse stray argv instead of guessing.
if len(sys.argv) != 1:
    sys.exit(
        "refusing: this script takes no arguments (it re-pins the ratchet baseline); pass none to pin"
    )

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "tldw_chatbook/css"
PKG = ROOT / "tldw_chatbook"

_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_DIM = re.compile(
    r"\b(?:padding|margin|width|height)(?:-(?:top|right|bottom|left))?\s*:\s*(?:[^;{}\s]+\s+)*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
)


def sheets() -> list[Path]:
    out: list[Path] = []
    for sub in ("core", "layout", "components", "features", "utilities"):
        out += sorted((CSS / sub).glob("*.tcss"))
    return out


# Dimension guard (ADR-161 task 11 close-out): refuse to write a baseline
# while any sheet outside the tokens file still carries a raw numeric
# dimension literal -- the governance test's floor is a hard zero, and a pin
# must never silently bless a regression. core/_variables.tcss is exempt:
# raw values are legal ONLY in token definitions (ADR-150).
dim_offenders = [
    f"{s.relative_to(CSS)}: {len(_DIM.findall(_COMMENT.sub('', s.read_text(encoding='utf-8'))))}"
    for s in sheets()
    if str(s.relative_to(CSS)) != "core/_variables.tcss"
    and _DIM.findall(_COMMENT.sub("", s.read_text(encoding="utf-8")))
]
if dim_offenders:
    sys.exit(
        "refusing to pin: raw numeric dimension literals remain (the "
        "dimension floor is a HARD ZERO, ADR-161 task 11):\n  "
        + "\n  ".join(dim_offenders)
    )

offenders = []
for path in sorted(PKG.rglob("*.py")):
    for write in inventory_styles(path.read_text(encoding="utf-8")):
        if write.violation:
            offenders.append(
                f"{path.relative_to(PKG)}:{write.line}: "
                f"{write.form} {write.property} ({write.value_kind})"
            )
if offenders:
    sys.exit(
        "refusing to pin: Python visual-style floor is HARD ZERO "
        "(ADR-161 task 12):\n  " + "\n  ".join(offenders)
    )

baseline: dict[str, dict[str, int]] = {"python_styles": {}}
Path(__file__).parent.joinpath("pattern_ratchet_baseline.json").write_text(
    json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps({k: sum(v.values()) for k, v in baseline.items()}))
