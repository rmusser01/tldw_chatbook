"""Reject unresolved merge conflicts in published User Guide pages."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFLICT_MARKER = re.compile(r"^(?:<{7}|>{7}|\|{7})(?: .*)?$|^={7}$")


def test_user_guide_has_no_merge_conflict_markers() -> None:
    """Every published page is free of Git conflict marker lines."""
    conflicts = []
    for page in sorted((REPO_ROOT / "Docs/User_Guide").rglob("*.md")):
        for line_number, line in enumerate(
            page.read_text(encoding="utf-8").splitlines(), 1
        ):
            if CONFLICT_MARKER.fullmatch(line):
                conflicts.append(f"{page.relative_to(REPO_ROOT)}:{line_number}")
    assert not conflicts, "Unresolved User Guide merge conflicts: " + ", ".join(
        conflicts
    )
