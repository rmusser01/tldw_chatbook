"""A hard DELETE of a character card must not be reachable from production code.

task-19566 F12: ``character_cards -> conversations -> messages`` carries
``ON DELETE CASCADE`` with foreign keys ON, so one hard ``DELETE`` of a
character card would take the user's entire chat history for it. Nothing in
production hard-deletes today -- the app soft-deletes (``UPDATE ... SET
deleted = 1``) -- but that safety is held by wiring, not by an invariant.
This census makes it an invariant: adding a ``DELETE FROM character_cards``
site anywhere under ``tldw_chatbook/`` turns this test red by file and line,
forcing the author to either use the soft-delete path or extend the explicit
allowlist below with a justification.
"""

import re
from pathlib import Path

PRODUCTION_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: Deliberate hard-delete sites. Empty today; every addition must cite why it
#: cannot take the soft-delete path and why the cascade is safe there.
ALLOWED_SITES: tuple[str, ...] = ()

_HARD_DELETE = re.compile(r"DELETE\s+FROM\s+character_cards\b", re.IGNORECASE)


def test_no_production_hard_delete_of_character_cards() -> None:
    hits: list[str] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        for mo in _HARD_DELETE.finditer(source):
            line = source.count("\n", 0, mo.start()) + 1
            hits.append(f"{path.relative_to(PRODUCTION_ROOT.parent)}:{line}")
    unexpected = [h for h in hits if not any(h == a for a in ALLOWED_SITES)]
    assert unexpected == [], (
        "Hard DELETE of character_cards found in production code "
        f"(cascade would erase its chat history): {unexpected}. "
        "Use the soft-delete path, or extend ALLOWED_SITES with a justification."
    )
