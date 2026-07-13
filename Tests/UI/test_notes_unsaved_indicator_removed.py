"""Task 173: the notes unsaved-indicator machinery (reactives + watchers + CSS)
was fully dead (widget never composed, reactives never mutated). These guard
against its presence / reintroduction and confirm the app still boots after
its removal."""
from pathlib import Path

import pytest

# Scope the scan to shipped source only — NOT this test file (which necessarily
# names the identifiers), Docs/, or backlog/.
_PKG = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_DEAD_IDENTIFIERS = [
    "notes_unsaved_changes",
    "notes_auto_save_status",
    "watch_notes_unsaved_changes",
    "watch_notes_auto_save_status",
    "notes-unsaved-indicator",
    "unsaved-indicator",
]


def _source_files():
    for pattern in ("**/*.py", "**/*.tcss"):
        for path in _PKG.glob(pattern):
            if "__pycache__" in path.parts:
                continue
            yield path


@pytest.mark.parametrize("identifier", _DEAD_IDENTIFIERS)
def test_dead_unsaved_indicator_identifier_removed(identifier):
    """No dead unsaved-indicator reference remains in shipped tldw_chatbook source."""
    hits = [
        str(p.relative_to(_PKG.parent))
        for p in _source_files()
        if identifier in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not hits, f"{identifier!r} still present in: {hits}"


@pytest.mark.asyncio
async def test_app_boots_after_removal():
    """The app mounts and the modular stylesheet loads with the CSS rules gone."""
    from tldw_chatbook.app import TldwCli
    app = TldwCli()
    async with app.run_test() as pilot:
        await pilot.pause()
