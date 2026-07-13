"""Task 173: the notes unsaved-indicator machinery (reactives + watchers + CSS)
was fully dead (widget never composed, reactives never mutated). These guard
against its presence / reintroduction and confirm the app still boots after
its removal."""
from pathlib import Path
from typing import Iterator, List, Tuple

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


def _source_files() -> Iterator[Path]:
    """Yield every shipped ``.py``/``.tcss`` file under ``tldw_chatbook`` (skipping caches)."""
    for pattern in ("**/*.py", "**/*.tcss"):
        for path in _PKG.glob(pattern):
            if "__pycache__" in path.parts:
                continue
            yield path


@pytest.fixture(scope="session")
def source_texts() -> List[Tuple[str, str]]:
    """Read every shipped source file once, shared across the parametrized checks.

    Returns:
        A list of ``(relative_path, file_text)`` pairs so the identifier scan
        below reads the tree a single time rather than once per identifier.
    """
    return [
        (str(p.relative_to(_PKG.parent)), p.read_text(encoding="utf-8", errors="ignore"))
        for p in _source_files()
    ]


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate HOME/XDG_DATA_HOME so the boot smoke can't touch the real config.

    Args:
        tmp_path: pytest's per-test temporary directory.
        monkeypatch: pytest's environment patcher.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / ".local" / "share"))


@pytest.mark.parametrize("identifier", _DEAD_IDENTIFIERS)
def test_dead_unsaved_indicator_identifier_removed(
    identifier: str, source_texts: List[Tuple[str, str]]
) -> None:
    """No dead unsaved-indicator reference remains in shipped tldw_chatbook source.

    Args:
        identifier: A dead symbol/class name that must not appear in any shipped source file.
        source_texts: The ``(path, text)`` pairs for every shipped source file (session fixture).
    """
    hits = [rel for rel, text in source_texts if identifier in text]
    assert not hits, f"{identifier!r} still present in: {hits}"


@pytest.mark.asyncio
async def test_app_boots_after_removal() -> None:
    """The app mounts and the modular stylesheet loads with the CSS rules gone."""
    from tldw_chatbook.app import TldwCli
    app = TldwCli()
    async with app.run_test() as pilot:
        await pilot.pause()
