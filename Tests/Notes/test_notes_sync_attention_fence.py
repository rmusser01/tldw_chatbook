"""A failed attention read keeps the last known answer (final review I3, T2-a).

TASK-34000.2 made the Notes tree, list and editor say "needs attention" for a
held sync folder. Its read helper turned ANY failure into its default, and the
default for "which folders are held" is the empty set. Both runtime reads are
producer calls, so they raise while a backup holds the producer fence: a visit
to Notes or a save during a backup wrote the empty set, repainted, and left a
held folder reading "Sync managed", "Ready" and "Saved" until something else
published. "Could not say" is not "nothing held".

T2-a, same function: the refresh wrote the folder set before its second await.
An exclusive successor that cancelled it there compared equal and skipped the
paint, so the answer was stored and never shown.

The runtime, store, database and vault are the production stack (the shared
real-stack harness). The host is the handful of attributes the helper reads;
only the canvas repaint is recorded, because no widget tree exists here.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Notes.notes_sync_tail_edit_support import Vault, build_owner
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_NOTES
from tldw_chatbook.UI.Library_Modules import library_notes_sync_attention as attention
from tldw_chatbook.UI.Library_Modules.library_notes_state import LibraryNotesState
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_NOTES_SOURCE_DATABASE,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

HELD = frozenset({"folder-1"})


@pytest.fixture
def vault(tmp_path: Path):
    selected = Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


async def _held_owner(vault: Vault):
    """A started runtime whose only root is held: its bound note was deleted."""

    owner = build_owner(vault)
    await owner.start()
    version = int(vault.note()["version"])
    assert await vault.scope_service.delete_note(
        scope="local_note", note_id="note-1", version=version, user_id="user-1"
    )
    assert await owner.note_changed("note-1") == ("root-1",)
    await owner.settle()
    assert owner.snapshot().roots[0].status == "needs_attention"
    return owner


class _Host:
    """What the attention helpers read of the Notes controller or screen."""

    def __init__(self, owner, *, view: str = "list") -> None:
        self.app_instance = SimpleNamespace(notes_sync_runtime_owner=owner)
        self._notes_state = LibraryNotesState()
        self._selected_note_id: str | None = None
        self._library_note_location: tuple[str, str, bool] = ("", "", False)
        self._library_notes_view = view
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES
        self._library_notes_source = LIBRARY_NOTES_SOURCE_DATABASE
        self.is_mounted = True
        self.presented = 0

    def _apply_library_note_presentation_state(self) -> None:
        self.presented += 1


@pytest.fixture
def painted(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every Notes canvas repaint the helper asks for."""

    paints: list[str] = []
    monkeypatch.setattr(
        attention, "_sync_library_canvas", lambda _host, name: paints.append(name)
    )
    return paints


async def test_a_refresh_during_a_backup_keeps_a_held_folder_painted_held(
    vault: Vault, painted: list[str]
) -> None:
    owner = await _held_owner(vault)
    try:
        host = _Host(owner)
        await attention.refresh_library_notes_sync_attention(host)
        assert host._notes_state.tree_attention_folder_ids == HELD
        assert painted == ["notes"]

        owner._maintenance_close_admission()  # a backup starts capturing the profile
        try:
            await attention.refresh_library_notes_sync_attention(host)
            during = frozenset(host._notes_state.tree_attention_folder_ids)
            paints_during = list(painted)
        finally:
            owner._maintenance_resume()
        # The backup is over and nothing was published, so nothing refreshes.
        after = frozenset(host._notes_state.tree_attention_folder_ids)
        assert owner.snapshot().roots[0].status == "needs_attention"

        assert during == HELD, "a read the fence refused painted the folder healthy"
        assert paints_during == ["notes"], "a failed read must not repaint"
        assert after == HELD
        # And the next real read still says held.
        await attention.refresh_library_notes_sync_attention(host)
        assert host._notes_state.tree_attention_folder_ids == HELD
        assert painted == ["notes"]
    finally:
        await owner.shutdown()


async def test_the_open_note_keeps_its_held_line_while_the_runtime_cannot_say(
    vault: Vault, painted: list[str]
) -> None:
    """The editor's own line, and the same rule: keep what was last known."""

    owner = build_owner(vault)
    await owner.start()
    try:
        host = _Host(owner, view="editor")
        host._selected_note_id = "note-1"
        # Hold the folder with the note still open: a failed pass.
        await owner._publish("root-1", "failed", "review_changes")
        await attention.load_library_note_location(host, "note-1")
        path, _written, held = host._library_note_location
        assert path == str(vault.file)
        assert held is True
        presented = host.presented

        owner._maintenance_close_admission()
        try:
            await attention.load_library_note_location(host, "note-1")
        finally:
            owner._maintenance_resume()

        assert host._library_note_location[0] == path, (
            "a refused read made a synced note look database-only"
        )
        assert host._library_note_location[2] is True, (
            "a refused read painted the held note healthy"
        )
        assert host.presented == presented, "a failed read must not repaint"
        assert host._notes_state.tree_attention_folder_ids == HELD
    finally:
        await owner.shutdown()


async def test_a_superseded_refresh_leaves_the_paint_to_its_successor(
    vault: Vault, painted: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T2-a. An exclusive worker cancels its predecessor at an await. The
    predecessor must not have stored the answer its successor then finds
    "unchanged" -- that left a held folder painted healthy."""

    owner = await _held_owner(vault)
    try:
        host = _Host(owner, view="editor")
        host._selected_note_id = "note-1"
        host._library_note_location = (str(vault.file), "", False)
        entered = asyncio.Event()
        release = asyncio.Event()
        real = owner.note_sync_needs_attention

        async def gated(note_id: str) -> bool:
            entered.set()
            await release.wait()
            return await real(note_id)

        monkeypatch.setattr(owner, "note_sync_needs_attention", gated)
        first = asyncio.create_task(
            attention.refresh_library_notes_sync_attention(host)
        )
        await asyncio.wait_for(entered.wait(), 10)
        first.cancel()  # what ``exclusive=True`` does to the older worker
        with pytest.raises(asyncio.CancelledError):
            await first
        assert painted == []

        release.set()
        await attention.refresh_library_notes_sync_attention(host)

        assert host._notes_state.tree_attention_folder_ids == HELD
        assert painted == ["notes"], "the successor skipped the paint"
    finally:
        await owner.shutdown()
