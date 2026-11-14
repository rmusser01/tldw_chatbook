"""File Notes delete-safety completion in the workspace (task-34383).

Restore refused an occupied destination with a bare "destination already
exists" and a missing parent with a raw errno line; neither offered a way
out. These tests pin the completed contract: both refusals name their reason
on the action line and reveal the "Export deleted copy" fallback, whose press
writes the exact deleted bytes to a new, absent path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Input

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.UI.test_library_file_notes_workspace import (
    _WorkspaceHarness,
    _static_text,
    _wait_until,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

WIDE = (235, 52)
DELETED_BYTES = b"the exact deleted payload\r\n"


def _action_status(workspace: LibraryFileNotesWorkspace) -> str:
    return _static_text(workspace, "#file-notes-action-status")


async def _press_restore(pilot, workspace: LibraryFileNotesWorkspace) -> None:
    await _wait_until(
        pilot,
        lambda: not workspace.query_one("#file-notes-restore", Button).disabled,
        "Restore button never became enabled",
    )
    workspace.query_one("#file-notes-restore").press()
    await pilot.pause()


@pytest.mark.asyncio
async def test_occupied_restore_refusal_offers_the_exact_export_fallback(
    tmp_path: Path,
) -> None:
    replica = FileNotesReplica(":memory:")
    root = tmp_path / "safety"
    root.mkdir()
    (root / "note.md").write_bytes(DELETED_BYTES)
    setup = FileNotesService(root, replica)
    opened = setup.open_file("note.md")
    assert (
        setup.delete_file("note.md", expected_hash=opened.content_hash).status
        == "ok"
    )
    # Someone recreates the path while it is tombstoned. This happens AFTER
    # the workspace has loaded Recently-deleted (a rescan would clear the
    # tombstone: the file is back, disk is authority), with polling idle so
    # the refusal is the deterministic next event.
    workspace = LibraryFileNotesWorkspace(
        root=root,
        replica=replica,
        poll_interval=60,
    )
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        (root / "note.md").write_bytes(b"a replacement now lives here\n")
        assert workspace.select_deleted("note.md")
        assert not workspace.query_one("#file-notes-export-deleted", Button).display

        await _press_restore(pilot, workspace)
        await _wait_until(
            pilot,
            lambda: "Restore refused" in _action_status(workspace),
            f"refusal never reported: {_action_status(workspace)!r}",
        )
        assert "never replaces" in _action_status(workspace)
        assert "Export deleted copy" in _action_status(workspace)
        fallback = workspace.query_one("#file-notes-export-deleted", Button)
        await _wait_until(
            pilot,
            lambda: fallback.display,
            "Export deleted copy fallback never appeared",
        )
        assert (root / "note.md").read_bytes() == b"a replacement now lives here\n"

        workspace.query_one("#file-notes-path", Input).value = "recovered.md"
        fallback.press()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (root / "recovered.md").exists(),
            "recovered.md never appeared on disk",
        )
        assert (root / "recovered.md").read_bytes() == DELETED_BYTES
        await _wait_until(
            pilot,
            lambda: "Exported the deleted bytes exactly" in _action_status(workspace),
            f"export receipt never arrived: {_action_status(workspace)!r}",
        )
        # The occupied file was never touched by the fallback.
        assert (root / "note.md").read_bytes() == b"a replacement now lives here\n"
    replica.close()


@pytest.mark.asyncio
async def test_missing_parent_restore_refusal_offers_the_fallback(
    tmp_path: Path,
) -> None:
    replica = FileNotesReplica(":memory:")
    root = tmp_path / "safety-missing"
    root.mkdir()
    nested = root / "gone"
    nested.mkdir()
    (nested / "note.md").write_bytes(DELETED_BYTES)
    setup = FileNotesService(root, replica)
    opened = setup.open_file("gone/note.md")
    assert (
        setup.delete_file(
            "gone/note.md",
            expected_hash=opened.content_hash,
        ).status
        == "ok"
    )
    nested.rmdir()

    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert workspace.select_deleted("gone/note.md")

        await _press_restore(pilot, workspace)
        await _wait_until(
            pilot,
            lambda: "Parent directory is missing" in _action_status(workspace),
            f"missing-parent reason never reported: {_action_status(workspace)!r}",
        )
        assert "Export deleted copy" in _action_status(workspace)
        assert not nested.exists()

        fallback = workspace.query_one("#file-notes-export-deleted", Button)
        await _wait_until(
            pilot,
            lambda: fallback.display,
            "Export deleted copy fallback never appeared",
        )
        workspace.query_one("#file-notes-path", Input).value = "flat.md"
        fallback.press()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (root / "flat.md").exists(),
            "flat.md never appeared on disk",
        )
        assert (root / "flat.md").read_bytes() == DELETED_BYTES
    replica.close()


@pytest.mark.asyncio
async def test_restore_refusal_marker_does_not_survive_a_root_change(
    tmp_path: Path,
) -> None:
    """PR #3016 review finding 8: the fallback marker is root-scoped.

    Root A's tombstone restore is refused and offers Export deleted copy;
    adopting root B -- which carries a SAME-NAMED tombstone -- must retire
    the marker: B's tombstone never had a refused restore, so the fallback
    stays hidden until B's own restore is refused.
    """
    replica = FileNotesReplica(":memory:")
    first = tmp_path / "refused-root"
    second = tmp_path / "next-root"
    first.mkdir()
    second.mkdir()
    (first / "note.md").write_bytes(b"root A deleted bytes\n")
    setup = FileNotesService(first, replica)
    opened = setup.open_file("note.md")
    assert (
        setup.delete_file("note.md", expected_hash=opened.content_hash).status
        == "ok"
    )
    # Root B arrives with its own same-named tombstone.
    (second / "note.md").write_bytes(b"root B deleted bytes\n")
    setup_b = FileNotesService(second, replica)
    opened_b = setup_b.open_file("note.md")
    assert (
        setup_b.delete_file("note.md", expected_hash=opened_b.content_hash).status
        == "ok"
    )

    workspace = LibraryFileNotesWorkspace(
        root=first,
        replica=replica,
        poll_interval=60,
    )
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        # Someone occupies root A's deleted path; the restore is refused.
        (first / "note.md").write_bytes(b"a replacement now lives here\n")
        assert workspace.select_deleted("note.md")
        await _press_restore(pilot, workspace)
        await _wait_until(
            pilot,
            lambda: "Restore refused" in _action_status(workspace),
            f"refusal never reported: {_action_status(workspace)!r}",
        )
        assert workspace.query_one("#file-notes-export-deleted", Button).display

        # Adopting root B retires the marker even though B lists a
        # same-named tombstone.
        assert await workspace.set_root(second, persist=False)
        await _wait_until(
            pilot,
            lambda: workspace.root == second.resolve(),
            "second root was not adopted",
        )
        assert workspace._deleted_paths == ("note.md",)
        assert workspace.select_deleted("note.md")
        fallback = workspace.query_one("#file-notes-export-deleted", Button)
        assert not fallback.display

        # The mechanism itself still works: refusing B's own restore
        # (the deleted path is occupied again) re-offers the fallback.
        (second / "note.md").write_bytes(b"a root B replacement now lives here\n")
        await _press_restore(pilot, workspace)
        await _wait_until(
            pilot,
            lambda: fallback.display,
            "fallback never reappeared for root B's own refusal",
        )
    replica.close()
