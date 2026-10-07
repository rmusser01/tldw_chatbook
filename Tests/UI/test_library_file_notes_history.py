"""Protected-file revision History affordance (task-34381).

The ``revisions`` table had only writers. These tests pin the read-path as a
user meets it: a bounded listing in the Folder-files workspace, verify
against the stored hash, exact export, and no-replace restore -- each with
its refusal shapes named on the action line (never as an editor conflict,
because history actions do not touch the open document).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from textual.widgets import OptionList

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.UI.test_library_file_notes_workspace import (
    _WorkspaceHarness,
    _static_text,
    _wait_until,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    FileNotesHistoryDialog,
    LibraryFileNotesWorkspace,
)

WIDE = (235, 52)


def _digest(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()


def _protected_file_with_checkpoint(
    tmp_path: Path,
    replica: FileNotesReplica,
) -> tuple[Path, bytes, bytes]:
    """One protected note on disk plus an older checkpoint revision."""
    root = tmp_path / "vault"
    root.mkdir()
    checkpoint_bytes = b"the checkpoint era body\n"
    current_bytes = b"the current body\n"
    (root / "important.md").write_bytes(current_bytes)
    root_key = str(root.resolve())
    replica.protect(root_key, "important.md")
    replica.checkpoint(
        root_key,
        "important.md",
        checkpoint_bytes,
        content_hash=_digest(checkpoint_bytes),
        session_key="session-1",
        created_at="2026-10-01T10:00:00Z",
    )
    return root, checkpoint_bytes, current_bytes


def _action_status(workspace: LibraryFileNotesWorkspace) -> str:
    return _static_text(workspace, "#file-notes-action-status")


async def _open_history_dialog(pilot, workspace: LibraryFileNotesWorkspace):
    """Expand the maintenance row (once) and open the History dialog."""
    if not workspace._maintenance_expanded:
        workspace.query_one("#file-notes-maintenance-toggle").press()
        await pilot.pause()
    await _wait_until(
        pilot,
        lambda: workspace.query_one("#file-notes-history").display,
        "History button never became visible",
    )
    workspace.query_one("#file-notes-history").press()
    await pilot.pause()
    await _wait_until(
        pilot,
        lambda: isinstance(pilot.app.screen, FileNotesHistoryDialog),
        "History dialog never opened",
    )
    dialog = pilot.app.screen
    assert isinstance(dialog, FileNotesHistoryDialog)
    return dialog


def _select_first_revision(dialog: FileNotesHistoryDialog) -> None:
    listing = dialog.query_one("#file-notes-history-list", OptionList)
    listing.highlighted = 0


@pytest.mark.asyncio
async def test_history_dialog_lists_verifies_and_exports_a_revision(
    tmp_path: Path,
) -> None:
    replica = FileNotesReplica(":memory:")
    root, checkpoint_bytes, _current_bytes = _protected_file_with_checkpoint(
        tmp_path,
        replica,
    )
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("important.md")
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "important.md",
            "important.md did not open",
        )

        dialog = await _open_history_dialog(pilot, workspace)
        listing = dialog.query_one("#file-notes-history-list", OptionList)
        # Bounded listing: newest first, one line per revision.
        await _wait_until(
            pilot,
            lambda: len(listing.options) == 1,
            "History listing never populated",
        )
        assert "pre_edit" in str(listing.options[0].prompt)
        assert "session-1" in str(listing.options[0].prompt)

        # Verify runs in the workspace and reports on the action line.
        _select_first_revision(dialog)
        dialog.query_one("#file-notes-history-verify").press()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: "Revision verified" in _action_status(workspace),
            f"verify status never arrived: {_action_status(workspace)!r}",
        )

        # Export writes the exact checkpoint bytes to a new, absent path.
        dialog = await _open_history_dialog(pilot, workspace)
        _select_first_revision(dialog)
        destination = dialog.query_one("#file-notes-history-destination")
        destination.value = "exported.md"
        dialog.query_one("#file-notes-history-export").press()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (root / "exported.md").exists(),
            "exported.md never appeared on disk",
        )
        assert (root / "exported.md").read_bytes() == checkpoint_bytes
        await _wait_until(
            pilot,
            lambda: "History export succeeded" in _action_status(workspace),
            f"export status never arrived: {_action_status(workspace)!r}",
        )
        # The open document was never marked conflicted by a history action.
        assert workspace.save_state == "saved"
    replica.close()


@pytest.mark.asyncio
async def test_history_restore_refuses_an_occupied_destination(
    tmp_path: Path,
) -> None:
    replica = FileNotesReplica(":memory:")
    root, checkpoint_bytes, _current_bytes = _protected_file_with_checkpoint(
        tmp_path,
        replica,
    )
    (root / "taken.md").write_bytes(b"do not touch\n")
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("important.md")
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "important.md",
            "important.md did not open",
        )

        # A refused restore names the no-replace reason on the action line.
        dialog = await _open_history_dialog(pilot, workspace)
        _select_first_revision(dialog)
        destination = dialog.query_one("#file-notes-history-destination")
        destination.value = "taken.md"
        dialog.query_one("#file-notes-history-restore").press()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: "never replaces" in _action_status(workspace),
            f"occupied refusal never arrived: {_action_status(workspace)!r}",
        )
        assert (root / "taken.md").read_bytes() == b"do not touch\n"

        # The same dialog restores to an absent path exactly once.
        dialog = await _open_history_dialog(pilot, workspace)
        _select_first_revision(dialog)
        destination = dialog.query_one("#file-notes-history-destination")
        destination.value = "restored.md"
        dialog.query_one("#file-notes-history-restore").press()
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: (root / "restored.md").exists(),
            "restored.md never appeared on disk",
        )
        assert (root / "restored.md").read_bytes() == checkpoint_bytes
    replica.close()


@pytest.mark.asyncio
async def test_history_destination_input_rejects_unsupported_text(
    tmp_path: Path,
) -> None:
    """PR #3016 review finding 4: the shared text validator, not just trim.

    The workspace's path input refuses text beyond the shared length bound
    before dispatch; the History dialog's destination must apply the same
    check instead of carrying any nonempty string onto a service call.
    """
    replica = FileNotesReplica(":memory:")
    root, _checkpoint_bytes, _current_bytes = _protected_file_with_checkpoint(
        tmp_path,
        replica,
    )
    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert await workspace.open_path("important.md")
        await _wait_until(
            pilot,
            lambda: workspace.current_path == "important.md",
            "important.md did not open",
        )

        dialog = await _open_history_dialog(pilot, workspace)
        _select_first_revision(dialog)
        destination = dialog.query_one("#file-notes-history-destination")
        # Far beyond the shared 4096-character bound the workspace path
        # input enforces.
        destination.value = "x" * 5000
        dialog.query_one("#file-notes-history-export").press()
        await pilot.pause()

        # The dialog refuses locally: it stays open with the reason shown,
        # and no request reached the workspace.
        assert pilot.app.screen is dialog
        assert "not supported" in _dialog_status(dialog)
        assert "History export" not in _action_status(workspace)
    replica.close()


def _dialog_status(dialog: FileNotesHistoryDialog) -> str:
    return _static_text(dialog, "#file-notes-history-status")


@pytest.mark.asyncio
async def test_history_action_is_rejected_after_the_root_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR #3016 review finding 13: no acting on another root.

    A History dialog's request carries the service and root generation that
    supplied its listing; if the root changed before the dismissed dialog's
    action runs, the action is refused before any service call.
    """
    from tldw_chatbook.Notes.file_notes_service import FileNotesService
    from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
        _HistoryActionRequest,
    )

    replica = FileNotesReplica(":memory:")
    root, _checkpoint_bytes, _current_bytes = _protected_file_with_checkpoint(
        tmp_path,
        replica,
    )
    other = tmp_path / "other-vault"
    other.mkdir()
    (other / "important.md").write_bytes(b"another root same name\n")
    replica.protect(str(other.resolve()), "important.md")

    calls: list[str] = []
    real_verify = FileNotesService.verify_revision

    def recording_verify(service_self, *args: object, **kwargs: object) -> object:
        calls.append(service_self.root_key)
        return real_verify(service_self, *args, **kwargs)

    monkeypatch.setattr(FileNotesService, "verify_revision", recording_verify)

    workspace = LibraryFileNotesWorkspace(root=root, replica=replica)
    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        current_service = workspace._service
        assert current_service is not None

        stale_request = _HistoryActionRequest(
            action="verify",
            relative_path="important.md",
            kind="pre_edit",
            session_key="session-1",
            service=object(),  # The service that supplied the listing.
            root_generation=workspace._root_generation + 5,
            revision_id=1,
        )
        await workspace._execute_history_action(stale_request)
        await pilot.pause()

        assert "root changed" in _action_status(workspace)
        assert calls == []

        # A current request still runs against the live service.
        fresh_request = _HistoryActionRequest(
            action="verify",
            relative_path="important.md",
            kind="pre_edit",
            session_key="session-1",
            service=current_service,
            root_generation=workspace._root_generation,
            revision_id=1,
        )
        await workspace._execute_history_action(fresh_request)
        await pilot.pause()
        await _wait_until(
            pilot,
            lambda: "Revision verified" in _action_status(workspace),
            f"verify status never arrived: {_action_status(workspace)!r}",
        )
        assert calls == [current_service.root_key]
    replica.close()
