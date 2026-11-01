"""Workspace invocation of replica retention (task-34382, ADR-218).

The retention policy itself is pinned in ``Tests/Notes/test_file_notes_retention.py``;
these tests pin the two invocation seams the AC names -- cleanup runs when the
selected root changes (before Recently-deleted is read) and when the workspace
session ends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.UI.test_library_file_notes_workspace import (
    _WorkspaceHarness,
    _wait_until,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Notes.file_notes_service import FileNotesService
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

WIDE = (235, 52)


@pytest.mark.asyncio
async def test_root_change_and_shutdown_each_enforce_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replica = FileNotesReplica(":memory:")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "one.md").write_text("one\n", encoding="utf-8")
    (second / "two.md").write_text("two\n", encoding="utf-8")
    workspace = LibraryFileNotesWorkspace(root=first, replica=replica)

    calls: list[str] = []
    real_enforce = FileNotesService.enforce_retention

    def recording_enforce(service_self: FileNotesService) -> object:
        calls.append(service_self.root_key)
        return real_enforce(service_self)

    monkeypatch.setattr(
        FileNotesService,
        "enforce_retention",
        recording_enforce,
    )

    async with _WorkspaceHarness(workspace).run_test(size=WIDE) as pilot:
        await _wait_until(pilot, lambda: workspace.initialized, "scan did not finish")
        assert workspace.root == first.resolve()

        assert await workspace.set_root(second, persist=False)
        await _wait_until(
            pilot,
            lambda: workspace.root == second.resolve(),
            "second root was not adopted",
        )
        # The root-change seam ran for the adopted root before the
        # Recently-deleted listing was read.
        assert calls.count(str(second.resolve())) >= 1

        await workspace.shutdown()
        # The session-end seam ran once more before the service retired.
        assert calls.count(str(second.resolve())) >= 2
    replica.close()
