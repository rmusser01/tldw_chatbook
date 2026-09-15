"""Session Git geometry and the trust prompt's own identifier (task-32615).

Three shapes, all measured at 235x52, 100x30 and 60x24:

* the commit workflow's ``1fr`` scroll took every spare row of the pane, so
  its Cancel/Review actions painted at the pane floor, ~20 rows below the
  Subject and Body they act on;
* the trust dialog let Textual fold the repository path it asks consent for,
  which for the assessor's own vault landed inside "vaul"/"t"; and
* an action receipt was the work pane's LAST child, so "Committed 1 session
  note …" painted under the Manage region's "Danger" heading.

Every assertion here was first run against the unfixed tree and recorded
failing; the RED text is on the task.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Static

# Stubs first in the local group: it registers the optional MLX modules the
# application imports below would otherwise probe.
import Tests.UI._optional_module_stubs  # noqa: F401
from Tests.UI.test_library_file_notes_git import (
    _commit_draft_projection,
    _PanelHarness,
)
from Tests.UI.test_library_file_notes_workspace import (
    _production_workspace_context,
    _wait_until,
)
from Tests.UI.test_library_notes_w5_folder_files_layout import (
    CRITIQUE_SIZES,
    WIDE,
    folder_files_vault,
    painted_lines,
)
from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica
from tldw_chatbook.Utils.Utils import fold_path_lines
from tldw_chatbook.Widgets.Library.library_file_notes_git_panel import (
    _TRUST_DIALOG_MESSAGE_WIDTH,
    LibraryFileNotesGitPanel,
    SessionGitTrustDialog,
)
from tldw_chatbook.Widgets.Library.library_file_notes_workspace import (
    LibraryFileNotesWorkspace,
)

pytestmark = pytest.mark.asyncio


async def test_the_commit_form_actions_sit_directly_under_its_fields() -> None:
    """task-32615 AC#1: no twenty blank rows between a field and its action.

    Born red with the footer at row 48 under a form ending at row 12 -- the
    ``1fr`` scroll claimed every spare row of the pane.
    """
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    async with _PanelHarness(panel).run_test(size=WIDE) as pilot:
        panel.render_commit_availability(_commit_draft_projection())
        panel.query_one("#file-notes-git-commit-staged", Button).press()
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        assert panel.commit_phase == "form"
        form = panel.query_one("#file-notes-git-commit-form")
        footer = panel.query_one("#file-notes-git-commit-footer")
        gap = footer.region.y - (form.region.y + form.region.height)
        assert 0 <= gap <= 1, (
            f"{gap} rows between the commit fields and their actions "
            f"(form {form.region}, footer {footer.region})"
        )


async def test_the_commit_actions_stay_on_screen_when_the_phase_outgrows_the_pane() -> (
    None
):
    """task-32615 AC#1: the ceiling must not become a way off the bottom.

    The negative control for the test above -- the exact regression
    ``_sync_row_list_height``'s comment warns about one surface over. On a
    short pane the scroll still shrinks and the footer still paints.
    """
    panel = LibraryFileNotesGitPanel()
    panel.styles.display = "block"
    async with _PanelHarness(panel).run_test(size=(40, 20)) as pilot:
        panel.render_commit_availability(_commit_draft_projection())
        panel.query_one("#file-notes-git-commit-staged", Button).press()
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        footer = panel.query_one("#file-notes-git-commit-footer")
        assert footer.region.height > 0
        assert footer.region.y + footer.region.height <= 20, footer.region


@pytest.mark.parametrize("size", CRITIQUE_SIZES)
async def test_the_trust_dialog_never_breaks_the_path_it_asks_consent_for(
    size: tuple[int, int],
) -> None:
    """task-32615 AC#2: the whole path, every component intact.

    Born red painting "/Users/.../crit4/B/" then "power/vault" -- a fold at
    whatever column ran out, which for the assessor's own path landed inside
    "vaul"/"t". A consent dialog may not mangle its own identifier, and it
    may not elide it either, so the break moved to the separators.
    """
    # Chosen so the unfixed fold lands INSIDE a component: character 54 of
    # this path is the "l" of "...vaults", which is what painted "vaul"/"t"
    # for the assessor. A path whose 54th column happens to be a separator
    # would pass either way and prove nothing.
    path = "/Users/macbook-dev/Documents/qa/library-notes-and-vaults/crit4/power/vault"
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test(size=size) as pilot:
        pilot.app.push_screen(SessionGitTrustDialog(path))
        await pilot.pause()
        await pilot.pause()
        label = pilot.app.screen.query_one(".dialog-message")
        painted = [line.rstrip() for line in painted_lines(label)]
        path_lines = [line for line in painted if line.startswith("/") or "/" in line]
        assert "".join(path_lines) == path, painted
        for line in path_lines[:-1]:
            assert line.endswith("/"), (
                f"line {line!r} breaks a path component in half"
            )


def test_the_trust_dialog_fold_width_is_the_width_the_dialog_paints() -> None:
    """task-32615 AC#2: the constant is the measurement, not a guess."""
    path = "/" + "/".join(f"segment-{index}" for index in range(12))
    for line in fold_path_lines(path, _TRUST_DIALOG_MESSAGE_WIDTH).split("\n"):
        assert len(line) <= _TRUST_DIALOG_MESSAGE_WIDTH, line


def test_the_fold_keeps_every_character_and_leaves_short_paths_alone() -> None:
    """The three cases the helper has to get right to be safe here."""
    # Nothing to do.
    assert fold_path_lines("/short/path", 54) == "/short/path"
    assert fold_path_lines("/anything", 0) == "/anything"
    # Windows separators break too -- `str(Path(...))` produces whichever the
    # host uses, the same reason `elide_path_middle` handles both.
    folded = fold_path_lines("C:\\Users\\rob\\Documents\\vault\\notes", 12)
    assert folded.replace("\n", "") == "C:\\Users\\rob\\Documents\\vault\\notes"
    assert all(line.endswith("\\") for line in folded.split("\n")[:-1]), folded
    # A single component longer than the width has nowhere to break: it takes
    # its own line rather than being cut.
    single = fold_path_lines("/" + "x" * 80, 20)
    assert single.replace("\n", "") == "/" + "x" * 80


@pytest.mark.parametrize("size", CRITIQUE_SIZES)
async def test_an_action_receipt_never_paints_under_the_danger_heading(
    tmp_path: Path, size: tuple[int, int]
) -> None:
    """task-32615 AC#3/AC#4: the receipt reads as a receipt at every size.

    Born red with "Committed 1 session note as c1db79e…; unrelated changes
    untouched." rendering under "Danger" -- the status line was the work
    pane's last child, below the Manage region's last heading.
    """
    replica = FileNotesReplica(":memory:")
    workspace = LibraryFileNotesWorkspace(root=folder_files_vault(tmp_path), replica=replica)
    async with _production_workspace_context(workspace, size=size) as pilot:
        assert await workspace.open_path("note-00.md")
        await _wait_until(
            pilot, lambda: workspace.current_path == "note-00.md", "did not open"
        )
        workspace.query_one("#file-notes-manage", Button).press()
        workspace._set_action_status(
            "Committed 1 session note as c1db79e; unrelated changes untouched."
        )
        await pilot.pause()
        await pilot.pause()
        receipt = workspace.query_one("#file-notes-action-status", Static)
        danger = next(
            widget
            for widget in workspace.query(".destination-section")
            if str(widget.renderable) == "Danger"
        )
        assert receipt.region.y < danger.region.y, (
            f"receipt at {receipt.region} under Danger at {danger.region} at {size}"
        )
    await workspace.shutdown()
    replica.close()
