"""Library ▸ Notes wave-4 group `sync-review` (task-32535).

The Keep-a-folder-synced review used to list sixty rows reading only
"Safe item N / Create a Library note" -- no path, no destination -- and
imported `.trash/`, `Templates/`, empty files and frontmatter that Import
once, one button to the left on the same vault, skips or lifts. These pins
hold the review to Import once's row grammar (path · what happens · where),
its collapsed uniform runs, its Skipped group, and the Obsidian toggle on
the sync setup form.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from textual.widgets import Checkbox, Collapsible, Input, Static

from Tests.Widgets.Library.test_library_notes_add_from_files_canvas import (
    _Host,
    _frame,
)
from tldw_chatbook.Library.library_notes_lasting_sync_state import (
    LastingSyncApplyBlocker,
    LastingSyncReview,
    LastingSyncReviewRow,
    LastingSyncSetup,
    initial_lasting_sync_snapshot,
)
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncAction,
    NotesSyncActionKind,
)
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationPlan
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRootSetup,
    NotesSyncRuntimeSnapshot,
    RuntimeBindingLabel,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)
from tldw_chatbook.Widgets.Library.library_notes_add_from_files_canvas import (
    LibraryNotesAddFromFilesCanvas,
)

pytestmark = pytest.mark.asyncio

TOKEN = "c" * 64
WIDE = (235, 52)


def _safe_row(index: int, *, folder: str = "Archive") -> LastingSyncReviewRow:
    return LastingSyncReviewRow(
        f"bind-{index}",
        "safe",
        "Create a Library note",
        action_id=f"act-{index}",
        relative_path=f"{folder}/Archived note {index:03d}.md",
        destination="Vault",
    )


def _review_snapshot(rows: tuple[LastingSyncReviewRow, ...], **counts: int):
    review = LastingSyncReview(
        root_id="root-1",
        observation_token=TOKEN,
        rows=rows,
        can_apply=True,
        apply_blocker=LastingSyncApplyBlocker.NONE,
        **counts,
    )
    return replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=review,
    )


async def test_review_row_reads_path_effect_and_where() -> None:
    """AC#1: one row names the file, the effect and the destination folder."""
    row = LastingSyncReviewRow(
        "bind-1",
        "safe",
        "Create a Library note",
        action_id="act-1",
        relative_path="Daily/2026-09-06.md",
        destination="Vault",
    )
    app = _Host(_review_snapshot((row,), safe_count=1))

    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        line = app.query_one("#notes-sync-review-row-0 .notes-sync-review-line", Static)
        assert (
            str(line.renderable)
            == "Daily/2026-09-06.md · Create a Library note · Vault"
        )
        heading = app.query_one(".notes-sync-review-group-heading", Static)
        assert str(heading.renderable) == "Create a Library note (1)"
        assert "Safe item 1" not in _frame(app)


async def test_sixty_creates_collapse_to_one_summary_row_with_a_disclosure() -> None:
    """AC#2: a uniform run collapses to one row, exactly as Import once does.

    The rows arrive in the plan's own order -- by binding id, a digest -- so
    the vault's Archive files are shuffled among its other folders. That is
    what the live review showed: sixty named rows and no run at all.
    """
    other = ("People", "Projects", "Reading")
    rows = tuple(_safe_row(index) for index in range(1, 61)) + tuple(
        _safe_row(100 + index, folder=folder) for index, folder in enumerate(other)
    )
    shuffled = tuple(sorted(rows, key=lambda row: hashlib.sha256(row.item_id.encode()).hexdigest()))
    assert [row.relative_path for row in shuffled] != [row.relative_path for row in rows]
    app = _Host(_review_snapshot(shuffled, safe_count=63))

    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        heading = app.query_one(".notes-sync-review-group-heading", Static)
        assert str(heading.renderable) == "Create a Library note (63)"
        runs = list(app.query(".notes-sync-run").results(Collapsible))
        assert len(runs) == 1
        title = str(runs[0].title)
        assert "Archive" in title
        assert "60 files" in title
        assert "Create a Library note" in title
        assert "Vault" in title
        assert runs[0].collapsed is True
        # Nothing is hidden: every row is one press away.
        assert len(runs[0].query(".library-notes-sync-review-row")) == 60


async def test_a_short_run_stays_as_rows() -> None:
    rows = tuple(_safe_row(index, folder="People") for index in range(1, 4))
    app = _Host(_review_snapshot(rows, safe_count=3))

    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        assert not app.query(".notes-sync-run")
        assert len(app.query(".library-notes-sync-review-row")) == 3


async def test_skipped_rows_group_under_a_skipped_heading_with_the_reason() -> None:
    """AC#3: a skipped file names itself and why, under its own heading."""
    rows = (
        _safe_row(1, folder="People"),
        LastingSyncReviewRow(
            "item-skip-0",
            "skipped",
            "Obsidian trash — skipped",
            relative_path=".trash/Old idea.md",
            reason="obsidian_trash",
        ),
        LastingSyncReviewRow(
            "item-skip-1",
            "skipped",
            "Empty file — nothing to import",
            relative_path="Inbox/Untitled.md",
            reason="empty_file",
        ),
    )
    app = _Host(_review_snapshot(rows, safe_count=1, skip_count=2))

    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        headings = [
            str(item.renderable)
            for item in app.query(".notes-sync-review-group-heading").results(Static)
        ]
        assert headings == ["Create a Library note (1)", "Skipped (2)"]
        lines = [
            str(item.renderable)
            for item in app.query(".notes-sync-review-line").results(Static)
        ]
        assert ".trash/Old idea.md · Obsidian trash — skipped" in lines
        assert "Inbox/Untitled.md · Empty file — nothing to import" in lines
        assert "1 safe · 0 need attention · 2 skipped" in _frame(app)


class _Importer:
    def begin_selection(self) -> None:
        raise AssertionError("setup must not enter import")


async def test_unnamed_rows_never_cost_the_user_the_review() -> None:
    """A name is decoration: a refused label projection must not eat the review.

    Regression pin: naming rows made `binding_labels` part of loading a
    review, and a runtime that refused it (a stale folder, an older port)
    left the review empty -- which also disarmed Activate, because the
    replacement review carried no observation token.
    """

    class _Refusing:
        def snapshot(self) -> NotesSyncRuntimeSnapshot:
            return NotesSyncRuntimeSnapshot("active", "sync_now", ())

        async def review_setup(self, setup: NotesSyncRootSetup) -> ReconciliationPlan:
            return ReconciliationPlan(
                root_id="setup-root",
                observation_token=TOKEN,
                safe_actions=(
                    NotesSyncAction("act-1", NotesSyncActionKind.CREATE_NOTE, "bind-1"),
                ),
                attention=(),
                skips=(),
                managed_placement_effects=(),
                deletion_groups=(),
            )

        async def binding_labels(
            self, root_id: str, binding_ids: tuple[str, ...]
        ) -> tuple[RuntimeBindingLabel, ...]:
            raise ValueError("stale_review")

        async def abandon_setup(self, root_id: str) -> None:
            return None

    controller = LibraryNotesSyncController(
        runtime=_Refusing(), import_controller=_Importer()
    )
    controller.choose_relationship("keep_synced")
    controller.set_setup("display_name", "Vault")
    controller.set_setup("folder", str(Path.home()))
    await controller.check_setup()

    review = controller.snapshot.review
    assert controller.snapshot.phase == "review"
    assert review.observation_token == TOKEN
    assert review.safe_count == 1
    assert review.rows[0].effect == "Create a Library note"
    # Unnamed, never wrongly named.
    assert (review.rows[0].relative_path, review.rows[0].destination) == ("", "")


class _SetupRuntime:
    def __init__(self) -> None:
        self.setups: list[NotesSyncRootSetup] = []

    def snapshot(self) -> NotesSyncRuntimeSnapshot:
        return NotesSyncRuntimeSnapshot("active", "sync_now", ())

    async def review_setup(self, setup: NotesSyncRootSetup) -> ReconciliationPlan:
        self.setups.append(setup)
        return ReconciliationPlan(
            root_id="setup-root",
            observation_token=TOKEN,
            safe_actions=(),
            attention=(),
            skips=(),
            managed_placement_effects=(),
            deletion_groups=(),
        )

    async def binding_labels(
        self, root_id: str, binding_ids: tuple[str, ...]
    ) -> tuple[RuntimeBindingLabel, ...]:
        return ()

    async def abandon_setup(self, root_id: str) -> None:
        return None


async def test_sync_setup_offers_the_obsidian_toggle_on_for_a_vault(
    tmp_path: Path,
) -> None:
    """AC#3: the toggle appears, on, only when the chosen folder is a vault."""
    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    plain = tmp_path / "plain"
    plain.mkdir()

    runtime = _SetupRuntime()
    controller = LibraryNotesSyncController(
        runtime=runtime, import_controller=_Importer()
    )
    assert controller.choose_relationship("keep_synced") == "configure"
    controller.set_setup("display_name", "Vault")
    controller.set_setup("folder", str(vault))
    setup = controller.snapshot.setup
    assert (setup.obsidian_vault, setup.obsidian_mode) == (True, True)

    app = _Host(controller.snapshot)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        toggle = app.query_one("#notes-sync-obsidian", Checkbox)
        assert toggle.value is True
        assert str(toggle.label) == "Obsidian vault"

    controller.set_setup("obsidian_mode", "off")
    await controller.check_setup()
    assert runtime.setups[-1].obsidian_mode is False

    controller.set_setup("obsidian_mode", "on")
    await controller.check_setup()
    assert runtime.setups[-1].obsidian_mode is True

    # A plain folder is never treated as a vault, whatever the toggle says.
    controller.set_setup("folder", str(plain))
    assert controller.snapshot.setup.obsidian_vault is False
    app = _Host(replace(controller.snapshot, phase="configure"))
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        assert not app.query("#notes-sync-obsidian")
    await controller.check_setup()
    assert runtime.setups[-1].obsidian_mode is False


async def test_choosing_a_vault_adds_the_toggle_to_the_already_open_setup_form(
    tmp_path: Path,
) -> None:
    """AC#3 on the live route: the form is open when the folder is picked.

    Regression pin (found live, not by the test above): `sync_state` keeps a
    fast path that patches the configure form's fields in place instead of
    recomposing, so the user's typed name survives a snapshot. The Obsidian
    row only EXISTS for a vault, so patching left it missing however correct
    the state was -- picking the vault in the running app changed the folder
    summary and nothing else.
    """

    vault = tmp_path / "vault"
    (vault / ".obsidian").mkdir(parents=True)
    plain = tmp_path / "plain"
    plain.mkdir()
    controller = LibraryNotesSyncController(
        runtime=_SetupRuntime(), import_controller=_Importer()
    )
    assert controller.choose_relationship("keep_synced") == "configure"
    controller.set_setup("display_name", "Vault")

    app = _Host(controller.snapshot)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        canvas = app.query_one(LibraryNotesAddFromFilesCanvas)
        assert not app.query("#notes-sync-obsidian")

        controller.set_setup("folder", str(vault))
        canvas.sync_state(controller.snapshot)
        await pilot.pause()

        assert app.query_one("#notes-sync-obsidian", Checkbox).value is True
        # The rebuild keeps what the user typed.
        assert app.query_one("#notes-sync-display-name", Input).value == "Vault"

        # And a plain folder takes the row away again.
        controller.set_setup("folder", str(plain))
        canvas.sync_state(controller.snapshot)
        await pilot.pause()
        assert not app.query("#notes-sync-obsidian")


def test_setup_state_defaults_the_toggle_on() -> None:
    setup = LastingSyncSetup()
    assert (setup.obsidian_vault, setup.obsidian_mode) == (False, True)
