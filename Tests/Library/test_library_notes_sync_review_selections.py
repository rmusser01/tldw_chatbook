"""Staged keep-synced choices survive leaving the review (TASK-34000.40 N-34).

The 2026-10-02 review staged "Keep both" for ``conflict.md · Both file and
note changed``, left the review and came back: the row read "No choice
selected." and nothing said the choice had gone. Every exit cleared the
staged choices -- ``abandon_setup`` (Escape, ‹ Notes), ``return_to_roots``
(the review's back control), a re-check, another root's projection and a
remount all ran ``_invalidate_review_authority``, which emptied
``_selections``. The choices are keyed by the review's observation token, so
a changed folder (a new token) already never matched them.

AC#6: staged choices survive leaving and re-entering the review while the
folder is unchanged; the back control names where it returns to; no stale
"Choice staged" line remains. Kept choices are staged intent only -- nothing
is written until the review's own Apply (CLAUDE.md, Notes Sync: no automatic
winner), so every re-entry arm also asserts that ``apply_reviewed`` never ran.

Three layers: the controller against the recording runtime port (fast, every
leave route), the controller against the REAL runtime over a real bound root
(the observation token of an unchanged folder is stable across checks --
without that, nothing here would ever match in production), and the mounted
Library screen (the painted back cue, choice and status line).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.Library_Modules.test_library_notes_sync_controller import (
    TOKEN,
    TOKEN_2,
    _conflict_plan,
    _ImportController,
    _Runtime,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_notes_files_sync_journey import (
    _close_real_conflict_stack,
    _JourneyHarness,
    _painted_widget_text,
    _seed_real_conflict_authority,
    _start_real_conflict_stack,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _seed_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Actor_Packs.persona_coordinator import (
    PersonaActorPackCoordinator,
    PersonaActorPackRecoveryResult,
)
from tldw_chatbook.UI.Library_Modules.library_notes_sync_controller import (
    LibraryNotesSyncController,
)
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: The status line a single staged choice answers with (unchanged copy).
STAGED_ONE = "Choice staged. No changes yet."
#: The selected-choice line a staged "Keep both" row paints.
SELECTED_KEEP_BOTH = "Selected: Keep both"
TOKEN_3 = "d" * 64
TOKEN_4 = "e" * 64


@pytest.fixture(autouse=True)
def _isolate_actor_pack_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unrelated startup recovery off the mounted journey's database."""

    monkeypatch.setattr(
        "tldw_chatbook.app.first_profile_created_this_session", lambda: False
    )
    monkeypatch.setattr(
        PersonaActorPackCoordinator,
        "recover",
        lambda _self: PersonaActorPackRecoveryResult(0, 0, 0, ()),
    )


def _controller(runtime: _Runtime) -> LibraryNotesSyncController:
    return LibraryNotesSyncController(
        runtime=runtime, import_controller=_ImportController()
    )


def _applied(runtime: _Runtime) -> list[tuple[object, ...]]:
    return [call for call in runtime.calls if call[0] == "apply_reviewed"]


async def _leave_by(route: str, controller: LibraryNotesSyncController) -> None:
    """Leave the review the way a user can, without touching the folder."""

    if route == "escape_or_notes":
        # Escape and ‹ Notes both end in ``_exit_library_notes_lasting_sync``.
        await controller.abandon_setup()
    elif route == "sync_folders":
        # The review's back control when it was opened from Manage sync folders.
        controller.return_to_roots()
    elif route == "add_from_files":
        await controller.abandon_setup()
        assert controller.begin_add_from_files() is True
    elif route == "remount":
        # The Library screen unmounting (another destination, then back).
        controller.invalidate_for_remount()
    else:  # pragma: no cover - a typo in the parametrization
        raise AssertionError(route)


@pytest.mark.parametrize(
    "route", ("escape_or_notes", "sync_folders", "add_from_files", "remount")
)
async def test_a_staged_choice_survives_leaving_and_rechecking_the_unchanged_folder(
    route: str,
) -> None:
    """AC#6: leave by any route, re-check the same folder, the choice is there."""

    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = _controller(runtime)
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")
    assert controller.snapshot.status_line == STAGED_ONE

    await _leave_by(route, controller)
    await controller.check_root("root-1")

    row = controller.snapshot.review.rows[0]
    assert controller.snapshot.review.observation_token == TOKEN
    assert row.selected_label == SELECTED_KEEP_BOTH, (
        f"after leaving by {route}, the re-entered review shows "
        f"{row.selected_label or 'No choice selected.'!r}"
    )
    # The status line says the kept choice is staged, not that nothing is.
    assert controller.snapshot.status_line == STAGED_ONE
    # Staged intent only: the re-entry wrote nothing.
    assert _applied(runtime) == []


async def test_a_changed_folder_drops_the_staged_choice_and_its_line() -> None:
    """A new observation (the folder changed) never inherits an old choice."""

    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = _controller(runtime)
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")
    await controller.abandon_setup()

    runtime.check_plan = _conflict_plan(token=TOKEN_2)
    await controller.check_root("root-1")

    assert controller.snapshot.review.rows[0].selected_label == ""
    assert "staged" not in controller.snapshot.status_line.casefold()
    assert controller.snapshot.status_line == (
        "Review the mutation-free check before applying changes."
    )
    assert _applied(runtime) == []


async def test_two_staged_choices_are_counted_and_kept_across_a_leave() -> None:
    """The line counts what is staged, and re-derives it on re-entry."""

    runtime = _Runtime()
    runtime.check_plan = _conflict_plan(bindings=("bind-1", "bind-2"))
    controller = _controller(runtime)
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")
    assert controller.snapshot.status_line == STAGED_ONE
    controller.stage_attention_choice("root-1", TOKEN, "bind-2", "Keep file")
    assert controller.snapshot.status_line == "2 choices staged. No changes yet."

    controller.return_to_roots()
    await controller.check_root("root-1")

    assert [row.selected_label for row in controller.snapshot.review.rows] == [
        SELECTED_KEEP_BOTH,
        "Selected: Keep file",
    ]
    assert controller.snapshot.status_line == "2 choices staged. No changes yet."
    assert _applied(runtime) == []


async def test_another_roots_review_keeps_the_choice_and_a_root_control_forgets_only_its_own() -> (
    None
):
    """Opening another root is navigation; pausing a root voids its own intent.

    The observation token's payload carries the root id
    (``notes_sync_reconciler._observation_token``), so one root's choices can
    never match another root's review. A root control (pause, resume,
    retarget, disconnect, recover, activate, undo) changes that root, so its
    staged choices are forgotten -- and nobody else's.
    """

    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = _controller(runtime)
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")

    runtime.check_plan = _conflict_plan(root_id="root-2", token=TOKEN_2)
    await controller.check_root("root-2")
    assert controller.snapshot.review.rows[0].selected_label == ""
    controller.stage_attention_choice("root-2", TOKEN_2, "bind-1", "Keep note")
    await controller.pause_root("root-2")

    runtime.check_plan = _conflict_plan()
    await controller.check_root("root-1")
    assert controller.snapshot.review.rows[0].selected_label == SELECTED_KEEP_BOTH

    runtime.check_plan = _conflict_plan(root_id="root-2", token=TOKEN_2)
    await controller.check_root("root-2")
    assert controller.snapshot.review.rows[0].selected_label == ""
    assert "staged" not in controller.snapshot.status_line.casefold()
    assert _applied(runtime) == []


async def test_kept_choices_are_bounded_to_the_three_latest_observations() -> None:
    """Memory hygiene: a fourth observation drops the least recently staged."""

    runtime = _Runtime()
    controller = _controller(runtime)
    for token in (TOKEN, TOKEN_2, TOKEN_3, TOKEN_4):
        runtime.check_plan = _conflict_plan(token=token)
        await controller.check_root("root-1")
        controller.stage_attention_choice("root-1", token, "bind-1", "Keep both")

    kept = {token for token, _binding in controller._selections}  # noqa: SLF001
    assert kept == {TOKEN_2, TOKEN_3, TOKEN_4}
    for token, expected in (
        (TOKEN, ""),
        (TOKEN_2, SELECTED_KEEP_BOTH),
        (TOKEN_4, SELECTED_KEEP_BOTH),
    ):
        runtime.check_plan = _conflict_plan(token=token)
        await controller.check_root("root-1")
        assert controller.snapshot.review.rows[0].selected_label == expected, token
    assert _applied(runtime) == []


async def test_a_real_unchanged_folder_rechecks_to_the_same_review_and_keeps_the_choice(
    tmp_path: Path,
) -> None:
    """Production's token is stable for an unchanged folder, so the kept
    choice is found again; an edit on disk makes a new token and drops it.

    The real runtime owner over a real bound root holding one two-sided
    conflict (``note-1``: "note side" in Notes, "file side" on disk).
    """

    notes_path, state_path, sync_root = _seed_real_conflict_authority(tmp_path)
    owner, database, interop, controller = await _start_real_conflict_stack(
        notes_path, state_path
    )
    try:
        await controller.sync_now("root-1")
        assert controller.snapshot.phase == "review", controller.snapshot.status_line
        first = controller.snapshot.review
        controller.stage_attention_choice(
            "root-1", first.observation_token, "binding-1", "Keep both"
        )
        assert controller.snapshot.status_line == STAGED_ONE

        controller.return_to_roots()
        await controller.sync_now("root-1")

        again = controller.snapshot.review
        assert controller.snapshot.phase == "review"
        assert again.observation_token == first.observation_token
        row = next(row for row in again.rows if row.item_id == "binding-1")
        assert row.selected_label == SELECTED_KEEP_BOTH
        assert controller.snapshot.status_line == STAGED_ONE
        # Nothing was written by staging, leaving or re-checking.
        note = database.get_note_by_id("note-1")
        assert note is not None and note["content"] == "note side"
        assert (sync_root / "note.md").read_text(encoding="utf-8") == "file side"
        assert database.count_notes() == 1

        (sync_root / "note.md").write_text("file side, edited", encoding="utf-8")
        controller.return_to_roots()
        await controller.sync_now("root-1")

        changed = controller.snapshot.review
        assert changed.observation_token != first.observation_token
        row = next(row for row in changed.rows if row.item_id == "binding-1")
        assert row.selected_label == ""
        assert "staged" not in controller.snapshot.status_line.casefold()
    finally:
        await _close_real_conflict_stack(owner, database, interop)


async def _press(screen, pilot, selector: str) -> None:
    button = await _wait_for_selector(screen, pilot, selector)
    assert isinstance(button, Button) and not button.disabled, selector
    button.press()


async def _check_changes_from_manage_sync_folders(screen, pilot) -> None:
    await _press(screen, pilot, "#notes-sync-root-check-0")
    await _wait_for_condition(
        pilot,
        lambda: screen._notes_state.view == "lasting_add"
        and screen._library_notes_sync_controller.snapshot.phase == "review"
        and bool(screen.query("#notes-sync-conflict-selected-0")),
        message=lambda: (
            "Check changes never reached its review: "
            f"view={screen._notes_state.view!r} "
            f"phase={screen._library_notes_sync_controller.snapshot.phase!r} "
            f"status={screen._library_notes_sync_controller.snapshot.status_line!r}"
        ),
    )


def _painted_of(host, screen, selector: str) -> str:
    found = screen.query(selector)
    return _painted_widget_text(host, found.first()) if found else ""


def _label(host, widget) -> str:
    """A button's painted label, without the focus frame (``┃ … ┃``)."""
    return _painted_widget_text(host, widget).strip("┃▊▎▌▐│ ")


def _footer(host, screen) -> str:
    found = screen.query(AppFooterStatus)
    return _painted_widget_text(host, found.first()) if found else ""


async def _wait_for_painted_back(host, screen, pilot, label: str) -> None:
    """Wait for the back control's PAINTED words (review R0 M-5): re-queried
    inside the wait, never one compositor read after a DOM-only wait."""

    def painted() -> str:
        found = screen.query("#notes-sync-back")
        return _label(host, found.first()) if found else ""

    await _wait_for_condition(
        pilot,
        lambda: painted() == label,
        message=lambda: f"the back control painted {painted()!r}, not {label!r}",
    )


async def test_the_review_names_its_way_back_and_paints_the_kept_choice(
    tmp_path: Path,
) -> None:
    """The mounted journey: Manage sync folders ▸ Check changes ▸ Keep both.

    The back control of a review opened from Manage sync folders returns
    there (``handle_library_notes_lasting_back``), so it says so:
    "‹ Sync folders", not "‹ Notes". Pressing it and checking again paints
    the kept choice and the status line. Fix round 1 (review R0 I-1):
    Escape goes where that control goes -- the footer says
    ``esc back to sync folders`` and the key lands on the folder list -- and
    the choice is still kept. Nothing reaches the note or the file.
    """

    notes_path, state_path, sync_root = _seed_real_conflict_authority(tmp_path)
    owner, database, interop, _controller_unused = await _start_real_conflict_stack(
        notes_path, state_path
    )
    try:
        app = _build_test_app()
        _seed_conversations(app, [], notes=_two_notes())
        app.notes_sync_runtime_owner = owner
        host = _JourneyHarness(app)
        async with host.run_test(size=(160, 45)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _press(screen, pilot, "#library-row-browse-notes")
            await _press(screen, pilot, "#library-notes-manage-sync-folders")
            await _check_changes_from_manage_sync_folders(screen, pilot)

            await _wait_for_painted_back(host, screen, pilot, "‹ Sync folders")
            await _press(screen, pilot, "#notes-sync-conflict-0-keep-both")
            await _wait_for_condition(
                pilot,
                lambda: _painted_of(host, screen, "#notes-sync-status") == STAGED_ONE,
                message=lambda: (
                    f"status painted {_painted_of(host, screen, '#notes-sync-status')!r}"
                ),
            )

            await _press(screen, pilot, "#notes-sync-back")
            await _wait_for_condition(
                pilot,
                lambda: screen._notes_state.view == "lasting_roots"
                and getattr(screen.focused, "id", None) == "notes-sync-roots-back",
                message=lambda: (
                    "‹ Sync folders did not return to Manage sync folders: "
                    f"view={screen._notes_state.view!r} focus={screen.focused!r}"
                ),
            )
            await _check_changes_from_manage_sync_folders(screen, pilot)
            await _wait_for_condition(
                pilot,
                lambda: _painted_of(host, screen, "#notes-sync-conflict-selected-0")
                == SELECTED_KEEP_BOTH,
                message=lambda: (
                    "the re-entered review paints "
                    f"{_painted_of(host, screen, '#notes-sync-conflict-selected-0')!r}"
                ),
            )
            assert _painted_of(host, screen, "#notes-sync-status") == STAGED_ONE

            # I-1: the footer names Escape's destination before the key ...
            await _wait_for_condition(
                pilot,
                lambda: "esc back to sync folders" in _footer(host, screen),
                message=lambda: f"the review's footer painted {_footer(host, screen)!r}",
            )
            await pilot.press("escape")
            # ... and the key goes where ‹ Sync folders goes: the folder list.
            await _wait_for_condition(
                pilot,
                lambda: screen._notes_state.view == "lasting_roots"
                and _painted_of(host, screen, "#notes-sync-roots-status")
                == "Sync folders refreshed.",
                message=lambda: (
                    "Escape did not return to Manage sync folders: "
                    f"view={screen._notes_state.view!r} "
                    f"roots status={_painted_of(host, screen, '#notes-sync-roots-status')!r}"
                ),
            )
            # The next key has somewhere to go: the folder list's own back
            # control, exactly where the ‹ Sync folders press leaves it.
            await _wait_for_condition(
                pilot,
                lambda: getattr(screen.focused, "id", None) == "notes-sync-roots-back",
                message=lambda: f"after Escape focus is on {screen.focused!r}",
            )
            await _check_changes_from_manage_sync_folders(screen, pilot)
            await _wait_for_condition(
                pilot,
                lambda: _painted_of(host, screen, "#notes-sync-conflict-selected-0")
                == SELECTED_KEEP_BOTH,
                message=lambda: (
                    "after Escape the review paints "
                    f"{_painted_of(host, screen, '#notes-sync-conflict-selected-0')!r}"
                ),
            )

        note = database.get_note_by_id("note-1")
        assert note is not None and note["content"] == "note side"
        assert (sync_root / "note.md").read_text(encoding="utf-8") == "file side"
    finally:
        await _close_real_conflict_stack(owner, database, interop)


async def test_a_review_started_from_add_from_files_says_notes_and_escape_returns_there(
    tmp_path: Path,
) -> None:
    """Negative control for I-1: a review the user reached from Notes (Add
    from files ▸ Keep a folder synced ▸ Check changes) offers "‹ Notes", its
    footer says ``esc back to notes``, and Escape returns to the Notes list
    (``lasting_origin == "setup"``, never "‹ Sync folders")."""

    notes_path, state_path, _sync_root = _seed_real_conflict_authority(
        tmp_path, diverge=False
    )
    owner, database, interop, _controller_unused = await _start_real_conflict_stack(
        notes_path, state_path
    )
    folder = tmp_path / "new-sync-folder"
    folder.mkdir()
    (folder / "plain.md").write_text("plain", encoding="utf-8")
    try:
        app = _build_test_app()
        _seed_conversations(app, [], notes=_two_notes())
        app.notes_sync_runtime_owner = owner
        host = _JourneyHarness(app)
        async with host.run_test(size=(160, 45)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _press(screen, pilot, "#library-row-browse-notes")
            await _press(screen, pilot, "#library-notes-add-from-files")
            await _press(screen, pilot, "#notes-add-keep-synced")
            await _wait_for_painted_back(host, screen, pilot, "‹ Notes")
            assert screen._notes_state.lasting_origin == "setup"
            controller = screen._library_notes_sync_controller
            controller.set_setup("display_name", "Plain folder")
            controller.set_setup("folder", str(folder))
            await _press(screen, pilot, "#notes-sync-check")
            await _wait_for_condition(
                pilot,
                lambda: controller.snapshot.phase == "review"
                and controller.snapshot.review.source == "setup",
                message=lambda: (
                    "the setup check never reached its review: "
                    f"phase={controller.snapshot.phase!r} "
                    f"status={controller.snapshot.status_line!r}"
                ),
            )
            await _wait_for_painted_back(host, screen, pilot, "‹ Notes")
            await _wait_for_condition(
                pilot,
                lambda: "esc back to notes" in _footer(host, screen),
                message=lambda: f"the review's footer painted {_footer(host, screen)!r}",
            )
            assert "sync folders" not in _footer(host, screen)
            await pilot.press("escape")
            await _wait_for_condition(
                pilot,
                lambda: screen._notes_state.view == "list",
                message=lambda: (
                    f"Escape left view={screen._notes_state.view!r}, not the Notes list"
                ),
            )
    finally:
        await _close_real_conflict_stack(owner, database, interop)


# --- Fix round 1 (review R0) -----------------------------------------------


def _receipt_snapshot(**changes: object):
    from dataclasses import replace

    from tldw_chatbook.Library.library_notes_lasting_sync_state import (
        initial_lasting_sync_snapshot,
    )

    return replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="receipt",
        status_line="1 applied · no conflicts remain.",
        receipt_line="1 applied · listed under Receipts",
        **changes,
    )


@pytest.mark.parametrize(
    ("back_label", "manage_shown"),
    (("‹ Sync folders", False), ("‹ Notes", True)),
)
async def test_a_receipt_names_the_folder_list_once(
    back_label: str, manage_shown: bool
) -> None:
    """M-1, one name for one place: a receipt reached from Manage sync
    folders already has "‹ Sync folders", so it does not also offer
    "Manage sync folders" (both opened the folder list). A receipt reached
    from Notes keeps "Manage sync folders" as the way onward."""

    from Tests.Widgets.Library.test_library_notes_add_from_files_canvas import (
        _Host,
        _frame,
    )

    app = _Host(_receipt_snapshot(back_label=back_label))
    async with app.run_test(size=(120, 36)) as pilot:
        await pilot.pause()
        assert bool(app.query("#notes-sync-manage-roots")) is manage_shown, _frame(app)
        assert app.query_one("#notes-sync-back", Button).label.plain == back_label


@pytest.mark.parametrize("action", ("retarget", "recover"))
async def test_an_inert_root_action_keeps_the_staged_choice(action: str) -> None:
    """M-2: Retarget (and Recover without an open entry) answer "unavailable
    in this release; no files or notes changed." -- nothing changed, so the
    staged choice is kept for the same observation token (T10-b: navigation
    only)."""

    runtime = _Runtime()
    runtime.check_plan = _conflict_plan()
    controller = _controller(runtime)
    await controller.check_root("root-1")
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")

    controller.stage_root_action("root-1", action)
    assert controller.snapshot.status_line == (
        f"{action.title()} is unavailable in this release; no files or notes changed."
    )
    await controller.check_root("root-1")

    assert controller.snapshot.review.rows[0].selected_label == SELECTED_KEEP_BOTH
    assert controller.snapshot.status_line == STAGED_ONE
    assert _applied(runtime) == []


async def test_a_migration_review_keeps_the_hold_reason_ahead_of_the_staged_line() -> (
    None
):
    """M-3: on a migration review whose Activate is held, the status line
    names the hold first and the staged choices after it -- on staging, on a
    re-entry that finds them, and on a refused Activate. One derivation."""

    from tldw_chatbook.Library.library_notes_lasting_sync_state import (
        migration_activation_blocked_line,
    )
    from tldw_chatbook.Notes.notes_sync_reconciler import (
        ReconciliationAttention,
        ReconciliationAttentionKind,
        ReconciliationItemSkip,
        ReconciliationPlan,
    )
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootRuntimeSnapshot,
        NotesSyncRuntimeSnapshot,
    )

    runtime = _Runtime()
    runtime.snapshot = lambda: NotesSyncRuntimeSnapshot(
        "active",
        "sync_now",
        (NotesSyncRootRuntimeSnapshot("root-1", "paused", "review_migration"),),
    )
    runtime.check_plan = ReconciliationPlan(
        root_id="root-1",
        observation_token=TOKEN,
        safe_actions=(),
        attention=(
            ReconciliationAttention(
                ReconciliationAttentionKind.CONFLICT, "both_sides_changed", "bind-1"
            ),
        ),
        skips=(),
        managed_placement_effects=(),
        deletion_groups=(),
        item_skips=(ReconciliationItemSkip("b.md", "mixed_newlines", bound=True),),
    )
    controller = _controller(runtime)
    hold = migration_activation_blocked_line(
        ("b.md · Mixed line endings — save it with one style",)
    )

    await controller.check_migration("root-1")
    assert controller.snapshot.status_line == hold
    controller.stage_attention_choice("root-1", TOKEN, "bind-1", "Keep both")
    assert controller.snapshot.status_line == f"{hold} {STAGED_ONE}"

    controller.return_to_roots()
    await controller.check_migration("root-1")
    assert controller.snapshot.review.rows[0].selected_label == SELECTED_KEEP_BOTH
    assert controller.snapshot.status_line == f"{hold} {STAGED_ONE}"

    assert await controller.activate_root("root-1", TOKEN) is False
    assert controller.snapshot.status_line == f"{hold} {STAGED_ONE}"
    assert not any(call[0] == "activate_root" for call in runtime.calls)
    assert _applied(runtime) == []
