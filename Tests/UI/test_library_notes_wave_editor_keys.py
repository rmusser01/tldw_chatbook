"""Library ▸ Notes critique wave -- editor-keys group.

Tasks 32131, 32132, 32133, 32138, 32139, 32142, and the wave-3 round
32246, 32247, 32252, 32253, 32267, 32268. See ``backlog/tasks/task-<id>*.md``
for the full acceptance criteria; each test below is named after the task
it pins.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
from textual.widgets import Button, Input, Static, TextArea

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Library.library_notes_state import LibraryNoteDeleteReceipt


def _build_notes_host(body: str | None = None) -> LibraryHarness:
    app = _build_test_app()
    note = {"title": "Research Note", "id": "note-1"}
    if body is not None:
        note["content"] = body
    _seed_conversations(app, _two_conversations(), notes=[note])
    return LibraryHarness(app)


#: ~35 KB over 1200 lines -- the size of the note task-32247 was measured on.
_LONG_NOTE_BODY = "\n".join(
    f"line {number:05d} alpha budget line for the long note repro"
    for number in range(1200)
)


async def _open_notes_list(screen, pilot):
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await pilot.pause()


def _first_note_row(screen) -> Button:
    """Return the first note row Button, tree-projection or flat-list."""
    rows = list(screen.query(".library-notes-row"))
    assert rows, "No note row Button found"
    return rows[0]


async def _open_first_note_in_info(screen, pilot):
    """Open Notes, select the first note, and switch to the Info pane."""
    await _open_notes_list(screen, pilot)
    _first_note_row(screen).press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()
    screen.query_one("#library-note-context", Button).press()
    await pilot.pause()


# --- task-32131: "/" focuses the filter without typing into it -------------


@pytest.mark.asyncio
async def test_slash_focuses_the_notes_filter_without_inserting_itself():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        row = _first_note_row(screen)
        row.focus()
        await pilot.pause()

        await pilot.press("/")
        await pilot.pause()

        filter_input = screen.query_one("#library-notes-filter", Input)
        assert screen.focused is filter_input
        assert filter_input.value == "", (
            f"'/' leaked into the filter it focused: {filter_input.value!r}"
        )

        await pilot.press("R", "e", "a", "d", "i", "n", "g")
        await pilot.pause()
        assert filter_input.value == "Reading"


@pytest.mark.asyncio
async def test_slash_types_normally_once_the_notes_filter_is_focused():
    """Fix round 1 Important 4 (controller ruling): "/" only ever acts as
    the focus-accelerator while the filter is NOT focused. Once focused,
    it must be a plain typeable character -- notes filter content can
    legitimately contain "/" (folder-style filters like "Work/Q3"), so
    the rail search box's re-arm-on-second-"/" behaviour (right for that
    box, wrong here) must NOT apply to this one.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        filter_input = screen.query_one("#library-notes-filter", Input)
        filter_input.focus()
        await pilot.pause()
        await pilot.press("W", "o", "r", "k", "/", "Q", "3")
        await pilot.pause()
        assert filter_input.value == "Work/Q3", (
            f"'/' typed inside the already-focused filter did not appear: "
            f"{filter_input.value!r}"
        )


# --- task-32132: delete confirmation stays put, footer follows focus,
# Tab is trapped ---------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_confirmation_renders_in_info_without_changing_mode():
    """AC#1: pressing Delete in Info must not snap the pane back to Edit."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note_in_info(screen, pilot)

        info_region = screen.query_one("#library-note-context-region")
        assert info_region.display, "Info wasn't active before Delete"

        screen.query_one("#library-note-context-delete", Button).press()
        await pilot.pause()

        assert info_region.display, (
            "Delete snapped the pane away from Info back to Edit"
        )
        assert not screen.query_one("#library-note-editor-region").display
        assert screen.query_one("#library-note-delete-confirmation").display
        assert screen.query_one("#library-note-context", Button).has_class(
            "is-active"
        )


@pytest.mark.asyncio
async def test_delete_confirmation_footer_follows_the_focused_button():
    """AC#2: the footer names the FOCUSED button's own Enter action."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note_in_info(screen, pilot)
        screen.query_one("#library-note-context-delete", Button).press()
        await pilot.pause()

        cancel_button = screen.query_one("#library-note-delete-cancel", Button)
        confirm_button = screen.query_one("#library-note-delete-confirm", Button)
        assert screen.focused is cancel_button

        shortcuts = dict(screen._library_notes_footer_shortcuts())
        assert shortcuts.get("enter") == "cancel", (
            f"Footer said Enter {shortcuts.get('enter')!r} while Cancel held "
            "focus, where Enter cancels"
        )

        confirm_button.focus()
        await pilot.pause()
        shortcuts = dict(screen._library_notes_footer_shortcuts())
        assert shortcuts.get("enter") == "delete", (
            f"Footer said Enter {shortcuts.get('enter')!r} while Delete held "
            "focus, where Enter deletes"
        )


@pytest.mark.asyncio
async def test_delete_confirmation_traps_tab_between_cancel_and_delete():
    """AC#3: Tab/Shift+Tab cycle only Cancel<->Delete while the prompt is open.

    task-32106 (PR #2571 re-review, NEW-2): the trap lives in
    ``LibraryScreen.on_key``, and the note editor fields now carry a PRIORITY
    tab binding that ``App._check_bindings`` resolves BEFORE ``on_key``. The
    trap therefore holds only because the prompt disables all four fields --
    Textual blurs a disabled widget and drops it from ``focusable``, so none
    of them can be in the binding chain. Asserted below so a change to
    read-only-instead-of-disabled fails here rather than silently letting Tab
    walk out of the prompt.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note_in_info(screen, pilot)
        screen.query_one("#library-note-context-delete", Button).press()
        await pilot.pause()

        cancel_button = screen.query_one("#library-note-delete-cancel", Button)
        confirm_button = screen.query_one("#library-note-delete-confirm", Button)
        assert screen.focused is cancel_button
        for field in (
            "#library-note-title",
            "#library-note-body",
            "#library-note-keywords",
            "#library-note-context-keywords",
        ):
            assert screen.query_one(field).disabled, (
                f"{field} stays live behind the delete prompt; its priority "
                "tab binding now beats the trap in on_key"
            )

        for _ in range(8):
            await pilot.press("tab")
            await pilot.pause()
            assert screen.focused in (cancel_button, confirm_button), (
                f"Tab escaped the delete prompt onto {screen.focused!r}"
            )
        # An even number of Tabs (8) returns to the starting button.
        assert screen.focused is cancel_button

        await pilot.press("shift+tab")
        await pilot.pause()
        assert screen.focused is confirm_button


@pytest.mark.asyncio
async def test_delete_confirmation_disables_the_other_info_buttons():
    """Fix round 1 Important 5: Info's Danger/Reuse & Export buttons stay
    LIVE behind the confirmation prompt (Info itself stays open per this
    task's own AC#1) unless explicitly disabled. Delete/Copy/Export/Use in
    Console must all be disabled while confirming -- and, concretely, a
    press on Use in Console must not navigate away with the delete
    admission still pending.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note_in_info(screen, pilot)
        screen.query_one("#library-note-context-delete", Button).press()
        await pilot.pause()
        assert screen._notes_state.confirming_delete is True

        for selector in (
            "#library-note-context-delete",
            "#library-note-context-copy",
            "#library-note-context-export-md",
            "#library-note-context-export-txt",
            "#library-note-context-use-in-console",
        ):
            button = screen.query_one(selector, Button)
            assert button.disabled, f"{selector} stayed live behind the prompt"

        use_in_console = screen.query_one(
            "#library-note-context-use-in-console", Button
        )
        use_in_console.press()
        await pilot.pause()
        assert screen._library_selected_row_id != "console", (
            "Use in Console navigated away with a delete admission pending"
        )
        assert screen._notes_state.confirming_delete is True, (
            "The delete confirmation was dismissed by a disabled button's press"
        )
        assert (
            screen._library_note_session.destructive_admission is not None
        ), "The delete admission was dropped by the (no-op) press"


@pytest.mark.asyncio
async def test_delete_confirmation_disables_the_context_back_button():
    """PR #2547 review (Qodo finding 4): Back was left out of the disabled
    selector list above, so it stayed live while every other Info action
    was disabled. Pressing it ran ``handle_library_note_context_back``,
    which clears ``_library_note_context`` without cancelling the pending
    admission -- displacing the confirmation prompt out of Info instead of
    leaving it in place or requiring Cancel/Delete.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note_in_info(screen, pilot)
        screen.query_one("#library-note-context-delete", Button).press()
        await pilot.pause()
        assert screen._notes_state.confirming_delete is True

        back_button = screen.query_one("#library-note-context-back", Button)
        assert back_button.disabled, "Back stayed live behind the prompt"

        back_button.press()
        await pilot.pause()
        assert screen._notes_state.confirming_delete is True, (
            "Back displaced the delete confirmation instead of staying inert"
        )
        info_region = screen.query_one("#library-note-context-region")
        assert info_region.display, (
            "Back's (no-op) press moved the pane away from Info"
        )


# --- task-32133: refused Escape notifies; a blank note reads as a draft ----


@pytest.mark.asyncio
async def test_refused_escape_notifies_why_and_what_to_do():
    """AC#1: a veto (e.g. a whitespace-padded title) must not be silent."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        title_input = screen.query_one("#library-note-title", Input)
        title_input.value = "Research Note "
        title_input.post_message(Input.Changed(title_input, "Research Note "))
        await pilot.pause()

        screen.app_instance.notify = MagicMock()
        await screen.action_library_notes_escape()
        await pilot.pause()

        assert screen._notes_state.view == "editor", (
            "Escape exited the editor despite the validation veto"
        )
        screen.app_instance.notify.assert_called_once_with(
            "Can't leave yet — fix the title or press Discard new note.",
            severity="warning",
        )


@pytest.mark.asyncio
async def test_back_button_notifies_on_a_non_validation_veto_kind():
    """Fix round 1 Important 1/2: the notify lives in the SHARED seam
    (``_exit_library_note_editor_guarded``), reached here through the
    "‹ Back to list" BUTTON, not Escape -- and a non-VALIDATION_VETO kind
    gets its own copy, not the title-specific mandated sentence."""
    from tldw_chatbook.Library.library_notes_session import (
        NoteFlushOutcome,
        NoteFlushOutcomeKind,
    )

    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        async def _fake_flush():
            return NoteFlushOutcome(
                NoteFlushOutcomeKind.CONFLICTED,
                "Conflict — review the choices below.",
            )

        screen._flush_library_note_save = _fake_flush
        screen.app_instance.notify = MagicMock()

        screen.query_one("#library-note-back", Button).press()
        await pilot.pause()

        assert screen._notes_state.view == "editor", (
            "The veto should have kept the editor open"
        )
        screen.app_instance.notify.assert_called_once_with(
            "Can't leave yet — this note changed elsewhere; "
            "choose Overwrite or Reload.",
            severity="warning",
        )


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (
            "VALIDATION_VETO",
            "Can't leave yet — fix the title or press Discard new note.",
        ),
        (
            "FAILED",
            "Can't leave yet — the save failed; press Save to retry or Discard.",
        ),
        (
            "CONFLICTED",
            "Can't leave yet — this note changed elsewhere; "
            "choose Overwrite or Reload.",
        ),
        (
            "BLOCKED",
            "Can't leave yet — another action is already in progress; "
            "wait for it to finish.",
        ),
        (
            "STALE",
            "Can't leave yet — the note changed while saving; try again.",
        ),
    ],
)
def test_exit_veto_message_covers_every_non_permitted_outcome_kind(kind, expected):
    """PR #2547 review (Qodo finding 2): the two UI tests above exercise
    only VALIDATION_VETO and CONFLICTED through the full editor; FAILED,
    BLOCKED, and STALE had no direct assertion, so their copy could drift
    silently. Direct unit coverage for every outcome the shared exit seam
    (``_exit_library_note_editor_guarded``) can actually pass in.
    """
    from tldw_chatbook.Library.library_notes_session import NoteFlushOutcomeKind
    from tldw_chatbook.UI.Screens.library_screen import (
        _library_note_editor_exit_veto_message,
    )

    assert (
        _library_note_editor_exit_veto_message(getattr(NoteFlushOutcomeKind, kind))
        == expected
    )


@pytest.mark.asyncio
async def test_fresh_blank_note_reads_as_a_draft_until_first_save():
    """AC#2: a blank note must not claim 'Saved' before anything is typed."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        # task-32356: `n` creates the blank note itself now -- there is no
        # chooser between the key and the editor.
        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        status = screen.query_one("#library-note-status", Static)
        # task-32358 refines task-32133 AC#2's wording, not its intent: an
        # untouched blank note still must not claim "Saved". The row is
        # committed by now, so the chip says what is actually at stake.
        assert str(status.renderable) == "Empty note — type to keep it", (
            f"A blank, untouched note claimed {status.renderable!r} before "
            "anything was typed or saved"
        )

        body = screen.query_one("#library-note-body", TextArea)
        body.text = "hello"
        body.post_message(TextArea.Changed(body))
        await pilot.pause()
        await pilot.pause()

        assert str(status.renderable) != "Empty note — type to keep it"


@pytest.mark.asyncio
async def test_keyword_only_edit_through_info_clears_the_draft_status():
    """PR #2547 review (Qodo finding 5): the main keywords field's handler
    clears ``_library_note_pending_blank_gc_id`` before autosaving, but the
    sibling Info (Context) properties field's handler did not -- so a user
    who opened Info on a fresh blank note and typed only a keyword there
    could autosave while the status kept claiming nothing was saved.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        # task-32356: `n` creates the blank note itself now -- there is no
        # chooser between the key and the editor.
        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        status = screen.query_one("#library-note-status", Static)
        assert str(status.renderable) == "Empty note — type to keep it"

        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()

        context_keywords = screen.query_one("#library-note-context-keywords", Input)
        context_keywords.value = "todo"
        context_keywords.post_message(Input.Changed(context_keywords, "todo"))
        await pilot.pause()
        await pilot.pause()

        assert str(status.renderable) != "Empty note — type to keep it", (
            "A keyword-only edit through Info left the note reading as an "
            "unsaved draft"
        )


# --- task-32138: ctrl+n and n both work on the landing and inside Notes ----


@pytest.mark.asyncio
async def test_ctrl_n_makes_a_note_from_the_landing():
    """AC#1: ctrl+n was inert on the landing; it must now make a new note.

    task-32356 kept the reach and changed the destination: the key used to
    land on the Create chooser and now creates the blank note it was going
    to choose. What this pin guards -- ctrl+n does the New note thing from
    the landing -- is unchanged.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        assert not screen._library_selected_row_id

        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-note-body")


@pytest.mark.asyncio
async def test_bare_n_still_makes_a_note_from_the_landing():
    """Regression guard: the existing bare-`n` landing accelerator survives."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-note-body")


@pytest.mark.asyncio
async def test_bare_n_also_makes_a_note_from_inside_notes():
    """AC#1: `n` was landing-only; it must now also work inside Notes,
    matching where ctrl+n already fires (the same ``check_action`` gate).

    Both keys share ``library_notes_new``, so task-32356's change reached
    both at once -- which is the point: one key creating while the other
    posed a nine-way question is exactly the drift AC#1 closed.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).focus()
        await pilot.pause()

        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-note-body")


# --- task-32139: one back-cue wording, sized by compact ---------------------


@pytest.mark.asyncio
async def test_back_cue_reads_the_same_in_edit_and_info_at_wide_sizes():
    """AC#1: Edit's "‹ Notes" and Info's "‹ Note" must read identically."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        edit_back = screen.query_one("#library-note-back", Button)
        assert str(edit_back.label) == "‹ Notes"

        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()
        info_back = screen.query_one("#library-note-context-back", Button)
        assert str(info_back.label) == str(edit_back.label), (
            f"Edit said {str(edit_back.label)!r}, Info said "
            f"{str(info_back.label)!r} for the identical Back action"
        )


@pytest.mark.asyncio
async def test_back_cue_reads_back_to_list_when_compact():
    """AC#1/guide: the compact editor must read '‹ Back to list', not
    '‹ Notes' -- task-32139's 60x24 evidence."""
    host = _build_notes_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        assert screen._notes_state.compact, "This size must measure compact"
        edit_back = screen.query_one("#library-note-back", Button)
        assert str(edit_back.label) == "‹ Back to list", edit_back.label


# --- task-32142: Preview title, one Saved, absolute timestamp, receipts ----


@pytest.mark.asyncio
async def test_preview_shows_the_title_above_the_rendered_body():
    """AC#1: Preview must show the note's own title, not just the body."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        screen.query_one("#library-note-preview", Button).press()
        await pilot.pause()

        preview_region = screen.query_one("#library-note-preview-region")
        title = screen.query_one("#library-note-preview-body-title", Static)
        body = screen.query_one("#library-note-preview-body")
        assert str(title.renderable) == "Research Note"
        assert title in preview_region.children
        assert list(preview_region.children).index(
            title
        ) < list(preview_region.children).index(body), (
            "The title must render above the body, inside Preview's own "
            "scrolling region"
        )


def test_preview_title_css_class_is_no_longer_dead():
    """Escalated minor (cheap): #library-note-preview-body-title carries
    `.destination-section` like Info's Properties/Danger headers, but no
    compact-mode selector reached it -- the class rendered no differently
    than plain text there (confirmed: a 1-line title measures height==1
    with or without the rule, so a live geometry assertion can't
    discriminate this one). Pin the source instead: the selector must
    exist, grouped with its Info siblings rather than inventing a new
    rule nothing else in this canvas has."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    css_source = (
        repo_root / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    ).read_text(encoding="utf-8")
    assert (
        "#library-shell-grid.library-notes-compact "
        "#library-note-preview-region .destination-section"
    ) in css_source


@pytest.mark.asyncio
async def test_info_shows_saved_only_once():
    """AC#2: Info must not print the same status text twice.

    task-32177: the duplicate ``#library-note-context-status`` widget this
    test used to find (only hidden, per task-32142) is now removed
    entirely rather than just hidden.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()

        status_static = screen.query_one("#library-note-status", Static)
        assert str(status_static.renderable) == "Saved"
        assert status_static.display is True
        assert not screen.query("#library-note-context-status"), (
            "The dead duplicate status widget should be removed, not just "
            "hidden"
        )


@pytest.mark.asyncio
async def test_info_properties_show_an_absolute_timestamp_beside_the_relative_one():
    """AC#3: the relative age alone ("Created 3m") decays into a guess."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[
            {
                "title": "Research Note",
                "id": "note-1",
                "created_at": "2026-07-01T10:00:00+00:00",
                "last_modified": "2026-07-07T11:57:00+00:00",
            }
        ],
    )
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()

        meta = str(screen.query_one("#library-note-context-meta", Static).renderable)
        assert "Created" in meta
        # An absolute local timestamp ("YYYY-MM-DD HH:MM"), not just a bare
        # relative age -- and never the raw ISO string reaching the user.
        assert re.search(r"Created \d{4}-\d{2}-\d{2} \d{2}:\d{2} · \S+ ago", meta), (
            f"No absolute timestamp beside the relative age: {meta!r}"
        )
        assert "T" not in meta.split("Created ", 1)[1].split(" · ")[0], (
            f"A raw ISO timestamp leaked into the meta line: {meta!r}"
        )


@pytest.mark.asyncio
async def test_delete_receipt_is_dismissed_leaving_the_list_for_add_from_files():
    """AC#4: a stale delete receipt must not survive into another workflow."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen._notes_state.delete_receipt = LibraryNoteDeleteReceipt(
            note_id="ghost-1", title="Deleted note", expected_version=1
        )
        screen.refresh(recompose=True)
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-notes-delete-receipt")

        screen.query_one("#library-notes-add-from-files", Button).press()
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: screen._notes_state.view == "lasting_add",
            message="Add from files never took over the view.",
        )
        assert screen._notes_state.delete_receipt is None, (
            "The delete receipt survived leaving the list for Add from files"
        )


@pytest.mark.asyncio
async def test_delete_receipt_is_dismissed_leaving_the_list_for_folder_files():
    """AC#4, the other named workflow: Folder files."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen._notes_state.delete_receipt = LibraryNoteDeleteReceipt(
            note_id="ghost-1", title="Deleted note", expected_version=1
        )
        screen.refresh(recompose=True)
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-notes-delete-receipt")

        screen.query_one("#library-notes-source-files", Button).press()
        await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: screen._file_notes_active(),
            message="Folder files never took over the view.",
        )
        assert screen._notes_state.delete_receipt is None, (
            "The delete receipt survived leaving the list for Folder files"
        )


# --- wave 3 --------------------------------------------------------------
# task-32246 Tab focus order, 32247 Ctrl+End, 32252 slash, 32253 Shift+Tab,
# 32267 Escape from Info, 32268 the delete prompt's position.


async def _open_first_note(screen, pilot):
    """Open Notes and press the first row into the editor."""
    await _open_notes_list(screen, pilot)
    _first_note_row(screen).press()
    await _wait_for_selector(screen, pilot, "#library-note-body")
    await pilot.pause()


# --- task-32246: Tab out of the body stays in the editor -------------------


@pytest.mark.asyncio
async def test_tab_out_of_the_body_lands_on_an_editor_control():
    """AC#1: Tab from the body must not leave the note editor.

    Live at dev 4a14b3f36f (caps 10/11): the body is the LAST focusable in
    ``#screen-content``, so one Tab wrapped the cycle round to its first --
    ``#library-notes-source-database``, the browse chrome's "Library notes"
    source switch, two panes away above the editor. It is a ``Button``, so
    every character typed next was swallowed, and its focus treatment is
    the same background-and-bold it already wears for ``-selected``, which
    is why the reconciler read the pane as having no focused control at all
    (this task's AC#4 -- ruled out as a separate defect, not reproduced as
    one).
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        body = screen.query_one("#library-note-body", TextArea)
        body.focus()
        await pilot.pause()

        await pilot.press("tab")
        await pilot.pause()

        work_pane = screen.query_one("#library-note-work-pane")
        landed = screen.focused
        assert landed is not None, "Tab out of the body left nothing focused"
        assert work_pane in landed.ancestors_with_self, (
            f"Tab out of the body left the note editor and landed on "
            f"{landed.id!r}"
        )
        assert landed.id == "library-note-back"


@pytest.mark.asyncio
async def test_typing_after_a_tab_out_of_the_body_is_named_by_the_footer():
    """AC#2: the characters must land visibly, or the footer must say where
    focus is. The landing control is a Button, so the footer carries it --
    through ``_library_focus_enter_label``, the same seam the delete prompt
    and the New-note canvas already use.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        body = screen.query_one("#library-note-body", TextArea)
        body.focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        shortcuts = dict(screen._library_notes_footer_shortcuts())
        assert shortcuts.get("enter") == "back to list", (
            f"The footer said nothing about the control Tab landed on: "
            f"{shortcuts!r}"
        )


@pytest.mark.asyncio
async def test_shift_tab_from_the_first_editor_control_returns_to_the_body():
    """The cycle closes both ways: Shift+Tab off the first editor control
    comes back to the body rather than walking into the browse chrome.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        screen.query_one("#library-note-back", Button).focus()
        await pilot.pause()
        await pilot.press("shift+tab")
        await pilot.pause()

        assert getattr(screen.focused, "id", None) == "library-note-body"


# --- task-32247: a document-end key that reaches the end -------------------


def test_every_ctrl_end_encoding_resolves_to_one_key_name():
    """AC#1's "in any encoding": the three sequences the critique tried are
    the SAME Textual key, so one binding covers all of them. Pinned against
    Textual's own table so an upgrade that renames the key fails here.
    """
    from textual._ansi_sequences import ANSI_SEQUENCES_KEYS
    from textual.keys import Keys

    assert ANSI_SEQUENCES_KEYS["\x1b[1;5F"] == (Keys.ControlEnd,)
    assert Keys.ControlEnd.value == "ctrl+end"


@pytest.mark.asyncio
async def test_ctrl_end_moves_the_caret_to_the_end_of_a_long_note():
    """AC#1/#3: on a 1200-line body, Ctrl+End must land at the document end.

    Live at dev 4a14b3f36f (cap 08): `\\x1b[1;5F` then "TAILEDIT" put the
    text at character 0 of 37,099. Cause PROVEN, not the inferred one --
    Textual 8.2.8's ``TextArea`` has no ``ctrl+end`` binding and no
    ``cursor_document_end`` action at all, so nothing upstream was
    swallowing anything: there was never a key to swallow.
    """
    host = _build_notes_host(_LONG_NOTE_BODY)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        body = screen.query_one("#library-note-body", TextArea)
        assert len(body.text) > 30_000, "This pin needs a genuinely long note"
        body.focus()
        body.move_cursor((0, 0))
        await pilot.pause()

        await pilot.press("ctrl+end")
        await pilot.pause()
        assert body.cursor_location == body.document.end, (
            f"Ctrl+End left the caret at {body.cursor_location} of "
            f"{body.document.end}"
        )

        await pilot.press("ctrl+home")
        await pilot.pause()
        assert body.cursor_location == (0, 0), (
            f"Ctrl+Home left the caret at {body.cursor_location}"
        )


@pytest.mark.asyncio
async def test_the_editor_footer_advertises_the_document_end_key():
    """AC#2: the key has to be on the footer beside the other editor keys."""
    host = _build_notes_host(_LONG_NOTE_BODY)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        shortcuts = dict(screen._library_notes_footer_shortcuts())
        assert shortcuts.get("ctrl+end") == "end of note", shortcuts


# --- task-32253: Shift+Tab into the Title must not select it ---------------


@pytest.mark.asyncio
async def test_shift_tab_into_the_title_keeps_the_title():
    """AC#1/#3: Shift+Tab into the Title then one character must leave the
    title intact apart from that character.

    Live at dev 4a14b3f36f (caps 33/34): "Ideas for study decks" became "!"
    on one keypress and the status line read "Saved 15:23" a moment later.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        body = screen.query_one("#library-note-body", TextArea)
        body.focus()
        await pilot.pause()

        await pilot.press("shift+tab")
        await pilot.pause()

        title = screen.query_one("#library-note-title", Input)
        assert screen.focused is title
        assert title.selection.start == title.selection.end, (
            f"Shift+Tab selected the title: {title.selection!r}"
        )
        assert title.cursor_position == len("Research Note"), (
            "The caret should stand at the end of the title, ready to extend it"
        )

        await pilot.press("!")
        await pilot.pause()
        assert title.value == "Research Note!", (
            f"One keystroke rewrote the title to {title.value!r}"
        )


# --- task-32252: "/" from a canvas with nothing focused --------------------


@pytest.mark.asyncio
async def test_slash_from_an_unfocused_notes_canvas_leaves_the_filter_empty():
    """AC#1/#3, reproducing the live STARTING STATE the sibling green test
    never constructs: no control focused anywhere (``set_focus(None)``), and
    an empty filter.

    ``test_slash_focuses_the_notes_filter_without_inserting_itself`` passes
    today because it focuses a note ROW first, so ``LibraryScreen.on_key``
    runs with a non-text focus and stops the key after focusing the filter.
    This test removes focus entirely, which is the state R/caps/61 shows.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)

        screen.set_focus(None)
        await pilot.pause()
        assert screen.focused is None, "This pin needs a genuinely unfocused canvas"

        await pilot.press("/")
        await pilot.pause()

        filter_input = screen.query_one("#library-notes-filter", Input)
        assert screen.focused is filter_input
        assert filter_input.value == "", (
            f"'/' leaked into the filter it focused from an unfocused canvas: "
            f"{filter_input.value!r}"
        )

        # AC#2: the task-32131 ruling holds -- once focused, "/" is a
        # literal character, so folder-style filters stay typeable.
        await pilot.press("W", "o", "r", "k", "/", "Q", "3")
        await pilot.pause()
        assert filter_input.value == "Work/Q3"


# --- task-32267: one Escape leaves Info ------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [LIBRARY_TEST_SIZE, (100, 30)])
async def test_one_escape_from_info_returns_to_the_editor(size):
    """AC#1/#2: a single Escape from the Info pane returns to the editor.

    Regression pin. The reported two-press behaviour did not reproduce at
    dev 4a14b3f36f from any of five entry states (see the task notes); the
    ladder step exists in ``action_library_notes_escape`` and this holds it
    at one press, wide and compact.
    """
    host = _build_notes_host()
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note_in_info(screen, pilot)
        assert screen.query_one("#library-note-context-region").display

        await pilot.press("escape")
        await pilot.pause()

        assert not screen.query_one("#library-note-context-region").display, (
            "Info survived the first Escape"
        )
        assert screen.query_one("#library-note-editor-region").display
        assert screen._notes_state.view == "editor"


@pytest.mark.asyncio
async def test_one_escape_returns_from_info_opened_by_keyboard():
    """AC#1, fix round 1 (review, 32267 gap): the KEYBOARD route in.

    The five entry states the task's notes record all reach Info by mouse
    press or by pressing the button programmatically. Task-32246's Tab trap
    makes Tab-to-Info-then-Enter the primary keyboard route into this pane,
    so that is the route this pins: from the body, Tab to
    ``#library-note-context``, Enter, then ONE Escape back to the editor
    with focus on a control the editor owns.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        screen.query_one("#library-note-body", TextArea).focus()
        await pilot.pause()
        info_button = screen.query_one("#library-note-context", Button)
        for _ in range(len(list(screen.focus_chain))):
            if screen.focused is info_button:
                break
            await pilot.press("tab")
            await pilot.pause()
        else:
            raise AssertionError(
                "Tab never reached the Info button from the body"
            )

        await pilot.press("enter")
        await pilot.pause()
        assert screen.query_one("#library-note-context-region").display, (
            "Enter on the Tab-reached Info button did not open Info"
        )

        await pilot.press("escape")
        await pilot.pause()

        assert not screen.query_one("#library-note-context-region").display, (
            "Info survived the first Escape on the keyboard route in"
        )
        assert screen.query_one("#library-note-editor-region").display
        assert screen._notes_state.view == "editor"
        work_pane = screen.query_one("#library-note-work-pane")
        assert screen.focused is not None
        assert work_pane in screen.focused.ancestors_with_self, (
            f"Escape left focus outside the editor, on {screen.focused!r}"
        )


# --- task-32268: the delete prompt sits with the control that raised it ----


@pytest.mark.asyncio
async def test_the_delete_prompt_renders_inside_info_beside_delete():
    """AC#1/#3: the prompt must render inside the Info border, adjacent to
    the Delete control that raised it.

    Live at dev 4a14b3f36f (cap 22): Info's border closed at row 31, Delete
    sat at row 27 inside it, and the prompt painted at rows 32-34 -- five
    rows below the button and outside the box.
    """
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note_in_info(screen, pilot)

        info = screen.query_one("#library-note-context-region")
        delete_button = screen.query_one("#library-note-context-delete", Button)
        delete_button.press()
        await pilot.pause()
        await pilot.pause()

        prompt = screen.query_one("#library-note-delete-confirmation")
        assert info in prompt.ancestors, (
            f"The prompt renders outside the Info border, under "
            f"{prompt.parent!r}"
        )
        children = list(info.children)
        assert children.index(prompt) == children.index(delete_button) + 1, (
            "The prompt must be the next thing after Delete, not further down"
        )
        assert info.region.contains_region(prompt.region), (
            f"The prompt at {prompt.region} escapes the Info box at "
            f"{info.region}"
        )
        assert prompt.region.y >= delete_button.region.y, (
            "The prompt must follow the control that raised it"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reach", "expected_enter"),
    [("tab", "back to list"), ("save", "save note")],
)
async def test_the_editor_exit_chip_survives_at_sixty_columns(reach, expected_enter):
    """task-32246 fix round 1 (review F1): the `enter …` chip must not evict
    the exit key at the narrow stage.

    Round 0 PREPENDED the chip. The real ``AppFooterStatus`` keeps only the
    leading chip at this width (``test_only_one_context_chip_paints_at_sixty_
    columns`` pins that budget), so after the very Tab task-32246 fixes the
    footer painted "enter back to list" ALONE -- no exit advertised at all,
    and on Save neither an exit nor a way back. Measured on the registered
    tier through the real footer widget, because the registered tuple is
    blind to the elision that decides this.
    """
    from Tests.UI.test_library_crit10_layout import _painted_footer

    host = _build_notes_host()
    async with host.run_test(size=(60, 20)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        if reach == "tab":
            screen.query_one("#library-note-body", TextArea).focus()
            await pilot.pause()
            await pilot.press("tab")
            await pilot.pause()
            assert getattr(screen.focused, "id", None) == "library-note-back"
        else:
            screen.query_one("#library-note-save", Button).focus()
            await pilot.pause()

        tier = screen._library_notes_footer_shortcuts()
        assert dict(tier).get("enter") == expected_enter, tier

        shown = await _painted_footer(tier, (60, 24))
        assert "esc notes" in shown, (
            f"the exit key was evicted from the narrow-stage footer by the "
            f"enter chip: registered {tier!r} painted {shown!r}"
        )
