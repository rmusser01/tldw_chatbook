"""Library ▸ Notes critique wave -- editor-keys group.

Tasks 32131, 32132, 32133, 32138, 32139, 32142. See
``backlog/tasks/task-32131*.md`` through ``task-32142*.md`` for the full
acceptance criteria; each test below is named after the task it pins.
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


def _build_notes_host() -> LibraryHarness:
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        notes=[{"title": "Research Note", "id": "note-1"}],
    )
    return LibraryHarness(app)


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
    """AC#3: Tab/Shift+Tab cycle only Cancel<->Delete while the prompt is open."""
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


@pytest.mark.asyncio
async def test_fresh_blank_note_reads_as_a_draft_until_first_save():
    """AC#2: a blank note must not claim 'Saved' before anything is typed."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.press("n")
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        blank.press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        status = screen.query_one("#library-note-status", Static)
        assert str(status.renderable) == "Draft — not saved yet", (
            f"A blank, untouched note claimed {status.renderable!r} before "
            "anything was typed or saved"
        )

        body = screen.query_one("#library-note-body", TextArea)
        body.text = "hello"
        body.post_message(TextArea.Changed(body))
        await pilot.pause()
        await pilot.pause()

        assert str(status.renderable) != "Draft — not saved yet"


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
        await pilot.press("n")
        blank = await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        blank.press()
        await _wait_for_selector(screen, pilot, "#library-note-body")
        await pilot.pause()

        status = screen.query_one("#library-note-status", Static)
        assert str(status.renderable) == "Draft — not saved yet"

        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()

        context_keywords = screen.query_one("#library-note-context-keywords", Input)
        context_keywords.value = "todo"
        context_keywords.post_message(Input.Changed(context_keywords, "todo"))
        await pilot.pause()
        await pilot.pause()

        assert str(status.renderable) != "Draft — not saved yet", (
            "A keyword-only edit through Info left the note reading as an "
            "unsaved draft"
        )


# --- task-32138: ctrl+n and n both work on the landing and inside Notes ----


@pytest.mark.asyncio
async def test_ctrl_n_opens_create_from_the_landing():
    """AC#1: ctrl+n was inert on the landing; it must now open Create."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        assert not screen._library_selected_row_id

        await pilot.press("ctrl+n")
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")


@pytest.mark.asyncio
async def test_bare_n_still_opens_create_from_the_landing():
    """Regression guard: the existing bare-`n` landing accelerator survives."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")


@pytest.mark.asyncio
async def test_bare_n_also_opens_create_from_inside_notes():
    """AC#1: `n` was landing-only; it must now also work inside Notes,
    matching where ctrl+n already fires (the same ``check_action`` gate)."""
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).focus()
        await pilot.pause()

        await pilot.press("n")
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")


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
    """AC#2: Info must not print the same status text twice."""
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
        context_status_static = screen.query_one(
            "#library-note-context-status", Static
        )
        assert str(status_static.renderable) == "Saved"
        assert status_static.display is True
        assert context_status_static.display is False, (
            "Info showed 'Saved' a second time, right above the panel"
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
