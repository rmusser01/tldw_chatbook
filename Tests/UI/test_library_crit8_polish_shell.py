"""Critique-8 polish/shell fixes for the Library landing, rail and Notes.

Covers tasks 32058, 32059, 32061, 32062, 32063, 32064, 32066, 32069, 32071
and 32072 -- the polish-shell group of the critique-8 fix wave.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual import events
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook import config as app_config
from tldw_chatbook.Library.library_content_evidence import LibraryContentEvidence
from tldw_chatbook.Library.library_rail_state import LibraryLifecycle
from tldw_chatbook.UI.Library_Modules.canvas_sync import _sync_library_canvas
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import LibraryLandingCanvas
from tldw_chatbook.Widgets.Library.library_note_work_pane import LibraryNoteWorkPane
from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
    StaticLibraryNotesScopeService,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    StaticLibraryNotesKeywordsService,
    _FakeSkillsScopeService,
    _LibraryEvidenceGates,
    _active_library_screen,
    _build_test_app,
    _new_library_onboarding_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_evidence_round,
    _wait_for_library_shell,
    _wait_for_selector,
)

#: The compact geometry the critique-8 live review ran at (register row 17).
COMPACT_TEST_SIZE = (100, 30)
WIDE_TEST_SIZE = (170, 48)


@pytest.mark.asyncio
async def test_library_landing_canvas_stays_beside_the_rail_at_compact_widths():
    """task-32066: docs and code disagreed; the CODE is the one that is right.

    `library.md` said the landing hides below 120 columns and the rail takes
    over. It does not, and that is deliberate: commit 1a6c293761 ("keep compact
    landing alongside rail", 2026-08-26) took the landing OUT of compact
    single-stage on purpose and pinned the two-pane result at 80 and 100
    columns, with a focus-stability contract attached
    (test_library_returning_landing_geometry_keyboard_and_compact_focus_stability
    keeps a focused Continue button through the resize -- hiding the canvas
    would drop that focus into nothing). The guide was corrected instead.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=COMPACT_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause()

        rail = screen.query_one("#library-rail")
        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)

        assert screen._notes_state.compact is True
        assert rail.display is True and rail.region.width > 0
        assert landing.region.width > 0, (
            "the compact landing keeps its pane beside the rail"
        )


@pytest.mark.asyncio
async def test_library_landing_canvas_paints_at_wide_widths():
    """The same route keeps the landing beside the rail above the breakpoint."""
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause()

        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)

        assert landing.region.width > 0


# --- task-32059: Get started survives a relaunch before the first visit ---


def test_get_started_survives_a_relaunch_before_the_first_library_visit(
    tmp_path, monkeypatch
) -> None:
    """task-32059: complete setup, quit, relaunch -- Get started must survive.

    ``coerce_library_lifecycle`` reads an absent lifecycle as EXPANDED once the
    profile was not created in the current run, so a user who finished first-run
    setup and quit before ever opening Library never saw the documented compact
    Get started rail. The lifecycle is now stamped at profile creation.
    """
    config_path = tmp_path / "relaunch-profile" / "config.toml"
    config_path.parent.mkdir(parents=True)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(
        app_config, "_FIRST_PROFILE_CREATED_THIS_SESSION", False, raising=False
    )
    app_config._CONFIG_CACHE = None
    app_config._CONFIG_CACHE_SOURCE = None
    app_config._SETTINGS_CACHE = None
    app_config._SETTINGS_CACHE_SOURCE = None

    # Run 1: the profile is created. Library is never opened.
    app_config.load_settings(force_reload=True)
    first_run = _build_test_app(preserve_profile_admission=True)
    assert first_run.library_new_profile_admission is True

    # Run 2: same config file, no longer created by this process.
    monkeypatch.setattr(
        app_config, "_FIRST_PROFILE_CREATED_THIS_SESSION", False, raising=False
    )
    app_config._CONFIG_CACHE = None
    app_config._CONFIG_CACHE_SOURCE = None
    app_config._SETTINGS_CACHE = None
    app_config._SETTINGS_CACHE_SOURCE = None
    app_config.load_settings(force_reload=True)
    second_run = _build_test_app()
    assert second_run.library_new_profile_admission is False

    assert (
        config_path.read_text(encoding="utf-8").count('lifecycle = "unknown"') == 1
    ), "profile creation must stamp the lifecycle into [library.rail_state]"
    screen = LibraryScreen(second_run)
    assert screen._library_lifecycle is LibraryLifecycle.UNKNOWN


# --- task-32058: a skill import updates the rail count and the list ------


@pytest.mark.asyncio
async def test_skill_import_updates_the_rail_count_and_the_list_in_place():
    """task-32058: an import left the rail at (2) and the list unchanged.

    The critique-8 review had to leave the Skills row and come back before the
    imported skill appeared. The import coordinator's terminal receipt asks the
    current screen to refresh its sources; the rail badge and the mounted list
    must both settle on the new population without a re-entry.
    """
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}, {"name": "translate"}],
    )
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-skills", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert "Skills (2)" in str(
            screen.query_one("#library-row-browse-skills", Button).label
        )

        service._available.append({"name": "summarize"})
        screen._present_library_skills_import_snapshot(refresh_sources=True)
        await _wait_for_condition(
            pilot,
            lambda: "Skills (3)"
            in str(screen.query_one("#library-row-browse-skills", Button).label),
            message="the rail count did not follow the import",
        )

        assert screen.query("#library-skill-row-summarize"), (
            "the imported skill must appear without re-entering the row"
        )


# --- task-32061: Escape restores the Notes list pane ---------------------


async def _open_first_tree_note(screen, pilot) -> None:
    """Open the first Database note through the folder tree the canvas renders."""
    screen.query_one("#library-row-browse-notes", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-row-0")
    screen.query_one("#library-notes-row-0", Button).press()
    await _wait_for_selector(screen, pilot, "#library-note-title")
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_escape_from_the_editor_restores_the_notes_list_pane():
    """task-32061: opening a note must not leave the list pane collapsed.

    Only Library navigation auto-closes for a wide Notes work session (see
    `Docs/User_Guide/library/notes.md`); the Notes list keeps whatever
    visibility it had before the editor opened, and Escape returns to it.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_tree_note(screen, pilot)
        assert screen.query_one("#library-notes-canvas").region.width > 0

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        notes_list = screen.query_one("#library-notes-canvas")
        assert notes_list.display is True
        assert notes_list.region.width > 0, (
            "Escape from the editor must leave the Notes list visible"
        )


@pytest.mark.asyncio
async def test_escape_does_not_reopen_a_notes_list_the_user_collapsed():
    """Restore means "the visibility it had", not "always open"."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        # Reconciliation with phase C (TASK-32089): task 2 unified the media and
        # notes browse routes onto ONE resident shell (`id_prefix="library-
        # browse"`), so the notes Items grip is now `#library-browse-items-grip`
        # -- it was `#library-notes-items-grip` on this test's own branch.
        screen.query_one("#library-browse-items-grip", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen._notes_state.reader_preferences.items_open is False

        screen.query_one("#library-notes-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert screen._notes_state.reader_preferences.items_open is False


def _new_fresh_profile_app(gates: _LibraryEvidenceGates):
    """A brand-new profile: settled STARTER rail, empty sources, real create seam.

    ``_new_library_onboarding_app`` stubs all six owners down to their evidence
    read, which leaves the browse canvas with no list services at all (the
    shell then paints its "sources are unavailable" callout instead of the
    Notes list). Put the real empty list services back and gate only the
    evidence read, so the STARTER -> GRADUATED transition stays under the
    test's control while the canvas behaves like the real one.
    """
    app = _new_library_onboarding_app(gates, starter=True)
    _seed_conversations(app, [], notes=[])
    for owner, attribute in (
        ("notes", "notes_scope_service"),
        ("media", "media_reading_scope_service"),
        ("conversations", "chat_conversation_scope_service"),
    ):
        setattr(
            getattr(app, attribute),
            "get_library_user_content_evidence",
            gates._async_call(owner),
        )
    return app


def _first_note_gates() -> _LibraryEvidenceGates:
    """Empty on entry, then user content -- the STARTER -> GRADUATED transition."""
    return _LibraryEvidenceGates(
        rounds=2,
        outcomes={
            "notes": [
                LibraryContentEvidence.EMPTY,
                LibraryContentEvidence.HAS_USER_CONTENT,
            ]
        },
    )


async def _open_the_first_note_editor(screen, pilot, gates) -> None:
    """Settle a fresh profile on Starter, then open the blank-note editor."""
    await _wait_for_evidence_round(pilot, gates, 0)
    gates.release_round(0)
    await _wait_for_condition(
        pilot,
        lambda: screen._library_lifecycle is LibraryLifecycle.STARTER,
        message="empty evidence did not settle Starter",
    )
    screen.query_one("#library-hub-action-new-note", Button).press()
    await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
    screen.query_one("#library-notes-create-blank", Button).press()
    await _wait_for_selector(screen, pilot, "#library-note-title")
    await pilot.pause()


async def _type(pilot, text: str) -> None:
    """Send ``text`` one key at a time through the pilot.

    This is NOT a burst, and the docstring that used to claim it was is the
    folklore that cost PR #2571 a whole review round: ``App._press_keys``
    awaits ``wait_for_idle(0)`` twice plus the animator between EVERY key,
    so the event loop fully drains between keystrokes. That is enough for
    task-32062's defect, which needs only a refresh to land between two
    keystrokes -- but never for anything whose trigger is "faster than the
    event loop". For that, see ``_burst`` below.
    """
    await pilot.press(*("space" if character == " " else character for character in text))


@pytest.mark.asyncio
async def test_first_note_on_a_fresh_profile_leaves_the_notes_list_visible():
    """task-32061 in the condition it was reported in: a FRESH profile.

    The earlier reproduction used a populated profile, which never runs the
    STARTER -> GRADUATED transition -- and that transition is what the
    critique saw collapse the list pane behind the first note.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)

            gates.release_all()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_lifecycle is LibraryLifecycle.GRADUATED,
                message="the first note did not graduate the profile",
            )
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            notes_list = screen.query_one("#library-notes-canvas")
            assert notes_list.display is True
            assert notes_list.region.width > 0, (
                "Escape after the first note must leave the Notes list visible"
            )
    finally:
        gates.release_all()


# --- task-32062: typing is never interrupted by the graduation ----------


@pytest.mark.asyncio
async def test_typing_the_first_note_survives_the_graduation_transition():
    """task-32062: 'My first note' + Tab + a body became one corrupted title.

    The first note graduates the profile, and that transition landed inside
    the ~0.4 s the reader was still typing.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)

            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()
            await _type(pilot, "My first note")

            # The graduation lands mid-typing, between the title and the body.
            gates.release_all()
            await _wait_for_condition(
                pilot,
                lambda: screen._library_lifecycle is LibraryLifecycle.GRADUATED,
                message="the first note did not graduate the profile",
            )

            screen.query_one("#library-note-body", TextArea).focus()
            await _type(pilot, "hello from jordan")
            await pilot.pause()

            assert (
                screen.query_one("#library-note-title", Input).value == "My first note"
            )
            assert (
                screen.query_one("#library-note-body", TextArea).text
                == "hello from jordan"
            )
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_notes_refresh_never_rebuilds_the_editor_the_reader_is_typing_in():
    """task-32062 AC#1: no recompose or focus reset while an editor has focus.

    Every Notes refresh (a save, a list reload, the first note landing in the
    list) routes through the canvas sync, which ended in an unconditional
    `refresh(recompose=True)` -- rebuilding the title Input and body TextArea
    under the reader's hands and dropping focus onto the list grip. The next
    keystroke then went somewhere else entirely.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            title = screen.query_one("#library-note-title", Input)
            title.focus()
            await pilot.pause()
            await _type(pilot, "My first note")

            # The real seam every Notes refresh routes through -- driving the
            # work pane's `sync_state` directly skips the follow-up this sync
            # queues, which is where the fix round's own regression hid.
            title_widget = title
            _sync_library_canvas(screen, "notes")
            await pilot.pause()
            await pilot.pause()
            title = screen.query_one("#library-note-title", Input)
            assert title is title_widget

            assert screen.query_one("#library-note-title", Input) is title, (
                "the focused editor must not be rebuilt under the reader"
            )
            assert title.value == "My first note"
            assert title.has_focus, (
                f"focus was reset mid-typing to {screen.focused!r}"
            )

            await pilot.press("tab")
            await _type(pilot, "hello")
            await pilot.pause()

            assert screen.query_one("#library-note-title", Input).value == (
                "My first note"
            )
            assert screen.query_one("#library-note-body", TextArea).text == "hello"
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_notes_sync_while_typing_leaves_focus_where_the_reader_moved_it():
    """Fix-round regression: the skipped refresh must strand no follow-up.

    `_sync_library_canvas(screen, "notes")` queues the Notes focus restore on
    the work pane. When the work pane declines to recompose (the reader is
    typing in it), that callback used to sit there until the NEXT recompose
    and then restore the identity captured before the sync -- yanking focus
    back into the editor the reader had deliberately left.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()
            await _type(pilot, "My first note")

            _sync_library_canvas(screen, "notes")
            await pilot.pause()

            keywords = screen.query_one("#library-note-keywords", Input)
            keywords.focus()
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            assert screen.focused is keywords, (
                f"the sync's stale restore pulled focus back to {screen.focused!r}"
            )
            assert screen.query_one("#library-note-title", Input).value == (
                "My first note"
            )
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_notes_sync_in_an_untouched_new_note_does_not_re_land_on_the_title():
    """The same lateness for `EditorReady`, which re-focuses an untouched title.

    It is posted from the work pane's `_after_recompose`, so a refresh held
    open while the editor has focus would fire it long after the reader chose
    another field.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            assert screen._library_note_session.untouched_create_token is not None
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()

            _sync_library_canvas(screen, "notes")
            await pilot.pause()

            keywords = screen.query_one("#library-note-keywords", Input)
            keywords.focus()
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()

            assert screen.focused is keywords, (
                f"a late EditorReady re-landed focus on {screen.focused!r}"
            )
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_stale_snapshot_never_rewrites_the_field_that_has_focus():
    """task-32062, the defect measured live: the in-place snapshot patch.

    `apply_session_state` rewrote the title Input from the screen's snapshot,
    which during a fast sentence is a keystroke or more behind. Assigning
    `Input.value` clamps the cursor to the shorter text, so the rest of the
    sentence was then typed at that stale position: "My first note" + Tab + a
    body within ~0.4 s stored the title "Mhello from jordan, testing the
    libraryy first note" with an empty body (live, fresh profile, 235x52).
    The focused field is its own authority -- the snapshot is built from its
    Changed events, so it can only be behind, never ahead.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            title = screen.query_one("#library-note-title", Input)
            title.focus()
            await pilot.pause()

            # The window the burst opens: the widget holds text the screen's
            # snapshot has not caught up with yet.
            with title.prevent(Input.Changed):
                title.value = "My first note"
            title.cursor_position = len("My first note")
            work = screen.query_one("#library-note-work-pane", LibraryNoteWorkPane)
            work.apply_session_state(screen._library_note_presentation_state())
            await pilot.pause()

            assert title.value == "My first note", (
                "a stale snapshot must not overwrite what the reader has typed"
            )
            assert title.cursor_position == len("My first note"), (
                "the cursor must not be dragged back into the middle of the text"
            )
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_notes_refresh_from_outside_the_editor_still_recomposes():
    """The skip is scoped to the reader's hands, not permanent.

    Fix round 2: the skipped rebuild is NOT re-applied at blur -- doing that
    stranded the sync's focus-restore callback, which then fired against the
    field the reader had moved to. The next refresh that arrives with focus
    outside the title/body paints the stored state instead.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            title = screen.query_one("#library-note-title", Input)
            title.focus()
            await pilot.pause()
            await _type(pilot, "My first note")

            _sync_library_canvas(screen, "notes")
            await pilot.pause()
            assert screen.query_one("#library-note-title", Input) is title

            screen.query_one("#library-note-back", Button).focus()
            await pilot.pause()
            _sync_library_canvas(screen, "notes")
            await pilot.pause()
            await pilot.pause()

            assert screen.query_one("#library-note-title", Input) is not title, (
                "a refresh with the reader's hands off the field must rebuild"
            )
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_notes_refresh_never_overwrites_the_keywords_being_typed():
    """Review of #2531: the keyword boxes are editable fields too.

    The first pass protected the title and the body only, so a refresh landing
    while the reader typed keywords rebuilt the pane under them and
    `apply_session_state` re-assigned the snapshot's (older) keyword text over
    what they had just typed.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            keywords = screen.query_one("#library-note-keywords", Input)
            keywords.focus()
            await pilot.pause()
            await _type(pilot, "retro")

            _sync_library_canvas(screen, "notes")
            await pilot.pause()
            await pilot.pause()

            assert screen.query_one("#library-note-keywords", Input) is keywords, (
                "the focused keyword box must not be rebuilt under the reader"
            )
            assert keywords.value == "retro"
            assert keywords.has_focus, (
                f"focus was reset mid-typing to {screen.focused!r}"
            )
    finally:
        gates.release_all()


# --- task-32063: one status line, one header, one toast ------------------


@pytest.mark.asyncio
async def test_notes_work_pane_does_not_repeat_the_list_pane_status_line():
    """task-32063: both Notes panes painted the same authority sentence.

    Live at 235x52 the list pane read "Library notes · Ready · Next: Create a
    note or add from files." (then "Library notes · Library database · …",
    before task-32218 dropped the second noun) while the work pane restated
    the same authority in its own header.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_tree_note(screen, pilot)

        list_line = str(
            screen.query_one("#library-notes-authority", Static).renderable
        )
        work_line = str(
            screen.query_one("#library-note-work-authority", Static).renderable
        )

        # task-32218 renamed the authority to the one noun the source strip
        # uses; the claim under test -- only ONE pane paints it -- is unchanged.
        assert list_line.startswith("Library notes")
        assert not work_line.startswith("Library notes"), (
            "the work pane must not restate the list pane's authority sentence"
        )
        assert work_line


@pytest.mark.asyncio
async def test_add_from_files_paints_one_header_not_four():
    """task-32063: the surface stacked a 140-character run-on over three more.

    Live capture `05-add-from-files-four-headers-before.txt`: the work pane's
    authority line, the canvas's own header, its status line and the phase
    purpose all restated the same thing.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        screen.query_one("#library-notes-add-from-files", Button).press()
        await _wait_for_selector(screen, pilot, "#notes-sync-authority")

        header = str(screen.query_one("#notes-sync-authority", Static).renderable)
        assert len(header) < 80, header
        assert "·" not in header, "the header must be one sentence, not a run-on"
        assert not screen.query("#library-note-work-authority"), (
            "a child canvas that paints its own header must not get a second one"
        )


def _graduation_notifications(app) -> list:
    sent: list = []
    app.notify = lambda message, **kwargs: sent.append((message, kwargs))
    return sent


@pytest.mark.asyncio
async def test_graduation_notice_is_silent_on_a_populated_profiles_first_visit():
    """task-32063: the notice fired on any transition into GRADUATED.

    A returning, already-populated profile has no stored lifecycle, settles to
    EXPANDED, and graduates on the first source read -- nothing became
    available, so nothing should announce it. Seen live on the seeded profile.
    """
    app = _build_test_app()
    sent = _graduation_notifications(app)
    screen = LibraryScreen(app)
    assert screen._library_lifecycle is LibraryLifecycle.EXPANDED

    screen._set_library_lifecycle(LibraryLifecycle.GRADUATED)
    screen._apply_graduation_notice(LibraryLifecycle.EXPANDED)

    assert sent == []


@pytest.mark.asyncio
async def test_graduation_notice_fires_when_the_compact_rail_gives_way():
    """The transition it exists for: the Get started rail becomes the full one."""
    app = _build_test_app()
    sent = _graduation_notifications(app)
    app.app_config.setdefault("library", {}).setdefault("rail_state", {})[
        "lifecycle"
    ] = "starter"
    screen = LibraryScreen(app)
    assert screen._library_lifecycle is LibraryLifecycle.STARTER

    screen._set_library_lifecycle(LibraryLifecycle.GRADUATED)
    screen._apply_graduation_notice(LibraryLifecycle.STARTER)

    assert sent == [
        ("Library tools are now available.", {"severity": "information"})
    ]


@pytest.mark.asyncio
async def test_graduation_notice_fires_for_a_new_profile_that_never_saw_starter():
    """Review of #2531: a new profile is stamped UNKNOWN and paints compact.

    Evidence that finds content aggregates straight to GRADUATED without ever
    settling on STARTER, so requiring STARTER silenced the notice for exactly
    the user whose hidden tools had just appeared.
    """
    app = _build_test_app()
    sent = _graduation_notifications(app)
    app.app_config.setdefault("library", {}).setdefault("rail_state", {})[
        "lifecycle"
    ] = "unknown"
    screen = LibraryScreen(app)
    assert screen._library_lifecycle is LibraryLifecycle.UNKNOWN

    screen._set_library_lifecycle(LibraryLifecycle.GRADUATED)
    screen._apply_graduation_notice(LibraryLifecycle.UNKNOWN)

    assert sent == [
        ("Library tools are now available.", {"severity": "information"})
    ]


@pytest.mark.asyncio
async def test_graduation_notice_is_a_toast_and_not_a_second_canvas_line():
    """task-32063 AC: the notice IS a toast -- one event, one surface.

    The first pass kept the in-canvas `#library-lifecycle-status` line beside
    the toast, which is the duplication the critique called out.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    sent = _graduation_notifications(app)
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen._set_library_lifecycle(LibraryLifecycle.STARTER)
        screen._set_library_lifecycle(LibraryLifecycle.GRADUATED)
        screen._apply_graduation_notice(LibraryLifecycle.STARTER)
        screen._sync_library_rail_lifecycle_presentation()
        await pilot.pause()

        assert sent[-1] == (
            "Library tools are now available.",
            {"severity": "information"},
        )
        status = screen.query_one("#library-lifecycle-status", Static)
        assert "Library tools are now available." not in str(status.renderable)
        assert status.display is False
        painted = "\n".join(
            "".join(segment.text for segment in strip)
            for strip in screen._compositor.render_strips()
        )
        assert "Library tools are now available." not in painted


# --- task-32064: the Chunking Lab strip and its exit --------------------


@pytest.mark.asyncio
async def test_chunking_lab_strip_lives_under_details_actions_with_a_gloss():
    """task-32064: the strip was the first interactive row on every canvas.

    A first-time reviewer pressed it and landed in a full-screen A/B tool with
    no explanation. It now sits under Details ▸ Actions and says what it does.
    """
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert not screen.query("#library-chunking-tools"), (
            "the Chunking Lab strip must not sit above the canvas"
        )
        lab = screen.query_one("#library-open-chunking-lab", Button)
        details_body = screen.query_one("#library-rail-section-body-details")
        assert details_body in lab.ancestors
        gloss = str(
            screen.query_one("#library-details-chunking-gloss", Static).renderable
        )
        assert gloss == "Chunking Lab — compare how text is split for search"


def test_chunking_lab_escape_returns_to_the_library_canvas() -> None:
    """task-32064: only the Lab's own Back left the screen; Escape did nothing."""
    from tldw_chatbook.UI.Screens.chunking_lab_screen import ChunkingLabScreen

    keys = {
        binding[0] if isinstance(binding, tuple) else binding.key
        for binding in ChunkingLabScreen.BINDINGS
    }
    assert "escape" in keys


@pytest.mark.asyncio
async def test_chunking_lab_escape_leaves_from_the_focused_sample_editor(
    tmp_path, monkeypatch
) -> None:
    """Review of #2531: the binding existed but vetoed itself where it matters.

    The Lab's sample area is a `TextArea` and is the first thing focused, so
    the guard that kept Escape for focused text controls left the reader stuck
    in the full-screen tool -- the opposite of the documented promise.
    """
    from Tests.UI.test_chunking_lab_screen import settle_lab
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Navigation.screen_registry import resolve_screen_route

    monkeypatch.setattr("tldw_chatbook.config.get_user_data_dir", lambda: tmp_path)
    app = _build_test_app()
    app._initial_screen_pushed = True

    async with app.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = resolve_screen_route("chunking_lab").load_screen_class()(app)
        await app.push_screen(screen)
        await screen.wait_until_ready()
        await settle_lab(app, screen, pilot)

        sample = screen.query_one("#lab-sample-text", TextArea)
        sample.focus()
        await pilot.pause()
        assert sample.has_focus

        routed: list[str] = []
        post_message = app.post_message

        def _record(message):
            if isinstance(message, NavigateToScreen):
                routed.append(message.screen_name)
            return post_message(message)

        monkeypatch.setattr(app, "post_message", _record)

        await pilot.press("escape")
        await settle_lab(app, screen, pilot)

        assert routed == [screen.return_route]


# --- task-32069: rail search clear, and three Study rows ----------------


@pytest.mark.asyncio
async def test_rail_search_box_clears_and_does_not_carry_a_query_across_canvases():
    """task-32069: the box kept the last query with no way to clear it."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        search = screen.query_one("#library-search-input", Input)
        search.value = "retro"
        await pilot.pause()

        clear = screen.query_one("#library-search-clear", Button)
        clear.press()
        await pilot.pause()
        assert screen.query_one("#library-search-input", Input).value == ""

        screen.query_one("#library-search-input", Input).value = "retro"
        screen._rag_search_state.query = "retro"
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")

        assert screen.query_one("#library-search-input", Input).value == "", (
            "a stale query must not follow the reader onto another canvas"
        )
        assert screen._rag_search_state.query == "retro", (
            "clearing the rail box must not discard the Search/RAG query itself"
        )


@pytest.mark.asyncio
async def test_study_section_renders_three_rows_not_six():
    """task-32069: three destinations spent six rail rows on a repeated hint."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        body = screen.query_one("#library-rail-section-body-study")
        rows = [child for child in body.children if isinstance(child, Button)]

        assert len(rows) == 3
        for row in rows:
            assert "see what carries over" not in str(row.label)
            assert row.styles.height.value == 1


# --- task-32072: first-run hand-off to Library ---------------------------


def test_wizard_summary_offers_a_route_into_library_import() -> None:
    """task-32072: the Summary offered "Explore Home" and never said where
    content lives, so a finished setup handed the user nowhere to put a file."""
    import inspect

    from tldw_chatbook.UI.Wizards import FirstRunSetupWizard as wizard_module

    source = inspect.getsource(wizard_module)
    assert '"Add your first document", id="setup-exit-library"' in source
    assert '@on(Button.Pressed, "#setup-exit-library")' in source


@pytest.mark.asyncio
async def test_get_started_steps_are_live_controls_that_unlock_in_sequence():
    """task-32072: "1 Add · 2 Find · 3 Use" named steps with no controls."""
    app = _build_test_app()
    app.app_config.setdefault("library", {}).setdefault("rail_state", {})[
        "lifecycle"
    ] = "starter"
    screen = LibraryScreen(app)
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-hub-step-import")

        assert not screen.query("#library-hub-orientation")
        steps = [
            screen.query_one(f"#library-hub-step-{name}", Button)
            for name in ("import", "find", "use")
        ]
        assert [str(step.label) for step in steps] == [
            "Import a file",
            "Find it",
            "Use it in Console",
        ]
        assert not steps[0].has_class("library-source-action-blocked")
        assert steps[1].has_class("library-source-action-blocked")
        assert steps[2].has_class("library-source-action-blocked")

        hint = str(screen.query_one("#library-hub-steps-hint", Static).renderable)
        assert "Import a file" in hint and hint.endswith(".")


def test_use_it_in_console_unlocks_on_a_selection_not_on_bare_results() -> None:
    """Review of #2531: the step unlocked on results the user had not picked.

    Staging refuses without a selected result, so the seemingly available
    control answered with the staging refusal instead of opening Console.
    """
    from types import MappingProxyType

    from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow

    app = _build_test_app()
    app.app_config.setdefault("library", {}).setdefault("rail_state", {})[
        "lifecycle"
    ] = "starter"
    screen = LibraryScreen(app)
    row = LibraryRagResultRow(
        result_id="r1",
        title="Retro notes",
        snippet="snippet",
        score=0.9,
        source_id="src-1",
        chunk_id="chunk-1",
        citations=(),
        provenance=MappingProxyType({"source_type": "notes"}),
    )
    screen._rag_search_state.results = (row,)
    screen._rag_search_state.selected_result_id = ""

    assert screen._library_landing_canvas_state().search_result_selected is False

    screen._rag_search_state.selected_result_id = "r1"
    assert screen._library_landing_canvas_state().search_result_selected is True


# --- task-32106: the editor-owned skip protects the editor, not the list ---


def _many_notes(count: int = 40) -> list[dict[str, str]]:
    """Enough notes for the Items pane to actually scroll at 100 columns."""
    return [
        {
            "id": f"n-{index}",
            "title": f"Note {index:02d}",
            "content": f"body {index}",
            "last_modified": "2026-09-01T00:00:00Z",
        }
        for index in range(count)
    ]


def _burst(app, *keys: str) -> None:
    """Post keys with NO awaits between them -- a real terminal burst.

    ``pilot.press`` is the opposite of this: ``App._press_keys`` awaits
    ``wait_for_idle(0)`` twice plus the animator between every key, so the
    whole event loop drains between keystrokes (measured on this machine:
    ~200 ms per key). The task itself records that "1 s gaps behave", so a
    drained loop can never reproduce AC#1 -- PR #2571 review, finding 1.
    """
    for key in keys:
        event = events.Key(key, key if len(key) == 1 else None)
        event.set_sender(app)
        app._driver.send_message(event)


@pytest.mark.asyncio
async def test_a_notes_refresh_at_the_tab_boundary_keeps_the_title():
    """task-32062: a refresh landing as focus leaves the title clobbered it.

    The title stops being its own authority the moment Tab moves focus off
    it, so a snapshot one keystroke behind could be written over it -- and
    assigning ``Input.value`` clamps the cursor to the shorter text. The
    per-field ``has_focus`` guards and the recompose skip are what keep the
    two fields apart here. (This is NOT the AC#1 burst: ``pilot.press``
    drains the loop between keys. See the burst test below.)
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()

            await _type(pilot, "My first note")
            _sync_library_canvas(screen, "notes")
            await pilot.press("tab")
            await _type(pilot, "hello from jordan")
            await pilot.pause()

            assert screen.query_one("#library-note-title", Input).value == (
                "My first note"
            )
            assert screen.query_one("#library-note-body", TextArea).text == (
                "hello from jordan"
            )
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_title_tab_body_burst_lands_the_body_in_the_body_field():
    """task-32106 AC#1: the reported gesture, driven as a REAL burst.

    ``Screen.BINDINGS``' ``Binding("tab", "app.focus_next")`` is not
    ``priority=True``, so ``Key(tab)`` is posted to the focused ``Input`` and
    has to bubble a message-queue hop per ancestor up to the Screen -- while
    the App keeps dequeuing the following keys and forwarding each to
    ``self.focused``, still the title. Reproduced in stock Textual 8 with
    nothing from this repo in it (PR #2571 review, finding 1):

        pilot  elapsed=1268.9ms  title='My first note'      body='hello'
        burst  elapsed=   0.2ms  title='My first notehello' body=''

    The editor's own fields take a priority Tab binding so the App resolves
    the focus move before it forwards the next key.
    """
    gates = _first_note_gates()
    app = _new_fresh_profile_app(gates)
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(235, 52)) as pilot:
            screen = _active_library_screen(host)
            await _open_the_first_note_editor(screen, pilot, gates)
            screen.query_one("#library-note-title", Input).focus()
            await pilot.pause()

            _burst(host, *"My first note".replace(" ", "_"), "tab", *"hello")
            await pilot.pause()
            await pilot.pause()

            assert screen.query_one("#library-note-title", Input).value == (
                "My_first_note"
            )
            assert screen.query_one("#library-note-body", TextArea).text == "hello"
    finally:
        gates.release_all()


@pytest.mark.asyncio
async def test_a_body_tab_burst_leaves_the_trailing_word_out_of_the_body():
    """The same defect one widget over -- the note BODY (coordinator addendum).

    A peer session reported typing vanishing after Tab out of a note body.
    Same root cause: bursting ``hello`` + Tab + ``world`` into the body left
    BOTH words in it (measured before the fix:
    ``'helloworldalpha budget line'``) because Tab's focus move landed after
    the burst. The body carries the same priority Tab binding as the fields
    around it.

    What this does NOT fix, and is a separate defect the peer's task keeps:
    Tab from the body lands on a Button, which silently swallows the keys
    that follow -- true at any typing speed, so it is a focus-ORDER problem,
    not this one.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_tree_note(screen, pilot)
        body = screen.query_one("#library-note-body", TextArea)
        # The priority binding is only correct while Tab means "leave the
        # field" here; under "indent" it would steal the key the TextArea
        # needs.
        assert body.tab_behavior == "focus"
        body.focus()
        await pilot.pause()
        before = body.text

        _burst(host, *"hello", "tab", *"world")
        await pilot.pause()
        await pilot.pause()

        after = screen.query_one("#library-note-body", TextArea).text
        assert after == f"hello{before}", after
        assert "world" not in after


@pytest.mark.asyncio
async def test_the_keywords_field_is_its_own_authority_while_focused():
    """task-32106 AC#2: the same rule as the title, one field over.

    Shipped by commit 97626354ee (PR #2531 review) and unpinned until now:
    ``apply_session_state`` used to write ``wide_keywords.value`` from a
    snapshot that could be a keystroke behind, and assigning ``Input.value``
    clamps the cursor to the shorter text -- so the rest of what was being
    typed landed at a stale position.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_tree_note(screen, pilot)
        work = screen.query_one("#library-note-work-pane", LibraryNoteWorkPane)
        keywords = screen.query_one("#library-note-keywords", Input)
        keywords.focus()
        await pilot.pause()
        with keywords.prevent(Input.Changed):
            keywords.value = "retro, half-typed"
        keywords.cursor_position = len("retro, half")

        state = work.presentation_state
        assert state is not None
        stale = replace(
            state,
            snapshot=replace(
                state.snapshot,
                draft=replace(state.snapshot.draft, keywords_text="stale"),
            ),
        )
        work.apply_session_state(stale)
        await pilot.pause()

        applied = screen.query_one("#library-note-keywords", Input)
        assert applied.value == "retro, half-typed"
        # The reported symptom was the cursor clamping to the shorter stale
        # text, so the cursor is the assertion that matters (PR #2571
        # review, finding 7).
        assert applied.cursor_position == len("retro, half")


@pytest.mark.asyncio
async def test_overlapping_syncs_mid_edit_keep_the_readers_place():
    """Two editor-owned syncs in flight must not restore a zero snapshot.

    PR #2571 Qodo finding 4 argued a second sync could capture the
    recomposed list's temporary zero offset before the first deferred
    restore ran, then apply that zero. It does not: the capture happens
    synchronously in ``_sync_library_canvas``, strictly BEFORE
    ``sync_state`` requests the recompose, and ``call_after_refresh``
    applies the offset before the next turn can issue another sync. Probed
    with 0, 1 and 2 pauses between the pair -- all three keep the offset.
    Pinned here so a change to that ordering fails loudly instead.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=COMPACT_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_tree_note(screen, pilot)
        screen.query_one("#library-note-title", Input).focus()
        await pilot.pause()
        screen.query_one("#library-notes-list").scroll_to(
            y=6, animate=False, force=True, immediate=True
        )
        await pilot.pause()
        scrolled = screen.query_one("#library-notes-list").scroll_offset
        assert scrolled.y > 0

        for pauses in (0, 1, 2):
            _sync_library_canvas(screen, "notes")
            for _ in range(pauses):
                await pilot.pause()
            _sync_library_canvas(screen, "notes")
            for _ in range(3):
                await pilot.pause()
            assert screen.query_one("#library-notes-list").scroll_offset == scrolled, (
                f"overlapping syncs with {pauses} pause(s) lost the offset"
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", (COMPACT_TEST_SIZE, WIDE_TEST_SIZE))
async def test_a_sync_mid_edit_repaints_the_list_pane_and_keeps_its_scroll(size):
    """task-32106 AC#3 + AC#4: the skip is scoped to the work pane instance.

    ``editor_has_focus`` asks whether the focused field is inside THIS canvas,
    so the Items pane beside the editor is not covered by it and still
    repaints -- moving the guard up to the screen would freeze the list
    silently (critique #9 D13 reads the other way round: a title the list
    never shows is a list whose DATA has not changed, not a list that stopped
    painting). What the repaint used to cost was the reader's place in the
    list: the offset went back to the top mid-sentence, because the follow-up
    that re-applies it is skipped while the editor owns focus (measured at
    100x30: 6 -> 0). It is re-applied on its own now, without touching focus.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_tree_note(screen, pilot)
        title = screen.query_one("#library-note-title", Input)
        title.focus()
        await pilot.pause()

        listing = screen.query_one("#library-notes-list")
        listing.scroll_to(y=6, animate=False, force=True, immediate=True)
        await pilot.pause()
        scrolled = screen.query_one("#library-notes-list").scroll_offset
        assert scrolled.y > 0, "the Items pane never scrolled; widen the fixture"
        rows_before = list(screen.query(".library-notes-row"))

        _sync_library_canvas(screen, "notes")
        await pilot.pause()
        await pilot.pause()

        rows_after = list(screen.query(".library-notes-row"))
        assert rows_after and rows_after[0] is not rows_before[0], (
            "the Items pane stopped repainting while the editor had focus"
        )
        assert screen.query_one("#library-note-title", Input) is title
        assert title.has_focus, f"focus moved to {screen.focused!r}"
        assert screen.query_one("#library-notes-list").scroll_offset == scrolled
