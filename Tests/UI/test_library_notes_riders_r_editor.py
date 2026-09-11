"""Library ▸ Notes critique wave -- r-editor riders group.

Tasks 32177, 32179. See ``backlog/tasks/task-32177*.md`` and
``task-32179*.md`` for the full acceptance criteria.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
from textual.widgets import Button

from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
    _FailingLibraryNoteDetailService,
)
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_notes_wave_editor_keys import (
    _build_notes_host,
    _first_note_row,
    _open_notes_list,
)


# --- task-32177 AC#1: one back-cue wording rule everywhere ------------------


@pytest.mark.asyncio
async def test_new_note_view_back_cue_reads_back_to_list_when_compact():
    """The New-note view's own Back button was left out of task-32139's
    unification and still hard-coded '‹ Notes' at every width."""
    host = _build_notes_host()
    async with host.run_test(size=(60, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).focus()
        await pilot.pause()

        # task-32356: `n` creates a note now, so the New-note VIEW this pin
        # is about is reached by the rail row that still opens it.
        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        back = screen.query_one("#library-notes-create-back", Button)
        assert str(back.label) == "‹ Back to list", back.label


@pytest.mark.asyncio
async def test_new_note_view_back_cue_reads_notes_when_wide():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).focus()
        await pilot.pause()

        # task-32356: `n` creates a note now, so the New-note VIEW this pin
        # is about is reached by the rail row that still opens it.
        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        back = screen.query_one("#library-notes-create-back", Button)
        assert str(back.label) == "‹ Notes", back.label


@pytest.mark.asyncio
async def test_load_retry_back_cue_reads_back_to_list_when_compact():
    """The note-loading/retry view's own Back button was also left out of
    task-32139's unification."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notes_scope_service = _FailingLibraryNoteDetailService(_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(60, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-load-retry")

        back = screen.query_one("#library-note-back", Button)
        assert str(back.label) == "‹ Back to list", back.label


@pytest.mark.asyncio
async def test_load_retry_back_cue_reads_notes_when_wide():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notes_scope_service = _FailingLibraryNoteDetailService(_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-load-retry")

        back = screen.query_one("#library-note-back", Button)
        assert str(back.label) == "‹ Notes", back.label


# --- task-32177 AC#3: public timestamp helper -------------------------------


def test_parse_browser_timestamp_is_exposed_publicly():
    """``library_notes_state`` imported a leading-underscore (module-private)
    name from another package. Assert the public spelling exists on the
    owning module and that the cross-package importer no longer reaches for
    the private one."""
    from tldw_chatbook.Workspaces import conversation_browser_state
    from tldw_chatbook.Library import library_notes_state

    assert hasattr(conversation_browser_state, "parse_browser_timestamp"), (
        "conversation_browser_state must expose a public parse_browser_timestamp"
    )

    source = inspect.getsource(library_notes_state)
    assert "_parse_browser_timestamp" not in source, (
        "library_notes_state still imports the private "
        "_parse_browser_timestamp name"
    )
    assert "parse_browser_timestamp" in source


# --- task-32179: duplicate screen method -------------------------------------


def test_seed_local_source_snapshot_from_cache_is_defined_exactly_once():
    """Python method resolution silently kept only the SECOND of two
    same-named definitions in the class body, leaving the first dead. Count
    top-level defs by name via AST (independent of runtime shadowing) and
    confirm the surviving definition is the real cache-applying
    implementation (accepts ``now``), not the old no-arg stub."""
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    source_path = Path(inspect.getsourcefile(LibraryScreen))
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "LibraryScreen"
    )
    definitions = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_seed_local_source_snapshot_from_cache"
    ]
    assert len(definitions) == 1, (
        f"Expected exactly one _seed_local_source_snapshot_from_cache "
        f"definition, found {len(definitions)}"
    )

    sig = inspect.signature(LibraryScreen._seed_local_source_snapshot_from_cache)
    assert "now" in sig.parameters, (
        "The surviving definition must be the real cache-applying "
        "implementation (accepts `now`), not the dead no-arg stub"
    )


# --- task-32177 AC#1, PR #2555 review: the wording must survive a resize ----
#
# Both views pick their Back wording at compose time. Crossing the compact
# breakpoint on an already-open view re-runs ``apply_compact_presentation``
# instead of recomposing, so a label that is only chosen in ``compose`` keeps
# the previous width's wording -- the initial-viewport tests above cannot see
# that. These cross the breakpoint in both directions.


@pytest.mark.asyncio
async def test_new_note_view_back_cue_follows_the_compact_breakpoint():
    host = _build_notes_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).focus()
        await pilot.pause()

        # task-32356: `n` creates a note now, so the New-note VIEW this pin
        # is about is reached by the rail row that still opens it.
        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        assert (
            str(screen.query_one("#library-notes-create-back", Button).label)
            == "‹ Notes"
        )

        await pilot.resize_terminal(60, 24)
        await pilot.pause()
        await pilot.pause()
        assert (
            str(screen.query_one("#library-notes-create-back", Button).label)
            == "‹ Back to list"
        ), "New note kept the wide wording after shrinking"

        await pilot.resize_terminal(*LIBRARY_TEST_SIZE)
        await pilot.pause()
        await pilot.pause()
        assert (
            str(screen.query_one("#library-notes-create-back", Button).label)
            == "‹ Notes"
        ), "New note kept the compact wording after widening"


@pytest.mark.asyncio
async def test_load_retry_back_cue_follows_the_compact_breakpoint():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notes_scope_service = _FailingLibraryNoteDetailService(_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_notes_list(screen, pilot)
        _first_note_row(screen).press()
        await _wait_for_selector(screen, pilot, "#library-note-load-retry")
        assert str(screen.query_one("#library-note-back", Button).label) == "‹ Notes"

        await pilot.resize_terminal(60, 24)
        await pilot.pause()
        await pilot.pause()
        assert (
            str(screen.query_one("#library-note-back", Button).label)
            == "‹ Back to list"
        ), "Load-retry kept the wide wording after shrinking"

        await pilot.resize_terminal(*LIBRARY_TEST_SIZE)
        await pilot.pause()
        await pilot.pause()
        assert (
            str(screen.query_one("#library-note-back", Button).label) == "‹ Notes"
        ), "Load-retry kept the compact wording after widening"
