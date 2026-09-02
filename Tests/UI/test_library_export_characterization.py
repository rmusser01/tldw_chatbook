"""Pre-extraction characterization pins for the Library Export subsystem.

Wave-2 Task 2 (`.superpowers/sdd/2026-09-02-library-decomposition-wave2-
cold-trio/task-2-brief.md`; recipe: `backlog/docs/library-decomposition-
recipe.md`; conversations series precedent: `Tests/UI/
test_library_conversations_characterization.py`). Before the export
subsystem's state PR moves any `_library_export*` field into
`LibraryExportState`, this file pins the CURRENT behavior of every export
``@on`` handler a plain ``grep -rn "#<button-id>" Tests/`` (the brief's
Step-1 enumeration script) reported as either wholly unreferenced or
referenced only for ``.disabled``/existence assertions -- never actually
``.press()``-ed through a real DOM/Pilot interaction.

Enumeration: an ``ast`` walk of ``LibraryScreen`` for method names
containing "export" found 51 methods (2026-09-02 snapshot, re-derived, not
carried over from planning). Of those, 20 distinct ``@on``-bound button/
input selectors were checked with a per-id ``grep -rn "#<id>" Tests/``
followed by a manual look at the surrounding lines for an actual
``.press()``/``.click()``/``insert_text_at_cursor`` call (not merely an id
reference, e.g. a ``.disabled`` assertion or an unbound
``LibraryScreen.<method>(fake, event)`` call, per the recipe §3 lesson on
bypass shapes). 15 of the 20 are already exercised this way; this file
pins the 5 genuine gaps:

- ``handle_library_export_cancel`` (``#library-export-cancel``) -- the
  button is ``query_one``-d only for a ``.display`` check
  (``test_library_export_cancel.py::test_cancel_button_visible_only_while_
  running``); the handler itself is only ever driven by three unbound
  ``LibraryScreen.handle_library_export_cancel(fake, None)`` calls in that
  same file -- never a real press.
- ``handle_library_export_description_changed``
  (``#library-export-description``) -- zero references anywhere in
  ``Tests/`` or ``tldw_chatbook/`` outside its own ``@on`` line and its
  compose site. Its sibling ``handle_library_export_name_changed`` IS
  driven through real typing (``name_input.insert_text_at_cursor(...)`` in
  ``test_library_shell.py``); the description field never is.
- ``handle_library_conversations_export_selected``
  (``#library-conversations-export-selected``) -- the id is referenced
  across ``test_library_selection_updates.py``,
  ``test_library_multiselect_conversations.py``, and
  ``test_library_shell.py``, but every occurrence is a ``.disabled``
  assertion; nothing presses it.
- ``handle_library_media_export_selected``
  (``#library-media-export-selected``) -- same pattern across
  ``test_library_media_side_by_side.py``, ``test_library_multiselect_
  media.py``, ``test_library_honesty_accessibility.py``, and
  ``test_library_shell.py``: only ``.disabled``/existence checks.
- ``choose_library_collection_legacy_recovery_export``
  (``#library-collections-legacy-recovery-export``) -- zero references
  anywhere in ``Tests/``. The write-path it eventually calls,
  ``_export_library_collection_legacy_recovery``, IS pinned by
  ``test_library_collections_capture_reader.py::
  test_legacy_recovery_inspector_and_export_reach_every_page`` -- but only
  via a direct ``await screen._export_library_collection_legacy_recovery(
  destination)`` call that bypasses this handler and its ``FileSave`` push
  entirely.

One id was checked and deliberately SKIPPED, with its rationale recorded
rather than silently omitted:

- ``#library-note-export-txt`` -- never itself pressed, but it shares its
  handler (``handle_library_note_export_text``) with
  ``#library-note-context-export-txt`` via a second ``@on`` decorator on
  the SAME method (no ``event.button.id`` branching inside the body); the
  sibling id IS pressed by
  ``test_library_shell.py::test_library_shell_note_export_pushes_file_
  save_dialog``, so the handler's actual behavior is already exercised.

Every OTHER export ``@on`` handler this enumeration considered (submit,
name-changed, quality/quality-choice, choose-destination, the note
markdown/context exports, the prompt/prompts/media/conversations/notes
bulk-export actions, the RAG canvas's Import-media recovery button) is
already driven through a real ``Button.Pressed``/``Input.Changed`` in one
of ``test_library_export_receipt.py``, ``test_library_shell.py``,
``test_library_choice_strips.py``, ``test_library_prompts_canvas.py``,
``test_library_notes_reader.py``, or
``test_library_multiselect_notes.py`` -- confirmed per-id, not assumed.
The 31 non-``@on`` private export helpers are all reached transitively by
that same well-covered submit/counts/destination/note/prompt pipeline
(mirrors the conversations exemplar's identical blanket finding for its
own 34 private helpers) and are not individually re-pinned here.

Every test below drives the screen only through DOM queries/presses and
public screen attributes (the pre-extraction ``_library_export_*`` names,
which the state PR's generated property shim keeps resolving identically),
per the recipe's byte-for-byte move discipline, so each keeps working
unmodified once the export series' controller PR relocates the method
bodies.
"""

from __future__ import annotations

import threading

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Library.library_export_scope import ExportScope
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_CONVERSATIONS,
    LIBRARY_ROW_BROWSE_MEDIA,
    LIBRARY_ROW_INGEST_EXPORT,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileSave
from Tests.UI.test_library_collections_capture_reader import _seed_legacy_records
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
    _wire_empty_export_prompts_db,
)

_GATE_TIMEOUT_SECONDS = 30.0


class _CancelAwareExportService:
    """Fake ``local_chatbook_service``: blocks until ``gate`` is set, then
    honors ``cancel_check`` -- mirrors ``test_library_shell.py``'s own
    ``_FakeLibraryExportService`` gating convention, extended to actually
    report a cancelled outcome so this file can pin the real
    press-cancel-observe-cancellation round trip without duplicating that
    class (which does not check ``cancel_check`` at all).
    """

    def __init__(self, gate: threading.Event) -> None:
        self._gate = gate
        self.export_calls: list[dict] = []

    async def export_chatbook(self, request_data, *, progress_callback=None, cancel_check=None):
        assert self._gate.wait(timeout=_GATE_TIMEOUT_SECONDS), "export gate never released"
        self.export_calls.append(dict(request_data))
        if cancel_check is not None and cancel_check():
            return {
                "success": False,
                "message": "Cancelled",
                "path": "",
                "dependency_info": {},
                "cancelled": True,
            }
        return {
            "success": True,
            "message": "",
            "path": request_data.get("output_path", ""),
            "dependency_info": {},
        }

    async def create_chatbook(self, **kwargs):
        return {"chatbook_id": 1, **kwargs}


@pytest.mark.asyncio
async def test_cancel_button_press_sets_the_cancel_event_and_settles_cancelled(
    tmp_path,
) -> None:
    """Characterization (pre-extraction): pins handle_library_export_cancel.

    Real press, not the unbound-fake pattern ``test_library_export_
    cancel.py`` uses for its own unit-level pins. Also exercises the
    downstream ``_marshal_library_export_cancelled``/
    ``_apply_library_export_cancelled`` completion path this handler's
    cancel event feeds.
    """
    app = _build_test_app()
    _wire_empty_export_prompts_db(app, "export-char-cancel-prompts")
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-char-cancel-media")
    app.media_db.add_media_with_keywords(title="M1", content="c1", media_type="video")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-char-cancel-ccn")
    app.chachanotes_db.add_conversation({"title": "Conv"})

    gate = threading.Event()
    service = _CancelAwareExportService(gate)
    app.local_chatbook_service = service

    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-destination")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_export_counts is not None,
            message="Export counts never landed.",
        )
        screen.refresh(recompose=True)
        await pilot.pause()

        screen._apply_library_export_destination(tmp_path / "out")
        await pilot.pause()

        submit = screen.query_one("#library-export-submit", Button)
        assert submit.disabled is False
        submit.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_export_running is True
        cancel_event = screen._library_export_cancel_event
        assert cancel_event is not None and not cancel_event.is_set()

        cancel_button = screen.query_one("#library-export-cancel", Button)
        cancel_button.press()
        await pilot.pause()

        assert cancel_event.is_set()
        assert screen._library_export_status == "Cancelling…"
        status_widget = screen.query_one("#library-export-status-line", Static)
        assert str(status_widget.renderable) == "Cancelling…"

        gate.set()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_export_running is False,
            message="Cancelled export never settled.",
        )
        await pilot.pause()

        assert screen._library_export_status == "Export cancelled."
        assert screen._library_export_error == ""
        assert len(service.export_calls) == 1


@pytest.mark.asyncio
async def test_description_input_updates_the_export_form_state() -> None:
    """Characterization (pre-extraction): pins handle_library_export_description_changed.

    ``#library-export-description`` has zero references anywhere in
    ``Tests/``. Its sibling ``#library-export-name`` IS driven through real
    typing elsewhere; this field never is.
    """
    app = _build_test_app()
    _wire_empty_export_prompts_db(app, "export-char-description-prompts")
    _seed_conversations(app, _two_conversations())
    app.media_db = MediaDatabase(":memory:", client_id="export-char-description-media")
    app.chachanotes_db = CharactersRAGDB(":memory:", client_id="export-char-description-ccn")
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{LIBRARY_ROW_INGEST_EXPORT}").press()
        await _wait_for_selector(screen, pilot, "#library-export-description")

        description_input = screen.query_one("#library-export-description", Input)
        screen.set_focus(description_input)
        await pilot.pause()
        assert screen.focused is description_input
        description_input.insert_text_at_cursor("quarterly bundle")
        await pilot.pause()

        assert description_input.value == "quarterly bundle"
        assert screen._library_export_form["description"] == "quarterly bundle"


@pytest.mark.asyncio
async def test_export_selected_conversations_opens_export_canvas_scoped_to_the_selection() -> None:
    """Characterization (pre-extraction): pins handle_library_conversations_export_selected.

    The button's id is referenced in three test files, always for a
    ``.disabled`` assertion; nothing presses it.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")

        screen.query_one("#library-conversations-select-toggle", Button).press()
        await pilot.pause()
        screen.query_one("#library-conversation-row-0", Button).press()
        await pilot.pause()
        assert screen._conversations_state.row_selection.is_selected("chat-1")

        export_selected = screen.query_one(
            "#library-conversations-export-selected", Button
        )
        assert export_selected.disabled is False
        export_selected.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT,
            message="Export-selected press never opened the Export canvas.",
        )
        await pilot.pause()

        assert screen._library_export_scope == ExportScope(
            kind="conversations", ids=("chat-1",)
        )
        assert screen._library_export_origin_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS


@pytest.mark.asyncio
async def test_export_selected_media_opens_export_canvas_scoped_to_the_selection() -> None:
    """Characterization (pre-extraction): pins handle_library_media_export_selected.

    Same pattern as the conversations sibling above: the id is referenced
    across four test files, always for a ``.disabled``/existence check.
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")

        screen.query_one("#library-media-select-toggle", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-row-0")),
            message="Media select controls never settled.",
        )
        # The scope service's default sort (most-recently-modified first)
        # lands "media-2" (last_modified 10:00) at row-0, "media-1" (08:00)
        # at row-1 -- the row's backing id is the scope service's own
        # ``local:media:<numeric backing id>`` scheme, not the raw seed
        # dict's "id" string.
        selected_row = screen.query_one("#library-media-row-0", Button)
        expected_id = selected_row.media_id
        selected_row.press()
        await pilot.pause()
        assert screen._library_media_row_selection.is_selected(expected_id)

        export_selected = screen.query_one("#library-media-export-selected", Button)
        assert export_selected.disabled is False
        export_selected.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_selected_row_id == LIBRARY_ROW_INGEST_EXPORT,
            message="Export-selected press never opened the Export canvas.",
        )
        await pilot.pause()

        assert screen._library_export_scope == ExportScope(
            kind="media", ids=(expected_id,)
        )
        assert screen._library_export_origin_row_id == LIBRARY_ROW_BROWSE_MEDIA


@pytest.mark.asyncio
async def test_legacy_recovery_export_button_pushes_file_save_dialog() -> None:
    """Characterization (pre-extraction): pins choose_library_collection_legacy_recovery_export.

    LIVE GAP recorded rather than fixed (out of this task's scope): a
    repo-wide ``grep -rn "library-collections-legacy-recovery-export"
    Tests/`` turns up zero hits. The write-path this handler eventually
    reaches, ``_export_library_collection_legacy_recovery``, IS pinned by
    ``test_library_collections_capture_reader.py::
    test_legacy_recovery_inspector_and_export_reach_every_page`` -- but
    only via a direct method call that bypasses this handler and the
    ``FileSave`` dialog it pushes entirely.
    """
    app = _build_test_app()
    _seed_legacy_records(app.local_library_collections_db, count=1)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-legacy-recovery")
        screen.query_one("#library-collections-legacy-recovery", Button).press()
        await _wait_for_selector(
            screen, pilot, "#library-collections-legacy-recovery-export"
        )

        screen.query_one(
            "#library-collections-legacy-recovery-export", Button
        ).press()
        for _ in range(150):
            if isinstance(host.screen_stack[-1], FileSave):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Legacy recovery export button never pushed a FileSave dialog."
            )

        dialog = host.screen_stack[-1]
        assert isinstance(dialog, FileSave)
        assert dialog._default_file == "legacy-collections-recovery.json"

        await host.pop_screen()
        await pilot.pause()
