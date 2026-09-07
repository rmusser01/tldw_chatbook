"""Pre-extraction characterization pins for the Library Collections subsystem.

Wave-2 Task 5 (``.superpowers/sdd/2026-09-02-library-decomposition-wave2-
cold-trio/task-5-report.md``; recipe: ``backlog/docs/library-decomposition-
recipe.md``; export series precedent: ``Tests/UI/
test_library_export_characterization.py``). Before the Collections
subsystem's state PR moves any ``_library_collections*`` field into
``LibraryCollectionsState``, this file pins the CURRENT behavior of every
Collections ``@on`` handler a per-id ``grep -rn "<id>" Tests/`` (the export
series' Step-1 method, re-run here) reported as never actually
``.press()``-ed/``.value =``-assigned/``Input.Submitted``-through a real
DOM/Pilot interaction -- not merely mentioned for a ``.disabled``/existence
assertion.

Enumeration: an ``ast`` walk of ``LibraryScreen`` for method names
containing "collection" (case-insensitive) found 67 methods (2026-09-02
snapshot, re-derived, not carried over from planning). Three of those --
``handle_library_prompts_collection``, ``_apply_library_prompt_collection``,
``_sync_library_prompt_collection_label`` -- are Prompts-owned (the
"Prompt Collections" feature, unrelated to Library Collections/captures;
confirmed by reading each body, per the recipe's own documented
``startswith``/substring-match trap) and excluded from this file's
enumeration and from the collections cluster entirely. Of the remaining 64,
42 carry a distinct ``@on`` decorator (41 once the 1 Prompts false-positive,
``handle_library_prompts_collection``, is dropped); every one of those 41
was checked with a per-id ``grep -rn`` across ``Tests/`` followed by a
manual read of the surrounding lines for an actual ``.press()``/
``.click()``/direct-value-assignment/``Input.Submitted`` interaction --
NOT a same-line grep alone, which the export series' own report already
flagged as unreliable (a query-then-press pair frequently spans two
lines, and a substring id like ``#library-collections-filters`` matches
inside ``#library-collections-filters-apply`` without being the same
selector). 23 of the 41 are already exercised this way (mostly via
``Tests/UI/test_library_collections_capture_reader.py`` and
``Tests/Live/test_library_collections_capture_walkthrough.py``, a
production-shaped walkthrough); one more
(``retry_library_collection_quick_capture``,
``#library-collections-capture-retry-confirm``) looked unpressed under a
same-line-only grep but is genuinely covered by
``test_unknown_quick_capture_preserves_draft_and_does_not_auto_retry``
once the press on the line AFTER its ``query_one`` call is read. This file
pins the 17 genuine gaps that remain, grouped into 4 tests by shared
setup (mirroring this codebase's own walkthrough-style tests, which
routinely exercise several related handlers inside one continuous Pilot
session rather than one Pilot session per handler):

- ``test_capture_filters_toggle_apply_clear_sort_and_free_text_reach_the_scope_service``
  pins ``toggle_library_collection_capture_filters``
  (``#library-collections-filters``, zero references anywhere despite the
  visually similar ``-apply``/``-clear`` siblings' ids appearing in an
  unrelated existence assertion), ``apply_library_collection_capture_filters``
  (``#library-collections-filters-apply``), ``clear_library_collection_
  capture_filters`` (``#library-collections-filters-clear``),
  ``cycle_library_collection_capture_sort`` (``#library-collections-sort``),
  and ``filter_library_collection_captures`` (``#library-collections-
  filter``, ``Input.Submitted`` -- only ever ``query_one``-d for an
  existence assertion, never actually submitted).
- ``test_item_row_press_and_scope_row_press_reach_the_capture_controller``
  pins ``select_library_collection_capture`` (``.library-collections-
  item-row`` -- every existing test either only counts/reads these rows or
  selects a capture via a direct, unbound-of-the-button
  ``await screen._select_library_collection_capture(identity)`` call that
  bypasses the ``@on`` handler and its ``event.button.capture_identity``
  extraction entirely) and ``select_library_collection_capture_scope``
  (``.library-collections-scope-row`` -- referenced only for a count/label
  assertion in ``test_library_collections_capture_reader.py``, never
  pressed).
- ``test_page_and_detail_retry_buttons_recover_after_a_transient_failure``
  pins ``retry_library_collection_captures`` (``#library-collections-page-
  retry``) and ``retry_library_collection_capture_detail``
  (``#library-collections-reader-retry``) -- both ids appear only in a
  ``query_one`` existence assertion built from a permanently-broken fixture
  (an authority-unavailable page), never in a real fail-once-then-recover
  round trip.
- ``test_detail_action_buttons_favorite_mark_read_open_original_highlight_delete_and_note_unlink``
  pins ``favorite_library_collection_capture`` (``#library-collections-
  favorite``, zero references anywhere), ``mark_library_collection_capture_
  read`` (``#library-collections-mark-read``, zero references anywhere),
  ``open_library_collection_capture_original`` (``#library-collections-
  open-original``, zero references anywhere), ``delete_library_collection_
  capture_highlight`` (``.library-collections-highlight-delete``, zero
  references anywhere -- its sibling ``save`` path IS pressed by
  ``test_real_local_capture_actions_persist_reader_results``, but nothing
  ever presses delete on the highlight that test itself saves), and
  ``unlink_library_collection_capture_note`` (``.library-collections-
  linked-note-unlink``, zero references anywhere -- same asymmetry: the
  ``link`` press is covered, the ``unlink`` press is not).
- ``test_quick_capture_refresh_hard_delete_cancel_and_legacy_recovery_close``
  pins ``refresh_library_collection_quick_capture`` (``#library-
  collections-capture-refresh`` -- only ever ``query_one``-d for a
  ``.disabled`` assertion), ``cancel_library_collection_capture_hard_
  delete`` (``#library-collections-hard-delete-cancel``, zero references
  anywhere -- its ``arm``/``confirm`` siblings ARE pressed by the
  walkthrough, but nothing ever backs out of the confirmation), and
  ``close_library_collection_legacy_recovery`` (``#library-collections-
  legacy-recovery-close``, zero references anywhere -- its ``inspect``
  sibling IS pressed by ``test_library_export_characterization.py``'s own
  legacy-recovery-export pin, but nothing ever closes the panel it opens).

No live bugs were found among the 17 -- every one is a coverage gap, not a
behavior bug (each handler's own current behavior, once actually driven
through the DOM, is exactly what its body says it should be). The
remaining 23 non-``@on`` and already-DOM-covered Collections methods are
reached transitively by these same well-covered flows and are not
individually re-pinned here, mirroring the export series' identical
blanket finding for its own 31 private helpers.

Every test below drives the screen only through DOM queries/presses/value
assignments and public screen attributes -- originally the pre-extraction
``_library_collections_*`` names, which resolved identically through the
state PR's generated property shim across the controller PR, and were
retargeted to ``screen._collections_state.<field>`` by wave-2 task 7
(collections series 3/3, cleanup PR) once that shim was deleted (recipe
§1's "Test retarget" step: assertions unchanged byte-for-byte, only the
receiver path moved). The pinned BEHAVIOR these tests characterize was
unaffected by that retarget.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.Library.collections_capture_models import CaptureSaveRequest
from Tests.UI.test_library_collections_capture_reader import _seed_legacy_records
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


async def _open_collections(screen, pilot) -> None:
    screen.query_one("#library-row-browse-collections", Button).press()
    await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")


@pytest.mark.asyncio
async def test_capture_filters_toggle_apply_clear_sort_and_free_text_reach_the_scope_service() -> None:
    """Characterization (pre-extraction): pins toggle/apply/clear filters,
    sort cycling, and the free-text filter submit -- five handlers with
    zero real presses anywhere in the suite today.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://alpha.example.test/a",
            title="Alpha capture",
            tags=("research",),
            text_content="Alpha body.",
        )
    )
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://beta.example.test/b",
            title="Beta capture",
            tags=("personal",),
            text_content="Beta body.",
        )
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_collections(screen, pilot)
        controller = screen._library_collections_capture_controller
        assert controller is not None

        # toggle_library_collection_capture_filters: opens the disclosure.
        assert screen._collections_state.filters_open is False
        screen.query_one("#library-collections-filters", Button).press()
        await pilot.pause()
        assert screen._collections_state.filters_open is True
        await _wait_for_selector(screen, pilot, "#library-collections-filters-apply")

        # apply_library_collection_capture_filters: a domain filter narrows
        # the requested scope.
        domain_input = await _wait_for_selector(
            screen, pilot, "#library-collections-filter-domain"
        )
        domain_input.value = "alpha.example.test"
        apply_button = await _wait_for_selector(
            screen, pilot, "#library-collections-filters-apply"
        )
        apply_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.requested_scope is not None
                and controller.state.requested_scope.domain == "alpha.example.test"
            ),
            message="Filters-apply never reached the scope service.",
        )
        assert controller.state.requested_scope.page == 1

        # clear_library_collection_capture_filters: the domain filter clears.
        clear_button = await _wait_for_selector(
            screen, pilot, "#library-collections-filters-clear"
        )
        clear_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.requested_scope is not None
                and controller.state.requested_scope.domain is None
            ),
            message="Filters-clear never reached the scope service.",
        )

        # cycle_library_collection_capture_sort: saved_desc -> saved_asc.
        assert controller.state.requested_scope.sort == "saved_desc"
        sort_button = await _wait_for_selector(screen, pilot, "#library-collections-sort")
        sort_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.requested_scope is not None
                and controller.state.requested_scope.sort == "saved_asc"
            ),
            message="Sort cycle never reached the scope service.",
        )

        # filter_library_collection_captures: a submitted free-text query
        # becomes the request's search term.
        filter_input = await _wait_for_selector(screen, pilot, "#library-collections-filter")
        screen.set_focus(filter_input)
        await pilot.pause()
        filter_input.insert_text_at_cursor("Beta")
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.requested_scope is not None
                and controller.state.requested_scope.search == "Beta"
            ),
            message="Free-text filter submit never reached the scope service.",
        )
        assert controller.state.requested_scope.page == 1


@pytest.mark.asyncio
async def test_item_row_press_and_scope_row_press_reach_the_capture_controller() -> None:
    """Characterization (pre-extraction): pins select_library_collection_
    capture (an item row press) and select_library_collection_capture_scope
    (a built-in scope row press) -- both currently only reached via a
    direct, unbound-of-the-button internal call or a plain existence
    assertion elsewhere in the suite.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://gamma.example.test/g",
            title="Gamma capture",
            status="reading",
            text_content="Gamma body.",
        )
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_collections(screen, pilot)
        controller = screen._library_collections_capture_controller
        assert controller is not None

        # select_library_collection_capture: a real item-row press.
        row = await _wait_for_selector(screen, pilot, "#library-collections-row-0")
        expected_identity = row.capture_identity
        row.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.selected_identity == expected_identity
                and controller.state.loaded_detail is not None
            ),
            message="Item-row press never reached the capture controller.",
        )

        # select_library_collection_capture_scope: a built-in scope-row press.
        screen.query_one("#library-collections-scope-reading", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                screen._collections_state.active_scope == "reading"
                and controller.state.requested_scope is not None
                and controller.state.requested_scope.statuses == ("reading",)
            ),
            message="Scope-row press never reached the capture controller.",
        )


@pytest.mark.asyncio
async def test_page_and_detail_retry_buttons_recover_after_a_transient_failure() -> None:
    """Characterization (pre-extraction): pins retry_library_collection_
    captures and retry_library_collection_capture_detail -- both ids are
    only ever asserted to EXIST (against a permanently-broken fixture)
    elsewhere in the suite, never actually pressed through a real
    fail-once-then-recover round trip.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://delta.example.test/d",
            title="Delta capture",
            text_content="Delta body.",
        )
    )

    real_list_page = scope.list_page
    list_page_calls = 0

    async def failing_once_list_page(request):
        nonlocal list_page_calls
        list_page_calls += 1
        if list_page_calls == 1:
            raise RuntimeError("controlled page load failure")
        return await real_list_page(request)

    scope.list_page = failing_once_list_page
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        # Entering the Collections rail row issues the first (failing) page
        # load, landing directly in the page_error/Retry state.
        screen.query_one("#library-row-browse-collections", Button).press()
        controller = screen._library_collections_capture_controller
        assert controller is not None
        retry_button = await _wait_for_selector(
            screen, pilot, "#library-collections-page-retry"
        )
        assert controller.state.page_error == "page_load_failed"

        # retry_library_collection_captures: the retry button recovers.
        retry_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.page_error is None
                and controller.state.page is not None
                and controller.state.page.items
            ),
            message="Page-retry press never recovered from the failure.",
        )
        assert list_page_calls == 2

        row = await _wait_for_selector(screen, pilot, "#library-collections-row-0")
        expected_identity = row.capture_identity

        real_get_detail = scope.get_detail
        detail_calls = 0

        async def failing_once_get_detail(identity):
            nonlocal detail_calls
            detail_calls += 1
            if detail_calls == 1:
                raise RuntimeError("controlled detail load failure")
            return await real_get_detail(identity)

        scope.get_detail = failing_once_get_detail
        row.press()
        detail_retry_button = await _wait_for_selector(
            screen, pilot, "#library-collections-reader-retry"
        )
        await _wait_for_condition(
            pilot,
            lambda: controller.state.detail_error == "detail_load_failed",
            message="Detail load never entered its failed state.",
        )

        # retry_library_collection_capture_detail: the reader-retry button
        # recovers.
        detail_retry_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.detail_error is None
                and controller.state.loaded_detail is not None
                and controller.state.loaded_detail.capture.identity
                == expected_identity
            ),
            message="Reader-retry press never recovered from the failure.",
        )
        assert detail_calls == 2


@pytest.mark.asyncio
async def test_detail_action_buttons_favorite_mark_read_open_original_highlight_delete_and_note_unlink(
    monkeypatch,
) -> None:
    """Characterization (pre-extraction): pins five detail-pane action
    handlers with zero references anywhere in the suite today -- favorite,
    mark-read, open-original, and the delete/unlink halves of the already-
    covered save-highlight/link-note pair.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    outcome = await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://epsilon.example.test/e",
            title="Epsilon capture",
            status="reading",
            favorite=False,
            text_content="Epsilon body.",
        )
    )
    assert outcome.capture is not None
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_collections(screen, pilot)
        controller = screen._library_collections_capture_controller
        assert controller is not None
        row = await _wait_for_selector(screen, pilot, "#library-collections-row-0")
        row.press()
        await _wait_for_condition(
            pilot,
            lambda: controller.state.loaded_detail is not None,
            message="Setup selection never settled.",
        )
        for _ in range(5):
            await pilot.pause()

        # favorite_library_collection_capture.
        assert controller.state.loaded_detail.capture.favorite is False
        favorite_button = await _wait_for_selector(
            screen, pilot, "#library-collections-favorite"
        )
        favorite_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.loaded_detail is not None
                and controller.state.loaded_detail.capture.favorite is True
            ),
            message="Favorite press never reached the capture controller.",
        )
        for _ in range(3):
            await pilot.pause()

        # mark_library_collection_capture_read.
        assert controller.state.loaded_detail.capture.status == "reading"
        mark_read_button = await _wait_for_selector(
            screen, pilot, "#library-collections-mark-read"
        )
        mark_read_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.loaded_detail is not None
                and controller.state.loaded_detail.capture.status == "read"
            ),
            message="Mark-read press never reached the capture controller.",
        )
        for _ in range(3):
            await pilot.pause()

        # open_library_collection_capture_original.
        opened_urls: list[str] = []
        monkeypatch.setattr(
            "webbrowser.open", lambda url: opened_urls.append(url) or True
        )
        open_original_button = await _wait_for_selector(
            screen, pilot, "#library-collections-open-original"
        )
        open_original_button.press()
        await pilot.pause()
        assert opened_urls == [controller.state.loaded_detail.capture.canonical_url]

        # save a highlight (already covered elsewhere) so this test can pin
        # its own DELETE half.
        highlights_mode_button = await _wait_for_selector(
            screen, pilot, "#library-collections-mode-highlights"
        )
        highlights_mode_button.press()
        quote_area = await _wait_for_selector(
            screen, pilot, "#library-collections-highlight-quote"
        )
        quote_area.text = "A quoted highlight"
        highlight_save_button = await _wait_for_selector(
            screen, pilot, "#library-collections-highlight-save"
        )
        highlight_save_button.press()
        await _wait_for_condition(
            pilot,
            lambda: len(screen._collections_state.highlights) == 1,
            message="Highlight setup did not persist.",
        )
        for _ in range(3):
            await pilot.pause()

        # delete_library_collection_capture_highlight.
        delete_button = await _wait_for_selector(
            screen, pilot, ".library-collections-highlight-delete"
        )
        delete_button.press()
        await _wait_for_condition(
            pilot,
            lambda: len(screen._collections_state.highlights) == 0,
            message="Highlight-delete press never reached the scope service.",
        )
        for _ in range(3):
            await pilot.pause()

        # link a note (already covered elsewhere) so this test can pin its
        # own UNLINK half.
        notes_mode_button = await _wait_for_selector(
            screen, pilot, "#library-collections-mode-notes"
        )
        notes_mode_button.press()
        linked_note_id_input = await _wait_for_selector(
            screen, pilot, "#library-collections-linked-note-id"
        )
        linked_note_id_input.value = "note-42"
        linked_note_save_button = await _wait_for_selector(
            screen, pilot, "#library-collections-linked-note-save"
        )
        linked_note_save_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.loaded_detail is not None
                and controller.state.loaded_detail.note_links
            ),
            message="Note-link setup did not persist.",
        )
        for _ in range(3):
            await pilot.pause()

        # unlink_library_collection_capture_note.
        unlink_button = await _wait_for_selector(
            screen, pilot, ".library-collections-linked-note-unlink"
        )
        unlink_button.press()
        await _wait_for_condition(
            pilot,
            lambda: bool(
                controller.state.loaded_detail is not None
                and not controller.state.loaded_detail.note_links
            ),
            message="Note-unlink press never reached the scope service.",
        )


@pytest.mark.asyncio
async def test_quick_capture_refresh_hard_delete_cancel_and_legacy_recovery_close(
    monkeypatch,
) -> None:
    """Characterization (pre-extraction): pins refresh_library_collection_
    quick_capture, cancel_library_collection_capture_hard_delete, and
    close_library_collection_legacy_recovery -- each id is only ever
    asserted for existence/``.disabled`` or has zero references at all
    elsewhere in the suite.
    """
    app = _build_test_app()
    scope = app.collections_capture_scope_service
    authority = scope.active_authority
    assert authority is not None
    outcome = await scope.save_capture(
        CaptureSaveRequest(
            authority.key,
            "https://zeta.example.test/z",
            title="Zeta capture",
            text_content="Zeta body.",
        )
    )
    assert outcome.capture is not None
    _seed_legacy_records(app.local_library_collections_db, count=1)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_collections(screen, pilot)
        controller = screen._library_collections_capture_controller
        assert controller is not None

        # refresh_library_collection_quick_capture: only renders once a save
        # outcome is uncertain (`save_outcome_unknown`); retains the draft
        # and re-settles the page/selected detail rather than auto-retrying.
        async def unknown_save(request):
            from tldw_chatbook.Library.collections_capture_models import (
                CaptureSaveOutcome,
            )

            return CaptureSaveOutcome(None, None, outcome_unknown=True)

        monkeypatch.setattr(scope, "save_capture", unknown_save)
        screen.query_one("#library-collections-quick-capture", Button).press()
        url_input = await _wait_for_selector(screen, pilot, "#library-collections-capture-url")
        url_input.value = "https://zeta.example.test/draft-retained"
        save_button = await _wait_for_selector(screen, pilot, "#library-collections-capture-save")
        save_button.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.save_outcome_unknown is True,
            message="Uncertain-save setup never settled.",
        )
        refresh_button = await _wait_for_selector(
            screen, pilot, "#library-collections-capture-refresh"
        )
        refresh_button.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.action_status
            == "Capture list refreshed. Confirm whether the URL is present before retrying.",
            message="Quick-capture refresh never reached the capture controller.",
        )
        retained_url_input = await _wait_for_selector(
            screen, pilot, "#library-collections-capture-url"
        )
        assert retained_url_input.value == "https://zeta.example.test/draft-retained"
        cancel_button = await _wait_for_selector(
            screen, pilot, "#library-collections-capture-cancel"
        )
        cancel_button.press()
        await pilot.pause()

        # cancel_library_collection_capture_hard_delete: backs out of the
        # confirmation without deleting anything.
        row = await _wait_for_selector(screen, pilot, "#library-collections-row-0")
        row.press()
        await _wait_for_condition(
            pilot,
            lambda: controller.state.loaded_detail is not None,
            message="Setup selection never settled.",
        )
        for _ in range(5):
            await pilot.pause()
        more_button = await _wait_for_selector(screen, pilot, "#library-collections-more")
        more_button.press()
        hard_delete_button = await _wait_for_selector(
            screen, pilot, "#library-collections-hard-delete"
        )
        hard_delete_button.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._collections_state.confirming_hard_delete is True,
            message="Hard-delete arm never opened its confirmation.",
        )
        hard_delete_cancel_button = await _wait_for_selector(
            screen, pilot, "#library-collections-hard-delete-cancel"
        )
        hard_delete_cancel_button.press()
        await pilot.pause()
        assert screen._collections_state.confirming_hard_delete is False
        assert controller.state.loaded_detail is not None
        assert controller.state.loaded_detail.capture.identity == outcome.capture.identity

        # close_library_collection_legacy_recovery: closes what its
        # `inspect` sibling opens (already covered by
        # test_library_export_characterization.py's own legacy-recovery-
        # export pin), clearing the preview lines.
        legacy_button = await _wait_for_selector(
            screen, pilot, "#library-collections-legacy-recovery"
        )
        legacy_button.press()
        await _wait_for_selector(
            screen, pilot, "#library-collections-legacy-recovery-content"
        )
        assert screen._collections_state.legacy_recovery_open is True
        assert screen._collections_state.legacy_recovery_lines
        legacy_close_button = await _wait_for_selector(
            screen, pilot, "#library-collections-legacy-recovery-close"
        )
        legacy_close_button.press()
        await pilot.pause()
        assert screen._collections_state.legacy_recovery_open is False
        assert screen._collections_state.legacy_recovery_lines == ()
