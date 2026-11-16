"""TASK-37360: native filter edits survive browse publication boundaries."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import replace

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_media_editor_dirty_escape import (
    SIZE,
    _host,
    _open_row_1,
    _painted,
)
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def _settled(screen, pilot, query: str) -> None:
    browse = screen._library_media_browse_controller
    await _wait_for_condition(
        pilot,
        lambda: browse.applied_scope is not None
        and browse.applied_scope.query == query
        and browse.requested_scope == browse.applied_scope
        and not browse.loading,
        message=lambda: f"exact query {query!r} did not settle: {browse.applied_scope!r}",
    )
    # Let the accepted publication's own widget pump finish its recompose.
    for _ in range(4):
        await pilot.pause()


async def test_paced_native_typing_keeps_prefix_focus_and_caret():
    """Each letter follows a settled publication, exercising the old prefill selection."""
    host, _app, _service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        await _settled(screen, pilot, "")
        await pilot.click("#library-media-filter")
        field = screen.query_one("#library-media-filter", Input)
        canvas = screen.query_one("#library-media-canvas")
        original_state = canvas.canvas
        published_before_changed = []

        def publish_before_changed(value: str) -> None:
            if value == "P" and not published_before_changed:
                published_before_changed.append(value)
                # A native value write is synchronous; Changed is still queued.
                canvas.sync_state(original_state, pager=canvas.pager)

        canvas.watch(field, "value", publish_before_changed, init=False)
        query = ""
        for char in "Product":
            await pilot.press(char)
            query += char
            await _settled(screen, pilot, query)
            current = screen.query_one("#library-media-filter", Input)
            assert current.value == query
            assert current is field
            assert screen.focused is field
            assert field.cursor_position == len(query)
            assert field.selection.start == field.selection.end == len(query)
        assert published_before_changed == ["P"]

        await pilot.press("left", "shift+left", "shift+left")
        selection = field.selection
        screen._sync_library_media_browse_state(None)
        for _ in range(4):
            await pilot.pause()
        assert screen.query_one("#library-media-filter", Input) is field
        assert screen.focused is field
        assert field.selection == selection
        assert field.value == "Product"
        # Multiple publications in one turn must reconcile the latest scope
        # without duplicate Clear ids or losing a re-added recovery action.
        for clearable in (False, True, False, True):
            canvas.sync_state(
                replace(canvas.canvas, scope_clearable=clearable), pager=canvas.pager
            )
        for _ in range(4):
            await pilot.pause()
        assert len(screen.query("#library-media-scope-clear")) == 1
        assert screen.query_one("#library-media-filter", Input) is field
        assert field.selection == selection
        # A Reader patch can adopt the newer display model before the browse
        # sync publishes its scope chrome. The old parent model is not proof
        # that the retained scope row already rendered that model.
        latest = replace(
            canvas.canvas,
            scope_line="Media · 1 of 3 · filter “Product” · all types · sort: Newest",
        )
        canvas.apply_reader_state(latest)
        canvas.sync_state(latest, pager=canvas.pager)
        for _ in range(4):
            await pilot.pause()
        assert (
            screen.query_one("#library-media-scope-line").content == latest.scope_line
        )
        assert screen.query_one("#library-media-filter", Input) is field
        assert field.selection == selection


@pytest.mark.parametrize("size", [SIZE, (80, 24)], ids=["wide", "compact"])
async def test_zero_results_backspace_and_clear_keep_the_filter_control(size):
    host, _app, _service = _host()
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        await _settled(screen, pilot, "")
        if size == (80, 24):
            # Opening the Reader gives it the compact stage. Reopen Items
            # through its native grip before clicking the visible filter.
            await pilot.click("#library-browse-items-grip")
            await _wait_for_condition(
                pilot,
                lambda: screen.query_one("#library-media-filter").region.width > 0,
                message="compact Items filter did not become visible",
            )
        await pilot.click("#library-media-filter")
        field = screen.query_one("#library-media-filter", Input)
        assert screen.focused is field
        await pilot.press("z")
        await _settled(screen, pilot, "z")
        assert not screen._media_state.reader_session.loaded_id
        if size == SIZE:
            empty = screen.query_one("#library-media-reader-empty")
            assert empty.region.width > 0 and empty.region.height > 0
            assert "Select a media item to read it here." in _painted(
                host, empty.region
            )
        assert screen.query_one("#library-media-filter", Input) is field
        assert screen.focused is field
        await pilot.press("backspace")
        await _settled(screen, pilot, "")
        assert field.value == ""
        await pilot.press("P")
        await _settled(screen, pilot, "P")
        screen.query_one("#library-media-filter-clear", Button).press()
        await _settled(screen, pilot, "")
        assert screen.query_one("#library-media-filter", Input) is field
        assert field.value == ""
        assert screen.query_one("#library-media-filter-clear", Button).disabled


async def test_older_completion_keeps_newer_native_edit_and_selection(monkeypatch):
    """Held executor work cannot replace a newer draft or its editing context."""
    host, _app, service = _host()
    search = service.search_media
    entered = {query: threading.Event() for query in ("P", "Pr")}
    release = {query: threading.Event() for query in entered}
    finished = {query: threading.Event() for query in entered}
    wait_finished = {query: threading.Event() for query in entered}

    def wait_for_release(query):
        try:
            return release[query].wait(30)
        finally:
            wait_finished[query].set()

    async def gated_search(**kwargs):
        query = kwargs.get("query")
        if query not in entered:
            return await search(**kwargs)
        entered[query].set()
        try:
            assert await asyncio.to_thread(wait_for_release, query), "gate timed out"
            return await search(**kwargs)
        finally:
            finished[query].set()

    monkeypatch.setattr(service, "search_media", gated_search)
    async with host.run_test(size=SIZE) as pilot:
        try:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_row_1(screen, pilot)
            await _settled(screen, pilot, "")
            await pilot.click("#library-media-filter")
            field = screen.query_one("#library-media-filter", Input)
            await pilot.press("P")
            await _wait_for_condition(
                pilot, entered["P"].is_set, message="older query never entered"
            )
            await pilot.press("r")
            await _wait_for_condition(
                pilot, entered["Pr"].is_set, message="newer query never entered"
            )
            await pilot.press("shift+left")
            selection = field.selection
            scope = screen._library_media_browse_controller.applied_scope
            scope_copy = str(screen.query_one("#library-media-scope-line").render())
            release["P"].set()
            await _wait_for_condition(
                pilot, finished["P"].is_set, message="older query did not finish"
            )
            for _ in range(4):
                await pilot.pause()
            assert screen._library_media_browse_controller.applied_scope == scope
            assert (
                str(screen.query_one("#library-media-scope-line").render())
                == scope_copy
            )
            assert screen.query_one("#library-media-filter", Input) is field
            assert field.value == "Pr"
            assert field.selection == selection
            assert screen.focused is field
            release["Pr"].set()
            await _settled(screen, pilot, "Pr")
            assert field.value == "Pr"
            assert field.selection == selection
            assert screen.focused is field
        finally:
            for gate in release.values():
                gate.set()
            for query, gate in entered.items():
                if gate.is_set():
                    await _wait_for_condition(
                        pilot,
                        wait_finished[query].is_set,
                        message=f"query {query!r} wait thread did not drain inside App",
                    )
                    await _wait_for_condition(
                        pilot,
                        finished[query].is_set,
                        message=f"query {query!r} did not drain inside App lifetime",
                    )


async def test_accepted_older_completion_keeps_newer_native_draft(monkeypatch):
    """Accept real worker results before the newer native edit's debounce fires."""
    host, _app, _service = _host()
    async with host.run_test(size=SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_row_1(screen, pilot)
        await _settled(screen, pilot, "")
        browse = screen._library_media_browse_controller
        apply_result = browse._apply
        pending = []
        accepted = []
        next_draft = None

        def hold_completion(result, **kwargs):
            if result.scope.query == "P":
                pending.append((result, kwargs))
                return False
            return apply_result(result, **kwargs)

        monkeypatch.setattr(browse, "_apply", hold_completion)
        await pilot.click("#library-media-filter")
        field = screen.query_one("#library-media-filter", Input)
        canvas = screen.query_one("#library-media-canvas")

        def accept_before_newer_debounce(value: str) -> None:
            if pending and value == next_draft:
                result, kwargs = pending.pop()
                assert apply_result(result, **kwargs), "older result was discarded"
                assert browse.applied_scope.query == "P"
                accepted.append((value, field.value, browse.applied_scope.query))

        canvas.watch(field, "value", accept_before_newer_debounce, init=False)
        try:
            for key, draft in (("r", "Pr"), ("backspace", ""), ("space", "P ")):
                await pilot.press("P")
                await _wait_for_condition(
                    pilot, lambda: bool(pending), message="P completion was not held"
                )
                next_draft = draft
                await pilot.press(key)
                assert accepted[-1] == (draft, draft, "P")
                assert screen.query_one("#library-media-filter", Input) is field
                assert screen.focused is field
                assert field.value == draft
                assert field.cursor_position == len(draft)
                await _settled(screen, pilot, draft.strip())
                assert field.value == draft
                if draft:
                    screen.query_one("#library-media-filter-clear", Button).press()
                    await _settled(screen, pilot, "")
        finally:
            for result, kwargs in pending:
                apply_result(result, **kwargs)
            pending.clear()
