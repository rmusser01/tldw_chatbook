"""Continuous Library Media traversal and authoritative filtering contracts."""

from __future__ import annotations

import asyncio
import threading
from types import MethodType, SimpleNamespace

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_library_media_side_by_side import (
    WIDE_SIZE,
    _build_media_test_app,
    _many_media_items,
    _open_media_list,
)
from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    LibraryProductionCSSHarness,
    StaticLibraryMediaScopeService,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_media_state import (
    MediaBrowseScope,
    build_library_media_browse_state,
    build_media_browse_result,
)
from tldw_chatbook.Library.library_media_reader_state import (
    LibraryMediaReaderSessionState,
    begin_selection,
    set_mode,
    settle_success,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen


class ControlledDetailMediaService(StaticLibraryMediaScopeService):
    """Gate each detail response independently without time-based sleeps."""

    def __init__(self, media_items):
        super().__init__(media_items)
        self.detail_entered: dict[int, threading.Event] = {}
        self.detail_release: dict[int, threading.Event] = {}
        self.detail_outcomes: dict[int, object] = {}
        self.progress_calls: list[dict[str, object]] = []
        self.progress_outcomes: dict[int, object] = {}

    def get_media_item(self, *, media_id, **kwargs):
        self.detail_calls.append({"media_id": media_id, **kwargs})
        entered = self.detail_entered.setdefault(media_id, threading.Event())
        release = self.detail_release.setdefault(media_id, threading.Event())
        entered.set()
        if not release.wait(timeout=5):
            raise RuntimeError(f"Timed out waiting to release detail {media_id}.")
        outcome = self.detail_outcomes.get(media_id)
        if isinstance(outcome, BaseException):
            raise outcome
        if outcome is not None:
            return outcome
        for index, item in enumerate(self.media_items):
            if self._backing_id(item, index) == media_id:
                return dict(item)
        return None

    def release(self, media_id: int, outcome: object | None = None) -> None:
        if outcome is not None:
            self.detail_outcomes[media_id] = outcome
        self.detail_release.setdefault(media_id, threading.Event()).set()

    def get_reading_progress(self, *, media_id, **kwargs):
        self.progress_calls.append({"media_id": media_id, **kwargs})
        return self.progress_outcomes.get(media_id)


def _flow_app(count: int = 65):
    app = _build_media_test_app()
    items = _many_media_items(count)
    _seed_conversations(app, _two_conversations(), media=items)
    service = ControlledDetailMediaService(items)
    app.media_reading_scope_service = service
    return app, service


def _row_identity(row: Button) -> tuple[str, int, str]:
    canonical_id = str(row.media_id)
    backing_id = int(canonical_id.rsplit(":", 1)[-1])
    title = str(row._library_media_title)
    return canonical_id, backing_id, title


@pytest.mark.asyncio
async def test_media_global_f6_reaches_content_scroller() -> None:
    """task-28003: F6 into the Reader lands on the scrollable content.

    Before this, the Reader pane's only F6 target was the Find button, so
    the content ScrollView (VirtualizedRawContent, can_focus) was reachable
    by mouse click alone -- keyboard scroll was dead on a fresh open
    (live-verified 2026-09-02). The content scroller is now the first F6
    candidate; Find stays reachable via "/".
    """
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items(3))
    host = LibraryGlobalKeyProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-find")
        content = await _wait_for_selector(
            screen, pilot, "#library-media-viewer-content-text"
        )
        assert content.can_focus  # scroll keys have somewhere to land
        rail = screen.query_one("#library-search-input", Input)
        items = screen.query_one("#library-media-filter", Input)
        rail.focus()
        await pilot.pause()

        for expected in (items, content, rail):
            await pilot.press("f6")
            await pilot.pause()
            assert screen.focused is expected


async def _wait_for_detail_call(
    service: ControlledDetailMediaService, backing_id: int
) -> None:
    try:
        started = await asyncio.to_thread(
            service.detail_entered.setdefault(backing_id, threading.Event()).wait,
            2,
        )
        if not started:
            raise TimeoutError
    except TimeoutError:
        pytest.fail(f"Detail call for backing id {backing_id} did not start.")


@pytest.mark.asyncio
async def test_arrow_traversal_updates_selection_immediately_but_loads_only_settled_row():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        first = screen.query_one("#library-media-row-0", Button)
        first_id, first_backing_id, _ = _row_identity(first)
        first.press()
        await _wait_for_detail_call(service, first_backing_id)
        service.release(first_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == first_id,
            message="Initial settled row did not load.",
        )
        service.detail_calls.clear()
        row_1 = _row_identity(screen.query_one("#library-media-row-1", Button))
        row_2 = _row_identity(screen.query_one("#library-media-row-2", Button))
        screen._select_library_media_reader_row(row_1[0], row_1[2], immediate=False)
        screen._select_library_media_reader_row(row_2[0], row_2[2], immediate=False)

        selected, expected_backing_id, _ = row_2
        assert screen._selected_media_id == selected
        await _wait_for_condition(
            pilot,
            lambda: bool(service.detail_calls),
            message="Settled traversal never dispatched detail.",
        )
        assert [call["media_id"] for call in service.detail_calls] == [
            expected_backing_id
        ]
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_enter_bypasses_selection_settle_window():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-1", Button)
        _, expected_backing_id, _ = _row_identity(row)
        row.focus()
        await pilot.press("enter")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.detail_calls),
            message="Enter did not dispatch detail immediately.",
        )
        assert service.detail_calls[0]["media_id"] == expected_backing_id
        service.release(expected_backing_id)


@pytest.mark.asyncio
async def test_escape_from_reader_focuses_loaded_row_and_down_advances():
    """task-28004: Escape from the Reader lands on the loaded ROW, not the filter.

    Live repro (2026-09-02): Escape focused the "Filter media" Input, so
    the natural next keystrokes were swallowed -- typed characters landed
    in the filter and Down was inert until a Tab. Landing on the loaded
    row keeps Escape-then-Down as the sequential-review gesture (Down
    moves the selection and auto-loads the adjacent item).
    """
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        # Load ROW 0 fully first (the reliable open-and-settle pattern the
        # sibling traversal test uses): a fully-loaded item is what the
        # Escape-then-Down selection path needs -- a still-pending selection
        # is disarmed, not re-selected, when focus moves.
        row_0 = screen.query_one("#library-media-row-0", Button)
        row_0_id, backing_id_0, _ = _row_identity(row_0)
        row_0.press()
        await _wait_for_detail_call(service, backing_id_0)
        service.release(backing_id_0)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == row_0_id,
            message="Row 0 never settled in the Reader.",
        )
        assert screen._selected_media_id == row_0_id

        # Put focus INSIDE the Reader pane, then Escape outward to Items.
        screen.query_one("#library-media-reader-find", Button).focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        focused = screen.focused
        assert focused is not None
        assert focused.has_class("library-media-row")
        assert str(getattr(focused, "media_id", "")) == row_0_id

        # The core fix: because focus is on the ROW (not the filter Input),
        # Down moves to the sibling row instead of being swallowed. Focus
        # movement is synchronous and deterministic; the downstream
        # auto-load-on-arrow is covered by
        # test_arrow_traversal_updates_selection_immediately_but_loads_only_settled_row.
        next_row_id = str(
            screen.query_one("#library-media-row-1", Button).media_id
        )
        await pilot.press("down")
        await pilot.pause()
        assert str(screen.focused.media_id) == next_row_id
        for media_id in tuple(service.detail_release):
            service.release(media_id)


async def _load_row_0(screen, service, pilot):
    """Open and fully settle row 0 in the Reader; return its canonical id."""
    row_0 = screen.query_one("#library-media-row-0", Button)
    row_0_id, backing_id_0, _ = _row_identity(row_0)
    row_0.press()
    await _wait_for_detail_call(service, backing_id_0)
    service.release(backing_id_0)
    await _wait_for_condition(
        pilot,
        lambda: screen._library_media_reader_session.loaded_id == row_0_id,
        message="Row 0 never settled in the Reader.",
    )
    return row_0_id


@pytest.mark.asyncio
async def test_s_key_enters_select_mode_and_space_toggles_a_row():
    """task-28012: keyboard-only bulk selection on the media list."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        assert screen._library_media_select_mode is False
        assert ("s", "select") in screen._library_footer_shortcuts_for_current_state()

        # "s" from a focused row enters select mode.
        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()
        assert screen._library_media_select_mode is True
        footer = screen._library_footer_shortcuts_for_current_state()
        assert ("space", "toggle selection") in footer
        assert ("s", "done selecting") in footer

        # Space toggles the focused row's selection.
        row = screen.query_one("#library-media-row-0", Button)
        row_id = str(row.media_id)
        row.focus()
        await pilot.pause()
        await pilot.press("space")
        await pilot.pause()
        assert screen._library_media_row_selection.is_selected(row_id)


@pytest.mark.asyncio
async def test_bracket_keys_walk_to_next_and_previous_item_in_the_reader():
    """task-28005: ] opens the next browse item, [ the previous, from the Reader."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row_0_id = await _load_row_0(screen, service, pilot)
        row_1_id = str(screen.query_one("#library-media-row-1", Button).media_id)
        assert screen._selected_media_id == row_0_id

        # ] walks DOWN the browse order (newest-first rows) to the next item.
        await pilot.press("]")
        await _wait_for_condition(
            pilot,
            lambda: screen._selected_media_id == row_1_id,
            message="] did not select the next item.",
        )
        # [ walks back to the previous item.
        await pilot.press("[")
        await _wait_for_condition(
            pilot,
            lambda: screen._selected_media_id == row_0_id,
            message="[ did not select the previous item.",
        )
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_prev_item_binding_disabled_at_the_first_item():
    """task-28005: [ is gated off (no-op) at the first item; ] stays active."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row_0_id = await _load_row_0(screen, service, pilot)
        # Row 0 is the first (top) item: no previous exists.
        assert screen.check_action("library_media_prev_item", ()) is False
        assert screen.check_action("library_media_next_item", ()) is True
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_reader_defaults_to_read_and_keeps_mode_across_local_items():
    """Reader mode is session state, not a per-detail display default."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        first = screen.query_one("#library-media-row-0", Button)
        first_id, first_backing_id, _ = _row_identity(first)
        assert screen._library_media_reader_session.mode == "read"

        first.press()
        await _wait_for_detail_call(service, first_backing_id)
        service.release(first_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == first_id,
            message="First local item never loaded.",
        )
        screen._library_media_reader_session = set_mode(
            screen._library_media_reader_session, "analysis"
        )
        screen._sync_library_media_viewer_or_recompose()
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-reader-mode-analysis")),
            message="Analysis mode did not compose.",
        )

        second = screen.query_one("#library-media-row-1", Button)
        second_id, second_backing_id, _ = _row_identity(second)
        second.press()
        await _wait_for_detail_call(service, second_backing_id)
        service.release(second_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == second_id,
            message="Second local item never loaded.",
        )
        assert screen._library_media_reader_session.mode == "analysis"
        assert screen.query("#library-media-reader-mode-analysis")
        assert not screen.query("#library-media-reader-mode-read")


@pytest.mark.asyncio
async def test_edit_metadata_from_read_routes_to_info_form_actions():
    """More's local metadata action must reveal its editable Info surface."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        canonical_id, backing_id, _ = _row_identity(row)
        row.press()
        await _wait_for_detail_call(service, backing_id)
        service.release(backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == canonical_id,
            message="Local detail never loaded for metadata editing.",
        )
        assert screen._library_media_reader_session.mode == "read"

        screen.query_one("#library-media-reader-more", Button).press()
        edit = await _wait_for_selector(screen, pilot, "#library-media-edit")
        edit.press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-form")

        assert screen._library_media_reader_session.mode == "info"
        for selector in (
            "#library-media-edit-title",
            "#library-media-edit-author",
            "#library-media-edit-url",
            "#library-media-edit-keywords",
            "#library-media-edit-save",
            "#library-media-edit-cancel",
        ):
            assert screen.query(selector)


@pytest.mark.asyncio
async def test_external_detail_without_original_exposes_no_empty_more_menu():
    """A server-only detail exposes only actions that it can actually perform."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await screen._open_library_external_media_detail("7")
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-reader-identity")),
            message=lambda: (
                "External server detail never mounted: "
                f"view={screen._library_media_view!r}, "
                f"row={screen._library_selected_row_id!r}, "
                f"session={screen._library_media_reader_session!r}, "
                f"detail={screen._library_media_detail!r}, "
                f"viewer={bool(screen.query('#library-media-viewer'))!r}."
            ),
        )

        assert screen.query_one("#library-media-reader-identity", Static).renderable == (
            "Server item · not in local Media list"
        )
        assert screen.app_instance.media_reading_scope_service.detail_calls[-1] == {
            "media_id": 7,
            "mode": "server",
            "include_content": True,
            "include_versions": True,
        }
        assert not any(
            row.loading or row.loaded
            for row in screen._build_library_media_state().rows
        )
        assert screen.query("#library-media-reader-find")
        assert screen.query("#library-media-use-in-chat")
        assert not screen.query("#library-media-reader-more")
        assert not screen.query("#library-media-read-later")
        assert not screen.query("#library-media-reader-mode-toolbar")
        assert not screen.query("#library-media-highlight-add")
        assert not screen.query("#library-media-edit")
        assert not screen.query("#library-media-delete")

        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen._library_media_reader_session.external_detail,
            message="Back did not clear the external-detail session.",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("late_outcome", [None, RuntimeError("late failure")])
async def test_late_external_a_cannot_replace_b_or_show_error(late_outcome):
    """A stale server detail cannot repaint a newer external Reader session."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        opening_a = asyncio.create_task(screen._open_library_external_media_detail("7"))
        await _wait_for_detail_call(service, 7)
        opening_b = asyncio.create_task(screen._open_library_external_media_detail("8"))
        await _wait_for_detail_call(service, 8)
        service.release(8)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_detail is not None
            and screen._library_media_reader_session.loaded_id == "server:media:8",
            message="External B did not load.",
        )

        service.release(7, late_outcome)
        await asyncio.gather(opening_a, opening_b)
        await pilot.pause()

        assert screen._library_media_reader_session.loaded_id == "server:media:8"
        assert screen._library_media_reader_session.error is None
        assert screen._library_media_detail["id"] == "media-8"


@pytest.mark.asyncio
async def test_info_describes_the_same_stored_text_representation_sent_to_console():
    """Info and Console share the authoritative handoff decision."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        _, backing_id, _ = _row_identity(row)
        source = next(
            item for item in service.media_items if item["id"] == f"media-{backing_id}"
        )
        source["type"] = "markdown"
        source["content"] = "# Stored Markdown\n\n**Exact source text**"
        row.press()
        await _wait_for_detail_call(service, backing_id)
        service.release(backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_backing_id == backing_id,
            message="Local detail never loaded for the Console handoff check.",
        )

        screen.query_one("#library-media-reader-select-info", Button).press()
        provenance = await _wait_for_selector(
            screen, pilot, "#library-media-reader-provenance"
        )
        payload = screen._selected_media_handoff_payload()

        assert payload is not None
        assert "Complete stored text excerpt" in str(provenance.renderable)
        assert "Content excerpt:\n# Stored Markdown\n\n**Exact source text**" in payload.body
        assert payload.body_truncated is False


@pytest.mark.asyncio
async def test_info_calls_empty_content_metadata_only_like_console():
    """Info must not call the Console's explicit no-content payload complete text."""
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        _, backing_id, _ = _row_identity(row)
        source = next(
            item for item in service.media_items if item["id"] == f"media-{backing_id}"
        )
        source["content"] = ""
        row.press()
        await _wait_for_detail_call(service, backing_id)
        service.release(backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_backing_id == backing_id,
            message="Empty-content detail did not load.",
        )

        screen.query_one("#library-media-reader-select-info", Button).press()
        provenance = await _wait_for_selector(
            screen, pilot, "#library-media-reader-provenance"
        )
        payload = screen._selected_media_handoff_payload()

        assert payload is not None
        assert "No stored content." in payload.body
        assert "Metadata only (no stored text)" in str(provenance.renderable)


@pytest.mark.asyncio
async def test_pending_banner_names_selected_b_and_loaded_a():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row_a = screen.query_one("#library-media-row-0", Button)
        canonical_a, backing_a, title_a = _row_identity(row_a)
        row_a.press()
        await _wait_for_detail_call(service, backing_a)
        service.release(backing_a)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == canonical_a,
            message="First detail did not load.",
        )
        row_b = screen.query_one("#library-media-row-1", Button)
        _, backing_b, title_b = _row_identity(row_b)
        row_b.press()
        await _wait_for_detail_call(service, backing_b)
        # task-22207: the banner is now a persistent display-gated widget,
        # so presence alone is vacuous -- wait for it to be PAINTED.
        await _wait_for_condition(
            pilot,
            lambda: (
                bool(screen.query("#library-media-viewer-loading"))
                and screen.query_one(
                    "#library-media-viewer-loading", Static
                ).display
            ),
            message="Reader never painted its pending banner.",
        )

        banner = screen.query_one("#library-media-viewer-loading", Static)
        copy = str(banner.renderable)
        assert title_b in copy
        assert title_a in copy
        service.release(backing_b)


@pytest.mark.asyncio
@pytest.mark.parametrize("late_outcome", [None, RuntimeError("late failure")])
async def test_late_completion_for_a_cannot_replace_loaded_b_or_show_error(
    late_outcome,
):
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row_a = screen.query_one("#library-media-row-0", Button)
        row_b = screen.query_one("#library-media-row-1", Button)
        _, backing_a, _ = _row_identity(row_a)
        canonical_b, backing_b, title_b = _row_identity(row_b)
        row_a.press()
        await _wait_for_detail_call(service, backing_a)
        screen._select_library_media_reader_row(canonical_b, title_b, immediate=True)
        await _wait_for_detail_call(service, backing_b)
        service.release(backing_b)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == canonical_b,
            message="B detail did not load.",
        )

        service.release(backing_a, late_outcome)
        await pilot.pause()
        assert screen._library_media_reader_session.loaded_id == canonical_b
        assert screen._library_media_reader_session.error is None
        assert screen._library_media_detail["id"] == f"media-{backing_b}"


@pytest.mark.asyncio
async def test_detail_failure_keeps_items_usable_and_reader_retryable():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        row = screen.query_one("#library-media-row-0", Button)
        _, backing_id, _ = _row_identity(row)
        row.press()
        await _wait_for_detail_call(service, backing_id)
        service.release(backing_id, RuntimeError("private"))
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.error is not None,
            message="Detail failure did not settle as retryable.",
        )

        assert screen.query_one("#library-media-row-1", Button).disabled is False
        assert screen._library_media_view == "viewer"
        assert screen._library_media_reader_session.pending_request is not None
        retry = screen.query_one("#library-media-reader-retry", Button)
        service.detail_outcomes.pop(backing_id)
        retry.press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_backing_id
            == backing_id,
            message="Reader-local Retry did not reload the failed item.",
        )
        assert screen._library_media_reader_session.error is None


def test_rows_expose_textual_loading_and_loaded_state():
    scope = MediaBrowseScope()
    result = build_media_browse_result(
        scope,
        {
            "items": [
                {
                    "id": "local:media:1",
                    "backing_media_id": 1,
                    "title": "A",
                    "media_type": "audio",
                    "updated_at": None,
                },
                {
                    "id": "local:media:2",
                    "backing_media_id": 2,
                    "title": "B",
                    "media_type": "video",
                    "updated_at": None,
                },
            ],
            "total": 2,
            "limit": 20,
            "offset": 0,
        },
    )

    state = build_library_media_browse_state(
        result,
        type_options=(),
        selected_id="local:media:2",
        loading_id="local:media:2",
        loaded_id="local:media:1",
    )

    assert state.rows[0].loaded is True
    assert state.rows[1].loading is True


@pytest.mark.asyncio
async def test_filter_uses_authoritative_search_and_restores_page_three_anchor():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-next")),
            message="Page 1 pager did not settle.",
        )
        screen.query_one("#library-media-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope.page == 2
                and bool(screen.query("#library-media-next"))
            ),
            message="Page 2 did not apply and settle its pager.",
        )
        screen.query_one("#library-media-next", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope.page == 3
                and bool(screen.query("#library-media-row-10"))
            ),
            message="Page 3 did not apply and settle its rows.",
        )
        anchor = screen.query_one("#library-media-row-10", Button)
        anchor_id, anchor_backing_id, _ = _row_identity(anchor)
        anchor.press()
        await _wait_for_detail_call(service, anchor_backing_id)
        service.release(anchor_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == anchor_id,
            message="Page-3 anchor did not load.",
        )

        media_filter = screen.query_one("#library-media-filter", Input)
        media_filter.value = "Media item 03"
        await _wait_for_condition(
            pilot,
            lambda: any(
                call.get("query") == "Media item 03" for call in service.search_calls
            ),
            message="Filter did not call authoritative search.",
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope.query
                == "Media item 03"
            ),
            message="Filter result did not apply.",
        )
        filtered_id = str(
            screen._library_media_browse_controller.retained_items[0]["id"]
        )
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_reader_session.selected_id == filtered_id
                and screen._library_media_reader_session.pending_request is not None
            ),
            message="The first authoritative filter result was not selected in Reader.",
        )
        screen.query_one("#library-media-filter-clear", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._library_media_browse_controller.applied_scope
                == MediaBrowseScope(page=3)
            ),
            message="Unfiltered page 3 was not restored.",
        )

        assert screen._selected_media_id == anchor_id
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == anchor_id,
            message="Clearing the filter did not restore the Reader anchor.",
        )
        assert (
            len({row.media_id for row in screen._build_library_media_state().rows})
            == 20
        )
        assert [
            call["offset"] for call in service.search_calls if not call.get("query")
        ][-1] == 40
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_entering_select_mode_cancels_pending_single_item_settlement():
    app, service = _flow_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-row-1", Button).focus()
        screen.query_one("#library-media-select-toggle", Button).press()
        await pilot.pause()

        assert screen._library_media_select_mode is True
        assert screen._library_media_reader_session.pending_request is None
        assert service.detail_calls == []


@pytest.mark.asyncio
async def test_delete_selects_following_row_then_previous_then_empty():
    """Single delete keeps traversal anchored to the pre-delete ordering."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        ordered = [
            _row_identity(screen.query_one(f"#library-media-row-{index}", Button))
            for index in range(3)
        ]
        previous_id, previous_backing_id, _ = ordered[0]
        middle_id, middle_backing_id, _ = ordered[1]
        following_id, following_backing_id, _ = ordered[2]

        screen.query_one("#library-media-row-1", Button).press()
        await _wait_for_detail_call(service, middle_backing_id)
        service.release(middle_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == middle_id,
            message="Middle item did not load before delete.",
        )

        screen._library_media_bulk_delete_in_flight = True
        screen._begin_library_media_mutation()
        await screen._delete_library_media_item(middle_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_delete_receipt_ids == (middle_id,),
            message="Single delete did not settle through the shared receipt.",
        )

        assert screen._library_media_reader_session.selected_id == following_id
        await _wait_for_detail_call(service, following_backing_id)
        service.release(following_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == following_id,
            message="Following item did not load after delete.",
        )

        screen._library_media_bulk_delete_in_flight = True
        screen._begin_library_media_mutation()
        await screen._delete_library_media_item(following_id)
        assert screen._library_media_reader_session.selected_id == previous_id
        await _wait_for_detail_call(service, previous_backing_id)
        service.release(previous_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == previous_id,
            message="Previous item did not load after deleting the final row.",
        )

        screen._library_media_bulk_delete_in_flight = True
        screen._begin_library_media_mutation()
        await screen._delete_library_media_item(previous_id)
        assert screen._library_media_reader_session.selected_id is None
        assert screen._library_media_reader_session.loaded_id is None
        assert screen._library_media_detail is None


@pytest.mark.asyncio
async def test_stale_detail_never_writes_progress_under_new_selected_id():
    """Progress persists under the mounted loaded item, not traversal intent."""
    loaded = LibraryMediaReaderSessionState(
        selected_id="local:media:1",
        selected_backing_id=1,
        selected_title="Loaded",
        loaded_id="local:media:1",
        loaded_backing_id=1,
        loaded_title="Loaded",
    )
    session = begin_selection(loaded, "local:media:2", 2, "Selected")
    calls = []

    async def update_reading_progress(**kwargs):
        calls.append(kwargs)

    async def run_service_call(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    workers = []
    fake = SimpleNamespace(
        _library_media_reader_session=session,
        _library_media_read_scroll_by_id={},
        _library_media_progress_pending_writes={},
        _library_media_progress_inflight_write=None,
        _library_media_progress_persisted_offsets={},
        _library_media_progress_write_worker=None,
        is_attached=True,
        app_instance=SimpleNamespace(
            media_reading_scope_service=SimpleNamespace(
                update_reading_progress=update_reading_progress
            )
        ),
        query_one=lambda *_args, **_kwargs: SimpleNamespace(
            scroller=SimpleNamespace(scroll_x=0, scroll_y=17)
        ),
        run_worker=lambda coro, **_kwargs: workers.append(coro),
    )
    fake._run_library_service_call = run_service_call
    _bind_progress_methods(fake)

    LibraryScreen._capture_library_media_loaded_progress(fake)
    await workers[0]
    stale_cached = LibraryScreen._cache_library_media_reading_progress(
        fake,
        loaded.request_generation,
        "local:media:1",
        {"scroll_x": 0, "scroll_y": 99},
    )

    assert calls == [
        {
            "mode": "local",
            "media_id": 1,
            "progress_data": {"scroll_x": 0, "scroll_y": 17},
        }
    ]
    assert stale_cached is False
    assert fake._library_media_read_scroll_by_id["local:media:1"] == (0, 17)


def test_progress_restores_after_loaded_content_mounts():
    """The loaded local identity owns the offset restored into its body."""
    calls = []
    fake = SimpleNamespace(
        _library_media_reader_session=LibraryMediaReaderSessionState(
            selected_id="local:media:7",
            selected_backing_id=7,
            selected_title="Seven",
            loaded_id="local:media:7",
            loaded_backing_id=7,
            loaded_title="Seven",
        ),
        _library_media_read_scroll_by_id={"local:media:7": (2, 19)},
        query_one=lambda *_args, **_kwargs: SimpleNamespace(
            scroller=SimpleNamespace(
                scroll_to=lambda **kwargs: calls.append(kwargs)
            )
        ),
    )

    LibraryScreen._restore_library_media_loaded_progress(fake, "local:media:7")

    assert calls == [{"x": 2, "y": 19, "animate": False, "force": True}]


@pytest.mark.asyncio
async def test_progress_loads_from_service_when_detail_has_no_embedded_progress():
    """Fresh sessions use the progress service after fenced detail ownership."""
    calls = []

    async def get_reading_progress(**kwargs):
        calls.append(kwargs)
        return {"scroll_x": 0, "scroll_y": 12}

    async def run_service_call(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    session = begin_selection(
        LibraryMediaReaderSessionState(), "local:media:1", 1, "One"
    )
    fake = SimpleNamespace(
        _library_media_reader_session=session,
        _library_media_read_scroll_by_id={},
        _library_media_progress_persisted_offsets={},
        app_instance=SimpleNamespace(
            media_reading_scope_service=SimpleNamespace(
                get_reading_progress=get_reading_progress
            )
        ),
        _run_library_service_call=run_service_call,
    )
    fake._required_library_media_backing_id = MethodType(
        LibraryScreen._required_library_media_backing_id, fake
    )
    fake._library_media_backing_id = MethodType(
        LibraryScreen._library_media_backing_id, fake
    )
    detail = {"id": 1, "content": "No embedded progress"}

    progress = await LibraryScreen._fetch_library_media_reading_progress(
        fake, "local:media:1"
    )
    cached = LibraryScreen._cache_library_media_reading_progress(
        fake, session.request_generation, "local:media:1", progress
    )

    assert "reading_progress" not in detail
    assert calls == [{"mode": "local", "media_id": 1}]
    assert cached is True
    assert fake._library_media_read_scroll_by_id["local:media:1"] == (0, 12)


def test_mode_change_preserves_per_item_read_scroll_for_session():
    """Leaving and returning to Read uses the loaded item's session snapshot."""
    restored = []
    scroller = SimpleNamespace(
        scroll_x=0,
        scroll_y=23,
        scroll_to=lambda **kwargs: restored.append(kwargs),
    )
    body = SimpleNamespace(scroller=scroller)
    fake = SimpleNamespace(
        _library_media_reader_session=LibraryMediaReaderSessionState(
            selected_id="local:media:4",
            selected_backing_id=4,
            selected_title="Four",
            loaded_id="local:media:4",
            loaded_backing_id=4,
            loaded_title="Four",
        ),
        _library_media_read_scroll_by_id={},
        app_instance=SimpleNamespace(media_reading_scope_service=None),
        query_one=lambda *_args, **_kwargs: body,
    )

    LibraryScreen._capture_library_media_loaded_progress(fake)
    fake._library_media_reader_session = set_mode(
        fake._library_media_reader_session, "analysis"
    )
    fake._library_media_reader_session = set_mode(
        fake._library_media_reader_session, "read"
    )
    LibraryScreen._restore_library_media_loaded_progress(fake, "local:media:4")

    assert restored == [{"x": 0, "y": 23, "animate": False, "force": True}]


def test_external_server_detail_does_not_use_local_progress_seam():
    """External detail has no local loaded-id progress ownership."""
    queried = []
    fake = SimpleNamespace(
        _library_media_reader_session=LibraryMediaReaderSessionState(
            selected_id="server:media:9",
            selected_backing_id=9,
            selected_title="Server",
            loaded_id="server:media:9",
            loaded_backing_id=9,
            loaded_title="Server",
            external_detail=True,
        ),
        _library_media_read_scroll_by_id={},
        query_one=lambda *_args, **_kwargs: queried.append(True),
        run_worker=lambda *_args, **_kwargs: pytest.fail("local progress worker ran"),
    )

    LibraryScreen._capture_library_media_loaded_progress(fake)

    assert queried == []


def _escape_fake_query_one(shell, find, *, mounted):
    from textual.css.query import NoMatches

    def query_one(selector, *_args):
        if selector == "#library-media-reader-shell":
            return shell
        if selector == "#library-media-content-search-controls" and not mounted:
            raise NoMatches(selector)
        return find

    return query_one


def _escape_fake(
    *,
    region: str,
    more_open: bool = False,
    find_mounted: bool | None = None,
    find_open: bool | None = None,
):
    layout = SimpleNamespace(items_open=True, library_open=True)
    library = SimpleNamespace(role="library", ancestors=())
    items = SimpleNamespace(role="items", ancestors=())
    reader = SimpleNamespace(role="reader", ancestors=())
    shell = SimpleNamespace(
        library=library,
        items=items,
        reader=reader,
        effective_layout=layout,
    )
    find = SimpleNamespace(role="find", ancestors=())
    owner = {"library": library, "items": items, "reader": reader, "find": find}[
        region
    ]
    focused = SimpleNamespace(ancestors=(owner,))
    # task-31237: the bar is collapsed by default, so the fake mounts it
    # only when the test's region IS the find bar (or explicitly asked).
    mounted = (region == "find") if find_mounted is None else find_mounted
    # task-31271 seam (a): DOM presence and screen state are the same thing
    # AT REST, but a recompose is one refresh behind in both directions --
    # ``find_open`` splits them so a test can sit inside that window.
    open_state = mounted if find_open is None else find_open
    calls = []
    fake = SimpleNamespace(
        focused=focused,
        _library_media_bulk_delete_in_flight=False,
        _library_media_editing=False,
        _library_media_confirming_delete=False,
        _library_media_editing_analysis=False,
        # task-31271 seam (a): the escape label reads the bar's STATE, not
        # its DOM presence (the DOM is one refresh behind). The viewer
        # mounts the bar for exactly ``find_open or content_query``, so the
        # fake keeps the two sides consistent -- a mounted bar with an
        # empty state is a shape production cannot produce.
        _library_media_find_open=open_state,
        _library_media_content_query="needle" if open_state else "",
        _library_media_content_match_index=1 if open_state else 0,
        _library_media_reader_session=LibraryMediaReaderSessionState(
            more_open=more_open
        ),
        query_one=_escape_fake_query_one(shell, find, mounted=mounted),
        _sync_library_media_viewer_or_recompose=lambda: calls.append("sync"),
        _focus_library_control=lambda selector: calls.append(("focus", selector)),
        _focus_library_media_items_pane=lambda: calls.append(("items-pane",)),
        _focus_library_rail_action=lambda selector: calls.append(("rail", selector)),
        _exit_library_media_viewer=lambda: calls.append("back"),
        _register_footer_shortcuts=lambda: calls.append("footer"),
        call_after_refresh=lambda callback, *args: callback(*args),
    )
    # task-31237: Escape's find branch routes through the shared close seam.
    fake._close_library_media_find = MethodType(
        LibraryScreen._close_library_media_find, fake
    )
    # task-31271 seam (a): Escape and its footer label read one seam now.
    fake._library_media_find_state = MethodType(
        LibraryScreen._library_media_find_state, fake
    )
    return fake, calls, shell, find


def test_escape_and_its_label_read_the_same_find_state():
    """task-31271 seam (a): the bar's DOM presence lags its state by one
    refresh, so Escape and the chip that describes it must both read the
    STATE -- otherwise, for that one refresh, the footer promises a close
    the key will not perform (or the reverse).
    """
    # Closed, but the bar is still mounted (the window right after Escape).
    fake, calls, _shell, _find = _escape_fake(
        region="reader", find_mounted=True, find_open=False
    )
    assert LibraryScreen._library_media_escape_label(fake) != "close find"
    LibraryScreen.action_library_media_viewer_back(fake)
    assert calls[:1] == [("items-pane",)]  # graduated panes, did not close find

    # Open, but the bar is not mounted yet (the window right after Find).
    fake, calls, _shell, _find = _escape_fake(
        region="reader", find_mounted=False, find_open=True
    )
    assert LibraryScreen._library_media_escape_label(fake) == "close find"
    LibraryScreen.action_library_media_viewer_back(fake)
    assert fake._library_media_find_open is False
    assert calls == ["sync", ("focus", "#library-media-reader-find")]


def test_escape_closes_more_find_confirmation_before_leaving_reader():
    """Every Reader transient consumes Escape before outward graduation."""
    fake, calls, _shell, _find = _escape_fake(region="reader", more_open=True)
    LibraryScreen.action_library_media_viewer_back(fake)
    assert fake._library_media_reader_session.more_open is False
    assert calls == ["sync", ("focus", "#library-media-reader-more")]

    fake, calls, _shell, _find = _escape_fake(region="find")
    LibraryScreen.action_library_media_viewer_back(fake)
    assert fake._library_media_content_query == ""
    assert calls == ["sync", ("focus", "#library-media-reader-find")]

    fake, calls, _shell, _find = _escape_fake(region="reader")
    fake._library_media_confirming_delete = True
    LibraryScreen.action_library_media_viewer_back(fake)
    assert fake._library_media_confirming_delete is False
    assert calls == ["sync"]

    fake, calls, _shell, _find = _escape_fake(region="reader")
    fake._library_media_editing = True
    LibraryScreen.action_library_media_viewer_back(fake)
    assert fake._library_media_editing is False
    assert calls == ["sync"]

    fake, calls, _shell, _find = _escape_fake(region="reader")
    fake._library_media_editing_analysis = True
    LibraryScreen.action_library_media_viewer_back(fake)
    assert fake._library_media_editing_analysis is False
    assert calls == ["sync"]


def test_escape_closes_an_open_find_bar_from_anywhere_in_the_reader():
    """Qodo on #2367: an open Find bar is a reader substate, focus-agnostic.

    F6 to the content pane must not leave the bar stranded — Escape from
    any reader-region focus closes it first (one press), and only the next
    Escape graduates panes. Items/Library-pane Escapes are unaffected.
    """
    fake, calls, _shell, _find = _escape_fake(region="reader", find_mounted=True)
    LibraryScreen.action_library_media_viewer_back(fake)
    assert fake._library_media_content_query == ""
    assert fake._library_media_find_open is False
    assert calls == ["sync", ("focus", "#library-media-reader-find")]

    # An Items-pane Escape never consumes the reader's find bar.
    fake, calls, shell, _find = _escape_fake(region="items", find_mounted=True)
    LibraryScreen.action_library_media_viewer_back(fake)
    assert calls == [("rail", "#library-search-input")]


def test_escape_label_names_close_find_while_the_bar_is_open():
    """The footer label matches the widened close (Qodo #2367 finding 4)."""
    fake, _calls, _shell, _find = _escape_fake(
        region="reader", find_mounted=True
    )
    assert LibraryScreen._library_media_escape_label(fake) == "close find"


def test_escape_moves_reader_to_items_then_library_then_screen_back():
    """One outward handler graduates through the effective pane hierarchy."""
    fake, calls, shell, _find = _escape_fake(region="reader")
    LibraryScreen.action_library_media_viewer_back(fake)
    # task-28004: the Items landing is the loaded row (falling back to the
    # filter only on an empty list), so Escape-then-Down keeps working.
    assert calls[:1] == [("items-pane",)]

    fake.focused = SimpleNamespace(ancestors=(shell.items,))
    calls.clear()
    LibraryScreen.action_library_media_viewer_back(fake)
    assert calls == [("rail", "#library-search-input")]

    fake.focused = SimpleNamespace(ancestors=(shell.library,))
    calls.clear()
    LibraryScreen.action_library_media_viewer_back(fake)
    assert calls == ["back"]


def test_escape_skips_responsively_collapsed_panes():
    """Collapsed intermediate roles are absent from outward graduation."""
    fake, calls, shell, _find = _escape_fake(region="reader")
    shell.effective_layout.items_open = False
    LibraryScreen.action_library_media_viewer_back(fake)
    assert calls[:1] == [("rail", "#library-search-input")]

    calls.clear()
    shell.effective_layout.library_open = False
    LibraryScreen.action_library_media_viewer_back(fake)
    assert calls == ["back", "footer"]


@pytest.mark.asyncio
async def test_hidden_panes_have_no_focusable_descendants_but_grips_remain_reachable():
    """A collapsed pane leaves only its fixed grip in the focus chain."""
    app, _service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        shell = screen.query_one("#library-media-reader-shell")
        screen.query_one("#library-media-items-grip", Button).press()
        await pilot.pause()

        assert shell.items.display is False
        assert shell.items.disabled is True
        assert all(
            widget is not shell.items and shell.items not in widget.ancestors
            for widget in screen.focus_chain
        )
        assert shell.items_grip in screen.focus_chain


def test_footer_advertises_only_working_current_actions():
    """The Escape hint names the currently effective focus destination."""
    fake, _calls, shell, _find = _escape_fake(region="reader")
    assert LibraryScreen._library_media_escape_label(fake) == "focus Items"
    fake.focused = SimpleNamespace(ancestors=(shell.items,))
    assert LibraryScreen._library_media_escape_label(fake) == "focus Library"
    shell.effective_layout.library_open = False
    assert LibraryScreen._library_media_escape_label(fake) == "back"


class CountingProgressMediaService(ControlledDetailMediaService):
    """Count settled progress writes without gating them (TASK-22210 probe)."""

    def __init__(self, media_items):
        super().__init__(media_items)
        self.progress_update_calls: list[dict[str, object]] = []

    def update_reading_progress(self, **kwargs):
        self.progress_update_calls.append(dict(kwargs))
        return dict(kwargs)


def _bind_progress_methods(fake) -> None:
    """Bind every progress-write method the screen currently defines."""
    for name in (
        "_write_library_media_loaded_progress",
        "_queue_library_media_progress_write",
        "_drain_library_media_progress_writes",
        "_library_media_progress_write_is_current",
    ):
        method = getattr(LibraryScreen, name, None)
        if method is not None:
            setattr(fake, name, MethodType(method, fake))


def _progress_capture_fake(
    *,
    workers: list,
    update_reading_progress,
    run_service_call,
    scroll_y: int = 17,
):
    """A capture-surface fake carrying the coalescing state slots."""
    fake = SimpleNamespace(
        _library_media_reader_session=LibraryMediaReaderSessionState(
            selected_id="local:media:3",
            selected_backing_id=3,
            selected_title="Three",
            loaded_id="local:media:3",
            loaded_backing_id=3,
            loaded_title="Three",
        ),
        _library_media_read_scroll_by_id={},
        _library_media_progress_pending_writes={},
        _library_media_progress_inflight_write=None,
        _library_media_progress_persisted_offsets={},
        _library_media_progress_write_worker=None,
        is_attached=True,
        app_instance=SimpleNamespace(
            media_reading_scope_service=SimpleNamespace(
                update_reading_progress=update_reading_progress
            )
        ),
        query_one=lambda *_args, **_kwargs: SimpleNamespace(
            scroller=SimpleNamespace(scroll_x=0, scroll_y=scroll_y)
        ),
        # Production run_worker returns a Worker that stays unfinished until
        # the drainer completes; the queue seam relies on that to spawn only
        # one drainer per burst.
        run_worker=lambda coro, **_kwargs: (
            workers.append(coro),
            SimpleNamespace(is_finished=False),
        )[1],
    )
    fake._run_library_service_call = run_service_call
    _bind_progress_methods(fake)
    return fake


@pytest.mark.asyncio
async def test_identical_consecutive_captures_settle_a_single_progress_write():
    """TASK-22210 probe: 30 unchanged captures coalesce to one settled write."""
    calls = []

    async def update_reading_progress(**kwargs):
        calls.append(kwargs)

    async def run_service_call(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    workers: list = []
    fake = _progress_capture_fake(
        workers=workers,
        update_reading_progress=update_reading_progress,
        run_service_call=run_service_call,
    )

    for _ in range(30):
        LibraryScreen._capture_library_media_loaded_progress(fake)
    for worker in workers:
        await worker

    assert len(workers) == 1, f"expected one queued writer, got {len(workers)}"
    assert len(calls) == 1, f"expected one settled write, got {len(calls)}"
    assert calls[0]["progress_data"] == {"scroll_x": 0, "scroll_y": 17}
    assert fake._library_media_read_scroll_by_id["local:media:3"] == (0, 17)


@pytest.mark.asyncio
async def test_offset_burst_settles_only_the_newest_value_per_item():
    """TASK-22210 probe: queued writes coalesce; only the latest offset lands."""
    calls = []

    async def update_reading_progress(**kwargs):
        calls.append(kwargs)

    async def run_service_call(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    workers: list = []
    scroller = SimpleNamespace(scroll_x=0, scroll_y=5)
    body = SimpleNamespace(scroller=scroller)
    fake = _progress_capture_fake(
        workers=workers,
        update_reading_progress=update_reading_progress,
        run_service_call=run_service_call,
    )
    fake.query_one = lambda *_args, **_kwargs: body

    for scroll_y in (5, 9, 17):
        scroller.scroll_y = scroll_y
        LibraryScreen._capture_library_media_loaded_progress(fake)
    # The drainer had no chance to run during the burst (the fake
    # run_worker defers); drain now and require last-write-wins.
    for worker in workers:
        await worker

    assert len(workers) == 1, f"expected one queued drainer, got {len(workers)}"
    assert [call["progress_data"] for call in calls] == [
        {"scroll_x": 0, "scroll_y": 17}
    ]
    assert fake._library_media_progress_persisted_offsets["local:media:3"] == (0, 17)


@pytest.mark.asyncio
async def test_capture_outside_read_never_persists_the_analysis_body_offset():
    """task-28026: the Analysis body reuses ``#library-media-viewer-content``,
    so capturing progress while a non-Read tab is active must NOT record that
    scroll offset as the transcript's reading progress."""
    calls = []

    async def update_reading_progress(**kwargs):
        calls.append(kwargs)

    async def run_service_call(call, *args, **kwargs):
        kwargs.pop("isolate_in_worker", None)
        return await call(*args, **kwargs)

    workers: list = []
    fake = _progress_capture_fake(
        workers=workers,
        update_reading_progress=update_reading_progress,
        run_service_call=run_service_call,
        scroll_y=42,
    )
    fake._library_media_reader_session = set_mode(
        fake._library_media_reader_session, "analysis"
    )

    LibraryScreen._capture_library_media_loaded_progress(fake)

    assert workers == []
    assert calls == []
    assert fake._library_media_read_scroll_by_id == {}


def test_find_from_analysis_opens_the_bar_on_the_analysis_tab():
    """task-31269: Find searches the tab you are reading. On the Analysis
    tab it opens the analysis bar in place (task-28026's Analysis->Read
    jump is retired) and hands the viewer a one-shot focus token; the
    query is untouched because the mode did not change."""
    session = set_mode(
        LibraryMediaReaderSessionState(
            selected_id="local:media:1",
            selected_backing_id=1,
            selected_title="A",
            loaded_id="local:media:1",
            loaded_backing_id=1,
            loaded_title="A",
        ),
        "analysis",
    )
    fake = SimpleNamespace(
        _library_media_reader_session=session,
        _library_media_content_query="needle",
        _library_media_content_match_index=5,
        _library_media_content_match_memo=(object(), "needle", (1, 2, 3), "analysis"),
        _library_media_find_open=False,
        _library_media_find_focus_pending=False,
        _sync_library_media_viewer_or_recompose=lambda: None,
        call_after_refresh=lambda *a, **k: None,
        _focus_library_media_content_search_input=lambda: None,
    )
    fake._reset_library_media_search_on_mode_change = MethodType(
        LibraryScreen._reset_library_media_search_on_mode_change, fake
    )
    fake._close_library_media_find = MethodType(
        LibraryScreen._close_library_media_find, fake
    )
    LibraryScreen.handle_library_media_reader_find(
        fake, SimpleNamespace(stop=lambda: None)
    )
    assert fake._library_media_reader_session.mode == "analysis"
    assert fake._library_media_find_open is True
    assert fake._library_media_find_focus_pending is True
    assert fake._library_media_content_query == "needle"


def test_media_content_matches_scopes_corpus_to_the_active_reader_mode():
    """task-28026: the match memo key includes reader mode, so the Analysis
    tab searches the analysis text and Read searches the transcript -- a tab
    switch never serves the other tab's matches from the one-slot memo."""
    detail = {"id": 1}
    state = SimpleNamespace(
        content="alpha\nbeta\nalpha",
        analysis="gamma\nalpha\ndelta",
    )
    fake = SimpleNamespace(
        _library_media_detail=detail,
        _library_media_content_query="alpha",
        _library_media_content_match_memo=None,
        _library_media_reader_session=LibraryMediaReaderSessionState(),
        _library_media_viewer_state_cached=lambda _detail: state,
    )

    # Read tab -> transcript corpus ("alpha" on lines 0 and 2).
    assert LibraryScreen._library_media_content_matches(fake) == (0, 2)
    memo_after_read = fake._library_media_content_match_memo
    assert memo_after_read is not None and memo_after_read[3] == "read"

    # Same detail + query, Analysis tab: the memo must MISS on the mode and
    # rescan the analysis text ("alpha" on line 1), not serve the cached
    # transcript matches.
    fake._library_media_reader_session = set_mode(
        fake._library_media_reader_session, "analysis"
    )
    assert LibraryScreen._library_media_content_matches(fake) == (1,)
    assert fake._library_media_content_match_memo[3] == "analysis"


def test_capture_matching_fetched_progress_skips_the_write():
    """TASK-22210 probe: an offset already in the DB never re-writes."""
    workers: list = []

    def update_reading_progress(**kwargs):  # pragma: no cover - must not run
        pytest.fail("unchanged offset reached the progress service")

    async def run_service_call(call, *args, **kwargs):  # pragma: no cover
        kwargs.pop("isolate_in_worker", None)
        return call(*args, **kwargs)

    fake = _progress_capture_fake(
        workers=workers,
        update_reading_progress=update_reading_progress,
        run_service_call=run_service_call,
        scroll_y=12,
    )
    session = begin_selection(
        LibraryMediaReaderSessionState(), "local:media:3", 3, "Three"
    )
    fake._library_media_reader_session = session
    cached = LibraryScreen._cache_library_media_reading_progress(
        fake,
        session.request_generation,
        "local:media:3",
        {"scroll_x": 0, "scroll_y": 12},
    )
    fake._library_media_reader_session = settle_success(
        session, session.request_generation, "local:media:3"
    )

    LibraryScreen._capture_library_media_loaded_progress(fake)

    for coro in workers:
        coro.close()
    assert cached is True
    assert workers == [], "capture queued a write for an unchanged offset"


@pytest.mark.asyncio
async def test_thirty_step_traversal_settles_at_most_one_progress_write():
    """TASK-22210 probe: a held-key traversal must not stack SQLite writers."""
    app = _build_media_test_app()
    items = _many_media_items(40)
    _seed_conversations(app, _two_conversations(), media=items)
    service = CountingProgressMediaService(items)
    app.media_reading_scope_service = service
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        first = screen.query_one("#library-media-row-0", Button)
        first_id, first_backing_id, _ = _row_identity(first)
        first.press()
        await _wait_for_detail_call(service, first_backing_id)
        service.release(first_backing_id)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_reader_session.loaded_id == first_id,
            message="Initial settled row did not load.",
        )
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-viewer-content")),
            message="Loaded content body never mounted.",
        )

        spawns: list = []
        original_run_worker = screen.run_worker

        def counting_run_worker(work, *args, **kwargs):
            if kwargs.get("group") == "library_media_reading_progress":
                spawns.append(kwargs.get("group"))
            return original_run_worker(work, *args, **kwargs)

        screen.run_worker = counting_run_worker
        # 30 traversal steps across the 20-row page: sweep down, then back up.
        step_indexes = list(range(1, 20)) + list(range(18, 7, -1))
        assert len(step_indexes) == 30
        rows = [
            _row_identity(screen.query_one(f"#library-media-row-{index}", Button))
            for index in step_indexes
        ]
        for canonical_id, _backing_id, title in rows:
            screen._select_library_media_reader_row(
                canonical_id, title, immediate=False
            )
            # Key repeat spans event-loop turns: give the drainer the chance
            # to finish between steps so a missing equality skip would
            # re-queue (and re-write) the unchanged offset every step.
            await pilot.pause()
        await _wait_for_condition(
            pilot,
            lambda: not any(
                worker.group == "library_media_reading_progress"
                and not worker.is_finished
                for worker in screen.workers
            ),
            message="Progress write workers never settled.",
        )

        # 30 traversal steps used to spawn 30 concurrent writers; the fix
        # settles exactly one (the loaded item's first snapshot -- every
        # later step captures the same unchanged offset).
        assert len(spawns) == 1, f"expected 1 worker spawn, got {len(spawns)}"
        assert len(service.progress_update_calls) == 1, (
            f"expected 1 settled progress write, "
            f"got {len(service.progress_update_calls)}"
        )
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_unmount_drains_pending_and_ambiguous_inflight_progress_writes():
    """TASK-22210 teardown: the last captured offsets survive screen teardown."""
    app = _build_media_test_app()
    items = _many_media_items(3)
    _seed_conversations(app, _two_conversations(), media=items)
    service = CountingProgressMediaService(items)
    app.media_reading_scope_service = service
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        # A queued-but-undrained value, and a write whose drainer may have
        # been cancelled mid-flight: both must be durable after unmount.
        screen._library_media_progress_pending_writes["local:media:1"] = (1, (0, 33))
        screen._library_media_progress_inflight_write = ("local:media:2", 2, (0, 7))

        await screen.on_unmount()

        written = {
            call["media_id"]: call["progress_data"]
            for call in service.progress_update_calls
        }
        assert written.get(1) == {"scroll_x": 0, "scroll_y": 33}
        assert written.get(2) == {"scroll_x": 0, "scroll_y": 7}
        assert screen._library_media_progress_pending_writes == {}
        assert screen._library_media_progress_inflight_write is None


# ---------------------------------------------------------------------------
# task-28027: Reader action-row accelerator keys (l / c / t)
# ---------------------------------------------------------------------------


def _reader_key_fake(*, view="viewer", external=False, substate=False, pending=False):
    """A minimal screen fake for the Reader action-key check_action gates."""
    from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_MEDIA

    media_id = "local:media:1"
    return SimpleNamespace(
        _library_selected_row_id=LIBRARY_ROW_BROWSE_MEDIA,
        _library_media_view=view,
        _selected_media_id=media_id,
        _library_media_reader_session=SimpleNamespace(
            external_detail=external,
            pending_request=object() if pending else None,
            # A settled Reader has its loaded id == the selected id.
            loaded_id=None if pending else media_id,
        ),
        _library_media_viewer_substate_active=lambda: substate,
    )


def test_reader_action_keys_gated_to_plain_local_viewer():
    """task-28027: l/c/t active only in a plain local Reader; c also for external."""
    plain = _reader_key_fake()
    for action in (
        "library_media_read_later",
        "library_media_use_in_console",
        "library_media_move_to_trash",
    ):
        assert LibraryScreen.check_action(plain, action, ()) is True, action

    # A sub-state (edit/confirm/analysis-edit) gates all three off.
    sub = _reader_key_fake(substate=True)
    for action in (
        "library_media_read_later",
        "library_media_use_in_console",
        "library_media_move_to_trash",
    ):
        assert LibraryScreen.check_action(sub, action, ()) is False, action

    # External (server) items: read-later/trash are local-only, Console works.
    ext = _reader_key_fake(external=True)
    assert LibraryScreen.check_action(ext, "library_media_read_later", ()) is False
    assert LibraryScreen.check_action(ext, "library_media_move_to_trash", ()) is False
    assert LibraryScreen.check_action(ext, "library_media_use_in_console", ()) is True

    # Not in the viewer at all -> all off.
    listing = _reader_key_fake(view="list")
    for action in (
        "library_media_read_later",
        "library_media_use_in_console",
        "library_media_move_to_trash",
    ):
        assert LibraryScreen.check_action(listing, action, ()) is False, action

    # Qodo #2317: while a detail request is pending (mid-load/traversal), the
    # displayed detail is not yet the selected item -> all off.
    loading = _reader_key_fake(pending=True)
    for action in (
        "library_media_read_later",
        "library_media_use_in_console",
        "library_media_move_to_trash",
    ):
        assert LibraryScreen.check_action(loading, action, ()) is False, action


def test_t_key_arms_delete_confirmation():
    """task-28027: 't' arms the same inline delete-confirm the button does."""
    fake = SimpleNamespace(
        _library_media_confirming_delete=False,
        _library_media_delete_receipt_ids=("stale",),
        _synced=0,
    )
    fake._sync_library_media_viewer_or_recompose = lambda: setattr(
        fake, "_synced", fake._synced + 1
    )
    LibraryScreen.action_library_media_move_to_trash(fake)
    assert fake._library_media_confirming_delete is True
    assert fake._library_media_delete_receipt_ids == ()
    assert fake._synced == 1


def test_c_key_opens_console_handoff():
    """task-28027: 'c' routes to the same Use-in-Console handoff as the button."""
    calls = []
    fake = SimpleNamespace(_open_selected_media_handoff=lambda: calls.append(1))
    LibraryScreen.action_library_media_use_in_console(fake)
    assert calls == [1]


def test_l_key_starts_read_later_toggle():
    """task-28027: 'l' starts the same read-it-later toggle as the button."""
    started = []
    fake = SimpleNamespace(
        _start_library_media_read_later_toggle=lambda: started.append(1)
    )
    LibraryScreen.action_library_media_read_later(fake)
    assert started == [1]


@pytest.mark.asyncio
async def test_t_key_in_reader_arms_delete_confirm_end_to_end():
    """task-28027: pressing 't' in the mounted Reader arms the delete confirm."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        assert screen._library_media_confirming_delete is False

        # Focus a non-input Reader control so the printable 't' reaches the
        # screen binding (not a search/filter Input).
        screen.query_one("#library-media-reader-find", Button).focus()
        await pilot.pause()
        await pilot.press("t")
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")
        assert screen._library_media_confirming_delete is True
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_l_key_in_reader_toggles_read_later_end_to_end():
    """task-28027 (Qodo #2317): 'l' in the mounted Reader saves read-it-later."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)

        screen.query_one("#library-media-reader-find", Button).focus()
        await pilot.pause()
        await pilot.press("l")
        await _wait_for_condition(
            pilot,
            lambda: bool(service.read_it_later_calls),
            message="'l' did not toggle read-it-later.",
        )
        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
async def test_c_key_in_reader_opens_console_handoff_end_to_end(monkeypatch):
    """task-28027 (Qodo #2317): 'c' in the mounted Reader routes to the handoff."""
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=WIDE_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)

        calls = []
        monkeypatch.setattr(
            screen, "_open_selected_media_handoff", lambda: calls.append(1)
        )
        screen.query_one("#library-media-reader-find", Button).focus()
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()
        assert calls == [1]
        for media_id in tuple(service.detail_release):
            service.release(media_id)
