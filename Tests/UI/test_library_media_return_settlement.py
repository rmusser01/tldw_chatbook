"""Media adaptive-stage presentation regressions for return settlement."""

from __future__ import annotations

import dataclasses

import pytest
from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.geometry import Size
from textual.widgets import Button

from Tests.UI.test_library_media_side_by_side import (
    COMPACT_SCROLL_SIZE,
    _build_media_test_app,
    _many_media_items,
    _open_media_list,
    _open_scrolled_compact_media_viewer,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _two_media_items,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.Library import (
    library_media_canvas as library_media_canvas_module,
)
from tldw_chatbook.Widgets.Library.library_media_reader_shell import (
    LibraryMediaReaderShell,
    MediaShellResized,
)


async def _open_compact_media(host, pilot) -> LibraryScreen:
    """Enter Media after selecting the compact presentation contract."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen._library_notes_compact = True
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-reader-shell")
    return screen


@pytest.mark.asyncio
async def test_first_frame_media_stage_projects_adaptive_compact_without_legacy_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    screen = LibraryScreen(app)
    monkeypatch.setattr(
        screen,
        "_reconcile_library_media_stage_presentation",
        lambda: False,
    )
    monkeypatch.setattr(screen, "_apply_library_notes_stage_visibility", lambda: None)
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        stage = screen.query_one("#library-shell-grid", Horizontal)

        assert stage.has_class("library-adaptive-compact")
        assert not stage.has_class("library-notes-compact")


@pytest.mark.asyncio
async def test_same_size_recompose_projects_media_stage_without_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    screen = LibraryScreen(app)
    monkeypatch.setattr(
        screen,
        "_reconcile_library_media_stage_presentation",
        lambda: False,
    )
    monkeypatch.setattr(screen, "_apply_library_notes_stage_visibility", lambda: None)
    host = LibraryProductionCSSHarness(app, screen=screen)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        previous_stage = screen.query_one("#library-shell-grid", Horizontal)

        await screen.recompose()
        replacement_stage = screen.query_one("#library-shell-grid", Horizontal)

        assert replacement_stage is not previous_stage
        assert replacement_stage.has_class("library-adaptive-compact")
        assert not replacement_stage.has_class("library-notes-compact")


@pytest.mark.asyncio
async def test_media_shell_lifecycle_reconciles_current_stage_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_compact_media(host, pilot)
        stage = screen.query_one("#library-shell-grid", Horizontal)
        shell = screen.query_one("#library-media-reader-shell", LibraryMediaReaderShell)
        stage.set_class(True, "library-notes-compact")
        stage.set_class(False, "library-adaptive-compact")

        layout_refreshes: list[bool] = []
        refresh = screen.refresh

        def count_layout_refresh(*args, **kwargs):
            layout_refreshes.append(bool(kwargs.get("layout", False)))
            return refresh(*args, **kwargs)

        monkeypatch.setattr(screen, "refresh", count_layout_refresh)

        assert shell.post_message(MediaShellResized())
        await pilot.pause()

        assert screen.query_one("#library-shell-grid", Horizontal) is stage
        assert stage.has_class("library-adaptive-compact")
        assert not stage.has_class("library-notes-compact")
        assert layout_refreshes == [True]

        assert shell.post_message(MediaShellResized())
        await pilot.pause()

        assert layout_refreshes == [True]


def _require_return_protocol():
    """Return the wished-for Task 2 protocol or fail as an intentional RED."""
    receipt_type = getattr(
        library_screen_module,
        "_LibraryMediaReturnReceipt",
        None,
    )
    settlement_type = getattr(
        library_screen_module,
        "_LibraryMediaReturnSettlement",
        None,
    )
    row_scroll_type = getattr(
        library_media_canvas_module,
        "LibraryMediaRowScroll",
        None,
    )
    geometry_type = getattr(
        library_media_canvas_module,
        "LibraryMediaRowGeometry",
        None,
    )
    geometry_message_type = getattr(
        library_media_canvas_module,
        "LibraryMediaRowGeometryChanged",
        None,
    )
    assert receipt_type is not None, "Task 2 return receipt is not implemented"
    assert settlement_type is not None, "Task 2 settlement request is not implemented"
    assert row_scroll_type is not None, "Task 2 row-scroll owner is not implemented"
    assert geometry_type is not None, "Task 2 geometry value is not implemented"
    assert geometry_message_type is not None, "Task 2 geometry message is not implemented"
    return (
        receipt_type,
        settlement_type,
        row_scroll_type,
        geometry_type,
        geometry_message_type,
    )


@pytest.mark.asyncio
async def test_viewer_return_capture_is_frozen_receipt_from_normal_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_type, *_ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-row-15")
        content_signature = screen._library_media_content_signature()
        layout_signature = screen._library_media_layout_signature()
        assert content_signature == (
            screen._library_media_browse_controller.applied_scope,
            tuple(
                str(item["id"])
                for item in screen._library_media_browse_controller.retained_items
            ),
        )
        assert layout_signature == (
            int(screen.size.width),
            int(screen.size.height),
            screen._library_notes_compact,
            screen._library_media_reader_preferences,
            screen._library_media_reader_layout,
        )

        capture_views: list[tuple[str, str]] = []
        real_content_signature = screen._library_media_content_signature
        real_layout_signature = screen._library_media_layout_signature

        def capture_content_signature() -> tuple[object, ...]:
            capture_views.append(("content", screen._library_media_view))
            return real_content_signature()

        def capture_layout_signature() -> tuple[object, ...]:
            capture_views.append(("layout", screen._library_media_view))
            return real_layout_signature()

        monkeypatch.setattr(
            screen,
            "_library_media_content_signature",
            capture_content_signature,
        )
        monkeypatch.setattr(
            screen,
            "_library_media_layout_signature",
            capture_layout_signature,
        )
        row = screen.query_one("#library-media-row-15", Button)
        screen._open_library_media_viewer(str(row.media_id))

        receipt = screen._library_media_viewer_return
        assert type(receipt) is receipt_type
        assert receipt.stable_id == row.media_id
        assert receipt.content_signature == content_signature
        assert receipt.layout_signature == layout_signature
        assert receipt.final_focus_policy == "row"
        assert receipt.final_focus_identity is None
        assert capture_views == [("content", "list"), ("layout", "list")]
        with pytest.raises(dataclasses.FrozenInstanceError):
            receipt.stable_id = "replacement"


@pytest.mark.asyncio
async def test_trash_return_capture_is_frozen_control_receipt_from_normal_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_type, *_ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-trash-open")
        trash_button = screen.query_one("#library-media-trash-open", Button)
        trash_button.focus()
        content_signature = screen._library_media_content_signature()
        layout_signature = screen._library_media_layout_signature()
        capture_views: list[str] = []
        real_content_signature = screen._library_media_content_signature
        real_layout_signature = screen._library_media_layout_signature

        def capture_content_signature() -> tuple[object, ...]:
            capture_views.append(screen._library_media_view)
            return real_content_signature()

        def capture_layout_signature() -> tuple[object, ...]:
            capture_views.append(screen._library_media_view)
            return real_layout_signature()

        monkeypatch.setattr(
            screen,
            "_library_media_content_signature",
            capture_content_signature,
        )
        monkeypatch.setattr(
            screen,
            "_library_media_layout_signature",
            capture_layout_signature,
        )
        trash_button.press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-canvas")

        receipt = screen._library_media_trash_return
        assert type(receipt) is receipt_type
        assert receipt.content_signature == content_signature
        assert receipt.layout_signature == layout_signature
        assert receipt.final_focus_policy == "control"
        assert receipt.final_focus_identity == "library-media-trash-open"
        assert capture_views == ["list", "list"]


@pytest.mark.asyncio
async def test_media_row_geometry_publishes_distinct_resize_payloads() -> None:
    (
        _,
        _,
        row_scroll_type,
        geometry_type,
        geometry_message_type,
    ) = _require_return_protocol()

    class GeometryHarness(App[None]):
        messages: list[object]

        def __init__(self) -> None:
            super().__init__()
            self.messages = []

        def compose(self) -> ComposeResult:
            yield row_scroll_type(id="geometry-owner")

        @on(geometry_message_type)
        def capture_geometry(self, message) -> None:
            message.stop()
            self.messages.append(message)

    app = GeometryHarness()
    async with app.run_test(size=(40, 12)) as pilot:
        owner = app.query_one("#geometry-owner", row_scroll_type)
        app.messages.clear()
        initial_revision = (
            owner.latest_geometry.revision
            if owner.latest_geometry is not None
            else 0
        )
        first_event = events.Resize(
            Size(17, 5),
            Size(17, 11),
            Size(19, 7),
        )
        owner.on_resize(first_event)
        owner.on_resize(first_event)
        owner.on_resize(
            events.Resize(
                Size(18, 5),
                Size(18, 13),
                Size(20, 7),
            )
        )
        await pilot.pause()

        assert len(app.messages) == 2
        first, second = app.messages
        assert type(first) is geometry_message_type
        assert first.owner is owner
        assert first.geometry == geometry_type(
            revision=initial_revision + 1,
            size=Size(17, 5),
            virtual_size=Size(17, 11),
            container_size=Size(19, 7),
        )
        assert type(second) is geometry_message_type
        assert second.owner is owner
        assert second.geometry.revision == initial_revision + 2
        assert second.geometry.size == Size(18, 5)
        assert second.geometry.virtual_size == Size(18, 13)
        assert second.geometry.container_size == Size(20, 7)
        assert owner.latest_geometry is second.geometry


@pytest.mark.asyncio
async def test_viewer_return_waits_for_geometry_then_scrolls_before_row_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        settlement_type,
        row_scroll_type,
        _,
        geometry_message_type,
    ) = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, media_id, scroll_offset = await _open_scrolled_compact_media_viewer(
            host,
            pilot,
        )
        assert scroll_offset == (0, 42)
        real_on_resize = row_scroll_type.on_resize
        monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)

        screen.query_one("#library-media-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-row-scroll")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None,
            message="Viewer return never armed an immutable settlement request.",
        )
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        request = screen._library_media_return_settlement
        assert type(request) is settlement_type
        assert request.owner_identity == id(owner)
        assert getattr(screen.focused, "media_id", None) != media_id
        assert screen._library_media_last_exact_settlement is None

        focus_observations: list[tuple[int, int]] = []
        real_set_focus = screen.set_focus

        def observe_final_focus(widget, *args, **kwargs):
            if getattr(widget, "media_id", None) == media_id:
                focus_observations.append(
                    (int(owner.scroll_x), int(owner.scroll_y))
                )
            return real_set_focus(widget, *args, **kwargs)

        monkeypatch.setattr(screen, "set_focus", observe_final_focus)
        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_exact_settlement is not None,
            message="Current-owner geometry did not settle the viewer return.",
        )

        assert (int(owner.scroll_x), int(owner.scroll_y)) == (0, 42)
        assert getattr(screen.focused, "media_id", None) == media_id
        assert focus_observations == [(0, 42)]
        settled_request, settled_revision = (
            screen._library_media_last_exact_settlement
        )
        assert settled_request.request_id == request.request_id
        assert settled_request.owner_identity == id(owner)
        assert settled_revision == owner.latest_geometry.revision
        assert screen._library_media_return_settlement is None

        duplicate = geometry_message_type(owner, owner.latest_geometry)
        assert screen.post_message(duplicate)
        await pilot.pause()
        assert screen._library_media_last_exact_settlement == (
            settled_request,
            settled_revision,
        )
        assert focus_observations == [(0, 42)]
        assert (int(owner.scroll_x), int(owner.scroll_y)) == (0, 42)


@pytest.mark.asyncio
async def test_authoritative_recompose_rearms_before_replacement_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, settlement_type, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, media_id, scroll_offset = await _open_scrolled_compact_media_viewer(
            host,
            pilot,
        )
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_exact_settlement is not None,
            message="Initial viewer return did not settle from owner geometry.",
        )
        initial_owner = screen.query_one(
            "#library-media-row-scroll",
            row_scroll_type,
        )
        initial_request = screen._library_media_last_exact_settlement[0]

        real_on_resize = row_scroll_type.on_resize
        monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)
        screen.refresh(recompose=True)
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-row-scroll"))
            and screen.query_one("#library-media-row-scroll", row_scroll_type)
            is not initial_owner,
            message="Authoritative recompose did not replace the Media owner.",
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None,
            message="Replacement owner never received fresh settlement authority.",
        )
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        request = screen._library_media_return_settlement
        assert type(request) is settlement_type
        assert request.request_id > initial_request.request_id
        assert request.owner_identity == id(owner)
        assert owner.latest_geometry is None
        assert getattr(screen.focused, "media_id", None) != media_id

        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_exact_settlement is not None
            and screen._library_media_last_exact_settlement[0].request_id
            == request.request_id,
            message="Replacement-owner geometry did not settle the return.",
        )

        assert (int(owner.scroll_x), int(owner.scroll_y)) == scroll_offset == (0, 42)
        assert getattr(screen.focused, "media_id", None) == media_id


@pytest.mark.asyncio
async def test_old_owner_and_below_floor_geometry_cannot_settle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _,
        settlement_type,
        row_scroll_type,
        _,
        geometry_message_type,
    ) = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-row-15")
        old_owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        await _wait_for_condition(
            pilot,
            lambda: old_owner.latest_geometry is not None,
            message="Original Media owner never published concrete geometry.",
        )
        old_geometry = old_owner.latest_geometry
        row = screen.query_one("#library-media-row-15", Button)
        row.focus()
        row.scroll_visible(animate=False, force=True, immediate=True)
        media_id = str(row.media_id)
        row.press()
        await _wait_for_selector(screen, pilot, "#library-media-back")

        real_on_resize = row_scroll_type.on_resize
        monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-row-scroll")
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None,
            message="Viewer return never armed its replacement-owner request.",
        )
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        assert owner is not old_owner
        assert getattr(screen.focused, "media_id", None) != media_id

        assert screen.post_message(geometry_message_type(old_owner, old_geometry))
        await pilot.pause()
        assert screen._library_media_last_exact_settlement is None
        assert getattr(screen.focused, "media_id", None) != media_id

        held: list[object] = []
        real_post_message = owner.post_message

        def hold_geometry(message) -> bool:
            if type(message) is geometry_message_type:
                held.append(message)
                return True
            return real_post_message(message)

        monkeypatch.setattr(owner, "post_message", hold_geometry)
        real_on_resize(
            owner,
            events.Resize(owner.size, owner.virtual_size, owner.container_size),
        )
        assert len(held) == 1
        geometry = owner.latest_geometry
        stage = screen.query_one("#library-shell-grid", Horizontal)
        stage.set_class(True, "library-notes-compact")
        assert screen._project_library_media_stage_classes(stage)
        current_request = screen._library_media_return_settlement
        assert type(current_request) is settlement_type
        assert current_request.presentation_epoch == (
            screen._library_media_presentation_epoch
        )
        assert current_request.exclusive_geometry_floor == geometry.revision

        assert screen.post_message(held[0])
        await pilot.pause()
        assert screen._library_media_last_exact_settlement is None
        assert getattr(screen.focused, "media_id", None) != media_id
