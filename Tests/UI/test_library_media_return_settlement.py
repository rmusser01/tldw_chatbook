"""Media adaptive-stage presentation regressions for return settlement."""

from __future__ import annotations

import dataclasses
from typing import get_type_hints

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


def _require_terminal_outcome_type():
    """Return Task 3's transient terminal-outcome alias or fail RED."""
    outcome_type = getattr(
        library_screen_module,
        "_LibraryMediaSettlementOutcome",
        None,
    )
    assert outcome_type is not None, "Task 3 settlement outcomes are not implemented"
    return outcome_type


def test_settlement_request_uses_concrete_current_tree_types() -> None:
    _, settlement_type, *_ = _require_return_protocol()

    annotations = get_type_hints(settlement_type)

    assert annotations["route_identity"] == tuple[object, ...]
    assert annotations["shell_identity"] is int
    assert annotations["items_host_identity"] is int
    assert annotations["owner_identity"] is int


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
            library_screen_module.resolve_media_reader_layout(
                screen._library_media_reader_layout.reader_width,
                screen._library_media_reader_preferences,
            ),
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

        screen._disarm_library_list_entry_focus()
        assert screen._library_media_last_exact_settlement is None


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


async def _open_pending_viewer_return(
    host,
    pilot,
    monkeypatch: pytest.MonkeyPatch,
    row_scroll_type,
):
    """Return a viewer Back request whose replacement owner has no geometry."""
    screen, media_id, scroll_offset = await _open_scrolled_compact_media_viewer(
        host,
        pilot,
    )
    real_on_resize = row_scroll_type.on_resize
    monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)
    screen.query_one("#library-media-back", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-row-scroll")
    await _wait_for_condition(
        pilot,
        lambda: screen._library_media_return_settlement is not None,
        message="Viewer return never armed without geometry.",
    )
    owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
    assert owner.latest_geometry is None
    return screen, owner, media_id, scroll_offset, real_on_resize


def _hold_next_owner_geometry(
    monkeypatch: pytest.MonkeyPatch,
    owner,
    row_scroll_type,
    geometry_message_type,
    real_on_resize,
):
    """Produce concrete owner geometry without delivering its custom message."""
    held: list[object] = []
    real_post_message = owner.post_message

    def hold_geometry(message) -> bool:
        if type(message) is geometry_message_type:
            held.append(message)
            return True
        return real_post_message(message)

    monkeypatch.setattr(owner, "post_message", hold_geometry)
    monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
    real_on_resize(
        owner,
        events.Resize(owner.size, owner.virtual_size, owner.container_size),
    )
    assert len(held) == 1
    assert owner.latest_geometry is held[0].geometry
    return held[0]


@pytest.mark.asyncio
async def test_failed_geometry_commit_rolls_back_and_same_revision_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, media_id, scroll_offset, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        geometry = _hold_next_owner_geometry(
            monkeypatch,
            owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        target = next(
            row
            for row in screen.query(".library-media-row")
            if str(getattr(row, "media_id", "") or "") == media_id
        )
        pre_scroll = (int(owner.scroll_x), int(owner.scroll_y))
        pre_focus = screen.focused
        desired_scroll_commits = 0
        target_focus_attempts = 0
        real_scroll_to = owner.scroll_to
        real_set_focus = screen.set_focus

        def observe_scroll_to(*args, **kwargs):
            nonlocal desired_scroll_commits
            if (kwargs.get("x"), kwargs.get("y")) == scroll_offset:
                desired_scroll_commits += 1
            return real_scroll_to(*args, **kwargs)

        def perturb_first_target_focus(widget, *args, **kwargs):
            nonlocal target_focus_attempts
            result = real_set_focus(widget, *args, **kwargs)
            if widget is target:
                target_focus_attempts += 1
                if target_focus_attempts == 1:
                    real_scroll_to(
                        x=scroll_offset[0],
                        y=scroll_offset[1] - 1,
                        animate=False,
                        force=True,
                        immediate=True,
                    )
            return result

        monkeypatch.setattr(owner, "scroll_to", observe_scroll_to)
        monkeypatch.setattr(screen, "set_focus", perturb_first_target_focus)

        receipt = screen._library_pending_list_entry_media_return
        request = screen._library_media_return_settlement
        assert receipt is not None
        assert request is not None
        screen._handle_library_media_row_geometry_changed(geometry)
        duplicate = geometry_message_type(owner, owner.latest_geometry)
        screen._handle_library_media_row_geometry_changed(duplicate)

        assert desired_scroll_commits == 1
        assert target_focus_attempts == 1
        assert screen._library_media_last_exact_settlement is None
        assert getattr(screen.focused, "media_id", None) != media_id
        assert screen.focused is pre_focus
        assert screen._library_notes_programmatic_focus_target is None
        assert (int(owner.scroll_x), int(owner.scroll_y)) == pre_scroll

        await pilot.pause()

        assert screen._library_pending_list_entry_focus is True
        assert screen._library_pending_list_entry_media_return is receipt
        assert screen._library_media_return_settlement is request
        assert screen._library_media_last_exact_settlement is None
        assert screen.focused is pre_focus
        assert (int(owner.scroll_x), int(owner.scroll_y)) == pre_scroll

        newer_messages: list[object] = []

        def capture_newer_geometry(message) -> bool:
            if type(message) is geometry_message_type:
                newer_messages.append(message)
                return True
            return row_scroll_type.post_message(owner, message)

        monkeypatch.setattr(owner, "post_message", capture_newer_geometry)
        owner.virtual_size = Size(
            owner.virtual_size.width,
            owner.virtual_size.height + 1,
        )
        real_on_resize(
            owner,
            events.Resize(owner.size, owner.virtual_size, owner.container_size),
        )
        assert len(newer_messages) == 1
        newer = newer_messages[-1]
        assert newer.geometry.revision == geometry.geometry.revision + 1

        screen._handle_library_media_row_geometry_changed(newer)
        screen._handle_library_media_row_geometry_changed(
            geometry_message_type(owner, owner.latest_geometry)
        )
        await pilot.pause()

        assert desired_scroll_commits == 2
        assert target_focus_attempts == 2
        assert screen._library_media_last_exact_settlement == (
            request,
            newer.geometry.revision,
        )
        assert getattr(screen.focused, "media_id", None) == media_id
        assert (int(owner.scroll_x), int(owner.scroll_y)) == scroll_offset


@pytest.mark.asyncio
async def test_real_compact_transition_floors_prechange_owner_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, _, _, real_on_resize = await _open_pending_viewer_return(
            host,
            pilot,
            monkeypatch,
            row_scroll_type,
        )
        queued = _hold_next_owner_geometry(
            monkeypatch,
            owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        before_epoch = screen._library_media_presentation_epoch
        assert screen._library_notes_compact is True
        identity = screen._capture_library_notes_focus_identity(stage_from_focus=True)

        screen._transition_library_notes_presentation(False, identity)

        stage = screen.query_one("#library-shell-grid", Horizontal)
        assert not stage.has_class("library-adaptive-compact")
        assert screen._library_media_presentation_epoch == before_epoch + 1
        assert screen._library_media_geometry_floor_owner_identity == id(owner)
        assert screen._library_media_geometry_floor == queued.geometry.revision
        equality_epoch = screen._library_media_presentation_epoch
        real_settlement_tree = screen._library_media_settlement_tree
        monkeypatch.setattr(
            screen,
            "_library_media_settlement_tree",
            lambda: pytest.fail("Equality projection queried settlement authority."),
        )
        screen._apply_library_notes_stage_visibility()
        assert screen._library_media_presentation_epoch == equality_epoch
        assert screen._reconcile_library_media_stage_presentation() is False
        assert screen._library_media_presentation_epoch == equality_epoch
        monkeypatch.setattr(
            screen,
            "_library_media_settlement_tree",
            real_settlement_tree,
        )

        assert screen.post_message(queued)
        await pilot.pause()
        assert screen._library_media_last_exact_settlement is None


@pytest.mark.asyncio
async def test_stale_programmatic_focus_releases_guard_before_user_refocus() -> None:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._library_notes_restoring_focus = False
        screen._mark_library_notes_user_interaction()
        target = screen.query_one("#library-media-type-filter", Button)
        live_focus = screen.query_one("#library-media-trash-open", Button)

        screen._library_notes_programmatic_focus_target = target
        screen.set_focus(target, scroll_visible=False)
        assert screen.focused is target
        screen.set_focus(live_focus, scroll_visible=False)
        assert screen.focused is live_focus

        await pilot.pause()

        assert screen.focused is live_focus
        assert screen._library_notes_programmatic_focus_target is None
        assert screen._library_media_view == "list"

        before_user_focus = screen._library_notes_focus_intent_generation
        screen.set_focus(target, scroll_visible=False)
        await pilot.pause()

        assert screen.focused is target
        assert target.has_focus
        assert screen._library_notes_focus_intent_generation == before_user_focus + 1


@pytest.mark.asyncio
async def test_live_user_row_focus_fences_earlier_queued_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, media_id, _, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        other = next(
            row
            for row in screen.query(".library-media-row")
            if str(getattr(row, "media_id", "")) != media_id
        )
        geometry_arm_focuses: list[object | None] = []
        real_arm = screen._arm_library_media_return_settlement

        def observe_geometry_arm_focus(*args, **kwargs):
            geometry_arm_focuses.append(screen.focused)
            return real_arm(*args, **kwargs)

        monkeypatch.setattr(
            screen,
            "_arm_library_media_return_settlement",
            observe_geometry_arm_focus,
        )
        other.focus(scroll_visible=False)
        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        assert owner.latest_geometry is not None

        await pilot.pause()
        assert geometry_arm_focuses == [other]
        assert screen._library_media_last_exact_settlement is None
        assert screen.focused is other
        assert screen._library_pending_list_entry_focus is False
        assert screen._library_pending_list_entry_media_return is None
        assert screen._library_media_return_settlement is None


@pytest.mark.asyncio
async def test_foreign_control_focus_cancels_pending_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, _, _, real_on_resize = await _open_pending_viewer_return(
            host,
            pilot,
            monkeypatch,
            row_scroll_type,
        )
        control = screen.query_one("#library-media-type-filter", Button)
        screen.set_focus(control, scroll_visible=False)
        screen.on_descendant_focus(events.DescendantFocus(control))

        assert screen._library_pending_list_entry_focus is False
        assert screen._library_pending_list_entry_media_return is None
        assert screen._library_media_return_settlement is None

        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        await pilot.pause()
        assert screen._library_media_last_exact_settlement is None
        assert screen.focused is control


@pytest.mark.asyncio
async def test_route_change_rejects_later_geometry_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, _, _, real_on_resize = await _open_pending_viewer_return(
            host,
            pilot,
            monkeypatch,
            row_scroll_type,
        )
        screen.query_one("#library-row-browse-notes", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_selected_row_id != "library-row-browse-media",
            message="Route did not leave Media.",
        )
        await _wait_for_condition(
            pilot,
            lambda: not owner.is_attached,
            message="Departed route did not detach its Media owner.",
        )
        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        queued = geometry_message_type(owner, owner.latest_geometry)

        assert screen.post_message(queued)
        await pilot.pause()
        assert screen._library_media_last_exact_settlement is None
        assert getattr(screen.focused, "media_id", None) is None


@pytest.mark.asyncio
async def test_detached_current_owner_rejects_later_geometry_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, _, _, real_on_resize = await _open_pending_viewer_return(
            host,
            pilot,
            monkeypatch,
            row_scroll_type,
        )
        await owner.remove()
        assert owner.is_attached is False
        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        queued = geometry_message_type(owner, owner.latest_geometry)

        assert screen.post_message(queued)
        await pilot.pause()
        assert screen._library_media_last_exact_settlement is None


@pytest.mark.asyncio
async def test_screen_unmount_revokes_complete_pending_return_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, _, _, real_on_resize = await _open_pending_viewer_return(
            host,
            pilot,
            monkeypatch,
            row_scroll_type,
        )
        queued = _hold_next_owner_geometry(
            monkeypatch,
            owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        assert screen._library_pending_list_entry_focus is True
        assert screen._library_pending_list_entry_media_return is not None
        assert screen._library_media_return_settlement is not None
        assert screen._library_list_entry_focus_timer is not None

        await host.pop_screen()

        assert host.screen is not screen
        assert screen.is_attached is False
        assert screen._library_pending_list_entry_focus is False
        assert screen._library_pending_list_entry_media_return is None
        assert screen._library_media_return_settlement is None
        assert screen._library_list_entry_focus_timer is None
        screen._handle_library_media_row_geometry_changed(queued)
        assert screen._library_media_last_exact_settlement is None


@pytest.mark.asyncio
async def test_screen_unmount_clears_retained_exact_settlement_proof() -> None:
    _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, _, _ = await _open_scrolled_compact_media_viewer(host, pilot)
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_exact_settlement is not None,
            message="Viewer return did not retain exact proof before unmount.",
        )
        assert screen._library_pending_list_entry_focus is True
        assert screen._library_list_entry_focus_timer is not None

        await host.pop_screen()

        assert screen._library_media_last_exact_settlement is None
        assert screen._library_pending_list_entry_focus is False
        assert screen._library_pending_list_entry_media_return is None
        assert screen._library_list_entry_focus_timer is None


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


@pytest.mark.asyncio
async def test_trash_back_exact_scroll_precedes_captured_control_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trash Back has one scroll-then-control commit and retains selection."""
    _require_terminal_outcome_type()
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-row-15")
        selected = screen.query_one("#library-media-row-15", Button)
        selected_id = str(selected.media_id)
        screen._selected_media_id = selected_id
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        owner.scroll_to(y=42, animate=False, force=True, immediate=True)
        scroll_offset = (int(owner.scroll_x), int(owner.scroll_y))
        assert scroll_offset == (0, 42)
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        opener.press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-back")

        captured_control_scrolls: list[tuple[int, int]] = []
        real_set_focus = screen.set_focus

        def observe_control_focus(widget, *args, **kwargs):
            if getattr(widget, "id", None) == "library-media-trash-open":
                current_owner = screen.query_one(
                    "#library-media-row-scroll",
                    row_scroll_type,
                )
                captured_control_scrolls.append(
                    (int(current_owner.scroll_x), int(current_owner.scroll_y))
                )
            return real_set_focus(widget, *args, **kwargs)

        monkeypatch.setattr(screen, "set_focus", observe_control_focus)
        screen.query_one("#library-media-trash-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_settlement_outcome is not None,
            message="Trash Back never published its terminal settlement outcome.",
        )

        replacement_owner = screen.query_one(
            "#library-media-row-scroll",
            row_scroll_type,
        )
        request_id, outcome, geometry_revision = (
            screen._library_media_last_settlement_outcome
        )
        assert request_id > 0
        assert outcome == "exact-settled"
        assert geometry_revision == replacement_owner.latest_geometry.revision
        assert captured_control_scrolls == [scroll_offset]
        assert screen.focused.id == "library-media-trash-open"
        assert screen._selected_media_id == selected_id
        assert (int(replacement_owner.scroll_x), int(replacement_owner.scroll_y)) == (
            scroll_offset
        )
        assert not hasattr(screen, "_restore_library_media_trash_return_focus")


@pytest.mark.asyncio
async def test_unavailable_trash_control_uses_exact_scroll_row_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layout-proven unavailable control cannot report full exact success."""
    _require_terminal_outcome_type()
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-row-15")
        selected = screen.query_one("#library-media-row-15", Button)
        selected_id = str(selected.media_id)
        screen._selected_media_id = selected_id
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        owner.scroll_to(y=42, animate=False, force=True, immediate=True)
        scroll_offset = (int(owner.scroll_x), int(owner.scroll_y))
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        opener.press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-back")

        real_on_resize = row_scroll_type.on_resize
        monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)
        screen.query_one("#library-media-trash-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None,
            message="Trash Back did not arm control-policy settlement authority.",
        )
        old_request = screen._library_media_return_settlement
        assert old_request is not None
        await pilot.resize_terminal(80, 24)
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None
            and screen._library_media_return_settlement.request_id
            > old_request.request_id,
            message="Responsive layout did not arm fresh control-policy authority.",
        )
        request = screen._library_media_return_settlement
        assert request is not None
        replacement_owner = screen.query_one(
            "#library-media-row-scroll",
            row_scroll_type,
        )
        target_control = screen.query_one("#library-media-trash-open", Button)
        target_control.disabled = True
        assert screen._library_media_last_settlement_outcome is None
        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        real_on_resize(
            replacement_owner,
            events.Resize(
                replacement_owner.size,
                replacement_owner.virtual_size,
                replacement_owner.container_size,
            ),
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_settlement_outcome is not None,
            message="Disabled captured control never reached its row fallback.",
        )

        assert screen._library_media_last_settlement_outcome[1] == (
            "exact-scroll-focus-fallback"
        )
        assert screen._library_media_last_settlement_outcome[0] == request.request_id
        assert getattr(screen.focused, "media_id", None) == selected_id
        assert (int(replacement_owner.scroll_x), int(replacement_owner.scroll_y)) == (
            scroll_offset
        )
        assert screen._selected_media_id == selected_id


@pytest.mark.asyncio
async def test_authoritative_content_revision_clamps_once_and_labels_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applied revision rejects old authority, then fresh geometry clamps once."""
    _require_terminal_outcome_type()
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, old_owner, media_id, _scroll_offset, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        old_request = screen._library_media_return_settlement
        assert old_request is not None
        old_geometry = _hold_next_owner_geometry(
            monkeypatch,
            old_owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)

        controller = screen._library_media_browse_controller
        service = app.media_reading_scope_service
        removed_id = next(
            str(item["id"])
            for item in controller.retained_items
            if str(item["id"]) != media_id
        )
        removed_backing = int(removed_id.rsplit(":", 1)[1])
        service.media_items = [
            item
            for index, item in enumerate(service.media_items)
            if service._backing_id(item, index) != removed_backing
        ]
        screen._request_library_media_browse(
            controller.mutation_refresh_scope,
            focus_identity=None,
        )
        await _wait_for_condition(
            pilot,
            lambda: not controller.loading
            and screen._library_media_content_signature()
            != old_request.content_signature,
            message="Authoritative reordered Media content never applied.",
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None
            and screen._library_media_return_settlement.request_id
            > old_request.request_id,
            message="Revised current tree never received fresh immutable authority.",
        )
        request = screen._library_media_return_settlement
        assert request is not None
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        assert request.content_signature == screen._library_media_content_signature()
        assert request.layout_signature == screen._library_media_layout_signature()
        assert not screen._settle_library_media_return_from_geometry(
            old_request,
            old_owner,
            old_geometry.geometry,
        )
        assert screen._library_media_last_settlement_outcome is None

        clamped_commits = 0
        real_scroll_to = owner.scroll_to

        def observe_clamped_scroll(*args, **kwargs):
            nonlocal clamped_commits
            if kwargs.get("immediate"):
                clamped_commits += 1
            return real_scroll_to(*args, **kwargs)

        monkeypatch.setattr(owner, "scroll_to", observe_clamped_scroll)
        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_settlement_outcome is not None,
            message="Fresh revised request did not reach its clamped commit.",
        )
        outcome = screen._library_media_last_settlement_outcome
        assert outcome == (
            request.request_id,
            "clamped-after-revision",
            owner.latest_geometry.revision,
        )
        assert clamped_commits == 1
        assert getattr(screen.focused, "media_id", None) == media_id
        assert screen._library_pending_list_entry_focus is False
        assert screen._library_media_return_settlement is None

        request_counter = screen._library_media_return_request_id
        owner_before_recompose = owner
        screen.refresh(recompose=True)
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-row-scroll"))
            and screen.query_one("#library-media-row-scroll", row_scroll_type)
            is not owner_before_recompose,
            message="Terminal revision proof did not cross a mounted recompose.",
        )
        await pilot.pause()

        assert screen._library_media_return_request_id == request_counter
        assert screen._library_media_return_settlement is None
        assert screen._library_media_last_settlement_outcome == outcome
        assert clamped_commits == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("signature_kind", ("content", "layout"))
async def test_existing_request_rejects_live_signature_drift(
    monkeypatch: pytest.MonkeyPatch,
    signature_kind: str,
) -> None:
    """An immutable request cannot consume geometry after either signature drifts."""
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, _media_id, _scroll_offset, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        request = screen._library_media_return_settlement
        assert request is not None
        geometry = _hold_next_owner_geometry(
            monkeypatch,
            owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        signature_method = f"_library_media_{signature_kind}_signature"
        original_signature = getattr(screen, signature_method)
        monkeypatch.setattr(
            screen,
            signature_method,
            lambda: original_signature() + (("drift", signature_kind),),
        )
        scroll_commits = 0
        real_scroll_to = owner.scroll_to

        def observe_scroll(*args, **kwargs):
            nonlocal scroll_commits
            if kwargs.get("immediate"):
                scroll_commits += 1
            return real_scroll_to(*args, **kwargs)

        monkeypatch.setattr(owner, "scroll_to", observe_scroll)
        screen._handle_library_media_row_geometry_changed(geometry)
        await pilot.pause()

        assert screen._library_media_return_settlement is None
        assert screen._library_media_last_settlement_outcome is None
        assert screen._library_media_last_successful_settlement is None
        assert scroll_commits == 0


@pytest.mark.asyncio
async def test_deadline_uses_one_current_geometry_fallback_and_never_requeues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ABA-bound deadline is one terminal liveness action, not readiness."""
    _require_terminal_outcome_type()
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, media_id, _scroll_offset, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        geometry = _hold_next_owner_geometry(
            monkeypatch,
            owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        request = screen._library_media_return_settlement
        assert request is not None
        rearms: list[int] = []
        real_arm = screen._arm_library_media_return_settlement

        def observe_rearm(*args, **kwargs):
            rearms.append(request.request_id)
            return real_arm(*args, **kwargs)

        monkeypatch.setattr(
            screen,
            "_arm_library_media_return_settlement",
            observe_rearm,
        )
        focus_attempts = 0
        real_set_focus = screen.set_focus

        def observe_focus(widget, *args, **kwargs):
            nonlocal focus_attempts
            if getattr(widget, "media_id", None) == media_id:
                focus_attempts += 1
            return real_set_focus(widget, *args, **kwargs)

        monkeypatch.setattr(screen, "set_focus", observe_focus)
        screen._expire_library_media_return_settlement(request.request_id)
        screen._expire_library_media_return_settlement(request.request_id)

        assert screen._library_media_last_settlement_outcome == (
            request.request_id,
            "clamped-after-settlement-failure",
            geometry.geometry.revision,
        )
        assert focus_attempts == 1
        assert rearms == []
        assert screen._library_media_return_settlement is None
        assert screen._library_pending_list_entry_focus is False
        assert screen._library_pending_list_entry_media_return is None
        assert screen._library_list_entry_focus_timer is None


@pytest.mark.asyncio
async def test_deadline_without_geometry_fails_once_with_metadata_only_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No eligible geometry yields one warning and no semantic content leak."""
    _require_terminal_outcome_type()
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, _owner, media_id, _scroll_offset, _real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        request = screen._library_media_return_settlement
        assert request is not None
        notices: list[tuple[str, str | None]] = []

        def capture_notice(message: str, *, severity: str | None = None, **_kwargs):
            notices.append((message, severity))

        monkeypatch.setattr(app, "notify", capture_notice)
        screen._expire_library_media_return_settlement(request.request_id)
        screen._expire_library_media_return_settlement(request.request_id)

        assert screen._library_media_last_settlement_outcome == (
            request.request_id,
            "layout-settlement-failed",
            None,
        )
        assert len(notices) == 1
        assert notices[0][1] == "warning"
        assert media_id not in notices[0][0]
        assert screen._library_media_return_settlement is None
        assert screen._library_pending_list_entry_focus is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale_fence",
    ("request", "compose", "lifecycle", "focus", "trash", "items"),
)
async def test_stale_request_generation_and_subview_fences_cannot_settle(
    monkeypatch: pytest.MonkeyPatch,
    stale_fence: str,
) -> None:
    """Each mutable authority fence rejects a real current-owner geometry."""
    _require_terminal_outcome_type()
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, media_id, _scroll_offset, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        request = screen._library_media_return_settlement
        assert request is not None
        geometry = _hold_next_owner_geometry(
            monkeypatch,
            owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )

        if stale_fence == "request":
            screen._library_media_return_settlement = dataclasses.replace(
                request,
                request_id=request.request_id + 1,
            )
            screen._settle_library_media_return_from_geometry(
                request,
                owner,
                geometry.geometry,
            )
        else:
            if stale_fence == "compose":
                screen._library_compose_generation += 1
            elif stale_fence == "lifecycle":
                screen._library_media_lifecycle_generation += 1
            elif stale_fence == "focus":
                screen._library_notes_focus_intent_generation += 1
            elif stale_fence == "trash":
                screen._library_media_view = "trash"
            elif stale_fence == "items":
                screen._library_media_reader_layout = dataclasses.replace(
                    screen._library_media_reader_layout,
                    items_open=False,
                )
            screen._handle_library_media_row_geometry_changed(geometry)
        await pilot.pause()

        assert screen._library_media_last_settlement_outcome is None
        assert screen._library_media_last_exact_settlement is None
        assert getattr(screen.focused, "media_id", None) != media_id


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_field", ("shell_identity", "items_host_identity"))
async def test_replaced_shell_and_items_identities_reject_current_geometry(
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    """Each replaced tree identity independently fences an otherwise-live request."""
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, owner, _media_id, _scroll_offset, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        request = screen._library_media_return_settlement
        assert request is not None
        geometry = _hold_next_owner_geometry(
            monkeypatch,
            owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        replacement_request = dataclasses.replace(
            request,
            **{identity_field: getattr(request, identity_field) + 1},
        )
        screen._library_media_return_settlement = replacement_request

        assert not screen._settle_library_media_return_from_geometry(
            replacement_request,
            owner,
            geometry.geometry,
        )
        assert screen._library_media_last_settlement_outcome is None
        assert screen._library_media_last_successful_settlement is None


@pytest.mark.asyncio
async def test_deadline_with_failed_nongeometry_fence_clears_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expiry after authority loss emits neither fallback outcome nor warning."""
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, _owner, _media_id, _scroll_offset, _real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        request = screen._library_media_return_settlement
        assert request is not None
        notices: list[tuple[str, str | None]] = []

        def capture_notice(message: str, *, severity: str | None = None, **_kwargs):
            notices.append((message, severity))

        monkeypatch.setattr(app, "notify", capture_notice)
        screen._library_notes_focus_intent_generation += 1
        screen._expire_library_media_return_settlement(request.request_id)

        assert notices == []
        assert screen._library_media_last_settlement_outcome is None
        assert screen._library_pending_list_entry_focus is False
        assert screen._library_pending_list_entry_media_return is None
        assert screen._library_media_return_settlement is None
        assert screen._library_list_entry_focus_timer is None


@pytest.mark.asyncio
async def test_another_viewer_back_request_invalidates_prior_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second mounted viewer round trip supersedes the first request."""
    _, _, row_scroll_type, _, geometry_message_type = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, old_owner, media_id, _scroll_offset, real_on_resize = (
            await _open_pending_viewer_return(
                host,
                pilot,
                monkeypatch,
                row_scroll_type,
            )
        )
        old_request = screen._library_media_return_settlement
        assert old_request is not None
        old_geometry = _hold_next_owner_geometry(
            monkeypatch,
            old_owner,
            row_scroll_type,
            geometry_message_type,
            real_on_resize,
        )
        monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)
        row = next(
            row
            for row in screen.query(".library-media-row")
            if str(getattr(row, "media_id", "") or "") == media_id
        )
        row.press()
        await _wait_for_selector(screen, pilot, "#library-media-back")
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None
            and screen._library_media_return_settlement.request_id
            > old_request.request_id,
            message="Second Back did not supersede the earlier settlement request.",
        )
        new_request = screen._library_media_return_settlement
        assert new_request is not None

        assert not screen._settle_library_media_return_from_geometry(
            old_request,
            old_owner,
            old_geometry.geometry,
        )
        assert screen._library_media_return_settlement is new_request
        assert screen._library_media_last_settlement_outcome is None


@pytest.mark.asyncio
async def test_post_exact_user_takeover_prevents_recompose_renewal() -> None:
    """After exact success, real user focus ends the outer recompose authority."""
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen, _media_id, _scroll_offset = await _open_scrolled_compact_media_viewer(
            host,
            pilot,
        )
        screen.query_one("#library-media-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_last_settlement_outcome is not None,
            message="Initial exact return never settled.",
        )
        assert screen._library_media_last_settlement_outcome[1] == "exact-settled"
        await pilot.pause()
        control = screen.query_one("#library-media-type-filter", Button)
        control.focus(scroll_visible=False)
        await pilot.pause()
        assert screen.focused is control
        assert screen._library_pending_list_entry_focus is False
        request_counter = screen._library_media_return_request_id
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)

        screen.refresh(recompose=True)
        await _wait_for_condition(
            pilot,
            lambda: bool(screen.query("#library-media-row-scroll"))
            and screen.query_one("#library-media-row-scroll", row_scroll_type) is not owner,
            message="User-takeover inverse did not cross a real recompose.",
        )
        await pilot.pause()

        assert screen._library_media_return_request_id == request_counter
        assert screen._library_media_return_settlement is None
        assert screen._library_pending_list_entry_focus is False


@pytest.mark.asyncio
async def test_trash_capture_uses_opener_identity_without_prefocus() -> None:
    """Trash captures its semantic opener, never arbitrary current focus."""
    receipt_type, *_ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        unrelated = screen.query_one("#library-media-type-filter", Button)
        unrelated.focus(scroll_visible=False)
        await pilot.pause()
        assert screen.focused is unrelated

        screen.query_one("#library-media-trash-open", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-canvas")

        receipt = screen._library_media_trash_return
        assert type(receipt) is receipt_type
        assert receipt.final_focus_policy == "control"
        assert receipt.final_focus_identity == "library-media-trash-open"
        assert receipt.final_focus_identity != unrelated.id


@pytest.mark.asyncio
@pytest.mark.parametrize("semantic_row_state", ("absent", "mismatched"))
async def test_control_exact_return_requires_matching_semantic_row(
    monkeypatch: pytest.MonkeyPatch,
    semantic_row_state: str,
) -> None:
    """A valid captured control cannot bypass exact semantic-row authority."""
    _, _, row_scroll_type, _, _ = _require_return_protocol()
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items())
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=COMPACT_SCROLL_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-row-15")
        semantic_row = screen.query_one("#library-media-row-15", Button)
        selected_id = str(semantic_row.media_id)
        screen._selected_media_id = selected_id
        original_owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        original_owner.scroll_to(y=42, animate=False, force=True, immediate=True)
        opener = screen.query_one("#library-media-trash-open", Button)
        opener.focus()
        opener.press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-back")

        real_on_resize = row_scroll_type.on_resize
        monkeypatch.setattr(row_scroll_type, "on_resize", lambda _owner, _event: None)
        screen.query_one("#library-media-trash-back", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._library_media_return_settlement is not None,
            message="Trash return did not arm control-policy authority.",
        )
        owner = screen.query_one("#library-media-row-scroll", row_scroll_type)
        current_row = next(
            row
            for row in screen.query(".library-media-row")
            if str(getattr(row, "media_id", "") or "") == selected_id
        )
        if semantic_row_state == "absent":
            await current_row.remove()
        else:
            current_row.media_id = f"mismatched:{selected_id}"
        assert not any(
            str(getattr(row, "media_id", "") or "") == selected_id
            for row in screen.query(".library-media-row")
        )
        scroll_commits = 0
        real_scroll_to = owner.scroll_to

        def observe_scroll(*args, **kwargs):
            nonlocal scroll_commits
            if kwargs.get("immediate"):
                scroll_commits += 1
            return real_scroll_to(*args, **kwargs)

        monkeypatch.setattr(owner, "scroll_to", observe_scroll)
        monkeypatch.setattr(row_scroll_type, "on_resize", real_on_resize)
        owner.on_resize(
            events.Resize(owner.size, owner.virtual_size, owner.container_size)
        )
        await pilot.pause()

        assert screen._library_media_last_settlement_outcome is None
        assert screen._library_media_last_successful_settlement is None
        assert getattr(screen.focused, "id", None) != "library-media-trash-open"
        assert scroll_commits == 0
