"""Paint, geometry, input, and focus contracts for Persona Buddy."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Container
from textual.geometry import Offset
from textual.screen import ModalScreen, Screen
from textual.widgets import Button, Input, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Persona_Buddy import (
    PersonaBuddyGeometry,
    PersonaBuddyPreferences,
    PersonaBuddySelection,
    PersonaBuddySnapshot,
    PersonaBuddyVisualSnapshot,
)
from tldw_chatbook.Widgets.Persona_Widgets.persona_buddy_widget import (
    PersonaBuddyWidget,
)


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            raise AssertionError("predicate did not become true before deadline")
        await asyncio.sleep(0.01)


class _FakeController:
    def __init__(
        self,
        *,
        size: tuple[int, int] = (28, 12),
        collapsed: bool = False,
        animate: bool = False,
        frames: tuple[str, ...] = ("BUDDY-A",),
        durations: tuple[int, ...] | None = None,
        loop: bool = True,
    ) -> None:
        self.preferences = PersonaBuddyPreferences(
            enabled=True,
            selection=PersonaBuddySelection("local", "persona-1"),
            collapsed=collapsed,
            geometry=PersonaBuddyGeometry(
                x=1_000_000,
                y=1_000_000,
                width=size[0],
                height=size[1],
            ),
        )
        self.generation = 1
        self.persisted: list[PersonaBuddyPreferences] = []
        self.resolve_sizes: list[tuple[int, int]] = []
        durations = durations or tuple(20 for _ in frames)
        prepared = tuple(
            SimpleNamespace(renderable=Text(label), duration_ms=duration)
            for label, duration in zip(frames, durations, strict=True)
        )
        self.visual = PersonaBuddyVisualSnapshot(
            available=True,
            reason=None,
            source="local",
            persona_id="persona-1",
            persona_revision=1,
            requested_state="idle",
            resolved_state="idle",
            animation_id="idle",
            graph_identity=None,
            cache_identity=None,
            frames=prepared,  # type: ignore[arg-type]
            frame_rate=50.0,
            loop=loop,
            animate=animate,
        )

    def current_preferences(self) -> PersonaBuddyPreferences:
        return self.preferences

    def snapshot(self) -> PersonaBuddySnapshot:
        return PersonaBuddySnapshot(
            generation=self.generation,
            selection=self.preferences.selection,
            state="idle",
            enabled=self.preferences.enabled,
            open=self.preferences.open,
            collapsed=self.preferences.collapsed,
            preferences_generation=self.generation,
            visual=self.visual,
        )

    async def update_preferences(
        self, preferences: PersonaBuddyPreferences
    ) -> PersonaBuddySnapshot:
        self.preferences = preferences
        self.generation += 1
        self.persisted.append(preferences)
        return self.snapshot()

    def apply_preferences_patch(self, **changes: object) -> int:
        self.preferences = replace(self.preferences, **changes)
        self.generation += 1
        return self.generation

    async def persist_preferences_revision(self, revision: int):
        assert revision <= self.generation
        self.persisted.append(self.preferences)
        return self.snapshot()

    async def resolve_current_visual(self, *, cols: int, lines: int):
        assert cols > 0 and lines > 0
        self.resolve_sizes.append((cols, lines))
        return self.visual


class _BuddyScreen(Screen):
    def __init__(self, controller: _FakeController) -> None:
        super().__init__()
        self.controller = controller
        self.buddy = PersonaBuddyWidget(
            controller=controller,
            view_generation=1,
            reconcile=lambda: None,
        )

    def compose(self) -> ComposeResult:
        with Container(id="buddy-flow-probe"):
            yield Input(id="focus-probe")
            yield Static("FLOW-END", id="flow-end")
        yield self.buddy


class _BlockingResolutionController(_FakeController):
    def __init__(self) -> None:
        super().__init__()
        self.resolve_started = asyncio.Event()
        self.resolve_release = asyncio.Event()
        self.active_resolves = 0
        self.max_active_resolves = 0

    async def resolve_current_visual(self, *, cols: int, lines: int):
        self.active_resolves += 1
        self.max_active_resolves = max(self.max_active_resolves, self.active_resolves)
        self.resolve_started.set()
        try:
            await self.resolve_release.wait()
            return self.visual
        finally:
            self.active_resolves -= 1


class _FreshEquivalentVisualController(_FakeController):
    async def resolve_current_visual(self, *, cols: int, lines: int):
        self.visual = replace(self.visual)
        return self.visual


class _CountingResolutionController(_FakeController):
    def __init__(self) -> None:
        super().__init__()
        self.resolve_calls = 0

    async def resolve_current_visual(self, *, cols: int, lines: int):
        self.resolve_calls += 1
        return self.visual


class _SlotSizedFrameController(_FakeController):
    async def resolve_current_visual(self, *, cols: int, lines: int):
        assert cols > 0 and lines > 1
        self.resolve_sizes.append((cols, lines))
        labels = ["TOP"] + [f"ROW-{index}" for index in range(1, lines - 1)]
        labels.append("BOTTOM")
        frame = SimpleNamespace(
            renderable=Text("\n".join(labels), style="bold white on rgb(20,80,140)"),
            duration_ms=20,
        )
        self.visual = replace(self.visual, frames=(frame,))  # type: ignore[arg-type]
        return self.visual


class _BuddyApp(ConsolidatedCSSApp):
    CSS_PATH = BUNDLED_STYLESHEET
    CSS = """
    #buddy-flow-probe { height: 1fr; }
    #flow-end { dock: bottom; height: 1; }
    """

    def __init__(self, controller: _FakeController) -> None:
        super().__init__()
        self.buddy_screen = _BuddyScreen(controller)

    async def on_mount(self) -> None:
        await self.push_screen(self.buddy_screen)


def _compositor_text(screen: Screen) -> str:
    return "\n".join(strip.text for strip in screen._compositor.render_strips())


def _mouse(event_type, *, x: int, y: int, button: int = 1, widget=None):
    return event_type(
        widget,
        x,
        y,
        0,
        0,
        button,
        False,
        False,
        False,
        screen_x=x,
        screen_y=y,
    )


@pytest.mark.asyncio
async def test_overlay_paints_without_flow_or_fr_budget():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        flow = app.screen.query_one("#buddy-flow-probe")
        await _wait_until(lambda: buddy.region.width == 28)
        assert buddy.styles.position == "absolute"
        assert buddy.styles.overlay == "screen"
        assert flow.region.height == 24
        assert app.screen.query_one("#flow-end").region.bottom == 24
        assert "Persona Buddy" in _compositor_text(app.screen)
        assert "BUDDY-A" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_drag_and_resize_are_viewport_bounded_and_persist_once():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        drag_handle = buddy.query_one("#persona-buddy-drag-handle", Static)
        initial_geometry = buddy._clamped_geometry(controller.preferences.geometry)
        assert initial_geometry.x + initial_geometry.width <= app.size.width
        assert initial_geometry.y + initial_geometry.height <= app.size.height

        buddy.on_mouse_down(
            _mouse(
                events.MouseDown,
                x=drag_handle.region.x + 1,
                y=drag_handle.region.y + 1,
                widget=drag_handle,
            )
        )
        await asyncio.sleep(0)
        assert app.screen.focused is buddy
        buddy.on_mouse_move(_mouse(events.MouseMove, x=-50, y=-50))
        assert buddy.absolute_offset == Offset(0, 0)
        assert controller.persisted == []
        buddy.refresh_from_controller()
        assert buddy.absolute_offset == Offset(0, 0)
        buddy.on_mouse_up(_mouse(events.MouseUp, x=-50, y=-50))
        await _wait_until(lambda: len(controller.persisted) == 1)

        corner_x = buddy.region.right - 1
        corner_y = buddy.region.bottom - 1
        buddy.on_mouse_down(_mouse(events.MouseDown, x=corner_x, y=corner_y))
        buddy.on_mouse_move(_mouse(events.MouseMove, x=500, y=500))
        assert buddy.region.right <= app.size.width
        assert buddy.region.bottom <= app.size.height
        assert len(controller.persisted) == 1
        buddy.on_mouse_up(_mouse(events.MouseUp, x=500, y=500))
        await _wait_until(lambda: len(controller.persisted) == 2)
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_keyboard_move_resize_reset_collapse_close_exact_bindings():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(100, 30)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        buddy.focus(scroll_visible=False)
        start = buddy.absolute_offset
        await pilot.press("h")
        assert buddy.absolute_offset.x == max(0, start.x - 1)
        await pilot.press("j", "k", "l")
        await pilot.press("H", "J", "K", "L")
        await _wait_until(lambda: len(controller.persisted) >= 8)
        await pilot.press("0")
        assert buddy.absolute_offset == Offset(72, 18)
        await pilot.press("c")
        await _wait_until(lambda: controller.preferences.collapsed)
        await _wait_until(lambda: "Buddy" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy.region.height == 3)
        await pilot.press("x")
        await _wait_until(lambda: not controller.preferences.open)

        keys = {binding.key for binding in PersonaBuddyWidget.BINDINGS}
        assert keys == {"h", "j", "k", "l", "H", "J", "K", "L", "0", "c", "x"}
        assert not any(key.startswith("ctrl+") for key in keys)


@pytest.mark.asyncio
async def test_tiny_viewport_uses_labelled_compact_control():
    controller = _FakeController(size=(28, 12))
    app = _BuddyApp(controller)
    async with app.run_test(size=(12, 4)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.has_class("persona-buddy-compact"))
        assert buddy.region.width <= 12
        assert buddy.region.height <= 4
        assert "Buddy" in _compositor_text(app.screen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("viewport", "compact"),
    [((28, 12), False), ((18, 6), True), ((12, 4), True), ((10, 2), True)],
)
async def test_exact_cell_layouts_keep_complete_labelled_controls(
    viewport: tuple[int, int], compact: bool
):
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=viewport):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width > 0 and buddy.region.height > 0)
        assert buddy.has_class("persona-buddy-compact") is compact
        assert buddy.region.right <= viewport[0]
        assert buddy.region.bottom <= viewport[1]

        text = _compositor_text(app.screen)
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        close = buddy.query_one("#persona-buddy-close", Button)
        controls = (collapse, close) if viewport[0] >= 18 else (collapse,)
        for control in controls:
            assert control.display
            assert control.region.width >= len(str(control.label))
            assert buddy.region.contains_region(control.region)
            target, _ = app.screen.get_widget_at(
                control.region.x + control.region.width // 2,
                control.region.y + control.region.height // 2,
            )
            assert target is control

        if compact:
            assert "Buddy" in text
            assert close.display is (viewport[0] >= 18)
            if close.display:
                assert "Close" in text
        else:
            assert "Fold" in text
            assert "Close" in text
            assert "hjkl move · HJKL size" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("viewport", [(18, 6), (12, 4), (10, 2)])
async def test_compact_keyboard_move_preserves_preferred_size_on_restore(
    viewport: tuple[int, int],
):
    controller = _FakeController(size=(28, 12))
    app = _BuddyApp(controller)
    async with app.run_test(size=viewport) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.has_class("persona-buddy-compact"))
        start = buddy.absolute_offset
        buddy.focus(scroll_visible=False)

        await pilot.press("k")
        await _wait_until(lambda: len(controller.persisted) == 1)

        preferred = controller.preferences.geometry
        assert (preferred.width, preferred.height) == (28, 12)
        assert preferred.y == max(0, start.y - 1)
        await pilot.resize_terminal(28, 12)
        await _wait_until(lambda: not buddy.has_class("persona-buddy-compact"))
        assert (buddy.region.width, buddy.region.height) == (28, 12)


@pytest.mark.asyncio
async def test_drag_while_viewport_becomes_compact_preserves_preferred_size():
    controller = _FakeController(size=(28, 12))
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        handle = buddy.query_one("#persona-buddy-drag-handle", Static)
        buddy.on_mouse_down(
            _mouse(
                events.MouseDown,
                x=handle.region.x + 1,
                y=handle.region.y,
                widget=handle,
            )
        )
        assert app.mouse_captured is buddy

        await pilot.resize_terminal(12, 4)
        await _wait_until(lambda: buddy.has_class("persona-buddy-compact"))
        buddy.on_mouse_move(_mouse(events.MouseMove, x=0, y=0))
        buddy.on_mouse_up(_mouse(events.MouseUp, x=0, y=0))
        await _wait_until(lambda: len(controller.persisted) == 1)

        preferred = controller.preferences.geometry
        assert (preferred.width, preferred.height) == (28, 12)
        await pilot.resize_terminal(28, 12)
        await _wait_until(lambda: not buddy.has_class("persona-buddy-compact"))
        await _wait_until(lambda: (buddy.region.width, buddy.region.height) == (28, 12))


@pytest.mark.asyncio
async def test_compact_keyboard_resize_intentionally_updates_preferred_size():
    controller = _FakeController(size=(28, 12))
    app = _BuddyApp(controller)
    async with app.run_test(size=(10, 2)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.has_class("persona-buddy-compact"))
        buddy.focus(scroll_visible=False)

        await pilot.press("L")
        await _wait_until(lambda: len(controller.persisted) == 1)

        preferred = controller.preferences.geometry
        assert (preferred.width, preferred.height) == (29, 12)
        assert (buddy.region.width, buddy.region.height) == (10, 1)


@pytest.mark.asyncio
async def test_labelled_buttons_click_without_starting_drag_and_reopen_collapsed():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        await pilot.click("#persona-buddy-collapse")
        await _wait_until(lambda: controller.preferences.collapsed)
        assert app.mouse_captured is None
        reopen = buddy.query_one("#persona-buddy-collapse", Button)
        assert str(reopen.label) == "Open Buddy"
        assert reopen.display
        await pilot.pause(0.25)
        target, _ = app.screen.get_widget_at(
            reopen.region.x + reopen.region.width // 2,
            reopen.region.y + reopen.region.height // 2,
        )
        assert target is reopen
        await pilot.click("#persona-buddy-collapse")
        await _wait_until(lambda: not controller.preferences.collapsed)
        await pilot.click("#persona-buddy-close")
        await _wait_until(lambda: not controller.preferences.open)
        assert app.mouse_captured is None


@pytest.mark.asyncio
async def test_state_repaint_never_steals_focus():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        focus_probe = app.screen.query_one("#focus-probe", Input)
        focus_probe.focus(scroll_visible=False)
        assert app.screen.focused is focus_probe
        buddy = app.screen.query_one(PersonaBuddyWidget)
        buddy.refresh_from_controller()
        await asyncio.sleep(0)
        assert app.screen.focused is focus_probe


@pytest.mark.asyncio
async def test_animation_cells_change_then_freeze_hidden_collapsed():
    controller = _FakeController(
        animate=True,
        frames=("FRAME-A", "FRAME-B"),
        durations=(300, 200),
        loop=False,
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "FRAME-A" in _compositor_text(app.screen))
        buddy._next_frame_at = 0
        buddy.advance_frame()
        await _wait_until(lambda: "FRAME-B" in _compositor_text(app.screen))
        assert buddy._next_frame_at - asyncio.get_running_loop().time() > 0.1
        buddy._next_frame_at = 0
        buddy.advance_frame()
        assert buddy.frame_index == 1

        controller.preferences = replace(controller.preferences, collapsed=True)
        buddy.refresh_from_controller()
        frozen = buddy.frame_index
        buddy.advance_frame()
        assert buddy.frame_index == frozen
        assert buddy.snapshot_polling_active

        buddy.display = False
        buddy.advance_frame()
        assert buddy.frame_index == frozen


@pytest.mark.asyncio
async def test_reduced_motion_paints_static_frame():
    controller = _FakeController(animate=False, frames=("STATIC-A", "STATIC-B"))
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "STATIC-A" in _compositor_text(app.screen))
        buddy.advance_frame()
        assert buddy.frame_index == 0
        assert "STATIC-A" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_resolution_is_single_owned_and_stale_completion_cannot_repaint():
    controller = _BlockingResolutionController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await asyncio.wait_for(controller.resolve_started.wait(), timeout=1)
        await asyncio.sleep(0.15)
        assert controller.max_active_resolves == 1
        previous = buddy._snapshot
        buddy._current_view = lambda _candidate: False
        controller.resolve_release.set()
        await asyncio.sleep(0.05)
        assert buddy._snapshot is previous


@pytest.mark.asyncio
async def test_fresh_equivalent_snapshots_preserve_animation_deadline_and_progress():
    controller = _FreshEquivalentVisualController(
        animate=True,
        frames=("POLL-A", "POLL-B"),
        durations=(180, 260),
        loop=False,
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        await _wait_until(lambda: "POLL-A" in _compositor_text(app.screen))
        await _wait_until(lambda: "POLL-B" in _compositor_text(app.screen), timeout=0.5)
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await asyncio.sleep(0.25)
        assert buddy.frame_index == 1
        assert "POLL-B" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_resolution_runs_once_until_semantic_authority_changes():
    controller = _CountingResolutionController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        await _wait_until(lambda: controller.resolve_calls == 1)
        await asyncio.sleep(0.62)
        assert controller.resolve_calls == 1

        controller.generation += 1
        await _wait_until(lambda: controller.resolve_calls == 2, timeout=0.3)


@pytest.mark.asyncio
async def test_resolution_uses_visible_frame_content_region_not_window_region():
    controller = _SlotSizedFrameController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: bool(controller.resolve_sizes))
        expected = (frame.content_region.width, frame.content_region.height)

        assert controller.resolve_sizes[-1] == expected
        assert controller.resolve_sizes[-1] != (
            buddy.content_region.width,
            buddy.content_region.height,
        )
        await _wait_until(
            lambda: (
                "TOP" in _compositor_text(app.screen)
                and "BOTTOM" in _compositor_text(app.screen)
            )
        )
        strips = app.screen._compositor.render_strips()
        painted_frame = "\n".join(
            strips[y].text[frame.region.x : frame.region.right]
            for y in range(frame.region.y, frame.region.bottom)
        )
        assert "TOP" in painted_frame
        assert "BOTTOM" in painted_frame


@pytest.mark.asyncio
async def test_frame_slot_resize_changes_resolution_authority_once():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: len(controller.resolve_sizes) == 1)
        original = frame.content_region.size
        frame.styles.height = max(1, original.height - 1)
        await pilot.pause()
        await _wait_until(lambda: frame.content_region.size != original)

        buddy.refresh_from_controller()

        await _wait_until(lambda: len(controller.resolve_sizes) == 2)
        assert controller.resolve_sizes[-1] == (
            frame.content_region.width,
            frame.content_region.height,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("hidden", "collapsed", "zero-frame"))
async def test_hidden_collapsed_or_zero_frame_slot_does_not_resolve(mode: str):
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: len(controller.resolve_sizes) == 1)

        if mode == "hidden":
            buddy.display = False
        elif mode == "collapsed":
            controller.preferences = replace(
                controller.preferences,
                collapsed=True,
            )
        else:
            frame.styles.height = 0
            await pilot.pause()
            await _wait_until(lambda: frame.content_region.height == 0)
        controller.generation += 1
        buddy.refresh_from_controller()
        await asyncio.sleep(0.2)

        assert len(controller.resolve_sizes) == 1


@pytest.mark.asyncio
async def test_animation_uses_frame_deadline_one_shots_and_static_has_no_timer():
    animated = _FakeController(
        animate=True,
        frames=("TIMER-A", "TIMER-B"),
        durations=(180, 260),
        loop=False,
    )
    app = _BuddyApp(animated)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy._frame_timer is not None)
        assert buddy._frame_timer._repeat == 0
        assert buddy._frame_timer._interval >= 0.17
        await _wait_until(lambda: buddy.frame_index == 1, timeout=0.4)
        assert buddy._frame_timer is None

    static = _FakeController(animate=False, frames=("STATIC-A", "STATIC-B"))
    static_app = _BuddyApp(static)
    async with static_app.run_test(size=(80, 24)):
        buddy = static_app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "STATIC-A" in _compositor_text(static_app.screen))
        assert buddy._frame_timer is None


@pytest.mark.asyncio
async def test_modal_covers_buddy_hit_target():
    class _BlockingModal(ModalScreen):
        def compose(self) -> ComposeResult:
            yield Static("MODAL BLOCKER", id="modal-blocker")

    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        point = (buddy.region.x + 1, buddy.region.y + 1)
        await app.push_screen(_BlockingModal())
        target, _region = app.screen.get_widget_at(*point)
        assert not isinstance(target, PersonaBuddyWidget)
        assert "MODAL BLOCKER" in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_capture_is_released_when_widget_unmounts_mid_drag():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        buddy.on_mouse_down(
            _mouse(
                events.MouseDown,
                x=buddy.query_one("#persona-buddy-drag-handle", Static).region.x + 1,
                y=buddy.query_one("#persona-buddy-drag-handle", Static).region.y + 1,
                widget=buddy.query_one("#persona-buddy-drag-handle", Static),
            )
        )
        assert app.mouse_captured is buddy
        await buddy.remove()
        assert app.mouse_captured is None
