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
        frame_sizes: tuple[tuple[int, int], ...] | None = None,
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
        self.viewport_generation = 0
        self.persisted: list[PersonaBuddyPreferences] = []
        self.preference_writes: list[dict[str, object]] = []
        self.resolve_sizes: list[tuple[int, int]] = []
        durations = durations or tuple(20 for _ in frames)
        frame_sizes = frame_sizes or tuple((24, 10) for _ in frames)
        prepared = tuple(
            SimpleNamespace(
                cache_identity=f"cache-{index}",
                graph_identity=None,
                asset_id=index + 1,
                asset_key=f"asset-{index}",
                asset_sha256=f"sha-{index}",
                manifest_frame_index=index,
                selected_frame=0,
                duration_ms=duration,
                width=width,
                height=height,
                paint_digest=f"paint-{index}-{label}",
                renderable=Text(label),
            )
            for index, (label, duration, (width, height)) in enumerate(
                zip(frames, durations, frame_sizes, strict=True)
            )
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
            viewport_generation=self.viewport_generation,
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
        self.preference_writes.append(changes)
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


class _BlockAfterAcceptedController(_FakeController):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.block = False
        self.direct_visual = self.visual
        self.resolve_started = asyncio.Event()
        self.resolve_release = asyncio.Event()
        self.stale_resolve_completed = asyncio.Event()
        self.next_resolve_started = asyncio.Event()
        self.blocked_resolves = 0

    async def resolve_current_visual(self, *, cols: int, lines: int):
        self.resolve_sizes.append((cols, lines))
        if not self.block:
            return self.visual
        self.blocked_resolves += 1
        visual = self.direct_visual
        if self.blocked_resolves == 1:
            self.resolve_started.set()
        else:
            self.next_resolve_started.set()
        await self.resolve_release.wait()
        self.resolve_release.clear()
        if self.blocked_resolves == 1:
            self.stale_resolve_completed.set()
        return visual


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


class _ViewportGenerationController(_FakeController):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.viewport_updates: list[int] = []

    def set_viewport_generation(self, generation: int) -> int:
        self.viewport_generation = generation
        self.viewport_updates.append(generation)
        return generation


class _SlotSizedFrameController(_FakeController):
    async def resolve_current_visual(self, *, cols: int, lines: int):
        assert cols > 0 and lines > 1
        self.resolve_sizes.append((cols, lines))
        labels = ["TOP".ljust(cols - len("RIGHT"), "-") + "RIGHT"]
        labels.extend(
            f"L{index}".ljust(cols - 1, "-") + "R" for index in range(1, lines - 1)
        )
        labels.append("BOTTOM".ljust(cols - len("RIGHT"), "-") + "RIGHT")
        frame = SimpleNamespace(
            cache_identity="slot-cache",
            graph_identity=None,
            asset_id=1,
            asset_key="slot-asset",
            asset_sha256="slot-sha",
            manifest_frame_index=0,
            selected_frame=0,
            width=cols,
            height=lines * 2,
            paint_digest=f"slot-{cols}-{lines}",
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


def _point_is_inside(widget, x: int, y: int) -> bool:
    return (
        widget.region.x <= x < widget.region.right
        and widget.region.y <= y < widget.region.bottom
    )


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


def _visual_with_frame(
    controller: _FakeController,
    *,
    label: str,
    width: object,
    height: object,
) -> PersonaBuddyVisualSnapshot:
    frame = SimpleNamespace(
        cache_identity=f"cache-{label}",
        graph_identity=None,
        asset_id=101,
        asset_key=f"asset-{label}",
        asset_sha256=f"sha-{label}",
        manifest_frame_index=0,
        selected_frame=0,
        duration_ms=20,
        width=width,
        height=height,
        paint_digest=f"paint-{label}",
        renderable=Text(label),
    )
    return replace(controller.visual, frames=(frame,))  # type: ignore[arg-type]


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
        assert "Persona Buddy" not in _compositor_text(app.screen)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))


@pytest.mark.asyncio
async def test_resting_buddy_contains_pet_and_icons_without_default_words():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))

        assert [child.id for child in buddy.children] == [
            "persona-buddy-frame",
            "persona-buddy-collapse",
            "persona-buddy-close",
        ]
        for removed_id in (
            "persona-buddy-header",
            "persona-buddy-drag-handle",
            "persona-buddy-status",
            "persona-buddy-hints",
        ):
            assert not buddy.query(f"#{removed_id}")

        resting = _compositor_text(app.screen)
        assert "BUDDY-A" in resting
        for default_word in (
            "Persona Buddy",
            "Drag",
            "Fold",
            "Close",
            "State",
            "Visual pending",
            "hjkl move",
            "HJKL size",
        ):
            assert default_word not in resting


@pytest.mark.asyncio
async def test_single_frame_touches_every_inner_content_edge():
    controller = _SlotSizedFrameController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: bool(controller.resolve_sizes))
        await _wait_until(lambda: "BOTTOM" in _compositor_text(app.screen))

        assert frame.region == buddy.content_region
        assert frame.content_region == buddy.content_region
        assert controller.resolve_sizes[-1] == buddy.content_region.size
        strips = app.screen._compositor.render_strips()
        painted = [
            strips[y].text[frame.region.x : frame.region.right]
            for y in range(frame.region.y, frame.region.bottom)
        ]
        source_rows = frame.renderable.plain.splitlines()
        assert source_rows[0].startswith("TOP")
        assert source_rows[0].endswith("RIGHT")
        assert len(source_rows) == frame.content_region.height
        assert all(len(row) == frame.content_region.width for row in source_rows)
        assert painted[0].strip()
        assert painted[-1].startswith("BOTTOM")
        assert painted[-1].endswith("RIGHT")
        assert all(row.startswith(("BOTTOM", "L")) for row in painted[1:])
        assert all(row.endswith(("RIGHT", "R")) for row in painted[1:])


@pytest.mark.asyncio
async def test_icon_controls_have_exact_labels_tooltips_and_hit_regions():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        await pilot.pause()
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        close = buddy.query_one("#persona-buddy-close", Button)

        assert (str(collapse.label), collapse.tooltip) == ("▾", "Fold")
        assert (str(close.label), close.tooltip) == ("×", "Close")
        assert collapse.region.width >= 3
        assert close.region.width >= 3
        assert collapse.region.y == buddy.content_region.y
        assert close.region.y == buddy.content_region.y
        assert collapse.region.x == buddy.content_region.x
        assert close.region.right == buddy.content_region.right
        for control in (collapse, close):
            target, _ = app.screen.get_widget_at(
                control.region.x + control.region.width // 2,
                control.region.y + control.region.height // 2,
            )
            assert target is control


@pytest.mark.asyncio
async def test_keyboard_focus_exposes_only_exact_action_label():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        close = buddy.query_one("#persona-buddy-close", Button)

        collapse.focus(scroll_visible=False)
        await pilot.pause()
        focused = _compositor_text(app.screen)
        assert "Fold" in focused
        assert "Close" not in focused

        close.focus(scroll_visible=False)
        await pilot.pause()
        focused = _compositor_text(app.screen)
        assert "Close" in focused
        assert "Fold" not in focused

        focus_rule = PersonaBuddyWidget.BUNDLED_CSS.split(
            "PersonaBuddyWidget .persona-buddy-control:focus {", 1
        )[1].split("}", 1)[0]
        assert "background: $accent;" in focus_rule
        assert "text-style: bold underline;" in focus_rule
        assert "outline: none;" in focus_rule
        assert "outline: heavy" not in focus_rule


@pytest.mark.asyncio
async def test_pet_surface_drags_but_icon_buttons_do_not():
    controller = _FakeController(
        frames=("PET",),
        frame_sizes=((12, 10),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: "PET" in _compositor_text(app.screen))
        controls = tuple(buddy.query(Button))

        for control in controls:
            buddy.on_mouse_down(
                _mouse(
                    events.MouseDown,
                    x=control.region.x + control.region.width // 2,
                    y=control.region.y + control.region.height // 2,
                    widget=control,
                )
            )
            assert buddy._interaction is None
            assert app.mouse_captured is None

        for y in range(frame.region.y, frame.region.bottom):
            for x in range(frame.region.x, frame.region.right):
                if any(_point_is_inside(control, x, y) for control in controls):
                    continue
                buddy.on_mouse_down(_mouse(events.MouseDown, x=x, y=y, widget=frame))
                expected_mode = (
                    "resize"
                    if x >= buddy.region.right - 2 and y >= buddy.region.bottom - 1
                    else "drag"
                )
                assert buddy._interaction is not None
                assert buddy._interaction[0] == expected_mode
                buddy.release_interaction_capture()


@pytest.mark.asyncio
async def test_drag_and_resize_are_viewport_bounded_and_persist_once():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        initial_geometry = buddy._clamped_geometry(controller.preferences.geometry)
        assert initial_geometry.x + initial_geometry.width <= app.size.width
        assert initial_geometry.y + initial_geometry.height <= app.size.height

        buddy.on_mouse_down(
            _mouse(
                events.MouseDown,
                x=frame.region.x + frame.region.width // 2,
                y=frame.region.y + 1,
                widget=frame,
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
        assert buddy.absolute_offset == Offset(74, 23)
        await pilot.press("c")
        await _wait_until(lambda: controller.preferences.collapsed)
        await _wait_until(
            lambda: buddy.query_one("#persona-buddy-collapse", Button).tooltip == "Open"
        )
        await _wait_until(lambda: buddy.region.height == 3)
        await pilot.press("x")
        await _wait_until(lambda: not controller.preferences.open)

        keys = {binding.key for binding in PersonaBuddyWidget.BINDINGS}
        assert keys == {"h", "j", "k", "l", "H", "J", "K", "L", "0", "c", "x"}
        assert not any(key.startswith("ctrl+") for key in keys)


@pytest.mark.asyncio
async def test_tiny_viewport_uses_icon_compact_control():
    controller = _FakeController(size=(28, 12))
    app = _BuddyApp(controller)
    async with app.run_test(size=(12, 4)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.has_class("persona-buddy-compact"))
        assert buddy.region.width <= 12
        assert buddy.region.height <= 4
        assert "▾" in _compositor_text(app.screen)
        assert "Buddy" not in _compositor_text(app.screen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("viewport", "compact"),
    [((28, 12), False), ((18, 6), True), ((12, 4), True), ((10, 2), True)],
)
async def test_exact_cell_layouts_keep_complete_icon_controls(
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
        controls = (collapse, close) if viewport[0] >= 14 else (collapse,)
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
            assert "▾" in text
            assert "Buddy" not in text
            assert close.display is (viewport[0] >= 14)
            if close.display:
                assert "×" in text
        else:
            assert "▾" in text
            assert "×" in text
            assert "Fold" not in text
            assert "Close" not in text
            assert "hjkl move · HJKL size" not in text


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
        await _wait_until(lambda: buddy.region.size == (26, 7))


@pytest.mark.asyncio
async def test_drag_while_viewport_becomes_compact_preserves_preferred_size():
    controller = _FakeController(size=(28, 12))
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        buddy.on_mouse_down(
            _mouse(
                events.MouseDown,
                x=frame.region.x + frame.region.width // 2,
                y=frame.region.y + 1,
                widget=frame,
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
        await _wait_until(lambda: buddy.region.size == (26, 7))


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
async def test_icon_buttons_click_without_starting_drag_and_reopen_collapsed():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.width == 28)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        await pilot.pause()
        await pilot.click("#persona-buddy-collapse")
        await _wait_until(lambda: controller.preferences.collapsed)
        assert app.mouse_captured is None
        reopen = buddy.query_one("#persona-buddy-collapse", Button)
        assert str(reopen.label) == "Open"
        assert reopen.tooltip == "Open"
        assert reopen.display
        collapsed_close = buddy.query_one("#persona-buddy-close", Button)
        assert (str(collapsed_close.label), collapsed_close.tooltip) == ("×", "Close")
        assert collapsed_close.display
        assert reopen.region.x == buddy.content_region.x
        assert collapsed_close.region.right == buddy.content_region.right
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
async def test_accepted_visual_uses_one_maximum_frame_box_without_jitter():
    controller = _FakeController(
        animate=True,
        frames=("SMALL", "LARGE"),
        frame_sizes=((8, 6), (12, 10)),
        durations=(300, 300),
        loop=False,
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "SMALL" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy.region.size == (14, 7))

        accepted = buddy._accepted_render
        assert accepted is not None
        assert (accepted.content_width, accepted.content_height) == (12, 5)
        stable_region = buddy.region

        buddy._next_frame_at = 0
        buddy.advance_frame()
        await _wait_until(lambda: "LARGE" in _compositor_text(app.screen))

        assert buddy.region == stable_region
        assert (accepted.content_width, accepted.content_height) == (12, 5)


@pytest.mark.asyncio
async def test_single_frame_fits_to_content_without_persisting_dimensions():
    controller = _FakeController(
        frames=("FITTED",),
        frame_sizes=((12, 10),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.size == (14, 7))

        assert controller.preferences.geometry.width == 28
        assert controller.preferences.geometry.height == 12
        assert controller.preference_writes == []
        assert controller.persisted == []
        assert controller.resolve_sizes == [(26, 10)]

        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(lambda: len(controller.resolve_sizes) == 2)

        assert controller.resolve_sizes[-1] == (26, 10)
        assert controller.preference_writes == []
        assert controller.persisted == []


@pytest.mark.asyncio
async def test_undersized_accepted_frame_keeps_icon_controls_operable():
    controller = _FakeController(
        frames=("FITTED",),
        frame_sizes=((12, 10),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "FITTED" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy.region.width == 14)
        await pilot.pause()
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        close = buddy.query_one("#persona-buddy-close", Button)
        frame = buddy.query_one("#persona-buddy-frame", Static)

        assert buddy.region.width == 14
        assert frame.content_region.width == 12
        assert buddy.region.right <= app.size.width
        for control in (collapse, close):
            assert buddy.region.contains_region(control.region)
            assert str(control.label) in _compositor_text(app.screen)
            target, _ = app.screen.get_widget_at(
                control.region.x + control.region.width // 2,
                control.region.y + control.region.height // 2,
            )
            assert target is control


@pytest.mark.asyncio
async def test_empty_direct_visual_uses_operable_fail_soft_box():
    controller = _FakeController(frames=())
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy._accepted_render is not None)

        accepted = buddy._accepted_render
        assert accepted is not None
        assert (accepted.content_width, accepted.content_height) == (10, 4)
        await _wait_until(lambda: buddy.region.size == (12, 6))


@pytest.mark.asyncio
@pytest.mark.parametrize(("width", "height"), ((0, 10), (12, True)))
async def test_invalid_direct_dimensions_cannot_replace_accepted_render(
    width: object,
    height: object,
):
    controller = _FakeController(
        frames=("CURRENT",),
        frame_sizes=((12, 10),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "CURRENT" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy.region.size == (14, 7))
        accepted = buddy._accepted_render
        accepted_region = buddy.region

        controller.visual = _visual_with_frame(
            controller,
            label="INVALID",
            width=width,
            height=height,
        )
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(
            lambda: (
                len(controller.resolve_sizes) == 2
                and buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )

        assert buddy._accepted_render is accepted
        assert buddy.region == accepted_region
        assert "CURRENT" in _compositor_text(app.screen)
        assert "INVALID" not in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_accepted_self_fit_does_not_advance_viewport_or_reresolve():
    controller = _ViewportGenerationController(
        frames=("FITTED",),
        frame_sizes=((12, 10),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: buddy.region.size == (14, 7))
        assert frame.content_region.size == (12, 5)
        accepted = buddy._accepted_render
        assert accepted is not None
        assert (accepted.content_width, accepted.content_height) == (12, 5)
        await asyncio.sleep(0.25)

        assert controller.viewport_updates == [1]
        assert controller.resolve_sizes == [(26, 10)]

        await pilot.resize_terminal(30, 11)
        await _wait_until(lambda: controller.viewport_updates == [1, 2])
        await _wait_until(lambda: len(controller.resolve_sizes) == 2)

        assert controller.resolve_sizes[-1] == (26, 9)


@pytest.mark.asyncio
async def test_prior_budget_snapshot_visual_cannot_refit_current_view():
    controller = _BlockAfterAcceptedController(
        frames=("CURRENT",),
        frame_sizes=((8, 6),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.size == (12, 6))
        assert "CURRENT" in _compositor_text(app.screen)
        accepted_region = buddy.region

        stale = _visual_with_frame(
            controller,
            label="PRIOR-BUDGET",
            width=20,
            height=12,
        )
        controller.block = True
        controller.direct_visual = stale
        controller.generation += 1
        buddy.refresh_from_controller()
        await asyncio.wait_for(controller.resolve_started.wait(), timeout=1)

        controller.preferences = replace(
            controller.preferences,
            geometry=replace(controller.preferences.geometry, width=40, height=18),
        )
        controller.visual = stale
        buddy.refresh_from_controller(schedule_resolution=False)
        controller.resolve_release.set()
        await asyncio.wait_for(controller.stale_resolve_completed.wait(), timeout=1)
        await asyncio.wait_for(controller.next_resolve_started.wait(), timeout=1)

        assert "CURRENT" in _compositor_text(app.screen)
        assert "PRIOR-BUDGET" not in _compositor_text(app.screen)
        assert buddy.region == accepted_region


@pytest.mark.asyncio
async def test_prior_viewport_direct_result_cannot_replace_accepted_render():
    controller = _BlockAfterAcceptedController(
        frames=("CURRENT",),
        frame_sizes=((8, 6),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: buddy.region.size == (12, 6))
        assert "CURRENT" in _compositor_text(app.screen)
        accepted_region = buddy.region

        stale = _visual_with_frame(
            controller,
            label="PRIOR-VIEWPORT",
            width=20,
            height=12,
        )
        controller.block = True
        controller.direct_visual = stale
        controller.generation += 1
        buddy.refresh_from_controller()
        await asyncio.wait_for(controller.resolve_started.wait(), timeout=1)

        controller.viewport_generation += 1
        controller.visual = stale
        buddy.refresh_from_controller(schedule_resolution=False)
        controller.resolve_release.set()
        await asyncio.wait_for(controller.stale_resolve_completed.wait(), timeout=1)
        await asyncio.wait_for(controller.next_resolve_started.wait(), timeout=1)

        assert "CURRENT" in _compositor_text(app.screen)
        assert "PRIOR-VIEWPORT" not in _compositor_text(app.screen)
        assert buddy.region == accepted_region


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
async def test_resolution_uses_preferred_pet_budget_not_fitted_window_region():
    controller = _SlotSizedFrameController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: bool(controller.resolve_sizes))

        assert controller.resolve_sizes[-1] == (26, 10)
        assert controller.resolve_sizes[-1] == frame.content_region.size
        assert buddy.region.size == (28, 12)
        strips = app.screen._compositor.render_strips()
        painted_frame = "\n".join(
            strips[y].text[frame.region.x : frame.region.right]
            for y in range(frame.region.y, frame.region.bottom)
        )
        cols, _lines = controller.resolve_sizes[-1]
        source_rows = frame.renderable.plain.splitlines()
        assert source_rows[0] == "TOP".ljust(cols - len("RIGHT"), "-") + "RIGHT"
        assert ("BOTTOM".ljust(cols - len("RIGHT"), "-") + "RIGHT") in painted_frame


@pytest.mark.asyncio
async def test_frame_slot_resize_does_not_feed_resolution_authority():
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
        await asyncio.sleep(0.2)

        assert controller.resolve_sizes == [(26, 10)]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("hidden", "collapsed"))
async def test_hidden_or_collapsed_buddy_does_not_resolve(mode: str):
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: len(controller.resolve_sizes) == 1)

        if mode == "hidden":
            buddy.display = False
        elif mode == "collapsed":
            controller.preferences = replace(
                controller.preferences,
                collapsed=True,
            )
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
                x=buddy.query_one("#persona-buddy-frame", Static).region.center[0],
                y=buddy.query_one("#persona-buddy-frame", Static).region.y + 1,
                widget=buddy.query_one("#persona-buddy-frame", Static),
            )
        )
        assert app.mouse_captured is buddy
        await buddy.remove()
        assert app.mouse_captured is None
