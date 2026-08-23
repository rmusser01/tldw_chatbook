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
        self.state = "idle"
        self.state_source: str | None = None
        self.state_owner: str | None = None
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
            state=self.state,
            state_source=self.state_source,
            state_owner=self.state_owner,
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


class _ModeSizedFrameController(_FakeController):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.visual = replace(
            self.visual,
            graph_identity="shared-graph",  # type: ignore[arg-type]
            cache_identity="shared-cache",  # type: ignore[arg-type]
        )

    async def resolve_current_visual(self, *, cols: int, lines: int):
        self.resolve_sizes.append((cols, lines))
        prefix = "THUMB" if (cols, lines) == (10, 4) else "FULL"
        prepared = []
        for index, frame in enumerate(self.visual.frames):
            values = vars(frame).copy()
            values.update(
                width=cols,
                height=lines * 2,
                paint_digest=f"{prefix.lower()}-{index}",
                renderable=Text(f"{prefix}-{index}"),
            )
            prepared.append(SimpleNamespace(**values))
        self.visual = replace(self.visual, frames=tuple(prepared))  # type: ignore[arg-type]
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
@pytest.mark.parametrize(
    ("state", "label"),
    (
        ("approval_needed", "Approval needed"),
        ("error", "Error"),
        ("offline", "Offline"),
    ),
)
async def test_only_actionable_states_replace_pet_with_fixed_path_free_text(
    state: str,
    label: str,
):
    controller = _FakeController(
        frames=("PET",),
        frame_sizes=((10, 8),),
    )
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: "PET" in _compositor_text(app.screen))
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()
        stable_region = buddy.region

        controller.state = state
        controller.state_source = "/private/provider/output"
        controller.state_owner = "assistant-secret-owner"
        controller.visual = replace(
            _visual_with_frame(
                controller,
                label="HIDDEN-ALERT-VISUAL",
                width=30,
                height=20,
            ),
            reason="hostile exception /Users/alice/private.txt",
        )
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and "HIDDEN-ALERT-VISUAL"
                in str(buddy._accepted_render.visual.frames[0].renderable)
            )
        )
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()
        await _wait_until(lambda: str(frame.renderable) == label)
        painted = _compositor_text(app.screen)

        assert "PET" not in painted
        assert "HIDDEN-ALERT-VISUAL" not in painted
        assert "/private/provider/output" not in painted
        assert "assistant-secret-owner" not in painted
        assert "hostile exception" not in painted
        assert "/Users/alice/private.txt" not in painted
        assert frame.has_class("persona-buddy-alert")
        rows = [frame.render_line(y).text for y in range(frame.region.height)]
        nonempty = [row for row in rows if row.strip()]
        assert 1 <= len(nonempty) <= 2
        assert " ".join(row.strip() for row in nonempty) == label
        for row in nonempty:
            left = len(row) - len(row.lstrip())
            right = len(row) - len(row.rstrip())
            assert abs(left - right) <= 1
        assert buddy.region == stable_region


@pytest.mark.asyncio
async def test_explicit_geometry_change_refits_current_alert_without_state_resize():
    controller = _FakeController(
        frames=("PET",),
        frame_sizes=((26, 20),),
    )
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "PET" in _compositor_text(app.screen))
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()
        normal_region = buddy.region
        assert normal_region.size == (28, 12)

        controller.state = "offline"
        controller.visual = _visual_with_frame(
            controller,
            label="HIDDEN-LARGE-ALERT",
            width=30,
            height=20,
        )
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and "HIDDEN-LARGE-ALERT"
                in str(buddy._accepted_render.visual.frames[0].renderable)
            )
        )
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()

        assert buddy.region == normal_region
        assert "Offline" in _compositor_text(app.screen)
        assert "HIDDEN-LARGE-ALERT" not in _compositor_text(app.screen)

        requested = replace(controller.preferences.geometry, width=20)
        controller.visual = _visual_with_frame(
            controller,
            label="HIDDEN-RESIZED-ALERT",
            width=18,
            height=20,
        )
        controller.apply_preferences_patch(geometry=requested)
        buddy.refresh_from_controller()
        await _wait_until(lambda: controller.resolve_sizes[-1] == (18, 10))
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and "HIDDEN-RESIZED-ALERT"
                in str(buddy._accepted_render.visual.frames[0].renderable)
            )
        )
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()

        assert buddy.region.size == (20, 12)
        assert buddy.region.width <= requested.width
        assert buddy.region.height <= requested.height
        assert "Offline" in _compositor_text(app.screen)
        assert "HIDDEN-RESIZED-ALERT" not in _compositor_text(app.screen)
        assert controller.preference_writes == [{"geometry": requested}]

        controller.state_source = "new-state-source-only"
        controller.visual = _visual_with_frame(
            controller,
            label="HIDDEN-STATE-ONLY-ALERT",
            width=8,
            height=8,
        )
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and "HIDDEN-STATE-ONLY-ALERT"
                in str(buddy._accepted_render.visual.frames[0].renderable)
            )
        )
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()

        assert buddy.region.size == (20, 12)
        assert "Offline" in _compositor_text(app.screen)
        assert "new-state-source-only" not in _compositor_text(app.screen)
        assert "HIDDEN-STATE-ONLY-ALERT" not in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_initial_actionable_mount_uses_minimum_until_non_alert_direct_result():
    controller = _FakeController(
        frames=("HIDDEN-INITIAL-ALERT-VISUAL",),
        frame_sizes=((30, 20),),
    )
    controller.state = "offline"
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and "HIDDEN-INITIAL-ALERT-VISUAL"
                in str(buddy._accepted_render.visual.frames[0].renderable)
            )
        )
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()

        assert buddy.region.size == (12, 6)
        assert "Offline" in _compositor_text(app.screen)
        assert "HIDDEN-INITIAL-ALERT-VISUAL" not in _compositor_text(app.screen)

        requested = replace(controller.preferences.geometry, width=20)
        controller.visual = _visual_with_frame(
            controller,
            label="HIDDEN-INITIAL-RESIZED-ALERT",
            width=18,
            height=20,
        )
        controller.apply_preferences_patch(geometry=requested)
        buddy.refresh_from_controller()
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and "HIDDEN-INITIAL-RESIZED-ALERT"
                in str(buddy._accepted_render.visual.frames[0].renderable)
            )
        )
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()

        assert buddy.region.size == (20, 12)
        assert "Offline" in _compositor_text(app.screen)
        assert "HIDDEN-INITIAL-RESIZED-ALERT" not in _compositor_text(app.screen)

        controller.state = "idle"
        controller.visual = _visual_with_frame(
            controller,
            label="CURRENT",
            width=14,
            height=10,
        )
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(lambda: "CURRENT" in _compositor_text(app.screen))
        await _wait_until(
            lambda: (
                buddy._resolution_worker is not None
                and buddy._resolution_worker.is_finished
            )
        )
        await pilot.pause()

        assert buddy.region.size == (16, 7)
        assert "Offline" not in _compositor_text(app.screen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    (
        "idle",
        "thinking",
        "speaking",
        "listening",
        "tool_running",
        "wake_armed",
        "explicit",
        "authored",
        "custom",
        "arbitrary /tmp/private-state",
    ),
)
async def test_non_actionable_states_remain_wordless(state: str):
    controller = _FakeController(frames=("PET-ONLY",))
    controller.state = state
    controller.state_source = "hostile-source"
    controller.state_owner = "hostile-owner"
    controller.visual = replace(controller.visual, reason="hostile-reason")
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)):
        await _wait_until(lambda: "PET-ONLY" in _compositor_text(app.screen))
        painted = _compositor_text(app.screen)

        assert state not in painted
        assert "hostile-source" not in painted
        assert "hostile-owner" not in painted
        assert "hostile-reason" not in painted


@pytest.mark.asyncio
async def test_alert_restores_latest_current_accepted_frame():
    controller = _FakeController(frames=("ORIGINAL-PET",))
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "ORIGINAL-PET" in _compositor_text(app.screen))

        controller.state = "offline"
        controller.visual = _visual_with_frame(
            controller,
            label="LATEST-PET",
            width=24,
            height=10,
        )
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and "LATEST-PET"
                in str(buddy._accepted_render.visual.frames[0].renderable)
            )
        )
        assert "Offline" in _compositor_text(app.screen)
        assert "LATEST-PET" not in _compositor_text(app.screen)

        controller.state = "idle"
        controller.generation += 1
        buddy.refresh_from_controller(schedule_resolution=False)

        assert "LATEST-PET" in _compositor_text(app.screen)
        assert "Offline" not in _compositor_text(app.screen)


@pytest.mark.asyncio
async def test_folded_mode_resolves_real_thumbnail_under_distinct_authority():
    controller = _ModeSizedFrameController(frames=("SOURCE",))
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "FULL-0" in _compositor_text(app.screen))
        full = buddy._accepted_render
        assert full is not None
        assert controller.resolve_sizes == [(26, 10)]

        controller.preferences = replace(controller.preferences, collapsed=True)
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(lambda: controller.resolve_sizes == [(26, 10), (10, 4)])
        await _wait_until(
            lambda: (
                buddy._accepted_render is not None
                and buddy._accepted_render.authority != full.authority
            )
        )
        await _wait_until(lambda: "THUMB-0" in _compositor_text(app.screen))

        folded = buddy._accepted_render
        assert folded is not None
        assert folded.authority != full.authority
        assert full.authority[-2:] == (False, (26, 10))
        assert folded.authority[-2:] == (True, (10, 4))
        assert folded.visual.graph_identity == full.visual.graph_identity
        assert folded.visual.cache_identity == full.visual.cache_identity
        assert "THUMB-0" in _compositor_text(app.screen)
        await asyncio.sleep(0.25)
        assert controller.resolve_sizes == [(26, 10), (10, 4)]


@pytest.mark.asyncio
async def test_folded_thumbnail_is_static_and_uses_open_close_icons():
    controller = _ModeSizedFrameController(
        collapsed=True,
        animate=True,
        frames=("SOURCE-A", "SOURCE-B"),
    )
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "THUMB-0" in _compositor_text(app.screen))
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        close = buddy.query_one("#persona-buddy-close", Button)

        assert controller.resolve_sizes == [(10, 4)]
        assert (str(collapse.label), collapse.tooltip) == ("▴", "Open")
        assert (str(close.label), close.tooltip) == ("×", "Close")
        assert buddy._frame_timer is None
        buddy._next_frame_at = 0
        buddy.advance_frame()
        assert buddy.frame_index == 0
        assert "THUMB-0" in _compositor_text(app.screen)
        assert "THUMB-1" not in _compositor_text(app.screen)
        buddy.on_mouse_down(
            _mouse(
                events.MouseDown,
                x=buddy.region.right - 1,
                y=buddy.region.bottom - 1,
            )
        )
        assert buddy._interaction is not None
        assert buddy._interaction[0] == "resize"
        buddy.release_interaction_capture()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("constraint", "effective_size", "paints_thumbnail"),
    (
        ("preferred", (10, 4), True),
        ("viewport", (10, 4), True),
        ("preferred", (9, 4), False),
        ("preferred", (10, 3), False),
        ("viewport", (9, 4), False),
        ("viewport", (10, 3), False),
    ),
)
async def test_only_effective_area_below_10x4_uses_two_button_fallback(
    constraint: str,
    effective_size: tuple[int, int],
    paints_thumbnail: bool,
):
    outer_size = (effective_size[0] + 2, effective_size[1] + 2)
    preferred = outer_size if constraint == "preferred" else (28, 12)
    viewport = outer_size if constraint == "viewport" else (80, 24)
    controller = _ModeSizedFrameController(size=preferred, collapsed=True)
    app = _BuddyApp(controller)

    async with app.run_test(size=viewport):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        close = buddy.query_one("#persona-buddy-close", Button)
        if paints_thumbnail:
            await _wait_until(lambda: "THUMB-0" in _compositor_text(app.screen))
            assert controller.resolve_sizes == [(10, 4)]
            assert not buddy.has_class("persona-buddy-compact")
        else:
            await _wait_until(lambda: buddy.has_class("persona-buddy-compact"))
            await asyncio.sleep(0.15)
            assert "THUMB-0" not in _compositor_text(app.screen)
            assert controller.resolve_sizes == []

        assert collapse.display
        assert close.display
        assert collapse.region.right <= close.region.x
        for control in (collapse, close):
            assert buddy.region.contains_region(control.region)
            target, _ = app.screen.get_widget_at(
                control.region.x + control.region.width // 2,
                control.region.y + control.region.height // 2,
            )
            assert target is control


@pytest.mark.asyncio
async def test_undersized_asset_uses_only_10x4_operable_minimum():
    controller = _FakeController(
        collapsed=True,
        frames=("P",),
        frame_sizes=((1, 1),),
    )
    app = _BuddyApp(controller)

    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: "P" in _compositor_text(app.screen))

        accepted = buddy._accepted_render
        assert accepted is not None
        assert controller.resolve_sizes == [(10, 4)]
        assert (accepted.content_width, accepted.content_height) == (10, 4)
        assert frame.content_region.size == (10, 4)
        assert buddy.region.size == (12, 6)
        rows = [frame.render_line(y).text for y in range(frame.region.height)]
        painted_y = next(index for index, row in enumerate(rows) if "P" in row)
        painted_x = rows[painted_y].index("P")
        assert painted_x in (4, 5)
        assert painted_y in (1, 2)


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
async def test_unchanged_refresh_does_not_reset_native_tooltips(monkeypatch):
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "BUDDY-A" in _compositor_text(app.screen))
        tooltip_updates = []
        monkeypatch.setattr(
            app.screen,
            "_update_tooltip",
            lambda widget: tooltip_updates.append(widget),
        )

        buddy.refresh_from_controller()
        buddy.refresh_from_controller()

        assert tooltip_updates == []


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

        app.screen.query_one("#focus-probe", Input).focus(scroll_visible=False)
        await pilot.pause()
        assert (str(collapse.label), str(close.label)) == ("▾", "×")
        unfocused = _compositor_text(app.screen)
        assert "Fold" not in unfocused
        assert "Close" not in unfocused

        focus_rule = PersonaBuddyWidget.BUNDLED_CSS.split(
            "PersonaBuddyWidget .persona-buddy-control:focus {", 1
        )[1].split("}", 1)[0]
        assert "background: $accent;" in focus_rule
        assert "text-style: bold underline;" in focus_rule
        assert "outline: none;" in focus_rule
        assert "outline: heavy" not in focus_rule


@pytest.mark.asyncio
@pytest.mark.parametrize(("collapsed", "glyph"), [(False, "▾"), (True, "▴")])
async def test_compact_focus_keeps_both_glyph_controls_clear_of_resize_grip(
    collapsed: bool, glyph: str
):
    controller = _FakeController(collapsed=collapsed)
    app = _BuddyApp(controller)
    async with app.run_test(size=(10, 2)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(
            lambda: (
                buddy.has_class("persona-buddy-compact")
                and buddy.region.size == (10, 1)
            )
        )
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        close = buddy.query_one("#persona-buddy-close", Button)

        def assert_compact_controls(focused: Button) -> None:
            assert [
                control.id for control in buddy.query(Button) if control.display
            ] == ["persona-buddy-collapse", "persona-buddy-close"]
            assert (str(collapse.label), str(close.label)) == (glyph, "×")
            assert (collapse.tooltip, close.tooltip) == (
                "Open" if collapsed else "Fold",
                "Close",
            )
            assert collapse.region.right <= close.region.x
            assert close.region.right <= buddy.region.right - 2
            assert "focus" in focused.pseudo_classes
            assert focused.styles.text_style.bold
            assert focused.styles.text_style.underline
            text = _compositor_text(app.screen)
            assert glyph in text
            assert "×" in text
            assert "Fold" not in text
            assert "Open" not in text
            assert "Close" not in text
            for control in (collapse, close):
                assert control.display
                for x in range(control.region.x, control.region.right):
                    target, _ = app.screen.get_widget_at(x, control.region.y)
                    assert target is control

        collapse.focus(scroll_visible=False)
        await pilot.pause()
        assert_compact_controls(collapse)
        await pilot.press("enter")
        await _wait_until(lambda: controller.preferences.collapsed is not collapsed)

        close.focus(scroll_visible=False)
        await pilot.pause()
        glyph = "▴" if controller.preferences.collapsed else "▾"
        collapsed = controller.preferences.collapsed
        assert_compact_controls(close)
        await pilot.press("enter")
        await _wait_until(lambda: not controller.preferences.open)
        assert buddy._interaction is None
        assert app.mouse_captured is None


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
    controller = _ModeSizedFrameController()
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
        assert buddy.absolute_offset == Offset(
            app.size.width - buddy.region.width,
            app.size.height - buddy.region.height,
        )
        await pilot.press("c")
        await _wait_until(lambda: controller.preferences.collapsed)
        await _wait_until(
            lambda: buddy.query_one("#persona-buddy-collapse", Button).tooltip == "Open"
        )
        await _wait_until(lambda: buddy.region.size == (12, 6))
        assert controller.resolve_sizes[-1] == (10, 4)
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
    [((28, 12), False), ((18, 6), False), ((12, 4), True), ((10, 2), True)],
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
        for control in (collapse, close):
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
            assert close.display
            assert "×" in text
        else:
            assert "▾" in text
            assert "×" in text
            assert "Fold" not in text
            assert "Close" not in text
            assert "hjkl move · HJKL size" not in text


@pytest.mark.asyncio
async def test_compact_controls_never_overlap_lower_right_resize_zone():
    controller = _FakeController()
    app = _BuddyApp(controller)
    async with app.run_test(size=(10, 2)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(
            lambda: (
                buddy.has_class("persona-buddy-compact")
                and buddy.region.size == (10, 1)
            )
        )
        await pilot.pause()
        close = buddy.query_one("#persona-buddy-close", Button)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        controls = tuple(control for control in buddy.query(Button) if control.display)

        assert [control.id for control in controls] == [
            "persona-buddy-collapse",
            "persona-buddy-close",
        ]
        assert not frame.display
        assert "BUDDY-A" not in _compositor_text(app.screen)
        assert "Buddy" not in _compositor_text(app.screen)
        resize_zone_x = buddy.region.right - 2
        assert close.region.right <= resize_zone_x
        for x in range(close.region.x, close.region.right):
            target, _ = app.screen.get_widget_at(x, close.region.y)
            assert target is close
            buddy.on_mouse_down(
                _mouse(events.MouseDown, x=x, y=close.region.y, widget=close)
            )
            assert buddy._interaction is None
            assert app.mouse_captured is None

        corner = (buddy.region.right - 1, buddy.region.bottom - 1)
        target, _ = app.screen.get_widget_at(*corner)
        assert target not in controls

        buddy.on_mouse_down(
            _mouse(events.MouseDown, x=corner[0], y=corner[1], widget=target)
        )

        assert buddy._interaction is not None
        assert buddy._interaction[0] == "resize"
        assert app.mouse_captured is buddy
        buddy.release_interaction_capture()

        await pilot.click(close, offset=(close.region.width - 1, 0))
        await _wait_until(lambda: not controller.preferences.open)
        assert buddy._interaction is None
        assert app.mouse_captured is None


@pytest.mark.asyncio
@pytest.mark.parametrize("viewport", [(12, 4), (10, 2)])
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
        collapse = buddy.query_one("#persona-buddy-collapse", Button)
        assert collapse.tooltip == "Fold"
        await pilot.click("#persona-buddy-collapse")
        await _wait_until(lambda: controller.preferences.collapsed)
        assert app.mouse_captured is None
        reopen = collapse
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
        await _wait_until(lambda: collapse.tooltip == "Fold")
        assert str(collapse.label) == "Fold"
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
async def test_actionable_alert_freezes_animation_until_current_pet_resumes():
    controller = _FakeController(
        animate=True,
        frames=("ALERT-A", "ALERT-B"),
        durations=(1_000, 1_000),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "ALERT-A" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy._frame_timer is not None)

        controller.state = "offline"
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(lambda: "Offline" in _compositor_text(app.screen))
        frozen = buddy.frame_index

        assert buddy._frame_timer is None
        buddy._next_frame_at = 0
        buddy.advance_frame()
        assert buddy.frame_index == frozen
        assert buddy._frame_timer is None
        assert "Offline" in _compositor_text(app.screen)
        assert "ALERT-B" not in _compositor_text(app.screen)

        resumed = _FakeController(
            animate=True,
            frames=("RESUME-A", "RESUME-B"),
            durations=(1_000, 1_000),
        )
        controller.state = "idle"
        controller.visual = resumed.visual
        controller.generation += 1
        buddy.refresh_from_controller()
        await _wait_until(lambda: "RESUME-A" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy._frame_timer is not None)

        remaining = buddy._next_frame_at - asyncio.get_running_loop().time()
        assert 0.5 <= remaining <= 1.0
        buddy._next_frame_at = 0
        buddy.advance_frame()
        await _wait_until(lambda: "RESUME-B" in _compositor_text(app.screen))


@pytest.mark.asyncio
async def test_reopen_waits_for_mode_matching_render_before_animation_resumes():
    controller = _BlockAfterAcceptedController(
        collapsed=True,
        animate=True,
        frames=("FOLDED-A", "FOLDED-B"),
        durations=(1_000, 1_000),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)):
        buddy = app.screen.query_one(PersonaBuddyWidget)
        await _wait_until(lambda: "FOLDED-A" in _compositor_text(app.screen))
        assert buddy._frame_timer is None

        full = _FakeController(
            animate=True,
            frames=("FULL-A", "FULL-B"),
            durations=(1_000, 1_000),
        ).visual
        controller.block = True
        controller.direct_visual = full
        controller.visual = full
        controller.preferences = replace(controller.preferences, collapsed=False)
        controller.generation += 1
        buddy.refresh_from_controller()
        await asyncio.wait_for(controller.resolve_started.wait(), timeout=1)
        frozen = buddy.frame_index

        assert "FOLDED-A" not in _compositor_text(app.screen)
        assert "FOLDED-B" not in _compositor_text(app.screen)
        assert buddy._frame_timer is None
        buddy._next_frame_at = 0
        buddy.advance_frame()
        assert buddy.frame_index == frozen
        assert buddy._frame_timer is None
        assert "FOLDED-A" not in _compositor_text(app.screen)
        assert "FOLDED-B" not in _compositor_text(app.screen)

        controller.resolve_release.set()
        await _wait_until(lambda: "FULL-A" in _compositor_text(app.screen))
        await _wait_until(lambda: buddy._frame_timer is not None)
        remaining = buddy._next_frame_at - asyncio.get_running_loop().time()
        assert 0.5 <= remaining <= 1.0


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
async def test_pending_smaller_budget_hides_oversized_accepted_render():
    controller = _BlockAfterAcceptedController(
        frames=("ACCEPTED-20X6",),
        frame_sizes=((20, 12),),
    )
    app = _BuddyApp(controller)
    async with app.run_test(size=(80, 24)) as pilot:
        buddy = app.screen.query_one(PersonaBuddyWidget)
        frame = buddy.query_one("#persona-buddy-frame", Static)
        await _wait_until(lambda: "ACCEPTED-20X6" in _compositor_text(app.screen))
        accepted = buddy._accepted_render
        assert accepted is not None
        assert (accepted.content_width, accepted.content_height) == (20, 6)

        replacement = _visual_with_frame(
            controller,
            label="REPLACEMENT-16X6",
            width=16,
            height=12,
        )
        controller.block = True
        controller.direct_visual = replacement
        controller.visual = replacement
        await pilot.resize_terminal(18, 8)
        await asyncio.wait_for(controller.resolve_started.wait(), timeout=1)
        await pilot.pause()

        assert controller.resolve_sizes[-1] == (16, 6)
        assert buddy._accepted_render is accepted
        assert buddy.has_class("persona-buddy-compact")
        assert not frame.display
        assert [control.id for control in buddy.query(Button) if control.display] == [
            "persona-buddy-collapse",
            "persona-buddy-close",
        ]
        assert "ACCEPTED-20X6" not in _compositor_text(app.screen)
        assert "REPLACEMENT-16X6" not in _compositor_text(app.screen)

        controller.resolve_release.set()
        await _wait_until(lambda: "REPLACEMENT-16X6" in _compositor_text(app.screen))

        assert buddy._accepted_render is not accepted
        assert not buddy.has_class("persona-buddy-compact")
        assert frame.display


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
        assert buddy._poll_timer is not None
        buddy._poll_timer.stop()
        await pilot.pause()
        original = frame.content_region.size
        frame.styles.height = max(1, original.height - 1)
        await pilot.pause()
        await _wait_until(lambda: frame.content_region.size != original)

        buddy.refresh_from_controller()
        await asyncio.sleep(0.2)

        assert controller.resolve_sizes == [(26, 10)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected_sizes"),
    (
        ("hidden", [(26, 10)]),
        ("collapsed", [(26, 10), (10, 4)]),
    ),
)
async def test_hidden_stops_resolution_but_collapsed_resolves_thumbnail(
    mode: str,
    expected_sizes: list[tuple[int, int]],
):
    controller = _ModeSizedFrameController()
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

        assert controller.resolve_sizes == expected_sizes


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
