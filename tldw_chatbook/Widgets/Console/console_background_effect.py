"""Pure Textual Console background effect renderer."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.segment import Segment
from rich.style import Style
from textual._time import get_time
from textual.app import ComposeResult
from textual.await_remove import AwaitRemove
from textual.containers import Container
from textual.content import Content
from textual.css.query import NoMatches, QueryError
from textual.dom import DOMNode
from textual.strip import Strip
from textual.style import Style as TextualStyle
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.UI.Navigation.surface_swap_guard import ensure_surface_swap_not_self_awaited
from tldw_chatbook.Utils.console_background_effects import (
    MAX_CONSOLE_BACKGROUND_FPS,
    MIN_CONSOLE_BACKGROUND_FPS,
    ConsoleBackgroundEffectSettings,
)
from tldw_chatbook.Widgets.Console.console_message_more_menu import (
    message_more_menus_on_screen,
)
from tldw_chatbook.Widgets.Console.console_selection_menu import (
    selection_menus_on_screen,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    ConsoleRoleplayMarkdown,
    ConsoleTranscript,
)
from tldw_chatbook.Widgets.Console.console_transcript_input import TranscriptInputVisit

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_chat_store import (
        ConsoleChatSession,
        ConsoleChatStore,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

_INTENSITY_DENSITY = {"low": 0.035, "medium": 0.07, "high": 0.11}
#: Bound the additional mounted graph independently of the visible window.
_RETAINED_TRANSCRIPT_ROWS = 128
_EFFECT_CHARS = {
    "snow": ".*+",
    "rain": ".:|",
    "matrix": "abcdefghijklmnopqrstuvwxyz0123456789",
}
_EFFECT_STYLES = {
    "snow": Style(color="#d7e8f7", dim=True),
    "rain": Style(color="#70a7d8", dim=True),
    "matrix": Style(color="#4ade80", dim=True),
}


@dataclass
class _Particle:
    x: int
    y: float
    speed: float
    char: str


class ConsoleBackgroundEffect(Widget):
    """Non-focusable animated text background effect for the Console transcript."""

    can_focus = False

    def __init__(
        self,
        settings: ConsoleBackgroundEffectSettings,
        *,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self._random = random.Random(seed)
        self._particles: list[_Particle] = []
        self._timer: Timer | None = None
        # task-261: `render_line` used to call `frame_text()` (a full W×H
        # grid build) once PER LINE, making each repaint O(W·H²). The frame
        # is instead computed once per (frame serial, width, height) and the
        # split lines reused for every `render_line` call of that repaint.
        # The serial bumps on every particle step (`_advance_frame`) and on
        # `update_settings`; width/height live in the key, so a resize
        # invalidates naturally. The cache lives on the widget instance.
        self._frame_serial: int = 0
        self._frame_cache: tuple[int, int, int, list[str]] | None = None

    @property
    def is_effect_active(self) -> bool:
        """Return whether the current settings should render an effect."""
        return self.settings.active

    @property
    def frame_rate(self) -> int:
        """Return renderer-clamped frames per second."""
        return max(
            MIN_CONSOLE_BACKGROUND_FPS,
            min(MAX_CONSOLE_BACKGROUND_FPS, int(self.settings.fps)),
        )

    def on_mount(self) -> None:
        """Start frame updates once the widget is mounted and active."""
        # Textual sets is_mounted after dispatching Mount.
        self.call_after_refresh(self._sync_timer)

    def on_unmount(self) -> None:
        """Stop frame updates when the widget leaves the DOM."""
        self._stop_timer()

    def update_settings(self, settings: ConsoleBackgroundEffectSettings) -> None:
        """Apply new settings, reset particles, and refresh the renderer.

        Args:
            settings: The new background-effect settings to render with.
        """
        self.settings = settings
        self._particles.clear()
        self._invalidate_frame_cache()
        self._sync_timer()
        self.refresh(layout=False)

    def frame_text(self, width: int, height: int) -> str:
        """Return a plain text frame for the requested dimensions."""
        width = max(0, width)
        height = max(0, height)
        if not self.is_effect_active or width == 0 or height == 0:
            return "\n".join(" " * width for _ in range(height))

        self._ensure_particle_count(width, height)
        grid = [[" " for _ in range(width)] for _ in range(height)]
        for particle in self._particles:
            y = int(particle.y)
            if 0 <= particle.x < width and 0 <= y < height:
                grid[y][particle.x] = particle.char
        return "\n".join("".join(row) for row in grid)

    def render_line(self, y: int) -> Strip:
        """Render a single background frame line.

        Args:
            y: Zero-based line index within the widget to render.

        Returns:
            The rendered strip for line ``y`` of the current frame.
        """
        width = self.size.width
        height = self.size.height
        if width <= 0 or height <= 0:
            return Strip.blank(max(0, width))
        lines = self._frame_lines(width, height)
        text = lines[y] if 0 <= y < len(lines) else " " * width
        style = _EFFECT_STYLES.get(self.settings.effect, Style(dim=True))
        return Strip([Segment(text, style)], width)

    def _frame_lines(self, width: int, height: int) -> list[str]:
        """Return the current frame's lines, computing them at most once per frame.

        The full-grid ``frame_text()`` build runs only when the cached frame's
        (serial, width, height) key no longer matches — i.e. after a particle
        step, a settings change, or a resize — so a repaint's H ``render_line``
        calls share one grid computation instead of building H grids.

        Args:
            width: Frame width in cells.
            height: Frame height in cells.

        Returns:
            The frame's rows, one string of length ``width`` per row.
        """
        cached = self._frame_cache
        if cached is not None:
            serial, cached_width, cached_height, lines = cached
            if (
                serial == self._frame_serial
                and cached_width == width
                and cached_height == height
            ):
                return lines
        lines = self.frame_text(width, height).splitlines()
        self._frame_cache = (self._frame_serial, width, height, lines)
        return lines

    def _invalidate_frame_cache(self) -> None:
        """Drop the cached frame so the next render recomputes the grid."""
        self._frame_serial += 1
        self._frame_cache = None

    def _sync_timer(self) -> None:
        self._stop_timer()
        if not self.is_effect_active or not self.is_mounted or not self.is_attached:
            return
        interval = 1 / self.frame_rate
        self._timer = self.set_interval(interval, self._advance_frame)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _advance_frame(self) -> None:
        # Skip while the screen/tab is inactive so hidden tabs burn no CPU.
        if not self.is_attached or not self.screen.is_active:
            return
        width = self.size.width
        height = self.size.height
        if not self.is_effect_active or width <= 0 or height <= 0:
            self.refresh(layout=False)
            return
        self._step_particles(width, height)
        self._invalidate_frame_cache()
        self.refresh(layout=False)

    def _target_particle_count(self, width: int, height: int) -> int:
        if not self.is_effect_active or width <= 0 or height <= 0:
            return 0
        density = _INTENSITY_DENSITY.get(
            self.settings.intensity, _INTENSITY_DENSITY["low"]
        )
        return max(1, min(width * height, int(width * height * density)))

    def _new_particle(self, width: int, height: int) -> _Particle:
        chars = _EFFECT_CHARS.get(self.settings.effect, _EFFECT_CHARS["snow"])
        speed_ranges = {
            "snow": (0.35, 1.0),
            "rain": (0.8, 1.8),
            "matrix": (0.6, 1.4),
        }
        min_speed, max_speed = speed_ranges.get(self.settings.effect, (0.5, 1.2))
        return _Particle(
            x=self._random.randrange(max(1, width)),
            y=float(self._random.randrange(max(1, height))),
            speed=self._random.uniform(min_speed, max_speed),
            char=self._random.choice(chars),
        )

    def _ensure_particle_count(self, width: int, height: int) -> None:
        target = self._target_particle_count(width, height)
        while len(self._particles) < target:
            self._particles.append(self._new_particle(width, height))
        if len(self._particles) > target:
            del self._particles[target:]

    def _step_particles(self, width: int, height: int) -> None:
        self._particles = [
            particle
            for particle in self._particles
            if 0 <= particle.x < width and particle.y + particle.speed < height
        ]
        for particle in self._particles:
            particle.y += particle.speed
            if self.settings.effect == "matrix":
                chars = _EFFECT_CHARS["matrix"]
                particle.char = self._random.choice(chars)
        self._ensure_particle_count(width, height)


class ConsoleTranscriptSurface(Container):
    """Container for a background effect plus the native Console transcript."""

    can_focus = False
    BUNDLED_CSS = """
    ConsoleTranscriptSurface > ConsoleTranscript > ConsoleBackgroundEffect {
        layer: console-background;
        dock: top;
        width: 100%;
        height: 100%;
    }

    ConsoleTranscriptSurface > ConsoleTranscript {
        layers: console-background default;
        dock: top;
        width: 100%;
        height: 100%;
    }
    """

    def __init__(
        self,
        settings: ConsoleBackgroundEffectSettings,
        *,
        transcript: ConsoleTranscript | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.settings = settings
        self.transcript = transcript or ConsoleTranscript(
            id="console-native-transcript"
        )

        # Textual's compositor does not expose sibling glyphs through a blank
        # transcript strip. Dock the dedicated renderer below message children.
        # A factory also restores that decoration after transcript recomposition.
        self.transcript.background_effect_factory = self._make_background_effect
        self._projection_lock = asyncio.Lock()
        self._authority: tuple[str, str] | None = None
        self._store: ConsoleChatStore | None = None
        self._active_key: _TranscriptKey | None = None
        self._previous: tuple[_TranscriptKey, ConsoleTranscript] | None = None
        self._retiring: AwaitRemove | None = None

    def _can_retain(self, viewport: ConsoleTranscript) -> bool:
        """Keep a bounded, passive text window on the mounted surface."""
        if (
            self.settings.active
            or not viewport._messages
            or len(viewport._messages) > _RETAINED_TRANSCRIPT_ROWS
            or len(viewport._row_widgets) > _RETAINED_TRANSCRIPT_ROWS
            or not viewport._windowing_enabled()
            or not viewport._is_following_tail()
            or viewport.selected_message_id is not None
            or viewport.pending_selection_id is not None
            or viewport.has_pending_selection_ui
            or viewport._voice_preview_projection is not None
            or viewport._refresh_lock.locked()
            or viewport._hydrating_scrollback
            or viewport._scrollback_hydration_scheduled
            or viewport._tailward_hydration_scheduled
            or viewport._reveal_scroll_target is not None
            or viewport._image_specs
            or viewport._generation_card_specs
            or viewport._video_card_specs
            or viewport.app.mouse_captured is viewport
            or any(
                _displayed_within(viewport, button)
                for button in viewport.walk_children(Button)
            )
            or any(
                menu.owner is viewport
                for menu in message_more_menus_on_screen(self.screen)
            )
            or any(
                menu._owner is viewport
                for menu in selection_menus_on_screen(self.screen)
            )
            or any(
                not worker.is_finished and viewport in worker.node.ancestors_with_self
                for worker in self.workers
            )
        ):
            return False
        # Native controls can turn an old click into a newly dated message.
        if not all(
            message.role in (ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT)
            and message.status == "complete"
            and not (
                message.attachments
                or message.image_data
                or message.raw_cli_presentation
                or message.activity_presentation
                or message.generation_metadata
            )
            for message in viewport._messages
        ):
            return False
        for markdown in viewport.walk_children(ConsoleRoleplayMarkdown):
            if (
                markdown.pending_update is not None
                and not markdown.pending_update.is_done
            ):
                return False
            if any(
                isinstance(content := block.content, Content)
                and any(
                    isinstance(span.style, TextualStyle)
                    and span.style.meta.get("@click")
                    for span in content.spans
                )
                for block in markdown.walk_children(Static)
            ):
                return False
        return True

    async def _retire_previous(self) -> None:
        if self._previous is not None:
            _, viewport = self._previous
            self._previous = None
            await self._retire(viewport)

    async def _retire(self, viewport: ConsoleTranscript) -> None:
        if viewport.parent is self and not viewport._pruning:
            ensure_surface_swap_not_self_awaited((viewport,), seam="Console transcript")
            self._retiring = viewport.remove()
            await self._finish_retirement()

    async def _finish_retirement(self) -> None:
        if self._retiring is not None:
            await asyncio.shield(self._retiring)
            self._retiring = None

    async def project(self, screen: ChatScreen, *, force: bool = False) -> bool | None:
        """Project the live session, then publish that exact visit."""
        async with self._projection_lock:
            await self._finish_retirement()
            store = screen._ensure_console_chat_store()
            runtime = screen._console_runtime()
            generation = runtime._attached_generation
            authority = runtime.authority_token, runtime.profile_authority
            if authority != self._authority or self._store is not store:
                await self._retire_previous()
                self._active_key = None
                self._authority, self._store = authority, store
            epoch = store.active_session_epoch()
            session_id = store.active_session_id
            key = _TranscriptKey.from_store(store, session_id)
            viewport = self.transcript
            old_viewport = viewport

            def current() -> bool:
                return (
                    self.is_attached
                    and viewport.parent is self
                    and not viewport._pruning
                    and runtime.view is screen
                    and runtime._attached_generation == generation
                    and (runtime.authority_token, runtime.profile_authority)
                    == authority
                    and runtime.chat_store is store
                    and getattr(screen.app_instance, "console_runtime", None) is runtime
                    and not runtime._disposed
                    and store.active_session_epoch() == epoch
                    and store.active_session_id == session_id
                    and (
                        key.is_live_in(store) if key is not None else session_id is None
                    )
                )

            if not current():
                return False
            previous = self._previous
            if previous is not None and (
                previous[1].parent is not self
                or not previous[0].is_live_in(store)
                or self.settings.active
            ):
                await self._retire_previous()
                if not current():
                    return False
                previous = None

            old_key = self._active_key
            changing = key != old_key
            candidate = None
            try:
                retain_old = False
                if changing:
                    old_viewport._transcript_input_visit = None
                    retain_old = (
                        key is not None
                        and old_key is not None
                        and old_key.id != key.id
                        and old_key.is_live_in(store)
                        and self._can_retain(old_viewport)
                    )
                    self._clear_transient_input(old_viewport)
                    if (
                        previous is not None
                        and previous[0] == key
                        and self._can_retain(previous[1])
                    ):
                        viewport = previous[1]
                    else:
                        await self._retire_previous()
                        if not current():
                            return False
                        if retain_old:
                            new_id = (
                                "console-retained-transcript"
                                if old_viewport.id == "console-native-transcript"
                                else "console-native-transcript"
                            )
                            candidate = ConsoleTranscript(
                                id=new_id, classes="console-transcript-viewport"
                            )
                            candidate.background_effect_factory = (
                                self._make_background_effect
                            )
                            candidate.display = False
                            candidate.visible = False
                            candidate._transcript_surface = self
                            await self.mount(candidate)
                            viewport = candidate
                if not current():
                    return False
                viewport._transcript_surface = self
                viewport._projection_height = (
                    old_viewport.size.height or self.size.height
                )
                if viewport is not old_viewport:
                    viewport.note_follow_intent()
                project = screen._project_native_console_transcript
                acknowledge = (
                    await project(viewport, current, force=True)
                    if force
                    else await project(viewport, current)
                )
                if not current():
                    return False
                if viewport is not old_viewport:
                    await self._finish_markdown(viewport)
                    if not current() or self.settings.active:
                        return False
                    retain_old = retain_old and self._can_retain(old_viewport)
                    # Hit testing can still consult the old compositor map.
                    # Visibility covers that interval without a :disabled restyle.
                    old_viewport.visible = False
                    old_viewport.display = False
                    focused = screen.focused
                    if (
                        focused is not None
                        and old_viewport in focused.ancestors_with_self
                    ):
                        focused.blur()
                    viewport.styles.visibility = None
                    viewport.display = True
                    self.transcript = viewport
                    self._previous = (old_key, old_viewport) if retain_old else None
                self._active_key = key
                prior_visit = getattr(viewport, "_transcript_input_visit", None)
                if prior_visit is None or not prior_visit.is_current():

                    def input_current() -> bool:
                        return (
                            getattr(viewport, "_transcript_input_visit", None) is visit
                            and current()
                            and viewport is self.transcript
                            and viewport.display
                            and screen.is_active
                        )

                    visit = TranscriptInputVisit(get_time(), input_current)
                    viewport._transcript_input_visit = visit
                if acknowledge is not None:
                    acknowledge()
                screen._sync_console_transcript_guidance()
                screen._last_native_transcript_session_id = session_id
                if viewport is not old_viewport and not retain_old:
                    await self._retire(old_viewport)
                return None
            finally:
                if candidate is not None and candidate is not self.transcript:
                    await self._retire(candidate)

    @staticmethod
    def _clear_transient_input(viewport: ConsoleTranscript) -> None:
        if viewport.has_pending_selection_ui or selection_menus_on_screen(
            viewport.screen
        ):
            viewport._remove_selection_menu()
            viewport.selection_manager.cancel()
            if viewport._kb_selection_row is not None:
                viewport._exit_keyboard_selection()
            viewport._selection_origin_row = None
        captured = viewport.app.mouse_captured
        if captured is not None and viewport in captured.ancestors_with_self:
            viewport.app.capture_mouse(None)

    @staticmethod
    async def _finish_markdown(viewport: ConsoleTranscript) -> None:
        for markdown in viewport.walk_children(ConsoleRoleplayMarkdown):
            if (update := markdown.pending_update) is not None and not update.is_done:
                await asyncio.shield(update)
            if markdown.pending_update is update:
                markdown.pending_update = None

    def _make_background_effect(self) -> ConsoleBackgroundEffect:
        return ConsoleBackgroundEffect(
            self.settings,
            id="console-transcript-background-effect",
            classes="console-background-effect",
        )

    def compose(self) -> ComposeResult:
        yield self.transcript

    def update_settings(self, settings: ConsoleBackgroundEffectSettings) -> None:
        """Update the mounted background effect child when available."""
        self.settings = settings
        if not self.is_mounted or self.transcript.parent is not self:
            return
        try:
            effect = self.transcript.query_one(
                "#console-transcript-background-effect",
                ConsoleBackgroundEffect,
            )
        except NoMatches:
            return
        effect.update_settings(settings)


@dataclass(frozen=True)
class _TranscriptKey:
    id: str
    incarnation: str
    binding: int

    @classmethod
    def from_store(
        cls, store: ConsoleChatStore, session_id: str | None
    ) -> _TranscriptKey | None:
        session: ConsoleChatSession | None = store._sessions.get(session_id)
        if session is None:
            return None
        return cls(
            session.id, session.incarnation_id, session.conversation_binding_revision
        )

    def is_live_in(self, store: ConsoleChatStore) -> bool:
        session = store._sessions.get(self.id)
        return (
            session is not None
            and session.incarnation_id == self.incarnation
            and session.conversation_binding_revision == self.binding
        )


def _displayed_within(viewport: ConsoleTranscript, button: Button) -> bool:
    """Exclude the viewport's own visibility from its descendant check."""
    for node in button.ancestors_with_self:
        if node is viewport:
            return True
        if not node.display or not node.visible:
            return False
    return False


def active_console_transcript(owner: DOMNode) -> ConsoleTranscript | None:
    """Return the attached viewport selected by the transcript surface."""
    try:
        surface = owner.query_one(
            "#console-transcript-surface", ConsoleTranscriptSurface
        )
    except QueryError:
        surface = None
    if isinstance(surface, ConsoleTranscriptSurface):
        view = surface.transcript
        if view.parent is surface and view.is_attached and view.display:
            return view
        return None
    try:
        return owner.query_one("#console-native-transcript", ConsoleTranscript)
    except QueryError:
        return None


async def sync_console_transcript(
    screen: ChatScreen, *, force: bool = False
) -> bool | None:
    """Project a prepared view. False means live work or teardown deferred it."""
    from tldw_chatbook.UI.Console_Modules.transcript import prepare_change_review_projection
    from tldw_chatbook.UI.Screens.chat_screen import (
        ChatScreen, _console_screen_is_torn_down,
    )

    if not await prepare_change_review_projection(screen, _console_screen_is_torn_down):
        return False
    try:
        surface = screen.query_one(
            "#console-transcript-surface", ConsoleTranscriptSurface
        )
    except QueryError:
        surface = None
    if isinstance(surface, ConsoleTranscriptSurface):
        return await surface.project(screen, force=force)
    else:
        transcript = active_console_transcript(screen)
        acknowledge = await ChatScreen._project_native_console_transcript(
            screen, transcript, force=force
        )
        if transcript is not None:
            if acknowledge is not None:
                acknowledge()
            screen._sync_console_transcript_guidance()
            screen._last_native_transcript_session_id = (
                screen._ensure_console_chat_store().active_session_id
            )


def acknowledge_console_transcript(
    screen: ChatScreen,
    transcript: ConsoleTranscript,
    messages: Sequence[ConsoleChatMessage],
    conversation_id: str | None,
    refresh_key: tuple,
    gate: tuple | None,
) -> None:
    """Acknowledge exact mounted receipts once their projection is published."""
    rendered = screen._rendered_console_terminal_receipts(
        transcript, messages, conversation_id=conversation_id
    )
    runtime_accessor = getattr(screen, "_console_runtime", None)
    runtime = runtime_accessor() if callable(runtime_accessor) else None
    acknowledge = getattr(runtime, "acknowledge_rendered_terminal_receipts", None)
    if rendered and callable(acknowledge):
        acknowledge(
            rendered,
            view=screen,
            attachment_generation=getattr(
                screen, "_console_runtime_attachment_generation", None
            ),
        )
    screen._last_native_transcript_refresh_key = refresh_key
    if gate is not None:
        screen._console_transcript_sync_token = gate
