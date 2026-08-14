"""Speech: the Lab destination's text-to-speech and speech-to-text screen."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.events import Click, Key
from textual.widget import Widget
from textual.widgets import Button, Static

from ...TTS import TTSPlaygroundSelectionPreset
from ...TTS.provider_ids import BUILT_IN_TTS_PROVIDER_IDS
from ..Lab_Modules.lab_speech_status import (
    SPEECH_CAPABILITY_SELECTOR,
    speech_capability_detail,
    speech_capability_text,
    speech_capability_tooltip,
    speech_local_dependency_availability,
)
from ..Lab_Modules.lab_workbench import LAB_RAIL_ROW_CLASS
from ..Speech.speech_playground_model import AXIS_CONTROLS
from ..Speech.speech_runtime_status import (
    speech_tts_navigation_target_from_context,
)
from ..Speech.speech_settings_contracts import SpeechTTSNavigationTarget
from ..STTS_Window import STTS_VIEW_KEYS, STTSWindow
from ..Workbench.workbench_state import WorkbenchHeaderState
from .lab_frame import LabScreen

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli

#: (section title, ((view key, label), ...)) in rail order.
#:
#: The emoji are deliberate, not decoration: they were the sidebar's only
#: per-item visual anchor, and dropping them was one of the things that made
#: the first attempt at this screen unreadable.
#:
#: View keys map to ``STTSWindow.current_view`` except the two that switch no
#: view at all -- ``voice-cloning`` pushes its own screen and ``effects`` is
#: unbuilt. Both are handled in ``_handle_rail_press``.
SPEECH_RAIL_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "Speech",
        (
            ("playground", "🎤 TTS Playground"),
            ("profiles", "🗣️ Voice Profiles"),
            ("settings", "⚙️ Studio TTS Preferences"),
            ("audiobook", "📚 AudioBook/Podcast"),
        ),
    ),
    (
        "Additional features",
        (
            ("voice-cloning", "🎙️ Voice Cloning"),
            ("dictation", "🔤 Speech Recognition"),
            ("effects", "🎵 Audio Effects"),
        ),
    ),
)

#: Rail rows that do not correspond to an ``STTSWindow.current_view``.
#: `voice-cloning` pushes its own screen rather than switching the view.
#: `effects` used to be here too, as a disabled row that opened nothing;
#: it has a placeholder view now, which explains itself.
#: Rail rows with no `STTSWindow.current_view` behind them. Empty now:
#: `effects` gained a placeholder view and `voice-cloning` became a
#: view instead of a pushed screen. Kept so the next such row has a
#: home, and so the branch that handles them stays exercised.
SPEECH_NON_VIEW_KEYS: frozenset[str] = frozenset()
_SPEECH_PLAYGROUND_AXES_STATE_KEY = "speech_playground_axes"
_MAX_PROCESS_LOCAL_AXIS_LENGTH = 4096


def _bounded_playground_axes(value: object) -> dict[str, str]:
    """Accept only bounded comparison axes for process-local screen restore."""

    if not isinstance(value, Mapping):
        return {}
    axes: dict[str, str] = {}
    for control_id in AXIS_CONTROLS:
        candidate = value.get(control_id)
        if (
            type(candidate) is not str
            or not candidate
            or len(candidate) > _MAX_PROCESS_LOCAL_AXIS_LENGTH
            or any(
                unicodedata.category(character) in {"Cc", "Cf", "Cs"}
                for character in candidate
            )
        ):
            continue
        axes[control_id] = candidate
    if axes.get("tts-provider-select") not in BUILT_IN_TTS_PROVIDER_IDS:
        return {}
    return axes


class STTSScreen(LabScreen):
    """Speech mode: view rail, capability line, and the legacy STTS window.

    The ``DestinationHeader`` above the rail is composed by the ``LabScreen``
    frame from this mode's ``lab_header_state()`` (a ``WorkbenchHeaderState``)
    and re-synced on every ``refresh_lab_status()`` pass.
    """

    #: Footer hint context (registered on mount; matches BINDINGS, ADR-031).
    #: The inherited Lab mode hints (``[ / ]``/``Enter``) register from
    #: ``LabScreen.on_mount``; these are the Speech playground keys.
    STTS_SHORTCUTS: tuple[tuple[str, str], ...] = (
        ("g", "generate"),
        ("r", "random text"),
        ("x", "clear"),
        ("p", "play"),
        ("s", "stop"),
    )

    # Screen-level plain-letter shortcuts for the mounted playground's
    # action_* methods (SpeechPlaybackMixin/SpeechSynthesisMixin), invoked
    # directly via `_playground()` rather than through the pane's own
    # BINDINGS (which use ctrl+ combos -- see speech_playground_pane.py).
    # These exist so the keys work from the landed state: the nav rail
    # holds initial focus, and a binding declared only on the pane would
    # never fire without in-pane focus.
    BINDINGS = [
        Binding("g", "generate_tts", "Generate Speech", show=False),
        Binding("r", "random_text", "Random Text", show=False),
        Binding("x", "clear_text", "Clear Text", show=False),
        Binding("p", "play_audio", "Play Audio", show=False),
        Binding("s", "stop_audio", "Stop Audio", show=False),
    ]

    def __init__(self, app_instance: "TldwCli", **kwargs: Any) -> None:
        """Create the Speech screen.

        Args:
            app_instance: The running application.
            kwargs: Forwarded to ``LabScreen``.
        """
        super().__init__(app_instance, "stts", **kwargs)
        self.stts_window: STTSWindow | None = None
        self._pending_navigation_context: (
            tuple[
                str,
                TTSPlaygroundSelectionPreset | None,
                SpeechTTSNavigationTarget | None,
            ]
            | None
        ) = None
        self._restored_playground_axes: dict[str, str] = {}
        self._speech_local_dependencies = None

    def _lab_footer_registration(self) -> tuple[str, tuple]:
        """Register the Speech hints in place of the frame's plain set.

        This screen must NOT define ``on_mount``: Textual dispatches every
        ``on_mount`` in the MRO for one Mount event, so the previous
        ``super().on_mount()`` here ran the Lab frame's handler twice --
        double-mounting the rail rows and crashing the app with
        ``DuplicateIds`` on every visit to Lab > Speech (TASK-2610).
        """
        return ("stts", self.STTS_SHORTCUTS + self.LAB_FOOTER_SHORTCUTS)

    def _playground(self):
        """Return the mounted playground pane, if any.

        ``STTSWindow._mount_view`` only ever mounts ``SpeechPlaygroundPane``
        for the ``playground`` view (this used to query the retired legacy
        playground widget -- TASK-2951 -- which was never mounted in
        production, making every mirrored action below a permanent no-op).
        """
        from ..Speech.speech_playground_pane import SpeechPlaygroundPane

        try:
            return self.query_one(SpeechPlaygroundPane)
        except Exception:  # noqa: BLE001 - playground not mounted
            return None

    def action_generate_tts(self) -> None:
        if widget := self._playground():
            widget.action_generate_tts()

    def action_random_text(self) -> None:
        if widget := self._playground():
            widget.action_random_text()

    def action_clear_text(self) -> None:
        if widget := self._playground():
            widget.action_clear_text()

    def action_play_audio(self) -> None:
        if widget := self._playground():
            widget.action_play_audio()

    def action_stop_audio(self) -> None:
        if widget := self._playground():
            widget.action_stop_audio()

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return the Speech header copy and derived readiness.

        Returns:
            Ready destination state. Individual local capabilities report
            their own availability and do not gate external providers.
        """
        return WorkbenchHeaderState(
            title="Speech",
            subtitle="Speech-to-text and text-to-speech tools.",
            status="ready",
        )

    def compose_lab_rail(self) -> ComposeResult:
        """Yield the two rail sections and their seven view rows."""
        dependencies = speech_local_dependency_availability(refresh=True)
        self._speech_local_dependencies = dependencies
        summary = Static(
            speech_capability_text(dependencies),
            id="speech-capability-summary",
            classes="speech-capability-status",
            markup=False,
        )
        summary.tooltip = speech_capability_tooltip(dependencies)
        yield summary

        for title, entries in SPEECH_RAIL_SECTIONS:
            yield Static(title, classes="lab-rail-section")
            for view_key, label in entries:
                row = Button(
                    label,
                    id=f"lab-speech-row-{view_key}",
                    classes=LAB_RAIL_ROW_CLASS,
                    # Audio Effects has no implementation. An enabled button
                    # whose only handler toasts "coming soon" is the
                    # dead-end-toast pattern; disabling says it once, in the
                    # control itself, exactly as the old sidebar did.
                )
                # Carried as an attribute rather than parsed back out of the
                # id, mirroring LLMScreen's lab_view_key.
                row.lab_view_key = view_key
                yield row

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield the local-speech recovery detail.

        Carries ``SPEECH_CAPABILITY_SELECTOR`` because this is the widget
        holding the recovery copy that selector names -- headline, why, the
        exact pip command, and where to go next. The inspector is the frame's
        region for exactly this: detail that must stay reachable without
        hovering, but must not crowd the rail.
        """
        dependencies = self._speech_local_dependencies
        if dependencies is None:
            dependencies = speech_local_dependency_availability(refresh=True)
            self._speech_local_dependencies = dependencies
        yield Static("Local speech", classes="lab-rail-section")
        yield Static(
            speech_capability_detail(dependencies),
            id=SPEECH_CAPABILITY_SELECTOR,
            classes="speech-capability-status",
            markup=False,
        )

    def build_lab_body(self) -> Widget:
        """Build the body: the window, which owns view switching.

        Returning the playground pane directly -- as this did while the
        rebuild was the only redesigned view -- left `self.stts_window` None
        forever, and every rail press hit its `is None` guard and did
        nothing. TTS Settings, AudioBook and Speech Recognition were all
        unreachable; only Voice Cloning worked, because it pushes a screen
        before that check.

        The window mounts `SpeechPlaygroundPane` for the playground view
        itself, so the rebuild is still what the user lands on.

        Returns:
            The ``STTSWindow``, mounted after first paint like every Lab
            body.
        """
        dependencies = self._speech_local_dependencies
        if dependencies is None:
            dependencies = speech_local_dependency_availability(refresh=True)
            self._speech_local_dependencies = dependencies
        self.stts_window = STTSWindow(
            self.app_instance,
            classes="window",
            playground_axis_values=self._restored_playground_axes,
            local_dependencies=dependencies,
        )
        self.stts_window.styles.height = "1fr"
        return self.stts_window

    def save_state(self) -> dict[str, object]:
        """Save only bounded process-local Playground comparison axes."""

        state = dict(super().save_state() or {})
        axes = self._restored_playground_axes
        if self.stts_window is not None:
            axes = self.stts_window.playground_axis_snapshot()
        bounded = _bounded_playground_axes(axes)
        if bounded:
            state[_SPEECH_PLAYGROUND_AXES_STATE_KEY] = bounded
        else:
            state.pop(_SPEECH_PLAYGROUND_AXES_STATE_KEY, None)
        return state

    def restore_state(self, state: dict[str, object]) -> None:
        """Seed bounded axes before the fresh deferred Speech body mounts."""

        super().restore_state(state)
        self._restored_playground_axes = _bounded_playground_axes(
            state.get(_SPEECH_PLAYGROUND_AXES_STATE_KEY)
            if isinstance(state, Mapping)
            else None
        )

    def on_lab_body_ready(self) -> None:
        """Bind the rail highlight to the window's ``current_view``.

        Registered here because the window does not exist earlier, and
        re-registered against the fresh instance after a screen-level
        recompose -- exactly as ``LLMScreen`` does. ``init=True`` seeds the
        highlight on arrival, which matters because ``STTSWindow`` sets
        ``current_view`` itself rather than waiting for a press.
        """
        if self.stts_window is None:
            # Redesigned panes own their own state; nothing to bind yet.
            return
        self.watch(self.stts_window, "current_view", self._sync_rail_active, init=True)
        self._apply_pending_navigation_context()

    def apply_navigation_context(self, context: Mapping[str, object]) -> None:
        """Retain one validated process-local Speech destination request."""

        if not isinstance(context, Mapping):
            return
        keys = set(context)
        view = context.get("view")
        if type(view) is not str or view not in STTS_VIEW_KEYS:
            return
        has_preset = "profile_preset" in context
        preset = context.get("profile_preset")
        if has_preset and (
            keys != {"view", "profile_preset"}
            or view != "playground"
            or type(preset) is not TTSPlaygroundSelectionPreset
        ):
            return
        navigation_target: SpeechTTSNavigationTarget | None = None
        if not has_preset and keys != {"view"}:
            if view != "playground" or not keys.issubset(
                {"view", "provider", "intent"}
            ):
                return
            navigation_target = speech_tts_navigation_target_from_context(
                {key: value for key, value in context.items() if key != "view"}
            )
            if navigation_target is None:
                return
        exact_preset = preset if has_preset else None
        self._pending_navigation_context = (
            view,
            exact_preset,
            navigation_target,
        )
        self._apply_pending_navigation_context()

    def _apply_pending_navigation_context(self) -> None:
        window = self.stts_window
        context = self._pending_navigation_context
        if window is None or context is None:
            return
        self._pending_navigation_context = None
        view, preset, navigation_target = context
        self.run_worker(
            window.request_view(
                view,
                profile_preset=preset,
                navigation_target=navigation_target,
            ),
            group="speech-view-navigation",
            exclusive=True,
            exit_on_error=False,
        )

    def _sync_rail_active(self, current_view: str) -> None:
        """Move the rail highlight to the row matching the active view.

        Args:
            current_view: The window's current view key.
        """
        for row in self.query(f".{LAB_RAIL_ROW_CLASS}").results(Button):
            row.set_class(
                getattr(row, "lab_view_key", None) == current_view, "is-active"
            )

    def on_key(self, _event: Key) -> None:
        """Do not let deferred profile restoration override keyboard intent."""

        if self.stts_window is not None:
            self.stts_window.cancel_profile_focus_restore()

    def on_click(self, _event: Click) -> None:
        """Do not let deferred profile restoration override pointer intent."""

        if self.stts_window is not None:
            self.stts_window.cancel_profile_focus_restore()

    @on(Button.Pressed, f".{LAB_RAIL_ROW_CLASS}")
    def _handle_rail_press(self, event: Button.Pressed) -> None:
        """Route a rail press to a view switch, or to its own action.

        The rows are this screen's children now, so ``STTSWindow``'s own
        ``on_button_pressed`` never sees them and its sidebar branches are
        unreachable from here -- this method owns that routing. That
        handler's *else* branch, which forwards presses to the active
        content widget, is untouched and still load-bearing.
        """
        event.stop()
        view_key = getattr(event.button, "lab_view_key", None)
        if view_key is None:
            return

        if view_key in SPEECH_NON_VIEW_KEYS:
            # `effects` composes disabled, so this is unreachable through the
            # UI; it remains the explicit "no view behind this key" branch.
            return

        if self.stts_window is None:
            logger.warning("Speech rail pressed before the body mounted; ignored.")
            return
        self.run_worker(
            self.stts_window.request_view(view_key),
            group="speech-view-navigation",
            exclusive=True,
            exit_on_error=False,
        )

    async def flush_pending_work(self) -> bool:
        """Protect a dirty Studio preference draft before screen navigation."""

        if self.stts_window is None:
            return True
        return await self.stts_window.confirm_studio_preferences_leave()
