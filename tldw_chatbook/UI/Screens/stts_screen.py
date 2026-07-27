"""Speech: the Lab destination's text-to-speech and speech-to-text screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Button, Static

from ..Lab_Modules.lab_speech_status import (
    SPEECH_CAPABILITY_SELECTOR,
    speech_capability_detail,
    speech_capability_text,
    speech_capability_tooltip,
    speech_dependencies_available,
)
from ..Lab_Modules.lab_workbench import LAB_RAIL_ROW_CLASS
from ..STTS_Window import STTSWindow
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
            ("settings", "⚙️ TTS Settings"),
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
SPEECH_NON_VIEW_KEYS = frozenset({"voice-cloning", "effects"})


class STTSScreen(LabScreen):
    """Speech mode: view rail, capability line, and the legacy STTS window."""

    def __init__(self, app_instance: "TldwCli", **kwargs: Any) -> None:
        """Create the Speech screen.

        Args:
            app_instance: The running application.
            kwargs: Forwarded to ``LabScreen``.
        """
        super().__init__(app_instance, "stts", **kwargs)
        self.stts_window: STTSWindow | None = None

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return the Speech header copy and derived readiness.

        Returns:
            Header state reading ``ready`` only when both local speech
            dependency groups import -- the same condition the rail's
            capability line states in words, rather than a constant.
        """
        return WorkbenchHeaderState(
            title="Speech",
            subtitle="Speech-to-text and text-to-speech tools.",
            status="ready" if speech_dependencies_available() else "blocked",
        )

    def compose_lab_rail(self) -> ComposeResult:
        """Yield the two rail sections and their six view rows."""
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
                    disabled=view_key == "effects",
                )
                # Carried as an attribute rather than parsed back out of the
                # id, mirroring LLMScreen's lab_view_key.
                row.lab_view_key = view_key
                yield row

        # One line, stating the fact. The full recovery taxonomy is ~14
        # rendered lines; inline here it buried the six rows above it, so it
        # lives in the inspector instead (compose_lab_inspector below).
        summary = Static(
            speech_capability_text(),
            id="speech-capability-summary",
            classes="speech-capability-status",
            markup=False,
        )
        summary.tooltip = speech_capability_tooltip()
        yield summary

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield the local-speech recovery detail.

        Carries ``SPEECH_CAPABILITY_SELECTOR`` because this is the widget
        holding the recovery copy that selector names -- headline, why, the
        exact pip command, and where to go next. The inspector is the frame's
        region for exactly this: detail that must stay reachable without
        hovering, but must not crowd the rail.
        """
        yield Static("Local speech", classes="lab-rail-section")
        yield Static(
            speech_capability_detail(),
            id=SPEECH_CAPABILITY_SELECTOR,
            classes="speech-capability-status",
            markup=False,
        )

    def build_lab_body(self) -> Widget:
        """Build the legacy STTS window.

        Returns:
            The ``STTSWindow``, mounted after first paint like every Lab body.
        """
        self.stts_window = STTSWindow(self.app_instance, classes="window")
        self.stts_window.styles.height = "1fr"
        return self.stts_window

    def on_lab_body_ready(self) -> None:
        """Bind the rail highlight to the window's ``current_view``.

        Registered here because the window does not exist earlier, and
        re-registered against the fresh instance after a screen-level
        recompose -- exactly as ``LLMScreen`` does. ``init=True`` seeds the
        highlight on arrival, which matters because ``STTSWindow`` sets
        ``current_view`` itself rather than waiting for a press.
        """
        if self.stts_window is None:
            return
        self.watch(
            self.stts_window, "current_view", self._sync_rail_active, init=True
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

        if view_key == "voice-cloning":
            from ..Voice_Cloning_Window import VoiceCloningWindow

            self.app.push_screen(VoiceCloningWindow())
            return
        if view_key in SPEECH_NON_VIEW_KEYS:
            # `effects` composes disabled, so this is unreachable through the
            # UI; it remains the explicit "no view behind this key" branch.
            return

        if self.stts_window is None:
            logger.warning("Speech rail pressed before the body mounted; ignored.")
            return
        self.stts_window.current_view = view_key
