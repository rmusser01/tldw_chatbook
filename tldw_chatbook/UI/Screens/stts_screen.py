"""
STTS (Speech-to-Text/Text-to-Speech) Screen
Screen wrapper for STTS functionality in screen-based navigation.
"""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from typing import Optional, TYPE_CHECKING
from loguru import logger

from ..Navigation.base_app_screen import BaseAppScreen
from ..STTS_Window import STTSWindow
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Workbench.workbench_widgets import DestinationHeader
from .lab_mode_strip import LabModeStrip

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class STTSScreen(BaseAppScreen):
    """Screen wrapper for Speech-to-Text/Text-to-Speech functionality."""

    #: Footer hint context (registered on mount; matches BINDINGS, ADR-031).
    STTS_SHORTCUTS: tuple[tuple[str, str], ...] = (
        ("g", "generate"),
        ("r", "random text"),
        ("x", "clear"),
        ("p", "play"),
        ("s", "stop"),
    )

    # Screen-level mirrors of TTSPlaygroundWidget.BINDINGS so the keys work
    # from the landed state (nav bar holds initial focus; widget bindings
    # only fire with in-window focus).
    BINDINGS = [
        Binding("g", "generate_tts", "Generate Speech", show=False),
        Binding("r", "random_text", "Random Text", show=False),
        Binding("x", "clear_text", "Clear Text", show=False),
        Binding("p", "play_audio", "Play Audio", show=False),
        Binding("s", "stop_audio", "Stop Audio", show=False),
    ]

    # Screen-specific state
    current_model: reactive[str] = reactive("")
    is_processing: reactive[bool] = reactive(False)
    audio_file_path: reactive[Optional[str]] = reactive(None)

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "stts", **kwargs)
        self.stts_window: Optional[STTSWindow] = None

    def _playground(self):
        """Return the playground widget, if mounted."""
        from ..STTS_Window import TTSPlaygroundWidget

        try:
            return self.query_one(TTSPlaygroundWidget)
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

    def compose_content(self) -> ComposeResult:
        """Compose the STTS screen with the STTS window and its destination header."""
        logger.info("Composing STTS screen")
        yield DestinationHeader(
            WorkbenchHeaderState(
                title="Speech",
                subtitle="Speech-to-text and text-to-speech tools.",
                status="ready",
                status_label="Speech tools ready",
            ),
            id="stts-destination-header",
        )
        yield LabModeStrip(active_route="stts", id="lab-mode-strip")
        self.stts_window = STTSWindow(self.app_instance, classes="window")
        # Leave room for the destination header above the window.
        self.stts_window.styles.height = "1fr"
        yield self.stts_window

    async def on_mount(self) -> None:
        """Initialize STTS services when screen is mounted."""
        logger.info("STTS screen mounted")
        self.register_footer_shortcuts(
            source="stts", shortcuts=self.STTS_SHORTCUTS
        )

        # Get the STTS window
        stts_window = self.stts_window or self.query_one(STTSWindow)

        # Initialize any services if needed
        if hasattr(stts_window, "initialize"):
            await stts_window.initialize()

    async def on_screen_suspend(self) -> None:
        """Clean up when screen is suspended (navigated away)."""
        logger.debug("STTS screen suspended")

        # Stop any ongoing audio processing
        if self.is_processing:
            stts_window = self.stts_window or self.query_one(STTSWindow)
            if hasattr(stts_window, "stop_processing"):
                await stts_window.stop_processing()
            self.is_processing = False

    async def on_screen_resume(self) -> None:
        """Restore state when screen is resumed."""
        logger.debug("STTS screen resumed")

        # Restore any necessary state
        stts_window = self.stts_window or self.query_one(STTSWindow)
        if hasattr(stts_window, "restore_state"):
            await stts_window.restore_state()
