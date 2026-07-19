"""Base screen class for all application screens."""

from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger

from textual.app import ComposeResult
from textual.geometry import Region
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Footer

from .main_navigation import MainNavigationBar

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class BaseAppScreen(Screen):
    """
    Base screen class for all application screens.
    Provides common functionality like navigation bar and state management.
    """

    DEFAULT_CSS = """
    BaseAppScreen {
        background: $background;
    }

    #screen-content {
        width: 100%;
        height: 1fr;
        min-height: 0;
        padding-top: 0;
    }
    """

    def __init__(self, app_instance: 'TldwCli', screen_name: str, **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.screen_name = screen_name
        self.state_data: Dict[str, Any] = {}

        logger.debug(f"Initializing {self.__class__.__name__} screen: {screen_name}")

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> "BaseAppScreen":
        """Recompose, releasing any stale mouse capture first.

        ``Widget.recompose()`` (what ``refresh(recompose=True)`` schedules)
        unconditionally removes and remounts every child. If ``App.
        mouse_captured`` currently points at one of those children -- e.g. an
        ``Input`` mid click/selection whose ``MouseUp`` hasn't arrived yet
        (plausible over textual-serve's websocket transport, where down/up
        travel as independently-timed messages) -- ``Input`` has no
        ``_on_hide`` handler to release the mouse on removal (unlike
        ``TextArea``/``ScrollBar``, Textual's other two mouse-capturing
        widgets, which both do). The result: ``mouse_captured`` is left
        referencing a removed widget forever. From then on EVERY mouse event
        anywhere in the app -- routed through ``Screen._forward_event``/
        ``_handle_mouse_move``, both of which special-case ``if self.app.
        mouse_captured: ... self.find_widget(widget)`` -- hits ``NoWidget``
        and is silently swallowed, permanently breaking click dispatch
        app-wide (keyboard input is unaffected: it never consults
        ``mouse_captured``). Only a real screen switch self-heals this,
        because ``App.push_screen``/``switch_screen``/``_replace_screen``
        already defensively call ``capture_mouse(None)`` before swapping
        screens -- but a same-screen ``BaseAppScreen`` content recompose
        (used throughout, e.g. the Library skills/prompts/notes in-canvas
        editors reopening via ``self.refresh(recompose=True)``) never got
        that same protection, so the app can get stuck with no screen switch
        able to fire either. Releasing the capture here, mirroring that
        existing Textual idiom, closes the gap at its root: any widget about
        to be recomposed away is released *before* it can be orphaned.
        """
        if recompose and self.is_running:
            try:
                self.app.capture_mouse(None)
            except Exception:
                logger.debug("Mouse-capture release before recompose skipped.", exc_info=True)
        return super().refresh(*regions, repaint=repaint, layout=layout, recompose=recompose)

    def compose(self) -> ComposeResult:
        """Compose the screen with navigation bar and content."""
        # Navigation bar at the top
        yield MainNavigationBar(active=self.screen_name, active_route=self.screen_name)
        
        # Content area below navigation
        with Container(id="screen-content"):
            yield from self.compose_content()

        # The app-level Ctrl+P "Palette Menu" binding is show=True, so it
        # already renders in the footer's binding list; the Footer's built-in
        # right-corner command-palette pill would duplicate it (UAT 2026-07:
        # "^p Palette Menu" shown twice).
        yield Footer(show_command_palette=False)
    
    def compose_content(self) -> ComposeResult:
        """Override in subclasses to provide screen-specific content."""
        yield Container()  # Default empty container
    
    def save_state(self) -> Dict[str, Any]:
        """Save the current state of the screen."""
        # Override in subclasses to save specific state
        return self.state_data
    
    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore a previously saved state."""
        # Override in subclasses to restore specific state
        self.state_data = state
    
    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        logger.info(f"Screen {self.screen_name} mounted")
    
    def on_unmount(self) -> None:
        """Called when the screen is unmounted."""
        logger.info(f"Screen {self.screen_name} unmounted")
