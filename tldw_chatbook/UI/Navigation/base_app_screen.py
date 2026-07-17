"""Base screen class for all application screens."""

from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger

from textual.app import ComposeResult
from textual.geometry import Region
from textual.screen import Screen
from textual.containers import Container

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
        #: (source, shortcuts) persisted so footer hints survive recompose.
        self._footer_shortcut_registration: Optional[tuple] = None

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
        # Imported locally (not at module level): `AppFooterStatus` imports
        # `UI.Navigation.shortcut_context`, and `UI/Navigation/__init__.py`
        # eagerly imports THIS module -- a module-level import here would be
        # a circular import (base_app_screen -> AppFooterStatus ->
        # UI.Navigation package init -> base_app_screen, partially
        # initialized).
        from ...Widgets.AppFooterStatus import AppFooterStatus

        # Navigation bar at the top
        yield MainNavigationBar(active=self.screen_name, active_route=self.screen_name)

        # Content area below navigation
        with Container(id="screen-content"):
            yield from self.compose_content()

        # Per-screen footer status bar (task-264): the App only ever mounts
        # ONE Footer-equivalent widget on its DEFAULT screen (app.py's own
        # compose()), which is occluded the moment any BaseAppScreen is
        # pushed on top -- `App.query_one`/`query` always resolve against
        # `App.default_screen` by design (see `App._get_dom_base`), so a
        # caller doing `self.app.query_one(AppFooterStatus)` from within a
        # pushed screen silently updates an invisible widget. Composing an
        # `AppFooterStatus` here gives every screen its OWN instance that
        # `self.query_one(AppFooterStatus)` (queried against the screen
        # itself) correctly resolves.
        footer = AppFooterStatus(id="screen-footer-status")
        # Screen-level recompose (settings' recompose=True reactives,
        # library/chat `refresh(recompose=True)` calls) re-runs THIS method
        # and replaces the footer with a fresh instance -- re-seed the
        # persisted registration so hints survive recompose. Safe pre-mount:
        # `set_workbench_shortcuts` updates child Statics the footer holds
        # as instance attributes.
        registration = getattr(self, "_footer_shortcut_registration", None)
        if registration is not None:
            footer.set_workbench_shortcuts(
                source=registration[0], shortcuts=registration[1]
            )
        yield footer
    
    def compose_content(self) -> ComposeResult:
        """Override in subclasses to provide screen-specific content."""
        yield Container()  # Default empty container

    def register_footer_shortcuts(
        self, *, source: str, shortcuts: tuple
    ) -> None:
        """Register a workbench shortcut set with this screen's footer.

        The registration is persisted on the screen so it survives a
        screen-level recompose (which replaces the footer widget -- see
        ``compose()``). Screens with a STATIC hint set should use this
        instead of talking to the footer directly; a screen whose context is
        dynamic and re-registered on every state transition (personas) may
        still drive ``set_shortcut_context`` itself.

        Args:
            source: Context owner tag (e.g. "console"); scopes clears.
            shortcuts: ``((key, label), ...)`` pairs to render.
        """
        registration = (source, tuple(shortcuts))
        self._footer_shortcut_registration = registration
        footer = self._footer_status()
        if footer is not None:
            footer.set_workbench_shortcuts(
                source=registration[0], shortcuts=registration[1]
            )

    def clear_footer_shortcuts(self, *, source: str) -> None:
        """Clear this screen's footer hints if ``source`` still owns them.

        Mirrors ``AppFooterStatus.clear_shortcut_context``'s source guard for
        the persisted copy, so a stale suspend cannot drop a newer owner's
        registration.
        """
        registration = getattr(self, "_footer_shortcut_registration", None)
        if registration is not None and registration[0] == source:
            self._footer_shortcut_registration = None
        footer = self._footer_status()
        if footer is not None:
            footer.clear_shortcut_context(source=source)

    def _footer_status(self):
        """This screen's own AppFooterStatus, or None before compose."""
        from ...Widgets.AppFooterStatus import AppFooterStatus  # noqa: PLC0415 -- circular (see compose)
        from textual.css.query import QueryError

        try:
            return self.query_one(AppFooterStatus)
        except QueryError:
            return None

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
