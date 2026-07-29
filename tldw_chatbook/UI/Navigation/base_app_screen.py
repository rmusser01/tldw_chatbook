"""Base screen class for all application screens."""

from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger

from textual.app import ComposeResult
from textual.geometry import Region
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Static

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

    def __init__(self, app_instance: "TldwCli", screen_name: str, **kwargs):
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
                logger.debug(
                    "Mouse-capture release before recompose skipped.", exc_info=True
                )
        return super().refresh(
            *regions, repaint=repaint, layout=layout, recompose=recompose
        )

    async def recompose(self) -> None:
        """Release any mouse capture again immediately before the actual
        teardown -- task-627.

        ``refresh(recompose=True)`` (overridden above) already releases
        capture at the moment it is CALLED, but Textual's own
        ``Widget.refresh(recompose=True)`` only *schedules* the real
        teardown (``self.call_next(self._check_recompose)``) -- it runs on
        a LATER iteration of the message loop, not synchronously. Live UAT
        (task-627) reproduced the exact "every mouse click silently
        swallowed app-wide" symptom the ``refresh()`` guard above was
        supposed to prevent: reproduced headlessly by injecting a NEW
        ``capture_mouse()`` call in that exact window (after ``refresh()``
        released the OLD capture, before the deferred recompose actually
        ran) and confirming ``App.mouse_captured`` was left pointing at the
        (now torn-down) widget afterward -- i.e. a real, exploitable gap:
        anything that captures the mouse in that window (a MouseDown on an
        Input/TextArea/ScrollBar arriving as a separately-timed message --
        entirely plausible over a laggy transport where down/up travel
        independently, as this app's textual-serve-driven UAT sessions do)
        leaks exactly like the original bug, since the earlier guard only
        ever checks capture state once, at ``refresh()``-call time.

        Overriding ``recompose()`` itself -- the coroutine Textual's
        deferred ``_check_recompose`` actually calls to perform the
        teardown -- releases capture as the very first synchronous
        statement of that same coroutine. asyncio only yields control at
        ``await`` points, so nothing else in the event loop can run between
        this release and ``super().recompose()`` initiating the real
        ``remove()``/``mount_all()`` teardown below it: this NARROWS the
        window to the teardown drain itself, it does not close it entirely
        (post-review correction, task-627: an EARLIER draft of this
        docstring overclaimed "closed entirely" -- a code-review probe
        proved that wrong). ``super().recompose()``'s own
        ``query_children("*")...remove()`` await lets each child's message
        pump drain before it's actually pruned; a message ALREADY queued on
        a CHILD's own pump before this method ever ran (e.g. a forwarded
        MouseDown not yet dispatched) can still be processed DURING that
        drain -- ``Input._on_mouse_down`` calls ``capture_mouse()``
        unconditionally, and ``Widget.capture_mouse()`` has no attachment
        guard, so it happily re-captures a widget that is mid-removal.
        Recomposing ALWAYS removes and remounts every child regardless of
        which specific widget currently holds capture (mirrors the
        `refresh()` guard's own reasoning), so the pre-teardown release
        above stays unconditional rather than trying to identify whether
        the captured widget is actually a descendant.

        The sweep below closes that residual gap: once ``recompose()`` has
        fully finished (removal AND remount both done), a capture that
        landed during the drain is by definition now pointing at a
        NO-LONGER-ATTACHED widget (nothing legitimately mounted during
        remount would already be captured) -- ``is_attached`` distinguishes
        that stale case from a widget a *later*, entirely unrelated
        interaction has since legitimately captured (which must be left
        alone).
        """
        if self.is_running:
            try:
                self.app.capture_mouse(None)
            except Exception:
                logger.debug(
                    "Mouse-capture release before recompose teardown skipped.",
                    exc_info=True,
                )
        await super().recompose()
        if self.is_running:
            captured = self.app.mouse_captured
            if captured is not None and not captured.is_attached:
                try:
                    self.app.capture_mouse(None)
                except Exception:
                    logger.debug(
                        "Stale post-recompose mouse-capture sweep skipped.",
                        exc_info=True,
                    )

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
            yield from self._compose_content_or_failure()

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

    def _compose_content_or_failure(self) -> ComposeResult:
        """Compose this screen's content, degrading to an error panel on failure.

        A destination that cannot build its own body must not take the app
        down with it. Textual composes a screen inside its mount pipeline, so
        an exception raised in ``compose_content`` is NOT raised back to
        whoever called ``switch_screen`` -- Textual records it on the App and
        exits the process. The navigation handler's try/except therefore
        cannot see it, and this is the only place that can.

        Concretely: the MCP canvases read ``Select.NULL`` (Textual 8+) while
        composing, so on an older Textual clicking MCP killed the whole app.

        Widgets already yielded before the failure stay mounted -- a partly
        built screen with a visible explanation beats a dead application.

        Returns:
            The subclass's content, or an error panel describing the failure.
        """
        try:
            yield from self.compose_content()
        except Exception as exc:
            logger.opt(exception=True).error(
                "Screen content failed to compose "
                f"(screen={self.screen_name!r}, exception_category={type(exc).__name__})."
            )
            yield Container(
                Static(
                    f"This screen failed to load.\n\n"
                    f"{type(exc).__name__}: {exc}\n\n"
                    "The rest of the app is unaffected -- use the navigation "
                    "bar above to go elsewhere. Details are in the log.",
                    id="screen-content-error-message",
                ),
                id="screen-content-error",
            )

    def compose_content(self) -> ComposeResult:
        """Override in subclasses to provide screen-specific content."""
        yield Container()  # Default empty container

    def register_footer_shortcuts(self, *, source: str, shortcuts: tuple) -> None:
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
