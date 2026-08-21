"""Base screen class for all application screens."""

import asyncio
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

    BUNDLED_CSS = """
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
        #: task-2854: the value handed to this screen's own ``MainNavigationBar``
        #: as ``active``/``active_route``. Defaults to ``screen_name`` -- today's
        #: behavior for every screen. A screen whose route is folded under
        #: another destination in ``shell_destinations.py`` for label/search
        #: purposes ONLY (e.g. Study is folded under Library, but renders none
        #: of Library's chrome -- no rail, no Library canvas) can override this
        #: to a value that resolves to no real destination (``""``), so its nav
        #: bar shows no highlighted tab instead of falsely claiming the owning
        #: destination is still on screen. The fold itself stays intact for
        #: every other consumer (Home's "Opens:" labels, the command palette's
        #: search aliases, screen routing) -- only this screen's OWN composed
        #: nav bar's highlight is affected.
        self.nav_bar_active: str = screen_name
        self._persona_buddy_view = None
        self._persona_buddy_view_generation = 0
        self._persona_buddy_reconcile_lock = asyncio.Lock()

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
        self.release_mouse_capture_for_teardown()
        await super().recompose()
        self.sweep_stale_mouse_capture()
        await self.reconcile_persona_buddy_view()

    def release_mouse_capture_for_teardown(self) -> None:
        """Release any mouse capture before removing widgets.

        Extracted from ``recompose`` (task-15475) so a screen that tears
        widgets down WITHOUT a screen recompose -- a region-scoped swap, which
        several screens now do instead of rebuilding themselves -- gets the
        same protection. The captured widget is not identified first: any
        teardown can orphan it, and ``Input`` has no ``_on_hide`` to release
        the mouse on removal, so the release stays unconditional exactly as
        ``recompose``'s own reasoning above requires.
        """
        if not self.is_running:
            return
        try:
            self.app.capture_mouse(None)
        except Exception:
            logger.debug("Mouse-capture release before teardown skipped.")

    def sweep_stale_mouse_capture(self) -> None:
        """Drop a capture left pointing at a no-longer-attached widget.

        The other half of ``release_mouse_capture_for_teardown``: a MouseDown
        already queued on a child's own pump can capture that child DURING the
        removal drain, after the pre-teardown release has run. ``is_attached``
        distinguishes that stale case from a capture a later, unrelated
        interaction legitimately holds (which must be left alone).
        """
        if not self.is_running:
            return
        captured = self.app.mouse_captured
        if captured is not None and not captured.is_attached:
            try:
                self.app.capture_mouse(None)
            except Exception:
                logger.debug("Stale post-teardown mouse-capture sweep skipped.")

    def compose(self) -> ComposeResult:
        """Compose the screen with navigation bar and content."""
        # Imported locally (not at module level): `AppFooterStatus` imports
        # `UI.Navigation.shortcut_context`, and `UI/Navigation/__init__.py`
        # eagerly imports THIS module -- a module-level import here would be
        # a circular import (base_app_screen -> AppFooterStatus ->
        # UI.Navigation package init -> base_app_screen, partially
        # initialized).
        from ...Widgets.AppFooterStatus import AppFooterStatus

        # Navigation bar at the top. task-2854: uses ``nav_bar_active`` (which
        # defaults to ``screen_name``, see ``__init__``), not ``screen_name``
        # directly, so a screen can opt out of a misleading destination
        # highlight without touching every other screen's behavior.
        yield MainNavigationBar(
            active=self.nav_bar_active, active_route=self.nav_bar_active
        )

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
        footer = AppFooterStatus(
            id="screen-footer-status",
            # task-17653: the footer token counter is retired — the Console
            # cost chip is the single token/cost surface, so no screen arms
            # the counter (chat used to, leaving it one write away from
            # duplicating the chip).
            show_token_count=False,
        )
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
        """Called when the screen is mounted.

        MRO contract: Textual's dispatcher invokes EVERY ``on_mount`` defined
        along the MRO for one Mount event, so a subclass handler must NOT
        call ``super().on_mount()`` -- that runs the parent handler a second
        time. A parent whose ``on_mount`` mounts widgets crashes with
        ``DuplicateIds`` when duplicated (TASK-2610: Lab > Speech). This base
        handler must therefore also stay idempotent: today it only logs, and
        anything heavier added here will run once per subclass that still
        carries a legacy ``super().on_mount()`` call.
        """
        logger.info(f"Screen {self.screen_name} mounted")
        self.call_after_refresh(self._schedule_persona_buddy_reconcile)

    def on_screen_resume(self) -> None:
        """Replay app-owned Buddy state after this screen is uncovered."""

        self.call_after_refresh(self._schedule_persona_buddy_reconcile)

    def on_unmount(self) -> None:
        """Called when the screen is unmounted."""
        self._persona_buddy_view_generation += 1
        view = self._persona_buddy_view
        self._persona_buddy_view = None
        if view is not None:
            view.release_interaction_capture()
        logger.info(f"Screen {self.screen_name} unmounted")

    @property
    def persona_buddy_view_generation(self) -> int:
        """Return this screen's current disposable Buddy-view generation."""

        return self._persona_buddy_view_generation

    def confirm_persona_buddy_unavailable(
        self,
        *,
        view: Any,
        controller: Any,
        snapshot: Any,
        visual: Any,
    ) -> bool:
        """Publish unavailable through the app's exact screen/view fence."""

        confirm = getattr(self.app_instance, "confirm_persona_buddy_unavailable", None)
        return bool(
            callable(confirm)
            and confirm(
                screen=self,
                view=view,
                view_generation=view.view_generation,
                controller=controller,
                snapshot=snapshot,
                visual=visual,
            )
        )

    def is_persona_buddy_confirmed_unavailable(
        self, controller: Any, snapshot: Any
    ) -> bool:
        """Query the app-owned marker for this exact controller authority."""

        unavailable = getattr(
            self.app_instance, "is_persona_buddy_confirmed_unavailable", None
        )
        return bool(callable(unavailable) and unavailable(controller, snapshot))

    def _schedule_persona_buddy_reconcile(self) -> None:
        """Schedule one idempotent mount reconciliation after screen paint."""

        if not self.is_attached:
            return
        self.run_worker(
            self.reconcile_persona_buddy_view,
            group="persona-buddy-view-reconcile",
            exclusive=True,
        )

    def sync_persona_buddy_reconciled_state(self) -> None:
        """Refresh screen-local Buddy affordances after view reconciliation."""

        return None

    async def reconcile_persona_buddy_view(self) -> bool:
        """Reconcile one generation; return whether no current view remains."""

        from ...Widgets.Persona_Widgets.persona_buddy_widget import (  # noqa: PLC0415
            PersonaBuddyWidget,
        )

        async with self._persona_buddy_reconcile_lock:
            controller = getattr(self.app_instance, "persona_buddy_controller", None)
            active = self.is_attached and self.app.screen is self
            snapshot = controller.snapshot() if controller is not None else None
            confirmed_unavailable = bool(
                snapshot is not None
                and self.is_persona_buddy_confirmed_unavailable(controller, snapshot)
            )
            desired = bool(
                active
                and snapshot is not None
                and snapshot.enabled
                and snapshot.open
                and snapshot.selection is not None
                and not confirmed_unavailable
            )

            current = self._persona_buddy_view
            if current is not None and not current.is_attached:
                current = None
                self._persona_buddy_view = None

            if not desired:
                self._persona_buddy_view_generation += 1
                if current is not None:
                    current.release_interaction_capture()
                    await current.remove()
                    if self._persona_buddy_view is current:
                        self._persona_buddy_view = None
                if self.is_attached and self.app.screen is self:
                    self.sync_persona_buddy_reconciled_state()
                return True

            if current is not None:
                current.refresh_from_controller()
                current.resume_resolution()
                self.sync_persona_buddy_reconciled_state()
                return False

            self._persona_buddy_view_generation += 1
            generation = self._persona_buddy_view_generation
            view = PersonaBuddyWidget(
                controller=controller,
                view_generation=generation,
                reconcile=self.app_instance.reconcile_persona_buddy_view,
                is_current=lambda candidate: bool(
                    generation == self._persona_buddy_view_generation
                    and self._persona_buddy_view is candidate
                    and self.is_attached
                    and self.app.screen is self
                ),
            )
            self._persona_buddy_view = view
            try:
                await self.mount(view)
            except BaseException:
                if self._persona_buddy_view is view:
                    self._persona_buddy_view = None
                if view.is_attached:
                    view.release_interaction_capture()
                    await view.remove()
                raise
            still_current = bool(
                generation == self._persona_buddy_view_generation
                and self._persona_buddy_view is view
                and self.is_attached
                and self.app.screen is self
            )
            if not still_current:
                view.release_interaction_capture()
                if view.is_attached:
                    await view.remove()
                if self._persona_buddy_view is view:
                    self._persona_buddy_view = None
                return True
            self.sync_persona_buddy_reconciled_state()
            return False
