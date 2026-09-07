"""Base screen class for all application screens."""

from typing import TYPE_CHECKING, Optional, Dict, Any
from loguru import logger

from textual.message import Message
from textual.app import ComposeResult
from textual.css.query import QueryError
from textual.geometry import Region
from textual.screen import Screen
from textual.containers import Container
from textual.widgets import Static

from .main_navigation import MainNavigationBar

if TYPE_CHECKING:
    from textual.widget import Widget

    from tldw_chatbook.app import TldwCli


class BaseAppScreen(Screen):
    """
    Base screen class for all application screens.
    Provides common functionality like navigation bar and state management.
    """

    class ContentsRebuilt(Message):
        """Report completion of a primary screen's child reconstruction."""

        def __init__(self, screen: "BaseAppScreen") -> None:
            super().__init__()
            self.screen = screen

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

    #: task-31825: below this shell height, DestinationHeader's stacked
    #: title/subtitle/status block (5 of 24 rows at the 80x24 floor --
    #: measured live in task-31419) gives up its subtitle row. Pinned to
    #: the floor tests' own 24 so "at/below the floor" and "compact" mean
    #: the same terminal size everywhere in this program.
    _DESTINATION_HEADER_COMPACT_FLOOR_HEIGHT = 24

    #: Textual's own breakpoint mechanism (`Screen._on_resize` ->
    #: `update_classes`, evaluated on every resize AND on the screen's own
    #: first layout pass -- a freshly pushed screen posts itself a Resize
    #: the moment it gets a real size, so this is already correct before
    #: the first paint, not just after a later resize). Every screen shares
    #: this base, so every DestinationHeader user gets the trigger with no
    #: per-screen code (`.shell-header-compact .workbench-header` in
    #: components/_workbench.tcss is the only other half of the wiring).
    VERTICAL_BREAKPOINTS = [
        (0, "shell-header-compact"),
        (_DESTINATION_HEADER_COMPACT_FLOOR_HEIGHT + 1, "shell-header-normal"),
    ]

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

        task-31946: this is also the ONE seam that puts KEYBOARD focus back
        after a whole-screen recompose. ``Widget.recompose()`` removes every
        child, so the widget holding focus goes with them and nothing
        re-picks a target on remount -- ``screen.focused`` ends ``None`` and
        the keyboard is dead until the user clicks. PR F patched that on the
        Library's Media route only (``LibraryScreen.refresh``); a background
        job tick or an ad-hoc repaint on any other route still dropped it.
        Capturing here rather than at each call site is what makes it one
        seam: every ``refresh(recompose=True)`` on any screen in the app
        goes through this method. The restore itself is
        ``restore_focus_after_recompose``, which subclasses override (the
        Library composes its Media rules into it) so there is exactly ONE
        restore callback per recompose, never two competing ones.
        """
        focus_identity: Optional[str] = None
        if recompose and self.is_running:
            try:
                self.app.capture_mouse(None)
            except Exception:
                logger.debug(
                    "Mouse-capture release before recompose skipped.", exc_info=True
                )
            focus_identity = self._focus_identity_for_recompose()
        result = super().refresh(
            *regions, repaint=repaint, layout=layout, recompose=recompose
        )
        if focus_identity is not None:
            # ``call_after_refresh`` (not ``call_next``): Textual only
            # SCHEDULES the teardown here -- ``Widget.refresh`` queues
            # ``_check_recompose`` via ``call_next``, which always runs
            # before the ``InvokeLater`` that ``call_after_refresh``
            # posts, so the new children exist by the time this runs.
            #
            # TWO hops on purpose. ``call_after_refresh`` posts an
            # ``InvokeLater`` message, so callbacks run in POST order --
            # and a subclass queues its own post-recompose work AFTER
            # calling ``super().refresh()``, i.e. behind this one. Posting
            # the real restore from inside the first hop puts it at the
            # BACK of that queue instead, so the restore always runs after
            # the subclass's own passes. Concretely (PR L review item 3):
            # the Library's stage-visibility pass hides containers, and a
            # widget that becomes hidden BLURS ITSELF (``Widget._on_hide``
            # -> ``blur()``), so a restore that landed before it would put
            # focus straight back to ``None``. ``Widget.focusable`` reads
            # ``visible``, which cannot see a hide that has not happened
            # yet -- ordering is the only fix.
            self.call_after_refresh(
                self._queue_focus_restore_after_recompose, focus_identity
            )
        return result

    def _queue_focus_restore_after_recompose(self, previous: Optional[str]) -> None:
        """Re-post the restore behind everything else this refresh queued."""
        self.call_after_refresh(self.restore_focus_after_recompose, previous)

    def _focus_identity_for_recompose(self) -> Optional[str]:
        """Id selector of whatever holds focus right now, or ``None``.

        ``None`` means "restore nothing", and covers both cases where a
        restore would be wrong rather than merely unhelpful: the screen had
        no focus to lose (a recompose that starts unfocused should not
        seize the keyboard), or the focused widget has no id and therefore
        cannot be identified again after the remount. Subclasses narrow
        this further -- the Library refuses to record a reader pane GRIP,
        which is where Textual dumps focus by accident and never a target
        worth restoring.
        """
        widget_id = getattr(self.focused, "id", None)
        return f"#{widget_id}" if widget_id else None

    def restore_focus_after_recompose(self, previous: Optional[str]) -> None:
        """Re-focus the equivalent widget after a recompose (task-31946).

        Only acts when the recompose actually LOST focus: an explicit
        follow-up that already focused something real (the Library's
        ``then=`` callbacks, a canvas-scoped restore) wins untouched, which
        is also what keeps this from double-firing with the Media restore
        ``LibraryScreen`` runs from its own override of this method.

        Order of preference:

        1. the same-id widget, when the remount produced one again (the
           overwhelmingly common case -- a recompose rebuilds the same
           tree) and it is still ``focusable``; ``Screen.set_focus``
           silently NO-OPS on a widget whose ``focusable`` is False, so a
           row that came back disabled must fall through rather than count
           as success;
        2. otherwise the first focusable widget INSIDE ``#screen-content``
           -- the defined fallback. Scoped to the content area on purpose
           (PR L review item 1): ``focus_chain[0]`` is the
           ``MainNavigationBar``'s first tab on every screen, where a
           blind Enter LEAVES THE SCREEN -- a key that was inert before
           this seam existed. Nothing screen-recomposes a ``LabScreen``
           today (``lab_frame.py`` re-lays its rail without a recompose),
           but its body mounts in a later ``call_after_refresh``, so a
           future recompose would find the captured id absent and land
           here -- on ``LabModeStrip``'s chips, where Enter switches Lab
           mode. Narrower than leaving the screen; excluding the chips
           is a follow-up if that path ever fires.
           The bare ``focus_chain[0]`` remains the last resort for a
           screen whose content area holds nothing focusable at all;
           mounted and keyboard-reachable still beats ``None``.

        ``scroll_visible=False``: this restores FOCUS, never scroll
        position (PR F measured the default re-scrolling a Media row back
        into view and reinstating a restore the user's wheel had cancelled).

        Args:
            previous: Identity from ``_focus_identity_for_recompose``, or
                ``None`` to leave focus alone.
        """
        if previous is None or not self.is_running:
            return
        focused = self.focused
        if focused is not None and focused.is_attached:
            return
        try:
            target = self.query_one(previous)
        except QueryError:
            target = None
        if target is None or not target.focusable:
            target = self._first_focusable_in_content()
        if target is not None:
            self.set_focus(target, scroll_visible=False)

    def _first_focusable_in_content(self) -> "Widget | None":
        """The fallback target: first focusable widget under content.

        Falls back to the screen's own first focusable widget only when
        the content area holds none (or has not been composed yet) --
        see ``restore_focus_after_recompose`` for why the nav bar must
        not be the first choice.
        """
        chain = self.focus_chain
        content = next(iter(self.query("#screen-content")), None)
        if content is not None:
            inside = next((w for w in chain if content in w.ancestors), None)
            if inside is not None:
                return inside
        return next(iter(chain), None)

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
        self.post_message(self.ContentsRebuilt(self))

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

    # TASK-24452 note: an `on_screen_suspend` override briefly lived here
    # releasing a screen-owned Buddy view's mouse capture (Qodo #2402
    # finding 3 -- reusable screens suspend instead of unmounting, so an
    # unmount-time release stops covering them). PR #2407 then moved the
    # Buddy overlay's lifetime to an app-level owner
    # (`UI/Navigation/persona_buddy_overlay.py`): screens no longer hold a
    # `_persona_buddy_view` at all, the owner's `is_current` fence rejects
    # interaction the moment `app.screen` changes, its retire/invalid paths
    # release capture per view, and Textual's own `switch_screen` calls
    # `capture_mouse(None)` before every swap. The concern is covered at
    # the owner; a screen-level hook would read an attribute that no
    # longer exists.

    def on_unmount(self) -> None:
        """Called when the screen is unmounted.

        MRO contract (TASK-31418, same rule as ``on_mount`` above): Textual's
        dispatcher invokes EVERY ``on_unmount`` defined along the MRO for one
        Unmount event, so a subclass handler must NOT call
        ``super().on_unmount()`` -- that runs this body a second time.
        Harmless today because this handler only logs, but the next
        non-idempotent teardown added here (a close, a release, a decrement)
        would double-fire in every subclass that still carries a
        ``super().on_unmount()`` call. Probed on Textual 8.2.8 with a
        two-level ``Screen`` subclass: base fired twice per unmount
        (``['child', 'base', 'base']``); the same double-fire was confirmed
        for ``on_mount`` and ``on_screen_resume``. See
        ``backlog/docs/lessons-textual.md`` for the full probe.
        """
        logger.info(f"Screen {self.screen_name} unmounted")
