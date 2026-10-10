"""TldwCli's screen navigation: ``NavigationMixin``.

Moved verbatim from ``app.py`` (TASK-33011, PR-G): resolving a navigation target, creating and
reusing destination screens (with their screen-owned CSS), the navigation lock and its
admission/drain/resume, overlay dismissal, the locked navigation itself, completion, failure
notification and navigation-bar resync. ``TldwCli`` mixes the class in before ``App``. The
``@on(NavigateToScreen)`` handler ``_dispatch_screen_navigation`` stays on ``TldwCli`` (Textual
only dispatches decorated handlers defined on Textual classes) and calls into these methods. The
``_SCREEN_OWNED_ROUTE_CSS`` class attribute stays on ``TldwCli`` too; ``_ensure_screen_owned_css``
reads it through ``self``.

Patch the names this code reads HERE: the bodies resolve free names through this module's
globals, so a patch on ``tldw_chatbook.app`` alone no longer reaches them. Where ``app.py`` still
reads the same name, patch both modules (``Tests/app_module_patches.py``).
``Tests/Architecture/test_app_extracted_patch_targets.py`` fails on an app-module patch that can
only have been meant for code that moved out.
"""

import asyncio
import inspect
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from loguru import logger
from textual.css.query import NoMatches, QueryError

from tldw_chatbook.Constants import TAB_CHAT, TAB_RESEARCH_WORKSPACE
from tldw_chatbook.UI.Navigation.main_navigation import (
    MainNavigationBar,
    NavigateToScreen,
)
from tldw_chatbook.UI.Navigation.screen_registry import (
    resolve_screen_route,
    resolve_screen_target,
    screen_load_error,
)
from tldw_chatbook.UI.Navigation.screen_state_store import ConsolePromptTargetProjection
from tldw_chatbook.UI.Navigation.shell_destinations import (
    get_shell_destination,
    resolve_shell_route,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tldw_chatbook.UI.Navigation.screen_state_store import RuntimeIdentity



class NavigationMixin:
    """``TldwCli`` members moved from ``app.py`` (TASK-33011); see the module docstring."""

    def _resolve_screen_navigation_target(self, target: str):
        """Normalize navigation aliases to a routed screen id and canonical current_tab value."""
        return resolve_screen_target(target)

    def _ensure_screen_owned_css(self, canonical_route: str) -> None:
        """Parse the route's screen-owned feature sheets on first visit.

        Mirrors Textual's ``App._load_screen_css`` (has_source guard, read,
        reparse, app-wide update) for sheets owned by the app rather than a
        screen. Never raises: a missing or unparsable sheet degrades to the
        unstyled state a fresh checkout without a CSS build already has,
        and the boot-time rebuild path owns repairing that.

        Args:
            canonical_route: The destination's canonical route id.
        """
        names = self._SCREEN_OWNED_ROUTE_CSS.get(canonical_route)
        if not names:
            return
        try:
            css_dir = Path(__file__).parent / "css"
            update = False
            for name in names:
                path = css_dir / name
                if path.is_file() and not self.stylesheet.has_source(str(path), ""):
                    self.stylesheet.read(path)
                    update = True
            if update:
                self.stylesheet.reparse()
                self.stylesheet.update(self)
        except Exception:
            logger.opt(exception=True).warning(
                "Screen-owned CSS load failed (route={}); continuing unstyled.",
                canonical_route,
            )

    def _create_navigation_screen(self, screen_name: str, screen_class: type):
        """Build a FRESH screen instance for every navigation.

        Args:
            screen_name: Routed screen id (used by callers for state keying;
                unused here, kept for signature stability at the seam).
            screen_class: The Screen subclass registered for the route.

        Returns:
            A newly constructed, never-mounted instance of ``screen_class``.

        Screens must never be cached and re-mounted: ``switch_screen``
        unmounts the outgoing screen, and re-mounting a previously-unmounted
        instance races its still-in-flight teardown under rapid tab
        switching -- child message pumps end up permanently stopped while
        the widgets stay attached (``mounted=True``), the compositor keeps
        presenting a stale frame, and every subsequent click is hit-tested
        into the dead tree and silently swallowed: a total, exception-free
        UI freeze (root-caused 2026-07-11). UX continuity across visits is
        owned by ``ScreenStateStore`` through each screen's
        ``save_state``/``restore_state`` boundary, not instance reuse.

        One documented exception, since task-15860: Console's message
        history is NOT in that snapshot. It lives in the app-owned
        ``ConsoleRuntime``'s ``ConsoleChatStore``, which outlives every
        ``ChatScreen``; Console's snapshot carries only view state (image
        view modes, the task-resume projection, the staged live-work
        launch). Two sources of truth is what that snapshot had become --
        a turn that ran while Console was unmounted persisted to
        ChaChaNotes and was then overwritten, unseen, by a snapshot taken
        before it (executed: ``Docs/superpowers/plans/2026-08-14-headless-
        wake-task-0-report.md``, P3b). Screens still die on navigation;
        only the runtime survives.

        Second documented exception, since TASK-24452: routes whose
        ``ScreenRoute.reusable`` flag is set bypass this constructor on
        warm visits entirely (``_reusable_navigation_screen``). That reuse
        is safe ONLY because the instance is INSTALLED: Textual's
        ``_replace_screen`` suspends an installed screen instead of
        removing it, so the teardown race above never starts -- the
        2026-07-11 freeze requires an unmount to be in flight, and an
        installed screen is never unmounted mid-session.
        """
        if screen_name == TAB_RESEARCH_WORKSPACE:
            return self._create_research_workspace_screen(screen_class)
        return screen_class(self)

    def _reusable_navigation_screen(
        self,
        current_tab_value: str,
        runtime_identity: "RuntimeIdentity",
    ) -> Any | None:
        """Return the cached installed instance for a reusable route, if any.

        TASK-24452. The cache is keyed by canonical route id and scoped to
        the runtime identity that built the instance: an identity change
        (the same scope ``ScreenStateStore`` keys its snapshots by) drops
        and uninstalls the cached screen rather than leaking one identity's
        live widget state into another's session.

        Args:
            current_tab_value: Canonical route id for the destination.
            runtime_identity: The current snapshot scope.

        Returns:
            The still-valid installed instance, or ``None`` when the route
            has not been visited yet (or its instance was invalidated).
        """
        cache = getattr(self, "_reusable_screen_instances", None)
        if cache is None:
            return None
        entry = cache.get(current_tab_value)
        if entry is None:
            return None
        cached_identity, screen = entry
        if cached_identity == runtime_identity:
            return screen
        # Identity flipped (local <-> server): the cached instance carries
        # the OLD identity's live widget state and must not be resumed under
        # the new one. Drop it from the cache first -- that alone guarantees
        # it is never reused -- then best-effort dispose. A screen still in
        # the stack cannot be uninstalled (Textual raises); it stays
        # installed-and-suspended until app exit: a bounded leak of one
        # instance per identity flip, preferred over a teardown race on the
        # rare path.
        cache.pop(current_tab_value, None)
        try:
            if all(screen not in stack for stack in self._screen_stacks.values()):
                self.uninstall_screen(screen)
                screen.remove()
        except Exception:
            logger.debug(
                "Could not dispose stale reusable screen (route=%s).",
                current_tab_value,
            )
        return None

    def _retain_reusable_navigation_screen(
        self,
        current_tab_value: str,
        runtime_identity: "RuntimeIdentity",
        screen: Any,
    ) -> None:
        """Install and cache a freshly built reusable screen (TASK-24452).

        Installing is the load-bearing half: only installed screens survive
        ``switch_screen`` without being unmounted. A failure to install
        falls back silently to the fresh-instance lifecycle -- the screen
        simply is not cached, and the next visit constructs again.
        """
        try:
            # id() in the name: a stale same-route instance can legitimately
            # outlive its cache entry (see the bounded-leak note in
            # `_reusable_navigation_screen`), and `install_screen` raises on
            # a duplicate name -- a lingering install must never block the
            # replacement's.
            self.install_screen(
                screen, name=f"tldw-reusable:{current_tab_value}:{id(screen)}"
            )
        except Exception:
            logger.warning(
                "Could not install reusable screen; falling back to "
                "per-visit construction (route=%s).",
                current_tab_value,
            )
            return
        cache = getattr(self, "_reusable_screen_instances", None)
        if cache is None:
            cache = {}
            self._reusable_screen_instances = cache
        cache[current_tab_value] = (runtime_identity, screen)

    def _screen_navigation_lock(self) -> asyncio.Lock:
        """Return the lock serializing `handle_screen_navigation` attempts.

        TASK-1230: `_dispatch_screen_navigation` (the App's real
        ``@on(NavigateToScreen)`` handler) now runs each navigation attempt
        as its own worker instead of awaiting it inline on the App's single
        message-processing task -- see that method's docstring for why.
        Workers are otherwise independent, and running two attempts
        concurrently would let them race on shared state in a way the old
        single-queue dispatch never allowed: ``self.current_tab``,
        ``switch_screen``'s screen stack, and -- inside
        ``_complete_screen_navigation``, itself called from within the
        guarded region -- ``self.screen_state_store.save()`` (snapshotting
        the OUTGOING screen) and ``.restore()`` (rehydrating the INCOMING
        one); two attempts interleaving there could save/restore the wrong
        screen's state or clobber a snapshot the other attempt just wrote.
        This lock preserves the old FIFO ordering: `asyncio.Lock` serves
        waiters in arrival order, so attempts still complete strictly one
        at a time, in the order their ``NavigateToScreen`` messages were
        dispatched -- confirmed by
        ``test_overlapping_navigate_requests_complete_in_fifo_order``,
        which reliably reorders without this lock -- the only change is
        that an attempt waiting on a confirm-navigation dialog no longer
        blocks the App from routing input to that very dialog while it
        waits its turn.
        """
        lock = getattr(self, "_screen_navigation_lock_instance", None)
        if lock is None:
            lock = asyncio.Lock()
            self._screen_navigation_lock_instance = lock
        return lock

    def _screen_navigation_close_admission(self) -> None:
        """Fence new route requests without cancelling accepted navigation."""
        self._screen_navigation_paused = True

    async def _screen_navigation_drain(self, deadline: float) -> bool:
        """Wait for admitted navigation before taking a screen snapshot."""
        if not getattr(self, "_screen_navigation_paused", False):
            raise RuntimeError("screen_navigation_not_paused")
        while True:
            setup = getattr(self, "_initial_screen_setup_task", None)
            pending = (
                getattr(self, "_screen_navigation_calls", None)
                or getattr(self, "_pending_flush_tasks", None)
                or any(not worker.is_finished for worker in getattr(self, "_screen_navigation_workers", ()))
                or (setup is not None and not setup.done())
            )
            mounted = (
                getattr(self, "_initial_screen_pushed", False) is True
                and getattr(self, "_ui_ready", False) is True
            )
            # A constructed but never-running app has no initial mount to
            # await. A live app must finish its actual first mount/setup.
            if not pending and (mounted or not self.is_running):
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(remaining, 0.02))

    def _screen_navigation_resume(self) -> None:
        """Reopen route requests after maintenance releases its screen view."""
        self._screen_navigation_paused = False

    async def handle_screen_navigation(self, message: NavigateToScreen) -> None:
        """Handle navigation to a different screen using switch_screen for better performance.

        Args:
            message: The navigation request. It reports completion ``False`` without
                navigating while navigation is paused or the app is shutting down;
                otherwise it runs under the navigation admission lock.
        """
        if getattr(self, "_screen_navigation_paused", False) or getattr(self, "_shutting_down", False):
            message.report_completion(False)
            return
        await self._run_admitted_screen_navigation(message)

    async def _run_admitted_screen_navigation(self, message: NavigateToScreen) -> None:
        """Complete an accepted route request under the existing FIFO lock."""
        calls = getattr(self, "_screen_navigation_calls", None)
        if calls is None:
            calls = self._screen_navigation_calls = {}
        task = asyncio.current_task()
        depth = calls.get(task, 0)
        calls[task] = depth + 1
        try:
            try:
                async with self._screen_navigation_lock():
                    succeeded = await self._handle_screen_navigation_locked(message)
            except asyncio.CancelledError:
                # Waiting to acquire the FIFO lock is cancellable too. Once the
                # destination owns Textual's stack, however, a later cancellation
                # cannot make the source route failed again. Cancellation still
                # belongs to the worker lifecycle and must remain observable.
                message.report_completion(message.target_ownership_committed)
                raise
            except Exception:
                if message.target_ownership_committed:
                    message.report_completion(True)
                else:
                    # task-2720: several steps in the locked body are legitimately
                    # unguarded (target resolution, runtime identity, snapshot
                    # restore, transition admission) and a transient error in any
                    # of them used to fail SILENTLY: no message, nav-bar highlight
                    # stuck on the destination, retry clicks no-opped. Recover the
                    # user-facing state, then re-raise so the worker hook still
                    # writes the `worker_failed` diagnostics line (ADR-029).
                    self._notify_navigation_failure(message.screen_name)
                    message.report_completion(False)
                raise
            message.report_completion(message.target_ownership_committed or succeeded)
        finally:
            if depth:
                calls[task] = depth
            else:
                calls.pop(task, None)

    def _navigation_outgoing_screen(self) -> Any:
        """Return the CONTENT screen a navigation is leaving.

        The screen stack is ``[Textual's default screen, the content screen,
        *pushed screens]``: startup pushes exactly one routed screen
        (``_push_initial_screen``) and every navigation replaces it, so
        index 1 is the tab the user is on and anything above it is an
        overlay. ``self.screen`` is the TOP of that stack, which is the
        overlay whenever one is open -- see ``_dismiss_navigation_overlays``
        for why that distinction is load-bearing.

        Returns:
            The content screen at the base of the stack, or ``self.screen``
            when the stack is too short for that position to exist (before
            the initial push, and in tests that drive the handler with no
            mounted stack at all).
        """
        try:
            stack = self._screen_stack
        except Exception:  # pragma: no cover - defensive; no mode, no stack
            return self.screen
        if len(stack) >= 2:
            return stack[1]
        return self.screen

    @staticmethod
    def _navigation_overlay_awaiter_pending(screen: Any) -> bool:
        """Report whether ``screen`` still owes a ``push_screen_wait`` result."""
        callbacks = getattr(screen, "_result_callbacks", None)
        if not callbacks:
            return False
        future = getattr(callbacks[-1], "future", None)
        return future is not None and not future.done()

    async def _dismiss_navigation_overlays(self, screen_name: str) -> bool:
        """Reduce the screen stack to its content screen before switching.

        TASK-16300. Textual's ``App.switch_screen``
        (``textual/app.py:3001-3032``) pops only ``self._screen_stack[-1]``
        and appends the new screen; ``_replace_screen`` then unmounts only
        that popped screen. So switching while ANY pushed screen sits above
        the content screen replaces THE OVERLAY and leaves the content
        screen resident in the stack -- mounted, message pump running,
        ``on_unmount`` never fired, its timers and controllers alive behind
        whatever the user is now looking at, and a second live instance of
        it created the moment they navigate back. That directly violates
        the invariant ``_create_navigation_screen`` documents (screens die
        on navigation; ``ScreenStateStore`` carries continuity instead),
        and it is the state the wake-integrity arc traced two live Console
        failures to (tasks 15970/15971).

        Overlays are dismissed rather than popped because ``switch_screen``
        and ``pop_screen`` both call ``_pop_result_callback()`` WITHOUT
        invoking it (``textual/app.py:3020``): a modal opened through
        ``push_screen_wait`` holds a future in that callback, so discarding
        it uncalled leaves the awaiting worker suspended forever -- it has
        no timeout and nothing else ever resolves it.
        ``Screen.dismiss(None)`` calls the callback first
        (``textual/screen.py:2048-2070``), so the awaiter resumes with the
        same ``None`` every user-driven close already delivers (``Escape``,
        ``action_dismiss``, a bare ``dismiss()``) -- the value existing
        callers, including the ones that map it to a decline, are already
        written against. Refusing to navigate while a modal is awaited was
        the alternative and is worse: awaited modals are the common kind,
        and a nav shortcut that silently no-ops is indistinguishable from a
        wedged app.

        Args:
            screen_name: Route being navigated to, for log context.

        Returns:
            ``True`` when the stack is reduced to its content screen and the
            switch may proceed; ``False`` when an overlay would not leave,
            in which case the caller must abort rather than switch and
            recreate the very leak this exists to prevent.
        """
        for _ in range(self._MAX_NAVIGATION_OVERLAY_DISMISSALS):
            stack = self._screen_stack
            if len(stack) <= 2:
                return True
            overlay = stack[-1]
            logger.info(
                "Dismissing pushed screen before navigating "
                "(route=%s, screen=%s, awaited=%s).",
                screen_name,
                type(overlay).__name__,
                self._navigation_overlay_awaiter_pending(overlay),
            )
            try:
                dismissed = overlay.dismiss(None)
                if inspect.isawaitable(dismissed):
                    await dismissed
            except Exception as exc:
                logger.warning(
                    "Pushed screen refused to dismiss before navigation "
                    "(route=%s, screen=%s, exception_category=%s).",
                    screen_name,
                    type(overlay).__name__,
                    type(exc).__name__,
                )
                return False
            stack = self._screen_stack
            if stack and stack[-1] is overlay:
                logger.warning(
                    "Pushed screen stayed on the stack after dismissal "
                    "(route=%s, screen=%s).",
                    screen_name,
                    type(overlay).__name__,
                )
                return False
        logger.warning(
            "Screen stack did not reduce to its content screen within %s "
            "dismissals (route=%s).",
            self._MAX_NAVIGATION_OVERLAY_DISMISSALS,
            screen_name,
        )
        return False

    async def _handle_screen_navigation_locked(self, message: NavigateToScreen) -> bool:
        """Body of `handle_screen_navigation`, run under its FIFO lock."""
        requested_screen = message.screen_name
        if not getattr(self, "_initial_screen_pushed", False):
            # Until the initial screen exists (splash screen still up, or the
            # post-splash startup push still in flight) the screen stack has
            # no result callback to pop and switch_screen raises IndexError.
            # Swallow the request; the user can re-issue it once the app is
            # interactive.
            logger.info(
                f"Ignoring navigation to {requested_screen}: "
                "initial screen not yet mounted"
            )
            return False

        # TASK-31807: refuse to navigate while a modal that has opted out of
        # stray-navigation dismissal is on the stack. The first-run setup
        # wizard is such a gate: it is pushed over the initial screen at
        # startup, and a navigation that arrives while it is still up is never
        # user-driven (its own Next/Back/Skip/Esc controls dismiss it directly
        # and post any follow-on navigation only AFTER it has left the stack).
        # The real trigger is a shell-destination key (F9/F10/ctrl+N ...)
        # leaking in during splash teardown -- the app's global bindings are
        # live on the just-mounted initial screen while the wizard's own push
        # is a `call_after_refresh` behind it. Left unguarded, that navigation
        # reaches `_dismiss_navigation_overlays` below and `dismiss(None)`s the
        # wizard, discarding onboarding with zero input (and stranding
        # `setup_started`, persisted from the wizard's `on_mount`). Ignore it,
        # keeping the wizard up. Returning False here is silent -- no
        # "couldn't open" toast, unlike the overlay-refusal path below.
        if any(
            getattr(screen, "blocks_stray_navigation", False)
            for screen in self._screen_stack
        ):
            logger.info(
                "Ignoring navigation to %s: a first-run setup gate is active "
                "and must be completed or explicitly cancelled first.",
                requested_screen,
            )
            return False

        screen_name, current_tab_value, screen_class = (
            self._resolve_screen_navigation_target(requested_screen)
        )
        logger.info(f"Navigating to screen: {requested_screen}")

        # NOT ``self.screen`` (TASK-16300): with a pushed screen on top --
        # the nav overflow menu, the command palette, a picker, a confirm
        # dialog -- ``self.screen`` IS that overlay, and every hook below
        # (flush, confirm, transition admission, and ``save_state`` inside
        # ``_complete_screen_navigation``) was asked of it. Overlays answer
        # none of them, so Console's busy-fleet confirmation never ran and
        # the tab being left was never snapshotted.
        current_screen = self._navigation_outgoing_screen()

        # Screens are never reused across navigations, so anything the
        # outgoing screen has not persisted is destroyed with its instance.
        # Give it one awaited chance to flush pending work (e.g. a Library
        # note edit whose debounced autosave has not fired); False vetoes
        # the switch, leaving the screen (and e.g. its save-conflict banner)
        # in place for the user.
        flush = getattr(current_screen, "flush_pending_work", None)
        if callable(flush):
            try:
                # TASK-34000.27: a flush that can name the destination in its
                # own veto toast ("Can't open Console yet: …") is told the
                # label; every other screen's flush is called as before.
                flush_result = (
                    flush(destination=self._navigation_destination_label(screen_name))
                    if self._flush_accepts_destination(flush)
                    else flush()
                )
                if inspect.isawaitable(flush_result):
                    # Shielded: giving up on the WAIT must not give up on the
                    # SAVE. The Library File Notes flush persists through
                    # `asyncio.to_thread`, which cannot be cancelled -- an
                    # unshielded `wait_for` killed the coroutine at that await
                    # while the thread kept writing, so `_save_draft` never ran
                    # its reconciliation: `_save_state` stayed "saving" (which
                    # makes `leave_allowed` False *forever*) and the cached
                    # `content_hash` stayed stale, so the next save reported a
                    # spurious conflict.
                    flush_task = asyncio.ensure_future(flush_result)
                    self._retain_unfinished_flush(flush_task, screen_name)
                    flush_result = await asyncio.wait_for(
                        asyncio.shield(flush_task),
                        timeout=self.NAVIGATION_FLUSH_TIMEOUT_SECONDS,
                    )
                if flush_result is False:
                    logger.info(
                        f"Navigation to {screen_name} vetoed by the outgoing "
                        "screen's pending-work flush"
                    )
                    # TASK-34000.27: the screen says why (Library, Console
                    # and Settings all toast before returning False); the
                    # app's part is the rollback. The bar framed the clicked
                    # destination optimistically, and left there it also
                    # swallowed the retry click (task-2720's trap, seen
                    # live: review S-17).
                    self._restore_nav_bar_highlight(
                        current_screen, screen_name=screen_name
                    )
                    return False
            except asyncio.TimeoutError:
                # Fail closed, exactly like a flush that raised: the pending
                # edits may exist ONLY in the outgoing screen, so keep it
                # mounted rather than discarding it on a save we can't
                # confirm. Abandoning the wait does not abandon the save --
                # the note-save worker is a separate task and keeps running.
                logger.warning(
                    "Screen flush timed out after %ss; staying put (route=%s).",
                    self.NAVIGATION_FLUSH_TIMEOUT_SECONDS,
                    screen_name,
                )
                try:
                    self.notify(
                        "Still saving pending changes; staying on this screen. "
                        "Try again in a moment.",
                        severity="warning",
                    )
                except Exception:
                    pass
                self._restore_nav_bar_highlight(
                    current_screen, screen_name=screen_name
                )
                return False
            except Exception as exc:
                # The outgoing instance may be the only place pending edits
                # still exist, so a failed flush must abort the transition.
                logger.warning(
                    "Screen flush failed (route=%s, exception_category=%s).",
                    screen_name,
                    type(exc).__name__,
                )
                try:
                    self.notify(
                        "Couldn't save pending changes before switching screens.",
                        severity="warning",
                    )
                except Exception:
                    pass
                self._restore_nav_bar_highlight(
                    current_screen, screen_name=screen_name
                )
                return False

        # TASK-1143 (F5): give the outgoing screen one awaited chance to
        # ASK before navigation proceeds. Mirrors the flush-veto seam
        # immediately above: False keeps the outgoing screen in place,
        # only here the decision comes from a user-facing confirmation
        # dialog rather than an unresolved save conflict. TASK-31520 note:
        # for REUSABLE routes leaving suspends rather than tears down, so
        # a screen whose only stake was "leaving destroys my work" should
        # return True unconditionally once flagged reusable (Console's
        # does -- its runs and approvals survive navigation now); the seam
        # itself stays for non-reusable screens and for hooks that gate on
        # something other than teardown.
        confirm_navigation = getattr(current_screen, "confirm_navigation", None)
        if callable(confirm_navigation):
            try:
                confirm_result = confirm_navigation()
                if inspect.isawaitable(confirm_result):
                    confirm_result = await confirm_result
                if confirm_result is False:
                    logger.info(
                        f"Navigation to {screen_name} vetoed by the outgoing "
                        "screen's confirm_navigation"
                    )
                    # The user answered "stay" in the screen's own dialog;
                    # the bar still has to agree with the stack (TASK-34000.27).
                    self._restore_nav_bar_highlight(
                        current_screen, screen_name=screen_name
                    )
                    return False
            except Exception as exc:
                # A broken confirm hook must not silently let navigation
                # proceed and tear down live work the user was never asked
                # about -- fail closed, same as the flush veto above.
                logger.warning(
                    "Screen navigation confirm failed (route=%s, exception_category=%s).",
                    screen_name,
                    type(exc).__name__,
                )
                try:
                    self.notify(
                        "Couldn't confirm leaving this screen; staying put.",
                        severity="warning",
                    )
                except Exception:
                    pass
                self._restore_nav_bar_highlight(
                    current_screen, screen_name=screen_name
                )
                return False

        release_navigation = None
        acquire_navigation = getattr(
            current_screen,
            "acquire_navigation_transition",
            None,
        )
        if callable(acquire_navigation):
            admission = acquire_navigation()
            if admission is False:
                logger.info(
                    f"Navigation to {screen_name} vetoed by the outgoing "
                    "screen's transition admission"
                )
                self._restore_nav_bar_highlight(
                    current_screen, screen_name=screen_name
                )
                return False
            release_navigation = admission
        try:
            return await self._complete_screen_navigation(
                message=message,
                requested_screen=requested_screen,
                screen_name=screen_name,
                current_tab_value=current_tab_value,
                screen_class=screen_class,
                current_screen=current_screen,
            )
        finally:
            if callable(release_navigation):
                release_navigation()

    def _retain_unfinished_flush(self, flush_task: Any, screen_name: str) -> None:
        """Keep every accepted flush alive until it finishes on its own.

        The navigation wait is shielded, so the flush keeps running after the
        app stops waiting -- but asyncio only holds a weak reference to a
        running task, so without a strong reference here it could be garbage
        collected mid-save. Retaining it also gives somewhere to consume the
        eventual result, which otherwise surfaces as "exception was never
        retrieved" noise long after the navigation that started it.

        Args:
            flush_task: The still-running flush task.
            screen_name: Route being navigated to, for log context.
        """
        pending = getattr(self, "_pending_flush_tasks", None)
        if pending is None:
            pending = set()
            self._pending_flush_tasks = pending
        pending.add(flush_task)

        def _finished(task: Any) -> None:
            pending.discard(task)
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.warning(
                    "Retained screen flush failed (route=%s, exception_category=%s).",
                    screen_name,
                    type(exc).__name__,
                )
            else:
                logger.info(
                    "Retained screen flush completed (route=%s).",
                    screen_name,
                )

        flush_task.add_done_callback(_finished)

    def _notify_navigation_failure(self, screen_name: str) -> None:
        """Tell the user a destination failed to open, without raising.

        Navigation failures are reported where they happen so the user is
        not left staring at an unchanged screen wondering whether the click
        registered. ``notify`` itself is guarded: this runs on the crash
        path, and a failure to display the message must not replace one
        escaping exception with another.
        """
        try:
            self.notify(
                f"Couldn't open {screen_name}. Staying on the current screen.",
                severity="error",
            )
        except Exception:
            logger.debug(f"Could not surface navigation failure for {screen_name!r}.")
        # The crash path rolls back the TOP screen's bar, as it always did;
        # the refusing branches of the locked body roll back the outgoing
        # content screen's (TASK-16300) through the same helper.
        try:
            top_screen = self.screen
        except Exception:
            top_screen = None
        self._restore_nav_bar_highlight(top_screen, screen_name=screen_name)

    def _restore_nav_bar_highlight(self, screen: Any, *, screen_name: str) -> None:
        """Roll ``screen``'s nav bar back to the route actually on the stack.

        task-2720: the nav bar highlighted the destination the moment it was
        clicked, before the navigation worker ran. Roll it back to the screen
        actually on the stack — otherwise the bar shows a destination that
        never loaded AND its already-active check swallows every retry click,
        leaving the destination unreachable until restart. TASK-34000.27:
        every refusing branch (flush veto, flush timeout, flush failure,
        confirm veto, confirm failure, admission veto) calls this too; before,
        only the crash path did, and a vetoed click left the bar framing the
        destination the user never reached.

        task-2854: use ``nav_bar_active``, not ``screen_name`` -- a screen
        whose route is folded under another destination for routing/label
        purposes only (e.g. Study folds under Library) sets
        ``nav_bar_active`` to a value that clears its own nav bar's
        highlight instead of falsely re-claiming the owning destination
        (see ``BaseAppScreen.nav_bar_active``). ``screen_name`` is kept as
        a fallback for any screen that predates that attribute.
        ``nav_bar_active`` may legitimately be ``""`` (Study's case), which
        must still reach ``restore_active`` -- ``resolve_shell_route("")``
        matches no destination, so every nav button loses ``is-active``
        rather than the call being skipped and the stale optimistic
        highlight surviving.

        Args:
            screen: The screen whose bar was optimistically highlighted
                (``None`` when the app has no screen yet: nothing to do).
            screen_name: The destination that failed to open, for the log.
        """
        if screen is None:
            return
        try:
            current_route = getattr(screen, "nav_bar_active", None)
            if current_route is None:
                current_route = getattr(screen, "screen_name", None)
            if isinstance(current_route, str):
                screen.query_one(MainNavigationBar).restore_active(current_route)
        except Exception:
            logger.debug(
                f"Could not roll back nav-bar state after failing to open "
                f"{screen_name!r}."
            )

    @staticmethod
    def _navigation_destination_label(screen_name: str) -> str:
        """The nav-bar label (``"Console"``) of the destination owning ``screen_name``.

        Returns:
            The label, or ``""`` for a route no destination owns -- the
            flush's veto toast then falls back to "Can't leave yet".
        """
        try:
            return get_shell_destination(
                resolve_shell_route(screen_name).destination_id
            ).label
        except Exception:
            return ""

    @staticmethod
    def _flush_accepts_destination(flush: Any) -> bool:
        """Whether ``flush_pending_work`` takes a ``destination`` keyword."""
        try:
            parameters = inspect.signature(flush).parameters
        except (TypeError, ValueError):
            return False
        return "destination" in parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )

    def _resync_navigation_bar_active(self, screen: Any) -> None:
        """Re-sync the visible screen's nav bar to its own route (CE-007).

        A navigation click optimistically highlights the clicked destination
        on the OUTGOING screen's own bar. Reusable routes survive navigation
        as suspended instances, so a warm return reinstated that screen with
        its bar still claiming the destination the user clicked away to; the
        bar's already-active guard then swallowed every later press of that
        destination (UAT re-run 2026-09-13, TASK-32534: nav-settings presses
        logged with zero "Navigation requested" lines, while palette routes
        kept working because they post ``NavigateToScreen`` directly).
        Re-syncing the visible bar to its own route after every completed
        navigation restores both the highlight and retry-ability; the
        navigation-failure path already does the same for the unchanged
        screen via ``_notify_navigation_failure`` -> ``restore_active``.
        """
        route = getattr(screen, "nav_bar_active", None)
        if route is None:
            route = getattr(screen, "screen_name", None)
        if not isinstance(route, str):
            return
        # Navigation targets may be duck-typed (test fakes without the
        # Textual widget API), so probe the seam instead of assuming it.
        query_one = getattr(screen, "query_one", None)
        if not callable(query_one):
            return
        try:
            bar = query_one(MainNavigationBar)
        except (NoMatches, QueryError):
            return
        restore_active = getattr(bar, "restore_active", None)
        if callable(restore_active):
            restore_active(route)

    async def _complete_screen_navigation(
        self,
        *,
        message: NavigateToScreen,
        requested_screen: str,
        screen_name: str,
        current_tab_value: str,
        screen_class: type | None,
        current_screen: Any,
    ) -> bool:
        """Save, construct, restore, and switch while transition admission is held."""
        runtime_identity = self._current_runtime_identity()
        outgoing_key = str(self.current_tab or "").strip()
        if not outgoing_key:
            outgoing_screen_name = getattr(current_screen, "screen_name", None)
            if isinstance(outgoing_screen_name, str) and outgoing_screen_name.strip():
                (
                    _outgoing_screen_name,
                    resolved_outgoing_key,
                    outgoing_screen_class,
                ) = self._resolve_screen_navigation_target(outgoing_screen_name.strip())
                if outgoing_screen_class is not None:
                    outgoing_key = resolved_outgoing_key

        # A Console snapshot that fails to be replaced below must not survive
        # with its published prompt-target projection attached (the stale
        # target `publish_console_prompt_target` would otherwise be read back
        # against). Since task-15860 this discards VIEW state only: Console's
        # sessions and transcripts live in the app-owned `ConsoleRuntime`
        # store, which no snapshot lifecycle can drop.
        if outgoing_key == TAB_CHAT:
            self.screen_state_store.discard(outgoing_key)

        save_state = getattr(current_screen, "save_state", None)
        if outgoing_key and callable(save_state):
            try:
                state = save_state()
                if isinstance(state, Mapping):
                    self.screen_state_store.save(
                        outgoing_key,
                        state,
                        runtime_identity,
                    )
                    if outgoing_key == TAB_CHAT:
                        projection_getter = getattr(
                            current_screen,
                            "console_prompt_target_projection",
                            None,
                        )
                        try:
                            projection = (
                                projection_getter()
                                if callable(projection_getter)
                                else None
                            )
                        except Exception:
                            projection = None
                        if isinstance(projection, ConsolePromptTargetProjection):
                            self.screen_state_store.publish_console_prompt_target(
                                outgoing_key,
                                projection,
                                runtime_identity,
                            )
                    logger.debug(
                        "Saved screen snapshot for canonical route: %s",
                        outgoing_key,
                    )
                else:
                    logger.warning(
                        "Screen snapshot save skipped (route=%s, reason=non_mapping).",
                        outgoing_key,
                    )
            except Exception as exc:
                logger.warning(
                    "Screen snapshot save failed (route=%s, exception_category=%s).",
                    outgoing_key,
                    type(exc).__name__,
                )

        if screen_class:
            # TASK-24459: the destination's split-off feature CSS must be in
            # the app stylesheet before its widgets first style themselves --
            # ahead of BOTH construction paths below (a fresh build styles at
            # mount; a reused instance restyles on resume).
            self._ensure_screen_owned_css(current_tab_value)
            # TASK-24452: a reusable route's screen survives navigation as an
            # installed (suspended, never unmounted) instance; warm visits
            # skip construction, mount, and snapshot-restore entirely.
            route = resolve_screen_route(screen_name)
            reusable_route = bool(route is not None and route.reusable)
            new_screen = (
                self._reusable_navigation_screen(current_tab_value, runtime_identity)
                if reusable_route
                else None
            )
            screen_reused = new_screen is not None
            if new_screen is None:
                try:
                    new_screen = self._create_navigation_screen(
                        screen_name, screen_class
                    )
                except Exception as exc:
                    # A destination that cannot even be constructed is a broken
                    # destination, never a dead app. This ran unguarded until
                    # 2026-07-28: the MCP canvases read `Select.NULL` (Textual 8+)
                    # at construction time, so on an older Textual the
                    # AttributeError escaped this handler and Textual exited the
                    # whole app rather than the user simply failing to reach MCP.
                    logger.opt(exception=True).error(
                        "Screen construction failed (route={}, exception_category={}).",
                        screen_name,
                        type(exc).__name__,
                    )
                    self._notify_navigation_failure(screen_name)
                    return False
                if reusable_route:
                    self._retain_reusable_navigation_screen(
                        current_tab_value, runtime_identity, new_screen
                    )

            if screen_reused:
                # The live instance IS the state -- restoring an older
                # snapshot over it would regress whatever the user last did
                # there. The snapshot machinery still runs for the OUTGOING
                # side above (projections published from `save_state` have
                # consumers beyond restore).
                logger.debug(
                    "Reusing installed screen instance for canonical "
                    "route: %s",
                    current_tab_value,
                )
            else:
                restored_state = self.screen_state_store.restore(
                    current_tab_value,
                    runtime_identity,
                )
                restore_state = getattr(new_screen, "restore_state", None)
                if restored_state is not None and callable(restore_state):
                    try:
                        restore_state(restored_state)
                        logger.debug(
                            "Restored screen snapshot for canonical route: %s",
                            current_tab_value,
                        )
                    except Exception as exc:
                        self.screen_state_store.discard(current_tab_value)
                        logger.warning(
                            "Screen snapshot restore failed "
                            "(route=%s, exception_category=%s).",
                            current_tab_value,
                            type(exc).__name__,
                        )

            navigation_context = getattr(message, "screen_context", {}) or {}
            if not navigation_context:
                navigation_context = self._LEGACY_ROUTE_LIBRARY_NAV_CONTEXT.get(
                    requested_screen, {}
                )
            prepared = None
            try:
                if message.require_character_inspection_admission:
                    current = message.is_current or (lambda: True)
                    prepare = getattr(new_screen, "prepare_character_inspection", None)
                    commit = getattr(new_screen, "commit_character_inspection", None)
                    if not current() or not callable(prepare) or not callable(commit):
                        return False
                    prepared = await prepare(navigation_context, is_current=current)
                    if prepared is None or not prepared.is_current():
                        return False
                    # Final validation, source acknowledgement, and exact Library
                    # commit form one synchronous boundary before overlay teardown.
                    if (
                        message.on_commit_started is not None
                        and not message.on_commit_started()
                    ):
                        return False
                    if not commit(prepared):
                        return False
                elif navigation_context and hasattr(new_screen, "apply_navigation_context"):
                    try:
                        result = new_screen.apply_navigation_context(navigation_context)
                        if inspect.isawaitable(result):
                            await result
                    except Exception as exc:
                        logger.warning(
                            "Navigation context application failed "
                            "(route=%s, exception_category=%s).",
                            current_tab_value,
                            type(exc).__name__,
                        )

                # TASK-16300: `switch_screen` replaces the TOP of the stack, so
                # the content screen has to BE the top before it runs -- see
                # `_dismiss_navigation_overlays`. Done here, after the veto
                # hooks and the construction of the incoming screen, so a
                # navigation that never happens never costs the user the dialog
                # they had open. Failing to reduce aborts: switching anyway is
                # exactly how the outgoing screen is left resident.
                if not await self._dismiss_navigation_overlays(screen_name):
                    logger.warning(
                        "Aborting navigation: a pushed screen would not leave "
                        "the stack (route=%s).",
                        screen_name,
                    )
                    self._notify_navigation_failure(screen_name)
                    return False

                # Textual replaces the top stack entry synchronously, then its
                # awaitable finishes mounting/removing. The source callback must
                # commit at that ownership transfer rather than after unrelated
                # bookkeeping below.
                try:
                    switch_result = self.switch_screen(new_screen)
                except Exception as exc:
                    if self._navigation_target_owns_stack(new_screen):
                        message.commit_target_ownership()
                        logger.warning(
                            "Screen switch raised after target ownership "
                            "(route=%s, exception_category=%s).",
                            screen_name,
                            type(exc).__name__,
                        )
                        raise
                    # Sibling of the construction guard above: a screen can also
                    # fail while composing/mounting (the MCP audit canvas reads
                    # `Select.NULL` inside compose()), and Textual surfaces that
                    # through switch_screen. Same rule -- report the broken
                    # destination instead of taking the app down with it.
                    logger.opt(exception=True).error(
                        "Screen mount failed (route={}, exception_category={}).",
                        screen_name,
                        type(exc).__name__,
                    )
                    self._notify_navigation_failure(screen_name)
                    return False

                if self._navigation_target_owns_stack(new_screen):
                    message.commit_target_ownership()

                try:
                    await switch_result
                except Exception as exc:
                    if self._navigation_target_owns_stack(new_screen):
                        message.commit_target_ownership()
                    if message.target_ownership_committed:
                        logger.warning(
                            "Screen mount reported after target ownership "
                            "(route=%s, exception_category=%s).",
                            screen_name,
                            type(exc).__name__,
                        )
                        raise
                    logger.opt(exception=True).error(
                        "Screen mount failed (route={}, exception_category={}).",
                        screen_name,
                        type(exc).__name__,
                    )
                    self._notify_navigation_failure(screen_name)
                    return False

                if self._navigation_target_owns_stack(new_screen):
                    message.commit_target_ownership()
                if not message.target_ownership_committed:
                    logger.error(
                        "Screen switch returned without target stack ownership (route=%s).",
                        screen_name,
                    )
                    self._notify_navigation_failure(screen_name)
                    return False

                try:
                    # Keep current_tab aligned to canonical tab ids even when routing uses aliases.
                    self.current_tab = current_tab_value

                    # task-18812: the exit rule runs only once the switch has
                    # SUCCEEDED -- flush vetoes, confirmations, admission, and mount
                    # failures above all `return` with the Console still resident, so
                    # clearing earlier would desync the app flag from the mounted
                    # screen's -focus class (the next toggle would do the wrong
                    # visible action).
                    self._clear_focus_if_leaving_console(screen_name)
                    # TASK-32534 (CE-007): the now-visible screen may be a
                    # reused instance whose nav bar still carries the
                    # optimistic highlight set by the departure click.
                    self._resync_navigation_bar_active(new_screen)

                    # ADR-171: only successful top-level navigation is a departure;
                    # modal suspension and automatic startup never set this token.
                    if outgoing_key == TAB_CHAT and current_tab_value != TAB_CHAT:
                        self._console_manual_read_departure = getattr(
                            self, "conversation_local_marks_service", None
                        )
                    elif current_tab_value == TAB_CHAT:
                        departed = getattr(self, "_console_manual_read_departure", None)
                        self._console_manual_read_departure = None
                        if departed is not None and departed is getattr(
                            self, "conversation_local_marks_service", None
                        ):
                            new_screen.run_worker(
                                new_screen._session.acknowledge_explicit_console_return(),
                                group="console-manual-read-return",
                                exclusive=True,
                            )

                except Exception as exc:
                    logger.warning(
                        "Post-switch bookkeeping failed after target ownership "
                        "(route=%s, exception_category=%s).",
                        screen_name,
                        type(exc).__name__,
                    )
                    raise

                logger.info(f"Successfully switched to {screen_name} screen")
                return True
            finally:
                if prepared is not None:
                    prepared.finish(
                        target_owned=message.target_ownership_committed
                        or self._navigation_target_owns_stack(new_screen)
                    )
        else:
            # No class for the route: unroutable target, or the screen module
            # failed to import (`load_screen_class` degrades ImportError/
            # AttributeError to None). Since TASK-23023 resolved the
            # Research_Workspace facade lazily, a submodule broken at install
            # time surfaces HERE at first navigation instead of killing the
            # whole app at boot -- and a log-only failure is exactly the
            # task-2720 defect (stuck nav highlight, swallowed retries, no
            # message). Tell the user and roll the nav bar back.
            logger.error(
                f"Unknown screen requested: {requested_screen} "
                f"({screen_load_error(requested_screen)})"
            )
            self._notify_navigation_failure(screen_name)
            return False

    def _navigation_target_owns_stack(self, target_screen: Any) -> bool:
        """Return whether Textual synchronously replaced the active stack target."""
        try:
            stack = self._screen_stack
        except Exception:
            return False
        return bool(stack) and stack[-1] is target_screen
