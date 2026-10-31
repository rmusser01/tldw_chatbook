"""Thin Console controller for the app-owned persistent Terminal manager."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from loguru import logger
from textual.css.query import NoMatches

from tldw_chatbook.Terminal.contracts import (
    TERMINAL_DISCLOSURE_LINES,
    TerminalLaunchRequest,
    TerminalLifecycle,
)
from tldw_chatbook.Terminal.launch import (
    ShellChoice,
    normalize_session_name,
    resolve_shell_choice,
    resolve_start_directory,
)
from tldw_chatbook.Terminal.session_manager import (
    TerminalSessionView,
    TerminalSubscriptionToken,
    TerminalViewState,
    TerminalViewToken,
)
from tldw_chatbook.Widgets.Console.console_terminal_messages import TerminalAction
from tldw_chatbook.Widgets.Console.console_terminal_session_modal import (
    ConsoleTerminalSessionModal,
    TerminalSessionFormResult,
    build_default_terminal_name,
)


def _spawn_async(factory: Callable[[], Awaitable[Any]]) -> None:
    """Start one controller-owned coroutine on the active Textual loop."""
    asyncio.create_task(factory())


class ConsoleTerminalController:
    """Route direct-user UI actions to one late-bound app-owned manager."""

    def __init__(
        self,
        *,
        terminal_runtime: Callable[[], Any],
        workspace_accessor: Callable[[], Any | None],
        selected_local_root: Callable[[], Path | None],
        account_home: Callable[[], Path],
        open_privacy_settings: Callable[[], None],
        confirm: Callable[[str, str], Awaitable[bool]],
        present_session_modal: Callable[
            [ConsoleTerminalSessionModal],
            Awaitable[TerminalSessionFormResult | None],
        ],
        marshal_to_ui: Callable[[Callable[[], None]], None],
        schedule_frame: Callable[[Callable[[], None]], None],
        shell_choices: Callable[[], tuple[ShellChoice, ...]],
        run_async: Callable[[Callable[[], Awaitable[Any]]], None] = _spawn_async,
    ) -> None:
        self._terminal_runtime = terminal_runtime
        self._workspace_accessor = workspace_accessor
        self._selected_local_root = selected_local_root
        self._account_home = account_home
        self._open_privacy_settings = open_privacy_settings
        self._confirm = confirm
        self._present_session_modal = present_session_modal
        self._marshal_to_ui = marshal_to_ui
        self._schedule_frame = schedule_frame
        self._shell_choices = shell_choices
        self._run_async = run_async
        self._view: TerminalViewToken | None = None
        self._subscription: TerminalSubscriptionToken | None = None
        self._mount_generation = 0
        self._frame_pending = False
        self._last_fingerprint: object | None = None
        self._visible_session_id: str | None = None
        #: TASK-33625.3: rail sides the user asked for during this visit.
        self._reclaimed_rails: frozenset[str] = frozenset()

    @property
    def is_open(self) -> bool:
        """Return whether this screen controller holds a current view token."""
        return self._view is not None

    def open_workspace(self) -> bool:
        """Attach one fresh view generation and queue its initial projection.

        Returns:
            ``True`` when a view is already open or attaches successfully;
            ``False`` when attachment or subscription fails closed.
        """
        if self._view is not None:
            return True
        runtime = self._terminal_runtime()
        try:
            view = runtime.attach_view()
        except Exception:
            self._status("Terminal view is unavailable.")
            return False
        self._mount_generation += 1
        generation = self._mount_generation
        self._view = view
        self._reclaimed_rails = frozenset()  # each visit starts from the yield
        self._last_fingerprint = None
        self._visible_session_id = None
        try:
            self._subscription = runtime.subscribe(
                lambda: self._manager_changed(generation)
            )
        except Exception:
            try:
                runtime.detach_view(view)
            except Exception:
                pass
            self._view = None
            self._status("Terminal view is unavailable.")
            return False
        self._queue_refresh(generation)
        return True

    def detach_workspace(self) -> bool:
        """Invalidate callbacks before releasing the manager view generation."""
        view = self._view
        subscription = self._subscription
        if view is None:
            return False
        self._mount_generation += 1
        self._view = None
        self._subscription = None
        self._frame_pending = False
        self._last_fingerprint = None
        self._visible_session_id = None
        runtime = self._terminal_runtime()
        if subscription is not None:
            try:
                runtime.unsubscribe(subscription)
            except Exception:
                pass
        try:
            return runtime.detach_view(view) is True
        except Exception as error:
            # TASK-32801.3: this module had no logger at all, so every
            # runtime failure here was reported only as a False return.
            # Type and message only -- the file sink runs with
            # diagnose=True, so a traceback would dump frame locals.
            logger.warning(
                "Terminal detach_view failed ({}): {}",
                type(error).__name__,
                error,
            )
            return False

    async def request_arm(self) -> bool:
        """Arm only after persisted unlock and manager-enforced disclosure."""
        runtime = self._terminal_runtime()
        if getattr(runtime, "permitted", False) is not True:
            self._open_privacy_settings()
            return False
        try:
            result = runtime.arm()
        except Exception:
            self._status("Terminal could not be armed.")
            return False
        if getattr(result, "disclosure_required", False) is True:
            confirmed = await self._confirm(
                "Arm Terminal for this launch?",
                "\n\n".join(TERMINAL_DISCLOSURE_LINES),
            )
            if not confirmed:
                return False
            try:
                result = runtime.arm(acknowledge_disclosure=True)
            except Exception:
                self._status("Terminal could not be armed.")
                return False
        armed = getattr(result, "armed", False) is True
        if not armed:
            self._status("Terminal remained unarmed.")
        return armed

    async def request_new_session(self) -> bool:
        """Collect, revalidate, and submit one bounded launch request."""
        runtime = self._terminal_runtime()
        if getattr(runtime, "permitted", False) is not True:
            self._open_privacy_settings()
            return False
        if getattr(runtime, "armed", False) is not True:
            self._status("Terminal is not armed for this launch.")
            return False
        try:
            projections = tuple(runtime.projections())
            existing_names = tuple(item.name for item in projections)
            choices = tuple(self._shell_choices())
            start_directory = resolve_start_directory(
                self._selected_local_root(),
                account_home=self._account_home(),
            )
        except Exception:
            self._status("Terminal launch choices are unavailable.")
            return False
        result = await self._present_session_modal(
            ConsoleTerminalSessionModal(
                mode="new",
                name=build_default_terminal_name(existing_names),
                shell_choices=choices,
                start_directory=start_directory,
                existing_names=existing_names,
            )
        )
        if result is None or result.shell is None or result.start_directory is None:
            return False
        try:
            name = normalize_session_name(
                result.name,
                existing_names=(item.name for item in runtime.projections()),
            )
            directory = resolve_start_directory(
                self._selected_local_root(),
                requested_directory=result.start_directory,
                account_home=self._account_home(),
            )
            shell = resolve_shell_choice(result.shell, choices)
            workspace = self._workspace_accessor()
            columns, rows = (
                workspace.terminal_size() if workspace is not None else (80, 24)
            )
            request = TerminalLaunchRequest(
                name=name,
                shell=shell.key,
                start_directory=str(directory),
                columns=columns,
                rows=rows,
            )
        except Exception:
            self._status("Terminal session values changed and are no longer valid.")
            return False
        try:
            created = await asyncio.to_thread(runtime.create_session, request)
        except Exception:
            self._status("Terminal session could not start.")
            return False
        admitted = getattr(created, "admitted", False) is True
        if not admitted:
            self._status("Terminal session was refused.")
            return False
        # TASK-33622.5: the manager selects a new session only when nothing
        # is selected, so Terminal 2 left Terminal 1 taking the keys, and no
        # create ever focused the viewport -- typed shell commands went to the
        # chat composer. Select the new session and hand it the keyboard.
        projection = getattr(created, "projection", None)
        session_id = getattr(projection, "session_id", None)
        if isinstance(session_id, str):
            self.request_focus(session_id)
        return True

    async def request_rename(self, session_id: str) -> bool:
        """Validate a user-visible rename before manager revalidation."""
        session = self._session_view(session_id)
        view = self._view
        if session is None or view is None:
            return False
        runtime = self._terminal_runtime()
        others = tuple(
            item.name for item in runtime.projections() if item.session_id != session_id
        )
        result = await self._present_session_modal(
            ConsoleTerminalSessionModal(
                mode="rename",
                name=session.projection.name,
                shell_choices=(),
                start_directory=self._account_home(),
                existing_names=others,
            )
        )
        if result is None:
            return False
        try:
            name = normalize_session_name(result.name, existing_names=others)
            return runtime.rename_session(session_id, name, view=view) is True
        except Exception:
            self._status("Terminal session could not be renamed.")
            return False

    async def request_close(self, session_id: str) -> bool:
        """Confirm termination for active shells, then request bounded cleanup."""
        view = self._view
        if view is None:
            return False
        projection = self._terminal_runtime().projection(session_id)
        if projection is None:
            return False
        if projection.lifecycle in {
            TerminalLifecycle.RUNNING,
            TerminalLifecycle.DRAINING,
        }:
            confirmed = await self._confirm(
                "Close Terminal session?",
                "Closing this session will terminate its shell and running programs.",
            )
            if not confirmed:
                return False
        try:
            receipt = self._terminal_runtime().close_session(session_id, view=view)
        except Exception:
            self._status("Terminal session could not be closed.")
            return False
        return receipt is not None

    def reclaim_rails(self, *, left_open: bool | None, right_open: bool | None) -> None:
        """Exempt the rail sides the user just asked for from the Terminal's yield.

        TASK-33625.3: under a narrow Terminal both rails yield by default,
        but Alt+C, Alt+I or a handle must still open (or close) a rail;
        otherwise the toggle writes a preference and shows nothing.

        Args:
            left_open: The Context rail request, or ``None`` when untouched.
            right_open: The Inspect rail request, or ``None`` when untouched.
        """
        sides = (("left", left_open), ("right", right_open))
        self._reclaimed_rails |= {side for side, value in sides if value is not None}

    def yield_rails(self, rail_state: Any, available_columns: int | None) -> Any:
        """Return ``rail_state`` with the narrow-width Terminal rail yield applied.

        Args:
            rail_state: The effective ``ConsoleRailState`` after every other rule.
            available_columns: Current Console width, when known.

        Returns:
            The rail state the Terminal center renders with.
        """
        from tldw_chatbook.Chat.console_rail_state import (
            yield_console_rails_to_terminal,
        )

        return yield_console_rails_to_terminal(
            rail_state, available_columns, reclaimed=self._reclaimed_rails
        )

    def owns_widget(self, widget: object | None) -> bool:
        """Return whether ``widget`` is a visible part of the mounted workspace.

        TASK-33622.5: the screen's pane/focus bookkeeping asks this so the
        Terminal center counts as the Console's center pane while it is open.

        Args:
            widget: Focused (or candidate) widget, or ``None``.

        Returns:
            ``True`` only for the attached workspace or a descendant of it
            whose whole ancestor chain is displayed.
        """
        workspace = self._workspace_accessor()
        if workspace is None or not getattr(workspace, "is_attached", False):
            return False
        if widget is not workspace and workspace not in getattr(
            widget, "ancestors", ()
        ):
            return False
        current: Any = widget
        while current is not None:
            if current.display is False or current.styles.display == "none":
                return False
            current = getattr(current, "parent", None)
        return True

    def focus_target_ids(self) -> tuple[str, ...]:
        """Return the workspace's focus candidates, most useful first.

        TASK-33622.5: the live session's viewport when one is selected and
        armed; otherwise the one control that moves the user forward.

        Returns:
            Widget ids for the screen's displayed-first focus walk.
        """
        runtime = self._terminal_runtime()
        if (
            getattr(runtime, "armed", False) is True
            and getattr(runtime, "selected_session_id", None) is not None
        ):
            return ("console-terminal-viewport", "console-terminal-return")
        return (
            "console-terminal-retry",
            "console-terminal-new",
            "console-terminal-arm",
            "console-terminal-open-settings",
            "console-terminal-return",
        )

    def request_focus(self, session_id: str) -> bool:
        """Select a retained session through the current view generation."""
        view = self._view
        if view is None:
            return False
        try:
            focused = self._terminal_runtime().focus_session(session_id, view=view)
        except Exception as error:
            # TASK-32801.3: this module had no logger at all, so every
            # runtime failure here was reported only as a False return.
            # Type and message only -- the file sink runs with
            # diagnose=True, so a traceback would dump frame locals.
            logger.warning(
                "Terminal focus_session failed ({}): {}",
                type(error).__name__,
                error,
            )
            return False
        if focused is True:
            workspace = self._workspace_accessor()
            if workspace is not None:
                workspace.focus_terminal()
            return True
        return False

    def request_retry_cleanup(self, session_id: str) -> bool:
        """Request the sole user-authorized fresh cleanup attempt."""
        view = self._view
        if view is None:
            return False
        try:
            return (
                self._terminal_runtime().retry_cleanup(session_id, view=view)
                is not None
            )
        except Exception as error:
            # TASK-32801.3: this module had no logger at all, so every
            # runtime failure here was reported only as a False return.
            # Type and message only -- the file sink runs with
            # diagnose=True, so a traceback would dump frame locals.
            logger.warning(
                "Terminal retry_cleanup failed ({}): {}",
                type(error).__name__,
                error,
            )
            return False

    def send_key(self, data: bytes) -> bool:
        """Offer one encoded key only for the currently selected session."""
        selected = self._selected_session_id()
        view = self._view
        if selected is None or view is None:
            return False
        try:
            result = self._terminal_runtime().send_key(selected, data, view=view)
        except Exception as error:
            # TASK-32801.3: this module had no logger at all, so every
            # runtime failure here was reported only as a False return.
            # Type and message only -- the file sink runs with
            # diagnose=True, so a traceback would dump frame locals.
            logger.warning(
                "Terminal send_key failed ({}): {}",
                type(error).__name__,
                error,
            )
            return False
        return getattr(result, "accepted", False) is True

    def send_paste(self, text: str, *, bracketed: bool) -> bool:
        """Offer one atomic paste through the manager-owned input actor."""
        selected = self._selected_session_id()
        view = self._view
        if selected is None or view is None:
            return False
        try:
            result = self._terminal_runtime().send_paste(
                selected,
                text,
                bracketed=bracketed,
                view=view,
            )
        except Exception:
            return False
        return getattr(result, "accepted", False) is True

    async def request_resize(self, columns: int, rows: int) -> bool:
        """Offer and apply the latest selected-session resize for this view."""
        selected = self._selected_session_id()
        view = self._view
        if selected is None or view is None:
            return False
        runtime = self._terminal_runtime()
        try:
            offered = runtime.resize_session(
                selected,
                columns=columns,
                rows=rows,
                view=view,
            )
            if offered is not True:
                return False
            return await runtime.apply_pending_resize(selected, view=view) is True
        except Exception:
            return False

    async def handle_action(
        self, action: TerminalAction, session_id: str | None = None
    ) -> bool:
        """Route one typed workspace action without involving the model queue."""
        if action == "open-settings":
            self._open_privacy_settings()
            return True
        if action == "arm":
            return await self.request_arm()
        if action == "new":
            return await self.request_new_session()
        if action == "jump-live":
            workspace = self._workspace_accessor()
            if workspace is None:
                return False
            workspace.jump_live()
            return True
        if action == "focus" and session_id is not None:
            workspace = self._workspace_accessor()
            if workspace is None:
                return False
            workspace.focus_terminal()
            return True
        if session_id is None:
            return False
        if action == "select":
            return self.request_focus(session_id)
        if action == "rename":
            return await self.request_rename(session_id)
        if action == "close":
            return await self.request_close(session_id)
        if action == "retry":
            return self.request_retry_cleanup(session_id)
        return False

    def _manager_changed(self, generation: int) -> None:
        if generation != self._mount_generation or self._view is None:
            return
        self._marshal_to_ui(lambda: self._queue_refresh(generation))

    def _queue_refresh(self, generation: int) -> None:
        if (
            generation != self._mount_generation
            or self._view is None
            or self._frame_pending
        ):
            return
        self._frame_pending = True
        self._schedule_frame(lambda: self._refresh(generation))

    def _refresh(self, generation: int) -> None:
        if generation != self._mount_generation or self._view is None:
            return
        self._frame_pending = False
        runtime = self._terminal_runtime()
        try:
            view_state = runtime.view_state(self._view)
            permitted = runtime.permitted is True
            armed = runtime.armed is True
        except Exception:
            return
        if not isinstance(view_state, TerminalViewState):
            return
        fingerprint = _projection_fingerprint(permitted, armed, view_state)
        if fingerprint == self._last_fingerprint:
            return
        workspace = self._workspace_accessor()
        if workspace is not None:
            try:
                workspace.project(
                    permitted=permitted,
                    armed=armed,
                    view_state=view_state,
                )
                selected = view_state.selected_session_id if armed else None
                if selected is not None and selected != self._visible_session_id:
                    columns, rows = workspace.terminal_size()
                    self._run_async(
                        lambda: self._resize_generation(
                            generation,
                            selected,
                            columns,
                            rows,
                        )
                    )
                self._visible_session_id = selected
            except NoMatches:
                # Textual may briefly retain a mounted parent after removing its
                # children during recompose. Keep this projection retryable.
                return
        self._last_fingerprint = fingerprint

    async def _resize_generation(
        self,
        generation: int,
        session_id: str,
        columns: int,
        rows: int,
    ) -> bool:
        """Resize only if the originally visible view generation remains current."""
        if generation != self._mount_generation or self._view is None:
            return False
        if self._selected_session_id() != session_id:
            return False
        return await self.request_resize(columns, rows)

    def _session_view(self, session_id: str) -> TerminalSessionView | None:
        view = self._view
        if view is None:
            return None
        try:
            state = self._terminal_runtime().view_state(view)
        except Exception:
            return None
        if not isinstance(state, TerminalViewState):
            return None
        return next(
            (
                session
                for session in state.sessions
                if session.projection.session_id == session_id
            ),
            None,
        )

    def _selected_session_id(self) -> str | None:
        view = self._view
        if view is None:
            return None
        try:
            state = self._terminal_runtime().view_state(view)
        except Exception:
            return None
        return (
            state.selected_session_id if isinstance(state, TerminalViewState) else None
        )

    def _status(self, message: str) -> None:
        workspace = self._workspace_accessor()
        setter = getattr(workspace, "set_status", None)
        if callable(setter):
            setter(message)


def sync_terminal_center_chrome(screen: Any) -> None:
    """Collapse the chat composer while the Terminal owns the Console center.

    TASK-33622.5 (owner decision 2026-10-27): typing into a visible composer
    under a host terminal was a mode error -- shell commands became chat
    drafts and Enter sent them to the model. The composer is hidden AND
    unfocusable here (``focus()`` still lands on a ``display: none`` widget),
    the footer names Terminal keys, and chat comes back with "Return to
    transcript". The user's own composer collapse is left untouched.

    Args:
        screen: The Console ``ChatScreen`` whose center was just swapped.
    """
    terminal_open = screen._console_terminal_open is True
    composer = screen._console_composer_or_none()
    if composer is not None:
        if terminal_open:
            screen._console_unknown_send_armed = None
            popup = screen._console_command_popup_or_none()
            if popup is not None:
                popup.hide()
        composer.display = not terminal_open
        composer.can_focus = (
            not terminal_open
            and not screen._console_composer_collapsed
            and not screen._console_setup_modal_blocking()
        )
    screen._register_console_footer_shortcuts()


def capture_center_reading(screen: Any, center: Any) -> None:
    """Remember the transcript reading position before the Terminal replaces it.

    TASK-33622.8: the swap removes the transcript region, and its return
    mounts a fresh one; this is the position "Return to transcript" restores.

    Args:
        screen: The Console ``ChatScreen`` swapping its center.
        center: The center about to be removed (only a transcript region
            carries a reading state).
    """
    capture = getattr(center, "capture_reading_state", None)
    if callable(capture):
        # Keyed by conversation: Return must never apply this offset to
        # another conversation activated while the Terminal was open.
        screen._console_terminal_return_reading = (
            _active_console_session_id(screen),
            capture(),
        )


def _active_console_session_id(screen: Any) -> str | None:
    """Return the Console store's active session id, or ``None`` without one."""
    store = getattr(screen, "_console_chat_store", None)
    return getattr(store, "active_session_id", None)


def finish_center_swap(screen: Any) -> None:
    """Settle chrome, focus and content after the Console center was swapped.

    TASK-33622.8: returning mounts a fresh transcript center, and nothing
    re-synced it -- the tab strip showed only "New tab", a populated chat
    read "No messages yet." (the transcript sync gate still held the
    pre-swap token), and focus sat on the transcript, where "y" opened
    Trace. The return now re-renders tabs and transcript, restores the
    captured reading position, and focuses the composer.

    Args:
        screen: The Console ``ChatScreen`` whose center was just mounted.
    """
    sync_terminal_center_chrome(screen)
    # TASK-33625.3: a narrow Terminal takes the rails, and Return gives them back.
    screen._sync_console_rail_visibility_if_changed(
        screen._current_console_rail_state()
    )
    if screen._console_terminal_open:
        screen.call_after_refresh(screen._focus_open_console_terminal)
        return
    reading = screen._console_terminal_return_reading
    screen._console_terminal_return_reading = None
    # The mounted transcript is new: its sync gate must not skip it.
    screen._console_transcript_sync_token = None
    screen.run_worker(
        _resync_returned_transcript(screen, reading),
        group="console-sync",
        exclusive=True,
    )
    screen.call_after_refresh(
        screen._focus_console_workbench_target, "console-native-composer"
    )


async def _resync_returned_transcript(screen: Any, reading: Any) -> None:
    """Render the returned center, then put the reader back where they were.

    ``reading`` is the ``(session id, reading state)`` pair captured before the
    swap, or ``None``; a different active conversation keeps its own position.
    """
    await screen._sync_native_console_chat_ui()
    if reading is None:
        return
    session_id, state = reading
    if session_id != _active_console_session_id(screen):
        return
    screen.call_after_refresh(screen._restore_console_transcript_reading_state, state)


def _projection_fingerprint(
    permitted: bool,
    armed: bool,
    state: TerminalViewState,
) -> tuple[object, ...]:
    """Ignore hidden screen generations while retaining all safe metadata."""
    return (
        permitted,
        armed,
        state.selected_session_id,
        tuple(
            (
                session.projection,
                session.shell,
                session.start_directory,
                session.columns,
                session.rows,
                session.cleanup_receipt,
                (
                    session.screen
                    if session.projection.session_id == state.selected_session_id
                    else None
                ),
            )
            for session in state.sessions
        ),
    )
