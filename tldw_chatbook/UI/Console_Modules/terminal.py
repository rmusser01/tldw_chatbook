"""Thin Console controller for the app-owned persistent Terminal manager."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

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
        except Exception:
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
        return admitted

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

    def request_focus(self, session_id: str) -> bool:
        """Select a retained session through the current view generation."""
        view = self._view
        if view is None:
            return False
        try:
            focused = self._terminal_runtime().focus_session(session_id, view=view)
        except Exception:
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
        except Exception:
            return False

    def send_key(self, data: bytes) -> bool:
        """Offer one encoded key only for the currently selected session."""
        selected = self._selected_session_id()
        view = self._view
        if selected is None or view is None:
            return False
        try:
            result = self._terminal_runtime().send_key(selected, data, view=view)
        except Exception:
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
