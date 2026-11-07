"""Late-bound Console workflow views; the native coordinator owns accepted work."""

from __future__ import annotations

import asyncio
import sqlite3

from loguru import logger
from pydantic import ValidationError
from textual.widgets import Button

from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
from tldw_chatbook.Workspaces.registry_service import WorkspaceRegistryServiceError

WORKFLOW_VIEW_ERRORS = (
    ValueError,
    RuntimeError,
    OSError,
    sqlite3.Error,
    WorkspaceRegistryServiceError,
)


_VIEW_CONTEXTS = frozenset(
    {
        "History",
        "Workflow history",
        "Setup",
        "Prepare plan",
        "Review",
        "Checkpoint",
        "Approve",
        "Accept",
        "Revise",
        "Resume",
        "Pause",
        "Stop",
        "Recovery",
        "Remove",
    }
)
_VIEW_REASONS = frozenset(
    {
        "Open a Console chat first.",
        "Add a ready local folder in F4 Settings → Workspaces.",
        "Workflow belongs to another chat.",
        "workflow provider selection changed",
        "Displayed planning provider or model changed. Reopen setup.",
        "This review changed. Review the refreshed revision.",
        "stale_workflow_decision",
        "workflow_snapshot_superseded",
        "remaining_requires_clean_checkpoint",
        "Recovery required · uncertain_execution",
        "revision_conflict",
        "workflow_command_conflict",
        "workflow_already_exists",
        "generation_budget",
        "child_launch_budget",
        "model_call_budget",
        "tokens_budget",
        "output_tokens_budget",
        "wall_budget",
        "clock_unknown",
        "clock_reversed",
        "workflow_resources_still_live",
        "recovery_audit_required",
        "recovery_unavailable",
        "workflow_runtime_changed",
        "workflow_has_active_evidence",
    }
)
_ERROR_TYPES = (
    sqlite3.OperationalError,
    sqlite3.IntegrityError,
    sqlite3.ProgrammingError,
    sqlite3.DataError,
    sqlite3.InterfaceError,
    sqlite3.NotSupportedError,
    sqlite3.InternalError,
    sqlite3.DatabaseError,
    sqlite3.Error,
    WorkspaceRegistryServiceError,
    AutomaticWorkRefused,
    PermissionError,
    OSError,
    ValueError,
    RuntimeError,
)


def _error_metadata(error: BaseException) -> str:
    if isinstance(error, ValidationError):
        details = error.errors(
            include_url=False, include_context=False, include_input=False
        )
        if not details:
            return "ValidationError"
        first = details[0]
        field = first["loc"][0] if first["loc"] else None
        if field not in {
            "objective",
            "conversation_id",
            "binding",
            "policy",
            "targets",
            "planner",
            "phases",
            "evidence",
        }:
            field = "record"
        code = first["type"]
        if code not in {
            "value_error",
            "missing",
            "extra_forbidden",
            "string_type",
            "string_too_short",
            "string_too_long",
            "int_type",
            "tuple_type",
            "literal_error",
        }:
            code = "invalid_value"
        return f"ValidationError ({field}/{code})"
    category = next(
        (kind.__name__ for kind in _ERROR_TYPES if isinstance(error, kind)), "Exception"
    )
    if isinstance(error, sqlite3.Error):
        code = getattr(error, "sqlite_errorcode", None)
        if type(code) is int and 0 <= code <= 65535:
            return f"{category} (sqlite_code={code})"
    elif (
        len(error.args) == 1
        and type(error.args[0]) is str
        and error.args[0] in _VIEW_REASONS
    ):
        return f"{category} · {error.args[0]}"
    return category


def workflow_view_error(context: str, error: BaseException) -> str:
    """Describe an unavailable operation without copying private data out of its owner."""
    operation = (
        context if type(context) is str and context in _VIEW_CONTEXTS else "Workflow"
    )
    reason = _error_metadata(error)
    cause = _error_metadata(error.__cause__) if error.__cause__ is not None else "none"
    logger.warning("Workflow {} unavailable: {} (cause: {})", operation, reason, cause)
    if error.args == ("workflow_resources_still_live",):
        return f"{operation} unavailable · Work or cleanup is still active. Wait, then refresh."
    if error.args == ("recovery_audit_required",):
        return f"{operation} unavailable · Restart Chatbook to audit interrupted work, then review it."
    return f"{operation} unavailable · {reason}"


def workflows_enabled() -> bool:
    from tldw_chatbook.Agents.run_log import _setting
    from tldw_chatbook.config import coerce_bool_setting

    return coerce_bool_setting(_setting("workflow_runs_enabled", False), False)


class WorkflowViewReads:
    """Join finite presentation reads before the host retires its database owners."""

    def __init__(self):
        self.tasks = set()
        self.closed = False

    async def read(self, operation, /, *args, **kwargs):
        if self.closed:
            raise asyncio.CancelledError
        task = asyncio.create_task(operation(*args, **kwargs))
        self.tasks.add(task)
        task.add_done_callback(self._returned)
        return await asyncio.shield(task)

    def _returned(self, task):
        self.tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def drain(self):
        self.closed = True
        if self.tasks:
            returned = asyncio.gather(*self.tasks, return_exceptions=True)
            cancellation = None
            while not returned.done():
                try:
                    await asyncio.shield(returned)
                except asyncio.CancelledError as error:
                    cancellation = error
            if cancellation is not None:
                raise cancellation


class ConsoleWorkflowLine(Button):
    """One compact link to the exact foreground workflow or background review."""

    def __init__(self):
        super().__init__("", id="console-workflow-line")
        self.display = False
        self.target = None

    def show_workflow(self, conversation_id, workflow_id, label):
        self.target = (conversation_id, workflow_id)
        self.label = label
        self.display = True

    def on_button_pressed(self, event):
        if self.target:
            event.stop()
            self.screen._workflows.open_detail(*self.target)


class ConsoleWorkflowsController:
    def __init__(self, screen):
        self.screen = screen
        self.reads = WorkflowViewReads()
        self._unsubscribe = None
        self._active_id = None
        self._background = None

    def _coordinator(self):
        runtime = self.screen._console_runtime()
        if runtime.chat_controller is None:
            self.screen._ensure_console_chat_controller()
        return runtime.ensure_workflow_coordinator()

    def _session(self, conversation_id=None):
        store = self._coordinator().controller.store
        return next(
            (
                s
                for s in store.sessions()
                if (
                    (s.persisted_conversation_id or s.id) == conversation_id
                    if conversation_id is not None
                    else s.id == store.active_session_id
                )
            ),
            None,
        )

    def command(self, args):
        if not args.strip():
            self.open_history()
        elif args.strip().lower() == "new":
            self.open_setup()
        else:
            self.screen.notify(
                "Use /workflow for this chat’s history or /workflow new for setup.",
                severity="warning",
            )

    def open_history(self):
        from tldw_chatbook.Widgets.Console.console_workflow_status import (
            ConsoleWorkflowHistory,
        )

        try:
            session = self._session()
            if session is None:
                raise ValueError("Open a Console chat first.")
            chat = session.persisted_conversation_id or session.id
            self.screen.app.push_screen(
                ConsoleWorkflowHistory(
                    self._coordinator(), chat, self.open_detail, self.open_setup
                )
            )
        except WORKFLOW_VIEW_ERRORS as error:
            self.screen.notify(
                workflow_view_error("History", error), severity="warning", markup=False
            )

    def open_setup(self, conversation_id=None):
        from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified

        if not native_workflow_qualified():
            self.screen.notify(
                "Native workflows are unavailable on this platform.", severity="warning"
            )
            return
        if not workflows_enabled():
            self.screen.notify(
                "Enable workflow runs in F4 Settings → Console behavior.",
                severity="warning",
            )
            return
        try:
            session = self._session(conversation_id)
        except WORKFLOW_VIEW_ERRORS as error:
            self.screen.notify(
                workflow_view_error("Setup", error), severity="warning", markup=False
            )
            return
        if (
            session is None
            or session.ephemeral
            or session.runtime_backend != "local"
            or session.assistant_kind == "character"
        ):
            self.screen.notify(
                "Workflow planning requires a local, saved Console chat.",
                severity="warning",
            )
            return
        self.screen.run_worker(
            self._setup(session), group="workflow-setup-view", exclusive=True
        )

    async def _setup(self, session):
        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
        from tldw_chatbook.Widgets.Console.console_workflow_setup import (
            ConsoleWorkflowSetup,
        )

        try:
            coordinator = self._coordinator()
            conversation_id = session.persisted_conversation_id or session.id
            selection = coordinator.controller._provider_selection_for_session(
                session.id
            )
            planner = (
                selection.provider,
                selection.explicit_model or selection.configured_model or "",
            )
            choices = await self.reads.read(
                coordinator.service.call,
                coordinator.service.setup_choices,
                session.workspace_id,
            )
            if not choices[0]:
                raise ValueError(
                    "Add a ready local folder in F4 Settings → Workspaces."
                )
            if self._session() is not session or not self.screen.is_mounted:
                return

            def started(result):
                if result is not None and self.screen.is_mounted:
                    self.open_detail(result.request.conversation_id, result.id)

            self.screen.app.push_screen(
                ConsoleWorkflowSetup(
                    coordinator,
                    conversation_id,
                    planner,
                    AutomaticWorkLimits.from_settings("workflow_phase"),
                    choices,
                ),
                callback=started,
            )
        except WORKFLOW_VIEW_ERRORS as error:
            if not self.reads.closed and self.screen.is_mounted:
                self.screen.notify(
                    workflow_view_error("Setup", error)
                    + " Reopen /workflow new to retry.",
                    severity="warning",
                    markup=False,
                )

    def open_detail(self, conversation_id, workflow_id):
        from tldw_chatbook.Widgets.Console.console_workflow_status import (
            ConsoleWorkflowStatus,
        )

        self.screen.app.push_screen(
            ConsoleWorkflowStatus(
                self._coordinator(),
                conversation_id,
                workflow_id,
                open_changes=self.open_changes,
                open_fleet=self.open_fleet,
                pending_approvals=self.pending_approvals,
                open_approval=self.open_approval,
            )
        )

    def open_changes(self, snapshot, run_id, window):
        self.screen._open_change_review(
            run_id,
            conversation_id=snapshot.request.conversation_id,
            review_workspace_roots=(snapshot.request.binding.locator,),
            initial_snapshot_id=window,
        )

    def pending_approvals(self, conversation_id):
        session = self._session()
        return bool(
            session
            and (session.persisted_conversation_id or session.id) == conversation_id
            and self.screen._console_pending_approval_count()
        )

    def open_approval(self, modal, conversation_id):
        if self.pending_approvals(conversation_id) and modal.dismiss_safe_once(None):
            self.screen.call_after_refresh(self.screen.action_review_pending_approval)

    def open_fleet(self, modal, conversation_id, run_id):
        session = self._session()
        if (
            session is None
            or (session.persisted_conversation_id or session.id) != conversation_id
        ):
            self.screen.notify(
                "Switch to the workflow’s chat to inspect or steer its fleet.",
                severity="warning",
            )
            return
        if not modal.dismiss_safe_once(None):
            return
        if run_id:
            self.screen._reveal_console_inspector_rail()
            self.screen._agent._drill_into_console_agent_subagent(run_id)
        else:
            self.screen._agent.open_fleet_progress()

    def select_chat(self, session):
        if not self._unsubscribe:
            self._unsubscribe = self._coordinator().subscribe(self._published)
        chat = session.persisted_conversation_id or session.id if session else None
        if chat == self._active_id:
            return
        self._active_id = chat
        line = self.screen.query_one("#console-workflow-line", ConsoleWorkflowLine)
        line.display = False
        if self._background:
            self._published(self._background)
        if chat:
            self.screen.run_worker(
                self._load_line(chat), group="workflow-line-read", exclusive=True
            )

    async def _load_line(self, chat):
        row = None
        try:
            coordinator = self._coordinator()
            page = await self.reads.read(
                coordinator.service.call, coordinator.service.list_history, chat
            )
            row = next(
                (
                    r
                    for r in page
                    if r["status"] not in {"closed", "completed", "stopped", "removed"}
                ),
                None,
            )
            if row and self._active_id == chat and self.screen.is_mounted:
                full = await self.reads.read(coordinator.get, row["id"])
                if self._active_id == chat and self.screen.is_mounted:
                    self._show_line(chat, full.id, full.status)
        except WORKFLOW_VIEW_ERRORS as error:
            message = workflow_view_error("Workflow history", error)
            if (
                not self.reads.closed
                and self._active_id == chat
                and self.screen.is_mounted
            ):
                if row:
                    self._show_line(chat, row["id"], "review_unavailable")
                else:
                    self.screen.notify(
                        message + " Open /workflow to retry.",
                        severity="warning",
                        markup=False,
                    )

    def _show_line(self, chat, workflow_id, status):
        from tldw_chatbook.Widgets.Console.console_workflow_status import status_label

        label = f"Workflow · {status_label(status)}"
        if chat != self._active_id:
            label += f" · chat {chat}"
        if not self.screen.query("#console-workflow-line"):
            return
        self.screen.query_one(
            "#console-workflow-line", ConsoleWorkflowLine
        ).show_workflow(chat, workflow_id, label)

    def _published(self, view):
        if not self.screen.is_mounted:
            return
        attention = view.status in {
            "awaiting_phase_review",
            "awaiting_result_review",
            "recovery_required",
            "paused",
        }
        if self._background and self._background.id == view.id and attention:
            self._background = view
        elif self._background and self._background.id == view.id:
            self._background = None
            line = self.screen.query_one("#console-workflow-line", ConsoleWorkflowLine)
            if view.conversation_id != self._active_id and line.target == (
                view.conversation_id,
                view.id,
            ):
                line.target = None
                line.display = False
                if self._active_id:
                    self.screen.run_worker(
                        self._load_line(self._active_id),
                        group="workflow-line-read",
                        exclusive=True,
                    )
        if view.conversation_id != self._active_id:
            if not attention:
                return
            self._background = view
        self._show_line(view.conversation_id, view.id, view.status)

    async def detach(self):
        if self._unsubscribe:
            self._unsubscribe()
            self._unsubscribe = None
        await self.reads.drain()
