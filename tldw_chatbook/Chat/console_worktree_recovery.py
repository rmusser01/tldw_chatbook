"""Runtime-owned manual recovery; views hold no physical execution lifetime."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from tldw_chatbook.Agents.agent_worktree import WorktreeRefusal
from tldw_chatbook.Agents.agent_worktree_recovery import (
    WorktreeRecoveryOutcome,
    recover_agent_worktree,
)
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.DB.agent_worktrees import AgentWorktreeRepository
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

if TYPE_CHECKING:
    from tldw_chatbook.Agents.local_tool_provider import RunAdmittedWorkspaceRoot

    from .console_project_instructions import ProjectInstructionControlState


@dataclass(frozen=True)
class RecoveryIntent:
    """Pure owning-session values; no transcript, callbacks or runtime locks."""

    id: str
    workspace_id: str
    persisted_conversation_id: str | None
    ephemeral: bool
    project_instruction_state: ProjectInstructionControlState


def capture_intent(controller: Any, session_id: str) -> RecoveryIntent | None:
    """UI thread: copy only the owning session's identity and selection state."""
    session = next((s for s in controller.store.sessions() if s.id == session_id), None)
    if session is None:
        return None
    return RecoveryIntent(
        session.id,
        session.workspace_id,
        session.persisted_conversation_id,
        session.ephemeral,
        session.project_instruction_state,
    )


def validate_intent(
    controller: Any, session_id: str, intent: RecoveryIntent | None
) -> RunAdmittedWorkspaceRoot | None:
    """Worker: resolve exactly the captured named writable binding and fresh gates."""
    from .console_chat_controller import (
        capture_run_admitted_workspace_roots,
        project_instruction_authority_snapshot_is_current,
        resolve_project_instruction_binding,
    )

    if intent is None or not intent.persisted_conversation_id or intent.ephemeral:
        return None
    if not intent.project_instruction_state.working_folder_binding_id:
        return None
    registry = getattr(controller.app, "workspace_registry_service", None)
    try:
        selection = resolve_project_instruction_binding(intent, registry)
        if selection is None or not selection.allow_write:
            return None
        kill_switch = controller._console_tool_kill_switch_reader()

        def guard():
            if kill_switch is None or kill_switch():
                return False
            current = controller.app.call_from_thread(
                capture_intent, controller, session_id
            )
            return bool(
                current is not None
                and current.persisted_conversation_id
                == intent.persisted_conversation_id
                and project_instruction_authority_snapshot_is_current(
                    session_snapshot=current,
                    registry=registry,
                    expected_selection=selection,
                )
            )

        roots = capture_run_admitted_workspace_roots(
            session=intent,
            registry=registry,
            project_selection=selection,
            project_authority_guard=guard,
        )
        return roots[0] if len(roots) == 1 and roots[0].guard(True) else None
    except Exception:  # noqa: BLE001 - stale or unavailable authority fails closed
        return None


@dataclass(frozen=True)
class RecoveryPage:
    rows: tuple[dict[str, Any], ...] = ()
    conversation_id: str = ""
    repository: str = ""
    next_run_id: str | None = None
    message: str = ""


@dataclass
class RecoveryOperation:
    cancel: threading.Event
    task: asyncio.Task


class ConsoleWorktreeRecovery:
    """Independent manual Events, physical owners and bounded owner receipts."""

    def __init__(self, controller: Any, bridge: Any) -> None:
        self.controller = controller
        self.bridge = bridge
        self.operations: dict[str, RecoveryOperation] = {}
        self.receipts: OrderedDict[str, Any] = OrderedDict()
        self.closed = False

    async def list_work(
        self, session_id: str, after_run_id: str | None = None
    ) -> RecoveryPage:
        intent = self.controller.capture_worktree_recovery_intent(session_id)
        receipt = (
            self.receipts.get(intent.persisted_conversation_id)
            if intent is not None
            else None
        )

        def read():
            authority = validate_intent(self.controller, session_id, intent)
            if authority is None:
                return RecoveryPage(
                    message="Select a writable named repository in this conversation's project instructions."
                )
            db = AgentRunsDB(self.bridge.runs_db.db_path_str)
            try:
                rows = AgentWorktreeRepository(db).list_for_conversation(
                    intent.persisted_conversation_id,
                    workspace_id=authority.workspace_id,
                    binding_id=authority.binding_id,
                    limit=51,
                    after_run_id=after_run_id,
                )
                if not authority.guard(True):
                    return RecoveryPage(
                        message="Repository selection changed. Open recovery again."
                    )
                return RecoveryPage(
                    tuple(rows[:50]),
                    intent.persisted_conversation_id,
                    str(authority.root),
                    rows[49]["run_id"] if len(rows) > 50 else None,
                    receipt.message if receipt is not None else "",
                )
            finally:
                db.close()

        return await asyncio.to_thread(read)

    async def start(
        self, session_id: str, run_id: str, action: str
    ) -> WorktreeRecoveryOutcome | WorktreeRefusal:
        if self.closed or session_id in self.operations:
            return WorktreeRefusal(
                "recovery_busy",
                "This conversation already has a recovery operation or is closing.",
            )
        intent = self.controller.capture_worktree_recovery_intent(session_id)
        if intent is None:
            return WorktreeRefusal(
                "missing_session", "The owning conversation is closed."
            )
        cancel = threading.Event()
        owner = self.bridge.runtime_capacity.begin_execution(
            origin=WorkOrigin.MANUAL,
            conversation_id=intent.persisted_conversation_id or session_id,
        )

        def worker():
            db = None
            try:
                with owner.activate():
                    authority = validate_intent(self.controller, session_id, intent)
                    if authority is None:
                        return WorktreeRefusal(
                            "missing_authority",
                            "Select a current writable named repository.",
                        )
                    db = AgentRunsDB(self.bridge.runs_db.db_path_str)
                    return recover_agent_worktree(
                        db,
                        authority=authority,
                        conversation_id=intent.persisted_conversation_id,
                        run_id=run_id,
                        action=action,
                        request_confirmation=partial(
                            self.controller.request_worktree_merge_confirm,
                            session_id=session_id,
                            operation_cancel_event=cancel,
                        ),
                        should_cancel=cancel.is_set,
                    )
            finally:
                if db is not None:
                    db.close()
                owner.finish_root()

        task = asyncio.create_task(asyncio.to_thread(worker))
        self.operations[session_id] = RecoveryOperation(cancel, task)

        def completed(done):
            self.operations.pop(session_id, None)
            if not done.cancelled():
                try:
                    self.receipts[intent.persisted_conversation_id] = done.result()
                except Exception:  # noqa: BLE001 - keep a bounded failure receipt
                    self.receipts[intent.persisted_conversation_id] = WorktreeRefusal(
                        "recovery_failed",
                        "Recovery failed; inspect recorded work before retrying.",
                    )
                while len(self.receipts) > 32:
                    self.receipts.popitem(last=False)

        task.add_done_callback(completed)
        return await asyncio.shield(task)

    def cancel_session(self, session_id: str) -> None:
        if operation := self.operations.get(session_id):
            operation.cancel.set()

    def begin_close(self) -> None:
        """Fence admission synchronously before awaiting physical workers."""
        self.closed = True
        for operation in self.operations.values():
            operation.cancel.set()

    async def close(self) -> None:
        self.begin_close()
        await asyncio.gather(
            *(asyncio.shield(op.task) for op in tuple(self.operations.values())),
            return_exceptions=True,
        )
