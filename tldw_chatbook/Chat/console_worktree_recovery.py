"""Runtime-owned manual recovery; views hold no physical execution lifetime."""

from __future__ import annotations

import asyncio
import threading
from collections import OrderedDict
from concurrent.futures import Future
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
    """Captured session selection, with no transcript, callbacks or runtime locks.

    Attributes:
        id: Owning Console session identifier.
        workspace_id: Session's selected workspace identifier.
        persisted_conversation_id: Durable conversation identifier, if saved.
        ephemeral: Whether the session is temporary and cannot be recovered.
        project_instruction_state: Captured project-instruction selection;
            ``validate_intent`` resolves and rechecks its named folder binding.
    """

    id: str
    workspace_id: str
    persisted_conversation_id: str | None
    ephemeral: bool
    project_instruction_state: ProjectInstructionControlState


def capture_intent(controller: Any, session_id: str) -> RecoveryIntent | None:
    """Copy the owning session's identity and selection state on the UI thread.

    Args:
        controller: Console controller owning the session store and app runtime.
        session_id: Exact session to capture, independent of the active view.

    Returns:
        Pure session values for later worker validation, or ``None`` if the
        session is missing, the runtime is disposed, or the session is fenced.
        A captured intent alone grants no repository authority.
    """
    runtime = getattr(getattr(controller, "app", None), "console_runtime", None)
    if runtime is not None:
        try:
            runtime._raise_if_disposed_or_session_fenced(session_id)
        except RuntimeError:
            return None
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
    """Resolve the captured writable binding and fresh gates on a worker thread.

    Args:
        controller: Owning Console controller, including its workspace registry,
            kill-switch reader and UI-thread dispatch through the app.
        session_id: Session whose current selection must still match the intent.
        intent: Captured selection to validate, or ``None`` for no selection.

    Returns:
        The single admitted named root with a guard that rechecks the session,
        binding and kill switch, or ``None`` for temporary, missing, stale,
        nonwritable or unavailable authority. Resolution failures fail closed.
    """
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
    """Disposable projection of recorded work for one authorized conversation.

    Attributes:
        rows: Up to 50 durable ownership rows, ordered by ascending run ID.
        conversation_id: Persisted conversation ID, empty when authority fails.
        repository: Current selected root path for display, not stored authority;
            empty when authority fails.
        next_run_id: Last displayed run ID to pass as ``after_run_id`` when more
            rows exist, otherwise ``None``.
        message: Latest retained operation receipt or an authority refusal;
            empty when neither is available.
    """

    rows: tuple[dict[str, Any], ...] = ()
    conversation_id: str = ""
    repository: str = ""
    next_run_id: str | None = None
    message: str = ""


@dataclass
class RecoveryOperation:
    """Runtime-retained recovery task and its independent cancellation signal.

    Attributes:
        cancel: Cooperative stop request, separate from primary-turn or view
            cancellation; setting it does not prove physical completion.
        task: Retained worker-submission task, shielded from view waiters and
            awaited on shutdown until admitted physical work has finished.
    """

    cancel: threading.Event
    task: asyncio.Task


class ConsoleWorktreeRecovery:
    """Retain manual recovery across view changes until physical work finishes.

    Each session can own one operation. Completed results are retained for up
    to 32 persisted conversations; these receipts are process-local UI state.
    """

    def __init__(self, controller: Any, bridge: Any) -> None:
        """Bind recovery to its runtime's controller and agent resources.

        Args:
            controller: Owner of session capture and human confirmation.
            bridge: Owner of the runs database path and runtime capacity ledger.
        """
        self.controller = controller
        self.bridge = bridge
        self.operations: dict[str, RecoveryOperation] = {}
        self.receipts: OrderedDict[str, Any] = OrderedDict()
        self.closed = False

    async def list_work(
        self, session_id: str, after_run_id: str | None = None
    ) -> RecoveryPage:
        """Read a page for the session's current authorized repository.

        Authority validation and database reads run on a worker with its own
        database handle. Database errors propagate to the calling view.

        Args:
            session_id: Owning Console session to capture on the UI thread.
            after_run_id: Exclusive ascending run-ID cursor from the preceding
                page, or ``None`` for the first page.

        Returns:
            Up to 50 recorded rows with a continuation cursor and any retained
            receipt, or an empty page explaining missing or changed authority.

        Raises:
            ValueError: A supplied database identifier is invalid.
        """
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
        """Start one confirmed recovery and await its retained worker result.

        The operation owns a manual execution lease and cancellation event.
        Cancelling this caller leaves the shielded task registered until its
        physical worker finishes; use ``cancel_session`` to request a stop.
        Unexpected submission or database failures propagate to the caller and
        leave a failure receipt when the retained task settles.

        Args:
            session_id: Owning Console session to capture on the UI thread.
            run_id: Durable child-run identifier selected for recovery.
            action: Requested ``apply``, ``merge`` or ``discard`` operation;
                the recovery engine validates it and requests human confirmation.

        Returns:
            The engine's outcome or refusal. Closed admission, an already busy
            session, a missing session or unavailable authority also refuses.

        Raises:
            CapacityRefused: The shared runtime capacity ledger is closed.
            asyncio.CancelledError: This caller is cancelled while awaiting the
                result; the retained operation is not cancelled with it.
        """
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

        handoff: Future[None] = Future()

        def worker():
            # A failed submit may already have queued this wrapper. Only an
            # unrevoked claim may acquire DB or engine authority.
            if not handoff.set_running_or_notify_cancel():
                return None
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
                try:
                    if db is not None:
                        db.close()
                finally:
                    owner.finish_root()
                    handoff.set_result(None)

        async def submit():
            try:
                return await asyncio.to_thread(worker)
            except BaseException:
                if handoff.cancel():
                    # Positive non-entry, even if the executor queued a wrapper
                    # before raising: the cancelled gate forbids later entry.
                    owner.finish_root()
                else:
                    # Submission may raise after entry. Keep the operation
                    # registered until its admitted worker physically finishes.
                    await asyncio.shield(asyncio.wrap_future(handoff))
                raise

        task = asyncio.create_task(submit())
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
        """Request a cooperative stop without cancelling or awaiting the task.

        Args:
            session_id: Owning session; a session with no operation is a no-op.
        """
        if operation := self.operations.get(session_id):
            operation.cancel.set()

    def begin_close(self) -> None:
        """Permanently fence admission and request stops without waiting.

        Repeated calls are safe. Existing operations retain their execution
        ownership until physical completion; ``close`` awaits that completion.
        """
        self.closed = True
        for operation in self.operations.values():
            operation.cancel.set()

    async def close(self) -> None:
        """Fence admission, request stops and await all retained operations.

        Worker failures are collected without being re-raised. Cancellation of
        this waiter leaves the shielded operations running and admission closed.

        Raises:
            asyncio.CancelledError: The shutdown waiter itself is cancelled.
        """
        self.begin_close()
        await asyncio.gather(
            *(asyncio.shield(op.task) for op in tuple(self.operations.values())),
            return_exceptions=True,
        )
