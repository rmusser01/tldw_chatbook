"""Retain ordinary Console save and terminal work through cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .console_chat_store import (
        ConsoleChatStore,
        ConsoleDurableAcceptanceFingerprint,
        ConsoleDurableTurnCommit,
    )
    from .console_dispatch_checkpoint import ConsoleDurableTurnAcceptance


@dataclass(frozen=True, slots=True)
class ConsoleNativeCommitCompletion:
    """Actual callback outcome after its operation-owned cleanup has retired."""

    commit: ConsoleDurableTurnCommit | None
    error: BaseException | None
    caller_cancelled: bool


@dataclass(frozen=True, slots=True)
class ConsoleNativeSettlementCompletion:
    """Actual accepted settlement outcome after native cleanup has retired."""

    settled: bool
    error: BaseException | None
    caller_cancelled: bool


@dataclass(frozen=True, slots=True)
class _ConsoleNativeBinding:
    """Pure original-source binding for one synchronous Console callback."""

    store: ConsoleChatStore
    turn_owner: object
    persistence: object
    database: object
    settlement_helper: Callable[..., bool] | None = None


_native_commit_binding: contextvars.ContextVar[_ConsoleNativeBinding | None] = (
    contextvars.ContextVar("console_native_commit_binding", default=None)
)


async def _run_owned_native_call[Result](
    binding: _ConsoleNativeBinding,
    callback: Callable[[], Result],
) -> tuple[Result | None, BaseException | None, bool]:
    """Retire one of the two named Console callbacks on its original source."""
    store, persistence, database = (
        binding.store,
        binding.persistence,
        binding.database,
    )

    def invoke_bound() -> Result:
        if (
            store.persistence is not persistence
            or getattr(persistence, "db", None) is not database
        ):
            raise RuntimeError("Durable Console persistence owner changed.")
        token = _native_commit_binding.set(binding)
        try:
            return callback()
        finally:
            _native_commit_binding.reset(token)

    try:
        memory_backed = bool(getattr(database, "is_memory_db", False))
    except BaseException as error:  # noqa: BLE001 - transport exact failure and cancellation after physical retirement.
        return None, error, False
    if memory_backed:
        try:
            return invoke_bound(), None, False
        except BaseException as error:  # noqa: BLE001 - transport exact failure and cancellation after physical retirement.
            return None, error, False

    submit_resolved = threading.Event()
    submitted = False

    def invoke() -> Result | None:
        # submit() can enqueue a wrapper before thread creation raises. Only a
        # successfully returned Future permits this wrapper to touch storage.
        submit_resolved.wait()
        if not submitted:
            return None
        if (
            store.persistence is not persistence
            or getattr(persistence, "db", None) is not database
        ):
            raise RuntimeError("Durable Console persistence owner changed.")
        from tldw_chatbook.DB.base_db import operation_owned_connection
        from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

        with operation_owned_connection(database):
            if type(database) is CharactersRAGDB and not database.is_memory_db:
                from tldw_chatbook.Backup_Recovery.participants import _core_operation

                with _core_operation(database):
                    return invoke_bound()
            return invoke_bound()

    try:
        context = contextvars.copy_context()
        future = asyncio.get_running_loop().run_in_executor(None, context.run, invoke)
    except BaseException as error:  # noqa: BLE001 - transport exact failure and cancellation after physical retirement.
        submit_resolved.set()
        return None, error, False
    submitted = True
    submit_resolved.set()

    caller_cancelled = False
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError as error:
            # A callback can raise this exact error even if the task retains an
            # earlier cancellation count. Only cancellation of the await marks
            # the caller; consume the original callback exception below.
            if future.done() and not future.cancelled() and future.exception() is error:
                break
            caller_cancelled = True
        except BaseException:  # noqa: BLE001 - transport exact failure and cancellation after physical retirement.
            break
    try:
        return future.result(), None, caller_cancelled
    except BaseException as error:  # noqa: BLE001 - transport exact failure and cancellation after physical retirement.
        return None, error, caller_cancelled


#: TASK-33620.15.4: per store (by id, live only while its commit runs), when
#: each ordinary durable turn commit now in flight began (monotonic seconds).
_DURABLE_COMMITS_IN_FLIGHT: dict[int, list[float]] = {}


@contextlib.contextmanager
def durable_commit_window(store: object) -> Iterator[None]:
    """Mark a send's durable turn commit in flight on ``store`` until it ends.

    TASK-33620.15.4 (measured live 2026-10-23): the commit's ~10 statements
    run on a worker thread for 140-180 ms while the send waits for them, and
    UI-loop work in that window lengthened it (warm sends reached the
    provider 31 ms later when the Console painted during it). The Console
    reads this window (``durable_commit_started_at``) to keep its syncs
    quiet while it is open. It is keyed by store, not session, so any
    session's commit opens it for the whole Console. Ends on success,
    failure and cancellation alike.

    Args:
        store: The Console store whose turn is being committed.
    """
    key, started = id(store), time.monotonic()
    starts = _DURABLE_COMMITS_IN_FLIGHT.setdefault(key, [])
    starts.append(started)
    try:
        yield
    finally:
        starts.remove(started)
        if not starts and _DURABLE_COMMITS_IN_FLIGHT.get(key) is starts:
            del _DURABLE_COMMITS_IN_FLIGHT[key]


def durable_commit_started_at(store: object) -> float | None:
    """When ``store``'s earliest in-flight durable commit began, or None."""
    starts = _DURABLE_COMMITS_IN_FLIGHT.get(id(store))
    return min(starts) if starts else None


async def commit_durable_turn_owned(
    store: ConsoleChatStore,
    acceptance: ConsoleDurableTurnAcceptance,
) -> ConsoleNativeCommitCompletion:
    """Finish the captured save before reporting any caller cancellation.

    Args:
        store: Original store whose callback and persistence binding are captured.
        acceptance: Exact already-admitted ordinary turn.

    Returns:
        The original callback result or error, plus observed caller cancellation.
        This outcome does not establish rollback or permission authority.
    """
    try:
        persistence = store.persistence
        database = getattr(persistence, "db", None)
        callback = store.commit_durable_turn
    except BaseException as error:  # noqa: BLE001 - transport exact failure and cancellation after physical retirement.
        return ConsoleNativeCommitCompletion(None, error, False)
    binding = _ConsoleNativeBinding(store, acceptance, persistence, database)
    with durable_commit_window(store):
        return ConsoleNativeCommitCompletion(
            *await _run_owned_native_call(binding, lambda: callback(acceptance))
        )


async def settle_accepted_durable_turn_owned(
    store: ConsoleChatStore,
    preparation_id: str,
    *,
    fingerprint: ConsoleDurableAcceptanceFingerprint,
    terminal_state: str,
    content: str,
    persistence: object,
    database: object,
    metadata_json: str | None = None,
) -> ConsoleNativeSettlementCompletion:
    """Retire the exact accepted terminal write without blocking the owner loop.

    Args:
        store: Original store retaining the accepted turn.
        preparation_id: Exact accepted preparation.
        fingerprint: Immutable original acceptance owner.
        terminal_state: Only stopped or failed.
        content: Assistant terminal content.
        persistence: Original persistence captured by the controller owner.
        database: Original database captured by the controller owner.
        metadata_json: Optional terminal metadata object.

    Returns:
        Actual settlement/refusal or error, plus observed caller cancellation.
    """
    try:
        callback = store.settle_accepted_durable_turn
        # Retain the moved body alongside the original callback. Python still
        # binds that actual callback's live defaults at its original invocation.
        settlement_helper = _store_settle_accepted_durable_turn
    except BaseException as error:  # noqa: BLE001 - transport exact failure and cancellation after physical retirement.
        return ConsoleNativeSettlementCompletion(False, error, False)

    def settle() -> bool:
        if metadata_json is None:
            return callback(
                preparation_id,
                fingerprint=fingerprint,
                terminal_state=terminal_state,
                content=content,
            )
        return callback(
            preparation_id,
            fingerprint=fingerprint,
            terminal_state=terminal_state,
            content=content,
            metadata_json=metadata_json,
        )

    binding = _ConsoleNativeBinding(
        store, fingerprint, persistence, database, settlement_helper
    )
    result, error, cancelled = await _run_owned_native_call(binding, settle)
    return ConsoleNativeSettlementCompletion(bool(result), error, cancelled)




if TYPE_CHECKING:
    from .console_chat_controller import (
        AgentChatStartAuthorization,
        Any,
        CitationRepairContract,
        ConsoleChatMessage,
        ConsoleChatSession,
        ConsolePreparationOutcome,
        ConsoleProviderResolution,
        ConsoleSubmissionOrigin,
        ConsoleSubmitResult,
        ConsoleTurnExecutionContext,
        ConsoleTurnPreparation,
        MessageAttachment,
        TerminalCitationFinalizer,
        _PreparedSendContinuation,
    )
    from .console_chat_store import (
        ConsolePromotionTransactionContribution,
        ProviderContinuationCheckpoint,
        Sequence,
    )


def _controller_ordinary_native_commit_tasks(
    self, session_id: str | None = None
) -> tuple[asyncio.Task[Any], ...]:
    """Return exact save owners which must retire before storage teardown."""
    if not getattr(self, "_ordinary_native_commits", None):
        return ()
    with self._active_submit_tasks_lock:
        return tuple(
            dict.fromkeys(
                owner.task
                for owner in self._ordinary_native_commits.values()
                if session_id is None or owner.session.id == session_id
            )
        )


def _controller_ordinary_native_commit_retirements(
    self, session_id: str | None = None
) -> tuple[asyncio.Future[None], ...]:
    """Observe only the exact native lifetime, independent of provider work."""
    if not getattr(self, "_ordinary_native_commits", None):
        return ()
    with self._active_submit_tasks_lock:
        return tuple(
            owner.retired
            for owner in self._ordinary_native_commits.values()
            if session_id is None or owner.session.id == session_id
        )


def _controller_ordinary_native_commit_owner(self, preparation_id: str):
    if not getattr(self, "_ordinary_native_commits", None):
        return None
    with self._active_submit_tasks_lock:
        return self._ordinary_native_commits.get(preparation_id)


def _controller_register_ordinary_native_commit(self, session, acceptance):
    from . import console_chat_controller as _source_controller
    task = _source_controller.asyncio.current_task()
    if task is None:
        raise RuntimeError("An ordinary save requires its submit owner.")
    store = self.store
    persistence = store.persistence
    owner = _source_controller._OrdinaryNativeCommitOwner(
        store,
        persistence,
        getattr(persistence, "db", None),
        session,
        acceptance,
        task,
        _source_controller.asyncio.get_running_loop().create_future(),
    )
    with self._active_submit_tasks_lock:
        if acceptance.preparation_id in self._ordinary_native_commits:
            raise RuntimeError("An ordinary save is already owned.")
        store._retain_native_commit_owner(acceptance.preparation_id, owner)
        self._ordinary_native_commits[acceptance.preparation_id] = owner
    return owner


def _controller_release_ordinary_native_commit(self, owner):
    if owner is None:
        return
    preparation_id = owner.acceptance.preparation_id
    with self._active_submit_tasks_lock:
        if self._ordinary_native_commits.get(preparation_id) is not owner:
            return
        owner.store._release_native_commit_owner(preparation_id, owner)
        self._ordinary_native_commits.pop(preparation_id)
        if not owner.retired.done():
            owner.retired.set_result(None)


def _controller_ordinary_native_commit_current(self, owner):
    return (
        self.store is owner.store
        and owner.store.persistence is owner.persistence
        and getattr(owner.persistence, "db", None) is owner.database
        and next(
            (row for row in owner.store.sessions() if row.id == owner.session.id),
            None,
        )
        is owner.session
    )


def _controller_retain_original_native_commit_recovery(self, owner, commit=None):
    """Keep actual original acceptance without touching successor state."""
    from . import console_chat_controller as _source_controller
    preparation_id = owner.acceptance.preparation_id
    store = owner.store
    fingerprint = store.durable_acceptance_fingerprint_for(preparation_id)
    if commit is None and fingerprint is not None:
        commit = store.durable_turn_commit_for(
            preparation_id, fingerprint=fingerprint
        )
    if commit is not None and (
        store.persistence is owner.persistence
        and getattr(owner.persistence, "db", None) is owner.database
        and any(row is owner.session for row in store.sessions())
    ):
        try:
            store.publish_durable_turn_identity(owner.session.id, commit)
            store.publish_durable_recovery_owner(owner.session.id, commit)
        except Exception as error:  # noqa: BLE001 - preserve accepted durable recovery and report only the exception type.
            _source_controller.logger.warning(
                "Original saved turn retained (exception_type={})",
                type(error).__name__,
            )
    return _source_controller.ConsoleSubmitResult(
        commit is not None,
        False,
        "Saved turn retained for recovery."
        if commit is not None
        else "Prepared turn owner changed; draft kept.",
        session_id=owner.session.id,
        user_message_id=commit.user_message_id
        if commit is not None
        else owner.acceptance.user_message_id,
        assistant_message_id=commit.assistant_message_id
        if commit is not None
        else None,
        terminal_status=_source_controller.ConsoleRunStatus.BLOCKED,
        preparation_id=preparation_id,
        provider_started=False,
    )


async def _controller_settle_cancelled_native_commit(self, owner, continuation):
    """Settle an exact ACCEPTED owner without crossing provider dispatch."""
    from . import console_chat_controller as _source_controller
    from .console_native_commit import settle_accepted_durable_turn_owned
    commit = continuation.commit
    session_id = continuation.session_id
    explicit = owner.explicit_stop or self._accepted_cancellation_was_requested(
        session_id
    )
    terminal = "stopped" if explicit else "failed"
    content = (
        "Response stopped."
        if explicit
        else "Accepted turn failed before provider dispatch."
    )
    completion = await settle_accepted_durable_turn_owned(
        owner.store,
        continuation.preparation_id,
        fingerprint=continuation.fingerprint,
        terminal_state=terminal,
        content=content,
        persistence=owner.persistence,
        database=owner.database,
    )
    owner.caller_cancelled |= completion.caller_cancelled
    settled = completion.settled and completion.error is None
    if not settled:
        # A callback or owned connection cleanup can raise AFTER the real
        # terminal write. Reconcile existing original state, never replay it.
        try:
            with owner.store.durable_preparation_lock:
                if (
                    owner.store.durable_acceptance_fingerprint_for(
                        continuation.preparation_id
                    )
                    == continuation.fingerprint
                    and owner.store.durable_turn_commit_for(
                        continuation.preparation_id,
                        fingerprint=continuation.fingerprint,
                    )
                    is commit
                ):
                    assistant = owner.store.get_message(commit.assistant_message_id)
                    recovery = owner.store.dispatch_recovery_for_session(session_id)
                    settled = (
                        assistant.id == commit.assistant_message_id
                        and assistant.persisted_message_id
                        == commit.assistant_message_id
                        and assistant.parent_message_id == commit.user_message_id
                        and assistant.role is _source_controller.ConsoleMessageRole.ASSISTANT
                        and assistant.status == terminal
                        and assistant.assistant_generation_state == terminal
                        and assistant.content == content
                        and (
                            recovery is None
                            or recovery.assistant_message_id
                            != commit.assistant_message_id
                        )
                    )
        except Exception as error:  # noqa: BLE001 - preserve accepted durable recovery and report only the exception type.
            _source_controller.logger.warning(
                "Accepted turn settlement reconciliation failed (exception_type={})",
                type(error).__name__,
            )
    if not self._ordinary_native_commit_current(owner):
        if not settled:
            return self._retain_original_native_commit_recovery(owner, commit)
        # The original terminal write completed. Do not resurrect its
        # deleted ACCEPTED checkpoint or publish into a replacement store.
        return _source_controller.ConsoleSubmitResult(
            True,
            False,
            content,
            session_id=session_id,
            user_message_id=commit.user_message_id,
            assistant_message_id=commit.assistant_message_id,
            terminal_status=_source_controller.ConsoleRunStatus.STOPPED
            if explicit
            else _source_controller.ConsoleRunStatus.FAILED,
            origin=continuation.origin,
            queue_entry_id=continuation.queue_entry_id,
            committed_context_epoch=continuation.committed_context_epoch,
            preparation_id=continuation.preparation_id,
            provider_started=False,
        )
    if completion.error is not None:
        _source_controller.logger.warning(
            "Accepted turn settlement failed (exception_type={})",
            type(completion.error).__name__,
        )
    if not settled:
        owner.store.mark_dispatch_recovery_needed(
            session_id, commit.assistant_message_id
        )
        self._set_run_state(
            _source_controller.ConsoleRunState.blocked("Accepted turn is retained for recovery."),
            session_id=session_id,
        )
        return _source_controller.ConsoleSubmitResult(
            True,
            False,
            "Accepted turn is retained for recovery.",
            session_id=session_id,
            user_message_id=commit.user_message_id,
            assistant_message_id=commit.assistant_message_id,
            terminal_status=_source_controller.ConsoleRunStatus.BLOCKED,
            origin=continuation.origin,
            queue_entry_id=continuation.queue_entry_id,
            committed_context_epoch=continuation.committed_context_epoch,
            preparation_id=continuation.preparation_id,
            provider_started=False,
        )
    self._set_run_state(
        _source_controller.ConsoleRunState(
            _source_controller.ConsoleRunStatus.STOPPED if explicit else _source_controller.ConsoleRunStatus.FAILED,
            content,
        ),
        session_id=session_id,
    )
    self._release_ordinary_native_commit(owner)
    self._settle_accepted_preparation(continuation.preparation_id)
    with owner.store.durable_preparation_lock:
        current = self._durable_postcommit_continuations.get(
            continuation.preparation_id
        )
        if current is continuation:
            self._durable_postcommit_continuations.pop(continuation.preparation_id)
            self._release_retired_prepared_evidence(continuation)
            owner.store.retire_durable_acceptance(
                continuation.preparation_id, continuation.fingerprint
            )
    owner.store.release_durable_postcommit_activity(continuation.preparation_id)
    return _source_controller.ConsoleSubmitResult(
        True,
        False,
        content,
        session_id=session_id,
        user_message_id=commit.user_message_id,
        assistant_message_id=commit.assistant_message_id,
        terminal_status=_source_controller.ConsoleRunStatus.STOPPED
        if explicit
        else _source_controller.ConsoleRunStatus.FAILED,
        origin=continuation.origin,
        queue_entry_id=continuation.queue_entry_id,
        committed_context_epoch=continuation.committed_context_epoch,
        preparation_id=continuation.preparation_id,
        provider_started=False,
    )


async def _controller_accept_durable_turn(
    self,
    *,
    session: ConsoleChatSession,
    preparation: ConsoleTurnPreparation,
    preparation_outcome: ConsolePreparationOutcome | None,
    prepared_continuation: _PreparedSendContinuation | None,
    echoed_user: ConsoleChatMessage,
    staged_title: str,
    staged_attachments: tuple[MessageAttachment, ...],
    resolution: ConsoleProviderResolution,
    provider_messages: list[dict[str, Any]],
    trace_source_messages: tuple[dict[str, Any], ...],
    prefill: str | None,
    prefill_from_one_shot: bool,
    one_shot_prefill_revision: int | None,
    skill_bindings: tuple[Any, ...],
    skill_bundle_block: str,
    citation_repair_contract: CitationRepairContract | None,
    terminal_citation_finalizer: TerminalCitationFinalizer | None,
    turn_context: ConsoleTurnExecutionContext,
    origin: ConsoleSubmissionOrigin,
    queue_entry_id: str | None,
    committed_context_epoch: int,
    custody_acceptance_hook: Callable[[], None] | None,
    hook_context: str = "",
    chat_start_authorization: AgentChatStartAuthorization | None = None,
) -> ConsoleSubmitResult:
    """Commit one durable owner, then enter its idempotent effect chain."""
    from . import console_chat_controller as _source_controller
    if chat_start_authorization is not None:
        required_action = self._native_start_project_required_action(
            session, resolution, turn_context
        )
        if required_action is not None:
            chat_start_authorization.withdrawal_reason = required_action
            return _source_controller.ConsoleSubmitResult(
                False,
                False,
                "Background start needs project setup.",
                session_id=session.id,
            )
    staged_identity = self.store.staged_durable_turn_identity_for(
        preparation.preparation_id
    )
    identity = self.store.stage_durable_turn_identity(
        session.id,
        preparation.preparation_id,
        title=staged_identity.title
        if staged_identity is not None
        else staged_title,
    )
    owner_ids = self.store.stage_durable_turn_owner_ids(
        session.id,
        preparation.preparation_id,
        user_message_id=echoed_user.id,
    )
    # This used to read `echoed_user.parent_message_id`, which is the
    # PERSISTED parent id that `_persist_new_message` assigns -- and this
    # echo was appended with `persist=False`, so it was ALWAYS None. Every
    # checkpointed turn was therefore written as a fresh DB root, forking
    # the conversation away from its own history on the second send
    # (TASK-22060).
    #
    # Resolve the nearest PERSISTED ancestor from the store's own tree
    # bookkeeping instead -- the identical walk `_persist_new_message`
    # performs, so the two persistence paths agree by construction rather
    # than by coincidence. `None` is not an error here: it is the
    # documented "true persisted root" answer, which is exactly what the
    # store does with it, and `insert_with_messages` validates a parent
    # only when one is given.
    parent_message_id = self.store.durable_parent_for_message(echoed_user.id)
    contributions = (
        (preparation_outcome.contribution,)
        if preparation_outcome is not None
        and preparation_outcome.contribution is not None
        else ()
    )
    continuation_gate = self.prompt_queue_coordinator.continuation_contribution(
        session.id, queue_entry_id
    )
    if continuation_gate is not None:
        contributions = (*contributions, continuation_gate)
    acceptance = _source_controller.ConsoleDurableTurnAcceptance(
        conversation_id=identity.conversation_id,
        user_message_id=owner_ids.user_message_id,
        assistant_message_id=owner_ids.assistant_message_id,
        parent_message_id=parent_message_id,
        user_content=preparation.executed_draft,
        attachments=tuple(
            {
                "position": attachment.position,
                "data": attachment.data,
                "mime_type": attachment.mime_type,
                "display_name": attachment.display_name,
            }
            for attachment in staged_attachments
        ),
        preparation_id=preparation.preparation_id,
        attempt_id=preparation.attempt_id,
        origin=origin.value,
        queue_entry_id=queue_entry_id,
        frozen_authority=turn_context.library_authority,
        resolved_destination=turn_context.resolved_destination,
        reconstructability=_source_controller.ConsoleDispatchReconstructability(
            attachments_reconstructable=True,
            evidence_reconstructable=not hook_context
            and not bool(
                prepared_continuation is not None
                and (
                    prepared_continuation.staged_evidence_frozen
                    or prepared_continuation.staged_evidence is not None
                )
            ),
            prefill_reconstructable=(prefill is None and not prefill_from_one_shot),
            opaque_reference=f"opaque:{preparation.preparation_id}",
        ),
        contributions=contributions,
        continuation_receipt=self.prompt_queue_coordinator.continuation_receipt(
            session.id, queue_entry_id
        ),
        machine_followup_receipt=self.prompt_queue_coordinator.machine_followup_receipt(
            session.id, queue_entry_id
        ),
        agent_chat_start_attempt_id=(
            chat_start_authorization.request.attempt_id
            if chat_start_authorization
            else None
        ),
        agent_chat_start=(
            echoed_user.metadata.agent_chat_start
            if chat_start_authorization
            else None
        ),
        handoff_draft_revision=session.agent_handoff_revision
        if session.agent_handoff_state == "pending"
        else None,
    )
    from .console_send_diagnostics import record_send_stage
    native_owner = (
        self._register_ordinary_native_commit(session, acceptance)
        if chat_start_authorization is None
        else None
    )
    try:
        if chat_start_authorization is not None:
            if not await self._chat_start.accept(chat_start_authorization):
                return _source_controller.ConsoleSubmitResult(
                    False,
                    False,
                    "Background start was withdrawn.",
                    session_id=session.id,
                )
        elif not await self.store.drain_agent_handoff(session.id):
            return _source_controller.ConsoleSubmitResult(
                False, False, "Draft custody is unconfirmed.", session_id=session.id
            )
        record_send_stage("durable_commit")
        try:
            # TASK-22205: the ~10-statement BEGIN IMMEDIATE turn commit runs
            # off the event loop; the await is the dispatch-ordering barrier.
            if chat_start_authorization is not None:
                # Capture the exact store/database before the first await. Stop
                # may cancel its awaiter, but cannot detach this physical owner.
                from tldw_chatbook.DB.base_db import run_owned_db_call
                owned_store = chat_start_authorization.store
                owned_database = owned_store.persistence.db
                async def commit_owned_start():
                    if getattr(owned_database, "is_memory_db", False):
                        return owned_store.commit_durable_turn(acceptance)
                    return await run_owned_db_call(
                        owned_database, owned_store.commit_durable_turn, acceptance
                    )
                chat_start_authorization.commit_worker = _source_controller.asyncio.create_task(
                    commit_owned_start()
                )
                commit = await _source_controller.asyncio.shield(
                    chat_start_authorization.commit_worker
                )
            else:
                from .console_native_commit import commit_durable_turn_owned
                if not self._ordinary_native_commit_current(native_owner):
                    return self._retain_original_native_commit_recovery(
                        native_owner
                    )
                completion = await commit_durable_turn_owned(
                    native_owner.store, acceptance
                )
                native_owner.caller_cancelled = (
                    completion.caller_cancelled or native_owner.explicit_stop
                )
                native_owner.commit_error = completion.error
                commit = completion.commit
                if completion.error is not None:
                    fingerprint = (
                        native_owner.store.durable_acceptance_fingerprint_for(
                            preparation.preparation_id
                        )
                    )
                    recovered = (
                        native_owner.store.durable_turn_commit_for(
                            preparation.preparation_id, fingerprint=fingerprint
                        )
                        if fingerprint is not None
                        else None
                    )
                    if recovered is None:
                        raise completion.error
                    commit = recovered
                if commit is None:
                    raise RuntimeError(
                        "The ordinary save returned no committed owner."
                    )
                if not self._ordinary_native_commit_current(native_owner):
                    if custody_acceptance_hook is not None:
                        custody_acceptance_hook()
                    return self._retain_original_native_commit_recovery(
                        native_owner, commit
                    )
        except Exception as exc:  # noqa: BLE001 -- a failed commit is a retry, not a crash
            record_send_stage("durable_commit", "failed", error=exc)
            from tldw_chatbook.Agents.hooks_v2.continuations import (
                ContinuationAdmissionRefused,
            )
            if (
                native_owner is not None
                and not self._ordinary_native_commit_current(native_owner)
            ):
                return self._retain_original_native_commit_recovery(native_owner)
            if isinstance(exc, ContinuationAdmissionRefused):
                self._release_ordinary_native_commit(native_owner)
                # This exception originated inside the rolled-back transaction.
                # A generic persistence failure has no such certainty.
                cleanup_owner = self._preparation_by_id(preparation.preparation_id)
                owned_echo = bool(
                    cleanup_owner is not None
                    and cleanup_owner.session_id == session.id
                    and cleanup_owner.transient_user_message_id == echoed_user.id
                    and cleanup_owner.state is _source_controller.ConsoleTurnPreparationState.PAUSED
                )
                before_cleanup = self.store.conversation_context_epoch(session.id)
                self._abandon_preparation(preparation.preparation_id)
                try:
                    self.store.get_message(echoed_user.id)
                    echo_removed = False
                except KeyError:
                    echo_removed = True
                if (
                    owned_echo
                    and echo_removed
                    and self._preparation_by_id(preparation.preparation_id) is None
                ):
                    self.prompt_queue_coordinator.acknowledge_machine_rollback(
                        session.id,
                        queue_entry_id,
                        before_cleanup,
                        self.store.conversation_context_epoch(session.id),
                    )
                return self._block(
                    session.id, "Hook continuation is no longer current."
                )
            # TASK-22251: the user-facing copy stays deliberately generic, but
            # something must record WHICH failure occurred. `commit_durable_turn`
            # is a multi-step transaction -- conversation create, Library-policy
            # write, workspace validation, checkpoint insert -- and swallowing
            # the exception collapsed every one of them into a single sentence.
            # Two distinct causes ("Workspace registry is required for workspace
            # conversations" and "Unknown workspace: <id>") were previously
            # indistinguishable, and each needed a temporary print inside this
            # method to identify. Type only, never the message: an exception
            # string here can carry conversation or workspace identifiers.
            _source_controller.logger.warning(
                "Durable turn commit failed; turn refused (exception_type={})",
                type(exc).__name__,
            )
            if native_owner is not None:
                _pause_unreserved_failed_commit(
                    native_owner.store,
                    preparation.preparation_id,
                    preparation.attempt_id,
                )
            return _source_controller.ConsoleSubmitResult(
                False,
                False,
                "Couldn't save the prepared turn. Retry or cancel.",
                session_id=session.id,
                user_message_id=echoed_user.id,
                origin=origin,
                queue_entry_id=queue_entry_id,
                preparation_id=preparation.preparation_id,
            )
        if chat_start_authorization is not None and (
            not self._chat_start.authorizes(chat_start_authorization, session.id)
            or self._agent_bridge is not chat_start_authorization.bridge
            or self._fleet_wake.runtime_owner_id
            != chat_start_authorization.runtime_owner_id
        ):
            raise PermissionError("chat start commit owner changed")
        if acceptance.handoff_draft_revision is not None:
            self.store.publish_agent_handoff_consumed(
                session.id, acceptance.handoff_draft_revision
            )
        if chat_start_authorization is not None:
            await self._chat_start.confirm_receipt(
                chat_start_authorization, commit.checkpoint
            )
        # The durable turn now exists even if later publication needs recovery.
        # Arm once here; replaying postcommit effects must not rearm a settled turn.
        self._arm_run_hooks_stop(session.id)
        if custody_acceptance_hook is not None:
            custody_acceptance_hook()
        record_send_stage("durable_commit", "succeeded")
        fingerprint = self.store.durable_acceptance_fingerprint_for(
            preparation.preparation_id
        )
        if fingerprint is None:
            raise RuntimeError("Durable acceptance fingerprint is unavailable.")
        citation_repair_session = (
            _source_controller.ConsoleCitationRepairSession(
                contract=citation_repair_contract,
                resolution=resolution,
            )
            if citation_repair_contract is not None
            else None
        )
        continuation = _source_controller._DurablePostcommitContinuation(
            preparation_id=preparation.preparation_id,
            fingerprint=fingerprint,
            session_id=session.id,
            origin=origin,
            queue_entry_id=queue_entry_id,
            hook_continuation=acceptance.continuation_receipt is not None,
            machine_followup=acceptance.machine_followup_receipt is not None,
            clean_draft=preparation.executed_draft,
            commit=commit,
            echoed_user_id=echoed_user.id,
            resolution=resolution,
            provider_messages=provider_messages,
            trace_source_messages=trace_source_messages,
            trace_capture_mode=preparation.capture_mode,
            trace_request=None,
            prefill=prefill,
            prefill_from_one_shot=prefill_from_one_shot,
            one_shot_prefill_revision=one_shot_prefill_revision,
            skill_bindings=skill_bindings,
            skill_bundle_block=skill_bundle_block,
            citation_repair_session=citation_repair_session,
            turn_context=turn_context,
            prepared=prepared_continuation,
            committed_context_epoch=committed_context_epoch,
            stream_signals=self._admit_capture_policy(
                session.id,
                origin,
                frozen_capture_enabled=(
                    preparation.capture_mode is _source_controller.ConsoleTraceCaptureMode.CAPTURE_ON
                ),
                frozen_pii_redaction_enabled=preparation.pii_redaction_enabled,
                frozen_pii_ruleset_revision_id=(
                    preparation.pii_ruleset_revision_id
                ),
                frozen_next_trace_privacy_revision=(
                    preparation.next_trace_privacy_revision
                ),
            ),
            terminal_citation_finalizer=terminal_citation_finalizer,
            hook_context=hook_context,
        )
        with self.store.durable_preparation_lock:
            self.store.validate_durable_acceptance_fingerprint(fingerprint)
            existing = self._durable_postcommit_continuations.get(
                preparation.preparation_id
            )
            if existing is not None and existing.fingerprint != fingerprint:
                raise RuntimeError("Durable continuation owner changed.")
            self._durable_postcommit_continuations[preparation.preparation_id] = (
                continuation
            )
        try:
            trace_request = self._build_durable_trace_request(
                preparation=preparation,
                resolution=resolution,
                provider_messages=self._provider_messages_with_prefill(
                    provider_messages,
                    prefill,
                ),
                trace_source_messages=trace_source_messages,
                echoed_user_id=echoed_user.id,
                committed_user_id=commit.user_message_id,
                route=(
                    _source_controller.ConsoleRequestRoute.DIRECT_PREFILL
                    if prefill
                    else _source_controller.ConsoleRequestRoute.FRESH
                ),
            )
        except Exception as exc:
            preserve_cancelled_composer = False
            if native_owner is not None:
                if not self._ordinary_native_commit_current(native_owner):
                    return self._retain_original_native_commit_recovery(
                        native_owner, commit
                    )
                preserve_cancelled_composer = (
                    native_owner.caller_cancelled
                    or native_owner.commit_error is not None
                )
                # Physical save and its retained continuation are complete;
                # the existing trace handler must be able to retire them.
                self._release_ordinary_native_commit(native_owner)
            result = self._handle_durable_trace_provenance_failure(
                continuation, exc
            )
            return (
                _source_controller.replace(result, should_clear_draft=False)
                if preserve_cancelled_composer
                else result
            )
        if trace_request is not None:
            with self.store.durable_preparation_lock:
                current = self._durable_postcommit_continuations.get(
                    preparation.preparation_id
                )
                if current is not continuation:
                    raise RuntimeError("Durable continuation owner changed.")
                continuation = _source_controller.replace(continuation, trace_request=trace_request)
                self._durable_postcommit_continuations[
                    preparation.preparation_id
                ] = continuation
        if native_owner is not None and native_owner.commit_error is not None:
            return await self.resume_durable_postcommit(
                preparation.preparation_id, continue_to_provider=False
            )
        return await self.resume_durable_postcommit(preparation.preparation_id)
    finally:
        self._release_ordinary_native_commit(native_owner)


def _pause_unreserved_failed_commit(
    store: ConsoleChatStore, preparation_id: str, attempt_id: str
) -> bool:
    """Pause a send whose durable commit failed before the store reserved it.

    TASK-33621.53. The store pauses a failed commit as PERSISTENCE only once
    it holds the commit's reservation. A failure before that -- the store's
    preflight ("Durable identity owner changed." and its siblings) or the
    caller's "no committed owner" check -- left the preparation COMMITTING,
    which neither the card nor the shelf can end, so the conversation
    refused every later send. Take the same pause here, and only while
    nothing else owns the commit: no reservation in flight, no commit
    recorded, and the exact attempt still COMMITTING.

    Args:
        store: The store that ran the failed commit.
        preparation_id: The failed send's preparation.
        attempt_id: The attempt that failed; a newer attempt is left alone.

    Returns:
        True when the preparation is now PAUSED(PERSISTENCE).
    """
    from .console_turn_preparation import (
        ConsolePreparationPauseKind,
        ConsolePreparationTransition,
        ConsoleTurnPreparationState,
    )

    with store.durable_preparation_lock:
        if (
            preparation_id in store._durable_commit_in_flight
            or preparation_id in store._durable_commit_by_preparation
        ):
            return False
        current = store.preparation_by_id(preparation_id)
        if current is None or current.attempt_id != attempt_id:
            return False
        return (
            store.compare_and_set_preparation(
                current.session_id,
                ConsolePreparationTransition(
                    preparation_id=preparation_id,
                    expected_state=ConsoleTurnPreparationState.COMMITTING,
                    new_state=ConsoleTurnPreparationState.PAUSED,
                    pause_kind=ConsolePreparationPauseKind.PERSISTENCE,
                    new_attempt_id=None,
                ),
            )
            is not None
        )


def _store_retain_native_commit_owner(self, preparation_id: str, owner: object) -> None:
    """Protect queued cleanup with the controller's exact custody token."""
    if owner is None:
        raise TypeError("native commit owner must not be None")
    with self._preparation_lock:
        preparation = self._preparations_by_id.get(preparation_id)
        if preparation is None or preparation.session_id not in self._sessions:
            raise RuntimeError("Durable preparation is unavailable.")
        current = self._native_commit_owners_by_preparation.get(preparation_id)
        if current is owner:
            return
        if current is not None:
            raise RuntimeError("Durable native commit owner changed.")
        self._native_commit_owners_by_preparation[preparation_id] = owner


def _store_release_native_commit_owner(self, preparation_id: str, owner: object) -> bool:
    """Release cleanup protection only for this same retired operation."""
    with self._preparation_lock:
        if (
            owner is None
            or self._native_commit_owners_by_preparation.get(preparation_id)
            is not owner
        ):
            return False
        self._native_commit_owners_by_preparation.pop(preparation_id)
        return True


def _store_settle_accepted_durable_turn(
    self,
    preparation_id: str,
    *,
    fingerprint: ConsoleDurableAcceptanceFingerprint,
    terminal_state: str,
    content: str,
    metadata_json: str | None = None,
) -> bool:
    """Settle one live durable acceptance cancelled before provider entry.

        Args:
            preparation_id: Exact preparation retaining this committed turn.
            fingerprint: Original app-lifetime acceptance owner.
            terminal_state: Only stopped or failed; dispatch is never invented.
            content: Assistant terminal content.
            metadata_json: Optional terminal metadata object.

        Returns:
            True only after the existing atomic assistant/checkpoint settlement.
            A refused or failed claim preserves its recovery owner.
        """
    from . import console_chat_store as _source_store
    if not isinstance(fingerprint, _source_store.ConsoleDurableAcceptanceFingerprint):
        raise TypeError("fingerprint must be ConsoleDurableAcceptanceFingerprint")
    if terminal_state not in {"stopped", "failed"}:
        raise ValueError("Accepted turns may only settle as stopped or failed.")
    if preparation_id != fingerprint.preparation_id:
        return False
    from .console_native_commit import _ConsoleNativeBinding, _native_commit_binding
    binding = _native_commit_binding.get()
    if binding is not None and binding.store is self:
        if binding.turn_owner != fingerprint:
            return False
    else:
        persistence = self.persistence
        binding = _ConsoleNativeBinding(
            self, fingerprint, persistence, getattr(persistence, "db", None)
        )
    session_id = fingerprint.session_id
    assistant_id = fingerprint.assistant_message_id
    with self._generation_owner_scope(assistant_id):
        with self._preparation_lock:
            if (
                self.persistence is not binding.persistence
                or getattr(binding.persistence, "db", None) is not binding.database
            ):
                return False
            commit = self._durable_commit_by_preparation.get(preparation_id)
            preparation = self._preparations_by_id.get(preparation_id)
            session = self._sessions.get(session_id)
            current = self._dispatch_recoveries_by_session.get(session_id)
            if (
                self._durable_fingerprint_by_preparation.get(preparation_id)
                != fingerprint
                or commit is None
                or preparation is None
                or session is None
                or self._preparations_by_session.get(session_id) is not preparation
                or preparation.session_id != session_id
                or preparation.state is not _source_store.ConsoleTurnPreparationState.ACCEPTED
                or preparation.attempt_id != fingerprint.attempt_id
                or preparation.origin != fingerprint.origin
                or preparation.queue_entry_id != fingerprint.queue_entry_id
                or commit.identity.conversation_id != fingerprint.conversation_id
                or session.persisted_conversation_id != fingerprint.conversation_id
                or commit.user_message_id != fingerprint.user_message_id
                or commit.assistant_message_id != assistant_id
                or current is None
                or current.kind is not _source_store.ConsoleDispatchRecoveryKind.ACCEPTED
                or current.in_flight
                or not current.runtime_active
                or current.recovery_needed
                or current.preparation_id != preparation_id
                or current.conversation_id != fingerprint.conversation_id
                or current.assistant_message_id != assistant_id
                or current.checkpoint != commit.checkpoint
                or commit.checkpoint.state
                is not _source_store.ConsoleDispatchCheckpointState.ACCEPTED
                or commit.checkpoint.user_message_version
                != commit.user_message_version
                or commit.checkpoint.assistant_message_version
                != commit.assistant_message_version
                or assistant_id in self._generation_attempt_tokens
                or session_id in self._dispatch_recovery_generation_tokens
            ):
                return False
            nodes = self._nodes_by_session.get(session_id, {})
            user = nodes.get(fingerprint.user_message_id)
            assistant = nodes.get(assistant_id)
            if (
                user is None
                or assistant is None
                or user.role is not _source_store.ConsoleMessageRole.USER
                or assistant.role is not _source_store.ConsoleMessageRole.ASSISTANT
                or user.persisted_message_id != fingerprint.user_message_id
                or assistant.persisted_message_id != assistant_id
                or self._message_session_index.get(user.id) != session_id
                or self._message_session_index.get(assistant.id) != session_id
                or assistant.content
            ):
                return False
            self._dispatch_recovery_message_baselines[session_id] = self._snapshot(
                assistant
            )
            claimed = current.with_in_flight(True)
            self._dispatch_recoveries_by_session[session_id] = claimed
        settled = False
        try:
            settled = self._settle_dispatch_recovery(
                session_id,
                assistant_message_id=assistant_id,
                terminal_state=terminal_state,
                content=content,
                metadata_json=metadata_json,
                _native_binding=binding,
            )
            return settled
        finally:
            if not settled:
                # Release only our exact claim, never a replacement owner.
                with self._preparation_lock:
                    if (
                        self._dispatch_recoveries_by_session.get(session_id)
                        is claimed
                    ):
                        self._release_dispatch_recovery_action(
                            session_id, assistant_id
                        )
                        restored = self._dispatch_recoveries_by_session[session_id]
                        self._dispatch_recoveries_by_session[session_id] = (
                            restored.with_runtime_truth(
                                runtime_active=False, recovery_needed=True
                            )
                        )
                        if (
                            restored.checkpoint is not None
                            and restored.checkpoint.origin == "queued"
                            and restored.checkpoint.queue_entry_id is not None
                        ):
                            self._dispatch_recovery_queue_hydration_pending.add(
                                session_id
                            )


def _store_settle_dispatch_recovery(
    self,
    session_id: str,
    *,
    assistant_message_id: str,
    terminal_state: str,
    content: str,
    metadata_json: str | None = None,
    provider_continuation_json: str | None = None,
    provider_continuation: ProviderContinuationCheckpoint | None = None,
    contributions: Sequence[ConsolePromotionTransactionContribution] = (),
    on_durable_commit: Callable[[], object] | None = None,
    _native_binding: _ConsoleNativeBinding | None = None,
) -> bool:
    """Settle dispatch state while holding its generation owner."""
    from . import console_chat_store as _source_store
    with self._preparation_lock:
        current = self._dispatch_recoveries_by_session.get(session_id)
        if (
            current is None
            or not current.in_flight
            or current.assistant_message_id != assistant_message_id
            or current.checkpoint is None
        ):
            return False
        checkpoint = current.checkpoint
        ephemeral = current.kind in {
            _source_store.ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED,
            _source_store.ConsoleDispatchRecoveryKind.EPHEMERAL_DISPATCH_STARTED,
        }
    if terminal_state == "discarded":
        self._invalidate_generation_attempt_locked(assistant_message_id)
    try:
        message = self._message_or_raise(assistant_message_id)
    except KeyError:
        message = None
    receipt_metadata_before: (
        tuple[_source_store.MessageMetadata | None, _source_store.VideoGenerationMetadata | None] | None
    ) = None
    terminal_receipt_id: str | None = None
    if (
        not ephemeral
        and terminal_state in {"complete", "failed"}
        and message is None
    ):
        return False
    if (
        not ephemeral
        and terminal_state in {"complete", "failed"}
        and message is not None
    ):
        merged_metadata = (
            _source_store.json.loads(metadata_json) if metadata_json is not None else None
        )
        if merged_metadata is not None and type(merged_metadata) is not dict:
            raise _source_store.ConsoleDispatchSettlementError(
                "Terminal metadata must be an object."
            )
        receipt_metadata_before = self._terminal_metadata_snapshot(message)
        terminal_receipt_id = self._ensure_terminal_receipt(message.id)
        receipt_metadata_json = (
            message.video_metadata.to_json()
            if message.video_metadata is not None
            else message.metadata.to_json()
            if message.metadata is not None
            else None
        )
        # The caller may have joined Canvas cards or other local sidecars
        # that have not yet been published on the live message.
        if merged_metadata is None:
            merged_metadata = _source_store.json.loads(receipt_metadata_json or "{}")
        merged_metadata["terminal_receipt_id"] = terminal_receipt_id
        metadata_json = _source_store.json.dumps(
            merged_metadata, ensure_ascii=False, allow_nan=False
        )
    elif message is not None:
        self._pending_terminal_receipts.pop(message.id, None)
    committed_message_version: int | None = None
    if not ephemeral:
        persistence = _native_binding.persistence if _native_binding is not None else self.persistence
        repository = getattr(
            persistence,
            "console_dispatch_repository",
            None,
        )
        if repository is None:
            if message is not None and receipt_metadata_before is not None:
                self._restore_terminal_metadata(
                    message,
                    receipt_metadata_before,
                )
            return False
        try:
            if _native_binding is not None and (
                _native_binding.store is not self
                or self.persistence is not _native_binding.persistence
                or getattr(_native_binding.persistence, "db", None) is not _native_binding.database
            ):
                raise RuntimeError("Durable Console persistence owner changed.")
            result = repository.settle_with_assistant(
                _source_store.ConsoleAssistantSettlement(
                    assistant_message_id=assistant_message_id,
                    expected_checkpoint_state=checkpoint.state,
                    expected_checkpoint_revision=checkpoint.checkpoint_revision,
                    expected_user_message_version=checkpoint.user_message_version,
                    expected_assistant_message_version=(
                        checkpoint.assistant_message_version
                    ),
                    terminal_state=terminal_state,
                    content=content,
                    metadata_json=metadata_json,
                    terminal_receipt_id=terminal_receipt_id,
                    usage_json=(
                        message.usage.to_json()
                        if message is not None and message.usage is not None
                        else None
                    ),
                    provider_continuation_json=provider_continuation_json,
                    thinking_blocks_json=(
                        _source_store.dump_thinking_blocks_json(message.thinking)
                        if message is not None
                        else None
                    ),
                    contributions=tuple(contributions),
                )
            )
        except Exception:
            # TASK-32801.3: this is the terminal-settlement write. The
            # rollback below is correct and the caller's retry queue picks
            # the turn up again, but the failure itself left no trace
            # anywhere -- so a turn that never settles is indistinguishable
            # from one that settled cleanly, in the log and to the user.
            _source_store.logger.warning(
                "Console terminal settlement failed; rolled back and "
                "returning False (session_id={}, assistant_message_id={}, "
                "terminal_state={})",
                session_id,
                assistant_message_id,
                terminal_state,
            )
            if message is not None and receipt_metadata_before is not None:
                self._restore_terminal_metadata(
                    message,
                    receipt_metadata_before,
                )
            return False
        if result.status is not _source_store.ConsoleDispatchResultStatus.COMMITTED:
            if message is not None and receipt_metadata_before is not None:
                self._restore_terminal_metadata(
                    message,
                    receipt_metadata_before,
                )
            return False
        committed_message_version = result.committed_message_version
        if message is not None and terminal_receipt_id is not None:
            self._commit_terminal_receipt(message.id)
        if on_durable_commit is not None:
            on_durable_commit()
    if message is not None:
        message.content = content
        message.status = terminal_state
        message.assistant_generation_state = terminal_state
        if metadata_json is not None and message.video_metadata is None:
            restored_metadata = _source_store.MessageMetadata.from_json(metadata_json)
            if restored_metadata is not None:
                message.metadata = restored_metadata
        message.provider_continuation = provider_continuation
        message.provider_continuation_message_version = committed_message_version
        message.provider_continuation_actions_enabled = False
        self._stream_chunks_by_message.pop(message.id, None)
        self._stream_materialized_counts.pop(message.id, None)
        self._bump_payload_revision(session_id)
    with self._preparation_lock:
        if self._dispatch_recoveries_by_session.get(session_id) is not current:
            raise RuntimeError("Dispatch recovery owner changed during settlement.")
        self._dispatch_recoveries_by_session.pop(session_id, None)
        self._dispatch_recovery_message_baselines.pop(session_id, None)
        self._dispatch_recovery_generation_tokens.pop(session_id, None)
        self._dispatch_recovery_queue_hydration_pending.discard(session_id)
    return True
