"""One runtime-owned, bounded start for an already durable agent-created chat."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from tldw_chatbook.DB.base_db import operation_owned_connection, run_owned_db_call

from .console_chat_models import CONSOLE_GLOBAL_WORKSPACE_ID
from .console_turn_context import ConsoleTurnConfigurationSnapshot
from .message_metadata import (
    AgentHandoffLaunchStatus,
    HANDOFF_LAUNCH_NOT_STARTED,
    HANDOFF_LAUNCH_STARTED,
    HANDOFF_LAUNCH_REVIEW_REQUIRED,
)


@dataclass(frozen=True, slots=True)
class AgentChatStartRequest:
    attempt_id: str
    source_run_id: str
    source_session_id: str
    source_session_incarnation: str
    conversation_id: str
    session_id: str
    session_incarnation: str
    draft_revision: int
    context_epoch: int
    opening_prompt: str = field(repr=False)
    configuration: ConsoleTurnConfigurationSnapshot = field(repr=False)
    workspace_id: str


@dataclass(frozen=True, slots=True)
class AgentChatStartOutcome:
    launch_status: AgentHandoffLaunchStatus
    reason: str | None = None


_AUTHORIZATION_KEY = object()


class AgentChatStartAuthorization:
    """Exact live coordinator authority; persisted provenance never creates it."""

    def __init__(
        self, owner: Any, request: AgentChatStartRequest, context: Any, *, key: object
    ) -> None:
        if key is not _AUTHORIZATION_KEY:
            raise PermissionError("coordinator authorization required")
        self.owner = owner
        self.request = request
        self.context = context
        self.bridge = owner._controller._agent_bridge
        self.store = owner._controller.store
        self.database = self.bridge.runs_db
        self.ledger = self.database.automatic_work
        self.capacity_owner = owner._controller._fleet_wake
        self.runtime_owner_id = self.capacity_owner.runtime_owner_id
        self.token = object()
        self.accepted = False
        self.receipted = False
        self.withdrawn = False
        self.outcome = asyncio.get_running_loop().create_future()
        self.task: asyncio.Task | None = None
        self.preparation_id: str | None = None
        self.validating_state: Any = None
        self.running = False
        self.provider_worker: asyncio.Task | None = None
        self.commit_worker: asyncio.Task | None = None
        self.withdrawal_reason: str | None = None


class ConsoleChatStartCoordinator:
    """Own starts until cleanup without adding a scheduler or retry queue."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller
        self._active: dict[str, AgentChatStartAuthorization] = {}
        self._tasks: set[asyncio.Task] = set()
        self._disposed = False

    def tasks(self) -> tuple[asyncio.Task, ...]:
        """Return currently owned work for runtime drain and verification."""
        return tuple(self._tasks)

    async def withdraw_for_manual(self, session_id: str) -> None:
        """Drain only the exact prepared target before ordinary manual admission."""
        item = self._active.get(session_id)
        if item is None or item.accepted:
            return
        self.withdraw_prepared(session_id, "manual_send")
        if item.task is not None:
            await asyncio.gather(item.task, return_exceptions=True)

    def is_accepted(self, session_id: str) -> bool:
        """Return whether this runtime still owns an accepted target start."""
        item = self._active.get(session_id)
        return bool(item and item.accepted and self.authorizes(item, session_id))

    def is_prepared(self, session_id: str) -> bool:
        item = self._active.get(session_id)
        return bool(item and not item.accepted and not item.withdrawn)

    def authorizes(
        self, authorization: AgentChatStartAuthorization | None, session_id: str
    ) -> bool:
        """Return whether this authorization still belongs to the target session.

        Args:
            authorization: Candidate start authorization, or None.
            session_id: Target session ID whose active authorization is checked.

        Returns:
            True when this coordinator is live, the authorization is unwithdrawn
            and owned by it and its current store, the active entry is this exact
            object, and the target exists with the requested session incarnation.
            This check establishes neither acceptance nor current primary visibility.
        """
        if not isinstance(authorization, AgentChatStartAuthorization):
            return False
        target = self._controller.store._sessions.get(session_id)
        return bool(
            not self._disposed
            and not authorization.withdrawn
            and authorization.owner is self
            and self._controller.store is authorization.store
            and self._active.get(session_id) is authorization
            and target is not None
            and target.incarnation_id == authorization.request.session_incarnation
        )

    def _source_live(self, request: AgentChatStartRequest) -> bool:
        controller = self._controller
        return controller._chat_creation_source_live(
            {
                "session_id": request.source_session_id,
                "source_incarnation": request.source_session_incarnation,
                "source_run_id": request.source_run_id,
                "source_message_id": controller._active_assistant_message_ids.get(
                    request.source_session_id
                ),
            }
        )

    def _target_unchanged(self, request: AgentChatStartRequest) -> bool:
        controller = self._controller
        store = controller.store
        target = store._sessions.get(request.session_id)
        return bool(
            target
            and target.incarnation_id == request.session_incarnation
            and target.workspace_id == request.workspace_id
            and target.persisted_conversation_id == request.conversation_id
            and target.agent_handoff_state == "pending"
            and target.agent_handoff_revision == request.draft_revision
            and target.draft == request.opening_prompt
            and target.settings == request.configuration.session_settings
            and store.conversation_context_epoch(target.id) == request.context_epoch
            and not store.pending_attachments(target.id)
            and not store.session_one_shot_prefill(target.id)
            and controller._has_explicit_staged_evidence(target.id) is False
            and not controller.prompt_queue_coordinator.controls_generation(target.id)
        )

    @staticmethod
    async def _drain_owned(task: asyncio.Task) -> Any:
        """A cancelled awaiter must not detach physical work from its slot."""
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    async def run_provider_worker(self, session_id: str, operation: Any) -> Any:
        """Await an operation and retain accepted provider-task custody.

        Args:
            session_id: Session whose active start may own the worker.
            operation: Provider coroutine to await.

        Returns:
            The operation's result. Without an accepted active item, the coroutine
            is awaited directly. An accepted item stores a task and shields it
            from awaiter cancellation; operation errors still propagate.
        """
        item = self._active.get(session_id)
        if item is None or not item.accepted:
            return await operation
        item.provider_worker = asyncio.create_task(operation)
        return await asyncio.shield(item.provider_worker)

    async def _publish_outcome(
        self,
        request: AgentChatStartRequest,
        outcome: AgentChatStartOutcome,
        *,
        store: Any = None,
    ) -> AgentChatStartOutcome:
        from .message_metadata import AgentHandoffLaunchMetadata

        launch = AgentHandoffLaunchMetadata(
            "start", outcome.launch_status, outcome.reason
        )
        store = store if store is not None else self._controller.store
        publication = asyncio.create_task(
            run_owned_db_call(
                store.persistence.db,
                store.persistence.update_agent_handoff_launch,
                request.conversation_id,
                launch,
            )
        )
        try:
            await asyncio.shield(publication)
        except asyncio.CancelledError:
            await self._drain_owned(publication)
            raise
        except Exception as exc:
            logger.warning(
                "Chat start failed (phase=outcome, exception_type={})",
                type(exc).__name__,
            )
            outcome = AgentChatStartOutcome(
                HANDOFF_LAUNCH_REVIEW_REQUIRED, "outcome_unconfirmed"
            )
            launch = AgentHandoffLaunchMetadata(
                "start", outcome.launch_status, outcome.reason
            )
        target = store._sessions.get(request.session_id)
        if (
            self._controller.store is store
            and target is not None
            and target.incarnation_id == request.session_incarnation
        ):
            target.agent_handoff_launch = launch
        return outcome

    async def _resolve(
        self,
        item: AgentChatStartAuthorization,
        status: AgentHandoffLaunchStatus,
        reason: str | None = None,
    ) -> None:
        if not item.outcome.done():
            outcome = await self._publish_outcome(
                item.request, AgentChatStartOutcome(status, reason), store=item.store
            )
            item.outcome.set_result(outcome)

    def _runtime_refusal(self, request: AgentChatStartRequest) -> str | None:
        """Recheck current native support and enablement at each admission fence."""
        from tldw_chatbook.Agents.agent_service import _coerce_autowake_enabled
        from tldw_chatbook.config import get_cli_setting

        controller = self._controller
        if getattr(controller, "_maintenance_paused", False):
            return "maintenance"
        if (
            self._disposed
            or controller._disposed
            or not controller._agent_runtime_enabled
            or not _coerce_autowake_enabled(
                get_cli_setting("agents", "autowake_enabled", True)
            )
        ):
            return "runtime_disabled"
        target = controller.store._sessions.get(request.session_id)
        if (
            not controller._agent_dispatch_is_eligible(target, prefill=None)
            or target.runtime_backend != "local"
        ):
            return "runtime_unavailable"
        return None

    def _destination_available(self, request: AgentChatStartRequest) -> bool:
        global_scope = request.workspace_id in (None, CONSOLE_GLOBAL_WORKSPACE_ID)
        return self._controller._chat_creation_destination_available(
            scope_type="global" if global_scope else "workspace",
            workspace_id=None if global_scope else request.workspace_id,
        )

    async def start(self, request: AgentChatStartRequest) -> AgentChatStartOutcome:
        """Return at refusal or both acceptance fences while owning the target task."""
        controller = self._controller
        reason = self._runtime_refusal(request)
        if reason is None and not self._destination_available(request):
            reason = "destination_unavailable"
        if reason is not None:
            return await self._publish_outcome(
                request, AgentChatStartOutcome(HANDOFF_LAUNCH_NOT_STARTED, reason)
            )
        if (
            request.configuration.session_settings is not None
            and not request.configuration.session_settings.provider.strip()
        ):
            return await self._publish_outcome(
                request,
                AgentChatStartOutcome(
                    HANDOFF_LAUNCH_NOT_STARTED, "provider_unconfigured"
                ),
            )
        if not self._source_live(request):
            return await self._publish_outcome(
                request,
                AgentChatStartOutcome(HANDOFF_LAUNCH_NOT_STARTED, "source_unavailable"),
            )
        if not self._target_unchanged(request) or request.session_id in self._active:
            return await self._publish_outcome(
                request,
                AgentChatStartOutcome(HANDOFF_LAUNCH_NOT_STARTED, "target_changed"),
            )
        ledger = getattr(controller._agent_bridge.runs_db, "automatic_work", None)
        if ledger is None:
            return await self._publish_outcome(
                request,
                AgentChatStartOutcome(
                    HANDOFF_LAUNCH_NOT_STARTED, "lineage_unavailable"
                ),
            )
        item = AgentChatStartAuthorization(self, request, None, key=_AUTHORIZATION_KEY)
        owner = controller._fleet_wake
        if not owner.try_claim_automatic_primary(request.session_id, item.token):
            return await self._publish_outcome(
                request, AgentChatStartOutcome(HANDOFF_LAUNCH_NOT_STARTED, "capacity")
            )
        self._active[request.session_id] = item
        # Initial ledger work belongs to runtime shutdown before its first await.
        item.task = asyncio.create_task(self._run(item))
        self._tasks.add(item.task)
        item.task.add_done_callback(self._tasks.discard)
        try:
            return await asyncio.shield(item.outcome)
        except asyncio.CancelledError:
            self.withdraw_prepared(request.session_id, "caller_cancelled")
            raise

    async def _prepare(self, item: AgentChatStartAuthorization) -> bool:
        """Drain initial preparation before any withdrawal can settle its reservation."""
        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits
        from tldw_chatbook.Agents.automatic_work_runtime import AutomaticWorkContext

        request = item.request
        owner = item.capacity_owner
        ledger = item.ledger
        preparation = asyncio.create_task(
            run_owned_db_call(
                item.database,
                ledger.prepare_chat_start,
                attempt_id=request.attempt_id,
                source_run_id=request.source_run_id,
                target_conversation_id=request.conversation_id,
                target_session_id=request.session_id,
                target_session_incarnation=request.session_incarnation,
                owner_id=item.runtime_owner_id,
                draft_revision=request.draft_revision,
                context_epoch=request.context_epoch,
                request_fingerprint=hashlib.sha256(
                    request.opening_prompt.encode()
                ).hexdigest(),
                limits=AutomaticWorkLimits.from_settings(),
            )
        )
        try:
            attempt = await asyncio.shield(preparation)
        except asyncio.CancelledError:
            item.withdrawn = True
            attempt = await self._drain_owned(preparation)
        item.context = AutomaticWorkContext(
            ledger,
            attempt.chain_id,
            item.runtime_owner_id,
            request.attempt_id,
            attempt_kind="chat_start",
        )
        return self.authorizes(item, request.session_id) and self._source_live(request)

    @staticmethod
    def _restrict_cleanup(item: AgentChatStartAuthorization) -> None:
        """Retain uncertain admission without letting registration skip custody."""
        try:
            item.ledger._restrict_chat_start(
                item.request.attempt_id,
                owner_id=item.runtime_owner_id,
                chain_id=item.context.chain_id if item.context is not None else None,
            )
        except BaseException as exc:
            logger.warning(
                "Chat start failed (phase=restriction, exception_type={})",
                type(exc).__name__,
            )
            try:
                # Ordinary registration may fail repeatedly. Preserve denial
                # with the same pre-established key, without filesystem or SQL.
                item.ledger._retain_chat_start_restriction(
                    item.request.attempt_id,
                    owner_id=item.runtime_owner_id,
                    chain_id=item.context.chain_id
                    if item.context is not None
                    else None,
                )
            except BaseException as retain_error:
                logger.warning(
                    "Chat start failed (phase=restriction_fallback, exception_type={})",
                    type(retain_error).__name__,
                )

    async def _run(self, item: AgentChatStartAuthorization) -> None:
        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
        from .console_chat_models import (
            ConsoleSubmissionOrigin,
            ConsoleRunState,
            ConsoleRunStatus,
        )

        controller, request = self._controller, item.request
        item.running = True
        status, reason = HANDOFF_LAUNCH_NOT_STARTED, "preflight_refused"
        controller._agent_wake_turn_sessions.add(request.session_id)
        preparation_started = False
        preparation_refused = False
        try:
            if item.withdrawn:
                return
            preparation_started = True
            if not await self._prepare(item):
                reason = "source_unavailable"
                return
            with item.context.scope():
                await controller.submit_draft(
                    request.opening_prompt,
                    session_id=request.session_id,
                    origin=ConsoleSubmissionOrigin.AGENT_CHAT_START,
                    chat_start_authorization=item,
                    configuration=request.configuration,
                    accepted_attachments=(),
                )
            if item.accepted and not item.receipted:
                status, reason = HANDOFF_LAUNCH_REVIEW_REQUIRED, "receipt_unconfirmed"
        except AutomaticWorkRefused:
            preparation_refused = item.context is None
            reason = "preparation_refused"
        except asyncio.CancelledError:
            if item.accepted:
                status, reason = HANDOFF_LAUNCH_REVIEW_REQUIRED, "interrupted"
            elif item.context is None:
                status, reason = (
                    HANDOFF_LAUNCH_REVIEW_REQUIRED,
                    "preparation_unconfirmed",
                )
        except Exception as exc:
            logger.warning(
                "Chat start failed (phase={}, exception_type={})",
                "accepted" if item.accepted else "preaccept",
                type(exc).__name__,
            )
            if item.accepted:
                status, reason = HANDOFF_LAUNCH_REVIEW_REQUIRED, "interrupted"
            elif item.context is None:
                status, reason = (
                    HANDOFF_LAUNCH_REVIEW_REQUIRED,
                    "preparation_unconfirmed",
                )
            else:
                reason = "start_failed"
        except BaseException:
            status, reason = HANDOFF_LAUNCH_REVIEW_REQUIRED, "interrupted"
        finally:
            ledger = item.ledger
            settlement_needed = preparation_started and not preparation_refused
            if settlement_needed and not item.receipted:
                # Deny siblings before draining any accepted uncertainty; durable
                # settlement itself still waits for every physical owner.
                self._restrict_cleanup(item)
            # The physical conversation transaction remains a distinct owner.
            if item.commit_worker is not None:
                try:
                    await self._drain_owned(item.commit_worker)
                except BaseException as exc:
                    logger.warning(
                        "Chat start failed (phase=commit_drain, exception_type={})",
                        type(exc).__name__,
                    )
                    status, reason = (
                        HANDOFF_LAUNCH_REVIEW_REQUIRED,
                        "receipt_unconfirmed",
                    )
            if item.provider_worker is not None:
                try:
                    await self._drain_owned(item.provider_worker)
                except BaseException as exc:
                    logger.warning(
                        "Chat start failed (phase=provider_drain, exception_type={})",
                        type(exc).__name__,
                    )
                    status, reason = (
                        HANDOFF_LAUNCH_REVIEW_REQUIRED,
                        "worker_unconfirmed",
                    )
            try:
                if not settlement_needed:
                    pass
                elif item.receipted:
                    settled = await self._drain_owned(
                        asyncio.create_task(
                            run_owned_db_call(
                                item.database,
                                ledger.complete_chat_start,
                                request.attempt_id,
                                owner_id=item.runtime_owner_id,
                            )
                        )
                    )
                    if not settled:
                        raise RuntimeError("settlement refused")
                    ledger._clear_chat_start_restriction(
                        request.attempt_id, owner_id=item.runtime_owner_id
                    )
                elif item.accepted:
                    settled = await self._drain_owned(
                        asyncio.create_task(
                            run_owned_db_call(
                                item.database,
                                ledger.mark_chat_start_review_required,
                                request.attempt_id,
                                owner_id=item.runtime_owner_id,
                            )
                        )
                    )
                    status, reason = (
                        HANDOFF_LAUNCH_REVIEW_REQUIRED,
                        "receipt_unconfirmed",
                    )
                    if not settled:
                        raise RuntimeError("settlement refused")
                else:
                    # A Python exception/false return is not evidence of rollback.
                    # The transaction alone can prove a prepared refund.
                    refunded = await self._drain_owned(
                        asyncio.create_task(
                            run_owned_db_call(
                                item.database,
                                ledger.abort_chat_start,
                                request.attempt_id,
                                owner_id=item.runtime_owner_id,
                            )
                        )
                    )
                    if refunded:
                        ledger._clear_chat_start_restriction(
                            request.attempt_id, owner_id=item.runtime_owner_id
                        )
                    else:
                        settled = await self._drain_owned(
                            asyncio.create_task(
                                run_owned_db_call(
                                    item.database,
                                    ledger.mark_chat_start_review_required,
                                    request.attempt_id,
                                    owner_id=item.runtime_owner_id,
                                )
                            )
                        )
                        status, reason = (
                            HANDOFF_LAUNCH_REVIEW_REQUIRED,
                            "receipt_unconfirmed",
                        )
                        if not settled:
                            raise RuntimeError("settlement refused")
            except BaseException as exc:
                logger.warning(
                    "Chat start failed (phase=settlement, exception_type={})",
                    type(exc).__name__,
                )
                status, reason = (
                    HANDOFF_LAUNCH_REVIEW_REQUIRED,
                    "settlement_unconfirmed",
                )
                absent = False
                if not item.accepted and not item.receipted and item.context is None:
                    try:
                        absent = await self._drain_owned(
                            asyncio.create_task(
                                run_owned_db_call(
                                    item.database,
                                    ledger._confirm_chat_start_absent,
                                    request.attempt_id,
                                    owner_id=item.runtime_owner_id,
                                )
                            )
                        )
                        if absent:
                            ledger._clear_chat_start_restriction(
                                request.attempt_id, owner_id=item.runtime_owner_id
                            )
                    except BaseException as absence_error:
                        absent = False
                        logger.warning(
                            "Chat start failed (phase=absence, exception_type={})",
                            type(absence_error).__name__,
                        )
                if absent:
                    status, reason = HANDOFF_LAUNCH_NOT_STARTED, "start_failed"
                else:
                    self._restrict_cleanup(item)
            try:
                await self._drain_owned(
                    asyncio.create_task(
                        self._resolve(
                            item,
                            status,
                            item.withdrawal_reason
                            if status == HANDOFF_LAUNCH_NOT_STARTED
                            and item.withdrawal_reason
                            else reason,
                        )
                    )
                )
            except BaseException as exc:
                logger.warning(
                    "Chat start failed (phase=outcome_drain, exception_type={})",
                    type(exc).__name__,
                )
                if not item.outcome.done():
                    item.outcome.set_result(
                        AgentChatStartOutcome(
                            HANDOFF_LAUNCH_REVIEW_REQUIRED, "outcome_unconfirmed"
                        )
                    )
            finally:
                try:
                    # A manual winner may already own a different preparation.
                    preparation = controller.store.preparation_for_session(
                        request.session_id
                    )
                    if (
                        not item.accepted
                        and controller.store is item.store
                        and preparation is not None
                        and preparation.preparation_id == item.preparation_id
                    ):
                        controller._abandon_preparation(preparation.preparation_id)
                except BaseException as exc:
                    logger.warning(
                        "Chat start failed (phase=abandonment, exception_type={})",
                        type(exc).__name__,
                    )
                finally:
                    try:
                        if (
                            not item.accepted
                            and controller.store is item.store
                            and item.validating_state is not None
                            and controller.run_state_for(request.session_id)
                            is item.validating_state
                        ):
                            controller._set_run_state(
                                ConsoleRunState(ConsoleRunStatus.IDLE),
                                session_id=request.session_id,
                            )
                    except BaseException as exc:
                        logger.warning(
                            "Chat start failed (phase=run_state, exception_type={})",
                            type(exc).__name__,
                        )
                    finally:
                        try:
                            controller._agent_wake_turn_sessions.discard(
                                request.session_id
                            )
                            if self._active.get(request.session_id) is item:
                                self._active.pop(request.session_id, None)
                        finally:
                            item.capacity_owner.release_automatic_primary(
                                request.session_id, item.token
                            )

    async def accept(self, authorization: AgentChatStartAuthorization) -> bool:
        """Serialize the ledger ownership cutoff with manual/source withdrawal."""
        from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkLimits

        item, request = authorization, authorization.request
        if not await self._controller.store.drain_agent_handoff(request.session_id):
            return False
        reason = self._runtime_refusal(request)
        if reason is None and (
            self._controller._agent_bridge is not item.bridge
            or item.context.owner_id != self._controller._fleet_wake.runtime_owner_id
        ):
            reason = "runtime_unavailable"
        if reason is None and not self._destination_available(request):
            reason = "destination_unavailable"
        if reason is not None:
            item.withdrawal_reason = reason
            return False
        # Own source reads too; no await through the durable ownership cutoff.
        with operation_owned_connection(item.database):
            if (
                not self.authorizes(item, request.session_id)
                or item.accepted
                or not self._source_live(request)
                or not self._target_unchanged(request)
                or not self._controller._fleet_wake.try_claim_automatic_primary(
                    request.session_id, item.token
                )
            ):
                return False
            with item.database.connection() as conn:
                previous_timeout = int(
                    conn.execute("PRAGMA busy_timeout").fetchone()[0]
                )
                conn.execute("PRAGMA busy_timeout=0")
                try:
                    # Assign custody before fallible timeout/handle restoration.
                    item.accepted = item.context.ledger.accept_chat_start(
                        request.attempt_id,
                        owner_id=item.context.owner_id,
                        limits=AutomaticWorkLimits.from_settings(),
                    )
                except sqlite3.OperationalError as exc:
                    if (getattr(exc, "sqlite_errorcode", 0) & 0xFF) in {
                        sqlite3.SQLITE_BUSY,
                        sqlite3.SQLITE_LOCKED,
                    }:
                        item.withdrawal_reason = "ledger_contended"
                    raise
                finally:
                    conn.execute(f"PRAGMA busy_timeout={previous_timeout}")
        return item.accepted

    async def confirm_receipt(
        self, authorization: AgentChatStartAuthorization, checkpoint: Any
    ) -> None:
        """Open automatic call authority only after the matching conversation fence."""
        item = authorization
        if (
            not self.authorizes(item, item.request.session_id)
            or not item.accepted
            or checkpoint.agent_chat_start_attempt_id != item.request.attempt_id
            or checkpoint.conversation_id != item.request.conversation_id
        ):
            raise PermissionError("chat start receipt mismatch")
        item.context.mark_accepted()
        item.receipted = True
        await self._resolve(item, HANDOFF_LAUNCH_STARTED)

    def withdraw_prepared(self, session_id: str, reason: str) -> bool:
        """Withdraw target or source preparations; accepted targets stay independent."""
        withdrawn = False
        for item in tuple(self._active.values()):
            if (
                session_id
                not in {item.request.session_id, item.request.source_session_id}
                or item.accepted
                or item.withdrawn
            ):
                continue
            item.withdrawn = True
            withdrawn = True
            item.withdrawal_reason = reason
            if item.task is not None and item.running:
                item.task.cancel()
        return withdrawn

    def dispose(self) -> None:
        """Fence future admission and drain owned tasks through controller shutdown."""
        self._disposed = True
        for item in tuple(self._active.values()):
            if not item.accepted:
                item.withdrawn = True
            item.withdrawal_reason = "shutdown"
            if item.task is not None and item.running:
                item.task.cancel()
