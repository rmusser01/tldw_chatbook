"""Prepare an existing runtime receipt without retaining a Console view."""

from __future__ import annotations

import asyncio
import functools
import sys
from dataclasses import dataclass, field, replace
from types import MethodType
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.app_services import read_app_config
from tldw_chatbook import config
from tldw_chatbook.config import first_send_persistence_source_current as _native_config_current
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from .console_chat_models import ConsoleSubmissionOrigin
from .console_ordinary_tree import _function_record, _function_unchanged
from .console_preparation_reads import run_preparation_read

if TYPE_CHECKING:
    from . import console_runtime as native_host
    from .console_received_intent import ConsoleReceivedTurnIntent


def _attachment_receipt(controller, session_id):
    from .console_received_turn import received_turn_claim_for

    claim = received_turn_claim_for(controller.store, session_id)
    runtime = controller._hooks_v2_runtime
    if claim is None or runtime is None:
        return None
    record = runtime._turn_custody.get(claim.request_id)
    if record is None or record.received_claim is not claim:
        return None
    runtime._require_received_custody_current(record, controller)
    return record


def received_attachment_metadata(controller, session_id):
    """Read the original receipt's staged prefix without transferring it."""
    record = _attachment_receipt(controller, session_id)
    if record is None:
        return None
    if record.request.input_origin == "command":
        return ()
    if record.inputs.attachments_transferred:
        return record.inputs.attachments
    expected = record.request.attachment_ids
    pending = tuple(record.store.pending_attachments(session_id))[: len(expected)]
    if tuple(item.attachment_id for item in pending) != expected:
        raise RuntimeError("Pending attachments changed before physical admission.")
    return pending


def admit_received_attachments(controller, session_id):
    """Transfer only behind the originating asynchronous physical admission."""
    record = _attachment_receipt(controller, session_id)
    if record is None:
        return None
    if record.request.input_origin == "command":
        return ()
    if not record.inputs.attachments_transferred:
        record.inputs.attachments = record.store.transfer_pending_attachments_to_turn(
            session_id, record.turn_id, record.request.attachment_ids
        )
        record.inputs.attachments_transferred = True
    return record.inputs.attachments


def stock_native_methods(owner: Any, originals: tuple) -> bool:
    """Preserve custom outer and nested callback signatures and affinity."""
    return all(
        isinstance(callback := getattr(owner, name, None), MethodType)
        and callback.__self__ is owner
        and callback.__func__ is function
        and function.__code__ is code
        for name, function, code in originals
    )


def preparation_native_runner(
    controller: Any, session_id: str | None = None, *, require_current=None
):
    """Capture one finite operation's sources before creating its native worker."""
    from tldw_chatbook import config

    app, store = controller.app, controller.store
    runtime = getattr(controller, "_hooks_v2_runtime", None)
    app_config = getattr(app, "app_config", None)
    identity = config.current_config_identity()
    session = next((row for row in store.sessions() if row.id == session_id), None)
    incarnation = getattr(session, "incarnation_id", None)
    binding = getattr(session, "conversation_binding_revision", None)

    def current():
        import sys

        if require_current is not None:
            require_current()
        if (
            sys.modules.get("tldw_chatbook.config") is not config
            or controller.app is not app
            or controller.store is not store
            or getattr(app, "app_config", None) is not app_config
            or config.current_config_identity() != identity
            or controller._disposed
            or controller._shutdown_requested.is_set()
            or getattr(controller, "_hooks_v2_runtime", None) is not runtime
            or (
                session_id is not None
                and (
                    session is None
                    or next(
                        (row for row in store.sessions() if row.id == session_id), None
                    )
                    is not session
                    or session.incarnation_id != incarnation
                    or session.conversation_binding_revision != binding
                )
            )
            or (
                runtime is not None
                and (
                    runtime._disposed
                    or runtime._chat_controller is not controller
                    or runtime._chat_store is not store
                    or session_id in runtime._admission_fenced_sessions
                )
            )
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

    async def run(callback):
        return await run_preparation_read(
            callback,
            creator=controller,
            session_id=session_id,
            reads=controller._preparation_reads,
            observers=() if runtime is None else (runtime._preparation_reads,),
            require_current=current,
        )

    return run


@dataclass(slots=True)
class ReceivedPreparationSource:
    """Receipt-time source identities; these grant no execution authority."""

    app: Any = field(repr=False)
    app_config: Any = field(repr=False)
    store: Any = field(repr=False)
    controller: Any = field(repr=False)
    controller_app: Any = field(repr=False)
    config: Any = field(repr=False)
    config_identity: Any = field(repr=False)
    permissions: Any = field(repr=False)
    configuration_preparation: Any = field(default=None, repr=False)
    prompt_scope: Any = field(default=None, repr=False)
    prompt_sources: tuple = field(default=(), repr=False)


def _prompt_scope_service(app):
    """Read the optional source at the caller's exact capture/fence position."""
    try:
        return app.prompt_scope_service
    except AttributeError:
        return None


def received_preparation_source(runtime, *, configuration_preparation=None):
    from tldw_chatbook import config

    return ReceivedPreparationSource(
        runtime._app,
        getattr(runtime._app, "app_config", None),
        runtime._chat_store,
        runtime._chat_controller,
        getattr(runtime._chat_controller, "app", None),
        config,
        config.current_config_identity(),
        runtime._hook_permissions,
        configuration_preparation,
        _prompt_scope_service(runtime._app),
    )


def require_received_source(runtime, record, source, *, inputs=True):
    import sys

    runtime._raise_if_disposed_or_session_fenced(record.session_id)
    for backend, identity in source.prompt_sources:
        try:
            now = source.prompt_scope.mixin_source_identity(mode=backend)
        except Exception:  # noqa: BLE001 - source uncertainty refuses admission.
            raise RecoveryRequired("console_snapshot_owner_changed") from None
        if now[0] != identity[0] or any(a is not b for a, b in zip(now[1:], identity[1:])):
            raise RecoveryRequired("console_snapshot_owner_changed")
    if (
        runtime._app is not source.app
        or _prompt_scope_service(source.app) is not source.prompt_scope
        or getattr(source.app, "app_config", None) is not source.app_config
        or runtime._chat_store is not source.store
        or runtime._chat_controller is not source.controller
        or getattr(source.controller, "app", None) is not source.controller_app
        or getattr(source.controller, "store", source.store) is not source.store
        or getattr(source.controller, "_disposed", False)
        or (
            getattr(source.controller, "_shutdown_requested", None) is not None
            and source.controller._shutdown_requested.is_set()
        )
        or sys.modules.get("tldw_chatbook.config") is not source.config
        or source.config.current_config_identity() != source.config_identity
        or (
            source.permissions is not None
            and runtime._hook_permissions is not source.permissions
        )
        or (
            record.received_claim is not None
            and not source.store.received_turn_is_current(record.received_claim)
        )
        or (
            inputs
            and not source.store.session_inputs_are_current(
                record.received_intent.inputs,
                include_draft=record.received_intent._pressed_inputs is None,
            )
        )
        or (
            inputs
            and record.received_intent.inputs.input_origin == "composer"
            and (
                runtime.snapshot_console_staged_evidence(record.session_id)[0]
                is not record.received_intent.staged_evidence_launch
                or runtime.snapshot_console_staged_evidence(record.session_id)[1]
                != record.received_intent.staged_evidence_revision
            )
        )
    ):
        raise RecoveryRequired("console_snapshot_owner_changed")


async def run_received_intent(runtime, record, source):
    """Keep normal receipt identity bound throughout its preparation driver."""
    from contextlib import nullcontext

    from .console_received_turn import bind_received_turn_claim

    scope = (
        bind_received_turn_claim(source.store, record.received_claim)
        if record.received_claim is not None
        else nullcontext()
    )
    with scope:
        return await _run_received_intent_bound(runtime, record, source)


async def _run_received_intent_bound(runtime, record, source):
    """Use existing review, configuration, queue and submit owners in order."""
    from .console_chat_models import ConsoleSubmissionOrigin
    from .console_prompt_queue import QueueMutationStatus
    from .console_turn_context import ConsoleTurnCustodyRequest

    # The lazy custody task never performs native work in the caller's intake.
    await asyncio.sleep(0)
    require_received_source(runtime, record, source)
    intent = record.received_intent
    controller, store = source.controller, source.store
    from .console_prompt_mixins import MixinSnapshotUnavailableError
    preferences = intent.inputs.prompt_mixin_preferences
    if preferences is None:
        raise MixinSnapshotUnavailableError("mixin_preferences_unavailable")
    mixins = await controller.prompt_mixin_service().capture(
        record.session_id, expected_revision=preferences.revision
    )
    require_received_source(runtime, record, source)
    chat_start = getattr(controller, "_chat_start", None)
    if chat_start is not None:
        await chat_start.withdraw_for_manual(record.session_id)
        require_received_source(runtime, record, source)

    def snapshot():
        owner = runtime.ensure_hook_permissions()
        if source.permissions is None:
            source.permissions = owner
        require_received_source(runtime, record, source)
        from tldw_chatbook.Agents.hook_permissions import HookPermissions

        if isinstance(owner, HookPermissions):
            # ADR-230 decision 3: the same snapshot() read, also keeping the
            # targets it published, so this attempt's later preparation can
            # share it instead of repeating the full read.
            read = owner.authority_read()
            if read.snapshot.ready:
                from .console_runtime import _prepare_legacy_hook_import

                _prepare_legacy_hook_import(runtime)
            return read
        return owner.snapshot()

    async def read_snapshot():
        result = await run_preparation_read(
            snapshot,
            creator=runtime,
            session_id=record.session_id,
            reads=runtime._preparation_reads,
            observers=(controller._preparation_reads,),
            require_current=lambda: require_received_source(runtime, record, source),
        )
        # Imported after the worker read, so a cold owner import stays there.
        from tldw_chatbook.Agents.hook_permissions import HookAuthorityRead

        if type(result) is HookAuthorityRead:
            return result.snapshot, result
        return result, None

    review, authority = await read_snapshot()
    if not review.ready:
        result = await runtime.request_initial_hook_review(
            record.session_id,
            record.turn_id,
            record.received_claim.generation
            if record.received_claim is not None
            else intent.inputs.draft_revision,
            review,
        )
        require_received_source(runtime, record, source)
        if result.kind != "ready":
            raise RuntimeError("Send cancelled; draft kept.")
        # The review answer is no authority: re-read, and share only this read.
        review, authority = await read_snapshot()
    if not review.ready:
        raise RuntimeError("Hooks changed; Send again.")
    from .console_hook_preparation import ConsoleHookAttemptRead

    # Bound to this attempt only and passed explicitly to its own submission;
    # each later consumer re-validates it or reads fresh (ADR-230 decision 3).
    hook_read = (
        ConsoleHookAttemptRead(record.session_id, authority)
        if authority is not None
        else None
    )
    require_received_source(runtime, record, source)
    prepared_skills = None
    preparation = source.configuration_preparation
    if preparation is not None and preparation.trust_source is not None:
        from .console_configuration_preparation import (
            capture_console_skill_catalog_owned,
            finish_console_skill_catalog_owned,
        )

        def current():
            require_received_source(runtime, record, source)

        reads, observers = controller._preparation_reads, (runtime._preparation_reads,)
        catalog = await capture_console_skill_catalog_owned(
            source.app,
            store,
            controller,
            session_id=record.session_id,
            turn_id=record.turn_id,
            skill_workspace_id=intent.selection.skill_workspace_id,
            preparation=preparation,
            reads=reads,
            observers=observers,
            require_current=current,
        )
        prepared_skills = await finish_console_skill_catalog_owned(
            catalog,
            reads=reads,
            observers=observers,
            require_current=current,
        )
    configuration = await controller.capture_turn_configuration_snapshot(
        record.session_id,
        selection=intent.selection,
        **({} if prepared_skills is None else {"_prepared_skills": prepared_skills}),
    )
    configuration = replace(configuration, prompt_mixins=mixins)
    require_received_source(runtime, record, source)
    configuration = replace(
        configuration, recovery_user_origin=intent.recovery_user_origin
    )
    if any(
        item.file_type == "image"
        and item.insert_mode == "attachment"
        and item.data is not None
        for item in intent.inputs.attachments
    ) and not configuration.capabilities.get("vision", False):
        from .attachment_core import vision_block_reason
        from .console_chat_models import ConsoleMessageRole

        # Configuration and the received source were checked immediately
        # above. Use that frozen verdict, not a fresh capability/config read.
        # This branch precedes request promotion, so generic runtime recovery
        # has no request on which to project its refusal.
        refusal = vision_block_reason(
            configuration.provider_selection.provider,
            configuration.effective_model,
            is_capable=lambda _provider, _model: False,
        )
        store.append_message(
            record.session_id, role=ConsoleMessageRole.SYSTEM, content=str(refusal)
        )
        raise RuntimeError(
            "The selected model cannot accept this attachment; draft kept."
        )

    if intent.queue_revision is not None:
        from .console_chat_controller import _CONSOLE_RECEIVED_QUEUE_METHOD

        function, code = _CONSOLE_RECEIVED_QUEUE_METHOD
        method = controller.queue_prompt
        kwargs = {
            "text": intent.inputs.draft,
            "expected_revision": intent.queue_revision,
            "configuration": configuration,
        }
        if not (
            isinstance(method, MethodType)
            and method.__self__ is controller
            and method.__func__ is function
            and function.__code__ is code
        ):
            raise RecoveryRequired("console_snapshot_owner_changed")

        def before_admission():
            if not stock_native_methods(
                controller, (("queue_prompt", function, code),)
            ):
                raise RecoveryRequired("console_snapshot_owner_changed")
            require_received_source(runtime, record, source)

        kwargs["_before_admission"] = before_admission
        if intent.inputs.input_origin == "command":
            kwargs["_command_inputs"] = intent.inputs
        queued = await method(record.session_id, **kwargs)
        if queued.applied:
            controller.prompt_mixin_service().publish_accepted_usage(
                record.session_id, configuration.prompt_mixins,
                observed_revision=preferences.revision if preferences is not None else None,
            )
            try:
                if intent.inputs.input_origin == "composer":
                    committed = store.commit_session_input_draft(intent.inputs)
                    if committed or intent._pressed_inputs is not None:
                        runtime._project_received_input(
                            record, draft_committed=committed
                        )
            except Exception:  # noqa: BLE001, S110 - presentation cannot retract an already accepted domain outcome.
                # Queue admission already won; presentation cannot retract it.
                pass
            return queued
        require_received_source(runtime, record, source)
        if queued.status is not QueueMutationStatus.REROUTE_NORMAL_SEND:
            raise RuntimeError(queued.detail or "Prompt queue admission refused.")
        claim = store.claim_received_turn(
            record.session_id,
            record.turn_id,
            draft_revision=(
                intent.inputs.draft_revision
                if intent.inputs.input_origin == "composer"
                else None
            ),
            _allow_draft_change=intent._pressed_inputs is not None,
        )
        if claim is None:
            raise RuntimeError(
                "Console session already has a received or prepared turn."
            )
        record.received_claim = claim
        runtime._note_received_admission_changed(store, record.session_id)

    require_received_source(runtime, record, source)
    record.inputs.staged_evidence_revision = intent.staged_evidence_revision
    record.request = ConsoleTurnCustodyRequest(
        turn_id=record.turn_id,
        session_id=record.session_id,
        draft=intent.inputs.draft,
        configuration=configuration,
        attachment_ids=tuple(item.attachment_id for item in intent.inputs.attachments),
        one_shot_prefill=intent.inputs.one_shot_prefill,
        one_shot_prefill_revision=intent.inputs.prefill_revision,
        staged_evidence_launch=intent.staged_evidence_launch,
        input_origin=intent.inputs.input_origin,
        command_id=intent.inputs.command_id,
        command_inputs=intent.inputs
        if intent.inputs.input_origin == "command"
        else None,
    )
    return await runtime._run_custodied_turn(
        record,
        origin=ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id=None,
        queue_authorization=None,
        wake_authorization=None,
        goal_authorization=None,
        raise_on_refusal=True,
        hook_read=hook_read,
        first_send_source=(
            source.config_identity,
            source.app_config,
            source.controller,
            source.store,
        ),
    )


def accept_received_intent(
    self, intent: ConsoleReceivedTurnIntent, *, _configuration_preparation=None,
    terminal_callback=None,
) -> str:
    """Reserve bounded authored input before any hook or configuration read."""
    from .console_received_intent import ConsoleReceivedTurnIntent

    if type(intent) is not ConsoleReceivedTurnIntent:
        raise TypeError("intent must be ConsoleReceivedTurnIntent")
    if intent.inputs.input_origin == "command" and self._workflow_closing:
        raise RuntimeError("Console owner is shutting down.")
    self._raise_if_disposed_or_session_fenced(intent.session_id)
    store = self._chat_store
    if store is None or self._chat_controller is None:
        raise RuntimeError("Console chat owner is unavailable.")
    if not store.session_inputs_are_current(intent.inputs, include_draft=intent._pressed_inputs is None):
        raise RuntimeError("Console input changed; Send again.")
    if _configuration_preparation is not None:
        from .console_configuration_preparation import (
            require_received_configuration_preparation,
        )

        require_received_configuration_preparation(
            _configuration_preparation,
            self._app,
            store,
            self._chat_controller,
            session_id=intent.session_id,
        )
    source = received_preparation_source(
        self, configuration_preparation=_configuration_preparation
    )
    preferences = intent.inputs.prompt_mixin_preferences
    if preferences is not None:
        try:
            source.prompt_sources = tuple(
                (backend, source.prompt_scope.mixin_source_identity(mode=backend))
                for backend in dict.fromkeys(row.source.backend for row in preferences.entries if row.enabled)
            )
        except Exception:  # noqa: BLE001 - source uncertainty refuses admission.
            from .console_prompt_mixins import MixinSnapshotUnavailableError
            raise MixinSnapshotUnavailableError("mixin_source_unavailable") from None
    claim = None
    if intent.queue_revision is None:
        claim = store.claim_received_turn(
            intent.session_id,
            intent.turn_id,
            draft_revision=(
                intent.inputs.draft_revision
                if intent.inputs.input_origin == "composer"
                else None
            ),
            _allow_draft_change=intent._pressed_inputs is not None,
        )
        if claim is None:
            raise RuntimeError(
                "Console session already has a received or prepared turn."
            )
    record = coroutine = None
    try:
        if claim is not None and self._response_rules is not None:
            self._response_rules.invalidate(intent.session_id, "user_input")
        record = self._register_custody(
            None,
            store=store,
            received_claim=claim,
            received_intent=intent,
        )
        coroutine = run_received_intent(self, record, source)
        record.task = self._create_custody_task(coroutine)
        record.task.add_done_callback(
            functools.partial(
                self._finish_custodied_turn,
                turn_id=record.turn_id,
                recover_before_acceptance=True,
                terminal_callback=terminal_callback,
            )
        )
    except BaseException:
        if coroutine is not None:
            coroutine.close()
        if record is not None:
            self._release_custody(record.turn_id)
        elif claim is not None:
            store.release_received_turn(claim)
        raise
    try:
        self._note_received_admission_changed(store, intent.session_id)
        view = self.view
        project = getattr(view, "_project_console_received_preparing", None)
        if callable(project):
            project(intent.session_id)
    except Exception as error:  # noqa: BLE001 - presentation cannot retract an already accepted domain outcome.
        logger.debug(
            "Received turn projection failed (exception_type={})",
            type(error).__name__,
        )
    return record.turn_id




def accept_turn(
    self,
    request: native_host.ConsoleTurnCustodyRequest,
    *,
    origin: native_host.ConsoleSubmissionOrigin = ConsoleSubmissionOrigin.MANUAL,
    queue_entry_id: str | None = None,
    queue_authorization: native_host.Any | None = None,
    wake_authorization: native_host.Any | None = None,
    goal_authorization: native_host.Any | None = None,
    chat_start_authorization: native_host.Any | None = None,
    terminal_callback: native_host.Callable[[bool], None] | None = None,
    recover_before_acceptance: bool = True,
) -> str:
    """Synchronously retain input metadata for asynchronous physical admission."""

    from . import console_runtime as native_host
    with self._first_send_source_scope():
        from tldw_chatbook import config

        first_send_source = (
            config.current_config_identity(),
            read_app_config(self._app),
            self._chat_controller,
            self._chat_store,
        )
        self._raise_if_disposed_or_session_fenced(request.session_id)
        if request.configuration.session_id != request.session_id:
            raise ValueError("Turn configuration belongs to another session.")
        store = self._chat_store
        if store is None:
            raise RuntimeError("Console chat store is unavailable.")
        if request.input_origin == "command":
            if self._workflow_closing:
                raise RuntimeError("Console owner is shutting down.")
            if request.command_inputs is None or not store.session_inputs_are_current(
                request.command_inputs
            ):
                raise RuntimeError("Console command target configuration changed.")
        if origin is ConsoleSubmissionOrigin.AGENT_CHAT_START:
            controller = self._chat_controller
            if (
                controller is None
                or controller._hooks_v2_runtime is not self
                or controller.store is not store
                or not controller._chat_start.authorizes(chat_start_authorization, request.session_id)
                or chat_start_authorization.task is not asyncio.current_task()
                or chat_start_authorization.request.configuration is not request.configuration
                or chat_start_authorization.request.opening_prompt != request.draft
            ):
                raise PermissionError("Chat Start custody changed.")
        elif chat_start_authorization is not None:
            raise ValueError("Chat Start authorization requires its original submission origin.")
        pressed = request._pressed_inputs
        if pressed is not None:
            from .console_received_intent import ConsoleSessionInputSnapshot

            if (
                type(pressed) is not ConsoleSessionInputSnapshot
                or pressed._store_ref() is not store
                or pressed.session_id != request.session_id
                or not store.session_inputs_are_current(pressed, include_draft=False)
            ):
                raise RuntimeError("Pressed Console input changed before admission.")
        claim = store.claim_received_turn(
            request.session_id,
            request.turn_id,
            origin=origin,
            draft_revision=pressed.draft_revision
            if pressed is not None
            else request.draft_revision,
            _allow_draft_change=pressed is not None,
        )
        if claim is None:
            raise RuntimeError("Console session already has an admission owner.")
        attachments = ()
        try:
            self._note_received_admission_changed(store, request.session_id)
            if (
                origin is native_host.ConsoleSubmissionOrigin.MANUAL
                and self._response_rules is not None
            ):
                self._response_rules.invalidate(request.session_id, "user_input")
            record = self._register_custody(
                request,
                attachments,
                self._staged_evidence_lease_revision(
                    request.staged_evidence_launch, request.session_id
                ),
                store=store,
                received_claim=claim,
            )
        except BaseException:
            try:
                store.restore_transferred_pending_attachments(
                    request.session_id, attachments
                )
            finally:
                store.release_received_turn(claim)
            raise
        if self._archive_recovery is not None:
            self._archive_recovery.accept_user_proof(
                request.recovery_user_origin,
                request.session_id,
                request.turn_id,
            )
        coroutine = self._run_custodied_turn(
            record,
            first_send_source=first_send_source,
            origin=origin,
            queue_entry_id=queue_entry_id,
            queue_authorization=queue_authorization,
            wake_authorization=wake_authorization,
            goal_authorization=goal_authorization,
            chat_start_authorization=chat_start_authorization,
            raise_on_refusal=recover_before_acceptance,
        )
        try:
            record.task = self._create_custody_task(coroutine)
        except BaseException:
            coroutine.close()
            try:
                store.restore_transferred_pending_attachments(
                    request.session_id, attachments
                )
            finally:
                self._release_custody(record.turn_id)
            raise
        record.task.add_done_callback(
            native_host.functools.partial(
                self._finish_custodied_turn,
                turn_id=record.turn_id,
                recover_before_acceptance=recover_before_acceptance,
                terminal_callback=terminal_callback,
            )
        )
        return record.turn_id


async def _run_custodied_turn(
    self,
    record: native_host._ConsoleTurnCustodyRecord,
    *,
    origin: native_host.ConsoleSubmissionOrigin,
    queue_entry_id: str | None,
    queue_authorization: native_host.Any | None,
    wake_authorization: native_host.Any | None,
    goal_authorization: native_host.Any | None,
    chat_start_authorization: native_host.Any | None = None,
    raise_on_refusal: bool,
    hook_read: native_host.Any = None,
    first_send_source: tuple,
) -> native_host.Any:
    """Run one screen-free turn using only its frozen custody record.

    ``hook_read`` is the received intent's one full hook authority read
    (``ConsoleHookAttemptRead``); it goes, as an argument, only to this
    record's own initial submission (ADR-230 decision 3).
    """

    from . import console_runtime as native_host
    request = record.request
    controller = self._chat_controller
    if request is None or controller is None:
        raise RuntimeError("Console controller is unavailable for runtime custody.")

    def require_received():
        self._require_received_custody_current(record, controller)

    require_received()
    if self._archive_recovery is not None and record.received_intent is not None:
        self._archive_recovery.accept_user_proof(
            request.recovery_user_origin,
            request.session_id,
            request.turn_id,
        )
    if self._archive_recovery is not None and origin in {
        native_host.ConsoleSubmissionOrigin.MANUAL,
        native_host.ConsoleSubmissionOrigin.QUEUED,
    }:
        await self._archive_recovery.prepare_for_admission(request.session_id)
        require_received()
    if record.archive_conversation_id:
        from tldw_chatbook.Chat.conversation_archive_actions import (
            conversation_send_refusal,
        )

        refusal = await conversation_send_refusal(
            self._app, record.archive_conversation_id
        )
        if refusal:
            notify = getattr(self._app, "notify", None)
            if callable(notify):
                notify(refusal, severity="warning")
            # TASK-33621.2: the parked turn's shelf entry states this
            # refusal too, like a controller refusal's below.
            raise native_host._ConsoleTurnRefusedError(
                "Console conversation is unavailable for submission.",
                reason=refusal,
            )

    def mark_durable_acceptance() -> None:
        publish_received_acceptance(self, record, controller)
        self._observe_remote_custody(
            record.turn_id, state="accepted", native_id=record.turn_id
        )

    async def submit() -> native_host.Any:
        require_received()
        if chat_start_authorization is None:
            controller.prompt_queue_coordinator.bind_turn_request(request, origin=origin)
        from .console_received_turn import bind_received_turn_claim

        from .console_chat_controller import ConsoleChatController

        shared = (
            {"_hook_read": hook_read}
            if hook_read is not None
            and getattr(controller.submit_draft, "__func__", None)
            is ConsoleChatController.submit_draft
            else {}
        )
        if request.input_origin == "command":
            shared["preserve_composer"] = True
        issuing_store, issuing_claim = record.store, record.received_claim
        try:
            with bind_received_turn_claim(issuing_store, issuing_claim):
                if chat_start_authorization is not None:
                    return await controller.submit_draft(
                        request.draft,
                        session_id=request.session_id,
                        origin=origin,
                        chat_start_authorization=chat_start_authorization,
                        configuration=request.configuration,
                        accepted_attachments=(),
                        custody_acceptance_hook=mark_durable_acceptance,
                    )
                return await controller.submit_draft(
                    request.draft,
                    session_id=request.session_id,
                    origin=origin,
                    queue_entry_id=queue_entry_id,
                    queue_authorization=queue_authorization,
                    wake_authorization=wake_authorization,
                    goal_authorization=goal_authorization,
                    configuration=native_host.replace(
                        request.configuration,
                        recovery_user_origin=request.recovery_user_origin,
                        skill_context_maximum={
                            **request.configuration.skill_context_maximum,
                            "plugin_turn_id": request.turn_id,
                        },
                    ),
                    accepted_attachments=received_attachment_metadata(
                        controller, request.session_id
                    ),
                    captured_one_shot_prefill=request.one_shot_prefill,
                    captured_one_shot_prefill_revision=(
                        request.one_shot_prefill_revision
                    ),
                    staged_evidence_launch=request.staged_evidence_launch,
                    staged_evidence_capture=self._capture_frozen_console_staged_rag,
                    staged_evidence_release=native_host.functools.partial(
                        self.release_console_staged_evidence,
                        revision=record.inputs.staged_evidence_revision,
                        session_id=record.session_id,
                    ),
                    custody_acceptance_hook=mark_durable_acceptance,
                    **shared,
                )
        finally:
            # Nonpromoting routes release after physical submit completion,
            # before the enclosing prompt chain admits its next request.
            if issuing_store.release_received_turn(issuing_claim):
                self._note_received_admission_changed(issuing_store, request.session_id)

    result = (
        await submit()
        if origin
        in {
            native_host.ConsoleSubmissionOrigin.AGENT_WAKE,
            native_host.ConsoleSubmissionOrigin.GOAL_ITERATION,
            native_host.ConsoleSubmissionOrigin.AGENT_CHAT_START,
        }
        else await controller.run_prompt_chain(
            session_id=request.session_id,
            initial_turn=submit,
        )
    )
    held_preparation_id = getattr(result, "preparation_id", None)
    if (
        not bool(getattr(result, "accepted", False))
        and not record.inputs.durable_accepted
        and raise_on_refusal
        # TASK-34350: a send held at the compaction threshold is waiting
        # on the user's answer in its own card, not refused; recording it
        # as a turn recovery too showed one send on two surfaces.
        and not (
            held_preparation_id is not None
            and controller.context_compaction_hold(held_preparation_id) is not None
        )
    ):
        from .console_unsent_turn import card_holds_refusal

        raise native_host._ConsoleTurnRefusedError(
            "Console turn was refused before durable acceptance.",
            reason=str(getattr(result, "visible_copy", "") or ""),
            preparation_id=held_preparation_id,
            card_held=card_holds_refusal(controller, held_preparation_id),
        )
    run_state_for = getattr(controller, "run_state_for", None)
    run_state = run_state_for(request.session_id) if callable(run_state_for) else None
    if (
        origin is native_host.ConsoleSubmissionOrigin.MANUAL
        and bool(getattr(result, "accepted", False))
        and getattr(run_state, "status", None) is native_host.ConsoleRunStatus.COMPLETED
    ):
        await self._record_successful_first_send(first_send_source)
    return result


_NATIVE_CONFIG_CHECK = _function_record(_native_config_current)


async def _persist_first_send_when_quiescent(
    self, config_module, identity, app_config, controller, store
) -> None:
    """Run the original one-key save only beyond all original lifetimes."""

    from . import console_runtime as native_host
    check = _NATIVE_CONFIG_CHECK

    def original_config():
        return (
            _native_config_current is check[0]
            and _function_unchanged(check)
            and check[0](config)
        )

    wakeup = self._first_send_wakeup
    try:
        while True:
            wakeup.clear()
            registry = getattr(controller, "prompt_queue_registry", None)
            queued = bool(
                registry is not None
                and any(state.total_count for state in registry._states.values())
            )
            with self._execution_capacity_lock:
                if (
                    not self._first_send_sources
                    and not self._ordinary_turns
                    and not self._turn_custody
                    and not queued
                ):
                    self._first_send_publishing = True
                    break
            await wakeup.wait()

        def current():
            return (
                sys.modules.get("tldw_chatbook.config") is config_module
                and original_config()
                and vars(self._app).get("console_runtime") is self
                and vars(self._app).get("app_config") is app_config
                and self._chat_controller is controller
                and self._chat_store is store
                and config_module.current_config_identity() == identity
            )

        if not current():
            raise RuntimeError("Original first-send source changed.")
        producer = native_host.asyncio.create_task(
            native_host.asyncio.to_thread(
                config_module.apply_settings_mutation_to_cli_config,
                {"console.onboarding": {"first_send_completed": True}},
                mutation_precondition=current,
            )
        )
        cancelled = False
        while not producer.done():
            try:
                await native_host.asyncio.shield(producer)
            except native_host.asyncio.CancelledError:
                cancelled = True
        result = producer.result()
        if cancelled:
            raise native_host.asyncio.CancelledError
        if result.conflict or result.failure_phase is not None:
            raise RuntimeError("Original first-send persistence failed.")
    except Exception as exc:  # noqa: BLE001 - the completed turn stays valid
        native_host.logger.warning(
            "Failed to persist Console first-send completion (exception_type={})",
            type(exc).__name__,
        )
    finally:
        with self._execution_capacity_lock:
            self._first_send_publishing = False
        self._wake_first_send()

def project_received_input(self, record, *, draft_committed=False) -> None:
    """Project a domain CAS into only the original attached composer."""
    intent = record.received_intent
    request = record.request
    if intent is not None and intent.inputs.input_origin != "composer":
        return
    if request is not None and request.input_origin != "composer":
        return
    if intent is not None:
        inputs, stash = intent._pressed_inputs, intent._pressed_stash
        generation = intent.view_attachment_generation
    elif request is not None and request._pressed_inputs is not None:
        inputs, stash = request._pressed_inputs, request._pressed_stash
        generation = request._pressed_attachment_generation
    else:
        return
    if self._attached_generation != generation:
        return
    project = getattr(self.view, "_project_console_received_input", None)
    if callable(project):
        if draft_committed:
            saved_project = getattr(self.view, "_project_console_saved_input", None)
            if callable(saved_project):
                saved_project(
                    record.session_id,
                    _captured_stash=stash,
                    _captured_inputs=intent.inputs if intent is not None else inputs,
                )
                return
        if inputs is not None:
            project(
                record.session_id,
                _captured_stash=stash,
                _captured_inputs=inputs,
            )
        else:
            project(record.session_id)



def require_received_custody_current(self, record, controller) -> None:
    """Refuse source or admission displacement before initial submit effects."""
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    self._raise_if_disposed_or_session_fenced(record.session_id)
    store, claim = record.store, record.received_claim
    if (
        store is None
        or claim is None
        or self._chat_store is not store
        or self._chat_controller is not controller
        or not store.received_turn_is_current(claim)
        or (
            isinstance(controller, ConsoleChatController)
            and controller.store is not store
        )
    ):
        raise RuntimeError("Received turn owner changed.")



def publish_received_acceptance(self, record, controller) -> None:
    record.inputs.durable_accepted = True
    intent = record.received_intent
    request = record.request
    captured = intent.inputs if intent is not None else (request._pressed_inputs if request is not None else None)
    pressed = intent._pressed_inputs if intent is not None else captured
    if captured is not None and captured.input_origin == "composer":
        try:
            store, claim = record.store, record.received_claim
            # Saved acceptance is a fact even when its original owner
            # changed or Stop won. Those outcomes retain the composer.
            if (
                self._disposed
                or record.session_id in self._admission_fenced_sessions
                or self._chat_store is not store
                or self._chat_controller is not controller
                or controller.store is not store
                or claim is None
                or not store.received_turn_matches_session(claim)
            ):
                return
            preparation_id = controller._active_submit_preparations.get(
                asyncio.current_task()
            )
            native_owner = controller._ordinary_native_commit_owner(preparation_id)
            if native_owner is not None and (
                native_owner.caller_cancelled
                or native_owner.explicit_stop
                or native_owner.commit_error is not None
                or not controller._ordinary_native_commit_current(native_owner)
            ):
                return
            committed = store.commit_session_input_draft(captured)
            if (intent is not None and request is not None
                    and captured.prompt_mixin_preferences is not None):
                controller.prompt_mixin_service().publish_accepted_usage(
                    record.session_id, request.configuration.prompt_mixins,
                    observed_revision=captured.prompt_mixin_preferences.revision,
                )
            if committed or pressed is not None:
                self._project_received_input(record, draft_committed=committed)
        except Exception as error:  # noqa: BLE001 - presentation cannot retract an already accepted domain outcome.
            logger.debug(
                "Received input projection failed (exception_type={})",
                type(error).__name__,
            )


def resolve_received_provider(controller, selection):
    """Retain only stock provider readiness through its existing native owner."""
    from .console_provider_gateway import _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS

    gateway = controller.provider_gateway
    kwargs = {}
    if stock_native_methods(gateway, _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS):
        with controller._active_submit_tasks_lock:
            session_id = controller._active_submit_tasks.get(asyncio.current_task())

        def current_gateway():
            if controller.provider_gateway is not gateway or not stock_native_methods(
                gateway, _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS
            ):
                raise RuntimeError("Console provider owner changed.")

        kwargs["_run_native"] = preparation_native_runner(
            controller, session_id, require_current=current_gateway
        )
    return gateway.resolve_for_send(selection, **kwargs)


async def capture_received_library_policy(controller, coordinator, session_id):
    """Capture the original repository and DB for one stock execution read."""
    from .console_library_policy_coordinator import (
        _CONSOLE_LIBRARY_EXECUTION_NATIVE_METHODS,
    )

    kwargs = {}
    if stock_native_methods(coordinator, _CONSOLE_LIBRARY_EXECUTION_NATIVE_METHODS):
        store = controller.store
        repository = coordinator.repository
        database = getattr(repository, "db", None)

        def current_library():
            if (
                not stock_native_methods(coordinator, _CONSOLE_LIBRARY_EXECUTION_NATIVE_METHODS)
                or controller.store is not store
                or store.library_policy_coordinator is not coordinator
                or coordinator.repository is not repository
                or getattr(repository, "db", None) is not database
            ):
                raise RuntimeError("Console Library owner changed.")

        kwargs["_run_native"] = preparation_native_runner(
            controller, session_id, require_current=current_library
        )
    return await coordinator.capture_for_execution(session_id, **kwargs)
