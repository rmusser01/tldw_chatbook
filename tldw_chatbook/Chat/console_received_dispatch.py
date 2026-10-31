"""Prepare an existing runtime receipt without retaining a Console view."""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from .console_preparation_reads import run_preparation_read

if TYPE_CHECKING:
    from .console_received_intent import ConsoleReceivedTurnIntent



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
    )


def require_received_source(runtime, record, source, *, inputs=True):
    import sys

    runtime._raise_if_disposed_or_session_fenced(record.session_id)
    if (
        runtime._app is not source.app
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
                record.received_intent.inputs
            )
        )
        or (
            inputs
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
    chat_start = getattr(controller, "_chat_start", None)
    if chat_start is not None:
        await chat_start.withdraw_for_manual(record.session_id)
        require_received_source(runtime, record, source)

    def snapshot():
        owner = runtime.ensure_hook_permissions()
        if source.permissions is None:
            source.permissions = owner
        require_received_source(runtime, record, source)
        return owner.snapshot()

    async def read_snapshot():
        return await run_preparation_read(
            snapshot,
            creator=runtime,
            session_id=record.session_id,
            reads=runtime._preparation_reads,
            observers=(controller._preparation_reads,),
            require_current=lambda: require_received_source(runtime, record, source),
        )

    review = await read_snapshot()
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
        review = await read_snapshot()
    if not review.ready:
        raise RuntimeError("Hooks changed; Send again.")
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
    require_received_source(runtime, record, source)
    if any(
        item.file_type == "image"
        and item.insert_mode == "attachment"
        and item.data is not None
        for item in intent.inputs.attachments
    ) and not configuration.capabilities.get("vision", False):
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
        queued = await method(record.session_id, **kwargs)
        if queued.applied:
            try:
                if store.commit_session_input_draft(intent.inputs):
                    runtime._project_received_input(record)
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
            draft_revision=intent.inputs.draft_revision,
        )
        if claim is None:
            raise RuntimeError(
                "Console session already has a received or prepared turn."
            )
        record.received_claim = claim
        runtime._note_received_admission_changed(store, record.session_id)

    require_received_source(runtime, record, source)
    attachments = store.transfer_pending_attachments_to_turn(
        record.session_id,
        record.turn_id,
        tuple(item.attachment_id for item in intent.inputs.attachments),
    )
    record.inputs.attachments = attachments
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
    )
    return await runtime._run_custodied_turn(
        record,
        origin=ConsoleSubmissionOrigin.MANUAL,
        queue_entry_id=None,
        queue_authorization=None,
        wake_authorization=None,
        goal_authorization=None,
        raise_on_refusal=True,
    )


def accept_received_intent(
    self, intent: ConsoleReceivedTurnIntent, *, _configuration_preparation=None
) -> str:
    """Reserve bounded authored input before any hook or configuration read."""
    from .console_received_intent import ConsoleReceivedTurnIntent

    if type(intent) is not ConsoleReceivedTurnIntent:
        raise TypeError("intent must be ConsoleReceivedTurnIntent")
    self._raise_if_disposed_or_session_fenced(intent.session_id)
    store = self._chat_store
    if store is None or self._chat_controller is None:
        raise RuntimeError("Console chat owner is unavailable.")
    if not store.session_inputs_are_current(intent.inputs):
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
    claim = None
    if intent.queue_revision is None:
        claim = store.claim_received_turn(
            intent.session_id,
            intent.turn_id,
            draft_revision=intent.inputs.draft_revision,
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
                terminal_callback=None,
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



def project_received_input(self, record) -> None:
    """Project a domain CAS into only the original attached composer."""
    intent = record.received_intent
    view = self.view
    if (
        intent is None
        or self._attached_generation != intent.view_attachment_generation
    ):
        return
    project = getattr(view, "_project_console_received_input", None)
    if callable(project):
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
    if intent is not None:
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
            if store.commit_session_input_draft(intent.inputs):
                self._project_received_input(record)
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
