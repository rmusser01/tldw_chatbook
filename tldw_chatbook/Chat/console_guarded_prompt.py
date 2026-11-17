"""The original prompt transaction under one retained settings capture.

Only its fixed System mutation and compensating replacement may enter native
settings work. Library acknowledgement belongs to this same actual Task.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from inspect import CORO_CLOSED, getattr_static, getcoroutinestate
from types import FunctionType
from typing import Any

from .console_domain_services import _definition, _definition_current
from .console_parent_settings import (
    ParentSettingsCapture,
    ParentSettingsDeferred,
    ParentSettingsOriginChanged,
    ParentSettingsSourceChanged,
    due_settings_phase,
    issue_parent_settings_capture,
    require_parent_settings_capture,
    settle_parent_action_defaults,
)


@dataclass(slots=True)
class _GuardedTransaction:
    transaction: object
    settings_state: tuple
    phase: str = "draft"
    first_operation: object = None
    draft_write: object = None
    mutation_error: BaseException | None = None
    settled: bool = False
    completed: bool = False
    cancelled: bool = False
    result: bool | None = None


class _GuardedWriteAborted(Exception):
    """The exact first durability Task was canceled before any body entry."""


def raise_selected_refusal(error: BaseException) -> None:
    """Propagate selected refusals through original best-effort adapters.

    Args:
        error: The original error; ordinary legacy errors remain best effort.
    """
    from .console_control_settings import (
        ConsoleDefaultConvergenceRequired,
        ConsoleSettingsPending,
    )
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    if isinstance(
        error,
        (
            ParentSettingsDeferred,
            ParentSettingsOriginChanged,
            ParentSettingsSourceChanged,
            ConsoleDefaultConvergenceRequired,
            ConsoleSettingsPending,
            RecoveryRequired,
        ),
    ):
        raise error


def _require_callbacks(owner, screen):
    from tldw_chatbook.UI.Console_Modules import prompts, wiring

    binding = wiring._selected_guarded_prompt_bindings.get(owner)
    original, code = wiring._selected_prompt_wiring_definition
    if (
        type(owner) is not prompts.ConsolePromptsController
        or owner._screen is not screen
        or screen._prompts is not owner
        or owner.app_instance is not screen.app_instance
        or binding is None
        or binding[0]() is not screen
        or wiring.build_console_controllers is not original
        or original.__code__ is not code
    ):
        raise ParentSettingsSourceChanged("Guarded prompt construction changed")
    for name, codes in wiring._selected_guarded_prompt_accessor_codes:
        callback = getattr_static(owner, name, None)
        registered = next((ref() for edge, ref in binding[1] if edge == name), None)
        if (
            callback is not registered
            or type(callback) is not FunctionType
            or callback.__code__ not in codes
            or callback.__globals__ is not vars(wiring)
            or callback.__defaults__ is not None
            or callback.__kwdefaults__ is not None
            or callback.__closure__ is None
            or len(callback.__closure__) != 1
            or callback.__closure__[0].cell_contents is not screen
        ):
            raise ParentSettingsSourceChanged("Guarded prompt callback changed")
    for name, function, body in prompts._selected_guarded_prompt_helpers:
        if getattr_static(owner, name) is not function or function.__code__ is not body:
            raise ParentSettingsSourceChanged(
                "Guarded prompt transaction definition changed"
            )
    for name, record in _PROMPT_PROPERTIES:
        installed = getattr_static(owner, name)
        if (
            type(installed) is not property
            or installed.fget is not record[0]
            or not _definition_current(record)
        ):
            raise ParentSettingsSourceChanged(
                "Guarded prompt callback property changed"
            )


def _controls(owner):
    from tldw_chatbook.UI.Console_Modules import wiring

    _require_definitions()
    binding = wiring._selected_guarded_prompt_bindings.get(owner)
    screen = (
        binding[0]() if binding is not None else getattr_static(owner, "_screen", None)
    )
    app = getattr_static(screen, "app_instance", None) if screen is not None else None
    runtime = getattr_static(app, "console_runtime", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    _require_callbacks(owner, screen)
    controls._require_graph()
    if controls._domain.presentation._view is not screen or runtime.view is not screen:
        raise ParentSettingsOriginChanged("Guarded prompt view changed")
    return controls


def queue_guarded_prompt(
    owner: ConsolePromptsController,
    application: Any,
    snapshot: Any,
    *,
    claim: tuple[Any, Any] | None = None,
) -> ParentSettingsCapture | None:
    """Admit the whole modal or Library action.

    Args:
        owner: The original constructed prompt controller.
        application: The original guarded application.
        snapshot: The original composer checkpoint, when required.
        claim: Original Library handoff store and claim, if present.

    Returns:
        The retained selected capture, or None for the legacy Boolean route.

    Raises:
        ParentSettingsSourceChanged: Original construction or source changed.
        ParentSettingsOriginChanged: The captured view or target changed.
    """
    try:
        controls = _controls(owner)
    except BaseException:
        if claim is not None:
            _require_handoffs(claim[0])
            claim[0].release_prompt_claim(claim[1])
        raise
    if controls is None:
        return None
    capture = None
    try:
        capture = issue_parent_settings_capture(
            controls,
            "guarded-prompt",
            (owner, application, snapshot, claim),
        )
        controls._queue_session_capture(capture, capture)
    except BaseException:
        if capture is None:
            if claim is not None:
                _require_handoffs(claim[0])
                claim[0].release_prompt_claim(claim[1])
        elif (
            capture.operation_task is None
            and capture in controls._parent_settings_captures
        ):
            from .console_parent_settings import finish_parent_settings_capture

            finish_parent_settings_capture(capture)
        raise
    capture.applied.add_done_callback(
        lambda result: None if result.cancelled() else result.exception()
    )
    return capture


async def join_guarded_prompt(capture: ParentSettingsCapture) -> Any:
    """Join actual completion before preserving observer cancellation.

    Args:
        capture: The exact admitted action, including its Library claim.

    Returns:
        The original completed action result.

    Raises:
        asyncio.CancelledError: The observer was canceled, after settlement.
    """
    cancelled = False
    while not capture.applied.done():
        try:
            await asyncio.shield(capture.applied)
        except asyncio.CancelledError:
            cancelled = True
        except BaseException:
            break
    value = capture.applied.result()
    if cancelled:
        raise asyncio.CancelledError
    return value


async def capture_prompt_target(owner: ConsolePromptsController) -> Any:
    """Retain original target capture before either async search entrance.

    Args:
        owner: The original prompt controller for the current attachment.

    Returns:
        The original target or its original no-target result.

    Raises:
        ParentSettingsSourceChanged: Original definitions or source changed.
        ParentSettingsOriginChanged: The exact created or active target changed.
    """
    from .console_control_settings import ConsoleDefaultConvergenceRequired

    controls = _controls(owner)
    if controls is None:
        return owner._capture_prompt_replace_target()
    if controls._store.active_session_id is None:
        screen, store, sources = (
            owner._screen,
            controls._store,
            controls._domain.sources,
        )
        stamp, revision = (
            controls._domain.presentation._stamp(),
            controls.source_revision,
        )
        _require_target_ensure(screen)
        try:
            return owner._capture_prompt_replace_target()
        except ConsoleDefaultConvergenceRequired as due:
            # This is the original ensure's result, never an active-session
            # lookup used to manufacture a creation witness after a callback.
            witness = getattr(due, "created_session", None)
            if type(witness) is not tuple or len(witness) != 8:
                raise
            (
                controller,
                actual_store,
                session,
                incarnation,
                settings,
                settings_revision,
                binding,
                task,
            ) = witness
            controls._require_graph()
            controls._domain.require_current(sources)
            _require_callbacks(owner, screen)
            _require_target_ensure(screen)
            if controls.source_revision != revision:
                raise ParentSettingsDeferred(
                    "Prompt creation source was fenced"
                ) from due
            if (
                controller is not screen._session
                or actual_store is not store
                or task is not asyncio.current_task()
                or store._sessions.get(session.id) is not session
                or store.active_session_id != session.id
                or session.incarnation_id != incarnation
                or session.settings is not settings
                or session.settings_revision != settings_revision
                or session.conversation_binding_revision != binding
                or not controls._domain.presentation._current_visit(stamp)
            ):
                raise ParentSettingsOriginChanged(
                    "Prompt creation result changed before continuation"
                ) from due
    capture = issue_parent_settings_capture(controls, "prompt-target", (owner,))
    controls._queue_session_capture(capture, capture)
    return await join_guarded_prompt(capture)


def _require_target_ensure(screen):
    from tldw_chatbook.UI.Console_Modules import session, wiring

    if (
        type(screen._session) is not session.ConsoleSessionController
        or screen._session._screen is not screen
    ):
        raise ParentSettingsSourceChanged("Prompt target ensure controller changed")
    binding = wiring._selected_guarded_prompt_bindings.get(screen._prompts)
    if binding is None or binding[2]() is not screen._session:
        raise ParentSettingsSourceChanged("Prompt target creation registration changed")
    for name, codes in wiring._selected_guarded_creation_accessor_codes:
        callback = getattr_static(screen._session, name, None)
        registered = next(
            (reference() for edge, reference in binding[3] if edge == name), None
        )
        if (
            callback is not registered
            or type(callback) is not FunctionType
            or callback.__code__ not in codes
            or callback.__globals__ is not vars(wiring)
            or callback.__defaults__ is not None
            or callback.__kwdefaults__ is not None
            or callback.__closure__ is None
            or len(callback.__closure__) != 1
            or callback.__closure__[0].cell_contents is not screen
        ):
            raise ParentSettingsSourceChanged("Prompt target creation callback changed")
    for name, record in _SCREEN_DEFINITIONS:
        if getattr_static(screen, name) is not record[0] or not _definition_current(
            record
        ):
            raise ParentSettingsSourceChanged("Prompt target screen edge changed")
    for name, record in _CREATION_PROPERTIES:
        installed = getattr_static(screen._session, name)
        if (
            type(installed) is not property
            or installed.fget is not record[0]
            or not _definition_current(record)
        ):
            raise ParentSettingsSourceChanged("Prompt target creation property changed")
    for name, record in _CREATION_DEFINITIONS:
        if getattr_static(screen._session, name) is not record[
            0
        ] or not _definition_current(record):
            raise ParentSettingsSourceChanged(
                "Prompt target creation definition changed"
            )
    for name, function, code in session._selected_due_ensure_definitions:
        if (
            getattr_static(screen._session, name) is not function
            or function.__code__ is not code
        ):
            raise ParentSettingsSourceChanged("Prompt target ensure definition changed")
    for name, record in session._CONSOLE_DEFAULT_CONVERGENCE_READERS:
        if getattr_static(screen._session, name) is not record[
            0
        ] or not _definition_current(record):
            raise ParentSettingsSourceChanged(
                "Prompt target convergence definition changed"
            )


async def run_prompt_target(capture: ParentSettingsCapture) -> Any:
    """Run original target capture under its registered action Task.

    Args:
        capture: The fixed prompt-target capture.

    Returns:
        The original target or its original no-target result.
    """
    from .console_control_settings import ConsoleDefaultConvergenceRequired

    owner = _owner(capture)
    _require_target_ensure(owner._screen)
    try:
        return owner._capture_prompt_replace_target()
    except ConsoleDefaultConvergenceRequired:
        request = await settle_parent_action_defaults(capture)
        with due_settings_phase(request):
            return _owner(capture)._capture_prompt_replace_target()


def _require_handoffs(handoffs):
    if type(handoffs) is not PendingHandoffStore:
        raise ParentSettingsSourceChanged("Guarded prompt handoff owner changed")
    for name, record in _HANDOFF_DEFINITIONS:
        if getattr_static(handoffs, name) is not record[0] or not _definition_current(
            record
        ):
            raise ParentSettingsSourceChanged(
                "Guarded prompt handoff definition changed"
            )


def _finish_library_claim(capture, *, acknowledge):
    claim_info = capture.checked_config[3]
    if claim_info is None or capture.guarded_claim_settled:
        return
    handoffs, claim = claim_info
    _require_handoffs(handoffs)
    if acknowledge:
        handoffs.acknowledge(claim)
    else:
        handoffs.release_prompt_claim(claim)
    capture.guarded_claim_settled = True


def finish_guarded_capture(capture: ParentSettingsCapture) -> None:
    """Release the original Library claim only for a settled no-entry action.

    Args:
        capture: The exact capture whose actual operation has settled.

    Raises:
        ParentSettingsSourceChanged: The actual operation is still live.
    """
    _require_definitions()
    if capture.operation_task is not None and not capture.operation_task.done():
        raise ParentSettingsSourceChanged("Guarded prompt actual Task is still live")
    if capture.guarded_transaction is None:
        _finish_library_claim(capture, acknowledge=False)


def _owner(capture, *, presentation=True):
    _require_definitions()
    require_parent_settings_capture(capture, presentation=presentation)
    if capture.operation_task is not asyncio.current_task() or not capture.started:
        raise ParentSettingsSourceChanged("Guarded prompt Task changed")
    owner = capture.checked_config[0]
    screen = capture.stamp[1]
    _require_callbacks(owner, screen)
    for name, record in _SCREEN_DEFINITIONS:
        if getattr_static(screen, name) is not record[0] or not _definition_current(
            record
        ):
            raise ParentSettingsSourceChanged("Guarded prompt screen edge changed")
    if owner._chat_store_accessor() is not capture.controls._store:
        raise ParentSettingsSourceChanged("Guarded prompt store changed")
    for name, record in _STORE_DEFINITIONS:
        if getattr_static(capture.controls._store, name) is not record[
            0
        ] or not _definition_current(record):
            raise ParentSettingsSourceChanged("Guarded prompt store definition changed")
    if presentation:
        composer = owner._composer_accessor()
        if composer is not None:
            _require_composer(composer, screen)
    return owner


def _require_composer(composer, screen):
    if type(composer) is not ConsoleComposerBar:
        raise ParentSettingsSourceChanged("Guarded prompt composer changed")
    for name, record in _COMPOSER_DEFINITIONS:
        if getattr_static(composer, name) is not record[0] or not _definition_current(
            record
        ):
            raise ParentSettingsSourceChanged(
                "Guarded prompt composer definition changed"
            )
    observer = composer._authored_draft_observer
    if (
        getattr(observer, "__self__", None) is not screen
        or getattr(observer, "__func__", None) is not _AUTHORED_DEFINITION[0]
        or not _definition_current(_AUTHORED_DEFINITION)
    ):
        raise ParentSettingsSourceChanged("Guarded prompt authored observer changed")


def _settings_state(session):
    return (
        session.settings,
        session.generation_settings_revision,
        session.settings_revision,
        session.context_policy_revision,
    )


def begin_guarded_draft_write(
    screen: ChatScreen,
    store: ConsoleChatStore,
    session_id: str,
    draft: str,
    authored_token: object | None,
) -> tuple | None:
    """Record the next original draft mutation of an actual guarded phase.

    Args:
        screen: The original attached Screen.
        store: The actual Console store.
        session_id: The original mutation target.
        draft: Text passed to the original setter.
        authored_token: Original authored identity, when supplied.

    Returns:
        The exact one-use expected-result record, or None outside this phase.

    Raises:
        ParentSettingsOriginChanged: An intervening draft changed ownership.
    """
    runtime = getattr_static(screen.app_instance, "console_runtime", None)
    controls = getattr(
        getattr(runtime, "_execution_service", None), "_local_controls", None
    )
    if controls is None:
        return None
    task = asyncio.current_task()
    capture = next(
        (
            item
            for item in controls._parent_settings_captures
            if item.purpose == "guarded-prompt"
            and item.operation_task is task
            and item.guarded_transaction is not None
        ),
        None,
    )
    if capture is None or capture.guarded_transaction.phase not in {
        "draft",
        "compensation-draft",
    }:
        return None
    _owner(capture, presentation=False)
    state = capture.guarded_transaction
    transaction = state.transaction
    with store._preparation_lock:
        session = store._sessions.get(session_id)
        if (
            store is not controls._store
            or session is not capture.session
            or state.draft_write is not None
            or transaction.witness_error is not None
            or transaction.draft_expected
            != (session.draft, session.draft_revision, session._draft_authored_token)
        ):
            error = ParentSettingsOriginChanged(
                "Guarded prompt draft callback changed its owner"
            )
            transaction.witness_error = error
            raise error
        expected = (
            draft,
            session.draft_revision
            + int(
                session.draft != draft
                or (
                    authored_token is not None
                    and session._draft_authored_token != authored_token
                )
            ),
            authored_token
            if authored_token is not None
            else session._draft_authored_token,
        )
    state.draft_write = (capture, session, expected)
    return state.draft_write


def finish_guarded_draft_write(record: tuple | None, returned_session: Any) -> None:
    """Consume the issued draft result without adopting callback changes.

    Args:
        record: Exact begin_guarded_draft_write record, or None.
        returned_session: Actual session returned by the original setter.

    Raises:
        ParentSettingsSourceChanged: The record was copied or already consumed.
    """
    if record is None:
        return
    capture, session, expected = record
    _owner(capture, presentation=False)
    state = capture.guarded_transaction
    if state.draft_write is not record:
        raise ParentSettingsSourceChanged(
            "Guarded prompt draft result was not issued or was consumed"
        )
    state.draft_write = None
    transaction = state.transaction
    with capture.controls._store._preparation_lock:
        if (
            returned_session is not session
            or capture.controls._store._sessions.get(session.id) is not session
            or expected
            != (session.draft, session.draft_revision, session._draft_authored_token)
        ):
            transaction.witness_error = ParentSettingsOriginChanged(
                "Guarded prompt draft callback changed resulting state"
            )
        transaction.draft_expected = expected


def _require_transaction_state(capture):
    visible = _require_transaction_context(capture)
    state = capture.guarded_transaction
    if (
        state.settings_state[0] is not capture.session.settings
        or state.settings_state[1:] != _settings_state(capture.session)[1:]
    ):
        raise ParentSettingsOriginChanged("Guarded prompt resulting state changed")
    return visible


def _require_transaction_context(capture):
    state = capture.guarded_transaction
    transaction = state.transaction
    _owner(capture, presentation=False)
    session = capture.session
    if (
        transaction.store is not capture.controls._store
        or transaction.session_id != session.id
        or transaction.draft_expected
        != (session.draft, session.draft_revision, session._draft_authored_token)
        or state.draft_write is not None
        or transaction.witness_error is not None
    ):
        raise ParentSettingsOriginChanged("Guarded prompt resulting state changed")
    composer = transaction.composer
    _require_composer(composer, capture.stamp[1])
    # A different visit may compensate the unchanged old backend, but never
    # restores its checkpoint into the new view. Same-composer edits also win.
    visible = (
        capture.port._current_visit(capture.stamp)
        and capture.controls._store.active_session_id == capture.session_id
    )
    if visible and (
        capture.checked_config[0]._composer_accessor() is not composer
        or composer.capture_transaction_checkpoint() != transaction.composer_result
    ):
        raise ParentSettingsOriginChanged("Guarded prompt composer was edited")
    return visible


def require_guarded_operation(
    capture: ParentSettingsCapture, mutation: tuple | None
) -> bool:
    """Validate only the sealed transaction's original settings phases.

    Args:
        capture: The actual registered guarded action.
        mutation: Its fixed component or compensating replacement operation.

    Returns:
        Whether this is the fixed compensation phase.

    Raises:
        ParentSettingsSourceChanged: The transaction or fixed phase changed.
        ParentSettingsOriginChanged: Transaction-owned resulting state changed.
    """
    _owner(capture, presentation=False)
    state = capture.guarded_transaction
    if state is None or mutation is None:
        raise ParentSettingsSourceChanged("Guarded prompt operation has no transaction")
    if state.phase == "system" and mutation == (
        "system_prompt",
        state.transaction.application.system_text,
    ):
        return False
    if (
        state.phase == "compensation"
        and state.settled
        and mutation[0] == "replacement"
        and mutation[1] is state.transaction.settings
    ):
        _require_transaction_state(capture)
        return True
    raise ParentSettingsSourceChanged("Guarded prompt operation phase changed")


def _settle_first_failure(capture):
    """Prove the first operation ended before admitting compensation."""
    state = capture.guarded_transaction
    operation = state.first_operation
    if operation is None:
        state.settled = True  # The draft-only synchronous phase issued no native work.
        return
    task = operation.actual_task
    closed = (
        operation.actual_coroutine is not None
        and getcoroutinestate(operation.actual_coroutine) == CORO_CLOSED
    )
    if (
        operation.authority_failed
        or not operation.finished.done()
        or not (
            (task is not None and task.done())
            or (task is None and closed and not operation.apply_started)
        )
        or (
            operation.native_entered
            and (operation.retirement is None or not operation.retirement.confirmed)
        )
    ):
        raise ParentSettingsSourceChanged(
            "Guarded prompt first operation settlement is unproven"
        )
    operation.current(presentation=False)
    if operation.owned_state is not None:
        state.settings_state = operation.owned_state
    # The failed operation's own barrier is settled independently from the
    # stricter compensation test. A newer draft must refuse rollback without
    # stranding an already-retired settings operation in the pending map.
    pending = getattr(operation.store, "_selected_settings_pending", {})
    if pending.get(operation.session.id) is operation:
        del pending[operation.session.id]
    operation.failed_settled = True
    state.settled = True


async def _retain_guarded_write(capture, operation):
    from .console_control_settings import retain_settings_operation

    task = asyncio.current_task()
    before = task.cancelling()
    try:
        return await retain_settings_operation(operation)
    except asyncio.CancelledError:
        # The original native/post operation is already joined by this helper.
        # Finish its transaction/ACK, then preserve the cancellation outcome.
        if operation.commit is None or not operation.retired:
            if (
                operation is capture.guarded_transaction.first_operation
                and operation.actual_task is not None
                and operation.actual_task.done()
                and operation.actual_task.cancelled()
                and not operation.apply_started
                and not operation.native_entered
                and operation.retirement is None
                and operation.actual_coroutine is not None
                and getcoroutinestate(operation.actual_coroutine) == CORO_CLOSED
                and operation.finished.done()
                and not operation.authority_failed
            ):
                capture.guarded_transaction.cancelled = True
                raise _GuardedWriteAborted(
                    "Guarded prompt write was canceled before entry"
                ) from operation.error
            raise
        return operation.commit
    finally:
        capture.guarded_transaction.cancelled |= task.cancelling() > before


def _guarded_result(capture, value):
    capture.guarded_transaction.result = value
    if capture.guarded_transaction.cancelled:
        raise asyncio.CancelledError
    return value


def _display_guarded_result(capture, persisted, error):
    owner = _owner(capture)
    _require_transaction_state(capture)
    transaction = capture.guarded_transaction.transaction
    if error is None:
        return owner._finish_guarded_prompt_application(
            transaction, persisted, selected=True
        )
    if transaction.application.apply_system:
        try:
            owner._sync_console_system_prompt_surfaces()
        except Exception as display_error:
            raise_selected_refusal(display_error)
    return owner._report_guarded_prompt_failure(transaction, error, selected=True)


def _advance_guarded_post_settings(capture, before, request):
    """Advance completed-post state only through this due call's actual writes."""
    from .console_parent_settings import _due_settings_read_current

    _due_settings_read_current(request)
    _owner(capture)
    _require_transaction_context(capture)
    state = capture.guarded_transaction
    if (
        not state.completed
        or state.settings_state is not before
        or request.source is not capture
        or not request.active
        or request.consumed
    ):
        raise ParentSettingsSourceChanged("Guarded post due phase changed")
    from .console_parent_settings import _advance_parent_post_settings

    state.settings_state = _advance_parent_post_settings(capture, before, request)


async def _finish_guarded_display(capture, persisted=True, error=None):
    from .console_control_settings import ConsoleDefaultConvergenceRequired

    try:
        return _display_guarded_result(capture, persisted, error)
    except ConsoleDefaultConvergenceRequired:
        _require_transaction_state(capture)
        before = capture.guarded_transaction.settings_state
        request = await settle_parent_action_defaults(capture)
        with due_settings_phase(request):
            if capture.guarded_transaction.completed:
                _advance_guarded_post_settings(capture, before, request)
            return _display_guarded_result(capture, persisted, error)


async def _compensate(capture, error):
    from .console_control_settings import PendingSettingsApply

    state = capture.guarded_transaction
    transaction = state.transaction
    _owner(capture, presentation=False)
    with transaction.store._preparation_lock:
        _settle_first_failure(capture)
        visible = _require_transaction_state(capture)
        state.phase = "compensation-draft"
    if visible:
        transaction.composer.rollback_transaction(transaction.checkpoint)
    record = begin_guarded_draft_write(
        capture.stamp[1],
        transaction.store,
        transaction.session_id,
        transaction.original_draft,
        None,
    )
    returned = None
    try:
        returned = transaction.store.set_session_draft(
            transaction.session_id, transaction.original_draft
        )
    finally:
        finish_guarded_draft_write(record, returned)
    if visible:
        transaction.composer_result = (
            transaction.composer.capture_transaction_checkpoint()
        )
    _require_transaction_state(capture)
    state.phase = "compensation"
    if transaction.application.apply_system:
        operation = PendingSettingsApply(
            capture.controls,
            capture,
            capture,
            mutation=("replacement", transaction.settings),
        )
        await _retain_guarded_write(capture, operation)
        state.settings_state = operation.owned_state
    state.completed = True
    if (
        visible
        and capture.port._current_visit(capture.stamp)
        and capture.controls._store.active_session_id == capture.session_id
    ):
        return await _finish_guarded_display(capture, error=error)
    return False


def _library_preflight(capture, owner):
    handoffs, claim = capture.checked_config[3]
    _require_handoffs(handoffs)
    if (
        owner.app_instance.pending_handoffs is not handoffs
        or not handoffs.is_current_claim(claim)
    ):
        raise ParentSettingsOriginChanged("Library prompt claim changed")
    if owner._console_setup_blocked_reason():
        owner.app_instance.notify(
            owner._LIBRARY_PROMPT_INSERT_BLOCKED_COPY, severity="warning"
        )
        return "blocked"
    owner._sync_console_session_draft()
    return "ready" if owner._composer_accessor() is not None else "missing"


async def run_guarded_prompt(capture: ParentSettingsCapture) -> bool:
    """Complete the original transaction and terminal Library acknowledgement.

    Args:
        capture: The exact registered guarded prompt action.

    Returns:
        The original Boolean application result.

    Raises:
        asyncio.CancelledError: After required transaction and claim settlement.
    """
    from .console_control_settings import (
        ConsoleDefaultConvergenceRequired,
        PendingSettingsApply,
    )

    owner = _owner(capture)
    _owner_arg, application, snapshot, claim_info = capture.checked_config
    acknowledge = False
    try:
        if claim_info is not None:
            try:
                ready = _library_preflight(capture, owner)
            except ConsoleDefaultConvergenceRequired:
                request = await settle_parent_action_defaults(capture)
                with due_settings_phase(request):
                    ready = _library_preflight(capture, owner)
            if ready != "ready":
                acknowledge = ready == "blocked"
                return False
        with capture.controls._store._preparation_lock:
            transaction = owner._prepare_guarded_prompt_application(
                application, snapshot
            )
            if transaction is None:
                acknowledge = True
                return False
            state = capture.guarded_transaction = _GuardedTransaction(
                transaction, _settings_state(capture.session)
            )
            transaction.composer_result = transaction.checkpoint
            transaction.draft_expected = (
                capture.session.draft,
                capture.session.draft_revision,
                capture.session._draft_authored_token,
            )
            _require_transaction_state(capture)
        try:
            owner._mutate_guarded_prompt_draft(transaction, record_owned=True)
            _require_transaction_state(capture)
        except Exception as error:
            state.mutation_error = error
        persisted = True
        try:
            if state.mutation_error is not None:
                raise state.mutation_error
            if application.apply_system:
                state.phase = "system"
                operation = PendingSettingsApply(
                    capture.controls,
                    capture,
                    capture,
                    mutation=("system_prompt", application.system_text),
                )
                state.first_operation = operation
                _session, persisted = await _retain_guarded_write(capture, operation)
                state.settings_state = operation.owned_state
            if not transaction.composer_changed:
                # UI loss after the completed System mutation is a delivery
                # refusal, not an exception in the original mutation itself.
                try:
                    _owner(capture)
                    _require_transaction_state(capture)
                except ParentSettingsOriginChanged:
                    acknowledge = True
                    return _guarded_result(capture, False)
                try:
                    transaction.composer.invalidate_improvement_undo()
                finally:
                    transaction.composer_result = (
                        transaction.composer.capture_transaction_checkpoint()
                    )
        except Exception as error:
            state.mutation_error = error
            if isinstance(error, ParentSettingsOriginChanged):
                _settle_first_failure(capture)
                acknowledge = True
                if (
                    capture.port._current_visit(capture.stamp)
                    and capture.controls._store.active_session_id == capture.session_id
                ):
                    return _guarded_result(
                        capture,
                        owner._report_guarded_prompt_failure(
                            transaction, error, restored=False, selected=True
                        ),
                    )
                return _guarded_result(capture, False)
            raise_selected_refusal(error)
            try:
                result = await _compensate(capture, error)
            except ParentSettingsOriginChanged:
                # A newer draft/settings owner wins. Refusal is not rollback.
                if (
                    capture.port._current_visit(capture.stamp)
                    and capture.controls._store.active_session_id == capture.session_id
                ):
                    result = owner._report_guarded_prompt_failure(
                        transaction, error, restored=False, selected=True
                    )
                else:
                    result = False
            acknowledge = True
            return _guarded_result(capture, result)
        acknowledge = True
        state.completed = True
        try:
            _owner(capture)
            _require_transaction_state(capture)
        except ParentSettingsOriginChanged:
            return _guarded_result(capture, False)
        return _guarded_result(
            capture, await _finish_guarded_display(capture, persisted)
        )
    finally:
        if claim_info is not None:
            if acknowledge or (
                capture.guarded_transaction is not None
                and capture.guarded_transaction.completed
            ):
                _finish_library_claim(capture, acknowledge=True)
            elif capture.guarded_transaction is None:
                _finish_library_claim(capture, acknowledge=False)


def _require_definitions():
    if any(
        globals().get(name) is not record[0] or not _definition_current(record)
        for name, record in _DEFINITIONS
    ):
        raise ParentSettingsSourceChanged("Guarded prompt defining source changed")


from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleComposerBar
from tldw_chatbook.UI.Console_Modules.prompts import ConsolePromptsController
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
    _publish_console_authored_draft,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Navigation.pending_handoff_store import PendingHandoffStore
from .console_chat_store import ConsoleChatStore

_PROMPT_PROPERTIES = tuple(
    (name, _definition(getattr_static(ConsolePromptsController, name).fget))
    for name in (
        "_console_composer_or_none",
        "_ensure_console_chat_store",
        "_sync_console_session_draft",
        "_ensure_active_console_session_settings",
        "_console_setup_blocked_reason",
        "_focus_console_composer_if_needed",
        "_sync_console_system_prompt_surfaces",
        "_sync_console_command_popup",
    )
)
_STORE_DEFINITIONS = tuple(
    (name, _definition(getattr(ConsoleChatStore, name)))
    for name in (
        "set_session_draft",
        "session_settings",
        "session_input_snapshot",
    )
)
_AUTHORED_DEFINITION = _definition(_publish_console_authored_draft)
_CREATION_PROPERTIES = tuple(
    (name, _definition(getattr_static(ConsoleSessionController, name).fget))
    for name in (
        "_ensure_console_chat_store",
        "_workspace_initial_session_title",
        "_provider_readiness_app_config",
    )
)
_CREATION_DEFINITIONS = (
    (
        "_blank_console_session_settings",
        _definition(ConsoleSessionController._blank_console_session_settings),
    ),
)
_HANDOFF_DEFINITIONS = tuple(
    (name, _definition(getattr(PendingHandoffStore, name)))
    for name in (
        "is_current_claim",
        "acknowledge",
        "release_prompt_claim",
    )
)
_SCREEN_DEFINITIONS = tuple(
    (name, _definition(getattr(ChatScreen, name)))
    for name in (
        "_console_composer_or_none",
        "_ensure_console_chat_store",
        "_publish_console_authored_draft",
        "_sync_console_chat_core_state",
        "_sync_console_rail_system_line",
        "_sync_console_settings_summary",
        "_sync_console_control_bar",
        "_sync_console_command_popup",
        "_focus_console_composer_if_needed",
        "_console_setup_blocked_reason",
        "_console_ordered_resume_pending",
    )
)

_COMPOSER_DEFINITIONS = tuple(
    (name, _definition(getattr(ConsoleComposerBar, name)))
    for name in (
        "capture_transaction_checkpoint",
        "capture_draft_snapshot",
        "rollback_transaction",
        "replace_snapshot_as_paste",
        "insert_text_as_paste",
        "move_cursor_end",
        "draft_text",
        "invalidate_improvement_undo",
        "_sync_hidden_input",
    )
)
_DEFINITIONS = tuple(
    (name, _definition(value))
    for name, value in tuple(globals().items())
    if type(value) is FunctionType and value.__module__ == __name__
)
