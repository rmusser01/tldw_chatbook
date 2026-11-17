"""Body-free recovery displays; executable checkpoints remain with the backend."""

from __future__ import annotations

from dataclasses import dataclass

from .console_chat_models import (
    ConsoleDispatchRecoveryAction,
    ConsoleDispatchRecoveryActionId,
    ConsoleDispatchRecoveryKind,
    ConsoleDispatchRecoveryState,
)
from .console_chat_store import (
    ConsoleSettingsComponent,
    ConsoleSettingsPersistenceFailure,
    ConsoleSettingsPolicyFailureLabel,
)
from .console_execution_projection_values import (
    _boolean,
    _count,
    _enum,
    _enum_record,
    _integer,
    _optional,
    _record,
    _text,
    _tuple,
    decode_continuation_projection,
    encode_continuation_projection,
)
from .console_execution_protocol import ConsoleProtocolError
from .console_trace_call_recovery import (
    TraceCallProviderContact,
    TraceProviderRejection,
)
from .console_turn_preparation import ContextCompactionHold, ConsolePreparationPauseKind
from ..UI.Console_Modules.provider_continuation_recovery import (
    ProviderContinuationRecoveryState,
    TraceCallRecoveryState,
)


@dataclass(frozen=True, slots=True)
class ConsoleSettingsFailureDisplay:
    component: ConsoleSettingsComponent
    revision: int
    policy_failure_label: ConsoleSettingsPolicyFailureLabel | None

    def __post_init__(self) -> None:
        context = self.component is ConsoleSettingsComponent.CONTEXT_POLICY
        if context != (self.policy_failure_label is not None):
            raise ValueError("Settings failure copy does not match its component")


def _failure(value, decode):
    return _record(
        value,
        ConsoleSettingsFailureDisplay,
        {
            "component": lambda v, d: _enum_record(v, ConsoleSettingsComponent, d),
            "revision": _count,
            "policy_failure_label": _optional(
                lambda v, d: _enum_record(v, ConsoleSettingsPolicyFailureLabel, d)
            ),
        },
        decode,
    )


def _action(value, decode):
    return _record(
        value,
        ConsoleDispatchRecoveryAction,
        {
            "action_id": lambda v, d: _enum_record(
                v, ConsoleDispatchRecoveryActionId, d
            ),
            "label": _text,
            "enabled": _boolean,
            "disabled_reason": _text,
        },
        decode,
    )


def _dispatch(value, decode):
    # Deliberately exclude checkpoint, queue/preparation handles and all bodies.
    return _record(
        value,
        ConsoleDispatchRecoveryState,
        {
            "kind": lambda v, d: _enum_record(v, ConsoleDispatchRecoveryKind, d),
            "assistant_message_id": _text,
            "conversation_id": _text,
            "visible_copy": _text,
            "actions": _tuple(_action),
            "warning": _text,
            "error_code": _optional(_text),
            "in_flight": _boolean,
            "runtime_active": _boolean,
            "recovery_needed": _boolean,
        },
        decode,
    )


def _rejection(value, decode):
    return _record(
        value,
        TraceProviderRejection,
        {
            "provider": _text,
            "status_code": _integer,
            "meaning": _text,
        },
        decode,
    )


def _contact(value, decode):
    return _record(
        value,
        TraceCallProviderContact,
        {
            "contacted": _enum("yes", "no", "unknown"),
            "rejection": _optional(_rejection),
            "capture_unconfirmed": _boolean,
        },
        decode,
    )


def _hold(value, decode):
    return _record(
        value,
        ContextCompactionHold,
        {
            "session_id": _text,
            "used_tokens": _integer,
            "trigger_tokens": _integer,
            "budget_tokens": _integer,
            "estimated": _boolean,
        },
        decode,
    )


def _trace(value, decode):
    return _record(
        value,
        TraceCallRecoveryState,
        {
            "preparation_id": _text,
            "temporary_capture": _boolean,
            "context_hold": _optional(_hold),
            "provenance": _boolean,
            "library_reason": _text,
            "resend_pause": _optional(
                lambda v, d: _enum_record(v, ConsolePreparationPauseKind, d)
            ),
            "provider_contact": _optional(_contact),
        },
        decode,
    )


def _continuation(value, decode):
    return (
        decode_continuation_projection(value)
        if decode
        else encode_continuation_projection(value)
    )


@dataclass(frozen=True, slots=True)
class ConsoleRecoveryDisplay:
    dispatch: ConsoleDispatchRecoveryState | None
    continuation: ProviderContinuationRecoveryState | None
    trace: TraceCallRecoveryState | None
    settings_failures: tuple[ConsoleSettingsFailureDisplay, ...]


def recovery_display(value, decode):
    result = _record(
        value,
        ConsoleRecoveryDisplay,
        {
            "dispatch": _optional(_dispatch),
            "continuation": _optional(_continuation),
            "trace": _optional(_trace),
            "settings_failures": _tuple(_failure),
        },
        decode,
    )
    failures = result.settings_failures if decode else value.settings_failures
    if len({row.component for row in failures}) != len(failures):
        raise ConsoleProtocolError("duplicate_settings_failure_component")
    return result


def capture_recovery_display(controller, session) -> ConsoleRecoveryDisplay:
    """Use the original resident recovery readers and their display derivations."""
    from . import console_predispatch_block as block
    from . import console_turn_preparation as preparation_module
    from ..UI.Console_Modules import provider_continuation_recovery as display
    from .console_domain_services import _definition_current

    if any(
        getattr(module, name) is not record[0] or not _definition_current(record)
        for module, name, record in _DISPLAY_READERS
    ):
        raise RuntimeError("Console recovery display source changed")
    dispatch = block.presented_owner(
        controller.store.dispatch_recovery_for_presentation(session.id)
    )
    continuation = display.provider_continuation_recovery_state(
        controller.provider_continuation_recovery_message(),
        replay_available=controller.provider_continuation_replay_available(),
    )
    preparation = controller.trace_call_recovery_preparation()
    options = {}
    if preparation is not None:
        if preparation.pause_kind is ConsolePreparationPauseKind.RETRIEVAL:
            outcome = controller.preparation_outcome(preparation.preparation_id)
            options["library_reason"] = preparation_module.library_pause_copy(
                getattr(outcome, "error_code", None)
            )
        else:
            options["provider_contact"] = controller.trace_call_provider_contact(
                preparation.preparation_id
            )
    trace = display.trace_call_recovery_state(
        preparation,
        context_hold=(
            controller.context_compaction_hold(preparation.preparation_id)
            if preparation is not None
            else None
        ),
        **options,
    )
    failures = []
    for component, failure in session.settings_persistence_failures.items():
        if (
            type(failure) is not ConsoleSettingsPersistenceFailure
            or component is not failure.component
        ):
            raise ConsoleProtocolError("invalid_original_settings_failure")
        failures.append(
            ConsoleSettingsFailureDisplay(
                component, failure.revision, failure.policy_failure_label
            )
        )
    return ConsoleRecoveryDisplay(dispatch, continuation, trace, tuple(failures))


from .console_domain_services import _definition
from . import console_predispatch_block as _block
from . import console_turn_preparation as _preparation
from ..UI.Console_Modules import provider_continuation_recovery as _display

_DISPLAY_READERS = tuple(
    (module, name, _definition(getattr(module, name)))
    for module, name in (
        (_block, "presented_owner"),
        (_block, "present_predispatch_block"),
        (_preparation, "library_pause_copy"),
        (_display, "provider_continuation_recovery_state"),
        (_display, "trace_call_recovery_state"),
    )
)
