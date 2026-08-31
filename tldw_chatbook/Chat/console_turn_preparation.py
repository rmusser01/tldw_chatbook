"""Pure in-memory state for one admitted Console user turn."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
from tldw_chatbook.Chat.console_dispatch_checkpoint import (
    ConsoleDispatchCheckpointValidationError,
    dump_console_resolved_destination_json,
    dump_console_turn_library_authority_json,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    SavedRevisionTraceProvenance,
    TraceProvenancePersistenceError,
    admit_message_provenance,
    request_route_provenance,
    trace_provenance_admission_transaction,
)
from tldw_chatbook.Chat.console_trace_service import TraceCallPersistenceError

if TYPE_CHECKING:
    from tldw_chatbook.Chat.console_prepared_request import PreparedConsoleRequest
    from tldw_chatbook.Chat.console_semantic_revision import (
        SemanticRevisionCoordinator,
    )


_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z", re.ASCII)
CONSOLE_PREPARATION_DRAFT_MAX_BYTES = 4 * 1024 * 1024
CONSOLE_PREPARATION_TITLE_MAX_BYTES = 4096
CONSOLE_PREPARATION_ID_COLLECTION_MAX_ITEMS = 1024


class ConsoleTurnPreparationValidationError(ValueError):
    """A preparation value violated its strict bounded in-memory contract."""


class ConsoleTurnPreparationState(str, Enum):
    """Closed lifecycle states for one admitted Console turn."""

    PREPARING = "preparing"
    READY = "ready"
    COMMITTING = "committing"
    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"
    DISPATCHED = "dispatched"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    SETTLED = "settled"


class ConsolePreparationPauseKind(str, Enum):
    """Closed pre-dispatch reasons that require an explicit user decision."""

    RETRIEVAL = "retrieval"
    PERSISTENCE = "persistence"
    DESTINATION_CHANGED = "destination_changed"
    TRACE_PROVENANCE = "trace_provenance"
    TRACE_CALL = "trace_call"
    TEMPORARY_CAPTURE = "temporary_capture"


PAUSE_ACTIONS: Mapping[ConsolePreparationPauseKind, tuple[str, ...]] = MappingProxyType(
    {
        ConsolePreparationPauseKind.RETRIEVAL: ("retry", "bypass", "cancel"),
        ConsolePreparationPauseKind.PERSISTENCE: ("retry", "cancel"),
        ConsolePreparationPauseKind.DESTINATION_CHANGED: ("retry", "cancel"),
        ConsolePreparationPauseKind.TRACE_PROVENANCE: (
            "retry",
            "send_without_capture",
            "cancel",
        ),
        ConsolePreparationPauseKind.TRACE_CALL: (
            "retry",
            "send_without_capture",
            "cancel",
        ),
        ConsolePreparationPauseKind.TEMPORARY_CAPTURE: (
            "save_and_send",
            "send_without_capture",
            "cancel",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class ConsoleTurnPreparation:
    """All preparation-owned inputs for one immediate or queued turn."""

    preparation_id: str
    attempt_id: str
    session_id: str
    origin: Literal["manual", "queued"]
    queue_entry_id: str | None
    executed_draft: str
    execution_context: ConsoleTurnExecutionContext
    transient_user_message_id: str | None
    attachment_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    prefill_id: str | None
    queue_generation: int | None
    pre_send_title: str
    pre_send_conversation_id: str | None
    state: ConsoleTurnPreparationState
    pause_kind: ConsolePreparationPauseKind | None
    one_shot_bypass: bool
    ephemeral: bool
    one_shot_capture_off: bool = False
    capture_mode: ConsoleTraceCaptureMode = ConsoleTraceCaptureMode.CAPTURE_ON
    pii_redaction_enabled: bool = False
    pii_ruleset_revision_id: str | None = None

    def __post_init__(self) -> None:
        """Reject malformed, mutable, or internally inconsistent state."""
        _validate_preparation(self)


@dataclass(frozen=True, slots=True)
class ConsolePreparationTransition:
    """One expected-state compare-and-set request for a preparation."""

    preparation_id: str
    expected_state: ConsoleTurnPreparationState
    new_state: ConsoleTurnPreparationState
    pause_kind: ConsolePreparationPauseKind | None
    new_attempt_id: str | None

    def __post_init__(self) -> None:
        """Reject malformed transition values before state evaluation."""
        _validate_identifier(self.preparation_id, "transition preparation ID")
        if type(self.expected_state) is not ConsoleTurnPreparationState:
            _invalid("transition expected state")
        if type(self.new_state) is not ConsoleTurnPreparationState:
            _invalid("transition new state")
        if (
            self.pause_kind is not None
            and type(self.pause_kind) is not ConsolePreparationPauseKind
        ):
            _invalid("transition pause kind")
        if self.new_attempt_id is not None:
            _validate_identifier(self.new_attempt_id, "transition attempt ID")


def initial_preparation_state(
    auto_retrieve: ConsoleAutoRetrieve,
) -> ConsoleTurnPreparationState:
    """Return the first state after authority and destination resolution.

    ``Never`` enters ``ready`` directly, so it neither runs nor displays a Library
    preparation stage. Automatic enters ``preparing``.
    """
    if auto_retrieve is ConsoleAutoRetrieve.NEVER:
        return ConsoleTurnPreparationState.READY
    if auto_retrieve is ConsoleAutoRetrieve.AUTOMATIC:
        return ConsoleTurnPreparationState.PREPARING
    raise TypeError("auto_retrieve must be ConsoleAutoRetrieve")


def preparation_actions(preparation: ConsoleTurnPreparation) -> tuple[str, ...]:
    """Return the action data available in the current precommit state."""
    if not _preparation_is_valid(preparation):
        return ()
    if preparation.state is ConsoleTurnPreparationState.PAUSED:
        if preparation.pause_kind is None:
            return ()
        return PAUSE_ACTIONS.get(preparation.pause_kind, ())
    if preparation.pause_kind is not None:
        return ()
    if preparation.state in {
        ConsoleTurnPreparationState.PREPARING,
        ConsoleTurnPreparationState.READY,
    }:
        return ("cancel",)
    return ()


def apply_preparation_transition(
    preparation: ConsoleTurnPreparation,
    transition: ConsolePreparationTransition,
) -> ConsoleTurnPreparation:
    """Apply one legal expected-state transition, otherwise return unchanged.

    Returning the same immutable object for a stale, racing, repeated, or
    malformed transition makes action delivery idempotent without granting a
    caller any state beyond the frozen transition matrix.
    """
    if not _preparation_is_valid(preparation) or not _transition_is_valid(transition):
        return preparation
    if (
        transition.preparation_id != preparation.preparation_id
        or transition.expected_state is not preparation.state
        or not _transition_is_legal(preparation, transition)
    ):
        return preparation

    attempt_id = transition.new_attempt_id or preparation.attempt_id
    execution_context = preparation.execution_context
    if transition.new_attempt_id is not None:
        execution_context = _execution_context_with_attempt(
            execution_context,
            transition.new_attempt_id,
        )
    bypass = preparation.one_shot_bypass or (
        preparation.state is ConsoleTurnPreparationState.PAUSED
        and preparation.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
        and transition.new_state is ConsoleTurnPreparationState.READY
    )
    return replace(
        preparation,
        attempt_id=attempt_id,
        execution_context=execution_context,
        state=transition.new_state,
        pause_kind=(
            transition.pause_kind
            if transition.new_state is ConsoleTurnPreparationState.PAUSED
            else None
        ),
        one_shot_bypass=bypass,
        one_shot_capture_off=(
            False
            if transition.new_attempt_id is not None
            else preparation.one_shot_capture_off
        ),
        capture_mode=(
            ConsoleTraceCaptureMode.CAPTURE_ON
            if transition.new_attempt_id is not None
            and preparation.one_shot_capture_off
            else preparation.capture_mode
        ),
    )


def pause_for_trace_provenance_failure(
    preparation: ConsoleTurnPreparation,
    failure: TraceProvenancePersistenceError,
) -> ConsoleTurnPreparation:
    """Pause an interactive Capture-On admission without exposing failure content."""

    if not isinstance(failure, TraceProvenancePersistenceError):
        raise TypeError("failure must be TraceProvenancePersistenceError")
    if preparation.origin != "manual":
        raise failure
    return apply_preparation_transition(
        preparation,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=ConsoleTurnPreparationState.COMMITTING,
            new_state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.TRACE_PROVENANCE,
            new_attempt_id=None,
        ),
    )


def pause_for_trace_call_failure(
    preparation: ConsoleTurnPreparation,
    failure: TraceCallPersistenceError,
) -> ConsoleTurnPreparation:
    """Pause an interactive call whose durable boundary was not committed."""

    if not isinstance(failure, TraceCallPersistenceError):
        raise TypeError("failure must be TraceCallPersistenceError")
    if preparation.origin != "manual":
        raise failure
    return apply_preparation_transition(
        preparation,
        ConsolePreparationTransition(
            preparation_id=preparation.preparation_id,
            expected_state=preparation.state,
            new_state=ConsoleTurnPreparationState.PAUSED,
            pause_kind=ConsolePreparationPauseKind.TRACE_CALL,
            new_attempt_id=None,
        ),
    )


def admit_preparation_trace_provenance(
    preparation: ConsoleTurnPreparation,
    *,
    database: object,
    coordinator: "SemanticRevisionCoordinator",
    message_ids: tuple[str, ...],
) -> tuple[ConsoleTurnPreparation, tuple[SavedRevisionTraceProvenance, ...]]:
    """Admit saved revisions atomically or return the fail-closed turn state."""

    if preparation.state is not ConsoleTurnPreparationState.COMMITTING:
        raise TraceProvenancePersistenceError()
    if preparation.capture_mode is ConsoleTraceCaptureMode.CAPTURE_OFF:
        return preparation, ()
    failed = False
    descriptors: tuple[SavedRevisionTraceProvenance, ...] = ()
    try:
        with trace_provenance_admission_transaction(database) as cursor:
            descriptors = admit_message_provenance(
                cursor,
                coordinator=coordinator,
                message_ids=message_ids,
            )
    except Exception:
        failed = True
    if failed:
        return (
            pause_for_trace_provenance_failure(
                preparation,
                TraceProvenancePersistenceError(),
            ),
            (),
        )
    return preparation, descriptors


def build_console_request_for_preparation(
    preparation: ConsoleTurnPreparation,
    messages: Sequence[Mapping[str, Any]],
    *,
    route: ConsoleRequestRoute,
    actor_id: str | None = None,
    chain_id: str | None = None,
    **request_kwargs: Any,
) -> "PreparedConsoleRequest":
    """Build semantics under the exact capture mode admitted by a turn."""

    if not _preparation_is_valid(preparation):
        raise TraceProvenancePersistenceError()
    metadata = request_kwargs.pop("metadata_provenance", None)
    if preparation.capture_mode is ConsoleTraceCaptureMode.CAPTURE_ON:
        metadata = tuple(metadata or ()) + (
            request_route_provenance(
                route,
                actor_id=actor_id,
                chain_id=chain_id,
            ),
        )
    from tldw_chatbook.Chat.console_prepared_request import build_console_request

    return build_console_request(
        messages,
        capture_mode=preparation.capture_mode,
        metadata_provenance=metadata,
        **request_kwargs,
    )


def admit_one_shot_capture_off(
    preparation: ConsoleTurnPreparation,
    *,
    new_preparation_id: str,
    new_attempt_id: str,
) -> ConsoleTurnPreparation:
    """Create a fresh interactive preparation for explicit one-shot Capture Off.

    The new identities and detached execution context prevent a partially built
    Capture-On request, descriptor aggregate, or frozen policy from being reused.
    Callers must rebuild provider-neutral preparation from this ``READY`` state.
    """

    if not _preparation_is_valid(preparation):
        return preparation
    if (
        preparation.origin != "manual"
        or preparation.state is not ConsoleTurnPreparationState.PAUSED
        or preparation.pause_kind
        not in {
            ConsolePreparationPauseKind.TRACE_PROVENANCE,
            ConsolePreparationPauseKind.TRACE_CALL,
            ConsolePreparationPauseKind.TEMPORARY_CAPTURE,
        }
    ):
        return preparation
    try:
        _validate_identifier(new_preparation_id, "preparation ID")
        _validate_identifier(new_attempt_id, "attempt ID")
    except ConsoleTurnPreparationValidationError:
        return preparation
    if (
        new_preparation_id == preparation.preparation_id
        or new_attempt_id == preparation.attempt_id
    ):
        return preparation
    return replace(
        preparation,
        preparation_id=new_preparation_id,
        attempt_id=new_attempt_id,
        execution_context=_execution_context_with_attempt(
            preparation.execution_context,
            new_attempt_id,
        ),
        state=ConsoleTurnPreparationState.READY,
        pause_kind=None,
        one_shot_capture_off=True,
        capture_mode=ConsoleTraceCaptureMode.CAPTURE_OFF,
    )


def pause_temporary_capture_on(
    preparation: ConsoleTurnPreparation,
) -> ConsoleTurnPreparation:
    """Pause a temporary Capture-On send before durable trace admission.

    Args:
        preparation: Valid pre-dispatch preparation to evaluate.

    Returns:
        A paused manual preparation when durable capture needs an explicit user
        choice, otherwise the original preparation.

    Raises:
        TraceCallPersistenceError: If a non-interactive origin reaches this
            interactive pause boundary.
    """

    if not _preparation_is_valid(preparation):
        return preparation
    if (
        not preparation.ephemeral
        or preparation.capture_mode is ConsoleTraceCaptureMode.CAPTURE_OFF
    ):
        return preparation
    if preparation.origin != "manual":
        raise TraceCallPersistenceError()
    if (
        preparation.state is not ConsoleTurnPreparationState.READY
        or preparation.pause_kind is not None
    ):
        return preparation
    return replace(
        preparation,
        state=ConsoleTurnPreparationState.PAUSED,
        pause_kind=ConsolePreparationPauseKind.TEMPORARY_CAPTURE,
    )


def admit_promoted_temporary_capture(
    preparation: ConsoleTurnPreparation,
    execution_context: ConsoleTurnExecutionContext,
) -> ConsoleTurnPreparation:
    """Resume the exact Capture-On preparation after durable promotion.

    Args:
        preparation: Paused temporary Capture-On preparation.
        execution_context: Fresh durable authority with unchanged configuration
            and resolved destination.

    Returns:
        The ready durable preparation when every frozen identity matches,
        otherwise the original preparation.
    """

    if not _preparation_is_valid(preparation):
        return preparation
    if (
        preparation.origin != "manual"
        or not preparation.ephemeral
        or preparation.capture_mode is not ConsoleTraceCaptureMode.CAPTURE_ON
        or preparation.state is not ConsoleTurnPreparationState.PAUSED
        or preparation.pause_kind
        is not ConsolePreparationPauseKind.TEMPORARY_CAPTURE
        or not isinstance(execution_context, ConsoleTurnExecutionContext)
        or execution_context.session_id != preparation.session_id
        or execution_context.configuration
        != preparation.execution_context.configuration
        or execution_context.resolved_destination
        != preparation.execution_context.resolved_destination
    ):
        return preparation
    return replace(
        preparation,
        attempt_id=execution_context.library_authority.attempt_id,
        execution_context=execution_context,
        state=ConsoleTurnPreparationState.READY,
        pause_kind=None,
        ephemeral=False,
    )


def _transition_is_legal(
    preparation: ConsoleTurnPreparation,
    transition: ConsolePreparationTransition,
) -> bool:
    """Validate the closed transition matrix and its attempt/pause shape."""
    current = preparation.state
    new = transition.new_state
    pause = preparation.pause_kind

    if current is ConsoleTurnPreparationState.PAUSED and pause is None:
        return False
    if current is not ConsoleTurnPreparationState.PAUSED and pause is not None:
        return False
    if new is ConsoleTurnPreparationState.PAUSED:
        if transition.new_attempt_id is not None:
            return False
        if current is ConsoleTurnPreparationState.PREPARING:
            return transition.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
        if current is ConsoleTurnPreparationState.COMMITTING:
            return transition.pause_kind in {
                ConsolePreparationPauseKind.PERSISTENCE,
                ConsolePreparationPauseKind.DESTINATION_CHANGED,
                ConsolePreparationPauseKind.TRACE_PROVENANCE,
            }
        if current in {
            ConsoleTurnPreparationState.ACCEPTED,
            ConsoleTurnPreparationState.DISPATCH_STARTED,
        }:
            return transition.pause_kind is ConsolePreparationPauseKind.TRACE_CALL
        return False
    if transition.pause_kind is not None:
        return False

    if new is ConsoleTurnPreparationState.CANCELLED:
        return transition.new_attempt_id is None and current in {
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
            ConsoleTurnPreparationState.PAUSED,
        }

    if current is ConsoleTurnPreparationState.PAUSED:
        if pause is ConsolePreparationPauseKind.RETRIEVAL:
            if new is ConsoleTurnPreparationState.PREPARING:
                return _has_new_attempt(preparation, transition)
            return (
                new is ConsoleTurnPreparationState.READY
                and transition.new_attempt_id is None
            )
        if pause is ConsolePreparationPauseKind.PERSISTENCE:
            return (
                new is ConsoleTurnPreparationState.COMMITTING
                and transition.new_attempt_id is None
            )
        if pause is ConsolePreparationPauseKind.DESTINATION_CHANGED:
            return (
                new is ConsoleTurnPreparationState.COMMITTING
                and transition.new_attempt_id is None
            )
        if pause is ConsolePreparationPauseKind.TRACE_PROVENANCE:
            return (
                new is ConsoleTurnPreparationState.COMMITTING
                and transition.new_attempt_id is None
            )
        if pause is ConsolePreparationPauseKind.TRACE_CALL:
            return (
                new is ConsoleTurnPreparationState.ACCEPTED
                and transition.new_attempt_id is None
            )
        return False

    legal_without_new_attempt = {
        (
            ConsoleTurnPreparationState.PREPARING,
            ConsoleTurnPreparationState.READY,
        ),
        (ConsoleTurnPreparationState.READY, ConsoleTurnPreparationState.COMMITTING),
        (
            ConsoleTurnPreparationState.COMMITTING,
            ConsoleTurnPreparationState.ACCEPTED,
        ),
        (
            ConsoleTurnPreparationState.ACCEPTED,
            ConsoleTurnPreparationState.DISPATCH_STARTED,
        ),
        (ConsoleTurnPreparationState.ACCEPTED, ConsoleTurnPreparationState.SETTLED),
        (
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            ConsoleTurnPreparationState.DISPATCHED,
        ),
        (
            ConsoleTurnPreparationState.DISPATCH_STARTED,
            ConsoleTurnPreparationState.SETTLED,
        ),
        (
            ConsoleTurnPreparationState.DISPATCHED,
            ConsoleTurnPreparationState.SETTLED,
        ),
    }
    if (current, new) in legal_without_new_attempt:
        return transition.new_attempt_id is None
    if (
        current is ConsoleTurnPreparationState.DISPATCH_STARTED
        and new is ConsoleTurnPreparationState.DISPATCH_STARTED
    ):
        return _has_new_attempt(preparation, transition)
    return False


def _has_new_attempt(
    preparation: ConsoleTurnPreparation,
    transition: ConsolePreparationTransition,
) -> bool:
    """Return whether a retry supplies a distinct nonblank attempt identity."""
    value = transition.new_attempt_id
    return (
        type(value) is str
        and _IDENTIFIER_RE.fullmatch(value) is not None
        and value != preparation.attempt_id
    )


def _invalid(field: str) -> None:
    """Raise one bounded validation error without echoing caller data."""
    raise ConsoleTurnPreparationValidationError(
        f"Invalid Console turn preparation {field}."
    )


def _validate_identifier(value: object, field: str) -> None:
    """Require the established bounded opaque Console identifier grammar."""
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        _invalid(field)


def _validate_optional_identifier(value: object, field: str) -> None:
    """Validate an absent or bounded opaque identifier."""
    if value is not None:
        _validate_identifier(value, field)


def _validate_identifier_tuple(value: object, field: str) -> None:
    """Require a bounded immutable collection of unique opaque identifiers."""
    if (
        type(value) is not tuple
        or len(value) > CONSOLE_PREPARATION_ID_COLLECTION_MAX_ITEMS
    ):
        _invalid(field)
    for item in value:
        _validate_identifier(item, field)
    if len(set(value)) != len(value):
        _invalid(field)


def _validate_text(value: object, field: str, maximum_bytes: int) -> None:
    """Require nonblank UTF-8 text within an explicit byte bound."""
    if type(value) is not str or not value.strip():
        _invalid(field)
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        _invalid(field)
    if size > maximum_bytes:
        _invalid(field)


def _validate_execution_context(preparation: ConsoleTurnPreparation) -> None:
    """Require the real complete context and its owning identities."""
    context = preparation.execution_context
    if not isinstance(context, ConsoleTurnExecutionContext):
        _invalid("execution context")
    try:
        dump_console_turn_library_authority_json(context.library_authority)
        dump_console_resolved_destination_json(context.resolved_destination)
    except (
        AttributeError,
        ConsoleDispatchCheckpointValidationError,
        OverflowError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        _invalid("execution context")
    if context.session_id != preparation.session_id:
        _invalid("execution context session")
    if context.library_authority.attempt_id != preparation.attempt_id:
        _invalid("execution context attempt")


def _validate_preparation(preparation: ConsoleTurnPreparation) -> None:
    """Validate the complete closed construction contract."""
    _validate_identifier(preparation.preparation_id, "preparation ID")
    _validate_identifier(preparation.attempt_id, "attempt ID")
    _validate_identifier(preparation.session_id, "session ID")
    _validate_optional_identifier(
        preparation.transient_user_message_id,
        "transient USER message ID",
    )
    _validate_identifier_tuple(preparation.attachment_ids, "attachment IDs")
    _validate_identifier_tuple(preparation.evidence_ids, "evidence IDs")
    _validate_optional_identifier(preparation.prefill_id, "prefill ID")
    _validate_optional_identifier(
        preparation.pre_send_conversation_id,
        "pre-send conversation ID",
    )
    if preparation.executed_draft:
        _validate_text(
            preparation.executed_draft,
            "executed draft",
            CONSOLE_PREPARATION_DRAFT_MAX_BYTES,
        )
    elif not preparation.attachment_ids:
        _invalid("executed draft")
    _validate_text(
        preparation.pre_send_title,
        "pre-send title",
        CONSOLE_PREPARATION_TITLE_MAX_BYTES,
    )
    if type(preparation.origin) is not str or preparation.origin not in {
        "manual",
        "queued",
    }:
        _invalid("origin")
    if preparation.origin == "manual":
        if (
            preparation.queue_entry_id is not None
            or preparation.queue_generation is not None
        ):
            _invalid("manual queue authority")
    else:
        _validate_identifier(preparation.queue_entry_id, "queue entry ID")
        if (
            type(preparation.queue_generation) is not int
            or preparation.queue_generation < 0
        ):
            _invalid("queue generation")
    if type(preparation.state) is not ConsoleTurnPreparationState:
        _invalid("state")
    if (
        preparation.pause_kind is not None
        and type(preparation.pause_kind) is not ConsolePreparationPauseKind
    ):
        _invalid("pause kind")
    if preparation.state is ConsoleTurnPreparationState.PAUSED:
        if preparation.pause_kind is None:
            _invalid("paused state")
    elif preparation.pause_kind is not None:
        _invalid("non-paused state")
    if type(preparation.one_shot_bypass) is not bool:
        _invalid("one-shot bypass flag")
    if type(preparation.ephemeral) is not bool:
        _invalid("ephemeral flag")
    if type(preparation.one_shot_capture_off) is not bool:
        _invalid("one-shot Capture Off flag")
    if type(preparation.capture_mode) is not ConsoleTraceCaptureMode:
        _invalid("capture mode")
    if type(preparation.pii_redaction_enabled) is not bool:
        _invalid("PII redaction flag")
    if preparation.pii_redaction_enabled:
        _validate_identifier(
            preparation.pii_ruleset_revision_id,
            "PII ruleset revision ID",
        )
    elif preparation.pii_ruleset_revision_id is not None:
        _invalid("PII ruleset revision ID")
    if preparation.one_shot_capture_off and (
        preparation.capture_mode is not ConsoleTraceCaptureMode.CAPTURE_OFF
    ):
        _invalid("Capture Off mode")

    _validate_execution_context(preparation)
    auto_retrieve = preparation.execution_context.library_authority.policy.auto_retrieve
    if auto_retrieve is ConsoleAutoRetrieve.NEVER and (
        preparation.state is ConsoleTurnPreparationState.PREPARING
        or (
            preparation.state is ConsoleTurnPreparationState.PAUSED
            and preparation.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
        )
    ):
        _invalid("Never-mode state")
    if preparation.one_shot_bypass and (
        auto_retrieve is not ConsoleAutoRetrieve.AUTOMATIC
        or preparation.state is ConsoleTurnPreparationState.PREPARING
        or (
            preparation.state is ConsoleTurnPreparationState.PAUSED
            and preparation.pause_kind is ConsolePreparationPauseKind.RETRIEVAL
        )
    ):
        _invalid("one-shot bypass state")
    if preparation.one_shot_capture_off and (
        preparation.origin != "manual"
        or preparation.state is ConsoleTurnPreparationState.PREPARING
        or (
            preparation.state is ConsoleTurnPreparationState.PAUSED
            and preparation.pause_kind
            in {
                ConsolePreparationPauseKind.RETRIEVAL,
                ConsolePreparationPauseKind.TRACE_PROVENANCE,
                ConsolePreparationPauseKind.TRACE_CALL,
            }
        )
    ):
        _invalid("one-shot Capture Off state")


def _preparation_is_valid(preparation: object) -> bool:
    """Return false for any corrupted or constructor-bypassing preparation."""
    if not isinstance(preparation, ConsoleTurnPreparation):
        return False
    try:
        _validate_preparation(preparation)
    except (AttributeError, ConsoleTurnPreparationValidationError, TypeError):
        return False
    return True


def _transition_is_valid(transition: object) -> bool:
    """Return false for a transition corrupted after construction."""
    if not isinstance(transition, ConsolePreparationTransition):
        return False
    try:
        _validate_identifier(
            transition.preparation_id,
            "transition preparation ID",
        )
        if type(transition.expected_state) is not ConsoleTurnPreparationState:
            return False
        if type(transition.new_state) is not ConsoleTurnPreparationState:
            return False
        if (
            transition.pause_kind is not None
            and type(transition.pause_kind) is not ConsolePreparationPauseKind
        ):
            return False
        if transition.new_attempt_id is not None:
            _validate_identifier(
                transition.new_attempt_id,
                "transition attempt ID",
            )
    except (AttributeError, ConsoleTurnPreparationValidationError, TypeError):
        return False
    return True


def _execution_context_with_attempt(
    context: ConsoleTurnExecutionContext,
    attempt_id: str,
) -> ConsoleTurnExecutionContext:
    """Return the same detached context with one fresh authority attempt."""
    return ConsoleTurnExecutionContext(
        configuration=context.configuration,
        library_authority=replace(context.library_authority, attempt_id=attempt_id),
        resolved_destination=context.resolved_destination,
    )
