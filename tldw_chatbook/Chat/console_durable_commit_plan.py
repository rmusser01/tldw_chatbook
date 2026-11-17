"""Local sealed durable plans; these objects grant no native admission."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .console_chat_store import (
        ConsoleChatSession,
        ConsoleDurableAcceptanceFingerprint,
        ConsoleStagedConversationIdentity,
        _ConsoleStagedDurableOwnerIds,
        _ConsoleDurableCommitReservation,
    )
    from .console_dispatch_checkpoint import (
        ConsoleDurableTurnAcceptance,
        ConsoleDispatchCheckpoint,
    )
    from .console_received_dispatch import (
        ReceivedPreparationSource,
        DomainReceivedPreparationSource,
    )
    from .console_received_intent import ConsoleReceivedTurnIntent
    from .console_received_turn import ConsoleReceivedTurnClaim
    from .console_runtime import ConsoleRuntime, _ConsoleTurnCustodyRecord
    from .console_turn_preparation import ConsoleTurnPreparation
    from .console_library_policy import ConsoleLibraryPolicyCandidate
    from .console_context_policy import ConsoleContextPolicyOverrides
    from .console_project_instructions import ProjectInstructionControlState
    from .console_generation_settings_metadata import ConsoleGenerationSettingsSnapshot
    from .console_session_settings import ConsoleSessionSettings


def seal(value: Any) -> Any:
    """Copy canonical containers into immutable plan-owned values."""
    if (
        value is None
        or type(value) in {bool, int, float, str, bytes}
        or isinstance(value, Enum)
    ):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType({key: seal(item) for key, item in value.items()})
    if type(value) in {tuple, list}:
        return tuple(seal(item) for item in value)
    if is_dataclass(value) and value.__dataclass_params__.frozen:
        return replace(
            value,
            **{
                item.name: seal(getattr(value, item.name))
                for item in fields(value)
                if item.init
            },
        )
    raise TypeError("Separated durable plans require immutable canonical inputs.")


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class DurablePlanSource:
    runtime: ConsoleRuntime
    record: _ConsoleTurnCustodyRecord
    source: ReceivedPreparationSource | DomainReceivedPreparationSource
    task: asyncio.Task[Any]
    claim: ConsoleReceivedTurnClaim | None
    intent: ConsoleReceivedTurnIntent
    source_fields: tuple = ()
    binding: Any = None

    def require_current(
        self, preparation: ConsoleTurnPreparation | None = None
    ) -> None:
        from .console_domain_services import _definition_current

        dispatch = _DISPATCH_SOURCE
        if sys.modules.get(dispatch.__name__) is not dispatch or any(
            dispatch.__dict__.get(name) is not definition[0]
            or not _definition_current(definition)
            for name, definition in _RECEIVED_DEFINITIONS
        ):
            raise RuntimeError("Durable received source definition changed.")
        if (
            self.claim is None
            or self.runtime._turn_custody.get(self.record.turn_id) is not self.record
            or self.record.task is not self.task
            or self.record.store is not self.source.store
            or self.record.received_claim is not self.claim
            or self.record.received_intent is not self.intent
            or any(
                getattr(self.source, name) is not value
                for name, value in self.source_fields
            )
        ):
            raise RuntimeError("Durable received input owner changed.")
        dispatch.require_received_source(
            self.runtime,
            self.record,
            self.source,
            inputs=preparation is None,
            promoted_preparation=preparation,
            source_task=self.task,
        )


_received_plan_source: ContextVar[DurablePlanSource | None] = ContextVar(
    "console_durable_plan_received_source", default=None
)


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _LegacyCommitCall:
    store: Any
    acceptance: Any
    thread: threading.Thread


_legacy_commit_call: ContextVar[_LegacyCommitCall | None] = ContextVar(
    "console_legacy_commit_call", default=None
)


@contextmanager
def legacy_commit_call(store, acceptance):
    call = _LegacyCommitCall(store, acceptance, threading.current_thread())
    token = _legacy_commit_call.set(call)
    try:
        yield
    finally:
        _legacy_commit_call.reset(token)


def current_legacy_commit_call(store, acceptance):
    call = _legacy_commit_call.get()
    if (
        call is not None
        and call.store is store
        and call.acceptance is acceptance
        and call.thread is threading.current_thread()
    ):
        return call
    return None


@contextmanager
def bind_durable_plan_source(runtime, record, source):
    task = asyncio.current_task()
    if task is None:
        raise RuntimeError("Durable source requires its owning task.")
    receipt = DurablePlanSource(
        runtime, record, source, task, record.received_claim, record.received_intent
    )
    token = _received_plan_source.set(receipt)
    try:
        yield
    finally:
        _received_plan_source.reset(token)


def capture_durable_plan_source(store: ConsoleChatStore) -> DurablePlanSource:
    receipt = _received_plan_source.get()
    if (
        receipt is None
        or receipt.task is not asyncio.current_task()
        or receipt.source.store is not store
        or receipt.record.received_claim is not receipt.claim
        or receipt.record.received_intent is not receipt.intent
    ):
        raise RuntimeError(
            "Separated durable plan requires its original received task."
        )
    return replace(
        receipt,
        source_fields=tuple(
            (item.name, getattr(receipt.source, item.name))
            for item in fields(receipt.source)
        ),
        binding=receipt,
    )


def record_rerouted_durable_claim(runtime, record, source, claim) -> None:
    """Anchor only the claim just issued by the original queue-reroute owner."""
    receipt = _received_plan_source.get()
    if (
        receipt is None
        or receipt.task is not asyncio.current_task()
        or receipt.runtime is not runtime
        or receipt.record is not record
        or receipt.source is not source
        or receipt.claim is not None
        or record.received_intent is not receipt.intent
        or record.received_claim is not claim
        or runtime._turn_custody.get(record.turn_id) is not record
        or record.task is not receipt.task
        or not source.store.received_turn_is_current(claim)
    ):
        raise RuntimeError("Durable reroute claim owner changed.")
    _received_plan_source.set(replace(receipt, claim=claim))


def require_reservation_source(source, read):
    if (
        type(source) is not DurablePlanSource
        or read is None
        or read.source is not source
        or read.task is not source.task
        or source.binding is not _received_plan_source.get()
    ):
        raise RuntimeError(
            "Separated durable reservation requires its issued source receipt."
        )


def require_plan_methods(store, persistence):
    """Qualify the finite selected callbacks against their original definitions."""
    from .console_domain_services import _definition_current

    for receiver, records in (
        (store, _STORE_DEFINITIONS),
        (persistence, _PERSISTENCE_DEFINITIONS),
    ):
        for name, definition in records:
            method = getattr(receiver, name, None)
            if (
                getattr(method, "__self__", None) is not receiver
                or getattr(method, "__func__", None) is not definition[0]
                or not _definition_current(definition)
            ):
                raise RuntimeError("Durable plan callback source changed.")
    for name, definition in _FINGERPRINT_DEFINITIONS:
        method = getattr(store, name, None)
        function = getattr(method, "__func__", method)
        if function is not definition[0] or not _definition_current(definition):
            raise RuntimeError("Durable fingerprint source changed.")


def require_plan_contributions(acceptance: ConsoleDurableTurnAcceptance) -> None:
    """Accept only the two immutable contribution types emitted by stock Send."""
    from .console_domain_services import _definition_current

    for contribution in acceptance.contributions:
        records = _CONTRIBUTION_DEFINITIONS.get(type(contribution))
        if records is None:
            raise RuntimeError("Durable contribution requires the legacy profile.")
        for name, definition in records:
            method = getattr(contribution, name, None)
            if (
                getattr(method, "__self__", None) is not contribution
                or getattr(method, "__func__", None) is not definition[0]
                or not _definition_current(definition)
            ):
                raise RuntimeError("Durable contribution source changed.")


@dataclass(frozen=True, slots=True)
class ConsoleDurableRecoveryProjection:
    """Body-free expected facts; the full fingerprint is not a database column."""

    fingerprint: ConsoleDurableAcceptanceFingerprint
    user_content_sha256: str
    parent_message_id: str | None
    attachments: tuple[tuple[int, str, str, str], ...]
    frozen_authority_json: str
    resolved_destination_json: str
    reconstructability_json: str
    project_context_sha256: str
    first_persist: bool
    generation_revision: int
    context_policy_revision: int
    roleplay_identity_revision: int
    user_root_fork: bool
    agent_chat_start: Any
    context_policy_initial_revision: int | None
    initial_user_version: int = 1
    initial_assistant_version: int = 1
    initial_checkpoint_revision: int = 1
    checkpoint: ConsoleDispatchCheckpoint | None = None


def recovery_projection(
    acceptance,
    fingerprint,
    project_json,
    first_persist,
    generation_revision,
    context_policy_revision,
    roleplay_identity_revision,
    context_policy_overrides,
):
    from .console_dispatch_checkpoint import (
        dump_console_turn_library_authority_json,
        dump_console_resolved_destination_json,
        dump_console_dispatch_reconstructability_json,
    )
    from .console_dispatch_repository import ConsoleDispatchRepository

    from .console_chat_store import MAX_PENDING_ATTACHMENTS

    # Larger custom/legacy inputs retain their synchronous save contract, but
    # cannot supply a bounded selected-service recovery projection.
    if len(acceptance.attachments) > MAX_PENDING_ATTACHMENTS:
        return None
    projection = ConsoleDurableRecoveryProjection(
        fingerprint,
        hashlib.sha256(acceptance.user_content.encode("utf-8")).hexdigest(),
        acceptance.parent_message_id,
        tuple(
            (position, hashlib.sha256(data).hexdigest(), mime, name if position else "")
            for position, data, mime, name in ConsoleDispatchRepository._validated_attachments(
                acceptance
            )
        ),
        dump_console_turn_library_authority_json(acceptance.frozen_authority),
        dump_console_resolved_destination_json(acceptance.resolved_destination),
        dump_console_dispatch_reconstructability_json(acceptance.reconstructability),
        hashlib.sha256((project_json or "").encode("utf-8")).hexdigest(),
        first_persist,
        generation_revision,
        context_policy_revision,
        roleplay_identity_revision,
        acceptance.user_root_fork,
        acceptance.agent_chat_start,
        None
        if context_policy_overrides is None or context_policy_overrides.is_empty
        else 1,
    )
    encoded = json.dumps(
        ConsoleChatStore._canonical_fingerprint_value(projection),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return projection if len(encoded.encode("utf-8")) <= 16_384 else None


def session_inputs(session):
    """Capture only the actual fields consumed by reserve and receipt revisions."""
    names = (
        "incarnation_id",
        "conversation_binding_revision",
        "persisted_conversation_id",
        "title",
        "workspace_id",
        "settings",
        "settings_revision",
        "generation_settings_revision",
        "context_policy_revision",
        "context_policy_overrides",
        "identity_revision",
        "staged_context_policy_failure_label",
        "staged_context_policy_failure_revision",
        "thinking_history_policy",
        "runtime_backend",
        "assistant_kind",
        "assistant_id",
        "assistant_authority_id",
        "persona_memory_mode",
        "character_id",
        "character_name",
        "speech_preferences",
        "project_instruction_state",
        "user_display_name_override",
        "character_system_template",
        "draft_revision",
        "attachment_revision",
        "one_shot_prefill_revision",
        "agent_handoff_revision",
        "ephemeral",
    )
    return tuple((name, seal(getattr(session, name))) for name in names) + (
        ("library_policy", seal(session.library_policy_holder.snapshot)),
    )


@dataclass(frozen=True, slots=True, eq=False, repr=False)
class _ConsoleDurableCommitPlan:
    store: ConsoleChatStore
    persistence: Any
    persistence_commit: Any
    database: Any
    received_acceptance: ConsoleDurableTurnAcceptance
    acceptance: ConsoleDurableTurnAcceptance
    source: DurablePlanSource | None
    legacy_call: _LegacyCommitCall | None
    preparation: ConsoleTurnPreparation
    session: ConsoleChatSession
    inputs: tuple
    settings_owner: ConsoleSessionSettings | None
    identity: ConsoleStagedConversationIdentity
    owners: _ConsoleStagedDurableOwnerIds
    reservation: _ConsoleDurableCommitReservation
    fingerprint: ConsoleDurableAcceptanceFingerprint
    policy_candidate: ConsoleLibraryPolicyCandidate
    conversation_kwargs: Mapping[str, object]
    context_policy_overrides: ConsoleContextPolicyOverrides | None
    project_state: ProjectInstructionControlState
    project_json: str | None
    first_persist: bool
    generation_snapshot: ConsoleGenerationSettingsSnapshot | None
    generation_revision: int
    context_policy_revision: int
    policy_failure_label: Any
    roleplay_identity_revision: int
    recovery: ConsoleDurableRecoveryProjection | None


@dataclass(frozen=True, slots=True)
class ConsoleExactCommitOutcome:
    status: Literal[
        "matching_durable_acceptance_observed",
        "matching_pair_terminal_observed",
        "matching_pair_continuation_observed",
        "no_matching_durable_rows_observed",
        "conflict_or_unknown",
    ]
    checkpoint: ConsoleDispatchCheckpoint | None = None
    detail: str | None = None
    full_fingerprint_verified: bool = False


from . import console_received_dispatch as _DISPATCH_SOURCE
from .console_domain_services import _definition
from .console_chat_store import ConsoleChatStore
from .chat_persistence_service import ChatPersistenceService

_RECEIVED_DEFINITIONS = _DISPATCH_SOURCE._DOMAIN_SOURCE_DEFINITIONS
_STORE_DEFINITIONS = tuple(
    (name, _definition(getattr(ConsoleChatStore, name)))
    for name in (
        "reserve_durable_commit_plan",
        "commit_reserved_durable_turn",
        "abandon_reserved_durable_commit",
    )
)
_PERSISTENCE_DEFINITIONS = (
    ("commit_durable_turn", _definition(ChatPersistenceService.commit_durable_turn)),
)
_FINGERPRINT_DEFINITIONS = tuple(
    (
        name,
        _definition(
            getattr(
                getattr(ConsoleChatStore, name),
                "__func__",
                getattr(ConsoleChatStore, name),
            )
        ),
    )
    for name in (
        "_canonical_fingerprint_value",
        "_contribution_fingerprint",
        "_durable_acceptance_fingerprint",
    )
)

from .library_preparation import LibraryPreparationContribution
from .console_prompt_mixin_service import MixinAttemptContribution

_CONTRIBUTION_DEFINITIONS = {
    cls: tuple((name, _definition(getattr(cls, name))) for name in names)
    for cls, names in (
        (LibraryPreparationContribution, ("durable_acceptance_fingerprint", "write")),
        (
            MixinAttemptContribution,
            ("durable_acceptance_fingerprint", "write", "write_exact"),
        ),
    )
}
