"""Strict, bounded contracts for device-local Console dispatch recovery."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, cast
from urllib.parse import urlsplit

from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicySnapshot,
)
from tldw_chatbook.Chat.console_endpoint_provenance import (
    EPHEMERAL_SESSION_ENDPOINT_OMITTED,
    ConsoleEndpointProvenance,
)
from tldw_chatbook.Chat.console_transaction_contribution import (
    ConsolePromotionTransactionContribution,
    ConsoleTransactionContribution,
)


CHECKPOINT_AUTHORITY_MAX_BYTES = 4096
CHECKPOINT_DESTINATION_MAX_BYTES = 2048
CHECKPOINT_RECONSTRUCTABILITY_MAX_BYTES = 2048

_CANONICAL_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    allow_nan=False,
)
_POLICY_KEYS = (
    "auto_retrieve",
    "assistant_access",
    "policy_revision",
    "source",
    "error_code",
)
_AUTHORITY_KEYS = (
    "policy",
    "direct_library_tools",
    "source_types",
    "scope_snapshot",
    "provider_intent",
    "attempt_id",
)
_SCOPE_KEYS = ("note_ids", "media_ids", "conversations_allowed")
_INTENT_KEYS = ("provider", "model", "endpoint")
_DESTINATION_KEYS = ("provider", "model", "endpoint_identity", "egress_class")
_RECONSTRUCTABILITY_KEYS = (
    "attachments_reconstructable",
    "evidence_reconstructable",
    "prefill_reconstructable",
    "opaque_reference",
)
_POLICY_SOURCES = {
    "new_session",
    "durable",
    "missing",
    "temporary",
    "unavailable",
}
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")
_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_OPAQUE_REFERENCE_RE = re.compile(r"opaque:[A-Za-z0-9][A-Za-z0-9._:-]{0,199}\Z")


class ConsoleDispatchCheckpointValidationError(ValueError):
    """A checkpoint value violated its strict bounded storage contract."""


class ConsoleDispatchCheckpointState(str, Enum):
    """States owned by the device-local operational checkpoint."""

    ACCEPTED = "accepted"
    DISPATCH_STARTED = "dispatch_started"


class ConsoleEgressClass(str, Enum):
    """Conservative provider destination classification."""

    ON_DEVICE = "on_device"
    PRIVATE_NETWORK = "private_network"
    PUBLIC_NETWORK = "public_network"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConsoleLibraryItemScopeSnapshot:
    """Exact Library item scope frozen for one executing turn."""

    note_ids: tuple[str, ...]
    media_ids: tuple[str, ...]
    conversations_allowed: bool


@dataclass(frozen=True, slots=True)
class ConsoleProviderIntent:
    """Credential-free provider selection captured before resolution."""

    provider: str
    model: str | None
    endpoint: str | None
    endpoint_provenance: ConsoleEndpointProvenance = (
        ConsoleEndpointProvenance.DURABLE_CONFIGURATION
    )


@dataclass(frozen=True, slots=True)
class ConsoleTurnLibraryAuthority:
    """Immutable maximum Library authority for one turn and its subagents."""

    policy: ConsoleLibraryPolicySnapshot
    direct_library_tools: bool
    source_types: tuple[str, ...]
    scope_snapshot: ConsoleLibraryItemScopeSnapshot
    provider_intent: ConsoleProviderIntent
    attempt_id: str


@dataclass(frozen=True, slots=True)
class ConsoleResolvedDestination:
    """Credential-free effective provider destination."""

    provider: str
    model: str | None
    endpoint_identity: str
    egress_class: ConsoleEgressClass
    endpoint_provenance: ConsoleEndpointProvenance = (
        ConsoleEndpointProvenance.DURABLE_CONFIGURATION
    )

    @property
    def identity_key(self) -> tuple[str, str | None, str, ConsoleEgressClass]:
        """Return the stable destination identity used by retry revalidation."""
        return (self.provider, self.model, self.endpoint_identity, self.egress_class)


@dataclass(frozen=True, slots=True)
class ConsoleDispatchReconstructability:
    """Presence flags used to decide whether exact retry can be reconstructed."""

    attachments_reconstructable: bool
    evidence_reconstructable: bool
    prefill_reconstructable: bool
    opaque_reference: str | None


@dataclass(frozen=True, slots=True)
class ConsoleDispatchCheckpoint:
    """One validated active-path durable dispatch recovery owner."""

    assistant_message_id: str
    user_message_id: str
    conversation_id: str
    preparation_id: str
    attempt_id: str
    state: ConsoleDispatchCheckpointState
    checkpoint_revision: int
    user_message_version: int
    assistant_message_version: int
    origin: Literal["manual", "queued"]
    queue_entry_id: str | None
    frozen_authority: ConsoleTurnLibraryAuthority
    resolved_destination: ConsoleResolvedDestination
    reconstructability: ConsoleDispatchReconstructability


@dataclass(frozen=True, slots=True)
class ConsoleDurableTurnAcceptance:
    """Inputs committed for one accepted durable manual or queued turn."""

    conversation_id: str
    user_message_id: str
    assistant_message_id: str
    parent_message_id: str | None
    user_content: str
    attachments: tuple[Mapping[str, object], ...]
    preparation_id: str
    attempt_id: str
    origin: Literal["manual", "queued"]
    queue_entry_id: str | None
    frozen_authority: ConsoleTurnLibraryAuthority
    resolved_destination: ConsoleResolvedDestination
    reconstructability: ConsoleDispatchReconstructability
    contributions: tuple[ConsoleTransactionContribution, ...]


@dataclass(frozen=True, slots=True)
class ConsoleDispatchTransition:
    """Expected-version accepted/dispatch-started state transition."""

    assistant_message_id: str
    expected_state: ConsoleDispatchCheckpointState
    expected_checkpoint_revision: int
    expected_user_message_version: int
    expected_assistant_message_version: int
    new_state: ConsoleDispatchCheckpointState
    new_attempt_id: str


@dataclass(frozen=True, slots=True)
class ConsoleAssistantSettlement:
    """Expected-version terminal assistant settlement."""

    assistant_message_id: str
    expected_checkpoint_state: ConsoleDispatchCheckpointState
    expected_checkpoint_revision: int
    expected_user_message_version: int
    expected_assistant_message_version: int
    terminal_state: Literal["complete", "stopped", "failed", "discarded"]
    content: str
    metadata_json: str | None
    usage_json: str | None = None
    provider_continuation_json: str | None = None
    thinking_blocks_json: str | None = field(default=None, repr=False)
    contributions: tuple[ConsolePromotionTransactionContribution, ...] = field(
        default=(), repr=False
    )


@dataclass(frozen=True, slots=True)
class ConsoleContinuationHandoff:
    """Expected-version ownership transfer to ADR-063 continuation."""

    assistant_message_id: str
    expected_checkpoint_revision: int
    expected_user_message_version: int
    expected_assistant_message_version: int
    provider_continuation_json: str


class ConsoleDispatchResultStatus(str, Enum):
    """Bounded dispatch repository read/write outcomes."""

    COMMITTED = "committed"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class ConsoleDispatchReadResult:
    """Validated recovery read or explicit inert outcome."""

    status: ConsoleDispatchResultStatus
    checkpoint: ConsoleDispatchCheckpoint | None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class ConsoleDispatchWriteResult:
    """Dispatch write outcome with exact committed message proof."""

    status: ConsoleDispatchResultStatus
    checkpoint: ConsoleDispatchCheckpoint | None
    committed_message_version: int | None
    committed_payload_hash: str | None


def _strict_load(value: object) -> dict[str, object]:
    if type(value) is not str:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")

    def reject_constant(_value: str) -> None:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ConsoleDispatchCheckpointValidationError(
                    "Invalid checkpoint data."
                )
            result[key] = item
        return result

    try:
        decoded = json.loads(
            value,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise ConsoleDispatchCheckpointValidationError(
            "Invalid checkpoint data."
        ) from exc
    if type(decoded) is not dict:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(dict[str, object], decoded)


def _exact_mapping(value: object, keys: tuple[str, ...]) -> Mapping[str, object]:
    if type(value) is not dict or tuple(cast(dict[str, object], value)) != keys:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(Mapping[str, object], value)


def _string(value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not cast(str, value).strip():
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(str, value)


def _identifier(value: object, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(cast(str, value)) is None:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(str, value)


def _error_code(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _ERROR_CODE_RE.fullmatch(cast(str, value)) is None:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(str, value)


def _opaque_reference(value: object) -> str | None:
    if value is None:
        return None
    if (
        type(value) is not str
        or _OPAQUE_REFERENCE_RE.fullmatch(cast(str, value)) is None
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(str, value)


def _identifier_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not list:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    items = tuple(_identifier(item) for item in cast(list[object], value))
    if len(set(items)) != len(items):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(tuple[str, ...], items)


def _credential_free_endpoint(value: str | None) -> bool:
    if value is None:
        return True
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )


def _bounded_dump(value: Mapping[str, object], cap: int) -> str:
    try:
        rendered = _CANONICAL_ENCODER.encode(value)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ConsoleDispatchCheckpointValidationError(
            "Invalid checkpoint data."
        ) from exc
    if len(rendered.encode("utf-8")) > cap:
        raise ConsoleDispatchCheckpointValidationError("Checkpoint data is too large.")
    return rendered


def _authority_value(authority: ConsoleTurnLibraryAuthority) -> dict[str, object]:
    policy = authority.policy
    valid_source_shape = (
        (
            policy.source == "durable"
            and type(policy.policy_revision) is int
            and policy.policy_revision >= 1
            and policy.error_code is None
        )
        or (
            policy.source in {"new_session", "temporary"}
            and policy.policy_revision is None
            and policy.error_code is None
        )
        or (
            policy.source in {"missing", "unavailable"}
            and policy.policy_revision is None
            and policy.auto_retrieve is ConsoleAutoRetrieve.NEVER
            and policy.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
            and (policy.source != "missing" or policy.error_code is None)
        )
    )
    if (
        not isinstance(policy.auto_retrieve, ConsoleAutoRetrieve)
        or not isinstance(policy.assistant_access, ConsoleAssistantLibraryAccess)
        or policy.source not in _POLICY_SOURCES
        or not valid_source_shape
        or type(authority.direct_library_tools) is not bool
        or type(authority.scope_snapshot.conversations_allowed) is not bool
        or authority.source_types != AUTOMATIC_LIBRARY_SOURCE_TYPES
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    error_code = _error_code(policy.error_code)
    provider = _string(authority.provider_intent.provider)
    model = _string(authority.provider_intent.model, optional=True)
    endpoint = _string(
        (
            None
            if authority.provider_intent.endpoint_provenance
            == ConsoleEndpointProvenance.EPHEMERAL_SESSION
            else authority.provider_intent.endpoint
        ),
        optional=True,
    )
    if not _credential_free_endpoint(endpoint):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return {
        "policy": {
            "auto_retrieve": policy.auto_retrieve.value,
            "assistant_access": policy.assistant_access.value,
            "policy_revision": policy.policy_revision,
            "source": policy.source,
            "error_code": error_code,
        },
        "direct_library_tools": authority.direct_library_tools,
        "source_types": list(_validated_identifier_tuple(authority.source_types)),
        "scope_snapshot": {
            "note_ids": list(
                _validated_identifier_tuple(authority.scope_snapshot.note_ids)
            ),
            "media_ids": list(
                _validated_identifier_tuple(authority.scope_snapshot.media_ids)
            ),
            "conversations_allowed": authority.scope_snapshot.conversations_allowed,
        },
        "provider_intent": {
            "provider": provider,
            "model": model,
            "endpoint": endpoint,
        },
        "attempt_id": _identifier(authority.attempt_id),
    }


def _validated_identifier_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    items = tuple(_identifier(item) for item in cast(tuple[object, ...], value))
    if len(set(items)) != len(items):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return cast(tuple[str, ...], items)


def dump_console_turn_library_authority_json(
    authority: ConsoleTurnLibraryAuthority,
) -> str:
    """Serialize frozen authority to canonical bounded JSON."""
    if not isinstance(authority, ConsoleTurnLibraryAuthority):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return _bounded_dump(_authority_value(authority), CHECKPOINT_AUTHORITY_MAX_BYTES)


def parse_console_turn_library_authority_json(
    value: object,
) -> ConsoleTurnLibraryAuthority:
    """Strictly parse canonical bounded frozen authority JSON."""
    if (
        type(value) is not str
        or len(value.encode("utf-8")) > CHECKPOINT_AUTHORITY_MAX_BYTES
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    data = _exact_mapping(_strict_load(value), _AUTHORITY_KEYS)
    policy = _exact_mapping(data["policy"], _POLICY_KEYS)
    scope = _exact_mapping(data["scope_snapshot"], _SCOPE_KEYS)
    intent = _exact_mapping(data["provider_intent"], _INTENT_KEYS)
    error_code = _error_code(policy["error_code"])
    try:
        auto_retrieve = ConsoleAutoRetrieve(_string(policy["auto_retrieve"]))
        assistant_access = ConsoleAssistantLibraryAccess(
            _string(policy["assistant_access"])
        )
    except ValueError as exc:
        raise ConsoleDispatchCheckpointValidationError(
            "Invalid checkpoint data."
        ) from exc
    valid_source_shape = (
        (
            policy["source"] == "durable"
            and type(policy["policy_revision"]) is int
            and cast(int, policy["policy_revision"]) >= 1
            and error_code is None
        )
        or (
            policy["source"] in {"new_session", "temporary"}
            and policy["policy_revision"] is None
            and error_code is None
        )
        or (
            policy["source"] in {"missing", "unavailable"}
            and policy["policy_revision"] is None
            and auto_retrieve is ConsoleAutoRetrieve.NEVER
            and assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
            and (policy["source"] != "missing" or error_code is None)
        )
    )
    source_types = _identifier_tuple(data["source_types"])
    if (
        type(data["direct_library_tools"]) is not bool
        or type(scope["conversations_allowed"]) is not bool
        or policy["source"] not in _POLICY_SOURCES
        or not valid_source_shape
        or source_types != AUTOMATIC_LIBRARY_SOURCE_TYPES
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    endpoint = _string(intent["endpoint"], optional=True)
    if not _credential_free_endpoint(endpoint):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    try:
        authority = ConsoleTurnLibraryAuthority(
            policy=ConsoleLibraryPolicySnapshot(
                auto_retrieve=auto_retrieve,
                assistant_access=assistant_access,
                policy_revision=cast(int | None, policy["policy_revision"]),
                source=cast(
                    Literal[
                        "new_session",
                        "durable",
                        "missing",
                        "temporary",
                        "unavailable",
                    ],
                    policy["source"],
                ),
                error_code=error_code,
            ),
            direct_library_tools=cast(bool, data["direct_library_tools"]),
            source_types=source_types,
            scope_snapshot=ConsoleLibraryItemScopeSnapshot(
                note_ids=_identifier_tuple(scope["note_ids"]),
                media_ids=_identifier_tuple(scope["media_ids"]),
                conversations_allowed=cast(bool, scope["conversations_allowed"]),
            ),
            provider_intent=ConsoleProviderIntent(
                provider=cast(str, _string(intent["provider"])),
                model=_string(intent["model"], optional=True),
                endpoint=endpoint,
            ),
            attempt_id=cast(str, _identifier(data["attempt_id"])),
        )
    except ValueError as exc:
        raise ConsoleDispatchCheckpointValidationError(
            "Invalid checkpoint data."
        ) from exc
    if dump_console_turn_library_authority_json(authority) != value:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return authority


def dump_console_resolved_destination_json(
    destination: ConsoleResolvedDestination,
) -> str:
    """Serialize a credential-free resolved destination to canonical JSON."""
    if not isinstance(destination, ConsoleResolvedDestination) or not isinstance(
        destination.egress_class, ConsoleEgressClass
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    endpoint_identity = cast(
        str,
        _string(
            EPHEMERAL_SESSION_ENDPOINT_OMITTED
            if destination.endpoint_provenance
            == ConsoleEndpointProvenance.EPHEMERAL_SESSION
            else destination.endpoint_identity
        ),
    )
    if not _credential_free_endpoint(endpoint_identity):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return _bounded_dump(
        {
            "provider": _string(destination.provider),
            "model": _string(destination.model, optional=True),
            "endpoint_identity": endpoint_identity,
            "egress_class": destination.egress_class.value,
        },
        CHECKPOINT_DESTINATION_MAX_BYTES,
    )


def parse_console_resolved_destination_json(
    value: object,
) -> ConsoleResolvedDestination:
    """Strictly parse a canonical bounded resolved destination."""
    if (
        type(value) is not str
        or len(value.encode("utf-8")) > CHECKPOINT_DESTINATION_MAX_BYTES
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    data = _exact_mapping(_strict_load(value), _DESTINATION_KEYS)
    endpoint_identity = cast(str, _string(data["endpoint_identity"]))
    if not _credential_free_endpoint(endpoint_identity):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    try:
        destination = ConsoleResolvedDestination(
            provider=cast(str, _string(data["provider"])),
            model=_string(data["model"], optional=True),
            endpoint_identity=endpoint_identity,
            egress_class=ConsoleEgressClass(_string(data["egress_class"])),
            endpoint_provenance=(
                ConsoleEndpointProvenance.EPHEMERAL_SESSION
                if endpoint_identity == EPHEMERAL_SESSION_ENDPOINT_OMITTED
                else ConsoleEndpointProvenance.DURABLE_CONFIGURATION
            ),
        )
    except ValueError as exc:
        raise ConsoleDispatchCheckpointValidationError(
            "Invalid checkpoint data."
        ) from exc
    if dump_console_resolved_destination_json(destination) != value:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return destination


def dump_console_dispatch_reconstructability_json(
    reconstructability: ConsoleDispatchReconstructability,
) -> str:
    """Serialize bounded reconstructability flags and an opaque reference."""
    if not isinstance(reconstructability, ConsoleDispatchReconstructability) or any(
        type(value) is not bool
        for value in (
            reconstructability.attachments_reconstructable,
            reconstructability.evidence_reconstructable,
            reconstructability.prefill_reconstructable,
        )
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return _bounded_dump(
        {
            "attachments_reconstructable": reconstructability.attachments_reconstructable,
            "evidence_reconstructable": reconstructability.evidence_reconstructable,
            "prefill_reconstructable": reconstructability.prefill_reconstructable,
            "opaque_reference": _opaque_reference(reconstructability.opaque_reference),
        },
        CHECKPOINT_RECONSTRUCTABILITY_MAX_BYTES,
    )


def parse_console_dispatch_reconstructability_json(
    value: object,
) -> ConsoleDispatchReconstructability:
    """Strictly parse canonical bounded reconstructability JSON."""
    if (
        type(value) is not str
        or len(value.encode("utf-8")) > CHECKPOINT_RECONSTRUCTABILITY_MAX_BYTES
    ):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    data = _exact_mapping(_strict_load(value), _RECONSTRUCTABILITY_KEYS)
    if any(type(data[key]) is not bool for key in _RECONSTRUCTABILITY_KEYS[:3]):
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    reconstructability = ConsoleDispatchReconstructability(
        attachments_reconstructable=cast(bool, data["attachments_reconstructable"]),
        evidence_reconstructable=cast(bool, data["evidence_reconstructable"]),
        prefill_reconstructable=cast(bool, data["prefill_reconstructable"]),
        opaque_reference=_opaque_reference(data["opaque_reference"]),
    )
    if dump_console_dispatch_reconstructability_json(reconstructability) != value:
        raise ConsoleDispatchCheckpointValidationError("Invalid checkpoint data.")
    return reconstructability
