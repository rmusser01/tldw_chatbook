"""Pure contracts for device-local Console Library policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ConsoleAutoRetrieve(str, Enum):
    """Whether Console automatically retrieves Library evidence on send."""

    NEVER = "never"
    AUTOMATIC = "automatic"


class ConsoleAssistantLibraryAccess(str, Enum):
    """Whether the assistant may use the built-in Library capability."""

    BLOCKED = "blocked"
    ALLOWED = "allowed"


AUTOMATIC_LIBRARY_SOURCE_TYPES: tuple[str, ...] = (
    "notes",
    "media",
    "conversations",
)

_POLICY_CORRUPT_ERROR_CODE = "corrupt_policy"
_POLICY_READ_ERROR_CODE = "policy_read_error"


@dataclass(frozen=True, slots=True)
class ConsoleLibraryMigrationSeed:
    """Sanitized legacy automatic-retrieval value for a schema migration."""

    auto_retrieve_on_send: bool

    def __post_init__(self) -> None:
        """Reject values that have not passed config-layer boolean coercion."""
        if type(self.auto_retrieve_on_send) is not bool:
            raise TypeError("auto_retrieve_on_send must be a bool")


@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyDefaults:
    """Policy captured when a new local Console session is created."""

    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess


@dataclass(frozen=True, slots=True)
class ConsoleConversationLibraryPolicy:
    """One durable, device-local policy row for a conversation."""

    conversation_id: str
    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess
    policy_revision: int
    updated_at: str


@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicySnapshot:
    """Effective policy authority made safe for a session or execution turn."""

    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess
    policy_revision: int | None
    source: Literal["new_session", "durable", "missing", "temporary", "unavailable"]
    error_code: str | None = None


@dataclass(slots=True)
class ConsoleLibraryPolicyHolder:
    """Mutable in-process policy state for one Console session."""

    snapshot: ConsoleLibraryPolicySnapshot
    explicitly_staged: bool = False
    save_pending: bool = False


@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyCandidate:
    """The two user-editable values used for an insert or CAS write."""

    auto_retrieve: ConsoleAutoRetrieve
    assistant_access: ConsoleAssistantLibraryAccess


@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyReadResult:
    """A durable policy read with its effective safe snapshot."""

    snapshot: ConsoleLibraryPolicySnapshot
    durable_policy: ConsoleConversationLibraryPolicy | None


class ConsoleLibraryPolicyWriteStatus(str, Enum):
    """Bounded outcomes from a durable policy write."""

    COMMITTED = "committed"
    CONFLICT = "conflict"
    MISSING_CONVERSATION = "missing_conversation"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ConsoleLibraryPolicyWriteResult:
    """Result of an insert or compare-and-swap policy write."""

    status: ConsoleLibraryPolicyWriteStatus
    snapshot: ConsoleLibraryPolicySnapshot


def _safe_snapshot(
    *,
    source: Literal["missing", "unavailable"],
    error_code: str | None = None,
) -> ConsoleLibraryPolicySnapshot:
    """Return the never/blocked policy used when durable authority is absent."""
    return ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=None,
        source=source,
        error_code=error_code,
    )


def _is_valid_durable_policy(policy: ConsoleConversationLibraryPolicy) -> bool:
    """Return whether a runtime dataclass instance satisfies its wire contract."""
    return (
        isinstance(policy.conversation_id, str)
        and bool(policy.conversation_id.strip())
        and isinstance(policy.auto_retrieve, ConsoleAutoRetrieve)
        and isinstance(policy.assistant_access, ConsoleAssistantLibraryAccess)
        and type(policy.policy_revision) is int
        and policy.policy_revision >= 0
        and isinstance(policy.updated_at, str)
        and bool(policy.updated_at.strip())
    )


def normalize_policy_read(raw_policy: object) -> ConsoleLibraryPolicyReadResult:
    """Normalize one repository read without treating failure as permission.

    Args:
        raw_policy: A durable policy instance, no row (``None``), a repository
            exception, or malformed persistence output.

    Returns:
        A durable snapshot for a valid row, otherwise a safe Never/Blocked
        snapshot with bounded error information.
    """
    if isinstance(raw_policy, ConsoleConversationLibraryPolicy) and _is_valid_durable_policy(
        raw_policy
    ):
        policy = raw_policy
        return ConsoleLibraryPolicyReadResult(
            snapshot=ConsoleLibraryPolicySnapshot(
                auto_retrieve=policy.auto_retrieve,
                assistant_access=policy.assistant_access,
                policy_revision=policy.policy_revision,
                source="durable",
            ),
            durable_policy=policy,
        )
    if raw_policy is None:
        return ConsoleLibraryPolicyReadResult(
            snapshot=_safe_snapshot(source="missing"),
            durable_policy=None,
        )
    error_code = (
        _POLICY_READ_ERROR_CODE
        if isinstance(raw_policy, BaseException)
        else _POLICY_CORRUPT_ERROR_CODE
    )
    return ConsoleLibraryPolicyReadResult(
        snapshot=_safe_snapshot(source="unavailable", error_code=error_code),
        durable_policy=None,
    )
