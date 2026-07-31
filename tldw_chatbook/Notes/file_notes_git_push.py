"""Pure contracts, parsers, and command builders for guarded File Notes push."""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import unicodedata
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass, field
from enum import Enum
from typing import Literal
from urllib.parse import SplitResult, urlsplit
from weakref import WeakKeyDictionary

PushContractErrorCode = Literal[
    "invalid_destination_ref",
    "invalid_endpoint",
    "invalid_configuration",
    "invalid_object_id",
    "unsafe_text",
    "invalid_command_context",
]
RemoteRefState = Literal[
    "parent",
    "candidate",
    "missing",
    "divergent",
    "malformed",
]
PushPorcelainState = Literal["accepted", "rejected", "malformed"]
PushOutcomeState = Literal[
    "already_published",
    "succeeded",
    "failed_no_update_observed",
    "uncertain",
]
PushRecoveryState = Literal["succeeded", "uncertain", "needs_attention"]
PushTransport = Literal["https", "ssh"]
PushDestinationPolicyState = Literal["ready", "blocked", "stale"]

_ERROR_MESSAGES: dict[PushContractErrorCode, str] = {
    "invalid_destination_ref": "The destination branch ref is not allowed.",
    "invalid_endpoint": "The configured push endpoint is not allowed.",
    "invalid_configuration": "The configured push destination is not allowed.",
    "invalid_object_id": "A complete lowercase Git object ID is required.",
    "unsafe_text": "Text contains characters that cannot be displayed safely.",
    "invalid_command_context": "The private Git command context is invalid.",
}
_OBJECT_ID_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SSH_USER_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
_DNS_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?")
_REPOSITORY_PATH_PATTERN = re.compile(r"[A-Za-z0-9._~/-]+")
_SCP_ENDPOINT_PATTERN = re.compile(
    r"(?P<user>[A-Za-z0-9._-]+)@"
    r"(?P<host>\[[^\]]+\]|[^:/\\]+):"
    r"(?P<path>.+)"
)
_MAX_REMOTE_OUTPUT_BYTES = 64 * 1024
_REMOTE_REF_STATES = frozenset(
    {"parent", "candidate", "missing", "divergent", "malformed"}
)
_PORCELAIN_STATES = frozenset({"accepted", "rejected", "malformed"})
_OUTCOME_COPY: dict[PushOutcomeState, tuple[str, str, bool]] = {
    "already_published": (
        "Already published",
        (
            "The configured destination currently points to this commit. "
            "No push was started by Chatbook."
        ),
        False,
    ),
    "succeeded": (
        "Succeeded",
        (
            "Git reported the exact update accepted, and the configured "
            "destination currently points to this commit."
        ),
        False,
    ),
    "failed_no_update_observed": (
        "Failed with no update currently observed",
        (
            "Git reported an unsuccessful push, every owned process ended, "
            "and the configured destination currently points to the reviewed "
            "parent. Remote-side work may still be pending or may occur later."
        ),
        False,
    ),
    "uncertain": (
        "Uncertain",
        (
            "Chatbook cannot currently prove whether the destination accepted "
            "the update. Do not push again automatically. Check the original "
            "destination again without pushing."
        ),
        True,
    ),
}
_RECOVERY_COPY: dict[str, tuple[PushRecoveryState, str, str, bool]] = {
    "candidate": (
        "succeeded",
        "Succeeded",
        (
            "A query-only check currently observes the candidate at the "
            "original destination. The observation does not establish the "
            "cause of the update. No push was sent by this check."
        ),
        False,
    ),
    "parent": (
        "uncertain",
        "Uncertain",
        (
            "A query-only check currently observes the reviewed parent, so "
            "the prior attempt remains uncertain. Remote-side work may still "
            "be pending. No push was sent by this check."
        ),
        True,
    ),
    "unprovable": (
        "uncertain",
        "Uncertain",
        (
            "A query-only check currently cannot prove the candidate at the "
            "original destination, so the prior attempt remains uncertain. "
            "No push was sent by this check."
        ),
        True,
    ),
    "needs_attention": (
        "needs_attention",
        "Needs attention",
        (
            "A query-only check currently cannot prove the candidate at the "
            "original destination. No push was sent by this check."
        ),
        True,
    ),
}
_RECOVERY_COPY_VALUES = frozenset(_RECOVERY_COPY.values())
_DESTINATION_POLICY_MESSAGES: dict[PushDestinationPolicyState, str] = {
    "ready": (
        "The exact local candidate and configured destination passed local "
        "policy checks. No remote contact has started."
    ),
    "blocked": (
        "The exact local candidate or configured destination did not pass "
        "local policy checks. No remote contact was made."
    ),
    "stale": (
        "The local candidate or repository authority changed before policy "
        "proof completed. No remote contact was made."
    ),
}


class PushContractError(ValueError):
    """Bounded refusal from a pure guarded-push contract.

    Attributes:
        code: Stable machine-readable refusal category.
    """

    def __init__(self, code: PushContractErrorCode) -> None:
        """Initialize a sanitized contract refusal.

        Args:
            code: Stable refusal category.
        """
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class PushIncludedNote:
    """Control-safe literal provenance for one included note.

    Attributes:
        group_id: Process-local session group identity.
        display_text: Exact literal note label. This value is not pre-escaped
            Rich markup and MUST be rendered with ``markup=False``.
    """

    group_id: int
    display_text: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.group_id, bool)
            or not isinstance(self.group_id, int)
            or self.group_id < 0
            or not _is_safe_display_text(self.display_text)
            or not self.display_text.strip()
        ):
            raise PushContractError("unsafe_text")


@dataclass(frozen=True, slots=True)
class PushCandidateProjection:
    """Sanitized immutable projection of one exact guarded-push candidate.

    Attributes:
        subject: Exact control-safe literal commit subject. This value is not
            pre-escaped Rich markup and MUST be rendered with ``markup=False``.
        included_notes: Literal note labels governed by
            :class:`PushIncludedNote`.
    """

    local_branch_ref: str
    parent_oid: str
    candidate_oid: str
    subject: str
    included_notes: tuple[PushIncludedNote, ...]

    def __post_init__(self) -> None:
        validate_destination_ref(self.local_branch_ref)
        _validate_object_id(self.parent_oid)
        _validate_object_id(self.candidate_oid)
        if (
            len(self.parent_oid) != len(self.candidate_oid)
            or self.parent_oid == self.candidate_oid
        ):
            raise PushContractError("invalid_object_id")
        if (
            not _is_safe_display_text(self.subject)
            or not self.subject.strip()
            or "\n" in self.subject
            or "\r" in self.subject
        ):
            raise PushContractError("unsafe_text")
        notes = tuple(self.included_notes)
        if any(not isinstance(note, PushIncludedNote) for note in notes):
            raise PushContractError("unsafe_text")
        object.__setattr__(self, "included_notes", notes)

    @property
    def included_note_count(self) -> int:
        """Return the fixed number of included session notes."""
        return len(self.included_notes)

    @property
    def transition(self) -> str:
        """Return the literal parent-to-candidate transition."""
        return f"{self.parent_oid} → {self.candidate_oid}"


@dataclass(frozen=True, slots=True)
class PushDestinationProjection:
    """Selectable endpoint identity without the effective endpoint value."""

    scheme: PushTransport
    host: str
    port: int
    repository_path: str
    destination_ref: str
    ssh_user: str | None = None

    def __post_init__(self) -> None:
        if self.scheme not in {"https", "ssh"}:
            raise PushContractError("invalid_endpoint")
        if _normalize_host(self.host) != self.host:
            raise PushContractError("invalid_endpoint")
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise PushContractError("invalid_endpoint")
        if not _is_safe_display_repository_path(self.repository_path):
            raise PushContractError("invalid_endpoint")
        validate_destination_ref(self.destination_ref)
        if self.scheme == "https" and self.ssh_user is not None:
            raise PushContractError("invalid_endpoint")
        if self.scheme == "ssh" and not _is_valid_ssh_user(self.ssh_user):
            raise PushContractError("invalid_endpoint")

    @property
    def selectable_details(self) -> tuple[tuple[str, str], ...]:
        """Return literal-rendering endpoint rows without reconstructing a URL."""
        rows: list[tuple[str, str]] = [
            ("Scheme", self.scheme),
            ("Host", self.host),
            ("Port", str(self.port)),
        ]
        if self.ssh_user is not None:
            rows.append(("SSH user", self.ssh_user))
        rows.extend(
            (
                ("Repository path", self.repository_path),
                ("Destination ref", self.destination_ref),
            )
        )
        return tuple(rows)

    @property
    def certificate_verification_required(self) -> bool:
        """Return whether normal HTTPS certificate verification is required."""
        return self.scheme == "https"

    @property
    def host_key_verification_required(self) -> bool:
        """Return whether normal OpenSSH host-key verification is required."""
        return self.scheme == "ssh"


@dataclass(frozen=True, slots=True)
class PushAuthorizationProjection:
    """Sanitized policy disclosure shown before any remote contact."""

    destination: PushDestinationProjection

    def __post_init__(self) -> None:
        if type(self.destination) is not PushDestinationProjection:
            raise PushContractError("unsafe_text")

    @property
    def action_label(self) -> str:
        """Return the explicit authorization action copy."""
        return "Authorize and check"

    @property
    def terminal_prompts_disabled(self) -> bool:
        """Return the fixed noninteractive policy."""
        return True

    @property
    def helper_contact_possible(self) -> bool:
        """Return whether approved existing helpers may be contacted."""
        return True

    @property
    def trusts_remote_content(self) -> bool:
        """Return whether authorization asserts trust in remote content."""
        return False


@dataclass(frozen=True, slots=True)
class PushDestinationPolicyResult:
    """Sanitized result of local-only candidate and destination proof."""

    state: PushDestinationPolicyState
    message: str
    authorization: PushAuthorizationProjection | None = None

    def __post_init__(self) -> None:
        expected = _DESTINATION_POLICY_MESSAGES.get(self.state)
        if (
            expected is None
            or self.message != expected
            or (
                (self.state == "ready")
                != (type(self.authorization) is PushAuthorizationProjection)
            )
        ):
            raise PushContractError("unsafe_text")

    @property
    def remote_contact_started(self) -> bool:
        """Return the fixed local-only boundary for this task."""
        return False


@dataclass(frozen=True, slots=True)
class PushReviewProjection:
    """Sanitized immutable final review of one exact remote update."""

    candidate: PushCandidateProjection
    destination: PushDestinationProjection
    configured_remote_label: str

    def __post_init__(self) -> None:
        if (
            type(self.candidate) is not PushCandidateProjection
            or type(self.destination) is not PushDestinationProjection
            or not _is_safe_display_text(self.configured_remote_label)
            or not self.configured_remote_label.strip()
        ):
            raise PushContractError("unsafe_text")

    @property
    def exact_lease(self) -> str:
        """Return the exact destination-to-parent compare-and-swap."""
        return f"{self.destination.destination_ref}:{self.candidate.parent_oid}"

    @property
    def exact_refspec(self) -> str:
        """Return the exact candidate-to-destination refspec."""
        return f"{self.candidate.candidate_oid}:{self.destination.destination_ref}"

    @property
    def hooks_bypassed(self) -> bool:
        """Return the fixed local pre-push-hook policy."""
        return True

    @property
    def later_note_edits_remain_local(self) -> bool:
        """Return whether later note edits can broaden this review."""
        return True


@dataclass(frozen=True, slots=True)
class PushOutcomeProjection:
    """Bounded point-in-time copy for one guarded-push outcome."""

    state: PushOutcomeState
    title: str
    message: str
    recovery_available: bool = False

    def __post_init__(self) -> None:
        expected = _OUTCOME_COPY.get(self.state) if type(self.state) is str else None
        if (
            expected is None
            or type(self.title) is not str
            or type(self.message) is not str
            or type(self.recovery_available) is not bool
            or (self.title, self.message, self.recovery_available) != expected
        ):
            raise PushContractError("unsafe_text")

    @property
    def point_in_time(self) -> bool:
        """Return that the copy describes only a current observation."""
        return True


@dataclass(frozen=True, slots=True)
class PushRecoveryProjection:
    """Sanitized query-only recovery projection for an uncertain attempt."""

    destination: PushDestinationProjection
    state: PushRecoveryState
    title: str
    message: str
    can_check_again: bool

    def __post_init__(self) -> None:
        if (
            type(self.destination) is not PushDestinationProjection
            or type(self.state) is not str
            or type(self.title) is not str
            or type(self.message) is not str
            or type(self.can_check_again) is not bool
            or (
                self.state,
                self.title,
                self.message,
                self.can_check_again,
            )
            not in _RECOVERY_COPY_VALUES
        ):
            raise PushContractError("unsafe_text")

    @property
    def query_only(self) -> bool:
        """Return that recovery never sends another update."""
        return True


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PushAuthorizationHandle:
    """Opaque identity-only capability for one destination authorization."""

    def __new__(cls) -> PushAuthorizationHandle:
        raise TypeError("Push authorization handles have no public constructor.")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PushReviewHandle:
    """Opaque identity-only single-use capability for one push review."""

    def __new__(cls) -> PushReviewHandle:
        raise TypeError("Push review handles have no public constructor.")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class PushRecoveryHandle:
    """Opaque identity-only capability for one query-only recovery check."""

    def __new__(cls) -> PushRecoveryHandle:
        raise TypeError("Push recovery handles have no public constructor.")


@dataclass(frozen=True, slots=True)
class RemoteRefObservation:
    """Closed classification of one exact ``ls-remote --refs`` response."""

    state: RemoteRefState
    observed_oid: str | None = None

    def __post_init__(self) -> None:
        if self.state not in _REMOTE_REF_STATES:
            raise PushContractError("invalid_object_id")
        if self.state in {"parent", "candidate", "divergent"}:
            _validate_object_id(self.observed_oid)
        elif self.observed_oid is not None:
            raise PushContractError("invalid_object_id")


@dataclass(frozen=True, slots=True)
class PushPorcelainResult:
    """Closed classification of exact push-porcelain destination output."""

    state: PushPorcelainState

    def __post_init__(self) -> None:
        if self.state not in _PORCELAIN_STATES:
            raise PushContractError("unsafe_text")


class PushDiagnosticCategory(str, Enum):
    """Closed sanitized categories for discarded remote diagnostic bytes."""

    AUTHENTICATION_FAILED = "authentication_failed"
    HOST_VERIFICATION_FAILED = "host_verification_failed"
    REMOTE_REJECTED = "remote_rejected"
    TRANSPORT_FAILED = "transport_failed"
    INVALID_RESPONSE = "invalid_response"
    UNKNOWN_FAILURE = "unknown_failure"


_DIAGNOSTIC_MESSAGES: dict[PushDiagnosticCategory, str] = {
    PushDiagnosticCategory.AUTHENTICATION_FAILED: (
        "Existing noninteractive authentication was not accepted."
    ),
    PushDiagnosticCategory.HOST_VERIFICATION_FAILED: (
        "The SSH host identity could not be verified."
    ),
    PushDiagnosticCategory.REMOTE_REJECTED: (
        "The destination rejected the exact requested ref update."
    ),
    PushDiagnosticCategory.TRANSPORT_FAILED: (
        "The configured destination could not be reached securely."
    ),
    PushDiagnosticCategory.INVALID_RESPONSE: (
        "The destination returned a response Chatbook could not verify."
    ),
    PushDiagnosticCategory.UNKNOWN_FAILURE: (
        "The remote operation failed without a safe diagnostic category."
    ),
}


@dataclass(frozen=True, slots=True)
class PushDiagnostic:
    """Sanitized bounded diagnostic after raw bytes have been discarded."""

    category: PushDiagnosticCategory
    message: str

    def __post_init__(self) -> None:
        if (
            type(self.category) is not PushDiagnosticCategory
            or type(self.message) is not str
            or self.message != _DIAGNOSTIC_MESSAGES[self.category]
        ):
            raise PushContractError("unsafe_text")


class _FrozenPushEndpoint:
    """Private immutable effective endpoint paired with its safe projection."""

    __slots__ = ("__weakref__", "projection")

    def __new__(cls) -> _FrozenPushEndpoint:
        raise TypeError("Frozen push endpoints have no public constructor.")

    def __repr__(self) -> str:
        return "_FrozenPushEndpoint(<opaque>)"

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot assign to frozen push endpoint")


class TransportAdmission:
    """Immutable production admission for secure network transports only."""

    __slots__ = ("_test_local_bare",)

    def __init__(self) -> None:
        """Create the only production transport policy."""
        object.__setattr__(self, "_test_local_bare", False)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot change transport admission")


@dataclass(frozen=True, slots=True)
class _AdmittedPushTransport:
    """Service-private endpoint admission without exposing configured text."""

    configured_identity: str
    destination: PushDestinationProjection
    endpoint: _FrozenPushEndpoint
    test_local_bare: bool = False


@dataclass(frozen=True, slots=True)
class _GitConfigFact:
    """One relevant Git config fact with a pre-hashed source identity."""

    scope: str
    origin_identity: str = field(repr=False)
    key: str
    value: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            self.scope not in {"system", "global", "local", "worktree"}
            or len(self.origin_identity) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.origin_identity
            )
            or not self.key
            or _contains_unsafe_text(self.key)
            or _contains_unsafe_text(self.value)
        ):
            raise PushContractError("invalid_configuration")


@dataclass(frozen=True, slots=True)
class _ResolvedPushConfiguration:
    """Service-private exact configured destination and its fingerprint."""

    tracking_remote: str = field(repr=False)
    merge_ref: str
    endpoint_source: Literal["pushurl", "url"]
    configuration_fingerprint: str
    transport: _AdmittedPushTransport


def _local_bare_transport_admission_for_tests() -> TransportAdmission:
    """Issue an explicit nonproduction local-bare admission for tests."""
    admission = object.__new__(TransportAdmission)
    object.__setattr__(admission, "_test_local_bare", True)
    return admission


def _canonical_test_local_bare_path(effective_endpoint: str) -> str:
    """Return one canonical absolute test-only path or refuse it."""
    if type(effective_endpoint) is not str:
        raise PushContractError("invalid_endpoint")
    local_path = effective_endpoint
    if local_path.startswith("file://"):
        local_path = local_path.removeprefix("file://")
    if (
        not local_path.startswith("/")
        or local_path.startswith("//")
        or local_path != os.path.normpath(local_path)
        or local_path != os.path.abspath(local_path)
        or _contains_unsafe_text(local_path)
        or any(character in local_path for character in ("\0", "?", "#"))
    ):
        raise PushContractError("invalid_endpoint")
    return local_path


def _admit_push_transport(
    admission: TransportAdmission,
    effective_endpoint: str,
    destination_ref: str,
) -> _AdmittedPushTransport:
    """Admit one configured endpoint under an immutable transport policy."""
    if type(admission) is not TransportAdmission:
        raise PushContractError("invalid_endpoint")
    try:
        allow_local_bare = admission._test_local_bare
    except AttributeError:
        raise PushContractError("invalid_endpoint") from None

    if allow_local_bare:
        local_path = _canonical_test_local_bare_path(effective_endpoint)
        endpoint = _freeze_test_local_bare_endpoint(
            admission,
            local_path,
            destination_ref,
        )
        _value, destination = _read_frozen_endpoint(endpoint)
        configured_identity = hashlib.sha256(
            (
                "test-local-bare\0"
                + local_path
                + "\0"
                + destination_ref
            ).encode("utf-8")
        ).hexdigest()
        return _AdmittedPushTransport(
            configured_identity=configured_identity,
            destination=destination,
            endpoint=endpoint,
            test_local_bare=True,
        )

    endpoint = _freeze_push_endpoint(effective_endpoint, destination_ref)
    normalized, destination = _read_frozen_endpoint(endpoint)
    configured_identity = hashlib.sha256(
        (normalized + "\0" + destination_ref).encode("utf-8")
    ).hexdigest()
    return _AdmittedPushTransport(
        configured_identity=configured_identity,
        destination=destination,
        endpoint=endpoint,
    )


def _resolve_push_configuration(
    facts: tuple[_GitConfigFact, ...],
    local_branch_ref: str,
    admission: TransportAdmission,
) -> _ResolvedPushConfiguration:
    """Resolve one frozen destination from bounded relevant config facts."""
    validate_destination_ref(local_branch_ref)
    if (
        type(facts) is not tuple
        or any(type(fact) is not _GitConfigFact for fact in facts)
        or type(admission) is not TransportAdmission
    ):
        raise PushContractError("invalid_configuration")
    branch_name = local_branch_ref.removeprefix("refs/heads/")
    tracking = _config_values(facts, "branch", branch_name, "remote")
    merges = _config_values(facts, "branch", branch_name, "merge")
    if (
        len(tracking) != 1
        or len(merges) != 1
        or tracking[0] == "."
        or not tracking[0].strip()
    ):
        raise PushContractError("invalid_configuration")
    tracking_remote = tracking[0]
    try:
        merge_ref = validate_destination_ref(merges[0])
    except PushContractError:
        raise PushContractError("invalid_configuration") from None

    branch_push_remote = _config_values(
        facts,
        "branch",
        branch_name,
        "pushremote",
    )
    push_default = _config_values(facts, "remote", None, "pushdefault")
    if len(branch_push_remote) > 1 or len(push_default) > 1:
        raise PushContractError("invalid_configuration")
    selected_remote = (
        branch_push_remote[0]
        if branch_push_remote
        else push_default[0] if push_default else tracking_remote
    )
    if selected_remote != tracking_remote:
        raise PushContractError("invalid_configuration")

    _validate_push_security_facts(facts, tracking_remote)
    push_urls = _config_values(
        facts,
        "remote",
        tracking_remote,
        "pushurl",
    )
    fetch_urls = _config_values(
        facts,
        "remote",
        tracking_remote,
        "url",
    )
    if len(push_urls) > 1 or (not push_urls and len(fetch_urls) != 1):
        raise PushContractError("invalid_configuration")
    endpoint_source: Literal["pushurl", "url"] = (
        "pushurl" if push_urls else "url"
    )
    configured_endpoint = push_urls[0] if push_urls else fetch_urls[0]
    effective_endpoint = _rewrite_push_endpoint(
        configured_endpoint,
        facts,
        use_push_rewrites=endpoint_source == "url",
    )
    try:
        transport = _admit_push_transport(
            admission,
            effective_endpoint,
            merge_ref,
        )
    except PushContractError:
        raise PushContractError("invalid_configuration") from None
    return _ResolvedPushConfiguration(
        tracking_remote=tracking_remote,
        merge_ref=merge_ref,
        endpoint_source=endpoint_source,
        configuration_fingerprint=_configuration_fingerprint(
            facts,
            local_branch_ref,
            endpoint_source,
        ),
        transport=transport,
    )


def _config_values(
    facts: tuple[_GitConfigFact, ...],
    section: str,
    subsection: str | None,
    name: str,
) -> tuple[str, ...]:
    return tuple(
        fact.value
        for fact in facts
        if _config_key_matches(fact.key, section, subsection, name)
    )


def _config_key_matches(
    key: str,
    section: str,
    subsection: str | None,
    name: str,
) -> bool:
    components = key.split(".")
    if subsection is None:
        return (
            len(components) == 2
            and components[0].lower() == section
            and components[1].lower() == name
        )
    return (
        len(components) >= 3
        and components[0].lower() == section
        and ".".join(components[1:-1]) == subsection
        and components[-1].lower() == name
    )


def _validate_push_security_facts(
    facts: tuple[_GitConfigFact, ...],
    remote: str,
) -> None:
    remote_blocked_names = {
        "push",
        "pushoption",
        "receivepack",
        "vcs",
    }
    local_security_names = {
        "sslcainfo",
        "sslcapath",
        "sslcert",
        "sslcertpasswordprotected",
        "sslkey",
        "extraheader",
        "proxy",
        "proxyauthmethod",
    }
    for fact in facts:
        lowered = fact.key.lower()
        value = fact.value.strip().lower()
        if (
            lowered == "push.pushoption"
            or any(
                _config_key_matches(fact.key, "remote", remote, name)
                for name in remote_blocked_names
            )
        ):
            raise PushContractError("invalid_configuration")
        if _config_key_matches(fact.key, "remote", remote, "mirror"):
            if value not in {"false", "no", "off", "0"}:
                raise PushContractError("invalid_configuration")
        if (
            (lowered == "http.sslverify" or lowered.endswith(".sslverify"))
            and value in {"false", "no", "off", "0"}
        ):
            raise PushContractError("invalid_configuration")
        if fact.scope not in {"local", "worktree"}:
            continue
        last_name = lowered.rsplit(".", 1)[-1]
        if (
            (
                lowered.startswith("credential.")
                and last_name == "helper"
            )
            or lowered in {"core.sshcommand", "ssh.variant"}
            or (
                lowered.startswith("http.")
                and last_name in local_security_names
            )
            or _config_key_matches(fact.key, "remote", remote, "proxy")
        ):
            raise PushContractError("invalid_configuration")


def _rewrite_push_endpoint(
    endpoint: str,
    facts: tuple[_GitConfigFact, ...],
    *,
    use_push_rewrites: bool,
) -> str:
    push_rules = (
        _rewrite_rules(facts, "pushinsteadof", endpoint)
        if use_push_rewrites
        else ()
    )
    rules = push_rules or _rewrite_rules(facts, "insteadof", endpoint)
    if not rules:
        return endpoint
    longest = max(len(prefix) for prefix, _replacement in rules)
    winners = tuple(rule for rule in rules if len(rule[0]) == longest)
    if len(winners) != 1:
        raise PushContractError("invalid_configuration")
    prefix, replacement = winners[0]
    return replacement + endpoint[len(prefix) :]


def _rewrite_rules(
    facts: tuple[_GitConfigFact, ...],
    name: str,
    endpoint: str,
) -> tuple[tuple[str, str], ...]:
    suffix = f".{name}"
    rules: list[tuple[str, str]] = []
    for fact in facts:
        lowered = fact.key.lower()
        if (
            not lowered.startswith("url.")
            or not lowered.endswith(suffix)
        ):
            continue
        replacement = fact.key[4 : len(fact.key) - len(suffix)]
        if replacement and endpoint.startswith(fact.value):
            rules.append((fact.value, replacement))
    return tuple(rules)


def _configuration_fingerprint(
    facts: tuple[_GitConfigFact, ...],
    local_branch_ref: str,
    endpoint_source: Literal["pushurl", "url"],
) -> str:
    digest = hashlib.sha256()
    for value in (
        "file-notes-push-config-v1",
        local_branch_ref,
        endpoint_source,
        *(
            component
            for fact in facts
            for component in (
                fact.scope,
                fact.origin_identity,
                fact.key,
                fact.value,
            )
        ),
    ):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _make_endpoint_registry() -> tuple[
    Callable[[_FrozenPushEndpoint, str], None],
    Callable[[_FrozenPushEndpoint], str | None],
]:
    values: WeakKeyDictionary[_FrozenPushEndpoint, str] = WeakKeyDictionary()

    def register(endpoint: _FrozenPushEndpoint, value: str) -> None:
        values[endpoint] = value

    def lookup(endpoint: _FrozenPushEndpoint) -> str | None:
        return values.get(endpoint)

    return register, lookup


_register_frozen_endpoint, _lookup_frozen_endpoint = _make_endpoint_registry()
del _make_endpoint_registry


def _make_test_local_endpoint_registry() -> tuple[
    Callable[[_FrozenPushEndpoint, str], None],
    Callable[[_FrozenPushEndpoint], str | None],
]:
    """Create the private possession registry for test-local endpoints."""
    values: WeakKeyDictionary[_FrozenPushEndpoint, str] = WeakKeyDictionary()

    def register(endpoint: _FrozenPushEndpoint, value: str) -> None:
        values[endpoint] = value

    def lookup(endpoint: _FrozenPushEndpoint) -> str | None:
        return values.get(endpoint)

    return register, lookup


(
    _register_test_local_endpoint,
    _lookup_test_local_endpoint,
) = _make_test_local_endpoint_registry()
del _make_test_local_endpoint_registry


def _issue_push_authorization_handle() -> PushAuthorizationHandle:
    """Issue one private identity-only authorization capability."""
    return object.__new__(PushAuthorizationHandle)


def _issue_push_review_handle() -> PushReviewHandle:
    """Issue one private identity-only review capability."""
    return object.__new__(PushReviewHandle)


def _issue_push_recovery_handle() -> PushRecoveryHandle:
    """Issue one private identity-only recovery capability."""
    return object.__new__(PushRecoveryHandle)


def validate_destination_ref(destination_ref: str) -> str:
    """Validate one exact existing ``refs/heads/*`` destination.

    Args:
        destination_ref: Full destination ref.

    Returns:
        The unchanged validated ref.

    Raises:
        PushContractError: If the value is relative, malformed, or unsafe.
    """
    if (
        not isinstance(destination_ref, str)
        or not destination_ref.startswith("refs/heads/")
        or _contains_unsafe_text(destination_ref)
    ):
        raise PushContractError("invalid_destination_ref")
    try:
        destination_ref.encode("utf-8")
    except UnicodeEncodeError:
        raise PushContractError("invalid_destination_ref") from None

    branch = destination_ref.removeprefix("refs/heads/")
    forbidden_characters = frozenset(" ~^:?*[\\")
    if (
        not branch
        or branch.startswith("-")
        or branch.endswith(".")
        or ".." in branch
        or "@{" in branch
        or "//" in branch
        or any(character in forbidden_characters for character in branch)
    ):
        raise PushContractError("invalid_destination_ref")
    components = branch.split("/")
    if any(
        not component
        or component in {".", ".."}
        or component.startswith(".")
        or component.endswith(".lock")
        for component in components
    ):
        raise PushContractError("invalid_destination_ref")
    return destination_ref


def parse_push_endpoint(
    effective_endpoint: str,
    destination_ref: str,
) -> PushDestinationProjection:
    """Parse a strict endpoint into safe selectable destination details.

    The returned projection never contains the effective endpoint string.

    Args:
        effective_endpoint: Exact configured push endpoint.
        destination_ref: Exact existing full destination ref.

    Returns:
        Sanitized endpoint identity for literal display.

    Raises:
        PushContractError: If the endpoint or ref is unsupported or ambiguous.
    """
    _, projection = _parse_push_endpoint(effective_endpoint, destination_ref)
    return projection


def _freeze_push_endpoint(
    effective_endpoint: str,
    destination_ref: str,
) -> _FrozenPushEndpoint:
    """Create the private endpoint value used by exact command builders."""
    normalized, projection = _parse_push_endpoint(
        effective_endpoint,
        destination_ref,
    )
    frozen = object.__new__(_FrozenPushEndpoint)
    object.__setattr__(frozen, "projection", projection)
    _register_frozen_endpoint(frozen, normalized)
    return frozen


def _freeze_test_local_bare_endpoint(
    admission: TransportAdmission,
    local_path: str,
    destination_ref: str,
) -> _FrozenPushEndpoint:
    """Issue an opaque local endpoint only from the private test capability."""
    try:
        allowed = (
            type(admission) is TransportAdmission
            and admission._test_local_bare is True
        )
    except AttributeError:
        allowed = False
    if not allowed:
        raise PushContractError("invalid_endpoint")
    canonical_path = _canonical_test_local_bare_path(local_path)
    validated_ref = validate_destination_ref(destination_ref)
    projection = PushDestinationProjection(
        scheme="https",
        host="local-test.invalid",
        port=443,
        repository_path="/test-only",
        destination_ref=validated_ref,
    )
    frozen = object.__new__(_FrozenPushEndpoint)
    object.__setattr__(frozen, "projection", projection)
    _register_test_local_endpoint(frozen, canonical_path)
    return frozen


def parse_ls_remote_refs(
    payload: bytes,
    destination_ref: str,
    parent_oid: str,
    candidate_oid: str,
) -> RemoteRefObservation:
    """Classify exact ``ls-remote --refs`` output and discard its bytes.

    Args:
        payload: Complete bounded stdout.
        destination_ref: Queried full destination ref.
        parent_oid: Exact reviewed parent.
        candidate_oid: Exact guarded candidate.

    Returns:
        A closed parent, candidate, missing, divergent, or malformed state.

    Raises:
        PushContractError: If caller-supplied authority is invalid.
    """
    validate_destination_ref(destination_ref)
    _validate_oid_pair(parent_oid, candidate_oid)
    if not isinstance(payload, bytes) or len(payload) > _MAX_REMOTE_OUTPUT_BYTES:
        return RemoteRefObservation("malformed")
    if not payload:
        return RemoteRefObservation("missing")
    if not payload.endswith(b"\n"):
        return RemoteRefObservation("malformed")

    records = payload[:-1].split(b"\n")
    if len(records) != 1:
        return RemoteRefObservation("malformed")
    fields = records[0].split(b"\t")
    if len(fields) != 2:
        return RemoteRefObservation("malformed")
    oid_bytes, ref_bytes = fields
    if ref_bytes != destination_ref.encode("utf-8"):
        return RemoteRefObservation("malformed")
    try:
        observed_oid = oid_bytes.decode("ascii")
    except UnicodeDecodeError:
        return RemoteRefObservation("malformed")
    if not _is_valid_object_id(observed_oid) or len(observed_oid) != len(parent_oid):
        return RemoteRefObservation("malformed")
    if observed_oid == parent_oid:
        return RemoteRefObservation("parent", observed_oid)
    if observed_oid == candidate_oid:
        return RemoteRefObservation("candidate", observed_oid)
    return RemoteRefObservation("divergent", observed_oid)


def parse_push_porcelain(
    payload: bytes,
    candidate_oid: str,
    destination_ref: str,
) -> PushPorcelainResult:
    """Classify one exact push-porcelain destination record.

    Raw endpoint headers and summary bytes are deliberately discarded.

    Args:
        payload: Complete bounded stdout from ``git push --porcelain``.
        candidate_oid: Exact pushed commit object ID.
        destination_ref: Exact full destination ref.

    Returns:
        Accepted, rejected, or malformed closed classification.

    Raises:
        PushContractError: If caller-supplied authority is invalid.
    """
    _validate_object_id(candidate_oid)
    validate_destination_ref(destination_ref)
    if (
        not isinstance(payload, bytes)
        or not payload
        or len(payload) > _MAX_REMOTE_OUTPUT_BYTES
        or not payload.endswith(b"\n")
    ):
        return PushPorcelainResult("malformed")

    status_lines = [
        line
        for line in payload[:-1].split(b"\n")
        if len(line) >= 2 and line[1:2] == b"\t"
    ]
    if len(status_lines) != 1:
        return PushPorcelainResult("malformed")
    fields = status_lines[0].split(b"\t")
    if len(fields) != 3 or not fields[2]:
        return PushPorcelainResult("malformed")
    expected_refspec = f"{candidate_oid}:{destination_ref}".encode("utf-8")
    if fields[1] != expected_refspec:
        return PushPorcelainResult("malformed")
    if fields[0] == b" ":
        return PushPorcelainResult("accepted")
    if fields[0] == b"!":
        return PushPorcelainResult("rejected")
    return PushPorcelainResult("malformed")


def classify_push_diagnostic(payload: bytes) -> PushDiagnostic:
    """Classify raw remote diagnostics into bounded static copy.

    Args:
        payload: Raw Git, SSH, helper, or server diagnostic bytes.

    Returns:
        A closed category and static message. No raw byte is retained.
    """
    sample = (
        payload[:_MAX_REMOTE_OUTPUT_BYTES].lower()
        if isinstance(payload, bytes)
        else b""
    )
    if _contains_any(
        sample,
        (
            b"host key verification failed",
            b"remote host identification has changed",
        ),
    ):
        category = PushDiagnosticCategory.HOST_VERIFICATION_FAILED
    elif _contains_any(
        sample,
        (
            b"permission denied",
            b"authentication failed",
            b"could not read username",
            b"publickey",
        ),
    ):
        category = PushDiagnosticCategory.AUTHENTICATION_FAILED
    elif _contains_any(
        sample,
        (
            b"pre-receive hook declined",
            b"remote rejected",
            b"[remote rejected]",
            b"failed to push some refs",
            b"stale info",
        ),
    ):
        category = PushDiagnosticCategory.REMOTE_REJECTED
    elif _contains_any(
        sample,
        (
            b"could not resolve host",
            b"connection refused",
            b"connection reset",
            b"connection timed out",
            b"network is unreachable",
            b"tls",
            b"ssl",
        ),
    ):
        category = PushDiagnosticCategory.TRANSPORT_FAILED
    elif _contains_any(
        sample,
        (
            b"protocol error",
            b"bad line length",
            b"invalid packet",
            b"unexpected disconnect",
        ),
    ):
        category = PushDiagnosticCategory.INVALID_RESPONSE
    else:
        category = PushDiagnosticCategory.UNKNOWN_FAILURE
    return PushDiagnostic(category, _DIAGNOSTIC_MESSAGES[category])


def _build_push_query_argv(
    git_executable: str,
    private_network_git_dir: str,
    endpoint: _FrozenPushEndpoint,
) -> tuple[str, ...]:
    """Build the exact read-only query vector for the frozen destination.

    Args:
        git_executable: Git executable selected by the owning service.
        private_network_git_dir: Isolated owner-only bare Git directory.
        endpoint: Private frozen endpoint and destination identity.

    Returns:
        Direct argument vector with no remote-name or implicit-ref behavior.

    Raises:
        PushContractError: If the private command context is invalid.
    """
    prefix = _build_network_git_prefix(
        git_executable,
        private_network_git_dir,
    )
    value, projection = _read_frozen_endpoint(endpoint)
    return (
        *prefix,
        "ls-remote",
        "--refs",
        "--",
        value,
        projection.destination_ref,
    )


def _build_push_argv(
    git_executable: str,
    private_network_git_dir: str,
    endpoint: _FrozenPushEndpoint,
    parent_oid: str,
    candidate_oid: str,
) -> tuple[str, ...]:
    """Build the exact one-commit compare-and-swap push vector.

    Args:
        git_executable: Git executable selected by the owning service.
        private_network_git_dir: Isolated owner-only bare Git directory.
        endpoint: Private frozen endpoint and destination identity.
        parent_oid: Exact reviewed remote parent.
        candidate_oid: Exact guarded candidate commit.

    Returns:
        Direct argument vector with explicit lease, refspec, and hook bypass.

    Raises:
        PushContractError: If command authority is incomplete or malformed.
    """
    _validate_oid_pair(parent_oid, candidate_oid)
    prefix = _build_network_git_prefix(
        git_executable,
        private_network_git_dir,
    )
    value, projection = _read_frozen_endpoint(endpoint)
    destination_ref = projection.destination_ref
    return (
        *prefix,
        "push",
        "--porcelain",
        "--no-verify",
        "--no-follow-tags",
        "--recurse-submodules=no",
        f"--force-with-lease={destination_ref}:{parent_oid}",
        "--",
        value,
        f"{candidate_oid}:{destination_ref}",
    )


def push_outcome_copy(state: PushOutcomeState) -> PushOutcomeProjection:
    """Return fixed honest point-in-time copy for one proven outcome state."""
    try:
        title, message, recovery_available = _OUTCOME_COPY[state]
    except (KeyError, TypeError):
        raise PushContractError("unsafe_text") from None
    return PushOutcomeProjection(
        state,
        title,
        message,
        recovery_available,
    )


def push_recovery_copy(
    destination: PushDestinationProjection,
    observation: RemoteRefObservation,
) -> PushRecoveryProjection:
    """Return query-only copy for a current retained-destination observation."""
    if observation.state == "candidate":
        copy_state = "candidate"
    elif observation.state == "parent":
        copy_state = "parent"
    elif observation.state == "malformed":
        copy_state = "unprovable"
    else:
        copy_state = "needs_attention"
    state, title, message, can_check_again = _RECOVERY_COPY[copy_state]
    return PushRecoveryProjection(
        destination,
        state,
        title,
        message,
        can_check_again,
    )


def _push_destination_policy_result(
    state: PushDestinationPolicyState,
    destination: PushDestinationProjection | None = None,
) -> PushDestinationPolicyResult:
    authorization = (
        PushAuthorizationProjection(destination)
        if state == "ready" and destination is not None
        else None
    )
    return PushDestinationPolicyResult(
        state,
        _DESTINATION_POLICY_MESSAGES[state],
        authorization,
    )


def _parse_push_endpoint(
    effective_endpoint: str,
    destination_ref: str,
) -> tuple[str, PushDestinationProjection]:
    validate_destination_ref(destination_ref)
    if (
        not isinstance(effective_endpoint, str)
        or not effective_endpoint
        or effective_endpoint != effective_endpoint.strip()
        or _contains_unsafe_text(effective_endpoint)
        or "\\" in effective_endpoint
    ):
        raise PushContractError("invalid_endpoint")

    if effective_endpoint.startswith(("https://", "ssh://")):
        return _parse_url_endpoint(effective_endpoint, destination_ref)
    return _parse_scp_endpoint(effective_endpoint, destination_ref)


def _parse_url_endpoint(
    effective_endpoint: str,
    destination_ref: str,
) -> tuple[str, PushDestinationProjection]:
    if "?" in effective_endpoint or "#" in effective_endpoint:
        raise PushContractError("invalid_endpoint")
    try:
        parsed = urlsplit(effective_endpoint)
        parsed_port = parsed.port
        host = parsed.hostname
    except ValueError:
        raise PushContractError("invalid_endpoint") from None
    if (
        parsed.scheme not in {"https", "ssh"}
        or not parsed.netloc
        or parsed.netloc.rsplit("@", 1)[-1].endswith(":")
        or not host
        or parsed.query
        or parsed.fragment
        or parsed.path == ""
    ):
        raise PushContractError("invalid_endpoint")

    normalized_host = _normalize_host(host)
    normalized_path = _normalize_url_repository_path(parsed.path)
    if parsed.scheme == "https":
        if (
            parsed.username is not None
            or parsed.password is not None
            or "@" in parsed.netloc
        ):
            raise PushContractError("invalid_endpoint")
        port = parsed_port if parsed_port is not None else 443
        ssh_user = None
    else:
        if parsed.password is not None or not _is_valid_ssh_user(parsed.username):
            raise PushContractError("invalid_endpoint")
        port = parsed_port if parsed_port is not None else 22
        ssh_user = parsed.username
    if not 1 <= port <= 65535:
        raise PushContractError("invalid_endpoint")

    normalized = _normalized_url(
        parsed,
        normalized_host,
        normalized_path,
        parsed_port,
        ssh_user,
    )
    projection = PushDestinationProjection(
        scheme=parsed.scheme,
        host=normalized_host,
        port=port,
        repository_path=normalized_path,
        destination_ref=destination_ref,
        ssh_user=ssh_user,
    )
    return normalized, projection


def _parse_scp_endpoint(
    effective_endpoint: str,
    destination_ref: str,
) -> tuple[str, PushDestinationProjection]:
    match = _SCP_ENDPOINT_PATTERN.fullmatch(effective_endpoint)
    if match is None:
        raise PushContractError("invalid_endpoint")
    user = match.group("user")
    host = match.group("host")
    raw_path = match.group("path")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not _is_valid_ssh_user(user):
        raise PushContractError("invalid_endpoint")
    if raw_path.startswith("~") and not raw_path.startswith("~/"):
        raise PushContractError("invalid_endpoint")
    normalized_host = _normalize_host(host)
    _validate_repository_path(raw_path, require_absolute=False)
    display_path = raw_path if raw_path.startswith(("/", "~/")) else f"~/{raw_path}"
    normalized = f"{user}@{_render_host(normalized_host)}:{raw_path}"
    return normalized, PushDestinationProjection(
        scheme="ssh",
        host=normalized_host,
        port=22,
        repository_path=display_path,
        destination_ref=destination_ref,
        ssh_user=user,
    )


def _normalized_url(
    parsed: SplitResult,
    host: str,
    path: str,
    parsed_port: int | None,
    ssh_user: str | None,
) -> str:
    authority = _render_host(host)
    if ssh_user is not None:
        authority = f"{ssh_user}@{authority}"
    if parsed_port is not None:
        authority = f"{authority}:{parsed_port}"
    return f"{parsed.scheme}://{authority}{path}"


def _normalize_host(host: str) -> str:
    if (
        not isinstance(host, str)
        or not host
        or host.startswith(".")
        or host.endswith(".")
        or _contains_unsafe_text(host)
    ):
        raise PushContractError("invalid_endpoint")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            normalized = host.encode("idna").decode("ascii").lower()
        except (UnicodeError, ValueError):
            raise PushContractError("invalid_endpoint") from None
        if len(normalized) > 253:
            raise PushContractError("invalid_endpoint")
        labels = normalized.split(".")
        if any(
            not label or len(label) > 63 or _DNS_LABEL_PATTERN.fullmatch(label) is None
            for label in labels
        ):
            raise PushContractError("invalid_endpoint")
        return normalized
    if isinstance(address, ipaddress.IPv6Address) and address.scope_id is not None:
        raise PushContractError("invalid_endpoint")
    return address.compressed.lower()


def _render_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _normalize_url_repository_path(path: str) -> str:
    _validate_repository_path(path, require_absolute=True)
    return path


def _validate_repository_path(
    path: str,
    *,
    require_absolute: bool,
) -> None:
    if (
        not isinstance(path, str)
        or not path
        or _contains_unsafe_text(path)
        or _REPOSITORY_PATH_PATTERN.fullmatch(path) is None
        or path.endswith("/")
        or "//" in path
        or (require_absolute and not path.startswith("/"))
    ):
        raise PushContractError("invalid_endpoint")
    components = path.lstrip("/").split("/")
    if any(
        not component or component in {".", ".."} or component.startswith("-")
        for component in components
    ):
        raise PushContractError("invalid_endpoint")


def _is_safe_display_repository_path(path: str) -> bool:
    try:
        raw_path = path[2:] if path.startswith("~/") else path
        _validate_repository_path(
            raw_path,
            require_absolute=path.startswith("/"),
        )
    except PushContractError:
        return False
    return path.startswith(("/", "~/"))


def _is_valid_ssh_user(user: str | None) -> bool:
    return (
        isinstance(user, str)
        and bool(user)
        and not user.startswith("-")
        and _SSH_USER_PATTERN.fullmatch(user) is not None
    )


def _build_network_git_prefix(
    git_executable: str,
    private_network_git_dir: str,
) -> tuple[str, ...]:
    if any(
        not isinstance(value, str) or not value or "\0" in value
        for value in (git_executable, private_network_git_dir)
    ):
        raise PushContractError("invalid_command_context")
    return (
        git_executable,
        f"--git-dir={private_network_git_dir}",
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "maintenance.auto=false",
        "-c",
        "gc.auto=0",
    )


def _read_frozen_endpoint(
    endpoint: _FrozenPushEndpoint,
) -> tuple[str, PushDestinationProjection]:
    if type(endpoint) is not _FrozenPushEndpoint:
        raise PushContractError("invalid_endpoint")
    value = _lookup_frozen_endpoint(endpoint)
    if value is None:
        local_path = _lookup_test_local_endpoint(endpoint)
        try:
            projection = endpoint.projection
            valid_test_projection = (
                type(projection) is PushDestinationProjection
                and projection.scheme == "https"
                and projection.host == "local-test.invalid"
                and projection.port == 443
                and projection.repository_path == "/test-only"
                and projection.ssh_user is None
                and validate_destination_ref(projection.destination_ref)
                == projection.destination_ref
            )
        except (AttributeError, PushContractError):
            valid_test_projection = False
        if (
            local_path is None
            or not valid_test_projection
            or _canonical_test_local_bare_path(local_path) != local_path
        ):
            raise PushContractError("invalid_endpoint")
        return local_path, projection
    try:
        normalized, projection = _parse_push_endpoint(
            value,
            endpoint.projection.destination_ref,
        )
    except (AttributeError, PushContractError):
        raise PushContractError("invalid_endpoint") from None
    if normalized != value or projection != endpoint.projection:
        raise PushContractError("invalid_endpoint")
    return value, projection


def _validate_oid_pair(parent_oid: str, candidate_oid: str) -> None:
    _validate_object_id(parent_oid)
    _validate_object_id(candidate_oid)
    if len(parent_oid) != len(candidate_oid) or parent_oid == candidate_oid:
        raise PushContractError("invalid_object_id")


def _validate_object_id(value: str | None) -> None:
    if not isinstance(value, str) or not _is_valid_object_id(value):
        raise PushContractError("invalid_object_id")


def _is_valid_object_id(value: str) -> bool:
    return _OBJECT_ID_PATTERN.fullmatch(value) is not None and any(
        character != "0" for character in value
    )


def _is_safe_display_text(value: str) -> bool:
    return isinstance(value, str) and not _contains_unsafe_text(value)


def _contains_unsafe_text(value: str) -> bool:
    if not isinstance(value, str):
        return True
    for character in value:
        codepoint = ord(character)
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs"} or codepoint in {0x2028, 0x2029}:
            return True
    return False


def _contains_any(payload: bytes, needles: tuple[bytes, ...]) -> bool:
    return any(needle in payload for needle in needles)


__all__ = [
    "PushAuthorizationHandle",
    "PushAuthorizationProjection",
    "PushCandidateProjection",
    "PushContractError",
    "PushDestinationPolicyResult",
    "PushDestinationProjection",
    "PushDiagnostic",
    "PushDiagnosticCategory",
    "PushIncludedNote",
    "PushOutcomeProjection",
    "PushPorcelainResult",
    "PushRecoveryHandle",
    "PushRecoveryProjection",
    "PushReviewHandle",
    "PushReviewProjection",
    "RemoteRefObservation",
    "TransportAdmission",
    "classify_push_diagnostic",
    "parse_ls_remote_refs",
    "parse_push_endpoint",
    "parse_push_porcelain",
    "push_outcome_copy",
    "push_recovery_copy",
    "validate_destination_ref",
]
