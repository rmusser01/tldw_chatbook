"""Pure contracts, parsers, and command builders for guarded File Notes push."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Callable
from dataclasses import FrozenInstanceError, dataclass
from enum import Enum
from typing import Literal
from urllib.parse import SplitResult, urlsplit
from weakref import WeakKeyDictionary

PushContractErrorCode = Literal[
    "invalid_destination_ref",
    "invalid_endpoint",
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

_ERROR_MESSAGES: dict[PushContractErrorCode, str] = {
    "invalid_destination_ref": "The destination branch ref is not allowed.",
    "invalid_endpoint": "The configured push endpoint is not allowed.",
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
_BIDI_CONTROL_CODEPOINTS = frozenset(
    {
        0x061C,
        0x200E,
        0x200F,
        *range(0x202A, 0x202F),
        *range(0x2066, 0x206A),
    }
)
_REMOTE_REF_STATES = frozenset(
    {"parent", "candidate", "missing", "divergent", "malformed"}
)
_PORCELAIN_STATES = frozenset({"accepted", "rejected", "malformed"})
_OUTCOME_STATES = frozenset(
    {
        "already_published",
        "succeeded",
        "failed_no_update_observed",
        "uncertain",
    }
)
_RECOVERY_STATES = frozenset({"succeeded", "uncertain", "needs_attention"})


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
    """Sanitized provenance for one note included by the guarded commit."""

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
    """Sanitized immutable projection of one exact guarded-push candidate."""

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
class PushReviewProjection:
    """Sanitized immutable final review of one exact remote update."""

    candidate: PushCandidateProjection
    destination: PushDestinationProjection

    def __post_init__(self) -> None:
        if (
            type(self.candidate) is not PushCandidateProjection
            or type(self.destination) is not PushDestinationProjection
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
        if (
            self.state not in _OUTCOME_STATES
            or not _is_safe_display_text(self.title)
            or not _is_safe_display_text(self.message)
            or type(self.recovery_available) is not bool
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
            or type(self.can_check_again) is not bool
            or self.state not in _RECOVERY_STATES
            or not _is_safe_display_text(self.title)
            or not _is_safe_display_text(self.message)
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


class _FrozenPushEndpoint:
    """Private immutable effective endpoint paired with its safe projection."""

    __slots__ = ("__weakref__", "projection")

    def __new__(cls) -> _FrozenPushEndpoint:
        raise TypeError("Frozen push endpoints have no public constructor.")

    def __repr__(self) -> str:
        return "_FrozenPushEndpoint(<opaque>)"

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot assign to frozen push endpoint")


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


def build_push_query_argv(
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


def build_push_argv(
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
    if state == "already_published":
        return PushOutcomeProjection(
            state,
            "Already published",
            (
                "The configured destination currently points to this commit. "
                "No push was started by Chatbook."
            ),
        )
    if state == "succeeded":
        return PushOutcomeProjection(
            state,
            "Succeeded",
            (
                "Git reported the exact update accepted, and the configured "
                "destination currently points to this commit."
            ),
        )
    if state == "failed_no_update_observed":
        return PushOutcomeProjection(
            state,
            "Failed with no update currently observed",
            (
                "Git reported an unsuccessful push, every owned process ended, "
                "and the configured destination currently points to the "
                "reviewed parent. Remote-side work may still be pending or may "
                "occur later."
            ),
        )
    if state == "uncertain":
        return PushOutcomeProjection(
            state,
            "Uncertain",
            (
                "Chatbook cannot currently prove whether the destination "
                "accepted the update. Do not push again automatically. Check "
                "the original destination again without pushing."
            ),
            recovery_available=True,
        )
    raise PushContractError("unsafe_text")


def push_recovery_copy(
    destination: PushDestinationProjection,
    observation: RemoteRefObservation,
) -> PushRecoveryProjection:
    """Return query-only copy for a current retained-destination observation."""
    no_push_copy = " No push was sent by this check."
    if observation.state == "candidate":
        return PushRecoveryProjection(
            destination,
            "succeeded",
            "Succeeded",
            (
                "A query-only check currently observes the candidate at the "
                "original destination. The observation does not establish the "
                f"cause of the update.{no_push_copy}"
            ),
            can_check_again=False,
        )
    if observation.state == "parent":
        return PushRecoveryProjection(
            destination,
            "uncertain",
            "Uncertain",
            (
                "A query-only check currently observes the reviewed parent, so "
                "the prior attempt remains uncertain. Remote-side work may "
                f"still be pending.{no_push_copy}"
            ),
            can_check_again=True,
        )
    return PushRecoveryProjection(
        destination,
        "needs_attention",
        "Needs attention",
        (
            "A query-only check currently cannot prove the candidate at the "
            f"original destination.{no_push_copy}"
        ),
        can_check_again=True,
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
        raise PushContractError("invalid_endpoint")
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
        if (
            category in {"Cc", "Cs"}
            or codepoint in _BIDI_CONTROL_CODEPOINTS
            or codepoint in {0x2028, 0x2029, 0xFEFF}
        ):
            return True
    return False


def _contains_any(payload: bytes, needles: tuple[bytes, ...]) -> bool:
    return any(needle in payload for needle in needles)


__all__ = [
    "PushAuthorizationHandle",
    "PushAuthorizationProjection",
    "PushCandidateProjection",
    "PushContractError",
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
    "build_push_argv",
    "build_push_query_argv",
    "classify_push_diagnostic",
    "parse_ls_remote_refs",
    "parse_push_endpoint",
    "parse_push_porcelain",
    "push_outcome_copy",
    "push_recovery_copy",
    "validate_destination_ref",
]
