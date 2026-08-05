"""Pure contracts and byte parsers for guarded File Notes commits."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

CommitContractErrorCode = Literal[
    "subject_required",
    "subject_too_long",
    "subject_multiline",
    "unsafe_text",
    "message_too_large",
    "invalid_identity",
    "malformed_staged_delta",
    "malformed_commit_object",
]
CommitReviewState = Literal["ready", "cancelled", "blocked"]
CommitReviewChangeType = Literal["New", "Modified", "Deleted", "Moved"]
CommitOutcomeState = Literal[
    "cancelled",
    "blocked",
    "succeeded",
    "failed_unchanged",
    "uncertain",
]

MAX_COMMIT_SUBJECT_CHARACTERS = 512
MAX_COMMIT_MESSAGE_BYTES = 64 * 1024

_ERROR_MESSAGES: dict[CommitContractErrorCode, str] = {
    "subject_required": "Commit subject is required.",
    "subject_too_long": "Commit subject exceeds 512 characters.",
    "subject_multiline": "Commit subject must be a single line.",
    "unsafe_text": "Text contains characters that cannot be previewed safely.",
    "message_too_large": "Commit message exceeds 64 KiB.",
    "invalid_identity": "Git identity is missing or invalid.",
    "malformed_staged_delta": "Staged Git data is malformed.",
    "malformed_commit_object": "Commit object data is malformed.",
}
_BIDI_FORMATTING_CODEPOINTS = frozenset(
    (*range(0x202A, 0x202F), *range(0x2066, 0x206A))
)
_OBJECT_ID_LENGTHS = frozenset({40, 64})
_MODE_PATTERN = re.compile(rb"[0-7]{6}")
_OBJECT_ID_PATTERN = re.compile(rb"[0-9a-fA-F]+")
_HEADER_NAME_PATTERN = re.compile(rb"[A-Za-z0-9-]+")
_TIMESTAMP_PATTERN = re.compile(rb"-?[0-9]+")
_OFFSET_PATTERN = re.compile(rb"[+-][0-9]{4}")


class CommitContractError(ValueError):
    """Bounded, path-free refusal from a pure commit contract.

    Attributes:
        code: Stable machine-readable refusal category.
    """

    def __init__(self, code: CommitContractErrorCode) -> None:
        """Initialize a contract refusal.

        Args:
            code: Stable refusal category.
        """
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class GitIdentity:
    """Confirmed Git identity without an ambient execution timestamp.

    Attributes:
        name: Effective Git author or committer name.
        email: Effective Git author or committer email.
    """

    name: str
    email: str

    def __post_init__(self) -> None:
        if (
            not self.name.strip()
            or not self.email.strip()
            or "<" in self.name
            or ">" in self.name
            or "<" in self.email
            or ">" in self.email
            or _contains_unsafe_text(self.name)
            or _contains_unsafe_text(self.email)
        ):
            raise CommitContractError("invalid_identity")

    @property
    def display(self) -> str:
        """Return the literal value intended for a ``markup=False`` widget."""
        return f"{self.name} <{self.email}>"


@dataclass(frozen=True, slots=True)
class CommitIncludedNote:
    """Sanitized display facts for one included session note.

    Attributes:
        group_id: Process-local session group identity.
        display_text: Control-safe note label for literal rendering.
        change_type: Git-semantic change proven by the complete staged delta.
    """

    group_id: int
    display_text: str
    change_type: CommitReviewChangeType


@dataclass(frozen=True, slots=True)
class CommitReviewProjection:
    """Presentation-only projection of an immutable commit review.

    Attributes:
        branch: Exact attached branch ref.
        old_commit: Reviewed parent commit object ID.
        message: Exact normalized UTF-8 message as text.
        included_notes: Sanitized included-note projections.
        author: Reviewed author name and email.
        committer: Reviewed committer name and email.
        hooks_bypassed: Whether repository hooks are bypassed.
        unsigned: Whether the reviewed commit is unsigned.
    """

    branch: str
    old_commit: str
    message: str
    included_notes: tuple[CommitIncludedNote, ...]
    author: GitIdentity
    committer: GitIdentity
    hooks_bypassed: bool = True
    unsigned: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "included_notes", tuple(self.included_notes))

    @property
    def included_note_count(self) -> int:
        """Return the reviewed number of included session notes."""
        return len(self.included_notes)

    @property
    def identity_display(self) -> tuple[tuple[str, str], ...]:
        """Return collapsed literal-rendering identity rows."""
        return format_git_identity_display(self.author, self.committer)


@dataclass(frozen=True, slots=True)
class CommitReviewHandle:
    """Opaque, process-memory capability for one guarded confirmation."""

    _token: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class CommitReviewResult:
    """Typed settlement of one commit-review request.

    Attributes:
        state: Review settlement category.
        handle: Opaque confirmation capability when ready.
        projection: Sanitized review when ready.
        message: Bounded recovery guidance when not ready.
    """

    state: CommitReviewState
    handle: CommitReviewHandle | None = None
    projection: CommitReviewProjection | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class CommitOutcome:
    """Typed terminal or recoverable result of one confirmation.

    Attributes:
        state: Guarded commit outcome category.
        message: Literal-rendering result or recovery guidance.
        qualification: Adjacent bounded policy qualification when applicable.
        commit_object_id: Proven new commit object ID on success.
        committed_note_count: Proven included-note count on success.
    """

    state: CommitOutcomeState
    message: str
    qualification: str | None = None
    commit_object_id: str | None = None
    committed_note_count: int = 0

    @property
    def recovery_required(self) -> bool:
        """Return whether exact retained proof must be checked again."""
        return self.state == "uncertain"


@dataclass(frozen=True, slots=True)
class CommitRecoveryProjection:
    """Sanitized projection of one retained uncertain commit attempt.

    Attributes:
        message: Literal-rendering recovery guidance.
        can_check_again: Whether full repository proof is already known safe.
            When false, a proof-only check may only re-observe the exact
            retained child and leave the attempt uncertain.
    """

    message: str
    can_check_again: bool


@dataclass(frozen=True, slots=True)
class RawStagedDeltaEntry:
    """One byte-preserving ``diff-index --raw -z`` staged-delta entry.

    Attributes:
        old_mode: Six-digit old Git file mode.
        new_mode: Six-digit new Git file mode.
        old_object_id: Old blob or tree object ID.
        new_object_id: New blob or tree object ID.
        status: Single-letter raw delta status.
        path: Uninterpreted repository-relative filename bytes.
    """

    old_mode: str
    new_mode: str
    old_object_id: str
    new_object_id: str
    status: str
    path: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class RawCommitObject:
    """Exact proof fields parsed from one raw normal commit object.

    Attributes:
        tree_object_id: Complete commit tree object ID.
        parent_object_id: Sole parent object ID.
        author: Parsed author name and email.
        committer: Parsed committer name and email.
        message: Exact bytes after the header separator.
        signature_headers: Detected commit-signature header names.
    """

    tree_object_id: str
    parent_object_id: str
    author: GitIdentity
    committer: GitIdentity
    message: bytes
    signature_headers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signature_headers",
            tuple(self.signature_headers),
        )

    @property
    def has_signature(self) -> bool:
        """Return whether a commit-signature header was present."""
        return bool(self.signature_headers)


def normalize_commit_message(subject: str, body: str = "") -> bytes:
    """Normalize and validate one exact UTF-8 Git commit message.

    Args:
        subject: Required single-line commit subject.
        body: Optional multiline commit body.

    Returns:
        Exact bytes shaped as ``subject\\n`` or
        ``subject\\n\\nbody\\n``.

    Raises:
        CommitContractError: If the message cannot be previewed exactly or
            violates a length bound.
    """
    normalized_subject = _normalize_newlines(subject)
    normalized_body = _normalize_newlines(body)
    if _contains_unsafe_text(
        normalized_subject,
        allow_newline=True,
        allow_tab=True,
    ) or _contains_unsafe_text(
        normalized_body,
        allow_newline=True,
        allow_tab=True,
    ):
        raise CommitContractError("unsafe_text")
    if "\n" in normalized_subject:
        raise CommitContractError("subject_multiline")

    trimmed_subject = normalized_subject.strip()
    if not trimmed_subject:
        raise CommitContractError("subject_required")
    if len(trimmed_subject) > MAX_COMMIT_SUBJECT_CHARACTERS:
        raise CommitContractError("subject_too_long")

    body_lines = normalized_body.split("\n")
    first_nonblank = 0
    while first_nonblank < len(body_lines) and not body_lines[first_nonblank].strip():
        first_nonblank += 1
    last_nonblank = len(body_lines)
    while last_nonblank > first_nonblank and not body_lines[last_nonblank - 1].strip():
        last_nonblank -= 1
    trimmed_body = "\n".join(body_lines[first_nonblank:last_nonblank])

    normalized = (
        f"{trimmed_subject}\n"
        if not trimmed_body
        else f"{trimmed_subject}\n\n{trimmed_body}\n"
    )
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError:
        raise CommitContractError("unsafe_text") from None
    if len(encoded) > MAX_COMMIT_MESSAGE_BYTES:
        raise CommitContractError("message_too_large")
    return encoded


def parse_git_identity(payload: bytes) -> GitIdentity:
    """Parse Git's effective ``name <email> timestamp offset`` ident.

    The timestamp and offset are validated but intentionally omitted from the
    result because confirmation binds names and emails while Git selects fresh
    execution timestamps.

    Args:
        payload: Raw stdout from ``git var GIT_*_IDENT`` or a raw commit
            identity header value.

    Returns:
        Confirmed name and email.

    Raises:
        CommitContractError: If any field is missing, malformed, non-UTF-8, or
            unsafe for exact literal display.
    """
    if not isinstance(payload, bytes):
        raise CommitContractError("invalid_identity")
    value = payload[:-1] if payload.endswith(b"\n") else payload
    parts = value.rsplit(b" ", 2)
    if len(parts) != 3:
        raise CommitContractError("invalid_identity")
    identity_bytes, timestamp, offset = parts
    if not _TIMESTAMP_PATTERN.fullmatch(timestamp) or not _valid_offset(offset):
        raise CommitContractError("invalid_identity")
    if not identity_bytes.endswith(b">"):
        raise CommitContractError("invalid_identity")
    delimiter = identity_bytes.rfind(b" <")
    if delimiter <= 0:
        raise CommitContractError("invalid_identity")

    name_bytes = identity_bytes[:delimiter]
    email_bytes = identity_bytes[delimiter + 2 : -1]
    try:
        name = name_bytes.decode("utf-8")
        email = email_bytes.decode("utf-8")
        return GitIdentity(name=name, email=email)
    except (UnicodeDecodeError, CommitContractError):
        raise CommitContractError("invalid_identity") from None


def format_git_identity_display(
    author: GitIdentity,
    committer: GitIdentity,
) -> tuple[tuple[str, str], ...]:
    """Build literal-rendering review rows for author and committer.

    Args:
        author: Reviewed author identity.
        committer: Reviewed committer identity.

    Returns:
        One ``Identity`` row when equal, otherwise separate ``Author`` and
        ``Committer`` rows.
    """
    if author == committer:
        return (("Identity", author.display),)
    return (
        ("Author", author.display),
        ("Committer", committer.display),
    )


def parse_raw_staged_delta(payload: bytes) -> tuple[RawStagedDeltaEntry, ...]:
    """Parse NUL-delimited output from ``diff-index --raw -z``.

    Args:
        payload: Complete raw command stdout.

    Returns:
        Frozen records whose paths remain uninterpreted bytes for an immediate
        proof comparison.

    Raises:
        CommitContractError: If any record is malformed or truncated. The
            diagnostic never contains filename bytes.
    """
    if not isinstance(payload, bytes):
        raise CommitContractError("malformed_staged_delta")
    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise CommitContractError("malformed_staged_delta")

    fields = payload[:-1].split(b"\0")
    if len(fields) % 2:
        raise CommitContractError("malformed_staged_delta")

    entries: list[RawStagedDeltaEntry] = []
    for index in range(0, len(fields), 2):
        header = fields[index]
        path = fields[index + 1]
        if not header.startswith(b":") or not path:
            raise CommitContractError("malformed_staged_delta")
        parts = header[1:].split(b" ")
        if len(parts) != 5 or any(not part for part in parts):
            raise CommitContractError("malformed_staged_delta")
        old_mode, new_mode, old_oid, new_oid, status = parts
        if (
            not _MODE_PATTERN.fullmatch(old_mode)
            or not _MODE_PATTERN.fullmatch(new_mode)
            or not _is_object_id(old_oid)
            or not _is_object_id(new_oid)
            or len(old_oid) != len(new_oid)
            or len(status) != 1
            or not 65 <= status[0] <= 90
        ):
            raise CommitContractError("malformed_staged_delta")
        entries.append(
            RawStagedDeltaEntry(
                old_mode=old_mode.decode("ascii"),
                new_mode=new_mode.decode("ascii"),
                old_object_id=old_oid.decode("ascii").lower(),
                new_object_id=new_oid.decode("ascii").lower(),
                status=status.decode("ascii"),
                path=path,
            )
        )
    return tuple(entries)


def parse_raw_commit_object(payload: bytes) -> RawCommitObject:
    """Parse exact proof fields from one raw Git commit object.

    Args:
        payload: Complete bytes from replacement-free ``cat-file commit``.

    Returns:
        The sole parent, tree, identities, exact message bytes, and detected
        commit-signature header names.

    Raises:
        CommitContractError: If the object is structurally malformed, lacks
            exactly one required header, or has invalid identity data.
    """
    if not isinstance(payload, bytes):
        raise CommitContractError("malformed_commit_object")
    separator = payload.find(b"\n\n")
    if separator < 0:
        raise CommitContractError("malformed_commit_object")
    header_payload = payload[:separator]
    message = payload[separator + 2 :]
    if not header_payload or b"\0" in header_payload or b"\r" in header_payload:
        raise CommitContractError("malformed_commit_object")

    headers: list[tuple[bytes, list[bytes]]] = []
    for line in header_payload.split(b"\n"):
        if line.startswith(b" "):
            if not headers:
                raise CommitContractError("malformed_commit_object")
            headers[-1][1].append(line[1:])
            continue
        delimiter = line.find(b" ")
        if delimiter <= 0:
            raise CommitContractError("malformed_commit_object")
        name = line[:delimiter]
        if not _HEADER_NAME_PATTERN.fullmatch(name):
            raise CommitContractError("malformed_commit_object")
        headers.append((name, [line[delimiter + 1 :]]))

    tree_values = _single_header_value(headers, b"tree")
    parent_values = _single_header_value(headers, b"parent")
    author_values = _single_header_value(headers, b"author")
    committer_values = _single_header_value(headers, b"committer")
    if (
        tree_values is None
        or parent_values is None
        or author_values is None
        or committer_values is None
        or not _is_commit_object_id(tree_values)
        or not _is_commit_object_id(parent_values)
        or len(tree_values) != len(parent_values)
    ):
        raise CommitContractError("malformed_commit_object")

    try:
        author = parse_git_identity(author_values)
        committer = parse_git_identity(committer_values)
    except CommitContractError:
        raise CommitContractError("malformed_commit_object") from None

    signature_headers = tuple(
        name.decode("ascii")
        for name, _ in headers
        if name == b"gpgsig" or name.startswith(b"gpgsig-")
    )
    return RawCommitObject(
        tree_object_id=tree_values.decode("ascii").lower(),
        parent_object_id=parent_values.decode("ascii").lower(),
        author=author,
        committer=committer,
        message=message,
        signature_headers=signature_headers,
    )


def _normalize_newlines(value: str) -> str:
    if not isinstance(value, str):
        raise CommitContractError("unsafe_text")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _contains_unsafe_text(
    value: str,
    *,
    allow_newline: bool = False,
    allow_tab: bool = False,
) -> bool:
    for character in value:
        codepoint = ord(character)
        if 0xD800 <= codepoint <= 0xDFFF:
            return True
        if codepoint in _BIDI_FORMATTING_CODEPOINTS or codepoint in {
            0x2028,
            0x2029,
        }:
            return True
        if codepoint < 0x20:
            if character == "\n" and allow_newline:
                continue
            if character == "\t" and allow_tab:
                continue
            return True
        if 0x7F <= codepoint <= 0x9F:
            return True
    return False


def _valid_offset(value: bytes) -> bool:
    if not _OFFSET_PATTERN.fullmatch(value):
        return False
    hours = int(value[1:3])
    minutes = int(value[3:5])
    return hours <= 23 and minutes <= 59


def _is_object_id(value: bytes) -> bool:
    return (
        len(value) in _OBJECT_ID_LENGTHS
        and _OBJECT_ID_PATTERN.fullmatch(value) is not None
    )


def _is_commit_object_id(value: bytes) -> bool:
    return _is_object_id(value) and any(character != 48 for character in value)


def _single_header_value(
    headers: list[tuple[bytes, list[bytes]]],
    name: bytes,
) -> bytes | None:
    matches = [values for header_name, values in headers if header_name == name]
    if len(matches) != 1 or len(matches[0]) != 1 or not matches[0][0]:
        return None
    return matches[0][0]
