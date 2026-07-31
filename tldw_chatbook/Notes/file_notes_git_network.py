"""Immutable owner-only Git execution context for guarded File Notes push.

This module performs no Git, SSH, credential-helper, or network execution. It
constructs and retains the exact local capability future guarded-push commands
must present. POSIX owner/mode and descriptor semantics are required; Windows
fails closed until an equivalent owner-only ACL implementation exists.

The private temporary root excludes other principals. Processes running as the
same effective UID, and root, remain inside the trusted application boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import sys
import tempfile
import threading
import weakref
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from tldw_chatbook.Notes.file_notes_git_push import (
    PushDestinationProjection,
    _FrozenPushEndpoint,
    _GitConfigFact,
    _build_push_argv,
    _build_push_query_argv,
    _read_frozen_endpoint,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    FileSystemIdentity,
    RepositoryIdentity,
)

NetworkContextErrorCode = Literal[
    "unsupported_platform",
    "invalid_context",
    "invalid_environment",
    "invalid_configuration",
    "invalid_source_objects",
    "unsafe_filesystem",
    "invalid_executable",
    "invalid_openssh",
]
NetworkRetentionPurpose = Literal["review", "active", "recovery"]
GitObjectFormat = Literal["sha1", "sha256"]

_ERROR_MESSAGES: dict[NetworkContextErrorCode, str] = {
    "unsupported_platform": (
        "Private network Git contexts are unsupported on this platform."
    ),
    "invalid_context": "The private network Git context is not available.",
    "invalid_environment": "The network Git environment is not allowed.",
    "invalid_configuration": (
        "The authorized network Git configuration is not allowed."
    ),
    "invalid_source_objects": (
        "The authorized source object directory is not available."
    ),
    "unsafe_filesystem": "The private network Git filesystem is not safe.",
    "invalid_executable": "The authorized executable is not available.",
    "invalid_openssh": "The authorized OpenSSH invocation is not available.",
}
_HEX_256 = re.compile(r"[0-9a-f]{64}")
_CREDENTIAL_HELPER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_GIT_BOOLEAN = frozenset(
    {"true", "false", "yes", "no", "on", "off", "1", "0"}
)
_SSH_SECRET = object()
_DIRECTORY_MODE = 0o700
_READ_ONLY_DIRECTORY_MODE = 0o500
_FILE_MODE = 0o600
_CONTEXT_PREFIX = ".chatbook-network-git-"
_ADAPTER_MODE = 0o700
_PLATFORM_CREDENTIAL_HELPERS: dict[str, frozenset[str]] = {
    "darwin": frozenset({"osxkeychain"}),
}
_SSH_ROUTE_HOST = "CHATBOOK_NETWORK_SSH_HOST"
_SSH_ROUTE_USER = "CHATBOOK_NETWORK_SSH_USER"
_SSH_ROUTE_PORT = "CHATBOOK_NETWORK_SSH_PORT"
_SSH_ROUTE_PATH = "CHATBOOK_NETWORK_SSH_PATH"
_SSH_ROUTE_AGENT = "CHATBOOK_NETWORK_SSH_AGENT"
_NO_AGENT = "none"
_OPENSSH_FIXED_ARGUMENTS = (
    "-F",
    "none",
    "-T",
    "-o",
    "SendEnv=GIT_PROTOCOL",
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=yes",
    "-o",
    "CheckHostIP=yes",
    "-o",
    "PreferredAuthentications=publickey",
    "-o",
    "PasswordAuthentication=no",
    "-o",
    "KbdInteractiveAuthentication=no",
    "-o",
    "ChallengeResponseAuthentication=no",
    "-o",
    "NumberOfPasswordPrompts=0",
    "-o",
    "ForwardAgent=no",
    "-o",
    "ForwardX11=no",
    "-o",
    "ClearAllForwardings=yes",
    "-o",
    "ProxyCommand=none",
    "-o",
    "ProxyJump=none",
    "-o",
    "CanonicalizeHostname=no",
    "-o",
    "ControlMaster=no",
    "-o",
    "ControlPath=none",
    "-o",
    "ControlPersist=no",
    "-o",
    "UpdateHostKeys=no",
    "-o",
    "KnownHostsCommand=none",
    "-o",
    "PermitLocalCommand=no",
    "-o",
    "RequestTTY=no",
)


class NetworkContextError(ValueError):
    """Sanitized fail-closed network-context refusal."""

    def __init__(self, code: NetworkContextErrorCode) -> None:
        """Initialize one bounded refusal.

        Args:
            code: Stable machine-readable refusal category.
        """
        self.code = code
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class _AuthorizedConfigFact:
    scope: Literal["system", "global"]
    origin_identity: str = field(repr=False)
    key: str
    value: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class _NetworkConfigRecord:
    configuration_fingerprint: str
    copy_fingerprint: str
    destination: PushDestinationProjection
    facts: tuple[_AuthorizedConfigFact, ...] = field(repr=False)


class NetworkConfigAuthorization:
    """Opaque exact authorization for copied noninteractive config facts."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NetworkConfigAuthorization:
        raise TypeError(
            "Network config authorizations require validated facts"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot change network config authorization")

    def __repr__(self) -> str:
        return "NetworkConfigAuthorization(<opaque>)"

    @property
    def configuration_fingerprint(self) -> str:
        """Return the already-resolved source configuration fingerprint."""
        return _read_network_config_authorization(
            self
        ).configuration_fingerprint

    @property
    def copy_fingerprint(self) -> str:
        """Return the ordered key/value/origin copy fingerprint."""
        return _read_network_config_authorization(self).copy_fingerprint

    @property
    def fact_count(self) -> int:
        """Return the number of exact facts authorized for copying."""
        return len(_read_network_config_authorization(self).facts)


@dataclass(frozen=True, slots=True)
class _SourceObjectRecord:
    path: Path = field(repr=False)
    identity: FileSystemIdentity
    object_format: GitObjectFormat
    owner: int
    group: int
    mode: int
    device: int
    identity_fingerprint: str


class SourceObjectDirectoryAuthorization:
    """Opaque identity-pinned read-only-alternate authorization."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> SourceObjectDirectoryAuthorization:
        raise TypeError(
            "Source object authorizations require a proved directory"
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot change source object authorization")

    def __repr__(self) -> str:
        return "SourceObjectDirectoryAuthorization(<opaque>)"

    @property
    def identity_fingerprint(self) -> str:
        """Return the path-and-filesystem-identity binding fingerprint."""
        return _read_source_object_authorization(self).identity_fingerprint


def _validated_network_config_record(
    facts: tuple[_GitConfigFact, ...],
    *,
    configuration_fingerprint: str,
    destination: PushDestinationProjection,
) -> _NetworkConfigRecord:
    """Authorize only an exact ordered set of safe copied Git config facts.

    The caller must pass only facts selected from an already double-read and
    policy-validated snapshot. This strict issuer never filters a mixed input;
    an unapproved fact rejects the entire authorization.

    Args:
        facts: Exact ordered facts proposed for the private config.
        configuration_fingerprint: Fingerprint of the complete resolved source
            configuration snapshot.
        destination: Sanitized exact authorized destination.

    Returns:
        Opaque authorization consumed by :class:`NetworkContextFactory`.

    Raises:
        NetworkContextError: If a fact, binding, or value is not allowlisted.
    """
    if (
        type(facts) is not tuple
        or type(destination) is not PushDestinationProjection
        or not _is_hex_fingerprint(configuration_fingerprint)
    ):
        raise NetworkContextError("invalid_configuration")
    authorized: list[_AuthorizedConfigFact] = []
    use_http_path_count = 0
    for fact in facts:
        if type(fact) is not _GitConfigFact or fact.scope not in {
            "system",
            "global",
        }:
            raise NetworkContextError("invalid_configuration")
        lowered = fact.key.lower()
        if lowered == "credential.helper":
            if fact.value and (
                _CREDENTIAL_HELPER.fullmatch(fact.value) is None
                or fact.value not in _supported_credential_helpers()
            ):
                raise NetworkContextError("invalid_configuration")
        elif lowered == "credential.usehttppath":
            use_http_path_count += 1
            if (
                use_http_path_count > 1
                or fact.value.lower() not in _GIT_BOOLEAN
            ):
                raise NetworkContextError("invalid_configuration")
        else:
            raise NetworkContextError("invalid_configuration")
        authorized.append(
            _AuthorizedConfigFact(
                fact.scope,
                fact.origin_identity,
                fact.key,
                fact.value,
            )
        )
    if authorized and destination.scheme != "https":
        raise NetworkContextError("invalid_configuration")
    copy_fingerprint = _config_copy_fingerprint(
        tuple(authorized),
        configuration_fingerprint,
        destination,
    )
    record = _NetworkConfigRecord(
        configuration_fingerprint,
        copy_fingerprint,
        destination,
        tuple(authorized),
    )
    return record


def _authorize_network_config_snapshot(
    facts: tuple[_GitConfigFact, ...],
    *,
    configuration_fingerprint: str,
    destination: PushDestinationProjection,
) -> NetworkConfigAuthorization:
    """Select supported credential facts from a proved complete snapshot.

    Unsupported credential-helper shapes are deliberately selected and then
    rejected by the strict issuer rather than silently dropped.
    """
    if type(facts) is not tuple or any(
        type(fact) is not _GitConfigFact for fact in facts
    ):
        raise NetworkContextError("invalid_configuration")
    selected = tuple(
        fact
        for fact in facts
        if (
            fact.key.lower().startswith("credential.")
            and fact.key.lower().endswith(
                ("helper", "usehttppath")
            )
        )
    )
    return _authorize_network_config_facts(
        selected,
        configuration_fingerprint=configuration_fingerprint,
        destination=destination,
    )


def _validated_source_object_record(
    path: str | os.PathLike[str],
    expected_identity: FileSystemIdentity,
    object_format: GitObjectFormat,
) -> _SourceObjectRecord:
    """Pin one already-proved canonical common object directory.

    Args:
        path: Canonical common Git object directory.
        expected_identity: Identity observed by the preceding local proof.
        object_format: Object format observed by the same local proof.

    Returns:
        Opaque identity-pinned authorization.

    Raises:
        NetworkContextError: If the directory is not the exact proved object.
    """
    _require_posix()
    if (
        type(expected_identity) is not FileSystemIdentity
        or type(object_format) is not str
        or object_format not in {"sha1", "sha256"}
    ):
        raise NetworkContextError("invalid_source_objects")
    try:
        candidate = Path(path)
        canonical = candidate.resolve(strict=True)
        metadata = candidate.stat(follow_symlinks=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise NetworkContextError("invalid_source_objects") from None
    if (
        not candidate.is_absolute()
        or candidate != canonical
        or os.pathsep in str(canonical)
        or "\0" in str(canonical)
        or "\n" in str(canonical)
        or "\r" in str(canonical)
        or not stat.S_ISDIR(metadata.st_mode)
        or _filesystem_identity(metadata) != expected_identity
        or metadata.st_uid not in {os.geteuid(), 0}
    ):
        raise NetworkContextError("invalid_source_objects")
    mode = stat.S_IMODE(metadata.st_mode)
    digest = hashlib.sha256()
    for value in (
        str(canonical),
        str(metadata.st_dev),
        str(metadata.st_ino),
        str(metadata.st_uid),
        str(metadata.st_gid),
        str(mode),
        object_format,
    ):
        _digest_text(digest, value)
    record = _SourceObjectRecord(
        canonical,
        expected_identity,
        object_format,
        metadata.st_uid,
        metadata.st_gid,
        mode,
        metadata.st_dev,
        digest.hexdigest(),
    )
    return record


def _make_authorization_registry():
    config_records: weakref.WeakKeyDictionary[
        NetworkConfigAuthorization,
        _NetworkConfigRecord,
    ] = weakref.WeakKeyDictionary()
    source_records: weakref.WeakKeyDictionary[
        SourceObjectDirectoryAuthorization,
        _SourceObjectRecord,
    ] = weakref.WeakKeyDictionary()

    def authorize_config(
        facts: tuple[_GitConfigFact, ...],
        *,
        configuration_fingerprint: str,
        destination: PushDestinationProjection,
    ) -> NetworkConfigAuthorization:
        record = _validated_network_config_record(
            facts,
            configuration_fingerprint=configuration_fingerprint,
            destination=destination,
        )
        authorization = object.__new__(NetworkConfigAuthorization)
        config_records[authorization] = record
        return authorization

    def read_config(
        authorization: NetworkConfigAuthorization,
    ) -> _NetworkConfigRecord:
        if type(authorization) is not NetworkConfigAuthorization:
            raise NetworkContextError("invalid_configuration")
        record = config_records.get(authorization)
        if record is None:
            raise NetworkContextError("invalid_configuration")
        return record

    def authorize_source(
        path: str | os.PathLike[str],
        expected_identity: FileSystemIdentity,
        object_format: GitObjectFormat,
    ) -> SourceObjectDirectoryAuthorization:
        record = _validated_source_object_record(
            path,
            expected_identity,
            object_format,
        )
        authorization = object.__new__(
            SourceObjectDirectoryAuthorization
        )
        source_records[authorization] = record
        return authorization

    def read_source(
        authorization: SourceObjectDirectoryAuthorization,
    ) -> _SourceObjectRecord:
        if type(authorization) is not SourceObjectDirectoryAuthorization:
            raise NetworkContextError("invalid_source_objects")
        record = source_records.get(authorization)
        if record is None:
            raise NetworkContextError("invalid_source_objects")
        return record

    return authorize_config, authorize_source, read_config, read_source


(
    _authorize_network_config_facts,
    _authorize_source_object_directory,
    _read_network_config_authorization,
    _read_source_object_authorization,
) = _make_authorization_registry()
del _make_authorization_registry


@dataclass(frozen=True, slots=True)
class NetworkCommandSettings:
    """Detached immutable direct-child settings for the process runner."""

    cwd: str = field(repr=False)
    environment: Mapping[str, str] = field(repr=False)
    environment_fingerprint: str
    stdin: None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        copied = dict(self.environment)
        if (
            not self.cwd
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in copied.items()
            )
        ):
            raise NetworkContextError("invalid_context")
        object.__setattr__(
            self,
            "environment",
            MappingProxyType(copied),
        )

    @property
    def stdin_closed(self) -> bool:
        """Return that future runner calls must attach the null device."""
        return True

    def __repr__(self) -> str:
        return (
            "NetworkCommandSettings(cwd=<private>, "
            "environment=<private>, stdin_closed=True)"
        )


@dataclass(frozen=True, slots=True)
class _PinnedAncestor:
    path: Path = field(repr=False)
    identity: FileSystemIdentity
    owner: int
    group: int
    mode: int
    device: int

    def validate(self) -> bool:
        try:
            metadata = self.path.stat(follow_symlinks=False)
        except OSError:
            return False
        return (
            stat.S_ISDIR(metadata.st_mode)
            and _filesystem_identity(metadata) == self.identity
            and metadata.st_uid == self.owner
            and metadata.st_gid == self.group
            and stat.S_IMODE(metadata.st_mode) == self.mode
            and metadata.st_dev == self.device
            and _safe_owned_directory_mode(metadata)
        )


@dataclass(frozen=True, slots=True)
class _PinnedExecutable:
    path: Path = field(repr=False)
    identity: FileSystemIdentity
    owner: int
    group: int
    mode: int
    device: int
    link_count: int
    ancestors: tuple[_PinnedAncestor, ...] = field(repr=False)

    def validate(self) -> bool:
        try:
            metadata = self.path.stat(follow_symlinks=False)
        except OSError:
            return False
        return (
            stat.S_ISREG(metadata.st_mode)
            and _filesystem_identity(metadata) == self.identity
            and metadata.st_uid == self.owner
            and metadata.st_gid == self.group
            and stat.S_IMODE(metadata.st_mode) == self.mode
            and metadata.st_dev == self.device
            and metadata.st_nlink == self.link_count
            and bool(metadata.st_mode & 0o111)
            and not metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            and all(ancestor.validate() for ancestor in self.ancestors)
        )


@dataclass(frozen=True, slots=True)
class _PinnedGitDispatchExecutable:
    entry: Path = field(repr=False)
    identity: FileSystemIdentity
    owner: int
    group: int
    mode: int
    device: int
    link_count: int
    target: _PinnedExecutable = field(repr=False)

    @property
    def path(self) -> Path:
        return self.target.path

    def validate(self) -> bool:
        try:
            metadata = self.entry.stat(follow_symlinks=False)
            resolved = self.entry.resolve(strict=True)
        except (OSError, RuntimeError):
            return False
        return (
            (stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode))
            and _filesystem_identity(metadata) == self.identity
            and metadata.st_uid == self.owner
            and metadata.st_gid == self.group
            and stat.S_IMODE(metadata.st_mode) == self.mode
            and metadata.st_dev == self.device
            and metadata.st_nlink == self.link_count
            and resolved == self.target.path
            and self.target.validate()
        )


@dataclass(frozen=True, slots=True)
class _PinnedSocket:
    path: Path = field(repr=False)
    identity: FileSystemIdentity
    owner: int
    group: int
    mode: int
    device: int
    ancestors: tuple[_PinnedAncestor, ...] = field(repr=False)

    def validate(self) -> bool:
        try:
            metadata = self.path.stat(follow_symlinks=False)
        except OSError:
            return False
        return (
            stat.S_ISSOCK(metadata.st_mode)
            and _filesystem_identity(metadata) == self.identity
            and metadata.st_uid == self.owner == os.geteuid()
            and metadata.st_gid == self.group
            and stat.S_IMODE(metadata.st_mode) == self.mode
            and metadata.st_dev == self.device
            and all(ancestor.validate() for ancestor in self.ancestors)
        )


@dataclass(frozen=True, slots=True)
class _PinnedDirectory:
    path: Path = field(repr=False)
    identity: FileSystemIdentity
    owner: int
    group: int
    mode: int
    device: int
    ancestors: tuple[_PinnedAncestor, ...] = field(repr=False)

    def validate(self) -> bool:
        try:
            metadata = self.path.stat(follow_symlinks=False)
        except OSError:
            return False
        return (
            stat.S_ISDIR(metadata.st_mode)
            and _filesystem_identity(metadata) == self.identity
            and metadata.st_uid == self.owner
            and metadata.st_gid == self.group
            and stat.S_IMODE(metadata.st_mode) == self.mode
            and metadata.st_dev == self.device
            and _safe_owned_directory_mode(metadata)
            and all(ancestor.validate() for ancestor in self.ancestors)
        )


@dataclass(frozen=True, slots=True)
class _CommandRecord:
    cwd: str = field(repr=False)
    git_dir: str = field(repr=False)
    environment: tuple[tuple[str, str], ...] = field(repr=False)
    environment_fingerprint: str

    def detached_settings(self) -> NetworkCommandSettings:
        return NetworkCommandSettings(
            cwd=self.cwd,
            environment=dict(self.environment),
            environment_fingerprint=self.environment_fingerprint,
        )


class OpenSSHInvocationSpec:
    """Immutable direct OpenSSH argv bound to a pinned executable identity."""

    __slots__ = ("_argv", "_executable")

    def __new__(
        cls,
        secret: object | None = None,
        *,
        argv: tuple[str, ...] = (),
        executable: _PinnedExecutable | None = None,
    ) -> OpenSSHInvocationSpec:
        if secret is not _SSH_SECRET or executable is None:
            raise TypeError("OpenSSH invocation specs are factory-issued")
        instance = super().__new__(cls)
        object.__setattr__(instance, "_argv", argv)
        object.__setattr__(instance, "_executable", executable)
        return instance

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot change OpenSSH invocation")

    def __repr__(self) -> str:
        return "OpenSSHInvocationSpec(<opaque>)"

    @property
    def argv(self) -> tuple[str, ...]:
        """Return the exact direct argv if the executable is still pinned."""
        if not self._executable.validate():
            raise NetworkContextError("invalid_openssh")
        return self._argv

    def _validate(self) -> bool:
        return self._executable.validate()


@dataclass(frozen=True, slots=True)
class _KnownEntry:
    relative_path: str
    kind: Literal["file", "directory", "executable"]
    identity: FileSystemIdentity
    owner: int
    group: int
    mode: int
    device: int
    link_count: int
    size: int
    digest: str | None = field(default=None, repr=False)


class _PrivateLayout:
    """Identity-pinned exact tree; same-UID mutation is inside trust boundary."""

    __slots__ = (
        "root",
        "parent",
        "_parent_entry",
        "_entries",
        "_removed",
    )

    def __init__(
        self,
        root: Path,
        parent: Path,
        parent_entry: _KnownEntry,
        entries: tuple[_KnownEntry, ...],
    ) -> None:
        self.root = root
        self.parent = parent
        self._parent_entry = parent_entry
        self._entries = entries
        self._removed: set[str] = set()

    @classmethod
    def capture(
        cls,
        root: Path,
        parent: Path,
        *,
        ssh_adapter: bool,
    ) -> _PrivateLayout:
        parent_entry = _capture_parent_entry(parent)
        relative_paths: tuple[
            tuple[
                str,
                Literal["file", "directory", "executable"],
                int,
            ],
            ...,
        ] = (
            (".", "directory", _DIRECTORY_MODE),
            ("repository.git", "directory", _DIRECTORY_MODE),
            ("repository.git/objects", "directory", _DIRECTORY_MODE),
            ("repository.git/objects/info", "directory", _DIRECTORY_MODE),
            ("repository.git/objects/pack", "directory", _DIRECTORY_MODE),
            ("repository.git/refs", "directory", _DIRECTORY_MODE),
            ("home", "directory", _READ_ONLY_DIRECTORY_MODE),
            ("xdg-config", "directory", _READ_ONLY_DIRECTORY_MODE),
            ("tmp", "directory", _READ_ONLY_DIRECTORY_MODE),
            ("repository.git/HEAD", "file", _FILE_MODE),
            ("repository.git/config", "file", _FILE_MODE),
            ("system.gitconfig", "file", _FILE_MODE),
            ("global.gitconfig", "file", _FILE_MODE),
        )
        if ssh_adapter:
            relative_paths = (
                *relative_paths,
                ("ssh-adapter", "executable", _ADAPTER_MODE),
            )
        entries = tuple(
            _capture_known_entry(
                root,
                root if relative == "." else root / relative,
                kind,
                relative,
                expected_mode=expected_mode,
            )
            for relative, kind, expected_mode in relative_paths
        )
        return cls(root, parent, parent_entry, entries)

    def validate(
        self,
        *,
        include_contents: bool = True,
        allow_partial_cleanup: bool = False,
    ) -> bool:
        try:
            if not _parent_entry_matches(self.parent, self._parent_entry):
                return False
            if self._removed and not allow_partial_cleanup:
                return False
            removed = self._removed if allow_partial_cleanup else set()
            expected_children: dict[str, set[str]] = {}
            for entry in self._entries:
                if entry.relative_path in removed:
                    continue
                if entry.relative_path == ".":
                    expected_children.setdefault(".", set())
                    continue
                parent, _, name = entry.relative_path.rpartition("/")
                expected_children.setdefault(parent or ".", set()).add(name)
                if entry.kind == "directory":
                    expected_children.setdefault(entry.relative_path, set())
            for entry in self._entries:
                path = (
                    self.root
                    if entry.relative_path == "."
                    else self.root / entry.relative_path
                )
                if entry.relative_path in removed:
                    if os.path.lexists(path):
                        return False
                    continue
                if not _known_entry_matches(
                    self.root,
                    path,
                    entry,
                    include_contents=include_contents,
                    allow_directory_link_drift=allow_partial_cleanup,
                ):
                    return False
                if entry.kind == "directory" and set(os.listdir(path)) != (
                    expected_children[entry.relative_path]
                ):
                    return False
        except (OSError, RuntimeError):
            return False
        return True

    def cleanup(self) -> bool:
        if not self.validate(
            include_contents=False,
            allow_partial_cleanup=True,
        ):
            return False
        files = sorted(
            (
                entry
                for entry in self._entries
                if entry.kind in {"file", "executable"}
            ),
            key=lambda entry: entry.relative_path.count("/"),
            reverse=True,
        )
        directories = sorted(
            (
                entry
                for entry in self._entries
                if entry.kind == "directory" and entry.relative_path != "."
            ),
            key=lambda entry: entry.relative_path.count("/"),
            reverse=True,
        )
        try:
            for entry in files:
                if entry.relative_path in self._removed:
                    continue
                path = self.root / entry.relative_path
                if not _known_entry_matches(
                    self.root,
                    path,
                    entry,
                    include_contents=False,
                ):
                    return False
                path.unlink()
                self._removed.add(entry.relative_path)
            for entry in directories:
                if entry.relative_path in self._removed:
                    continue
                path = self.root / entry.relative_path
                path.rmdir()
                self._removed.add(entry.relative_path)
            if not self.validate(
                include_contents=False,
                allow_partial_cleanup=True,
            ):
                return False
            self.root.rmdir()
        except OSError:
            return False
        return not self.root.exists()


class _LayoutBuilder:
    __slots__ = ("root", "_created")

    def __init__(self, root: Path) -> None:
        self.root = root
        self._created: list[tuple[Path, Literal["file", "directory"]]] = []

    def directory(self, relative_path: str) -> Path:
        path = self.root / relative_path
        path.mkdir(mode=_DIRECTORY_MODE)
        self._created.append((path, "directory"))
        path.chmod(_DIRECTORY_MODE)
        return path

    def file(
        self,
        relative_path: str,
        payload: bytes,
        *,
        executable: bool = False,
    ) -> Path:
        path = self.root / relative_path
        mode = _ADAPTER_MODE if executable else _FILE_MODE
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, mode)
        self._created.append((path, "file"))
        try:
            os.fchmod(descriptor, mode)
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short private config write")
                view = view[written:]
        finally:
            os.close(descriptor)
        return path

    def cleanup_partial(self) -> None:
        for path, kind in reversed(self._created):
            try:
                if kind == "directory":
                    path.rmdir()
                else:
                    path.unlink()
            except OSError:
                pass
        try:
            self.root.rmdir()
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class _ContextAuthority:
    command: _CommandRecord = field(repr=False)
    temporary_parent: _PinnedDirectory = field(repr=False)
    git_executable: _PinnedExecutable = field(repr=False)
    git_exec_directory: _PinnedDirectory = field(repr=False)
    python_executable: _PinnedExecutable | None = field(repr=False)
    dispatch_executables: tuple[_PinnedGitDispatchExecutable, ...] = field(
        repr=False
    )
    agent_socket: _PinnedSocket | None = field(repr=False)
    source_objects: _SourceObjectRecord = field(repr=False)
    configuration: _NetworkConfigRecord = field(repr=False)
    destination: PushDestinationProjection = field(repr=False)
    endpoint: str = field(repr=False)
    openssh: OpenSSHInvocationSpec | None = field(repr=False)


class NetworkContextLease:
    """Opaque single-release holder for review, active, or recovery work."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NetworkContextLease:
        raise TypeError("Network context leases are context-issued")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot change network context lease")

    def __repr__(self) -> str:
        return "NetworkContextLease(<opaque>)"

    def __copy__(self) -> NetworkContextLease:
        raise NetworkContextError("invalid_context")

    def __deepcopy__(self, _memo: object) -> NetworkContextLease:
        raise NetworkContextError("invalid_context")

    def release(self) -> bool:
        """Release this exact holder once."""
        return _release_network_context_lease(self)


class NetworkGitExecutionContext:
    """Opaque immutable capability for one private network Git directory."""

    __slots__ = ("__weakref__",)

    def __new__(cls) -> NetworkGitExecutionContext:
        raise TypeError("Network Git execution contexts are factory-issued")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot change network Git execution context")

    def __repr__(self) -> str:
        return "NetworkGitExecutionContext(<opaque>)"

    def __copy__(self) -> NetworkGitExecutionContext:
        raise NetworkContextError("invalid_context")

    def __deepcopy__(self, _memo: object) -> NetworkGitExecutionContext:
        raise NetworkContextError("invalid_context")

    @property
    def config_copy_fingerprint(self) -> str:
        """Return the ordered copied-fact fingerprint without private values."""
        return _network_context_config_copy_fingerprint(self)

    @property
    def cleaned(self) -> bool:
        """Return whether exact cleanup has completed."""
        return _network_context_is_cleaned(self)

    def command_settings(self) -> NetworkCommandSettings:
        """Return a detached view of live cwd/environment/stdin settings."""
        return _network_context_command_settings(self)

    def openssh_invocation(self) -> OpenSSHInvocationSpec | None:
        """Return the immutable direct SSH argv only for an SSH endpoint."""
        return _network_context_openssh_invocation(self)

    def build_query_argv(
        self,
        endpoint: _FrozenPushEndpoint,
    ) -> tuple[str, ...]:
        """Build a query argv only from this live context and frozen endpoint."""
        return _network_context_build_query_argv(self, endpoint)

    def build_push_argv(
        self,
        endpoint: _FrozenPushEndpoint,
        parent_oid: str,
        candidate_oid: str,
    ) -> tuple[str, ...]:
        """Build one exact CAS push argv only from this live context."""
        return _network_context_build_push_argv(
            self,
            endpoint,
            parent_oid,
            candidate_oid,
        )

    def retain(
        self,
        purpose: NetworkRetentionPurpose,
    ) -> NetworkContextLease:
        """Retain exact review/active/recovery ownership until release."""
        return _retain_network_context(self, purpose)

    def close(self) -> bool:
        """Request exact cleanup, deferred until every holder releases."""
        return _close_network_context(self)


def _make_network_context_registry():
    class _ContextLifecycle:
        __slots__ = (
            "authority",
            "layout",
            "lock",
            "leases",
            "close_requested",
            "cleaned",
        )

        def __init__(
            self,
            authority: _ContextAuthority,
            layout: _PrivateLayout,
        ) -> None:
            self.authority = authority
            self.layout = layout
            self.lock = threading.RLock()
            self.leases: dict[
                NetworkContextLease,
                NetworkRetentionPurpose,
            ] = {}
            self.close_requested = False
            self.cleaned = False

        def require_live(self) -> None:
            authority = self.authority
            with self.lock:
                if (
                    self.cleaned
                    or not self.layout.validate()
                    or not authority.temporary_parent.validate()
                    or not authority.git_executable.validate()
                    or not authority.git_exec_directory.validate()
                    or (
                        authority.python_executable is not None
                        and not authority.python_executable.validate()
                    )
                    or not all(
                        executable.validate()
                        for executable in authority.dispatch_executables
                    )
                    or (
                        authority.agent_socket is not None
                        and not authority.agent_socket.validate()
                    )
                    or not _source_record_matches(authority.source_objects)
                    or (
                        authority.openssh is not None
                        and not authority.openssh._validate()
                    )
                ):
                    raise NetworkContextError("invalid_context")

        def attempt_cleanup(self) -> bool:
            if self.layout.cleanup():
                self.cleaned = True
                return True
            return False

    class _LeaseRecord:
        __slots__ = ("context", "released")

        def __init__(self, context: NetworkGitExecutionContext) -> None:
            self.context = context
            self.released = False

    contexts: weakref.WeakKeyDictionary[
        NetworkGitExecutionContext,
        _ContextLifecycle,
    ] = weakref.WeakKeyDictionary()
    leases: weakref.WeakKeyDictionary[
        NetworkContextLease,
        _LeaseRecord,
    ] = weakref.WeakKeyDictionary()

    def read_context(
        context: NetworkGitExecutionContext,
    ) -> _ContextLifecycle:
        if type(context) is not NetworkGitExecutionContext:
            raise NetworkContextError("invalid_context")
        lifecycle = contexts.get(context)
        if lifecycle is None:
            raise NetworkContextError("invalid_context")
        return lifecycle

    def issue_context(
        authority: _ContextAuthority,
        layout: _PrivateLayout,
    ) -> NetworkGitExecutionContext:
        if (
            type(authority) is not _ContextAuthority
            or type(layout) is not _PrivateLayout
        ):
            raise NetworkContextError("invalid_context")
        context = object.__new__(NetworkGitExecutionContext)
        contexts[context] = _ContextLifecycle(authority, layout)
        return context

    def config_copy_fingerprint(
        context: NetworkGitExecutionContext,
    ) -> str:
        return read_context(context).authority.configuration.copy_fingerprint

    def is_cleaned(context: NetworkGitExecutionContext) -> bool:
        lifecycle = read_context(context)
        with lifecycle.lock:
            return lifecycle.cleaned

    def command_settings(
        context: NetworkGitExecutionContext,
    ) -> NetworkCommandSettings:
        lifecycle = read_context(context)
        with lifecycle.lock:
            lifecycle.require_live()
            return lifecycle.authority.command.detached_settings()

    def openssh_invocation(
        context: NetworkGitExecutionContext,
    ) -> OpenSSHInvocationSpec | None:
        lifecycle = read_context(context)
        with lifecycle.lock:
            lifecycle.require_live()
            openssh = lifecycle.authority.openssh
            if openssh is not None:
                openssh.argv
            return openssh

    def require_endpoint(
        context: NetworkGitExecutionContext,
        endpoint: _FrozenPushEndpoint,
    ) -> _ContextAuthority:
        lifecycle = read_context(context)
        with lifecycle.lock:
            lifecycle.require_live()
            try:
                endpoint_value, projection = _read_frozen_endpoint(endpoint)
            except ValueError:
                raise NetworkContextError("invalid_context") from None
            authority = lifecycle.authority
            if (
                endpoint_value != authority.endpoint
                or projection != authority.destination
            ):
                raise NetworkContextError("invalid_context")
            return authority

    def build_query_argv(
        context: NetworkGitExecutionContext,
        endpoint: _FrozenPushEndpoint,
    ) -> tuple[str, ...]:
        authority = require_endpoint(context, endpoint)
        return _build_push_query_argv(
            str(authority.git_executable.path),
            authority.command.git_dir,
            endpoint,
        )

    def build_push_argv(
        context: NetworkGitExecutionContext,
        endpoint: _FrozenPushEndpoint,
        parent_oid: str,
        candidate_oid: str,
    ) -> tuple[str, ...]:
        authority = require_endpoint(context, endpoint)
        expected_width = (
            40 if authority.source_objects.object_format == "sha1" else 64
        )
        if (
            len(parent_oid) in {40, 64}
            and len(candidate_oid) in {40, 64}
            and (
                len(parent_oid) != expected_width
                or len(candidate_oid) != expected_width
            )
        ):
            raise NetworkContextError("invalid_context")
        return _build_push_argv(
            str(authority.git_executable.path),
            authority.command.git_dir,
            endpoint,
            parent_oid,
            candidate_oid,
        )

    def retain_context(
        context: NetworkGitExecutionContext,
        purpose: NetworkRetentionPurpose,
    ) -> NetworkContextLease:
        if purpose not in {"review", "active", "recovery"}:
            raise NetworkContextError("invalid_context")
        lifecycle = read_context(context)
        with lifecycle.lock:
            if lifecycle.close_requested:
                raise NetworkContextError("invalid_context")
            lifecycle.require_live()
            lease = object.__new__(NetworkContextLease)
            leases[lease] = _LeaseRecord(context)
            lifecycle.leases[lease] = purpose
            return lease

    def release_lease(lease: NetworkContextLease) -> bool:
        if type(lease) is not NetworkContextLease:
            raise NetworkContextError("invalid_context")
        record = leases.get(lease)
        if record is None:
            raise NetworkContextError("invalid_context")
        lifecycle = read_context(record.context)
        with lifecycle.lock:
            if record.released:
                return False
            if lease not in lifecycle.leases:
                raise NetworkContextError("invalid_context")
            del lifecycle.leases[lease]
            record.released = True
            if lifecycle.close_requested and not lifecycle.leases:
                lifecycle.attempt_cleanup()
            return True

    def close_context(context: NetworkGitExecutionContext) -> bool:
        lifecycle = read_context(context)
        with lifecycle.lock:
            if lifecycle.cleaned:
                return True
            lifecycle.close_requested = True
            if lifecycle.leases:
                return False
            return lifecycle.attempt_cleanup()

    return (
        issue_context,
        config_copy_fingerprint,
        is_cleaned,
        command_settings,
        openssh_invocation,
        build_query_argv,
        build_push_argv,
        retain_context,
        release_lease,
        close_context,
    )


(
    _issue_network_context,
    _network_context_config_copy_fingerprint,
    _network_context_is_cleaned,
    _network_context_command_settings,
    _network_context_openssh_invocation,
    _network_context_build_query_argv,
    _network_context_build_push_argv,
    _retain_network_context,
    _release_network_context_lease,
    _close_network_context,
) = _make_network_context_registry()
del _make_network_context_registry


class NetworkContextFactory:
    """Immutable sole authority for fresh, never-reused network contexts."""

    __slots__ = (
        "_base_environment",
        "_temporary_parent",
        "_git_executable",
        "_git_exec_directory",
        "_python_executable_value",
        "_agent_socket_value",
        "_ssh_executable_value",
    )

    def __init__(
        self,
        *,
        environment: Mapping[str, str] | None = None,
        temporary_parent: str | os.PathLike[str] | None = None,
        git_executable: str | None = None,
        git_exec_path: str | os.PathLike[str],
        ssh_executable: str | None = None,
        allow_ssh_agent: bool = False,
    ) -> None:
        """Freeze the explicit ambient allowlist and executable selections.

        Args:
            environment: Ambient source inspected only through a fixed
                allowlist. Unrelated values are never retained.
            temporary_parent: Optional test seam for a safe canonical parent.
            git_executable: Optional exact Git executable selection.
            git_exec_path: Exact locally proved Git executable directory.
            ssh_executable: Optional exact OpenSSH executable selection.
            allow_ssh_agent: Whether to retain an explicit ``SSH_AUTH_SOCK``.
        """
        _require_posix()
        if type(allow_ssh_agent) is not bool:
            raise NetworkContextError("invalid_environment")
        source = os.environ if environment is None else environment
        base = _allowlisted_base_environment(source, allow_ssh_agent)
        path_value = dict(base).get("PATH", os.defpath)
        selected_git = git_executable or shutil.which("git", path=path_value)
        if selected_git is None:
            raise NetworkContextError("invalid_executable")
        object.__setattr__(self, "_base_environment", base)
        object.__setattr__(self, "_temporary_parent", temporary_parent)
        object.__setattr__(
            self,
            "_git_executable",
            _pin_executable(selected_git, path_value),
        )
        if self._git_executable.path.name != "git":
            raise NetworkContextError("invalid_executable")
        object.__setattr__(
            self,
            "_git_exec_directory",
            _pin_git_exec_directory(git_exec_path),
        )
        object.__setattr__(self, "_python_executable_value", sys.executable)
        agent_value = dict(base).get("SSH_AUTH_SOCK")
        object.__setattr__(
            self,
            "_agent_socket_value",
            agent_value,
        )
        object.__setattr__(self, "_ssh_executable_value", ssh_executable)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise FrozenInstanceError("cannot change network context factory")

    def __repr__(self) -> str:
        return "NetworkContextFactory(<opaque>)"

    def create(
        self,
        *,
        repository: RepositoryIdentity,
        source_objects: SourceObjectDirectoryAuthorization,
        configuration: NetworkConfigAuthorization,
        destination: PushDestinationProjection,
        endpoint: _FrozenPushEndpoint,
    ) -> NetworkGitExecutionContext:
        """Create one new private context without executing or discovering one.

        Args:
            repository: Exact trusted source repository identity.
            source_objects: Already-proved common object-directory capability.
            configuration: Exact authorized config-copy capability.
            destination: Exact sanitized authorized destination.
            endpoint: Exact frozen endpoint whose spelling Git will receive.

        Returns:
            Fresh opaque immutable context.

        Raises:
            NetworkContextError: If any filesystem or authority fact is stale.
        """
        _require_posix()
        if (
            type(repository) is not RepositoryIdentity
            or type(source_objects) is not SourceObjectDirectoryAuthorization
            or type(configuration) is not NetworkConfigAuthorization
            or type(destination) is not PushDestinationProjection
            or type(endpoint) is not _FrozenPushEndpoint
        ):
            raise NetworkContextError("invalid_context")
        try:
            endpoint_value, endpoint_projection = _read_frozen_endpoint(endpoint)
        except ValueError:
            raise NetworkContextError("invalid_context") from None
        source_record = _read_source_object_authorization(source_objects)
        config_record = _read_network_config_authorization(configuration)
        excluded_roots = (
            Path(repository.worktree_root),
            Path(repository.git_dir),
            Path(repository.git_common_dir),
        )
        if (
            config_record.destination != destination
            or endpoint_projection != destination
            or not _repository_matches(repository)
            or not _source_record_matches(source_record)
            or source_record.path
            != Path(repository.git_common_dir) / "objects"
            or not self._git_executable.validate()
            or not self._git_exec_directory.validate()
            or any(
                _path_is_within(self._git_exec_directory.path, root)
                for root in excluded_roots
            )
        ):
            raise NetworkContextError("invalid_context")
        parent_record = _safe_temporary_parent(self._temporary_parent)
        parent = parent_record.path
        ambient_search_path = dict(self._base_environment).get(
            "PATH",
            os.defpath,
        )
        runtime_path = _runtime_search_path(self._git_executable)
        credential_helpers = _pin_credential_helpers(
            config_record,
            self._git_exec_directory,
        )
        dispatch_executables = tuple(
            executable for _name, executable in credential_helpers
        )
        if destination.scheme == "https":
            transport_helper = _pin_git_dispatch_executable(
                "git-remote-https",
                self._git_exec_directory,
            )
            dispatch_executables = (
                transport_helper,
                *dispatch_executables,
            )
        agent_socket = (
            _pin_agent_socket(self._agent_socket_value)
            if (
                destination.scheme == "ssh"
                and self._agent_socket_value is not None
            )
            else None
        )
        python_executable = (
            _pin_executable(
                self._python_executable_value,
                ambient_search_path,
            )
            if destination.scheme == "ssh"
            else None
        )
        openssh, ssh_executable = self._build_openssh(
            destination,
            ambient_search_path,
            agent_socket,
        )
        network_executables = (
            self._git_executable,
            *dispatch_executables,
        )
        if python_executable is not None:
            network_executables = (*network_executables, python_executable)
        if ssh_executable is not None:
            network_executables = (*network_executables, ssh_executable)
        if any(
            _path_is_within(executable.path, root)
            for executable in network_executables
            for root in excluded_roots
        ):
            raise NetworkContextError("invalid_executable")
        root: Path | None = None
        builder: _LayoutBuilder | None = None
        try:
            root = Path(
                tempfile.mkdtemp(prefix=_CONTEXT_PREFIX, dir=str(parent))
            )
            root.chmod(_DIRECTORY_MODE)
            if (
                root.parent != parent
                or _path_is_within(root, Path(repository.worktree_root))
                or _path_is_within(root, Path(repository.git_dir))
                or _path_is_within(root, Path(repository.git_common_dir))
            ):
                raise NetworkContextError("unsafe_filesystem")
            builder = _LayoutBuilder(root)
            git_dir = builder.directory("repository.git")
            object_directory = builder.directory("repository.git/objects")
            builder.directory("repository.git/objects/info")
            builder.directory("repository.git/objects/pack")
            builder.directory("repository.git/refs")
            home = builder.directory("home")
            config_home = builder.directory("xdg-config")
            private_tmp = builder.directory("tmp")
            builder.file(
                "repository.git/HEAD",
                b"ref: refs/heads/chatbook-isolated\n",
            )
            builder.file(
                "repository.git/config",
                _render_private_config(
                    config_record.facts,
                    source_record.object_format,
                ),
            )
            system_config = builder.file("system.gitconfig", b"")
            global_config = builder.file("global.gitconfig", b"")
            ssh_adapter = root / "ssh-adapter" if openssh is not None else None
            ssh_routing = (
                ()
                if openssh is None
                else _ssh_runtime_routing(
                    endpoint_value,
                    destination,
                    agent_socket,
                )
            )
            environment = _build_context_environment(
                self._base_environment,
                runtime_path=runtime_path,
                git_exec_path=self._git_exec_directory.path,
                git_dir=git_dir,
                object_directory=object_directory,
                source_objects=source_record.path,
                system_config=system_config,
                global_config=global_config,
                home=home,
                config_home=config_home,
                private_tmp=private_tmp,
                ssh_adapter=ssh_adapter,
                ssh_routing=ssh_routing,
            )
            environment_fingerprint = _environment_fingerprint(environment)
            ssh_routing_fingerprint = _environment_fingerprint(ssh_routing)
            if openssh is not None:
                if python_executable is None:
                    raise NetworkContextError("invalid_executable")
                created_ssh_adapter = builder.file(
                    "ssh-adapter",
                    _render_ssh_adapter(
                        python=python_executable,
                        openssh=openssh,
                        routing=ssh_routing,
                        routing_fingerprint=ssh_routing_fingerprint,
                    ),
                    executable=True,
                )
                if created_ssh_adapter != ssh_adapter:
                    raise NetworkContextError("unsafe_filesystem")
            for child_directory in (home, config_home, private_tmp):
                child_directory.chmod(_READ_ONLY_DIRECTORY_MODE)
            layout = _PrivateLayout.capture(
                root,
                parent,
                ssh_adapter=ssh_adapter is not None,
            )
            if not layout.validate():
                raise NetworkContextError("unsafe_filesystem")
            command = _CommandRecord(
                cwd=str(git_dir),
                git_dir=str(git_dir),
                environment=tuple(environment),
                environment_fingerprint=environment_fingerprint,
            )
            authority = _ContextAuthority(
                command=command,
                temporary_parent=parent_record,
                git_executable=self._git_executable,
                git_exec_directory=self._git_exec_directory,
                python_executable=python_executable,
                dispatch_executables=dispatch_executables,
                agent_socket=agent_socket,
                source_objects=source_record,
                configuration=config_record,
                destination=destination,
                endpoint=endpoint_value,
                openssh=openssh,
            )
            return _issue_network_context(authority, layout)
        except NetworkContextError:
            if builder is not None:
                builder.cleanup_partial()
            elif root is not None:
                try:
                    root.rmdir()
                except OSError:
                    pass
            raise
        except (OSError, RuntimeError, ValueError):
            if builder is not None:
                builder.cleanup_partial()
            elif root is not None:
                try:
                    root.rmdir()
                except OSError:
                    pass
            raise NetworkContextError("unsafe_filesystem") from None

    def _build_openssh(
        self,
        destination: PushDestinationProjection,
        search_path: str,
        agent_socket: _PinnedSocket | None,
    ) -> tuple[OpenSSHInvocationSpec | None, _PinnedExecutable | None]:
        if destination.scheme != "ssh":
            return None, None
        selected = self._ssh_executable_value or shutil.which(
            "ssh",
            path=search_path,
        )
        if selected is None or destination.ssh_user is None:
            raise NetworkContextError("invalid_openssh")
        executable = _pin_executable(
            selected,
            search_path,
        )
        host = destination.host
        argv = (
            str(executable.path),
            *_OPENSSH_FIXED_ARGUMENTS,
            "-o",
            (
                "IdentityAgent=none"
                if agent_socket is None
                else f"IdentityAgent={agent_socket.path}"
            ),
            "-o",
            f"HostName={host}",
            "-p",
            str(destination.port),
            "-l",
            destination.ssh_user,
            "--",
            host,
        )
        return (
            OpenSSHInvocationSpec(
                _SSH_SECRET,
                argv=argv,
                executable=executable,
            ),
            executable,
        )


def _allowlisted_base_environment(
    ambient: Mapping[str, str],
    allow_ssh_agent: bool,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(ambient, Mapping):
        raise NetworkContextError("invalid_environment")
    allowed_names = ("PATH",)
    selected: list[tuple[str, str]] = []
    for name in allowed_names:
        if name not in ambient:
            continue
        value = ambient[name]
        if not _safe_environment_value(name, value):
            raise NetworkContextError("invalid_environment")
        selected.append((name, value))
    if not any(name == "PATH" for name, _value in selected):
        selected.append(("PATH", os.defpath))
    if allow_ssh_agent and "SSH_AUTH_SOCK" in ambient:
        value = ambient["SSH_AUTH_SOCK"]
        if not (
            isinstance(value, str)
            and bool(value)
            and "\0" not in value
            and "\n" not in value
            and "\r" not in value
        ):
            raise NetworkContextError("invalid_environment")
        selected.append(("SSH_AUTH_SOCK", value))
    return tuple(selected)


def _build_context_environment(
    base: tuple[tuple[str, str], ...],
    *,
    runtime_path: str,
    git_exec_path: Path,
    git_dir: Path,
    object_directory: Path,
    source_objects: Path,
    system_config: Path,
    global_config: Path,
    home: Path,
    config_home: Path,
    private_tmp: Path,
    ssh_adapter: Path | None,
    ssh_routing: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    values = dict(base)
    if ssh_adapter is None:
        if ssh_routing:
            raise NetworkContextError("invalid_environment")
        values.pop("SSH_AUTH_SOCK", None)
    values.update(
        {
            "GCM_GUI_PROMPT": "0",
            "GCM_INTERACTIVE": "Never",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(source_objects),
            "GIT_CONFIG_GLOBAL": str(global_config),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": str(system_config),
            "GIT_DIR": str(git_dir),
            "GIT_EXEC_PATH": str(git_exec_path),
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OBJECT_DIRECTORY": str(object_directory),
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(home),
            "LC_ALL": "C",
            "PATH": runtime_path,
            "SSH_ASKPASS_REQUIRE": "never",
            "TEMP": str(private_tmp),
            "TMP": str(private_tmp),
            "TMPDIR": str(private_tmp),
            "XDG_CONFIG_HOME": str(config_home),
        }
    )
    if ssh_adapter is not None:
        routing = dict(ssh_routing)
        if set(routing) != {
            _SSH_ROUTE_HOST,
            _SSH_ROUTE_USER,
            _SSH_ROUTE_PORT,
            _SSH_ROUTE_PATH,
            _SSH_ROUTE_AGENT,
        }:
            raise NetworkContextError("invalid_environment")
        values.update(
            {
                "GIT_SSH": str(ssh_adapter),
                "GIT_SSH_VARIANT": "ssh",
                **routing,
            }
        )
    return tuple(sorted(values.items()))


def _ssh_runtime_routing(
    endpoint_value: str,
    destination: PushDestinationProjection,
    agent_socket: _PinnedSocket | None,
) -> tuple[tuple[str, str], ...]:
    if destination.scheme != "ssh" or destination.ssh_user is None:
        raise NetworkContextError("invalid_openssh")
    repository_path = (
        destination.repository_path
        if endpoint_value.startswith("ssh://")
        else endpoint_value.rsplit(":", 1)[-1]
    )
    return tuple(
        sorted(
            {
                _SSH_ROUTE_HOST: destination.host,
                _SSH_ROUTE_USER: destination.ssh_user,
                _SSH_ROUTE_PORT: str(destination.port),
                _SSH_ROUTE_PATH: repository_path,
                _SSH_ROUTE_AGENT: (
                    _NO_AGENT
                    if agent_socket is None
                    else str(agent_socket.path)
                ),
            }.items()
        )
    )


def _environment_fingerprint(
    environment: Mapping[str, str] | tuple[tuple[str, str], ...],
) -> str:
    digest = hashlib.sha256()
    _digest_text(digest, "file-notes-network-environment-v1")
    pairs = (
        environment.items()
        if isinstance(environment, Mapping)
        else environment
    )
    for key, value in pairs:
        _digest_text(digest, key)
        _digest_text(digest, value)
    return digest.hexdigest()


def _config_copy_fingerprint(
    facts: tuple[_AuthorizedConfigFact, ...],
    configuration_fingerprint: str,
    destination: PushDestinationProjection,
) -> str:
    digest = hashlib.sha256()
    for value in (
        "file-notes-network-config-v1",
        configuration_fingerprint,
        destination.scheme,
        destination.host,
        str(destination.port),
        destination.repository_path,
        destination.destination_ref,
        destination.ssh_user or "",
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
        _digest_text(digest, value)
    return digest.hexdigest()


def _render_private_config(
    facts: tuple[_AuthorizedConfigFact, ...],
    object_format: GitObjectFormat,
) -> bytes:
    if (
        type(object_format) is not str
        or object_format not in {"sha1", "sha256"}
    ):
        raise NetworkContextError("invalid_source_objects")
    lines = [
        "[core]\n",
        (
            "\trepositoryFormatVersion = 0\n"
            if object_format == "sha1"
            else "\trepositoryFormatVersion = 1\n"
        ),
        "\tfileMode = true\n",
        "\tbare = true\n",
        "\tlogAllRefUpdates = false\n",
    ]
    if object_format == "sha256":
        lines.extend(
            (
                "[extensions]\n",
                "\tobjectFormat = sha256\n",
            )
        )
    if facts:
        lines.append("[credential]\n")
    for fact in facts:
        name = fact.key.rsplit(".", 1)[-1]
        rendered_value = f" {fact.value}" if fact.value else ""
        lines.append(f"\t{name} ={rendered_value}\n")
    return "".join(lines).encode("utf-8")


def _render_ssh_adapter(
    *,
    python: _PinnedExecutable,
    openssh: OpenSSHInvocationSpec,
    routing: tuple[tuple[str, str], ...],
    routing_fingerprint: str,
) -> bytes:
    interpreter = str(python.path)
    if (
        any(character.isspace() for character in interpreter)
        or len(interpreter.encode()) > 180
        or not _is_hex_fingerprint(routing_fingerprint)
        or not openssh._validate()
    ):
        raise NetworkContextError("invalid_executable")
    routing_keys = tuple(key for key, _value in routing)
    openssh_prefix = (
        str(openssh._executable.path),
        *_OPENSSH_FIXED_ARGUMENTS,
    )
    source = f"""#!{interpreter} -I
import hashlib
import os
import shlex
import sys

ROUTING_KEYS = {routing_keys!r}
EXPECTED_ROUTING_FINGERPRINT = {routing_fingerprint!r}
OPENSSH_PREFIX = {openssh_prefix!r}
HOST_KEY = {_SSH_ROUTE_HOST!r}
USER_KEY = {_SSH_ROUTE_USER!r}
PORT_KEY = {_SSH_ROUTE_PORT!r}
PATH_KEY = {_SSH_ROUTE_PATH!r}
AGENT_KEY = {_SSH_ROUTE_AGENT!r}
NO_AGENT = {_NO_AGENT!r}

def digest_text(digest, value):
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)

routing = {{}}
digest = hashlib.sha256()
digest_text(digest, "file-notes-network-environment-v1")
for key in ROUTING_KEYS:
    value = os.environ.get(key)
    if value is None:
        raise SystemExit(126)
    routing[key] = value
    digest_text(digest, key)
    digest_text(digest, value)
if digest.hexdigest() != EXPECTED_ROUTING_FINGERPRINT:
    raise SystemExit(126)

host = routing[HOST_KEY]
user = routing[USER_KEY]
port = routing[PORT_KEY]
repository_path = routing[PATH_KEY]
agent_socket = routing[AGENT_KEY]
expected_host = f"{{user}}@{{host}}"

args = sys.argv[1:]
send_protocol = args[:2] == ["-o", "SendEnv=GIT_PROTOCOL"]
if send_protocol:
    args = args[2:]
if args[:1] == ["-p"]:
    if args[:2] != ["-p", port]:
        raise SystemExit(126)
    args = args[2:]
if len(args) != 2 or args[0] != expected_host:
    raise SystemExit(126)
try:
    remote = shlex.split(args[1], posix=True)
except ValueError:
    raise SystemExit(126) from None
if remote not in (
    ["git-upload-pack", repository_path],
    ["git-receive-pack", repository_path],
):
    raise SystemExit(126)
remote_command = shlex.join((remote[0], repository_path))

child_environment = {{"LC_ALL": "C"}}
protocol = os.environ.get("GIT_PROTOCOL")
if send_protocol:
    if protocol != "version=2":
        raise SystemExit(126)
    child_environment["GIT_PROTOCOL"] = protocol
elif protocol is not None:
    raise SystemExit(126)
if agent_socket != NO_AGENT:
    if os.environ.get("SSH_AUTH_SOCK") != agent_socket:
        raise SystemExit(126)
    child_environment["SSH_AUTH_SOCK"] = agent_socket
openssh_argv = (
    *OPENSSH_PREFIX,
    "-o",
    (
        "IdentityAgent=none"
        if agent_socket == NO_AGENT
        else f"IdentityAgent={{agent_socket}}"
    ),
    "-o",
    f"HostName={{host}}",
    "-p",
    port,
    "-l",
    user,
    "--",
    host,
    remote_command,
)
os.execve(OPENSSH_PREFIX[0], openssh_argv, child_environment)
"""
    return source.encode("utf-8")


def _supported_credential_helpers() -> frozenset[str]:
    platform = "linux" if sys.platform.startswith("linux") else sys.platform
    return _PLATFORM_CREDENTIAL_HELPERS.get(platform, frozenset())


def _pin_git_dispatch_executable(
    name: str,
    git_exec_directory: _PinnedDirectory,
) -> _PinnedGitDispatchExecutable:
    exec_path_candidate = git_exec_directory.path / name
    try:
        metadata = exec_path_candidate.stat(follow_symlinks=False)
    except OSError:
        raise NetworkContextError("invalid_executable")
    if (
        exec_path_candidate.name != name
        or not (
            stat.S_ISREG(metadata.st_mode)
            or stat.S_ISLNK(metadata.st_mode)
        )
        or metadata.st_uid not in {os.geteuid(), 0}
        or (
            stat.S_ISREG(metadata.st_mode)
            and metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        )
    ):
        raise NetworkContextError("invalid_executable")
    target = _pin_executable(str(exec_path_candidate), os.defpath)
    return _PinnedGitDispatchExecutable(
        exec_path_candidate,
        _filesystem_identity(metadata),
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_dev,
        metadata.st_nlink,
        target,
    )


def _pin_credential_helpers(
    configuration: _NetworkConfigRecord,
    git_exec_directory: _PinnedDirectory,
) -> tuple[tuple[str, _PinnedGitDispatchExecutable], ...]:
    names = tuple(
        dict.fromkeys(
            fact.value
            for fact in configuration.facts
            if fact.key.lower() == "credential.helper" and fact.value
        )
    )
    pinned: list[tuple[str, _PinnedGitDispatchExecutable]] = []
    for name in names:
        if name not in _supported_credential_helpers():
            raise NetworkContextError("invalid_configuration")
        alias = f"git-credential-{name}"
        try:
            executable = _pin_git_dispatch_executable(
                alias,
                git_exec_directory,
            )
        except NetworkContextError:
            raise NetworkContextError("invalid_configuration") from None
        pinned.append((alias, executable))
    return tuple(pinned)


def _runtime_search_path(
    git: _PinnedExecutable,
) -> str:
    value = str(git.path.parent)
    if not _safe_environment_value("PATH", value):
        raise NetworkContextError("invalid_environment")
    selected = shutil.which("git", path=value)
    if selected is None or Path(selected).resolve() != git.path:
        raise NetworkContextError("invalid_executable")
    return value


def _safe_owned_directory_mode(metadata: os.stat_result) -> bool:
    if metadata.st_uid not in {os.geteuid(), 0}:
        return False
    group_or_other_write = metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    return not group_or_other_write or (
        metadata.st_uid == 0 and bool(metadata.st_mode & stat.S_ISVTX)
    )


def _capture_safe_ancestors(path: Path) -> tuple[_PinnedAncestor, ...]:
    captured: list[_PinnedAncestor] = []
    for ancestor in path.parents:
        try:
            metadata = ancestor.stat(follow_symlinks=False)
        except OSError:
            raise NetworkContextError("unsafe_filesystem") from None
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or not _safe_owned_directory_mode(metadata)
        ):
            raise NetworkContextError("unsafe_filesystem")
        captured.append(
            _PinnedAncestor(
                ancestor,
                _filesystem_identity(metadata),
                metadata.st_uid,
                metadata.st_gid,
                stat.S_IMODE(metadata.st_mode),
                metadata.st_dev,
            )
        )
    return tuple(captured)


def _pin_agent_socket(value: str) -> _PinnedSocket:
    if not _safe_environment_value("SSH_AUTH_SOCK", value):
        raise NetworkContextError("invalid_environment")
    try:
        candidate = Path(value)
        canonical = candidate.resolve(strict=True)
        metadata = candidate.stat(follow_symlinks=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise NetworkContextError("invalid_environment") from None
    if (
        not candidate.is_absolute()
        or candidate != canonical
        or not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise NetworkContextError("invalid_environment")
    try:
        ancestors = _capture_safe_ancestors(canonical)
    except NetworkContextError:
        raise NetworkContextError("invalid_environment") from None
    return _PinnedSocket(
        canonical,
        _filesystem_identity(metadata),
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_dev,
        ancestors,
    )


def _pin_executable(value: str, search_path: str) -> _PinnedExecutable:
    if not isinstance(value, str) or not value or "\0" in value:
        raise NetworkContextError("invalid_executable")
    selected = value if os.path.isabs(value) else shutil.which(value, path=search_path)
    if selected is None:
        raise NetworkContextError("invalid_executable")
    try:
        candidate = Path(selected).resolve(strict=True)
        metadata = candidate.stat(follow_symlinks=False)
    except (OSError, RuntimeError):
        raise NetworkContextError("invalid_executable") from None
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not candidate.is_absolute()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid not in {os.geteuid(), 0}
        or (metadata.st_uid != 0 and metadata.st_nlink != 1)
        or not metadata.st_mode & 0o111
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise NetworkContextError("invalid_executable")
    return _PinnedExecutable(
        candidate,
        _filesystem_identity(metadata),
        metadata.st_uid,
        metadata.st_gid,
        mode,
        metadata.st_dev,
        metadata.st_nlink,
        _capture_safe_ancestors(candidate),
    )


def _pin_git_exec_directory(
    value: str | os.PathLike[str],
) -> _PinnedDirectory:
    try:
        candidate = Path(value)
        canonical = candidate.resolve(strict=True)
        metadata = candidate.stat(follow_symlinks=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise NetworkContextError("invalid_executable") from None
    if (
        not candidate.is_absolute()
        or candidate != canonical
        or os.pathsep in str(canonical)
        or "\0" in str(canonical)
        or "\n" in str(canonical)
        or "\r" in str(canonical)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in {os.geteuid(), 0}
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise NetworkContextError("invalid_executable")
    try:
        ancestors = _capture_safe_ancestors(canonical)
    except NetworkContextError:
        raise NetworkContextError("invalid_executable") from None
    return _PinnedDirectory(
        canonical,
        _filesystem_identity(metadata),
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_dev,
        ancestors,
    )


def _safe_temporary_parent(
    configured: str | os.PathLike[str] | None,
) -> _PinnedDirectory:
    try:
        candidate = Path(tempfile.gettempdir() if configured is None else configured)
        parent = candidate.resolve(strict=True)
        metadata = parent.stat(follow_symlinks=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        raise NetworkContextError("unsafe_filesystem") from None
    if (
        not parent.is_absolute()
        or not stat.S_ISDIR(metadata.st_mode)
        or not _safe_owned_directory_mode(metadata)
    ):
        raise NetworkContextError("unsafe_filesystem")
    return _PinnedDirectory(
        parent,
        _filesystem_identity(metadata),
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_dev,
        _capture_safe_ancestors(parent),
    )


def _repository_matches(repository: RepositoryIdentity) -> bool:
    for path_value, expected in (
        (repository.worktree_root, repository.worktree_identity),
        (repository.git_dir, repository.git_dir_identity),
        (repository.git_common_dir, repository.git_common_dir_identity),
    ):
        try:
            path = Path(path_value)
            canonical = path.resolve(strict=True)
            metadata = path.stat(follow_symlinks=False)
        except (OSError, RuntimeError):
            return False
        if (
            path != canonical
            or not stat.S_ISDIR(metadata.st_mode)
            or _filesystem_identity(metadata) != expected
        ):
            return False
    return True


def _source_record_matches(record: _SourceObjectRecord) -> bool:
    try:
        metadata = record.path.stat(follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and _filesystem_identity(metadata) == record.identity
        and metadata.st_uid == record.owner
        and metadata.st_gid == record.group
        and stat.S_IMODE(metadata.st_mode) == record.mode
        and metadata.st_dev == record.device
    )


def _capture_known_entry(
    root: Path,
    path: Path,
    kind: Literal["file", "directory", "executable"],
    relative_path: str = ".",
    *,
    expected_mode: int | None = None,
) -> _KnownEntry:
    metadata = path.stat(follow_symlinks=False)
    mode = stat.S_IMODE(metadata.st_mode)
    required_mode = (
        {
            "directory": _DIRECTORY_MODE,
            "file": _FILE_MODE,
            "executable": _ADAPTER_MODE,
        }[kind]
        if expected_mode is None
        else expected_mode
    )
    is_regular = kind in {"file", "executable"}
    if (
        metadata.st_uid != os.geteuid()
        or metadata.st_dev != root.stat(follow_symlinks=False).st_dev
        or mode != required_mode
        or (kind == "directory") != stat.S_ISDIR(metadata.st_mode)
        or is_regular != stat.S_ISREG(metadata.st_mode)
        or (is_regular and metadata.st_nlink != 1)
    ):
        raise NetworkContextError("unsafe_filesystem")
    digest = _file_digest(path) if is_regular else None
    return _KnownEntry(
        relative_path,
        kind,
        _filesystem_identity(metadata),
        metadata.st_uid,
        metadata.st_gid,
        mode,
        metadata.st_dev,
        metadata.st_nlink,
        metadata.st_size,
        digest,
    )


def _capture_parent_entry(parent: Path) -> _KnownEntry:
    metadata = parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode):
        raise NetworkContextError("unsafe_filesystem")
    return _KnownEntry(
        ".",
        "directory",
        _filesystem_identity(metadata),
        metadata.st_uid,
        metadata.st_gid,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_dev,
        metadata.st_nlink,
        metadata.st_size,
    )


def _known_entry_matches(
    root: Path,
    path: Path,
    entry: _KnownEntry,
    *,
    include_contents: bool,
    allow_directory_link_drift: bool = False,
) -> bool:
    try:
        metadata = path.stat(follow_symlinks=False)
        root_device = root.stat(follow_symlinks=False).st_dev
    except OSError:
        return False
    matches = (
        _filesystem_identity(metadata) == entry.identity
        and metadata.st_uid == entry.owner
        and metadata.st_gid == entry.group
        and stat.S_IMODE(metadata.st_mode) == entry.mode
        and metadata.st_dev == entry.device == root_device
        and (
            metadata.st_nlink == entry.link_count
            or (allow_directory_link_drift and entry.kind == "directory")
        )
        and (entry.kind == "directory") == stat.S_ISDIR(metadata.st_mode)
        and (entry.kind in {"file", "executable"})
        == stat.S_ISREG(metadata.st_mode)
    )
    if (
        not matches
        or entry.kind not in {"file", "executable"}
        or not include_contents
    ):
        return matches
    return metadata.st_size == entry.size and _file_digest(path) == entry.digest


def _parent_entry_matches(parent: Path, entry: _KnownEntry) -> bool:
    """Validate a safe parent without treating unrelated sticky-dir churn as drift."""
    try:
        metadata = parent.stat(follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(metadata.st_mode)
        and _filesystem_identity(metadata) == entry.identity
        and metadata.st_uid == entry.owner
        and metadata.st_gid == entry.group
        and stat.S_IMODE(metadata.st_mode) == entry.mode
        and metadata.st_dev == entry.device
    )


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filesystem_identity(metadata: os.stat_result) -> FileSystemIdentity:
    return FileSystemIdentity(metadata.st_dev, metadata.st_ino)


def _safe_environment_value(name: str, value: object) -> bool:
    safe_text = (
        isinstance(name, str)
        and isinstance(value, str)
        and bool(value)
        and "\0" not in name
        and "=" not in name
        and "\0" not in value
        and "\n" not in value
        and "\r" not in value
    )
    if not safe_text:
        return False
    if name == "PATH":
        return all(
            component and Path(component).is_absolute()
            for component in value.split(os.pathsep)
        )
    if name == "SSH_AUTH_SOCK":
        return "%" not in value and "$" not in value
    return True


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        return path == parent or path.is_relative_to(parent)
    except (OSError, RuntimeError):
        return True


def _is_hex_fingerprint(value: object) -> bool:
    return isinstance(value, str) and _HEX_256.fullmatch(value) is not None


def _digest_text(digest: object, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _require_posix() -> None:
    if os.name != "posix" or not hasattr(os, "geteuid"):
        raise NetworkContextError("unsupported_platform")


__all__ = [
    "NetworkCommandSettings",
    "NetworkConfigAuthorization",
    "NetworkContextError",
    "NetworkContextFactory",
    "NetworkContextLease",
    "NetworkGitExecutionContext",
    "OpenSSHInvocationSpec",
    "SourceObjectDirectoryAuthorization",
]
