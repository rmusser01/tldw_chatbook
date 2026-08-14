"""Provider-neutral immutable model-artifact contracts."""

from __future__ import annotations

import errno
import hashlib
import ipaddress
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, Literal, NoReturn, Self, TypeVar
from urllib.parse import urlsplit

from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json
from tldw_chatbook.Utils.path_validation import validate_path_simple
from . import leases as _leases
from .gguf_admission import GGUFMetadata, inspect_gguf_structure, open_local_gguf
from .leases import (
    ArtifactLeaseError,
    ArtifactLeaseKey,
    ArtifactLeaseTimeoutError,
    ArtifactOperationLease,
    ArtifactOperationLeaseSet,
    LeaseMode,
)


_CANONICAL_COMPONENT = re.compile(
    r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?\Z",
    re.ASCII,
)
_PORTABLE_FILE_COMPONENT = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z",
    re.ASCII,
)
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_URL_HOST_LABEL = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z",
    re.ASCII,
)
_URL_PATH = re.compile(r"(?:/[A-Za-z0-9._~!$&'()*+,;=:@%\-]*)*\Z", re.ASCII)
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})", re.ASCII)
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in range(1, 10)),
    }
)
_MANAGED_PATH_COMPONENTS = frozenset(
    {"active", "locks", "manifest.json", "ready", "staging"}
)
_DESCRIPTOR_SCHEMA_VERSION = 1
_REF_KEYS = frozenset({"artifact_id", "revision", "variant"})
_FILE_KEYS = frozenset({"path", "size_bytes", "sha256"})
_DESCRIPTOR_STRING_FIELDS = (
    "model_id",
    "consumer",
    "model_family",
    "upstream_repository",
    "upstream_revision",
    "precision",
    "license_id",
    "usage_notice",
    "runtime_name",
    "runtime_version_constraint",
)
_DESCRIPTOR_KEYS = frozenset(
    {
        "schema_version",
        "reference",
        "model_id",
        "role",
        "format",
        "consumer",
        "model_family",
        "upstream_repository",
        "upstream_revision",
        "source_url",
        "precision",
        "expected_installed_bytes",
        "license_id",
        "license_url",
        "usage_notice",
        "runtime_name",
        "runtime_version_constraint",
        "supported_os",
        "supported_architectures",
        "provenance",
        "files",
        "dependencies",
    }
)
_MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_KEYS = frozenset({"schema_version", "descriptor"})
# TASK-1694: schema for the marker file (``download-stage.json``) that
# identifies one service-owned download-staging operation -- ported from
# codex/task-595-managed-downloads-v2's service.py (that branch's own
# port of TASK-595's "service-owned staging and finalization" seam; see
# Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-
# reconciliation.md item 1). Distinct from `_MANIFEST_SCHEMA_VERSION`
# above: the marker names the OPERATION, not the installed artifact.
_DOWNLOAD_STAGE_SCHEMA_VERSION = 1
_DOWNLOAD_STAGE_KEYS = frozenset(
    {"schema_version", "reference", "descriptor_fingerprint"}
)
_READINESS_SCHEMA_VERSION = 1
_READINESS_KEYS = frozenset(
    {"schema_version", "root", "closure", "closure_fingerprint"}
)
_LOCAL_COPY_CHUNK_BYTES = 1024 * 1024
_LOCAL_COPY_PROGRESS_BYTES = 64 * _LOCAL_COPY_CHUNK_BYTES


def _never_cancelled() -> bool:
    return False


def _raise_if_install_cancelled(cancelled: Callable[[], bool]) -> None:
    if cancelled():
        raise ArtifactStateError("artifact installation cancelled")


_ACTIVE_SCHEMA_VERSION = 1
_ACTIVE_KEYS = frozenset({"schema_version", "root"})
_LIFECYCLE_LEASE_KEY = ArtifactLeaseKey("!lifecycle", "1", "writer")

# TASK-595: reserved lease identity for the single global managed-acquisition
# session. Defined once here (not in the not-yet-written acquisition module,
# to avoid a circular import) so both reconcile()'s staging GC and the future
# acquisition runtime (Task 6+) key off the exact same ArtifactLeaseKey.
# "#managed-acquisition" can never collide with a real ArtifactRef.artifact_id
# because canonical artifact identifiers are validated against
# _CANONICAL_COMPONENT, which forbids a leading "#".
ACQUISITION_SESSION_LEASE_KEY = ArtifactLeaseKey(
    "#managed-acquisition", "session", "global"
)

# Non-blocking probe timeout (seconds) shared by every lease this service
# (or its ArtifactAcquisitionService caller) only ever tries opportunistically
# rather than waiting on: reconcile()'s two staging-GC lease probes below
# (_gc_install_staging_orphan, _gc_managed_staging), and acquisition.py's own
# acquisition-session lease attempt in provision() (imported from there, not
# redefined -- this 0.1s value was previously triplicated across all three
# call sites). An immediate, typed failure beats a hang: whoever else holds
# the lease is busy RIGHT NOW, not "worth waiting for" at any of these sites.
NONBLOCKING_LEASE_TIMEOUT_SECONDS = 0.1

# Naming convention for install()'s ephemeral staging tempdirs (see install()
# below, which names then leases then creates each one -- never a bare
# tempfile.mkdtemp, whose combined name+create step would leave the
# directory visible on disk before its orphan-detection lease is held).
# reconcile() uses this same prefix to recognize candidate orphans among
# staging's top-level entries.
_INSTALL_STAGING_PREFIX = "install-"

# TASK-1697 (PR-1165 review, P2): top-level naming convention for a
# service-owned download stage's CANONICAL operation directory (see
# ``_download_stage_paths``: ``staging/download-<fingerprint>/``).
# ``reconcile()``'s staging GC (``_gc_download_staging`` below) uses this
# prefix to recognize candidate download-stage entries among staging's
# top-level entries -- distinct from a stage's ephemeral PRE-publish
# ``.download-<random>/`` tempdir (leading dot, ``tempfile.mkdtemp``'s own
# random suffix, never this fixed prefix), which this GC deliberately does
# NOT recognize: nothing ever looks up a stage by that temp name again, so
# a leftover one is handled by ``_download_stage_for``'s own failure-path
# cleanup (``_remove_incomplete_download_stage``) instead of reconcile().
_DOWNLOAD_STAGE_PREFIX = "download-"

# Filename SUFFIX of a managed-download fetch-state sidecar, addressed as
# a SIBLING of the ``managed/<id>/<rev>/<variant>`` payload directory it
# describes (``.../<variant>.fetch-state.json``), never a child of it.
# LEGACY LAYOUT ONLY: since TASK-1694 moved acquisition onto
# service-owned ``download-<fingerprint>/`` stages (whose ``state/``
# subtree holds resume metadata outside the promoted ``payload/``),
# nothing writes this layout any more. The GC below keeps recognizing
# it so a directory left by an earlier build is still reclaimed.
_MANAGED_FETCH_SIDECAR_SUFFIX = ".fetch-state.json"
_PathSnapshot = tuple[int, int, int, int, int, int]
_NodeIdentity = tuple[int, int, int]


def _install_staging_lease_key(staging_name: str) -> ArtifactLeaseKey:
    """Return the lease key one in-flight ``install()`` call holds.

    ``install()`` acquires this key, non-reentrantly, for the full lifetime
    of one staging tempdir -- from directory creation until its final
    cleanup (success or failure). ``reconcile()`` tries the same key
    non-blocking before deleting a candidate ``install-*`` orphan: if it is
    still held, a live install owns that directory and it must survive;
    if it is free, the directory was abandoned (e.g. by a crashed process,
    whose OS-level lock is released automatically) and is safe to remove.

    Reserved under the "#install-staging" namespace, which -- like
    ``ACQUISITION_SESSION_LEASE_KEY``'s "#managed-acquisition" -- can never
    collide with a real ``ArtifactRef.lease_key()``, since canonical
    artifact identifiers never start with "#".

    Args:
        staging_name: Basename of the ``install-*`` staging tempdir (unique
            per call, generated before the directory itself is created --
            see ``install()``).

    Returns:
        The reserved lease key scoped to that one staging directory.
    """

    return ArtifactLeaseKey("#install-staging", staging_name, "lock")


def _path_snapshot(info: os.stat_result) -> _PathSnapshot:
    return (
        info.st_dev,
        info.st_ino,
        stat.S_IFMT(info.st_mode),
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _node_identity(info: os.stat_result) -> _NodeIdentity:
    return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))


class ArtifactDescriptorError(ValueError):
    """Base error for artifact descriptor parsing and validation."""


class ArtifactDescriptorValidationError(ArtifactDescriptorError):
    """Raised when a directly constructed descriptor value is invalid."""


class ArtifactDescriptorParseError(ArtifactDescriptorError):
    """Raised when a serialized descriptor does not match the versioned shape."""


class ArtifactError(RuntimeError):
    """Base error for stable model-artifact service failures."""


class ArtifactPathError(ArtifactError):
    """Raised when a managed or source path is unsafe or invalid."""


class ArtifactIntegrityError(ArtifactError):
    """Raised when artifact payload bytes do not match their descriptor."""


class ArtifactConflictError(ArtifactError):
    """Raised when an immutable destination already contains other state."""


class ArtifactStateError(ArtifactError):
    """Raised when a managed-store operation cannot complete safely."""


class ArtifactInUseError(ArtifactStateError):
    """Raised when an artifact cannot be deleted while it is leased."""


class ArtifactDependencyError(ArtifactStateError):
    """Raised when an exact installed dependency closure is invalid."""


class ArtifactNotReadyError(ArtifactStateError):
    """Raised when no valid readiness record exists for an artifact."""


class ArtifactNotInstalledError(ArtifactStateError):
    """Raised when an exact managed artifact root is not installed."""


class ArtifactRemovalAvailability(str, Enum):
    """Bounded advisory result for an exact artifact removal probe."""

    AVAILABLE = "available"
    BUSY = "busy"


_REMOVAL_CLEANUP_OWNER_ATTRIBUTE = "_artifact_removal_cleanup_owner"


def _is_removal_control(error: BaseException) -> bool:
    return not isinstance(error, Exception)


def _take_attached_removal_cleanup_owner(
    error: BaseException,
) -> ArtifactRemovalAuthority | None:
    owner = getattr(error, _REMOVAL_CLEANUP_OWNER_ATTRIBUTE, None)
    if not isinstance(owner, ArtifactRemovalAuthority):
        return None
    setattr(error, _REMOVAL_CLEANUP_OWNER_ATTRIBUTE, None)
    return owner


def _compose_removal_cleanup_owners(
    owner: ArtifactRemovalAuthority | None,
    additional: ArtifactRemovalAuthority | None,
) -> ArtifactRemovalAuthority | None:
    if additional is None:
        return owner
    if owner is None:
        return additional
    owner._absorb_cleanup_owner(additional)
    return owner


def _take_direct_removal_cleanup_owner(
    error: BaseException,
) -> ArtifactRemovalAuthority | None:
    owner: ArtifactRemovalAuthority | None = None
    if isinstance(error, ArtifactRemovalCleanupError):
        owner = ArtifactRemovalCleanupError.take_cleanup_owner(error)
    return _compose_removal_cleanup_owners(
        owner,
        _take_attached_removal_cleanup_owner(error),
    )


def _absorb_removal_cleanup_owner(
    owner: ArtifactRemovalAuthority,
    error: BaseException,
) -> None:
    attached = _take_direct_removal_cleanup_owner(error)
    if attached is not None:
        owner._absorb_cleanup_owner(attached)


def _move_removal_cleanup_owner(
    winner: BaseException,
    loser: BaseException,
) -> None:
    losing_owner = _take_direct_removal_cleanup_owner(loser)
    if losing_owner is None:
        return
    winning_owner = _take_attached_removal_cleanup_owner(winner)
    if winning_owner is None:
        winning_owner = losing_owner
    else:
        winning_owner._absorb_cleanup_owner(losing_owner)
    setattr(winner, _REMOVAL_CLEANUP_OWNER_ATTRIBUTE, winning_owner)


def _sanitize_removal_exception(error: BaseException) -> BaseException:
    """Return a bounded exception.

    Exception-group identity cannot survive because its message and children
    are read-only; its control family and recursively sanitized children do.
    """

    if isinstance(error, BaseExceptionGroup):
        cleanup_owner = _take_direct_removal_cleanup_owner(error)
        children: list[BaseException] = []
        for child in error.exceptions:
            bounded_child = _sanitize_removal_exception(child)
            cleanup_owner = _compose_removal_cleanup_owners(
                cleanup_owner,
                _take_direct_removal_cleanup_owner(bounded_child),
            )
            children.append(bounded_child)
        bounded = BaseExceptionGroup("artifact removal control group", children)
        if cleanup_owner is not None:
            setattr(bounded, _REMOVAL_CLEANUP_OWNER_ATTRIBUTE, cleanup_owner)
        error.__traceback__ = None
        error.__context__ = None
        error.__cause__ = None
        if hasattr(error, "__notes__"):
            error.__notes__ = []
        return bounded

    exit_code = error.code if isinstance(error, SystemExit) else None
    error.args = ()
    if isinstance(error, SystemExit):
        error.code = exit_code if exit_code is None or type(exit_code) is int else 1
    error.__traceback__ = None
    error.__context__ = None
    error.__cause__ = None
    if hasattr(error, "__notes__"):
        error.__notes__ = []
    return error


def _sanitize_and_absorb_removal_cleanup_owner(
    owner: ArtifactRemovalAuthority,
    error: BaseException,
) -> BaseException:
    """Bound one losing error and retain every cleanup authority it carries."""

    bounded_error = _sanitize_removal_exception(error)
    _absorb_removal_cleanup_owner(owner, bounded_error)
    return bounded_error


def _raise_removal_control(
    control: BaseException,
    *,
    cleanup_owner: ArtifactRemovalAuthority | None = None,
    ordinary_failure: bool = False,
) -> None:
    """Re-raise one exact sanitized control object with optional ownership."""

    control = _sanitize_removal_exception(control)
    if ordinary_failure:
        control.add_note("artifact removal also encountered an ordinary failure")
    if cleanup_owner is not None:
        cleanup_owner._make_cleanup_only()
        existing_owner = getattr(control, _REMOVAL_CLEANUP_OWNER_ATTRIBUTE, None)
        if isinstance(existing_owner, ArtifactRemovalAuthority):
            existing_owner._absorb_cleanup_owner(cleanup_owner)
            cleanup_owner = existing_owner
        setattr(control, _REMOVAL_CLEANUP_OWNER_ATTRIBUTE, cleanup_owner)
    try:
        raise control from None
    except BaseException as raised:
        raised.__context__ = None
        raised.__cause__ = None
        raise


class ArtifactRole(str, Enum):
    """An artifact's position in a dependency closure."""

    ROOT = "root"
    DEPENDENCY = "dependency"


class ArtifactFormat(str, Enum):
    """Supported immutable artifact formats."""

    ONNX = "onnx"
    GGUF = "gguf"


class ProvenanceClass(str, Enum):
    """Integrity and curation claims that may be persisted."""

    CHATBOOK_CURATED = "chatbook_curated"
    INTEGRITY_VERIFIED = "integrity_verified"
    LOCAL_INTEGRITY_RECORDED = "local_integrity_recorded"


_EnumT = TypeVar(
    "_EnumT",
    ArtifactRole,
    ArtifactFormat,
    ProvenanceClass,
)


def _is_windows_reserved(component: str) -> bool:
    basename = component.split(".", 1)[0]
    return basename.casefold() in _WINDOWS_RESERVED_BASENAMES


def _validate_canonical_component(field_name: str, value: object) -> None:
    if type(value) is not str or _CANONICAL_COMPONENT.fullmatch(value) is None:
        raise ArtifactDescriptorValidationError(
            f"{field_name} must be a lowercase ASCII portable path component"
        )
    if _is_windows_reserved(value):
        raise ArtifactDescriptorValidationError(
            f"{field_name} uses a Windows reserved device name"
        )


def _validate_nonempty_text(field_name: str, value: object) -> None:
    if type(value) is not str or not value or value != value.strip():
        raise ArtifactDescriptorValidationError(
            f"{field_name} must be a non-empty canonical string"
        )
    if any(not character.isprintable() for character in value):
        raise ArtifactDescriptorValidationError(
            f"{field_name} must not contain non-printable characters"
        )


def _valid_url_hostname(hostname: str, *, bracketed: bool) -> bool:
    if bracketed:
        try:
            ipaddress.IPv6Address(hostname)
        except ValueError:
            return False
        return True
    if ":" in hostname:
        return False
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    labels = ascii_hostname.removesuffix(".").split(".")
    return (
        bool(labels)
        and all(_URL_HOST_LABEL.fullmatch(label) is not None for label in labels)
        and len(ascii_hostname) <= 254
    )


def _valid_url_authority(authority: str, hostname: str) -> bool:
    bracketed = authority.startswith("[")
    if bracketed:
        closing_bracket = authority.find("]")
        if closing_bracket < 0:
            return False
        raw_hostname = authority[1:closing_bracket]
        suffix = authority[closing_bracket + 1 :]
        if suffix and (not suffix.startswith(":") or not suffix[1:].isdigit()):
            return False
    else:
        if "[" in authority or "]" in authority or authority.count(":") > 1:
            return False
        raw_hostname, separator, port_text = authority.rpartition(":")
        if not separator:
            raw_hostname = authority
        elif not port_text.isdigit():
            return False
    return raw_hostname.casefold() == hostname.casefold() and _valid_url_hostname(
        hostname,
        bracketed=bracketed,
    )


def _validate_url(field_name: str, value: object) -> None:
    _validate_nonempty_text(field_name, value)
    assert isinstance(value, str)
    if (
        "?" in value
        or "#" in value
        or "\\" in value
        or any(character.isspace() for character in value)
    ):
        raise ArtifactDescriptorValidationError(
            f"{field_name} must not include whitespace, a query, or a fragment"
        )
    if _INVALID_PERCENT_ESCAPE.search(value) is not None:
        raise ArtifactDescriptorValidationError(
            f"{field_name} contains an invalid percent escape"
        )
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except ValueError as error:
        raise ArtifactDescriptorValidationError(
            f"{field_name} must be a valid credential-free HTTP(S) URL"
        ) from error
    if (
        parsed.scheme not in {"http", "https"}
        or not value.startswith(f"{parsed.scheme}://")
        or parsed.hostname is None
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not _valid_url_authority(parsed.netloc, parsed.hostname)
        or _URL_PATH.fullmatch(parsed.path) is None
    ):
        raise ArtifactDescriptorValidationError(
            f"{field_name} must be a valid credential-free HTTP(S) URL"
        )


def _validate_tuple(
    field_name: str,
    value: object,
    item_type: type[object],
    *,
    nonempty: bool,
) -> None:
    if type(value) is not tuple:
        raise ArtifactDescriptorValidationError(f"{field_name} must be a tuple")
    if nonempty and not value:
        raise ArtifactDescriptorValidationError(f"{field_name} must not be empty")
    if any(type(item) is not item_type for item in value):
        raise ArtifactDescriptorValidationError(
            f"{field_name} contains an invalid item"
        )


@dataclass(frozen=True, order=True)
class ArtifactRef:
    """The exact immutable identity of one managed artifact."""

    artifact_id: str
    revision: str
    variant: str

    def __post_init__(self) -> None:
        """Validate canonical, portable identity components."""

        _validate_canonical_component("artifact_id", self.artifact_id)
        _validate_canonical_component("revision", self.revision)
        _validate_canonical_component("variant", self.variant)

    def lease_key(self) -> ArtifactLeaseKey:
        """Return the corresponding operation-lease identity."""

        return ArtifactLeaseKey(
            artifact_id=self.artifact_id,
            revision=self.revision,
            variant=self.variant,
        )

    def to_dict(self) -> dict[str, str]:
        """Return the JSON-safe canonical reference shape."""

        return {
            "artifact_id": self.artifact_id,
            "revision": self.revision,
            "variant": self.variant,
        }


@dataclass(frozen=True)
class ArtifactFile:
    """Expected metadata for one contained artifact payload file."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """Validate the portable relative path, size, and digest."""

        if type(self.path) is not str or not self.path or "\\" in self.path:
            raise ArtifactDescriptorValidationError(
                "path must be a non-empty forward-slash relative path"
            )
        components = self.path.split("/")
        if any(
            not component
            or _PORTABLE_FILE_COMPONENT.fullmatch(component) is None
            or _is_windows_reserved(component)
            or component.casefold() in _MANAGED_PATH_COMPONENTS
            for component in components
        ):
            raise ArtifactDescriptorValidationError(
                "path contains an unsafe or reserved portable component"
            )
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ArtifactDescriptorValidationError(
                "size_bytes must be a nonnegative integer"
            )
        if (
            type(self.sha256) is not str
            or _LOWERCASE_SHA256.fullmatch(self.sha256) is None
        ):
            raise ArtifactDescriptorValidationError(
                "sha256 must be exactly 64 lowercase hexadecimal characters"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe canonical file shape."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ArtifactDescriptor:
    """Immutable metadata required to verify and use one artifact."""

    reference: ArtifactRef
    model_id: str
    role: ArtifactRole
    format: ArtifactFormat
    consumer: str
    model_family: str
    upstream_repository: str
    upstream_revision: str
    source_url: str
    precision: str
    expected_installed_bytes: int
    license_id: str
    license_url: str
    usage_notice: str
    runtime_name: str
    runtime_version_constraint: str
    supported_os: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    provenance: tuple[ProvenanceClass, ...]
    files: tuple[ArtifactFile, ...]
    dependencies: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        """Validate the complete immutable descriptor contract."""

        if type(self.reference) is not ArtifactRef:
            raise ArtifactDescriptorValidationError("reference must be an ArtifactRef")
        if type(self.role) is not ArtifactRole:
            raise ArtifactDescriptorValidationError("role must be an ArtifactRole")
        if type(self.format) is not ArtifactFormat:
            raise ArtifactDescriptorValidationError("format must be an ArtifactFormat")

        for field_name in _DESCRIPTOR_STRING_FIELDS:
            _validate_nonempty_text(field_name, getattr(self, field_name))
        local_only = self.provenance == (ProvenanceClass.LOCAL_INTEGRITY_RECORDED,)
        if type(self.source_url) is not str:
            raise ArtifactDescriptorValidationError(
                "source_url must be a non-empty canonical string"
            )
        if self.source_url:
            _validate_url("source_url", self.source_url)
        elif not local_only:
            raise ArtifactDescriptorValidationError(
                "source_url may be empty only for local integrity provenance"
            )

        if type(self.license_url) is not str:
            raise ArtifactDescriptorValidationError(
                "license_url must be a non-empty canonical string"
            )
        if self.license_url:
            _validate_url("license_url", self.license_url)
        elif not (local_only and self.license_id == "unknown"):
            raise ArtifactDescriptorValidationError(
                "license_url may be empty only for unknown local-import licensing"
            )

        if self.precision != self.reference.variant:
            raise ArtifactDescriptorValidationError(
                "precision must equal reference.variant"
            )
        if (
            type(self.expected_installed_bytes) is not int
            or self.expected_installed_bytes < 0
        ):
            raise ArtifactDescriptorValidationError(
                "expected_installed_bytes must be a nonnegative integer"
            )

        _validate_tuple(
            "supported_os",
            self.supported_os,
            str,
            nonempty=True,
        )
        _validate_tuple(
            "supported_architectures",
            self.supported_architectures,
            str,
            nonempty=True,
        )
        for field_name, values in (
            ("supported_os", self.supported_os),
            ("supported_architectures", self.supported_architectures),
        ):
            for value in values:
                _validate_canonical_component(field_name, value)

        _validate_tuple(
            "provenance",
            self.provenance,
            ProvenanceClass,
            nonempty=True,
        )
        if len(set(self.provenance)) != len(self.provenance):
            raise ArtifactDescriptorValidationError(
                "provenance must not contain duplicates"
            )
        if {
            ProvenanceClass.INTEGRITY_VERIFIED,
            ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
        }.issubset(self.provenance):
            raise ArtifactDescriptorValidationError(
                "provenance cannot combine independently verified and locally "
                "recorded integrity"
            )

        _validate_tuple("files", self.files, ArtifactFile, nonempty=True)
        paths = [item.path for item in self.files]
        if len(set(paths)) != len(paths):
            raise ArtifactDescriptorValidationError("files contains a duplicate path")
        casefold_paths = [path.casefold() for path in paths]
        if len(set(casefold_paths)) != len(casefold_paths):
            raise ArtifactDescriptorValidationError(
                "files contains a case-insensitive path collision"
            )
        if self.expected_installed_bytes != sum(item.size_bytes for item in self.files):
            raise ArtifactDescriptorValidationError(
                "expected installed bytes must equal the sum of file sizes"
            )

        _validate_tuple(
            "dependencies",
            self.dependencies,
            ArtifactRef,
            nonempty=False,
        )
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ArtifactDescriptorValidationError(
                "dependencies contains a duplicate reference"
            )
        dependency_ids = [item.artifact_id for item in self.dependencies]
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ArtifactDescriptorValidationError(
                "dependencies contains conflicting revisions or variants for "
                "one artifact identity"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the strict versioned JSON-safe descriptor shape."""

        return {
            "schema_version": _DESCRIPTOR_SCHEMA_VERSION,
            "reference": self.reference.to_dict(),
            "model_id": self.model_id,
            "role": self.role.value,
            "format": self.format.value,
            "consumer": self.consumer,
            "model_family": self.model_family,
            "upstream_repository": self.upstream_repository,
            "upstream_revision": self.upstream_revision,
            "source_url": self.source_url,
            "precision": self.precision,
            "expected_installed_bytes": self.expected_installed_bytes,
            "license_id": self.license_id,
            "license_url": self.license_url,
            "usage_notice": self.usage_notice,
            "runtime_name": self.runtime_name,
            "runtime_version_constraint": self.runtime_version_constraint,
            "supported_os": list(self.supported_os),
            "supported_architectures": list(self.supported_architectures),
            "provenance": [item.value for item in self.provenance],
            "files": [item.to_dict() for item in self.files],
            "dependencies": [item.to_dict() for item in self.dependencies],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> ArtifactDescriptor:
        """Parse the strict versioned descriptor shape.

        Args:
            raw: Mapping containing exactly the version 1 descriptor fields.

        Returns:
            A validated immutable descriptor.

        Raises:
            ArtifactDescriptorParseError: If the shape or any value is invalid.
        """

        try:
            _require_exact_keys(raw, _DESCRIPTOR_KEYS, "descriptor")
            schema_version = raw["schema_version"]
            if type(schema_version) is not int or schema_version != 1:
                raise ArtifactDescriptorParseError(
                    "schema_version must be the integer 1"
                )
            reference = _parse_reference(raw["reference"], "reference")
            role = _parse_enum(raw["role"], ArtifactRole, "role")
            artifact_format = _parse_enum(
                raw["format"],
                ArtifactFormat,
                "format",
            )
            supported_os = _parse_string_list(
                raw["supported_os"],
                "supported_os",
            )
            supported_architectures = _parse_string_list(
                raw["supported_architectures"],
                "supported_architectures",
            )
            provenance = tuple(
                _parse_enum(item, ProvenanceClass, f"provenance[{index}]")
                for index, item in enumerate(
                    _require_list(raw["provenance"], "provenance")
                )
            )
            files = tuple(
                _parse_file(item, f"files[{index}]")
                for index, item in enumerate(_require_list(raw["files"], "files"))
            )
            dependencies = tuple(
                _parse_reference(item, f"dependencies[{index}]")
                for index, item in enumerate(
                    _require_list(raw["dependencies"], "dependencies")
                )
            )
            string_fields = {
                field_name: _require_string(raw[field_name], field_name)
                for field_name in (
                    *_DESCRIPTOR_STRING_FIELDS,
                    "source_url",
                    "license_url",
                )
            }
            expected_installed_bytes = raw["expected_installed_bytes"]
            if type(expected_installed_bytes) is not int:
                raise ArtifactDescriptorParseError(
                    "expected_installed_bytes must be an integer"
                )
            return cls(
                reference=reference,
                role=role,
                format=artifact_format,
                expected_installed_bytes=expected_installed_bytes,
                supported_os=supported_os,
                supported_architectures=supported_architectures,
                provenance=provenance,
                files=files,
                dependencies=dependencies,
                **string_fields,
            )
        except ArtifactDescriptorParseError:
            raise
        except ArtifactDescriptorValidationError as error:
            raise ArtifactDescriptorParseError(str(error)) from error


def _require_exact_keys(
    raw: object,
    expected: frozenset[str],
    context: str,
) -> None:
    if not isinstance(raw, Mapping):
        raise ArtifactDescriptorParseError(f"{context} must be a mapping")
    if any(type(key) is not str for key in raw):
        raise ArtifactDescriptorParseError(
            f"{context} has invalid keys: keys must be strings"
        )
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ArtifactDescriptorParseError(
            f"{context} has invalid keys: missing={missing}, unknown={unknown}"
        )


def _require_string(raw: object, context: str) -> str:
    if type(raw) is not str:
        raise ArtifactDescriptorParseError(f"{context} must be a string")
    return raw


def _require_list(raw: object, context: str) -> list[object]:
    if type(raw) is not list:
        raise ArtifactDescriptorParseError(f"{context} must be a list")
    return raw


def _parse_string_list(raw: object, context: str) -> tuple[str, ...]:
    values = _require_list(raw, context)
    return tuple(
        _require_string(item, f"{context}[{index}]")
        for index, item in enumerate(values)
    )


def _parse_enum(
    raw: object,
    enum_type: type[_EnumT],
    context: str,
) -> _EnumT:
    value = _require_string(raw, context)
    try:
        return enum_type(value)
    except ValueError as error:
        raise ArtifactDescriptorParseError(
            f"{context} has an unsupported value"
        ) from error


def _parse_reference(raw: object, context: str) -> ArtifactRef:
    _require_exact_keys(raw, _REF_KEYS, context)
    assert isinstance(raw, Mapping)
    try:
        return ArtifactRef(
            artifact_id=_require_string(raw["artifact_id"], f"{context}.artifact_id"),
            revision=_require_string(raw["revision"], f"{context}.revision"),
            variant=_require_string(raw["variant"], f"{context}.variant"),
        )
    except ArtifactDescriptorValidationError as error:
        raise ArtifactDescriptorParseError(f"{context}: {error}") from error


def _parse_file(raw: object, context: str) -> ArtifactFile:
    _require_exact_keys(raw, _FILE_KEYS, context)
    assert isinstance(raw, Mapping)
    size_bytes = raw["size_bytes"]
    if type(size_bytes) is not int:
        raise ArtifactDescriptorParseError(f"{context}.size_bytes must be an integer")
    try:
        return ArtifactFile(
            path=_require_string(raw["path"], f"{context}.path"),
            size_bytes=size_bytes,
            sha256=_require_string(raw["sha256"], f"{context}.sha256"),
        )
    except ArtifactDescriptorValidationError as error:
        raise ArtifactDescriptorParseError(f"{context}: {error}") from error


def closure_fingerprint(
    root: ArtifactRef,
    dependencies: Iterable[ArtifactRef],
) -> str:
    """Return the canonical identity of an exact dependency closure.

    Dependency order and duplicate references do not affect the result. The root
    is included exactly once.
    """

    if type(root) is not ArtifactRef:
        raise ArtifactDescriptorValidationError("root must be an ArtifactRef")
    try:
        closure = {root}
        for dependency in dependencies:
            if type(dependency) is not ArtifactRef:
                raise ArtifactDescriptorValidationError(
                    "dependencies must contain only ArtifactRef values"
                )
            closure.add(dependency)
    except TypeError as error:
        raise ArtifactDescriptorValidationError(
            "dependencies must be an iterable of ArtifactRef values"
        ) from error

    payload = json.dumps(
        [reference.to_dict() for reference in sorted(closure)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"artifact-closure-v1\0" + payload).hexdigest()


@dataclass(frozen=True)
class InstalledArtifact:
    """One visible entry in the managed installed-artifact tree."""

    path: Path
    descriptor: ArtifactDescriptor | None
    ready: bool
    active: bool
    error: str | None = None


@dataclass(frozen=True)
class LocalGGUFImportProgress:
    """Path-private progress emitted by one local GGUF import."""

    phase: Literal["copy", "inspect", "verify", "finalize"]
    file: str | None
    bytes_done: int
    bytes_total: int


@dataclass(frozen=True)
class LocalGGUFImportResult:
    """Exact managed reference and convergence outcome for an import."""

    reference: ArtifactRef
    already_installed: bool


def _ignore_local_import_progress(_progress: LocalGGUFImportProgress) -> None:
    """Accept one import progress update without side effects."""


def _local_gguf_descriptor(
    metadata: GGUFMetadata,
    *,
    digest: str,
    size_bytes: int,
) -> ArtifactDescriptor:
    """Build one path-private descriptor from staged GGUF facts."""

    revision = f"sha256-{digest}"
    variant = (
        f"filetype-{metadata.file_type}"
        if metadata.file_type is not None
        else "imported"
    )
    label = (metadata.model_name or "").strip() or f"Imported GGUF {digest[:8]}"
    reference = ArtifactRef(f"local-gguf-{digest[:16]}", revision, variant)
    return ArtifactDescriptor(
        reference=reference,
        model_id=label,
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.GGUF,
        consumer="unassigned",
        model_family=metadata.architecture,
        upstream_repository="local-import",
        upstream_revision=revision,
        source_url="",
        precision=variant,
        expected_installed_bytes=size_bytes,
        license_id="unknown",
        license_url="",
        usage_notice=(
            "Imported from a user-selected local file; license and runtime "
            "compatibility are not verified."
        ),
        runtime_name="unassigned",
        runtime_version_constraint="none",
        supported_os=("unassigned",),
        supported_architectures=("unassigned",),
        provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
        files=(ArtifactFile("model.gguf", size_bytes, digest),),
    )


@dataclass(frozen=True)
class ArtifactDiskUsage:
    """Logical managed-store byte totals and current filesystem capacity."""

    installed_bytes: int
    staging_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class ReconcileReport:
    """Deterministic results from an explicit managed-store reconciliation."""

    readiness_created: int
    state_removed: int
    corrupt_artifacts: tuple[Path, ...]
    staging_entries: tuple[Path, ...]
    staging_removed: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactDependencyHandle:
    """Verified exact managed dependencies and their payload paths."""

    references: tuple[ArtifactRef, ...]
    paths: tuple[tuple[ArtifactRef, Path], ...]

    @property
    def lease_keys(self) -> tuple[ArtifactLeaseKey, ...]:
        """Return exact dependency lease keys in canonical order."""

        return tuple(reference.lease_key() for reference in self.references)


@dataclass(frozen=True)
class ArtifactHandle:
    """A verified exact artifact closure and its managed payload paths."""

    root: ArtifactRef
    closure: tuple[ArtifactRef, ...]
    closure_fingerprint: str
    paths: tuple[tuple[ArtifactRef, Path], ...]

    @property
    def lease_keys(self) -> tuple[ArtifactLeaseKey, ...]:
        """Return exact closure lease keys in canonical order."""

        return tuple(reference.lease_key() for reference in self.closure)

    @property
    def resident_identity(self) -> tuple[ArtifactRef, str]:
        """Return the cache-safe identity of this resident closure."""

        return (self.root, self.closure_fingerprint)


@dataclass(frozen=True)
class _ManagedDownloadStage:
    """One service-owned resumable download operation inside staging.

    TASK-1694: ported from codex/task-595-managed-downloads-v2 (see the
    reconciliation doc referenced above ``_DOWNLOAD_STAGE_SCHEMA_VERSION``).
    ``payload`` is the ONLY subtree ``_finalize_download_stage`` ever
    renames into the immutable destination; ``marker`` and ``state`` never
    enter it, so resume metadata written into ``state`` cannot end up
    inside a promoted artifact by construction.
    """

    reference: ArtifactRef
    descriptor_fingerprint: str
    operation: Path
    marker: Path
    payload: Path
    state: Path
    operation_identity: _NodeIdentity
    marker_identity: _NodeIdentity
    payload_identity: _NodeIdentity | None
    state_identity: _NodeIdentity | None


@dataclass(frozen=True)
class _ReadinessRecord:
    root: ArtifactRef
    closure: tuple[ArtifactRef, ...]
    closure_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _READINESS_SCHEMA_VERSION,
            "root": self.root.to_dict(),
            "closure": [reference.to_dict() for reference in self.closure],
            "closure_fingerprint": self.closure_fingerprint,
        }


_HandleT = TypeVar("_HandleT", ArtifactHandle, ArtifactDependencyHandle)


class _LeasedArtifactHandle(Generic[_HandleT]):
    """Own shared operation leases for an already acquired handle."""

    def __init__(
        self,
        handle: _HandleT,
        lease_set: ArtifactOperationLeaseSet,
    ) -> None:
        self.handle: _HandleT = handle
        self._lease_set: ArtifactOperationLeaseSet | None = lease_set

    def close(self) -> None:
        """Release the owned shared lease set idempotently."""

        lease_set = self._lease_set
        if lease_set is not None:
            lease_set.release()
            self._lease_set = None

    def __enter__(self) -> Self:
        """Return this already acquired handle without reacquiring leases."""

        if self._lease_set is None:
            raise ArtifactStateError("leased artifact handle is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Release leases, preserving a body exception over cleanup failure."""

        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"lease context cleanup failed: {cleanup_error!r}")
            for note in getattr(cleanup_error, "__notes__", ()):
                exc.add_note(note)


class LeasedArtifactHandle(_LeasedArtifactHandle[ArtifactHandle]):
    """Own shared operation leases for an acquired artifact closure."""


class LeasedArtifactDependencyHandle(_LeasedArtifactHandle[ArtifactDependencyHandle]):
    """Own shared operation leases for verified exact dependencies."""


class ArtifactRemovalAuthority:
    """Own ordered exclusive leases for one exact installed artifact target."""

    def __init__(
        self,
        service: ModelArtifactService,
        reference: ArtifactRef,
        lifecycle_lease: ArtifactOperationLease | None,
        target_lease: ArtifactOperationLease | None,
        target_identity: _NodeIdentity | None = None,
    ) -> None:
        self._service = service
        self._reference = reference
        self._target_identity = target_identity
        self._lifecycle_lease: ArtifactOperationLease | None = lifecycle_lease
        self._target_lease: ArtifactOperationLease | None = target_lease
        self._committed = False
        self._additional_cleanup_owners: list[ArtifactRemovalAuthority] = []

    def commit(self) -> None:
        """Remove the pinned target once without reacquiring either lease."""

        if self._target_identity is None:
            raise ArtifactStateError("removal authority is cleanup-only")
        if self._target_lease is None or self._lifecycle_lease is None:
            raise ArtifactStateError("removal authority is closed")
        if self._committed:
            raise ArtifactStateError("removal authority already committed")
        self._service._revalidate_removal_target(
            self._reference,
            self._target_identity,
        )
        self._service._delete_under_leases(self._reference)
        self._committed = True

    def _admit_target(self, target_identity: _NodeIdentity) -> None:
        """Make this close-only owner commit-capable after exact target pinning."""

        if self._target_identity is not None:
            raise ArtifactStateError("removal authority target is already pinned")
        self._target_identity = target_identity

    def _make_cleanup_only(self) -> None:
        """Irrevocably revoke commit capability before cleanup can escape."""

        self._target_identity = None

    def _absorb_cleanup_owner(self, owner: ArtifactRemovalAuthority) -> None:
        """Compose another exact cleanup-only authority into this owner."""

        self._make_cleanup_only()
        owner._make_cleanup_only()
        if owner is self or any(
            existing is owner for existing in self._additional_cleanup_owners
        ):
            return
        self._additional_cleanup_owners.append(owner)

    def _has_cleanup(self) -> bool:
        return (
            self._target_lease is not None
            or self._lifecycle_lease is not None
            or any(owner._has_cleanup() for owner in self._additional_cleanup_owners)
        )

    def close(self) -> None:
        """Release target then lifecycle authority, retaining failures for retry."""

        self._make_cleanup_only()
        control: BaseException | None = None
        ordinary_error: BaseException | None = None
        for attribute in ("_target_lease", "_lifecycle_lease"):
            lease = getattr(self, attribute)
            if lease is None:
                continue
            try:
                lease.release()
            except BaseException as error:
                if not _is_removal_control(error):
                    if control is None:
                        ordinary_error = error
                        break
                    if ordinary_error is None:
                        ordinary_error = error
                    else:
                        _sanitize_and_absorb_removal_cleanup_owner(self, error)
                    continue
                if control is None:
                    control = error
                else:
                    bounded_error = _sanitize_removal_exception(error)
                    _move_removal_cleanup_owner(control, bounded_error)
                continue
            setattr(self, attribute, None)
        retained_owners: list[ArtifactRemovalAuthority] = []
        for owner in self._additional_cleanup_owners:
            try:
                owner.close()
            except BaseException as error:
                retained_owners.append(owner)
                if _is_removal_control(error):
                    if control is None:
                        control = error
                    else:
                        bounded_error = _sanitize_removal_exception(error)
                        _move_removal_cleanup_owner(control, bounded_error)
                elif ordinary_error is None:
                    ordinary_error = error
                else:
                    _sanitize_and_absorb_removal_cleanup_owner(self, error)
            else:
                if owner._has_cleanup():
                    retained_owners.append(owner)
        self._additional_cleanup_owners = retained_owners
        if control is not None:
            if ordinary_error is not None:
                ordinary_error = _sanitize_removal_exception(ordinary_error)
                _move_removal_cleanup_owner(control, ordinary_error)
            _raise_removal_control(
                control,
                ordinary_failure=ordinary_error is not None,
            )
        if ordinary_error is not None:
            raise ordinary_error

    def __enter__(self) -> Self:
        """Return this already acquired authority."""

        if self._target_identity is None:
            raise ArtifactStateError("removal authority is cleanup-only")
        if self._target_lease is None or self._lifecycle_lease is None:
            raise ArtifactStateError("removal authority is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        """Close while preserving body control flow over cleanup failure."""

        cleanup_error: BaseException | None = None
        try:
            self.close()
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is None:
            return
        cleanup_error = _sanitize_and_absorb_removal_cleanup_owner(
            self,
            cleanup_error,
        )
        if exc is not None and _is_removal_control(exc):
            _raise_removal_control(
                exc,
                cleanup_owner=self,
                ordinary_failure=not _is_removal_control(cleanup_error),
            )
        if exc is not None:
            _sanitize_and_absorb_removal_cleanup_owner(self, exc)
        if _is_removal_control(cleanup_error):
            _raise_removal_control(
                cleanup_error,
                cleanup_owner=self,
                ordinary_failure=exc is not None,
            )
        _raise_removal_cleanup_failure(
            self,
            note="artifact removal commit also failed" if exc is not None else None,
        )


class ArtifactRemovalCleanupError(ArtifactStateError):
    """Bounded failure carrying exact retryable removal cleanup ownership."""

    __slots__ = ("_cleanup_owner",)

    def __init__(self, cleanup_owner: ArtifactRemovalAuthority) -> None:
        cleanup_owner._make_cleanup_only()
        self._cleanup_owner: ArtifactRemovalAuthority | None = cleanup_owner
        super().__init__("artifact removal cleanup is incomplete")

    def take_cleanup_owner(self) -> ArtifactRemovalAuthority | None:
        """Transfer retained cleanup ownership exactly once."""

        owner = self._cleanup_owner
        self._cleanup_owner = None
        return owner


def _raise_removal_cleanup_failure(
    cleanup_owner: ArtifactRemovalAuthority,
    *,
    note: str | None = None,
) -> NoReturn:
    """Raise one bounded cleanup carrier without exposing superseded context."""

    failure = ArtifactRemovalCleanupError(cleanup_owner)
    if note is not None:
        failure.add_note(note)
    try:
        raise failure from None
    except ArtifactRemovalCleanupError as raised:
        raised.__context__ = None
        raised.__cause__ = None
        raise


def take_artifact_removal_cleanup_owner(
    error: BaseException,
) -> ArtifactRemovalAuthority | None:
    """Take exact removal cleanup ownership from one bounded failure/control."""

    return _take_direct_removal_cleanup_owner(error)


class ModelArtifactService:
    """Own verified immutable artifacts beneath one resolved local root."""

    def __init__(
        self,
        root: Path,
        *,
        lease_timeout_seconds: float = 5.0,
    ) -> None:
        """Create or open a managed artifact store.

        Args:
            root: Service-owned local store root.
            lease_timeout_seconds: Maximum wait for each writer lease.

        Raises:
            TypeError: If ``root`` is not a path.
            ValueError: If the timeout is invalid.
            ArtifactPathError: If the store layout cannot be established safely.
        """

        if not isinstance(root, Path):
            raise TypeError("root must be a Path")
        if (
            isinstance(lease_timeout_seconds, bool)
            or not isinstance(lease_timeout_seconds, (int, float))
            or not math.isfinite(lease_timeout_seconds)
            or lease_timeout_seconds < 0
        ):
            raise ValueError("lease_timeout_seconds must be finite and nonnegative")
        try:
            self._root = root.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as error:
            raise ArtifactPathError("failed to resolve artifact store root") from error
        self._lease_timeout_seconds = float(lease_timeout_seconds)
        self._artifacts_path = self._root / "artifacts"
        self._active_path = self._root / "active"
        self._ready_path = self._root / "ready"
        self._staging_path = self._root / "staging"
        self._locks_path = self._root / "locks"
        for path in (
            self._artifacts_path,
            self._active_path,
            self._ready_path,
            self._staging_path,
            self._locks_path,
        ):
            self._ensure_owned_directory(path)
        managed_roots = (
            self._root,
            self._artifacts_path,
            self._active_path,
            self._ready_path,
            self._staging_path,
            self._locks_path,
        )
        try:
            self._managed_root_identities = {
                path: _node_identity(path.stat(follow_symlinks=False))
                for path in managed_roots
            }
        except OSError as error:
            raise ArtifactPathError("failed to inspect managed store roots") from error
        for path in managed_roots:
            self._assert_managed_path(path)

    @property
    def artifacts_path(self) -> Path:
        """Return the managed immutable-artifact directory."""

        return self._artifacts_path

    @property
    def staging_path(self) -> Path:
        """Return the non-loadable same-filesystem staging directory."""

        return self._staging_path

    @property
    def locks_path(self) -> Path:
        """Return the managed lock-file root used by artifact operation leases.

        Exposed (TASK-595 Task 6) so ``ArtifactAcquisitionService`` -- a
        composed caller in a different module -- can acquire its own
        reserved-namespace leases (e.g. ``ACQUISITION_SESSION_LEASE_KEY``)
        against the exact same lock root this service uses internally,
        without reaching into a private attribute.
        """

        return self._locks_path

    def artifact_path(self, reference: ArtifactRef) -> Path:
        """Return the contained final path for one validated reference."""

        if type(reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        return (
            self._artifacts_path
            / reference.artifact_id
            / reference.revision
            / reference.variant
        )

    def readiness_path(self, reference: ArtifactRef) -> Path:
        """Return the contained readiness-record path for one exact reference."""

        if type(reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        return (
            self._ready_path
            / reference.artifact_id
            / reference.revision
            / f"{reference.variant}.json"
        )

    def active_path(self, artifact_id: str) -> Path:
        """Return the contained active-selector path for one artifact identity."""

        _validate_canonical_component("artifact_id", artifact_id)
        return self._active_path / f"{artifact_id}.json"

    def delete(self, reference: ArtifactRef) -> None:
        """Delete one exact artifact after invalidating affected derived state."""

        try:
            with self.acquire_removal_authority(reference) as authority:
                authority.commit()
        except ArtifactError:
            raise
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire or release artifact deletion leases"
            ) from error
        except OSError as error:
            raise ArtifactStateError("artifact deletion I/O failed") from error

    def probe_removal_availability(
        self,
        reference: ArtifactRef,
    ) -> ArtifactRemovalAvailability:
        """Return advisory available/busy truth without mutating store state."""

        if type(reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        lifecycle_lease: ArtifactOperationLease | None = None
        target_lease: ArtifactOperationLease | None = None
        primary_error: BaseException | None = None
        timed_out = False
        try:
            self._assert_managed_path(self._locks_path)
            lifecycle_lease = ArtifactOperationLease(
                self._locks_path,
                _LIFECYCLE_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=NONBLOCKING_LEASE_TIMEOUT_SECONDS,
            )
            lifecycle_lease.acquire()
            self._assert_managed_path(self._locks_path)
            target_lease = ArtifactOperationLease(
                self._locks_path,
                reference.lease_key(),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=NONBLOCKING_LEASE_TIMEOUT_SECONDS,
            )
            target_lease.acquire()
        except ArtifactLeaseTimeoutError:
            timed_out = True
        except BaseException as error:
            primary_error = error
        cleanup_owner = ArtifactRemovalAuthority(
            self,
            reference,
            lifecycle_lease,
            target_lease,
        )
        cleanup_error: BaseException | None = None
        try:
            cleanup_owner.close()
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            cleanup_error = _sanitize_and_absorb_removal_cleanup_owner(
                cleanup_owner,
                cleanup_error,
            )
        if primary_error is not None and _is_removal_control(primary_error):
            _raise_removal_control(
                primary_error,
                cleanup_owner=cleanup_owner if cleanup_owner._has_cleanup() else None,
                ordinary_failure=(
                    cleanup_error is not None and not _is_removal_control(cleanup_error)
                ),
            )
        if cleanup_error is not None and _is_removal_control(cleanup_error):
            if primary_error is not None:
                _sanitize_and_absorb_removal_cleanup_owner(
                    cleanup_owner,
                    primary_error,
                )
            _raise_removal_control(
                cleanup_error,
                cleanup_owner=cleanup_owner if cleanup_owner._has_cleanup() else None,
                ordinary_failure=primary_error is not None or timed_out,
            )
        if cleanup_error is not None:
            if primary_error is not None:
                _sanitize_and_absorb_removal_cleanup_owner(
                    cleanup_owner,
                    primary_error,
                )
            _raise_removal_cleanup_failure(
                cleanup_owner,
                note=(
                    "artifact removal probe also failed"
                    if primary_error is not None or timed_out
                    else None
                ),
            )
        if primary_error is not None:
            _sanitize_and_absorb_removal_cleanup_owner(
                cleanup_owner,
                primary_error,
            )
            if cleanup_owner._has_cleanup():
                _raise_removal_cleanup_failure(cleanup_owner)
            raise ArtifactStateError("artifact removal probe failed") from None
        return (
            ArtifactRemovalAvailability.BUSY
            if timed_out
            else ArtifactRemovalAvailability.AVAILABLE
        )

    def acquire_removal_authority(
        self,
        reference: ArtifactRef,
    ) -> ArtifactRemovalAuthority:
        """Acquire ordered exclusive authority over one exact installed target."""

        if type(reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        lifecycle_lease: ArtifactOperationLease | None = None
        target_lease: ArtifactOperationLease | None = None
        primary_error: BaseException | None = None
        try:
            self._assert_managed_path(self._locks_path)
            lifecycle_lease = ArtifactOperationLease(
                self._locks_path,
                _LIFECYCLE_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=self._lease_timeout_seconds,
            )
            lifecycle_lease.acquire()
            self._assert_managed_path(self._locks_path)
            target_lease = ArtifactOperationLease(
                self._locks_path,
                reference.lease_key(),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=self._lease_timeout_seconds,
            )
            target_lease.acquire()
        except BaseException as error:
            primary_error = error
        if primary_error is not None:
            cleanup_owner = ArtifactRemovalAuthority(
                self,
                reference,
                lifecycle_lease,
                target_lease,
            )
            cleanup_error: BaseException | None = None
            try:
                cleanup_owner.close()
            except BaseException as error:
                cleanup_error = error
            if cleanup_error is not None:
                cleanup_error = _sanitize_and_absorb_removal_cleanup_owner(
                    cleanup_owner,
                    cleanup_error,
                )
            if _is_removal_control(primary_error):
                _raise_removal_control(
                    primary_error,
                    cleanup_owner=(
                        cleanup_owner if cleanup_owner._has_cleanup() else None
                    ),
                    ordinary_failure=(
                        cleanup_error is not None
                        and not _is_removal_control(cleanup_error)
                    ),
                )
            if cleanup_error is not None and _is_removal_control(cleanup_error):
                _sanitize_and_absorb_removal_cleanup_owner(
                    cleanup_owner,
                    primary_error,
                )
                _raise_removal_control(
                    cleanup_error,
                    cleanup_owner=(
                        cleanup_owner if cleanup_owner._has_cleanup() else None
                    ),
                    ordinary_failure=True,
                )
            if cleanup_error is not None:
                _sanitize_and_absorb_removal_cleanup_owner(
                    cleanup_owner,
                    primary_error,
                )
                _raise_removal_cleanup_failure(
                    cleanup_owner,
                    note="artifact removal authority setup also failed",
                )
            _sanitize_and_absorb_removal_cleanup_owner(
                cleanup_owner,
                primary_error,
            )
            if cleanup_owner._has_cleanup():
                _raise_removal_cleanup_failure(cleanup_owner)
            if isinstance(primary_error, ArtifactLeaseTimeoutError):
                raise ArtifactInUseError(
                    "artifact is in use and cannot be deleted"
                ) from None
            if isinstance(primary_error, ArtifactLeaseError):
                lease_failure = ArtifactStateError(
                    "failed to acquire or release artifact deletion leases"
                )
                raise lease_failure from None
            raise ArtifactStateError("artifact removal authority failed") from None

        assert lifecycle_lease is not None
        assert target_lease is not None
        authority = ArtifactRemovalAuthority(
            self,
            reference,
            lifecycle_lease,
            target_lease,
        )
        target_error: BaseException | None = None
        target_identity: _NodeIdentity | None = None
        try:
            target_identity = self._removal_target_identity(reference)
        except BaseException as error:
            target_error = error
        if target_error is None:
            assert target_identity is not None
            authority._admit_target(target_identity)
            return authority

        target_cleanup_error: BaseException | None = None
        try:
            authority.close()
        except BaseException as error:
            target_cleanup_error = error
        if target_cleanup_error is not None:
            target_cleanup_error = _sanitize_and_absorb_removal_cleanup_owner(
                authority,
                target_cleanup_error,
            )
        if _is_removal_control(target_error):
            _raise_removal_control(
                target_error,
                cleanup_owner=authority if authority._has_cleanup() else None,
                ordinary_failure=(
                    target_cleanup_error is not None
                    and not _is_removal_control(target_cleanup_error)
                ),
            )
        if target_cleanup_error is not None and _is_removal_control(
            target_cleanup_error
        ):
            _sanitize_and_absorb_removal_cleanup_owner(
                authority,
                target_error,
            )
            _raise_removal_control(
                target_cleanup_error,
                cleanup_owner=authority,
                ordinary_failure=True,
            )
        if target_cleanup_error is not None:
            _sanitize_and_absorb_removal_cleanup_owner(
                authority,
                target_error,
            )
            _raise_removal_cleanup_failure(
                authority,
                note="artifact removal target setup also failed",
            )
        target_is_missing = type(target_error) is ArtifactStateError and (
            target_error.args == ("installed artifact does not exist",)
        )
        _sanitize_and_absorb_removal_cleanup_owner(
            authority,
            target_error,
        )
        if authority._has_cleanup():
            _raise_removal_cleanup_failure(authority)
        if target_is_missing:
            raise ArtifactStateError("installed artifact does not exist") from None
        raise ArtifactStateError("artifact removal authority failed") from None

    def reconcile(self) -> ReconcileReport:
        """Verify installed roots and reconcile derived state explicitly."""

        try:
            self._assert_managed_path(self._locks_path)
            with ArtifactOperationLease(
                self._locks_path,
                _LIFECYCLE_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                return self._reconcile_under_lifecycle()
        except ArtifactError:
            raise
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire or release artifact reconciliation leases"
            ) from error
        except OSError as error:
            raise ArtifactStateError("artifact reconciliation I/O failed") from error

    def _reconcile_under_lifecycle(self) -> ReconcileReport:
        for root in (
            self._locks_path,
            self._artifacts_path,
            self._ready_path,
            self._active_path,
            self._staging_path,
        ):
            self._assert_managed_path(root)

        staging_entries = self._staging_entries()
        corrupt_paths: set[Path] = set()
        descriptors: dict[ArtifactRef, ArtifactDescriptor] = {}
        for path, reference in self._installed_candidates():
            if reference is None:
                corrupt_paths.add(path)
                continue
            self._assert_managed_path(self._locks_path)
            with ArtifactOperationLeaseSet(
                self._locks_path,
                (reference.lease_key(),),
                LeaseMode.SHARED,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                try:
                    item = self._read_manifest(path)
                    if item.reference != reference:
                        raise ArtifactConflictError(
                            "manifest reference does not match its directory"
                        )
                except ArtifactError:
                    corrupt_paths.add(path)
                else:
                    descriptors[reference] = item

        verified: set[ArtifactRef] = set()
        corrupt_references: set[ArtifactRef] = set()
        expected_readiness: dict[ArtifactRef, _ReadinessRecord] = {}
        roots = sorted(
            reference
            for reference, item in descriptors.items()
            if item.role is ArtifactRole.ROOT
        )
        for root_reference in roots:
            try:
                closure = self._resolve_closure(root_reference)
            except ArtifactError:
                continue
            self._assert_managed_path(self._locks_path)
            with ArtifactOperationLeaseSet(
                self._locks_path,
                tuple(reference.lease_key() for reference in closure),
                LeaseMode.SHARED,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                try:
                    if self._resolve_closure(root_reference) != closure:
                        raise ArtifactDependencyError(
                            "dependency closure changed during reconciliation"
                        )
                    closure_valid = True
                    for reference in closure:
                        if reference in corrupt_references:
                            closure_valid = False
                            continue
                        if reference in verified:
                            continue
                        expected_role = (
                            ArtifactRole.ROOT
                            if reference == root_reference
                            else ArtifactRole.DEPENDENCY
                        )
                        try:
                            self._verify_installed(reference, expected_role)
                        except ArtifactError:
                            corrupt_references.add(reference)
                            corrupt_paths.add(self.artifact_path(reference))
                            closure_valid = False
                        else:
                            verified.add(reference)
                    if closure_valid:
                        expected_readiness[root_reference] = _ReadinessRecord(
                            root=root_reference,
                            closure=closure,
                            closure_fingerprint=closure_fingerprint(
                                root_reference,
                                closure,
                            ),
                        )
                except ArtifactDependencyError:
                    continue

        for reference, item in sorted(descriptors.items()):
            if reference in verified or reference in corrupt_references:
                continue
            self._assert_managed_path(self._locks_path)
            with ArtifactOperationLeaseSet(
                self._locks_path,
                (reference.lease_key(),),
                LeaseMode.SHARED,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                try:
                    self._verify_installed(reference, item.role)
                except ArtifactError:
                    corrupt_references.add(reference)
                    corrupt_paths.add(self.artifact_path(reference))
                else:
                    verified.add(reference)

        state_removed = 0
        readiness_created = 0
        preserved_readiness: set[ArtifactRef] = set()
        invalid_readiness: list[Path] = []
        for path in self._state_files(self._ready_path, 3):
            reference = self._readiness_ref_from_path(path)
            expected = (
                expected_readiness.get(reference) if reference is not None else None
            )
            try:
                current = (
                    self._read_readiness(reference) if reference is not None else None
                )
            except ArtifactStateError:
                current = None
            if reference is not None and expected is not None:
                if current == expected:
                    preserved_readiness.add(reference)
                continue
            invalid_readiness.append(path)
        for path in invalid_readiness:
            self._remove_state_path(path, "failed to remove invalid readiness state")
            state_removed += 1
        for reference, record in sorted(expected_readiness.items()):
            if reference in preserved_readiness:
                continue
            path = self.readiness_path(reference)
            if self._state_path_exists(path):
                mode = path.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                    self._remove_state_path(
                        path,
                        "failed to remove invalid readiness state",
                    )
                state_removed += 1
            self._write_readiness(record)
            readiness_created += 1

        for path in self._state_files(self._active_path, 1):
            artifact_id = self._active_id_from_path(path)
            try:
                selected = (
                    self._read_active(artifact_id) if artifact_id is not None else None
                )
            except ArtifactStateError:
                selected = None
            if selected not in expected_readiness:
                self._remove_state_path(
                    path,
                    "failed to remove invalid active selector",
                )
                state_removed += 1

        staging_removed = self._gc_staging(staging_entries)

        return ReconcileReport(
            readiness_created=readiness_created,
            state_removed=state_removed,
            corrupt_artifacts=tuple(
                sorted(corrupt_paths, key=lambda path: path.as_posix())
            ),
            staging_entries=staging_entries,
            staging_removed=staging_removed,
        )

    def _delete_under_leases(self, reference: ArtifactRef) -> None:
        target = self.artifact_path(reference)
        if not self._managed_path_exists(target):
            raise ArtifactStateError("installed artifact does not exist")
        self._assert_managed_path(target)

        invalidated_roots: set[ArtifactRef] = {reference}
        invalidated_paths: set[Path] = set()
        own_readiness = self.readiness_path(reference)
        if self._state_path_exists(own_readiness):
            invalidated_paths.add(own_readiness)
        for path in self._state_files(self._ready_path, 3):
            readiness_ref = self._readiness_ref_from_path(path)
            if readiness_ref is None:
                continue
            try:
                record = self._read_readiness(readiness_ref)
            except ArtifactStateError:
                continue
            if reference in record.closure:
                invalidated_roots.add(record.root)
                invalidated_paths.add(path)

        affected_artifact_ids = {
            reference.artifact_id,
            *(root.artifact_id for root in invalidated_roots),
        }
        active_paths: set[Path] = set()
        for artifact_id in affected_artifact_ids:
            path = self.active_path(artifact_id)
            if not self._state_path_exists(path):
                continue
            try:
                selected = self._read_active(artifact_id)
            except ArtifactStateError:
                active_paths.add(path)
            else:
                if selected == reference or selected in invalidated_roots:
                    active_paths.add(path)

        for path in sorted(invalidated_paths, key=lambda item: item.as_posix()):
            self._remove_state_path(path, "failed to remove invalidated readiness")
        for path in sorted(active_paths, key=lambda item: item.as_posix()):
            self._remove_state_path(path, "failed to remove affected active selector")
        try:
            shutil.rmtree(target)
        except OSError as error:
            raise ArtifactStateError("failed to remove installed artifact") from error
        for parent in (target.parent, target.parent.parent):
            try:
                self._assert_managed_path(parent)
                parent.rmdir()
            except OSError:
                break

    def _removal_target_identity(self, reference: ArtifactRef) -> _NodeIdentity:
        target = self.artifact_path(reference)
        if not self._managed_path_exists(target):
            raise ArtifactStateError("installed artifact does not exist")
        self._assert_managed_path(target)
        try:
            return _node_identity(target.stat(follow_symlinks=False))
        except OSError as error:
            raise ArtifactStateError("failed to pin artifact removal target") from error

    def _revalidate_removal_target(
        self,
        reference: ArtifactRef,
        expected_identity: _NodeIdentity,
    ) -> None:
        target = self.artifact_path(reference)
        if not self._managed_path_exists(target):
            raise ArtifactStateError("artifact removal target changed")
        self._assert_managed_path(target)
        try:
            current_identity = _node_identity(target.stat(follow_symlinks=False))
        except OSError as error:
            raise ArtifactStateError("failed to revalidate removal target") from error
        if current_identity != expected_identity:
            raise ArtifactStateError("artifact removal target changed")

    def _state_files(self, root: Path, record_depth: int) -> tuple[Path, ...]:
        self._assert_managed_path(root)
        before = _path_snapshot(root.stat(follow_symlinks=False))
        files: list[Path] = []

        def scan(directory: Path, depth: int) -> None:
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError as error:
                raise ArtifactPathError("failed to scan derived state") from error
            for entry in entries:
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError as error:
                    raise ArtifactPathError(
                        "failed to inspect derived state"
                    ) from error
                path = Path(entry.path)
                entry_depth = depth + 1
                if (
                    stat.S_ISDIR(mode)
                    and not stat.S_ISLNK(mode)
                    and entry_depth < record_depth
                ):
                    scan(path, entry_depth)
                else:
                    files.append(path)

        scan(root, 0)
        self._assert_managed_path(root)
        if _path_snapshot(root.stat(follow_symlinks=False)) != before:
            raise ArtifactPathError("derived state changed during traversal")
        return tuple(files)

    def _installed_candidates(
        self,
    ) -> tuple[tuple[Path, ArtifactRef | None], ...]:
        self._assert_managed_path(self._artifacts_path)
        inventory = self.list_installed()
        self._assert_managed_path(self._artifacts_path)
        if any(item.path == self._artifacts_path for item in inventory):
            raise ArtifactPathError("failed to scan artifacts during reconciliation")
        candidates = []
        for item in inventory:
            try:
                reference = (
                    item.descriptor.reference
                    if item.descriptor is not None
                    else ArtifactRef(*item.path.relative_to(self._artifacts_path).parts)
                )
            except (ArtifactDescriptorValidationError, TypeError, ValueError):
                reference = None
            candidates.append((item.path, reference))
        return tuple(candidates)

    def _staging_entries(self) -> tuple[Path, ...]:
        self._assert_managed_path(self._staging_path)
        before = _path_snapshot(self._staging_path.stat(follow_symlinks=False))
        try:
            entries = tuple(
                sorted(
                    (Path(entry.path) for entry in os.scandir(self._staging_path)),
                    key=lambda path: path.as_posix(),
                )
            )
        except OSError as error:
            raise ArtifactPathError("failed to scan staging entries") from error
        self._assert_managed_path(self._staging_path)
        if _path_snapshot(self._staging_path.stat(follow_symlinks=False)) != before:
            raise ArtifactPathError("staging changed during reconciliation scan")
        return entries

    def _gc_staging(self, staging_entries: tuple[Path, ...]) -> tuple[str, ...]:
        """Garbage-collect staging orphans, preserving resumable managed state.

        Three independent categories of staging content are reclaimed
        here; everything else (unrecognized top-level names -- including a
        download stage's ephemeral PRE-publish ``.download-<random>/``
        tempdir, see ``_DOWNLOAD_STAGE_PREFIX``) is left alone, matching
        reconcile()'s historical report-only behavior for arbitrary
        staging content:

        * ``install-*`` tempdirs abandoned by a crashed :meth:`install`
          call. A live in-flight call on any process still holds
          ``_install_staging_lease_key(name)``, so this is safe to attempt
          under the exclusive lifecycle lease this method's caller already
          holds -- no additional lease is required for this category.
        * ``managed/<id>/<rev>/<variant>`` download-staging directories
          whose ``fetch-state.json`` sidecar is missing, unparseable, or
          does not parse to the ``{"files": {...}}`` shape the fetch phase
          actually writes (see :meth:`_is_valid_managed_staging_entry`).
          LEGACY LAYOUT ONLY (see ``_MANAGED_FETCH_SIDECAR_SUFFIX``).
          Only removed when ``ACQUISITION_SESSION_LEASE_KEY`` is free
          (non-blocking); when busy, the entire ``managed`` subtree is left
          untouched for this reconcile pass so an in-progress acquisition's
          resumable state is never disturbed.
        * ``download-<fingerprint>/`` service-owned download stages
          (TASK-1697; see :meth:`_gc_download_staging`) whose reference is
          already installed -- redundant, since a resumed ``provision()``
          would skip straight past it. Same ``ACQUISITION_SESSION_LEASE_KEY``
          non-blocking gate as the legacy ``managed/`` category, and the
          same "never touch a resumable stage" guarantee.

        Args:
            staging_entries: Immediate children of the staging root, as
                returned by :meth:`_staging_entries`.

        Returns:
            Staging-relative POSIX paths of every entry removed.
        """

        removed: list[str] = []
        managed_root: Path | None = None
        download_operations: list[Path] = []
        for entry in staging_entries:
            if entry.name == "managed":
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise ArtifactPathError(
                        "failed to inspect staging entry"
                    ) from error
                if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                    managed_root = entry
                continue
            if entry.name.startswith(_DOWNLOAD_STAGE_PREFIX):
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except FileNotFoundError:
                    continue
                except OSError as error:
                    raise ArtifactPathError(
                        "failed to inspect staging entry"
                    ) from error
                if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                    download_operations.append(entry)
                continue
            if not entry.name.startswith(_INSTALL_STAGING_PREFIX):
                continue
            name = self._gc_install_staging_orphan(entry)
            if name is not None:
                removed.append(name)

        if managed_root is not None:
            removed.extend(self._gc_managed_staging(managed_root))

        if download_operations:
            removed.extend(self._gc_download_staging(tuple(download_operations)))

        return tuple(sorted(removed))

    def _gc_install_staging_orphan(self, entry: Path) -> str | None:
        """Remove one ``install-*`` staging directory if it is abandoned.

        Tries ``_install_staging_lease_key(entry.name)`` non-blocking: a
        live :meth:`install` call already holds this key for its whole
        staging lifetime, so a busy lease means the directory is still in
        use and must survive; a free lease means the directory was
        abandoned (its owner is gone, and the OS released the lease when
        that process or handle closed) and is safe to remove.

        Args:
            entry: Top-level staging directory matching the ``install-*``
                tempdir prefix used by :meth:`install`.

        Returns:
            The removed entry's staging-relative POSIX path, or None when
            an in-flight install still owns it.
        """

        self._assert_managed_path(self._locks_path)
        try:
            lease = ArtifactOperationLease(
                self._locks_path,
                _install_staging_lease_key(entry.name),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=NONBLOCKING_LEASE_TIMEOUT_SECONDS,
            )
            lease.acquire()
        except ArtifactLeaseTimeoutError:
            return None
        try:
            self._remove_state_path(entry, "failed to remove staging orphan")
        finally:
            lease.release()
        return entry.relative_to(self._staging_path).as_posix()

    def _gc_managed_staging(self, managed_root: Path) -> tuple[str, ...]:
        """GC orphaned managed-download staging entries.

        Requires ``ACQUISITION_SESSION_LEASE_KEY`` non-blocking; when busy
        (a managed acquisition may still be writing into this subtree),
        the whole managed staging tree is left untouched for this
        reconcile pass.

        Args:
            managed_root: The ``staging/managed`` directory (already
                confirmed to be a real, non-symlink directory).

        Returns:
            Staging-relative POSIX paths of every orphan entry removed.
        """

        self._assert_managed_path(self._locks_path)
        try:
            lease = ArtifactOperationLease(
                self._locks_path,
                ACQUISITION_SESSION_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=NONBLOCKING_LEASE_TIMEOUT_SECONDS,
            )
            lease.acquire()
        except ArtifactLeaseTimeoutError:
            return ()
        try:
            removed: list[str] = []
            for candidate in self._state_files(managed_root, 3):
                if self._is_valid_managed_staging_entry(managed_root, candidate):
                    continue
                self._remove_state_path(candidate, "failed to remove staging orphan")
                removed.append(candidate.relative_to(self._staging_path).as_posix())
            return tuple(removed)
        finally:
            lease.release()

    def _is_valid_managed_staging_entry(
        self,
        managed_root: Path,
        candidate: Path,
    ) -> bool:
        """Return whether ``candidate`` is part of a live managed download.

        The fetch-state sidecar for ``managed/<id>/<rev>/<variant>/`` lives
        as a SIBLING file -- ``managed/<id>/<rev>/<variant>.fetch-state.json``
        -- never inside the payload directory it describes (see
        acquisition.py's ``_fetch_sidecar_path`` for why). Both the payload
        directory AND its sidecar therefore surface as SEPARATE depth-three
        candidates from :meth:`_state_files`, and either one is classified
        by resolving its counterpart: a candidate is live only when BOTH
        the payload directory exists as a real (non-symlink) directory AND
        its sidecar parses to a JSON OBJECT with a ``"files"`` mapping --
        the exact shape ``ArtifactAcquisitionService``'s fetch phase always
        writes (see ``acquisition.py``'s ``_load_fetch_sidecar`` /
        ``_fetch_one_file``). Anything shallower, deeper, symlinked,
        missing its counterpart, or whose sidecar is unparseable or parses
        to something else entirely (e.g. a bare JSON list or string --
        valid JSON, but not a usable fetch-state record) is an orphan.
        ``_assert_managed_path`` rejects a symlink anywhere in the chain up
        to and including either path, so a symlinked payload directory or
        sidecar is correctly treated as an orphan without ever being
        followed.

        Args:
            managed_root: The ``staging/managed`` directory.
            candidate: One entry surfaced by scanning ``managed_root`` to a
                fixed depth of three (see :meth:`_state_files`) -- either
                the payload directory or its sibling sidecar file.

        Returns:
            True when the entry's resumable download state should survive
            garbage collection.
        """

        parts = candidate.relative_to(managed_root).parts
        if len(parts) != 3:
            return False
        name = parts[-1]
        if name.endswith(_MANAGED_FETCH_SIDECAR_SUFFIX):
            sidecar = candidate
            payload_dir = candidate.parent / name[: -len(_MANAGED_FETCH_SIDECAR_SUFFIX)]
        else:
            payload_dir = candidate
            sidecar = candidate.parent / f"{name}{_MANAGED_FETCH_SIDECAR_SUFFIX}"
        try:
            self._assert_managed_path(
                sidecar,
                allow_missing=True,
                target_must_be_directory=False,
            )
            payload = sidecar.read_text(encoding="utf-8")
            parsed = json.loads(payload)
        except (ArtifactPathError, OSError, ValueError):
            return False
        if not (isinstance(parsed, dict) and isinstance(parsed.get("files"), dict)):
            return False
        try:
            self._assert_managed_path(payload_dir)
        except (ArtifactPathError, OSError):
            return False
        return True

    def _gc_download_staging(self, operations: tuple[Path, ...]) -> tuple[str, ...]:
        """GC redundant service-owned download-stage operations (TASK-1697).

        Requires ``ACQUISITION_SESSION_LEASE_KEY`` non-blocking, exactly
        like :meth:`_gc_managed_staging`'s legacy-layout equivalent: a live
        ``provision()`` call holds this lease for its ENTIRE run (see
        ``ArtifactAcquisitionService.provision``), so when it is busy the
        whole batch is left untouched for this reconcile pass -- an
        in-progress acquisition's resumable state must never be disturbed.

        An operation is reclaimed only when BOTH:

        * :meth:`_download_stage_ownership` can prove it -- the marker is
          present, parses to the exact schema :meth:`_download_stage_for`
          writes, and its recorded ``descriptor_fingerprint`` matches the
          fingerprint encoded in the operation directory's OWN name. A
          marker that is missing, unparseable, or self-inconsistent in any
          way is treated as UNPROVABLE, not orphaned -- a deliberately more
          conservative choice than the legacy ``managed/`` GC (which
          reclaims a structurally-invalid entry outright): this layout's
          payload can hold a large, real, in-progress download, so a
          transient or partial marker read must never cost it.
        * the reference it names is ALREADY installed
          (``core.list_installed()``, checked via ``artifact_path``) -- the
          only on-disk-provable signal (reconcile() has no catalog) that
          this EXACT stage will never be resumed again: ``provision()``
          skips the fetch/pre-verify/install phases entirely for an
          already-installed reference, so nothing will ever open this
          stage for anything but a redundant discard (the same check
          ``ArtifactAcquisitionService._install_artifact`` /
          ``_finalize_download_stage`` itself makes on a retried finalize).
          A stage whose reference is NOT YET installed is exactly the
          "valid and resumable" case that must survive regardless of lease
          state -- this is what actually distinguishes an abandoned stage
          (e.g. a crash between ``_promote`` and
          ``_remove_finalized_download_stage`` on a LATER, successful
          resume) from one still waiting to be resumed.

        Args:
            operations: Top-level ``download-<fingerprint>/`` directories
                found directly under staging (already confirmed real,
                non-symlink directories).

        Returns:
            Staging-relative POSIX paths of every orphan operation removed.
        """

        self._assert_managed_path(self._locks_path)
        try:
            lease = ArtifactOperationLease(
                self._locks_path,
                ACQUISITION_SESSION_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=NONBLOCKING_LEASE_TIMEOUT_SECONDS,
            )
            lease.acquire()
        except ArtifactLeaseTimeoutError:
            return ()
        try:
            removed: list[str] = []
            for operation in operations:
                reference = self._download_stage_ownership(operation)
                if reference is None:
                    continue
                if not self._managed_path_exists(self.artifact_path(reference)):
                    continue
                self._remove_state_path(operation, "failed to remove staging orphan")
                removed.append(operation.relative_to(self._staging_path).as_posix())
            return tuple(removed)
        finally:
            lease.release()

    def _download_stage_ownership(self, operation: Path) -> ArtifactRef | None:
        """Prove (or refuse to prove) which reference owns ``operation``.

        Self-consistency is the ownership proof: reconcile() has no
        catalog/descriptor to check the marker's claims against (unlike
        :meth:`_read_download_stage_marker`, which always compares against
        a caller-supplied descriptor/reference) -- what it CAN check is
        that the marker's own recorded ``descriptor_fingerprint`` matches
        the fingerprint encoded in the name of the directory carrying it
        (``download-<fingerprint>``), which only happens if the marker was
        legitimately written by :meth:`_download_stage_for` for this exact
        operation.

        Containment is checked alongside ownership: ``operation``'s
        immediate children must be exactly its marker plus (optionally)
        ``payload/`` and ``state/``, none of them a symlink
        (:meth:`_download_stage_node_identity` asserts this per node), and
        neither subtree may itself contain a symlink anywhere
        (:meth:`_validate_download_stage_state` for ``state/``;
        :meth:`_assert_no_symlinks` for ``payload/``, which -- unlike
        ``state/`` -- holds arbitrary downloaded bytes, not JSON).

        Args:
            operation: A ``download-<fingerprint>/`` directory already
                confirmed to be a real, non-symlink directory.

        Returns:
            The ``ArtifactRef`` this operation's marker claims, or
            ``None`` when ownership or containment cannot be proven --
            callers must then leave ``operation`` completely alone.
        """

        marker = operation / "download-stage.json"
        payload = operation / "payload"
        state = operation / "state"
        name_fingerprint = operation.name[len(_DOWNLOAD_STAGE_PREFIX) :]
        if _LOWERCASE_SHA256.fullmatch(name_fingerprint) is None:
            return None
        try:
            entries = {entry.name: Path(entry.path) for entry in os.scandir(operation)}
        except OSError:
            return None
        payload_present = payload.name in entries
        state_present = state.name in entries
        expected = {marker.name}
        if payload_present:
            expected.add(payload.name)
        if state_present:
            expected.add(state.name)
        if set(entries) != expected:
            return None
        try:
            self._download_stage_node_identity(marker, directory=False)
            if payload_present:
                self._download_stage_node_identity(payload, directory=True)
            if state_present:
                self._download_stage_node_identity(state, directory=True)
        except (ArtifactPathError, OSError):
            return None
        try:
            with marker.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
            _require_exact_keys(raw, _DOWNLOAD_STAGE_KEYS, "download stage")
            assert isinstance(raw, Mapping)
            if (
                type(raw["schema_version"]) is not int
                or raw["schema_version"] != _DOWNLOAD_STAGE_SCHEMA_VERSION
                or type(raw["descriptor_fingerprint"]) is not str
                or raw["descriptor_fingerprint"] != name_fingerprint
            ):
                return None
            reference = _parse_reference(raw["reference"], "download stage.reference")
        except (
            ArtifactDescriptorError,
            AssertionError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ):
            return None
        if payload_present:
            try:
                self._assert_no_symlinks(payload)
            except ArtifactPathError:
                return None
        if state_present:
            try:
                self._validate_download_stage_state(state)
            except ArtifactPathError:
                return None
        return reference

    def _assert_no_symlinks(self, directory: Path) -> None:
        """Recursively reject a symlink anywhere inside ``directory``.

        Unlike :meth:`_validate_download_stage_state` (which additionally
        requires every JSON-suffixed file to actually parse -- ``state/``
        never holds anything but resume metadata), a download stage's
        ``payload/`` holds arbitrary downloaded file bytes, so only the
        symlink-containment property is checked here. Used by
        :meth:`_download_stage_ownership` to prove an orphan candidate's
        payload subtree is fully contained before it is ever handed to
        :meth:`_remove_state_path`.

        Args:
            directory: The subtree to walk.

        Raises:
            ArtifactPathError: A symlink was found anywhere under
                ``directory``, or the tree could not be scanned/inspected.
        """

        try:
            entries = tuple(os.scandir(directory))
        except OSError as error:
            raise ArtifactPathError(
                "failed to scan download staging payload"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError as error:
                raise ArtifactPathError(
                    "failed to inspect download staging payload"
                ) from error
            if stat.S_ISLNK(info.st_mode):
                raise ArtifactPathError("download staging payload contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                self._assert_no_symlinks(path)

    def _readiness_ref_from_path(self, path: Path) -> ArtifactRef | None:
        try:
            relative = path.relative_to(self._ready_path)
            if len(relative.parts) != 3 or path.suffix != ".json":
                return None
            return ArtifactRef(
                relative.parts[0],
                relative.parts[1],
                path.stem,
            )
        except (ArtifactDescriptorValidationError, ValueError):
            return None

    def _active_id_from_path(self, path: Path) -> str | None:
        try:
            relative = path.relative_to(self._active_path)
            if len(relative.parts) != 1 or path.suffix != ".json":
                return None
            artifact_id = path.stem
            _validate_canonical_component("artifact_id", artifact_id)
            return artifact_id
        except (ArtifactDescriptorValidationError, ValueError):
            return None

    def _remove_state_path(self, path: Path, message: str) -> None:
        self._assert_managed_path(path.parent)
        try:
            mode = path.stat(follow_symlinks=False).st_mode
            if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            raise ArtifactStateError(message) from error

    def _state_path_exists(self, path: Path) -> bool:
        self._assert_managed_path(path.parent, allow_missing=True)
        try:
            path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ArtifactPathError("failed to inspect derived state path") from error
        return True

    def activate(self, root_reference: ArtifactRef) -> ArtifactRef:
        """Verify or reuse one exact dependency closure, then select its root."""

        if type(root_reference) is not ArtifactRef:
            raise TypeError("root_reference must be an ArtifactRef")
        try:
            self._assert_managed_path(self._locks_path)
            with ArtifactOperationLease(
                self._locks_path,
                _LIFECYCLE_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                closure = self._resolve_closure(root_reference)
                keys = tuple(reference.lease_key() for reference in closure)
                self._assert_managed_path(self._locks_path)
                with ArtifactOperationLeaseSet(
                    self._locks_path,
                    keys,
                    LeaseMode.SHARED,
                    timeout_seconds=self._lease_timeout_seconds,
                ):
                    if self._resolve_closure(root_reference) != closure:
                        raise ArtifactDependencyError(
                            "dependency closure changed during activation"
                        )
                    expected = _ReadinessRecord(
                        root=root_reference,
                        closure=closure,
                        closure_fingerprint=closure_fingerprint(
                            root_reference,
                            closure,
                        ),
                    )
                    existing = self._try_read_readiness(root_reference)
                    if existing != expected:
                        self._remove_readiness(root_reference)
                        try:
                            for reference in closure:
                                expected_role = (
                                    ArtifactRole.ROOT
                                    if reference == root_reference
                                    else ArtifactRole.DEPENDENCY
                                )
                                self._verify_installed(reference, expected_role)
                            self._write_readiness(expected)
                        except ArtifactError:
                            self._remove_active_if_selects(root_reference)
                            raise
                    try:
                        active_path = self.active_path(root_reference.artifact_id)
                        self._assert_managed_path(
                            active_path,
                            allow_missing=True,
                            target_must_be_directory=False,
                        )
                        atomic_write_json(
                            active_path,
                            {
                                "schema_version": _ACTIVE_SCHEMA_VERSION,
                                "root": root_reference.to_dict(),
                            },
                        )
                    except OSError as error:
                        raise ArtifactStateError(
                            "failed to write active artifact selector"
                        ) from error
            return root_reference
        except ArtifactError:
            raise
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire artifact activation leases"
            ) from error
        except OSError as error:
            raise ArtifactStateError("artifact activation I/O failed") from error

    def acquire(self, root_reference: ArtifactRef) -> LeasedArtifactHandle:
        """Acquire shared leases for one unchanged strict readiness record."""

        if type(root_reference) is not ArtifactRef:
            raise TypeError("root_reference must be an ArtifactRef")
        record = self._read_readiness(root_reference)
        self._assert_managed_path(self._locks_path)
        lease_set = ArtifactOperationLeaseSet(
            self._locks_path,
            tuple(reference.lease_key() for reference in record.closure),
            LeaseMode.SHARED,
            timeout_seconds=self._lease_timeout_seconds,
        )
        try:
            lease_set.acquire()
        except ArtifactLeaseTimeoutError:
            raise
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire artifact closure leases"
            ) from error
        try:
            try:
                current = self._read_readiness(root_reference)
            except ArtifactStateError as error:
                raise ArtifactStateError(
                    "readiness record changed during artifact acquisition"
                ) from error
            if current != record:
                raise ArtifactStateError(
                    "readiness record changed during artifact acquisition"
                )
            paths = tuple(
                (reference, self.artifact_path(reference))
                for reference in record.closure
            )
            for _reference, path in paths:
                self._assert_managed_path(path)
            handle = ArtifactHandle(
                root=record.root,
                closure=record.closure,
                closure_fingerprint=record.closure_fingerprint,
                paths=paths,
            )
            return LeasedArtifactHandle(handle, lease_set)
        except BaseException as error:
            try:
                lease_set.release()
            except BaseException as cleanup_error:
                error.add_note(f"lease rollback cleanup failed: {cleanup_error!r}")
                for note in getattr(cleanup_error, "__notes__", ()):
                    error.add_note(note)
            raise

    def acquire_installed_root(
        self,
        reference: ArtifactRef,
    ) -> LeasedArtifactHandle:
        """Verify and lease one exact installed ROOT without derived state."""

        if type(reference) is not ArtifactRef:
            raise TypeError("reference must be an ArtifactRef")
        self._assert_managed_path(self._locks_path)
        lease_set = ArtifactOperationLeaseSet(
            self._locks_path,
            (reference.lease_key(),),
            LeaseMode.SHARED,
            timeout_seconds=self._lease_timeout_seconds,
        )
        try:
            lease_set.acquire()
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire installed artifact root lease"
            ) from error
        try:
            path = self.artifact_path(reference)
            self._assert_managed_path(path, allow_missing=True)
            try:
                path.stat(follow_symlinks=False)
            except FileNotFoundError:
                raise ArtifactNotInstalledError(
                    "exact artifact root is not installed"
                ) from None
            except OSError as error:
                raise ArtifactStateError(
                    "failed to inspect exact installed artifact root"
                ) from error
            self._verify_installed(reference, ArtifactRole.ROOT)
            canonical_root = path.resolve(strict=True)
            self._assert_managed_path(canonical_root)
            handle = ArtifactHandle(
                root=reference,
                closure=(reference,),
                closure_fingerprint=closure_fingerprint(reference, ()),
                paths=((reference, canonical_root),),
            )
            return LeasedArtifactHandle(handle, lease_set)
        except BaseException as error:
            try:
                lease_set.release()
            except BaseException as cleanup_error:
                error.add_note(f"lease rollback cleanup failed: {cleanup_error!r}")
                for note in getattr(cleanup_error, "__notes__", ()):
                    error.add_note(note)
            raise

    def acquire_dependencies(
        self,
        references: tuple[ArtifactRef, ...],
    ) -> LeasedArtifactDependencyHandle:
        """Verify and lease exact managed dependency artifacts."""

        if type(references) is not tuple:
            raise TypeError("references must be a tuple")
        if any(type(reference) is not ArtifactRef for reference in references):
            raise TypeError("references must contain only ArtifactRef values")
        ordered = tuple(sorted(set(references)))
        if not ordered:
            raise ValueError("references must contain at least one ArtifactRef")

        self._assert_managed_path(self._locks_path)
        lease_set = ArtifactOperationLeaseSet(
            self._locks_path,
            tuple(reference.lease_key() for reference in ordered),
            LeaseMode.SHARED,
            timeout_seconds=self._lease_timeout_seconds,
        )
        try:
            lease_set.acquire()
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire artifact dependency leases"
            ) from error
        try:
            for reference in ordered:
                self._verify_installed(reference, ArtifactRole.DEPENDENCY)
            paths = tuple(
                (reference, self.artifact_path(reference)) for reference in ordered
            )
            handle = ArtifactDependencyHandle(references=ordered, paths=paths)
            return LeasedArtifactDependencyHandle(handle, lease_set)
        except BaseException as error:
            try:
                lease_set.release()
            except BaseException as cleanup_error:
                error.add_note(f"lease rollback cleanup failed: {cleanup_error!r}")
                for note in getattr(cleanup_error, "__notes__", ()):
                    error.add_note(note)
            raise

    def list_installed(self) -> tuple[InstalledArtifact, ...]:
        """Return a deterministic manifest-only installed inventory."""

        installed: list[InstalledArtifact] = []

        def invalid(path: Path, message: str) -> None:
            installed.append(
                InstalledArtifact(
                    path=path,
                    descriptor=None,
                    ready=False,
                    active=False,
                    error=message,
                )
            )

        def state_flags(
            reference: ArtifactRef,
            role: ArtifactRole,
        ) -> tuple[bool, bool, str | None]:
            is_root = role is ArtifactRole.ROOT
            ready = False
            active = False
            errors: list[str] = []
            try:
                readiness = self._read_readiness(reference)
                ready = is_root and readiness.root == reference
            except ArtifactNotReadyError:
                pass
            except ArtifactStateError as error:
                errors.append(f"readiness: {error}")
            try:
                active_root = self._read_active(reference.artifact_id)
                active = is_root and active_root == reference
            except ArtifactNotReadyError:
                pass
            except ArtifactStateError as error:
                errors.append(f"active: {error}")
            return ready, active, "; ".join(errors) or None

        def visit(path: Path, depth: int) -> None:
            try:
                mode = path.stat(follow_symlinks=False).st_mode
            except OSError as error:
                invalid(path, f"cannot inspect managed entry: {error}")
                return
            if stat.S_ISLNK(mode):
                invalid(path, "managed entry is a symlink")
                return
            if not stat.S_ISDIR(mode):
                invalid(path, "managed entry is not a directory")
                return
            if depth == 3:
                try:
                    relative = path.relative_to(self._artifacts_path)
                    reference = ArtifactRef(*relative.parts)
                    descriptor = self._read_manifest(path)
                    if descriptor.reference != reference:
                        raise ArtifactConflictError(
                            "manifest reference does not match its directory"
                        )
                except (
                    ArtifactDescriptorError,
                    ArtifactError,
                    TypeError,
                    ValueError,
                ) as error:
                    invalid(path, str(error))
                    return
                ready, active, state_error = state_flags(
                    reference,
                    descriptor.role,
                )
                installed.append(
                    InstalledArtifact(
                        path=path,
                        descriptor=descriptor,
                        ready=ready,
                        active=active,
                        error=state_error,
                    )
                )
                return
            try:
                children = sorted(
                    (Path(entry.path) for entry in os.scandir(path)),
                    key=lambda child: child.name,
                )
            except OSError as error:
                invalid(path, f"cannot scan managed entry: {error}")
                return
            if not children:
                invalid(path, "managed entry has an incomplete directory identity")
                return
            for child in children:
                visit(child, depth + 1)

        try:
            self._assert_managed_path(self._artifacts_path)
            before = _path_snapshot(self._artifacts_path.stat(follow_symlinks=False))
            roots = sorted(
                (Path(entry.path) for entry in os.scandir(self._artifacts_path)),
                key=lambda child: child.name,
            )
            self._assert_managed_path(self._artifacts_path)
            if (
                _path_snapshot(self._artifacts_path.stat(follow_symlinks=False))
                != before
            ):
                raise ArtifactPathError(
                    "artifacts directory changed during inventory scan"
                )
            for root in roots:
                visit(root, 1)
            self._assert_managed_path(self._artifacts_path)
            if (
                _path_snapshot(self._artifacts_path.stat(follow_symlinks=False))
                != before
            ):
                raise ArtifactPathError(
                    "artifacts directory changed during inventory traversal"
                )
        except (ArtifactPathError, OSError, ValueError) as error:
            installed.clear()
            invalid(self._artifacts_path, f"cannot scan artifacts directory: {error}")
        return tuple(sorted(installed, key=lambda item: item.path.as_posix()))

    def disk_usage(self) -> ArtifactDiskUsage:
        """Return logical managed regular-file bytes and current free space."""

        try:
            installed_bytes = self._regular_tree_bytes(self._artifacts_path)
            staging_bytes = self._regular_tree_bytes(self._staging_path)
            self._assert_managed_path(self._root)
            self._assert_managed_path(self._artifacts_path)
            self._assert_managed_path(self._staging_path)
            free_bytes = shutil.disk_usage(self._root)[2]
            self._assert_managed_path(self._root)
            return ArtifactDiskUsage(
                installed_bytes=installed_bytes,
                staging_bytes=staging_bytes,
                free_bytes=free_bytes,
            )
        except ArtifactPathError:
            raise
        except OSError as error:
            raise ArtifactStateError("failed to account artifact disk usage") from error

    # -- TASK-1694: service-owned download-stage seam -----------------------
    #
    # Ported from codex/task-595-managed-downloads-v2's service.py (that
    # branch's own implementation of TASK-595's "service-owned staging and
    # finalization" design -- see Docs/superpowers/specs/2026-07-31-
    # verified-managed-model-downloads-design.md, "### 2. Service-owned
    # staging and finalization", and the reconciliation decision at
    # Docs/superpowers/reviews/2026-08-01-task-595-duplicate-implementation-
    # reconciliation.md, item 1).
    #
    # A "download stage" is a marked, contained operation directory
    # (``staging/download-<fingerprint>/``) with three children: a JSON
    # marker naming the exact reference and descriptor this stage belongs
    # to, a ``payload/`` subtree the downloader writes fetched files into,
    # and a ``state/`` subtree for resume metadata (e.g. a fetch-state
    # sidecar). ``_finalize_download_stage`` verifies and RENAMES only
    # ``payload/`` into the immutable destination -- ``marker``/``state``
    # never enter it, so resume metadata cannot end up inside a promoted
    # artifact by construction. This replaces the old
    # ``install(..., consume_source=True)`` promotion path for REMOTE
    # acquisition (``ArtifactAcquisitionService._install_artifact`` in
    # acquisition.py), which needed a sibling-file sidecar workaround to
    # keep resume state out of ``core.install``'s validated payload tree.
    # ``install()`` itself is unchanged and still used by local import,
    # whose copy/TOCTOU contract is different (see its own docstring).

    def _download_stage_for(
        self,
        descriptor: ArtifactDescriptor,
        *,
        create: bool,
    ) -> _ManagedDownloadStage | None:
        """Open or create the exact service-owned staging operation for a download."""

        if type(descriptor) is not ArtifactDescriptor:
            raise TypeError("descriptor must be an ArtifactDescriptor")
        if type(create) is not bool:
            raise TypeError("create must be a bool")
        operation, marker, payload, state, fingerprint = self._download_stage_paths(
            descriptor,
        )
        self._assert_managed_path(self._staging_path)
        if self._managed_path_exists(operation):
            return self._open_download_stage(
                descriptor,
                operation,
                marker,
                payload,
                state,
                fingerprint,
            )
        if not create:
            return None
        temporary: Path | None = None
        temporary_identity: _NodeIdentity | None = None
        try:
            temporary = Path(
                tempfile.mkdtemp(prefix=".download-", dir=self._staging_path)
            )
            temporary_identity = self._download_stage_node_identity(
                temporary,
                directory=True,
            )
        except OSError as error:
            raise ArtifactStateError(
                "failed to create download staging operation"
            ) from error
        try:
            temporary_marker = temporary / marker.name
            temporary_payload = temporary / payload.name
            temporary_state = temporary / state.name
            self._assert_managed_path(temporary)
            temporary_payload.mkdir()
            temporary_state.mkdir()
            atomic_write_json(
                temporary_marker,
                {
                    "schema_version": _DOWNLOAD_STAGE_SCHEMA_VERSION,
                    "reference": descriptor.reference.to_dict(),
                    "descriptor_fingerprint": fingerprint,
                },
            )
            temporary_stage = self._open_download_stage(
                descriptor,
                temporary,
                temporary_marker,
                temporary_payload,
                temporary_state,
                fingerprint,
            )
            self._assert_managed_path(self._locks_path)
            with ArtifactOperationLease(
                self._locks_path,
                descriptor.reference.lease_key(),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                if self._managed_path_exists(operation):
                    canonical_stage = self._open_download_stage(
                        descriptor,
                        operation,
                        marker,
                        payload,
                        state,
                        fingerprint,
                    )
                    self._discard_temporary_download_stage(temporary_stage)
                    return canonical_stage
                os.rename(temporary, operation)
            return self._open_download_stage(
                descriptor,
                operation,
                marker,
                payload,
                state,
                fingerprint,
            )
        except ArtifactError:
            if temporary is not None and temporary_identity is not None:
                self._remove_incomplete_download_stage(
                    temporary,
                    temporary_identity,
                    temporary / marker.name,
                    temporary / payload.name,
                    temporary / state.name,
                )
            raise
        except OSError as error:
            if temporary is not None and temporary_identity is not None:
                self._remove_incomplete_download_stage(
                    temporary,
                    temporary_identity,
                    temporary / marker.name,
                    temporary / payload.name,
                    temporary / state.name,
                )
            raise ArtifactStateError(
                "failed to initialize download staging operation"
            ) from error

    def _finalize_download_stage(
        self,
        descriptor: ArtifactDescriptor,
        stage: _ManagedDownloadStage,
    ) -> ArtifactRef:
        """Verify and promote a complete staged download without copying it."""

        if type(descriptor) is not ArtifactDescriptor:
            raise TypeError("descriptor must be an ArtifactDescriptor")
        if type(stage) is not _ManagedDownloadStage:
            raise TypeError("stage must be a _ManagedDownloadStage")
        destination = self.artifact_path(descriptor.reference)
        try:
            self._assert_managed_path(self._locks_path)
            # Keep finalization in the same writer domain as install and deletion.
            with ArtifactOperationLease(
                self._locks_path,
                _LIFECYCLE_LEASE_KEY,
                LeaseMode.EXCLUSIVE,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                with ArtifactOperationLease(
                    self._locks_path,
                    descriptor.reference.lease_key(),
                    LeaseMode.EXCLUSIVE,
                    timeout_seconds=self._lease_timeout_seconds,
                ):
                    checked = self._validate_download_stage_handle(descriptor, stage)
                    manifest_path = checked.payload / "manifest.json"
                    if self._managed_path_exists(manifest_path):
                        existing = self._read_manifest(checked.payload)
                        if existing != descriptor:
                            raise ArtifactConflictError(
                                "download staging payload has a different manifest"
                            )
                    self._verify_payload(
                        checked.payload,
                        descriptor.files,
                        allowed_files=frozenset({"manifest.json"}),
                    )
                    if self._managed_path_exists(destination):
                        self._verify_existing_destination(destination, descriptor)
                        self._discard_download_stage(checked)
                        return descriptor.reference
                    atomic_write_json(
                        manifest_path,
                        {
                            "schema_version": _MANIFEST_SCHEMA_VERSION,
                            "descriptor": descriptor.to_dict(),
                        },
                    )
                    # Recheck after the crash-recovery manifest write and before rename.
                    checked = self._validate_download_stage_handle(descriptor, checked)
                    self._verify_payload(
                        checked.payload,
                        descriptor.files,
                        allowed_files=frozenset({"manifest.json"}),
                    )
                    self._ensure_final_parent(destination.parent)
                    if self._managed_path_exists(destination):
                        raise ArtifactConflictError(
                            "immutable artifact destination already exists"
                        )
                    self._assert_managed_path(checked.payload)
                    self._assert_managed_path(destination.parent)
                    self._promote(checked.payload, destination)
                    self._assert_managed_path(destination)
                    self._remove_finalized_download_stage(checked)
            return descriptor.reference
        except ArtifactError:
            raise
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire artifact writer leases"
            ) from error
        except OSError as error:
            raise ArtifactStateError("download finalization I/O failed") from error

    def _discard_download_stage(self, stage: _ManagedDownloadStage) -> None:
        """Remove one exact marked operation without touching other staging data."""

        if type(stage) is not _ManagedDownloadStage:
            raise TypeError("stage must be a _ManagedDownloadStage")
        try:
            self._validate_download_stage_handle_by_marker(stage)
        except ArtifactPathError as full_stage_error:
            try:
                self._validate_download_stage_handle_by_marker_after_promotion(stage)
            except ArtifactPathError:
                raise full_stage_error
        try:
            shutil.rmtree(self._retire_download_stage_operation(stage.operation))
        except FileNotFoundError:
            raise ArtifactPathError(
                "download staging operation disappeared during discard"
            )
        except OSError as error:
            raise ArtifactStateError(
                "failed to discard download staging operation"
            ) from error

    def _discard_temporary_download_stage(self, stage: _ManagedDownloadStage) -> None:
        """Remove one verified losing temporary publication candidate."""

        if (
            stage.operation.parent != self._staging_path
            or not stage.operation.name.startswith(".download-")
            or stage.payload_identity is None
            or stage.state_identity is None
        ):
            raise ArtifactPathError("download staging temporary has unexpected paths")
        operation_identity = self._download_stage_node_identity(
            stage.operation,
            directory=True,
        )
        marker_identity = self._download_stage_node_identity(
            stage.marker,
            directory=False,
        )
        payload_identity = self._download_stage_node_identity(
            stage.payload,
            directory=True,
        )
        state_identity = self._download_stage_node_identity(stage.state, directory=True)
        if (
            operation_identity != stage.operation_identity
            or marker_identity != stage.marker_identity
            or payload_identity != stage.payload_identity
            or state_identity != stage.state_identity
        ):
            raise ArtifactPathError("download staging temporary identity changed")
        self._validate_download_stage_layout(
            stage.operation,
            stage.marker,
            stage.payload,
            stage.state,
        )
        self._read_download_stage_marker(
            stage.marker,
            stage.reference,
            stage.descriptor_fingerprint,
        )
        try:
            shutil.rmtree(stage.operation)
        except OSError as error:
            raise ArtifactStateError(
                "failed to discard download staging temporary"
            ) from error

    def _download_stage_paths(
        self,
        descriptor: ArtifactDescriptor,
    ) -> tuple[Path, Path, Path, Path, str]:
        fingerprint = hashlib.sha256(
            b"managed-download-stage-v1\0"
            + json.dumps(
                descriptor.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        operation = self._staging_path / f"download-{fingerprint}"
        return (
            operation,
            operation / "download-stage.json",
            operation / "payload",
            operation / "state",
            fingerprint,
        )

    def _open_download_stage(
        self,
        descriptor: ArtifactDescriptor,
        operation: Path,
        marker: Path,
        payload: Path,
        state: Path,
        fingerprint: str,
    ) -> _ManagedDownloadStage:
        operation_identity = self._download_stage_node_identity(
            operation,
            directory=True,
        )
        marker_identity = self._download_stage_node_identity(marker, directory=False)
        state_identity = (
            self._download_stage_node_identity(state, directory=True)
            if self._managed_path_exists(state)
            else None
        )
        payload_identity = (
            self._download_stage_node_identity(payload, directory=True)
            if self._managed_path_exists(payload)
            else None
        )
        self._read_download_stage_marker(marker, descriptor, fingerprint)
        self._validate_download_stage_layout(
            operation,
            marker,
            payload,
            state,
            payload_present=payload_identity is not None,
            state_present=state_identity is not None,
        )
        if payload_identity is not None and state_identity is None:
            raise ArtifactPathError("download staging state is missing")
        if state_identity is not None:
            self._validate_download_stage_state(state)
        return _ManagedDownloadStage(
            reference=descriptor.reference,
            descriptor_fingerprint=fingerprint,
            operation=operation,
            marker=marker,
            payload=payload,
            state=state,
            operation_identity=operation_identity,
            marker_identity=marker_identity,
            payload_identity=payload_identity,
            state_identity=state_identity,
        )

    def _validate_download_stage_handle(
        self,
        descriptor: ArtifactDescriptor,
        stage: _ManagedDownloadStage,
    ) -> _ManagedDownloadStage:
        operation, marker, payload, state, fingerprint = self._download_stage_paths(
            descriptor,
        )
        if (
            stage.reference != descriptor.reference
            or stage.descriptor_fingerprint != fingerprint
            or stage.operation != operation
            or stage.marker != marker
            or stage.payload != payload
            or stage.state != state
        ):
            raise ArtifactPathError("download staging handle does not match descriptor")
        opened = self._open_download_stage(
            descriptor,
            operation,
            marker,
            payload,
            state,
            fingerprint,
        )
        if (
            stage.payload_identity is None
            or opened.payload_identity is None
            or stage.state_identity is None
            or opened.state_identity is None
        ):
            raise ArtifactPathError("download staging payload is missing")
        if (
            opened.operation_identity != stage.operation_identity
            or opened.marker_identity != stage.marker_identity
            or opened.payload_identity != stage.payload_identity
            or opened.state_identity != stage.state_identity
        ):
            raise ArtifactPathError("download staging operation identity changed")
        return opened

    def _validate_download_stage_handle_by_marker(
        self,
        stage: _ManagedDownloadStage,
    ) -> None:
        self._validate_download_stage_handle_paths(stage)
        operation = stage.operation
        operation_identity = self._download_stage_node_identity(
            operation, directory=True
        )
        marker_identity = self._download_stage_node_identity(
            stage.marker, directory=False
        )
        payload_identity = self._download_stage_node_identity(
            stage.payload, directory=True
        )
        state_identity = self._download_stage_node_identity(stage.state, directory=True)
        if (
            operation_identity != stage.operation_identity
            or marker_identity != stage.marker_identity
            or payload_identity != stage.payload_identity
            or state_identity != stage.state_identity
        ):
            raise ArtifactPathError("download staging operation identity changed")
        self._validate_download_stage_layout(
            operation,
            stage.marker,
            stage.payload,
            stage.state,
        )
        self._read_download_stage_marker(
            stage.marker,
            stage.reference,
            stage.descriptor_fingerprint,
        )

    def _download_stage_node_identity(
        self,
        path: Path,
        *,
        directory: bool,
    ) -> _NodeIdentity:
        self._assert_managed_path(
            path,
            target_must_be_directory=directory,
        )
        try:
            info = path.stat(follow_symlinks=False)
        except OSError as error:
            raise ArtifactPathError(
                "failed to inspect download staging path"
            ) from error
        if stat.S_ISLNK(info.st_mode) or (
            not stat.S_ISDIR(info.st_mode)
            if directory
            else not stat.S_ISREG(info.st_mode)
        ):
            raise ArtifactPathError("download staging path has an invalid node type")
        return _node_identity(info)

    def _validate_download_stage_layout(
        self,
        operation: Path,
        marker: Path,
        payload: Path,
        state: Path,
        *,
        payload_present: bool = True,
        state_present: bool = True,
    ) -> None:
        try:
            entries = {entry.name: Path(entry.path) for entry in os.scandir(operation)}
        except OSError as error:
            raise ArtifactPathError(
                "failed to scan download staging operation"
            ) from error
        expected = {
            marker.name: marker,
        }
        if payload_present:
            expected[payload.name] = payload
        if state_present:
            expected[state.name] = state
        if entries != expected:
            raise ArtifactPathError("download staging operation has unexpected entries")

    def _validate_download_stage_handle_paths(
        self,
        stage: _ManagedDownloadStage,
    ) -> None:
        operation = stage.operation
        if operation.parent != self._staging_path:
            raise ArtifactPathError("download staging operation escapes staging root")
        if (
            stage.marker != operation / "download-stage.json"
            or stage.payload != operation / "payload"
            or stage.state != operation / "state"
            or operation.name != f"download-{stage.descriptor_fingerprint}"
        ):
            raise ArtifactPathError("download staging handle has unexpected paths")
        if _LOWERCASE_SHA256.fullmatch(stage.descriptor_fingerprint) is None:
            raise ArtifactPathError("download staging handle has invalid fingerprint")

    def _remove_incomplete_download_stage(
        self,
        operation: Path,
        operation_identity: _NodeIdentity,
        marker: Path,
        payload: Path,
        state: Path,
    ) -> None:
        """Remove one never-published temp download-stage directory this call created.

        PR-1165 review (P2, part ii): covers BOTH failure windows inside
        ``_download_stage_for``'s creation path -- a PRE-marker failure
        (``payload``/``state`` ``mkdir``, or the marker write itself,
        raised, so ``marker`` may not exist yet) AND a POST-marker failure
        (the marker was written, but the final ``os.rename`` into the
        canonical ``download-<fingerprint>/`` operation path raised).
        Before this fix, the POST-marker case was a real leak: the early
        ``if self._managed_path_exists(marker): return`` unconditionally
        left the fully-populated temp directory behind forever, since
        nothing -- including reconcile()'s staging GC, which only
        recognizes the CANONICAL ``download-<fingerprint>/`` name, never
        the ephemeral ``.download-<random>/`` one -- ever looks up a stage
        by its temp name again.

        Safe to remove in both cases because ``operation`` is confirmed,
        via ``operation_identity``, to still be the EXACT directory this
        call itself created moments ago via ``tempfile.mkdtemp`` -- a name
        unique enough that no other caller could ever have discovered or
        started relying on it (only the canonical path, which this
        directory was never successfully renamed to, is ever looked up by
        ``_download_stage_paths``). ``payload``/``state`` are also always
        still EMPTY at this point in practice: this helper is only ever
        reached before ``_fetch_artifact``/``_preverify_artifact`` write
        anything into a stage ``_download_stage_for`` hands back, so the
        non-empty guard below is defense-in-depth, not an expected path.

        Args:
            operation: The temp operation directory this call created.
            operation_identity: Its identity snapshot from immediately
                after creation, to detect any change below.
            marker: The temp marker path (``operation / marker.name``).
            payload: The temp payload path (``operation / payload.name``).
            state: The temp state path (``operation / state.name``).
        """

        try:
            if (
                self._download_stage_node_identity(
                    operation,
                    directory=True,
                )
                != operation_identity
            ):
                return
            entries = {entry.name: Path(entry.path) for entry in os.scandir(operation)}
            if set(entries) - {marker.name, payload.name, state.name}:
                return
            for path in (payload, state):
                if path.name not in entries:
                    continue
                self._download_stage_node_identity(path, directory=True)
                if tuple(os.scandir(path)):
                    return
            if marker.name in entries:
                self._download_stage_node_identity(marker, directory=False)
            for path in (state, payload):
                if path.name in entries:
                    path.rmdir()
            if marker.name in entries:
                marker.unlink()
            operation.rmdir()
        except (ArtifactError, OSError):
            return

    def _read_download_stage_marker(
        self,
        marker: Path,
        reference: ArtifactDescriptor | ArtifactRef,
        fingerprint: str,
    ) -> None:
        expected_reference = (
            reference.reference if type(reference) is ArtifactDescriptor else reference
        )
        try:
            with marker.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
            _require_exact_keys(raw, _DOWNLOAD_STAGE_KEYS, "download stage")
            assert isinstance(raw, Mapping)
            if (
                type(raw["schema_version"]) is not int
                or raw["schema_version"] != _DOWNLOAD_STAGE_SCHEMA_VERSION
                or _parse_reference(raw["reference"], "download stage.reference")
                != expected_reference
                or type(raw["descriptor_fingerprint"]) is not str
                or raw["descriptor_fingerprint"] != fingerprint
            ):
                raise ArtifactPathError(
                    "download stage marker does not match descriptor"
                )
        except ArtifactPathError:
            raise
        except (
            ArtifactDescriptorError,
            FileNotFoundError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise ArtifactPathError("download stage marker is invalid") from error

    def _validate_download_stage_state(self, state: Path) -> None:
        def scan(directory: Path) -> None:
            try:
                entries = tuple(os.scandir(directory))
            except OSError as error:
                raise ArtifactPathError(
                    "failed to scan download staging state"
                ) from error
            for entry in entries:
                path = Path(entry.path)
                try:
                    info = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise ArtifactPathError(
                        "failed to inspect download staging state"
                    ) from error
                if stat.S_ISLNK(info.st_mode):
                    raise ArtifactPathError("download staging state contains a symlink")
                if stat.S_ISDIR(info.st_mode):
                    scan(path)
                elif stat.S_ISREG(info.st_mode) and path.suffix == ".json":
                    try:
                        with path.open("r", encoding="utf-8") as handle:
                            json.load(
                                handle, object_pairs_hook=_reject_duplicate_json_keys
                            )
                    except (
                        OSError,
                        UnicodeError,
                        json.JSONDecodeError,
                        ValueError,
                    ) as error:
                        raise ArtifactPathError(
                            "download staging state contains invalid JSON"
                        ) from error
                else:
                    raise ArtifactPathError("download staging state has invalid entry")

        scan(state)

    def _remove_finalized_download_stage(self, stage: _ManagedDownloadStage) -> None:
        self._validate_download_stage_handle_by_marker_after_promotion(stage)
        try:
            shutil.rmtree(self._retire_download_stage_operation(stage.operation))
        except OSError as error:
            raise ArtifactStateError(
                "failed to clean finalized download staging"
            ) from error

    def _retire_download_stage_operation(self, operation: Path) -> Path:
        """Atomically move one validated canonical operation off its stable path."""

        retired = Path(
            tempfile.mkdtemp(prefix=".download-retired-", dir=self._staging_path)
        )
        try:
            retired.rmdir()
            os.rename(operation, retired)
        except OSError:
            try:
                retired.rmdir()
            except OSError:
                pass
            raise
        return retired

    def _validate_download_stage_handle_by_marker_after_promotion(
        self,
        stage: _ManagedDownloadStage,
    ) -> None:
        self._validate_download_stage_handle_paths(stage)
        if self._managed_path_exists(stage.payload):
            raise ArtifactPathError("download staging payload was not promoted")
        operation_identity = self._download_stage_node_identity(
            stage.operation,
            directory=True,
        )
        marker_identity = self._download_stage_node_identity(
            stage.marker,
            directory=False,
        )
        state_identity = (
            self._download_stage_node_identity(stage.state, directory=True)
            if self._managed_path_exists(stage.state)
            else None
        )
        if (
            operation_identity != stage.operation_identity
            or marker_identity != stage.marker_identity
            or state_identity != stage.state_identity
        ):
            raise ArtifactPathError("download staging operation identity changed")
        try:
            entries = {entry.name for entry in os.scandir(stage.operation)}
        except OSError as error:
            raise ArtifactPathError(
                "failed to scan finalized download staging"
            ) from error
        expected_entries = {stage.marker.name}
        if state_identity is not None:
            expected_entries.add(stage.state.name)
        if entries != expected_entries:
            raise ArtifactPathError("finalized download staging has unexpected entries")
        self._read_download_stage_marker(
            stage.marker,
            stage.reference,
            stage.descriptor_fingerprint,
        )

    # -- end TASK-1694 download-stage seam -----------------------------------

    def _create_install_staging(self) -> tuple[Path, ArtifactOperationLease]:
        """Create one leased operation-owned install staging directory."""

        self._assert_managed_path(self._staging_path)
        # Name before create so reconcile() cannot observe an unleased stage.
        staging_name = f"{_INSTALL_STAGING_PREFIX}{uuid.uuid4().hex}"
        staging = self._staging_path / staging_name
        self._assert_managed_path(self._locks_path)
        # Use the leases submodule so tests replacing the public writer-lease
        # name do not intercept this orphan-detection lease.
        staging_lease = _leases.ArtifactOperationLease(
            self._locks_path,
            _install_staging_lease_key(staging_name),
            LeaseMode.EXCLUSIVE,
            timeout_seconds=self._lease_timeout_seconds,
        )
        staging_lease.acquire()
        try:
            os.mkdir(staging, 0o700)
        except OSError as error:
            primary_error = ArtifactStateError(
                "failed to create artifact install staging directory"
            )
            try:
                try:
                    shutil.rmtree(staging)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    primary_error.add_note(
                        f"operation staging cleanup failed: {cleanup_error!r}"
                    )
            finally:
                try:
                    staging_lease.release()
                except BaseException as release_error:
                    primary_error.add_note(
                        f"staging operation lease release failed: {release_error!r}"
                    )
            raise primary_error from error
        return staging, staging_lease

    def _commit_verified_staging(
        self,
        descriptor: ArtifactDescriptor,
        staging: Path,
        *,
        cancelled: Callable[[], bool],
        on_finalizing: Callable[[], None] | None = None,
    ) -> bool:
        """Verify and converge or promote one operation-owned staging tree."""

        destination = self.artifact_path(descriptor.reference)
        self._assert_managed_path(self._locks_path)

        # ponytail: one lifecycle writer lock is enough until measured install
        # throughput justifies per-artifact writer coordination.
        with ArtifactOperationLease(
            self._locks_path,
            _LIFECYCLE_LEASE_KEY,
            LeaseMode.EXCLUSIVE,
            timeout_seconds=self._lease_timeout_seconds,
        ):
            with ArtifactOperationLease(
                self._locks_path,
                descriptor.reference.lease_key(),
                LeaseMode.EXCLUSIVE,
                timeout_seconds=self._lease_timeout_seconds,
            ):
                _raise_if_install_cancelled(cancelled)
                if self._managed_path_exists(destination):
                    self._verify_existing_destination(destination, descriptor)
                    return True
                self._verify_payload(
                    staging,
                    descriptor.files,
                    cancelled=cancelled,
                )
                _raise_if_install_cancelled(cancelled)
                atomic_write_json(
                    staging / "manifest.json",
                    {
                        "schema_version": _MANIFEST_SCHEMA_VERSION,
                        "descriptor": descriptor.to_dict(),
                    },
                )
                self._ensure_final_parent(destination.parent)
                if self._managed_path_exists(destination):
                    raise ArtifactConflictError(
                        "immutable artifact destination already exists"
                    )
                self._assert_managed_path(staging)
                self._assert_managed_path(destination.parent)
                _raise_if_install_cancelled(cancelled)
                if on_finalizing is not None:
                    on_finalizing()
                self._promote(staging, destination)
                self._assert_managed_path(destination)
                return False

    def import_local_gguf(
        self,
        source_file: Path,
        *,
        cancelled: Callable[[], bool] = _never_cancelled,
        progress: Callable[
            [LocalGGUFImportProgress], None
        ] = _ignore_local_import_progress,
    ) -> LocalGGUFImportResult:
        """Copy one local GGUF into the immutable managed artifact store.

        Args:
            source_file: User-owned regular GGUF selected for import.
            cancelled: Optional caller-thread-safe cancellation probe.
            progress: Synchronous path-private progress callback. Exceptions raised
                by the callback propagate to the caller.

        Returns:
            The exact managed reference and whether it already existed.

        Raises:
            TypeError: An argument has the wrong public API type.
            ArtifactLeaseTimeoutError: A managed writer lease remains busy.
            ArtifactError: Managed staging, verification, or promotion fails.
            GGUFError: The source path or staged GGUF structure is invalid.
        """

        if not isinstance(source_file, Path):
            raise TypeError("source_file must be a Path")
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        if not callable(progress):
            raise TypeError("progress must be callable")
        _raise_if_install_cancelled(cancelled)

        staging: Path | None = None
        staging_lease: ArtifactOperationLease | None = None
        try:
            staging, staging_lease = self._create_install_staging()
            self._assert_managed_path(staging)
            target = staging / "model.gguf"
            with open_local_gguf(source_file) as opened, open(target, "xb") as output:
                digest = hashlib.sha256()
                copied = 0
                last_progress = 0
                progress(
                    LocalGGUFImportProgress(
                        "copy",
                        "model.gguf",
                        copied,
                        opened.identity.size_bytes,
                    )
                )
                while chunk := opened.handle.read(_LOCAL_COPY_CHUNK_BYTES):
                    _raise_if_install_cancelled(cancelled)
                    output.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                    if copied - last_progress >= _LOCAL_COPY_PROGRESS_BYTES:
                        progress(
                            LocalGGUFImportProgress(
                                "copy",
                                "model.gguf",
                                copied,
                                opened.identity.size_bytes,
                            )
                        )
                        last_progress = copied
                if copied != last_progress:
                    progress(
                        LocalGGUFImportProgress(
                            "copy",
                            "model.gguf",
                            copied,
                            opened.identity.size_bytes,
                        )
                    )
                output.flush()
                os.fsync(output.fileno())
                opened.recheck()

            _raise_if_install_cancelled(cancelled)
            progress(LocalGGUFImportProgress("inspect", None, 0, copied))
            with target.open("rb", buffering=0) as staged:
                metadata = inspect_gguf_structure(staged, file_size=copied)
            descriptor = _local_gguf_descriptor(
                metadata,
                digest=digest.hexdigest(),
                size_bytes=copied,
            )
            progress(LocalGGUFImportProgress("verify", "model.gguf", 0, copied))
            already_installed = self._commit_verified_staging(
                descriptor,
                staging,
                cancelled=cancelled,
                on_finalizing=lambda: progress(
                    LocalGGUFImportProgress("finalize", None, copied, copied)
                ),
            )
            if not already_installed:
                staging = None
            return LocalGGUFImportResult(descriptor.reference, already_installed)
        except ArtifactError:
            raise
        except ArtifactLeaseTimeoutError:
            raise
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire artifact writer leases"
            ) from error
        except OSError as error:
            raise ArtifactStateError("artifact import I/O failed") from error
        finally:
            primary_error = sys.exception()
            try:
                if staging is not None:
                    try:
                        shutil.rmtree(staging)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        if primary_error is None:
                            raise ArtifactStateError(
                                "failed to clean operation-owned artifact staging"
                            ) from cleanup_error
                        primary_error.add_note(
                            f"operation staging cleanup failed: {cleanup_error!r}"
                        )
            finally:
                if staging_lease is not None:
                    try:
                        staging_lease.release()
                    except BaseException as release_error:
                        if primary_error is None:
                            raise ArtifactStateError(
                                "failed to release staging operation lease"
                            ) from release_error
                        primary_error.add_note(
                            f"staging operation lease release failed: {release_error!r}"
                        )

    def install(
        self,
        descriptor: ArtifactDescriptor,
        source_directory: Path,
        *,
        consume_source: bool = False,
        declared_files_only: bool = False,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> ArtifactRef:
        """Verify and promote one local source directory immutably.

        Args:
            descriptor: Strict descriptor for the artifact being installed.
            source_directory: Existing local directory containing its payload.
            consume_source: Move files instead of copying when source lies inside
                the service root. Requires source inside the root; raises
                ArtifactPathError otherwise. On EXDEV (cross-device link), falls
                back to copy+delete. Defaults to False (today's copy semantics).
            declared_files_only: Validate and snapshot only descriptor-declared
                paths, allowing unrelated user-owned entries to remain ignored.
            cancelled: Optional caller-thread-safe cancellation probe, polled
                between fixed-size local-copy chunks and install phases.

        Returns:
            The installed artifact's immutable reference.

        Raises:
            TypeError: An argument has the wrong public API type.
            ArtifactPathError: The source or managed destination is unsafe, or
                consume_source with a source outside the service root.
            ArtifactIntegrityError: A declared file fails size or digest checks.
            ArtifactConflictError: The immutable destination conflicts.
            ArtifactStateError: Artifact storage or lease operations fail.
        """

        if type(descriptor) is not ArtifactDescriptor:
            raise TypeError("descriptor must be an ArtifactDescriptor")
        if not isinstance(source_directory, Path):
            raise TypeError("source_directory must be a Path")
        if type(declared_files_only) is not bool:
            raise TypeError("declared_files_only must be a bool")
        if not callable(cancelled):
            raise TypeError("cancelled must be callable")
        _raise_if_install_cancelled(cancelled)
        try:
            source_directory = validate_path_simple(
                source_directory,
                probe_existing=False,
            )
        except ValueError as error:
            raise ArtifactPathError("source directory path is invalid") from error
        source_directory = Path(os.path.abspath(source_directory))
        source_snapshot = self._validate_payload_tree(
            source_directory,
            descriptor.files,
            declared_files_only=declared_files_only,
        )
        _raise_if_install_cancelled(cancelled)

        staging: Path | None = None
        staging_lease: ArtifactOperationLease | None = None
        try:
            staging, staging_lease = self._create_install_staging()
            self._assert_managed_path(staging)
            self._copy_payload(
                descriptor,
                source_directory,
                staging,
                consume_source=consume_source,
                cancelled=cancelled,
            )
            _raise_if_install_cancelled(cancelled)
            # When consume_source=True, files are intentionally moved from source.
            # Skip the unchanged-tree check in that case; it's expected to change.
            if not consume_source:
                if (
                    self._validate_payload_tree(
                        source_directory,
                        descriptor.files,
                        declared_files_only=declared_files_only,
                    )
                    != source_snapshot
                ):
                    raise ArtifactPathError("source tree changed during artifact copy")
            _raise_if_install_cancelled(cancelled)
            already_installed = self._commit_verified_staging(
                descriptor,
                staging,
                cancelled=cancelled,
            )
            if not already_installed:
                staging = None
            return descriptor.reference
        except ArtifactError:
            raise
        except ArtifactLeaseError as error:
            raise ArtifactStateError(
                "failed to acquire artifact writer leases"
            ) from error
        except OSError as error:
            raise ArtifactStateError("artifact installation I/O failed") from error
        finally:
            primary_error = sys.exception()
            try:
                if staging is not None:
                    try:
                        shutil.rmtree(staging)
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        if primary_error is None:
                            raise ArtifactStateError(
                                "failed to clean operation-owned artifact staging"
                            ) from cleanup_error
                        primary_error.add_note(
                            f"operation staging cleanup failed: {cleanup_error!r}"
                        )
            finally:
                if staging_lease is not None:
                    try:
                        staging_lease.release()
                    except BaseException as release_error:
                        if primary_error is None:
                            raise ArtifactStateError(
                                "failed to release staging operation lease"
                            ) from release_error
                        primary_error.add_note(
                            f"staging operation lease release failed: {release_error!r}"
                        )

    def _resolve_closure(
        self,
        root_reference: ArtifactRef,
    ) -> tuple[ArtifactRef, ...]:
        references_by_id = {root_reference.artifact_id: root_reference}
        visiting: set[ArtifactRef] = set()
        visited: set[ArtifactRef] = set()

        def visit(reference: ArtifactRef, role: ArtifactRole) -> None:
            if reference in visiting:
                raise ArtifactDependencyError(
                    "dependency cycle detected in exact artifact closure"
                )
            if reference in visited:
                return
            descriptor = self._load_installed_descriptor(reference, role)
            visiting.add(reference)
            try:
                for dependency in descriptor.dependencies:
                    if dependency == root_reference:
                        raise ArtifactDependencyError(
                            "root artifact cannot appear as a dependency"
                        )
                    existing = references_by_id.get(dependency.artifact_id)
                    if existing is not None and existing != dependency:
                        raise ArtifactDependencyError(
                            "dependency closure contains conflicting exact "
                            f"references for {dependency.artifact_id}"
                        )
                    references_by_id[dependency.artifact_id] = dependency
                    visit(dependency, ArtifactRole.DEPENDENCY)
            finally:
                visiting.remove(reference)
            visited.add(reference)

        visit(root_reference, ArtifactRole.ROOT)
        return tuple(sorted(visited))

    def _load_installed_descriptor(
        self,
        reference: ArtifactRef,
        expected_role: ArtifactRole,
    ) -> ArtifactDescriptor:
        try:
            descriptor = self._read_manifest(self.artifact_path(reference))
        except ArtifactError as error:
            raise ArtifactDependencyError(
                "missing or invalid exact installed artifact "
                f"{reference.artifact_id}@{reference.revision}/{reference.variant}"
            ) from error
        if descriptor.reference != reference:
            raise ArtifactDependencyError(
                "installed manifest reference does not match its exact directory"
            )
        if descriptor.role is not expected_role:
            raise ArtifactDependencyError(
                f"installed artifact {reference.artifact_id} has role "
                f"{descriptor.role.value}, expected {expected_role.value}"
            )
        return descriptor

    def _verify_installed(
        self,
        reference: ArtifactRef,
        expected_role: ArtifactRole,
    ) -> None:
        descriptor = self._load_installed_descriptor(reference, expected_role)
        try:
            self._verify_payload(
                self.artifact_path(reference),
                descriptor.files,
                allowed_files=frozenset({"manifest.json"}),
            )
        except ArtifactPathError as error:
            raise ArtifactIntegrityError(
                "installed artifact payload tree is invalid"
            ) from error

    def _read_readiness(self, reference: ArtifactRef) -> _ReadinessRecord:
        path = self.readiness_path(reference)
        try:
            self._assert_managed_path(
                path,
                allow_missing=True,
                target_must_be_directory=False,
            )
            mode = path.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ArtifactStateError(
                    "readiness record must be a regular non-symlink file"
                )
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
            _require_exact_keys(raw, _READINESS_KEYS, "readiness")
            assert isinstance(raw, Mapping)
            schema_version = raw["schema_version"]
            if (
                type(schema_version) is not int
                or schema_version != _READINESS_SCHEMA_VERSION
            ):
                raise ArtifactDescriptorParseError(
                    "readiness schema_version must be the integer 1"
                )
            root = _parse_reference(raw["root"], "readiness.root")
            if root != reference:
                raise ArtifactDescriptorParseError(
                    "readiness root does not match its state path"
                )
            closure = tuple(
                _parse_reference(item, f"readiness.closure[{index}]")
                for index, item in enumerate(
                    _require_list(raw["closure"], "readiness.closure")
                )
            )
            if not closure:
                raise ArtifactDescriptorParseError(
                    "readiness closure must not be empty"
                )
            if closure != tuple(sorted(set(closure))):
                raise ArtifactDescriptorParseError(
                    "readiness closure must be sorted and unique"
                )
            if root not in closure:
                raise ArtifactDescriptorParseError(
                    "readiness closure must contain its root"
                )
            fingerprint = _require_string(
                raw["closure_fingerprint"],
                "readiness.closure_fingerprint",
            )
            if _LOWERCASE_SHA256.fullmatch(
                fingerprint
            ) is None or fingerprint != closure_fingerprint(root, closure):
                raise ArtifactDescriptorParseError(
                    "readiness closure fingerprint does not match"
                )
            return _ReadinessRecord(
                root=root,
                closure=closure,
                closure_fingerprint=fingerprint,
            )
        except FileNotFoundError as error:
            raise ArtifactNotReadyError("artifact has no readiness record") from error
        except ArtifactNotReadyError:
            raise
        except ArtifactStateError:
            raise
        except (
            ArtifactDescriptorError,
            ArtifactPathError,
            NotADirectoryError,
            OSError,
            RecursionError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise ArtifactStateError("artifact readiness record is invalid") from error

    def _read_active(self, artifact_id: str) -> ArtifactRef:
        path = self.active_path(artifact_id)
        try:
            self._assert_managed_path(
                path,
                allow_missing=True,
                target_must_be_directory=False,
            )
            mode = path.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ArtifactStateError(
                    "active selector must be a regular non-symlink file"
                )
            with path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
            _require_exact_keys(raw, _ACTIVE_KEYS, "active")
            assert isinstance(raw, Mapping)
            schema_version = raw["schema_version"]
            if (
                type(schema_version) is not int
                or schema_version != _ACTIVE_SCHEMA_VERSION
            ):
                raise ArtifactDescriptorParseError(
                    "active schema_version must be the integer 1"
                )
            root = _parse_reference(raw["root"], "active.root")
            if root.artifact_id != artifact_id:
                raise ArtifactDescriptorParseError(
                    "active root does not match its selector path"
                )
            return root
        except FileNotFoundError as error:
            raise ArtifactNotReadyError(
                "artifact identity has no active selector"
            ) from error
        except ArtifactNotReadyError:
            raise
        except ArtifactStateError:
            raise
        except (
            ArtifactDescriptorError,
            ArtifactPathError,
            NotADirectoryError,
            OSError,
            RecursionError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise ArtifactStateError("artifact active selector is invalid") from error

    def _remove_active_if_selects(self, reference: ArtifactRef) -> None:
        path = self.active_path(reference.artifact_id)
        if not self._state_path_exists(path):
            return
        try:
            selected = self._read_active(reference.artifact_id)
        except ArtifactStateError:
            selected = reference
        if selected == reference:
            self._remove_state_path(
                path,
                "failed to remove invalid active selector",
            )

    def _try_read_readiness(
        self,
        reference: ArtifactRef,
    ) -> _ReadinessRecord | None:
        try:
            return self._read_readiness(reference)
        except (ArtifactNotReadyError, ArtifactStateError):
            return None

    def _remove_readiness(self, reference: ArtifactRef) -> None:
        path = self.readiness_path(reference)
        try:
            self._assert_managed_path(path.parent, allow_missing=True)
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            raise ArtifactStateError(
                "failed to remove invalid readiness record"
            ) from error

    def _write_readiness(self, record: _ReadinessRecord) -> None:
        path = self.readiness_path(record.root)
        self._ensure_state_parent(self._ready_path, path.parent)
        self._assert_managed_path(path.parent)
        try:
            atomic_write_json(path, record.to_dict())
        except OSError as error:
            raise ArtifactStateError("failed to write artifact readiness") from error

    def _ensure_state_parent(self, base: Path, parent: Path) -> None:
        relative = parent.relative_to(base)
        current = base
        for component in relative.parts:
            current = current / component
            self._assert_managed_path(current, allow_missing=True)
            self._ensure_owned_directory(current)
            self._assert_managed_path(current)

    def _ensure_owned_directory(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            mode = path.stat(follow_symlinks=False).st_mode
        except OSError as error:
            raise ArtifactPathError(
                f"failed to create managed directory {path.name}"
            ) from error
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise ArtifactPathError(f"managed path {path.name} must be a directory")

    def _assert_managed_path(
        self,
        path: Path,
        *,
        allow_missing: bool = False,
        target_must_be_directory: bool = True,
    ) -> None:
        # Portable stdlib checks cannot pin a directory against a coordinated
        # swap-and-restore. The service sole-writer invariant excludes that
        # managed-store race; identity snapshots reject observable changes.
        lexical_path = Path(os.path.abspath(path))
        try:
            relative = lexical_path.relative_to(self._root)
        except ValueError as error:
            raise ArtifactPathError(
                "managed path escapes the artifact store root"
            ) from error

        current = self._root
        chain = (self._root,) + tuple(
            self._root.joinpath(*relative.parts[:index])
            for index in range(1, len(relative.parts) + 1)
        )
        for index, current in enumerate(chain):
            try:
                info = current.stat(follow_symlinks=False)
            except FileNotFoundError as error:
                if allow_missing:
                    return
                raise ArtifactPathError("managed path component is missing") from error
            except (OSError, ValueError) as error:
                raise ArtifactPathError("failed to inspect managed path") from error
            is_target = index == len(chain) - 1
            if stat.S_ISLNK(info.st_mode):
                raise ArtifactPathError("managed path component is a symlink")
            if not stat.S_ISDIR(info.st_mode) and (
                not is_target or target_must_be_directory
            ):
                raise ArtifactPathError("managed path component is not a directory")
            expected = self._managed_root_identities.get(current)
            if expected is not None and _node_identity(info) != expected:
                raise ArtifactPathError("managed store root identity changed")
            try:
                resolved = current.resolve(strict=True)
                resolved.relative_to(self._root)
            except (OSError, RuntimeError, ValueError) as error:
                raise ArtifactPathError(
                    "managed path resolves outside the artifact store root"
                ) from error
            if resolved != current:
                raise ArtifactPathError(
                    "managed path contains a redirected path component"
                )

    def _managed_path_exists(self, path: Path) -> bool:
        self._assert_managed_path(
            path,
            allow_missing=True,
            target_must_be_directory=False,
        )
        try:
            path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        except (OSError, ValueError) as error:
            raise ArtifactPathError("failed to inspect managed destination") from error
        return True

    def _regular_tree_bytes(self, root: Path) -> int:
        self._assert_managed_path(root)
        try:
            before = _path_snapshot(root.stat(follow_symlinks=False))
        except OSError as error:
            raise ArtifactPathError("failed to inspect accounting directory") from error
        total = 0
        try:
            entries = sorted(os.scandir(root), key=lambda entry: entry.name)
        except OSError as error:
            raise ArtifactPathError("failed to scan accounting directory") from error
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            if stat.S_ISREG(entry_stat.st_mode):
                total += entry_stat.st_size
            elif stat.S_ISDIR(entry_stat.st_mode):
                total += self._regular_tree_bytes(Path(entry.path))
        self._assert_managed_path(root)
        try:
            after = _path_snapshot(root.stat(follow_symlinks=False))
        except OSError as error:
            raise ArtifactPathError("failed to recheck accounting directory") from error
        if after != before:
            raise ArtifactPathError("accounting directory changed during scan")
        return total

    def _ensure_final_parent(self, parent: Path) -> None:
        relative = parent.relative_to(self._artifacts_path)
        current = self._artifacts_path
        for component in relative.parts:
            current = current / component
            self._assert_managed_path(current, allow_missing=True)
            self._ensure_owned_directory(current)
            self._assert_managed_path(current)

    def _validate_payload_tree(
        self,
        root: Path,
        files: tuple[ArtifactFile, ...],
        *,
        allowed_files: frozenset[str] = frozenset(),
        declared_files_only: bool = False,
    ) -> tuple[tuple[str, _PathSnapshot], ...]:
        lexical_root = Path(os.path.abspath(root))
        try:
            resolved_root = lexical_root.resolve(strict=True)
            root_info = lexical_root.stat(follow_symlinks=False)
        except (FileNotFoundError, NotADirectoryError) as error:
            raise ArtifactPathError(
                "source_directory must be an existing directory"
            ) from error
        except (OSError, ValueError) as error:
            raise ArtifactPathError("failed to inspect source_directory") from error
        if resolved_root != lexical_root:
            raise ArtifactPathError(
                "artifact directory contains a symlinked or redirected ancestor"
            )
        root_mode = root_info.st_mode
        if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
            raise ArtifactPathError("source_directory must be a non-symlink directory")
        if declared_files_only:
            snapshots = {"": _path_snapshot(root_info)}
            for item in files:
                current = lexical_root
                parts = Path(item.path).parts
                for index, component in enumerate(parts):
                    current /= component
                    relative = current.relative_to(lexical_root).as_posix()
                    try:
                        info = current.stat(follow_symlinks=False)
                        resolved = current.resolve(strict=True)
                        resolved.relative_to(resolved_root)
                    except (FileNotFoundError, NotADirectoryError) as error:
                        raise ArtifactPathError(
                            f"source is missing declared path: {relative}"
                        ) from error
                    except (OSError, RuntimeError, ValueError) as error:
                        raise ArtifactPathError(
                            f"failed to inspect declared path: {relative}"
                        ) from error
                    is_file = index == len(parts) - 1
                    if stat.S_ISLNK(info.st_mode) or resolved != current:
                        raise ArtifactPathError(
                            f"declared path is symlinked or redirected: {relative}"
                        )
                    if is_file and not stat.S_ISREG(info.st_mode):
                        raise ArtifactPathError(
                            f"declared path is not a regular file: {relative}"
                        )
                    if not is_file and not stat.S_ISDIR(info.st_mode):
                        raise ArtifactPathError(
                            f"declared path ancestor is not a directory: {relative}"
                        )
                    snapshots[relative] = _path_snapshot(info)
            return tuple(sorted(snapshots.items()))
        expected_files = {item.path for item in files}
        permitted_files = expected_files | allowed_files
        expected_directories = {
            "/".join(Path(path).parts[:index])
            for path in expected_files
            for index in range(1, len(Path(path).parts))
        }
        actual_files: set[str] = set()
        source_snapshots = [("", _path_snapshot(root_info))]

        def scan(directory: Path, prefix: str = "") -> None:
            try:
                entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            except OSError as error:
                raise ArtifactPathError("failed to scan source_directory") from error
            for entry in entries:
                relative = f"{prefix}/{entry.name}" if prefix else entry.name
                try:
                    entry_info = entry.stat(follow_symlinks=False)
                except OSError as error:
                    raise ArtifactPathError(
                        f"failed to inspect source entry {relative}"
                    ) from error
                mode = entry_info.st_mode
                if stat.S_ISLNK(mode):
                    raise ArtifactPathError(f"source entry is a symlink: {relative}")
                if stat.S_ISDIR(mode):
                    if relative not in expected_directories:
                        raise ArtifactPathError(
                            f"source contains an undeclared directory: {relative}"
                        )
                    scan(Path(entry.path), relative)
                elif stat.S_ISREG(mode):
                    if relative not in permitted_files:
                        raise ArtifactPathError(
                            f"source contains an undeclared file: {relative}"
                        )
                    actual_files.add(relative)
                else:
                    raise ArtifactPathError(
                        f"source contains a special entry: {relative}"
                    )
                source_snapshots.append((relative, _path_snapshot(entry_info)))

        scan(lexical_root)
        missing = expected_files - actual_files
        if missing:
            raise ArtifactPathError(
                f"source is missing declared files: {sorted(missing)}"
            ) from FileNotFoundError("declared payload file is missing")
        return tuple(source_snapshots)

    def _copy_payload(
        self,
        descriptor: ArtifactDescriptor,
        source: Path,
        staging: Path,
        *,
        consume_source: bool = False,
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> None:
        """Copy or move declared payload into install staging.

        Args:
            descriptor: Artifact descriptor with file metadata.
            source: Source directory containing payload files.
            staging: Destination staging directory.
            consume_source: Move files with os.replace when the source lies
                inside this service's root (both stagings share it). EXDEV
                (bind-mount inside the root) degrades to copy+delete for that
                file — correctness over the disk optimization.
            cancelled: Cancellation probe polled at fixed copy chunks.

        Raises:
            ArtifactPathError: consume_source with a source outside the root,
                or when source path contains symlinks or redirects.
        """
        if consume_source:
            # Validate source is inside root and contains no symlinks or
            # redirects anywhere in its path hierarchy.
            try:
                self._assert_managed_path(source)
            except ArtifactPathError as error:
                raise ArtifactPathError(
                    "consume_source requires a source inside the service root "
                    "with no symlink or redirect path components"
                ) from error

        for item in descriptor.files:
            _raise_if_install_cancelled(cancelled)
            src = source / item.path
            dst = staging / item.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if consume_source:
                try:
                    os.replace(src, dst)
                    continue
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
            with src.open("rb") as source_file, dst.open("xb") as destination_file:
                while chunk := source_file.read(_LOCAL_COPY_CHUNK_BYTES):
                    destination_file.write(chunk)
                    _raise_if_install_cancelled(cancelled)
            if consume_source:
                src.unlink()

    def _verify_payload(
        self,
        root: Path,
        files: tuple[ArtifactFile, ...],
        *,
        allowed_files: frozenset[str] = frozenset(),
        cancelled: Callable[[], bool] = _never_cancelled,
    ) -> None:
        self._validate_payload_tree(
            root,
            files,
            allowed_files=allowed_files,
        )
        for item in files:
            path = root / item.path
            try:
                size = path.stat(follow_symlinks=False).st_size
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                        _raise_if_install_cancelled(cancelled)
            except OSError as error:
                raise ArtifactIntegrityError(
                    f"failed to verify payload file {item.path}"
                ) from error
            if size != item.size_bytes or digest.hexdigest() != item.sha256:
                raise ArtifactIntegrityError(
                    f"payload file does not match descriptor: {item.path}"
                )

    def _verify_existing_destination(
        self,
        destination: Path,
        descriptor: ArtifactDescriptor,
    ) -> None:
        existing = self._read_manifest(destination)
        if existing != descriptor:
            raise ArtifactConflictError(
                "immutable artifact destination has a different descriptor"
            )
        try:
            self._verify_payload(
                destination,
                descriptor.files,
                allowed_files=frozenset({"manifest.json"}),
            )
        except ArtifactPathError as error:
            raise ArtifactIntegrityError(
                "existing artifact payload tree is invalid"
            ) from error

    def _read_manifest(self, directory: Path) -> ArtifactDescriptor:
        self._assert_managed_path(directory)
        try:
            directory_mode = directory.stat(follow_symlinks=False).st_mode
            manifest_path = directory / "manifest.json"
            manifest_mode = manifest_path.stat(follow_symlinks=False).st_mode
            if (
                stat.S_ISLNK(directory_mode)
                or not stat.S_ISDIR(directory_mode)
                or stat.S_ISLNK(manifest_mode)
                or not stat.S_ISREG(manifest_mode)
            ):
                raise ArtifactConflictError(
                    "immutable artifact destination has no valid manifest"
                )
            with manifest_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
            _require_exact_keys(raw, _MANIFEST_KEYS, "manifest")
            assert isinstance(raw, Mapping)
            if (
                raw["schema_version"] != _MANIFEST_SCHEMA_VERSION
                or type(raw["schema_version"]) is not int
            ):
                raise ArtifactDescriptorParseError(
                    "manifest schema_version must be the integer 1"
                )
            descriptor_raw = raw["descriptor"]
            if not isinstance(descriptor_raw, Mapping):
                raise ArtifactDescriptorParseError(
                    "manifest descriptor must be a mapping"
                )
            return ArtifactDescriptor.from_dict(descriptor_raw)
        except ArtifactConflictError:
            raise
        except (
            ArtifactDescriptorError,
            FileNotFoundError,
            NotADirectoryError,
            OSError,
            RecursionError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise ArtifactConflictError(
                "immutable artifact destination has no valid matching manifest"
            ) from error

    def _promote(self, staging: Path, destination: Path) -> None:
        os.rename(staging, destination)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result
