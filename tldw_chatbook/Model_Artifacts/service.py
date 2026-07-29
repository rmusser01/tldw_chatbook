"""Provider-neutral immutable model-artifact contracts."""

from __future__ import annotations

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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlsplit

from tldw_chatbook.Utils.atomic_file_ops import atomic_write_json

from .leases import (
    ArtifactLeaseError,
    ArtifactLeaseKey,
    ArtifactOperationLease,
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
_LIFECYCLE_LEASE_KEY = ArtifactLeaseKey("!lifecycle", "1", "writer")
_PathSnapshot = tuple[int, int, int, int, int, int]
_NodeIdentity = tuple[int, int, int]


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
        _validate_url("source_url", self.source_url)
        _validate_url("license_url", self.license_url)

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
class ArtifactDiskUsage:
    """Logical managed-store byte totals and current filesystem capacity."""

    installed_bytes: int
    staging_bytes: int
    free_bytes: int


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
                installed.append(
                    InstalledArtifact(
                        path=path,
                        descriptor=descriptor,
                        ready=False,
                        active=False,
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

    def install(
        self,
        descriptor: ArtifactDescriptor,
        source_directory: Path,
    ) -> ArtifactRef:
        """Verify and promote one local source directory immutably."""

        if type(descriptor) is not ArtifactDescriptor:
            raise TypeError("descriptor must be an ArtifactDescriptor")
        if not isinstance(source_directory, Path):
            raise TypeError("source_directory must be a Path")
        source_directory = Path(os.path.abspath(source_directory))
        source_snapshot = self._validate_payload_tree(
            source_directory,
            descriptor.files,
        )

        staging: Path | None = None
        try:
            self._assert_managed_path(self._staging_path)
            staging = Path(
                tempfile.mkdtemp(
                    prefix="install-",
                    dir=self._staging_path,
                )
            )
            self._assert_managed_path(staging)
            self._copy_payload(descriptor, source_directory, staging)
            if (
                self._validate_payload_tree(
                    source_directory,
                    descriptor.files,
                )
                != source_snapshot
            ):
                raise ArtifactPathError("source tree changed during artifact copy")
            destination = self.artifact_path(descriptor.reference)
            self._assert_managed_path(self._locks_path)

            # ponytail: one lifecycle writer lock is enough until measured install throughput
            # justifies per-artifact writer coordination.
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
                    if self._managed_path_exists(destination):
                        self._verify_existing_destination(
                            destination,
                            descriptor,
                        )
                        return descriptor.reference
                    self._verify_payload(staging, descriptor.files)
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
                    self._promote(staging, destination)
                    self._assert_managed_path(destination)
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
            if staging is not None:
                primary_error = sys.exception()
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
        expected_files = {item.path for item in files}
        permitted_files = expected_files | allowed_files
        expected_directories = {
            "/".join(Path(path).parts[:index])
            for path in expected_files
            for index in range(1, len(Path(path).parts))
        }
        actual_files: set[str] = set()
        snapshots = [("", _path_snapshot(root_info))]

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
                snapshots.append((relative, _path_snapshot(entry_info)))

        scan(lexical_root)
        missing = expected_files - actual_files
        if missing:
            raise ArtifactPathError(
                f"source is missing declared files: {sorted(missing)}"
            )
        return tuple(snapshots)

    def _copy_payload(
        self,
        descriptor: ArtifactDescriptor,
        source: Path,
        staging: Path,
    ) -> None:
        for item in descriptor.files:
            destination = staging / item.path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / item.path, destination, follow_symlinks=False)

    def _verify_payload(
        self,
        root: Path,
        files: tuple[ArtifactFile, ...],
        *,
        allowed_files: frozenset[str] = frozenset(),
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
