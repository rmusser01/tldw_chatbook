"""Provider-neutral immutable model-artifact contracts."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlsplit

from .leases import ArtifactLeaseKey


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


class ArtifactDescriptorError(ValueError):
    """Base error for artifact descriptor parsing and validation."""


class ArtifactDescriptorValidationError(ArtifactDescriptorError):
    """Raised when a directly constructed descriptor value is invalid."""


class ArtifactDescriptorParseError(ArtifactDescriptorError):
    """Raised when a serialized descriptor does not match the versioned shape."""


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
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ArtifactDescriptorValidationError(
            f"{field_name} must not contain control characters"
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

        for field_name in (
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
        ):
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
                    "model_id",
                    "consumer",
                    "model_family",
                    "upstream_repository",
                    "upstream_revision",
                    "source_url",
                    "precision",
                    "license_id",
                    "license_url",
                    "usage_notice",
                    "runtime_name",
                    "runtime_version_constraint",
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
    enum_type: type[ArtifactRole] | type[ArtifactFormat] | type[ProvenanceClass],
    context: str,
) -> ArtifactRole | ArtifactFormat | ProvenanceClass:
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
