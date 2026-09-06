"""Immutable Canvas runtime-profile catalog loading and pure admission selection."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Literal

from .runtime_assets import RUNTIME_MANIFEST_BYTES, load_canvas_runtime_assets

PROFILE_CATALOG_BYTES = RUNTIME_MANIFEST_BYTES
PROFILE_UNAVAILABLE = "profile-unavailable"
_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OPERATIONS = frozenset({"create", "update", "rename", "load"})
_CATALOG_FIELDS = {
    "schema_version",
    "build_id",
    "policy_id",
    "default_diagram_profile",
    "profiles",
}
_ENTRY_FIELDS = {
    "profile_id",
    "manifest",
    "manifest_sha256",
    "executable",
    "reason",
    "library",
}
_LIBRARY_FIELDS = {"bytes", "files"}
_CONTRACT_FIELDS = {
    "engine",
    "facade",
    "plan",
    "grammar",
    "layout",
    "unicode",
    "quotas",
}
_IDENTITY_FIELDS = {"id", "sha256"}
_UNICODE_FIELDS = {"version", "segmentation", "width", "sha256"}
_QUOTA_FIELDS = {
    "id",
    "html_bytes",
    "script_bytes",
    "runtime_memory_bytes",
    "stack_bytes",
    "startup_milliseconds",
    "event_milliseconds",
    "pending_jobs",
    "dom_nodes",
    "css_rules",
    "patches_per_operation",
}


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    """Bounded source-free projection of one verified runtime manifest."""

    profile_id: str
    manifest_sha256: str
    executable: bool
    reason: str | None
    library_bytes: int

    def __post_init__(self) -> None:
        _validate_profile_id(self.profile_id)
        _validate_digest(self.manifest_sha256, "manifest_sha256")
        if type(self.executable) is not bool:
            raise ValueError("executable must be a boolean")
        if self.executable and self.reason is not None:
            raise ValueError("executable profiles cannot have a refusal reason")
        if not self.executable and not _is_safe_reason(self.reason):
            raise ValueError("unavailable profiles require a bounded refusal reason")
        if type(self.library_bytes) is not int or self.library_bytes < 0:
            raise ValueError("library_bytes must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ProfileSnapshot:
    """Verified process-owned runtime-profile admission authority."""

    build_id: str
    policy_id: str
    profiles: tuple[ProfileRecord, ...]
    default_diagram_profile: str | None

    def __post_init__(self) -> None:
        _validate_digest(self.build_id, "build_id")
        _validate_digest(self.policy_id, "policy_id")
        if type(self.profiles) is not tuple:
            raise ValueError("profiles must be a tuple")
        identities = [record.profile_id for record in self.profiles]
        if len(identities) != len(set(identities)):
            raise ValueError("profile identities must be unique")
        if self.default_diagram_profile is not None:
            _validate_profile_id(self.default_diagram_profile)


@dataclass(frozen=True, slots=True)
class ProfileResolution:
    """Source-free result of pure runtime-profile selection and admission."""

    profile_id: str
    executable: bool
    reason: str | None


def _validate_profile_id(value: object) -> str:
    if type(value) is not str or _PROFILE_ID.fullmatch(value) is None:
        raise ValueError("invalid Canvas runtime profile identity")
    return value


def _validate_digest(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lower-case SHA-256 identity")
    return value


def _is_safe_reason(value: object) -> bool:
    return (
        type(value) is str
        and 0 < len(value.encode("utf-8")) <= 64
        and _PROFILE_ID.fullmatch(value) is not None
    )


def _strict_json(data: bytes) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(data.decode("utf-8"), object_pairs_hook=object_pairs)
    if type(value) is not dict:
        raise ValueError("catalog must be an object")
    return value


def _read_bounded(resource: Any, limit: int) -> bytes:
    with resource.open("rb") as handle:
        value = handle.read(limit + 1)
    if len(value) > limit:
        raise ValueError("resource exceeds its packaged byte limit")
    return value


def _valid_metadata(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"bytes", "sha256"}
        and type(value["bytes"]) is int
        and value["bytes"] >= 0
        and type(value["sha256"]) is str
        and _SHA256.fullmatch(value["sha256"]) is not None
    )


def _validate_manifest_contract(manifest: dict[str, Any]) -> None:
    contract = manifest.get("profile_contract")
    if type(contract) is not dict or set(contract) != _CONTRACT_FIELDS:
        raise ValueError("runtime manifest omits a pinned profile contract")
    for field in ("engine", "facade", "plan", "grammar", "layout"):
        identity = contract[field]
        if type(identity) is not dict or set(identity) != _IDENTITY_FIELDS:
            raise ValueError("invalid runtime contract identity")
        if type(identity["id"]) is not str or not identity["id"]:
            raise ValueError("invalid runtime contract identifier")
        _validate_digest(identity["sha256"], f"{field}.sha256")
    unicode_contract = contract["unicode"]
    if type(unicode_contract) is not dict or set(unicode_contract) != _UNICODE_FIELDS:
        raise ValueError("invalid Unicode contract identity")
    for field in ("version", "segmentation", "width"):
        if type(unicode_contract[field]) is not str or not unicode_contract[field]:
            raise ValueError("invalid Unicode contract identifier")
    _validate_digest(unicode_contract["sha256"], "unicode.sha256")
    quotas = contract["quotas"]
    if type(quotas) is not dict or set(quotas) != _QUOTA_FIELDS:
        raise ValueError("invalid quota contract identity")
    if type(quotas["id"]) is not str or not quotas["id"]:
        raise ValueError("invalid quota contract identifier")
    for field in _QUOTA_FIELDS - {"id"}:
        if type(quotas[field]) is not int or quotas[field] <= 0:
            raise ValueError("invalid quota contract value")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_verified_snapshot() -> ProfileSnapshot:
    static = files("tldw_chatbook.Canvas").joinpath("static")
    catalog = _strict_json(
        _read_bounded(static.joinpath("profile-catalog.json"), PROFILE_CATALOG_BYTES)
    )
    if set(catalog) != _CATALOG_FIELDS or catalog.get("schema_version") != 1:
        raise ValueError("unsupported profile catalog schema")
    _validate_digest(catalog.get("build_id"), "build_id")
    _validate_digest(catalog.get("policy_id"), "policy_id")
    default = catalog.get("default_diagram_profile")
    if default is not None:
        _validate_profile_id(default)
    entries = catalog.get("profiles")
    if type(entries) is not list or not entries:
        raise ValueError("profile catalog must contain profiles")

    runtime_assets = load_canvas_runtime_assets()
    if not runtime_assets.enabled or runtime_assets.manifest_bytes is None:
        raise ValueError("runtime assets are unavailable")
    records: list[ProfileRecord] = []
    build_projection: list[dict[str, object]] = []
    policy_projection: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != _ENTRY_FIELDS:
            raise ValueError("invalid profile catalog entry")
        profile_id = _validate_profile_id(entry["profile_id"])
        if profile_id in seen:
            raise ValueError("duplicate profile identity")
        seen.add(profile_id)
        if entry["manifest"] != "runtime-manifest.json":
            raise ValueError("unsupported runtime manifest path")
        manifest_sha256 = _validate_digest(entry["manifest_sha256"], "manifest_sha256")
        if hashlib.sha256(runtime_assets.manifest_bytes).hexdigest() != manifest_sha256:
            raise ValueError("runtime manifest integrity mismatch")
        manifest = _strict_json(runtime_assets.manifest_bytes)
        if manifest.get("runtime_profile") != profile_id:
            raise ValueError("runtime profile identity reuse")
        _validate_manifest_contract(manifest)

        library = entry["library"]
        if type(library) is not dict or set(library) != _LIBRARY_FIELDS:
            raise ValueError("invalid library inventory")
        if type(library["bytes"]) is not int or library["bytes"] < 0:
            raise ValueError("invalid library byte inventory")
        library_files = library["files"]
        if type(library_files) is not dict:
            raise ValueError("invalid library file inventory")
        library_total = 0
        for name, metadata in library_files.items():
            _validate_profile_id(name)
            if not _valid_metadata(metadata):
                raise ValueError("invalid library file metadata")
            contents = _read_bounded(static.joinpath(name), RUNTIME_MANIFEST_BYTES)
            actual = {
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
            if actual != metadata:
                raise ValueError("library file integrity mismatch")
            library_total += len(contents)
        if library_total != library["bytes"]:
            raise ValueError("library byte inventory mismatch")

        record = ProfileRecord(
            profile_id=profile_id,
            manifest_sha256=manifest_sha256,
            executable=entry["executable"],
            reason=entry["reason"],
            library_bytes=library_total,
        )
        records.append(record)
        build_projection.append(
            {
                "library": library,
                "manifest_sha256": manifest_sha256,
                "profile_id": profile_id,
            }
        )
        policy_projection.append(
            {
                "executable": record.executable,
                "profile_id": profile_id,
                "reason": record.reason,
            }
        )
    if (
        _canonical_digest(sorted(build_projection, key=lambda item: item["profile_id"]))
        != catalog["build_id"]
    ):
        raise ValueError("profile build identity mismatch")
    policy_value = {
        "default_diagram_profile": default,
        "profiles": sorted(policy_projection, key=lambda item: item["profile_id"]),
    }
    if _canonical_digest(policy_value) != catalog["policy_id"]:
        raise ValueError("profile policy identity mismatch")
    if default is not None:
        admitted = {record.profile_id for record in records if record.executable}
        if default not in admitted:
            raise ValueError("diagram default is not admitted")
    return ProfileSnapshot(
        build_id=catalog["build_id"],
        policy_id=catalog["policy_id"],
        profiles=tuple(records),
        default_diagram_profile=default,
    )


def load_profile_snapshot() -> ProfileSnapshot:
    """Verify packaged catalog/runtime inputs and return their immutable projection."""

    try:
        return _load_verified_snapshot()
    except Exception as exc:
        raise ValueError("Canvas runtime profile catalog is unavailable") from exc


def resolve_profile(
    snapshot: ProfileSnapshot,
    *,
    operation: Literal["create", "update", "rename", "load"],
    parent_profile: str | None,
    has_diagrams: bool,
) -> ProfileResolution:
    """Select an exact profile from an owned snapshot without I/O or mutation."""

    if operation not in _OPERATIONS:
        raise ValueError("unsupported Canvas profile operation")
    if type(has_diagrams) is not bool:
        raise ValueError("has_diagrams must be a boolean")
    if operation != "create" and parent_profile is None:
        raise ValueError("this Canvas operation requires a parent profile")
    if parent_profile is not None:
        _validate_profile_id(parent_profile)

    if operation in {"rename", "load"}:  # noqa: SIM114 - mirrors normative precedence
        selected = parent_profile
    elif operation == "update" and parent_profile != "canvas-v1":
        selected = parent_profile
    elif has_diagrams:
        selected = snapshot.default_diagram_profile
    else:
        selected = "canvas-v1"
    if selected is None:
        return ProfileResolution(PROFILE_UNAVAILABLE, False, PROFILE_UNAVAILABLE)
    _validate_profile_id(selected)
    record = next(
        (item for item in snapshot.profiles if item.profile_id == selected), None
    )
    if record is None:
        return ProfileResolution(selected, False, PROFILE_UNAVAILABLE)
    return ProfileResolution(record.profile_id, record.executable, record.reason)


def runtime_snapshot_id(snapshot: ProfileSnapshot) -> str:
    """Return the canonical source-free cross-process identity for ``snapshot``."""

    projection = {
        "build_id": snapshot.build_id,
        "default_diagram_profile": snapshot.default_diagram_profile,
        "policy_id": snapshot.policy_id,
        "profiles": sorted(
            (
                {
                    "executable": record.executable,
                    "library_bytes": record.library_bytes,
                    "manifest_sha256": record.manifest_sha256,
                    "profile_id": record.profile_id,
                    "reason": record.reason,
                }
                for record in snapshot.profiles
            ),
            key=lambda item: item["profile_id"],
        ),
    }
    return _canonical_digest(projection)


__all__ = [
    "PROFILE_CATALOG_BYTES",
    "PROFILE_UNAVAILABLE",
    "ProfileRecord",
    "ProfileResolution",
    "ProfileSnapshot",
    "load_profile_snapshot",
    "resolve_profile",
    "runtime_snapshot_id",
]
