"""Read local recovery evidence before config imports, fallback, or composition.

This module deliberately uses only stdlib and the private native path reader.
There is no cleanup, catalog fallback, config parsing, or native qualification here.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from ..Utils.private_paths import _open_verified_parent
from .profile_paths import default_config_path, effective_config_path, lexical_path

MAX_RECORD = 1048576
MAX_RECORDS = 4096


class RecoveryRequired(RuntimeError):
    """A bounded reason code, never a local locator or original exception."""


def default_bootstrap_root() -> Path:
    return default_config_path().parent / "recovery-bootstrap"


@contextmanager
def pinned_directory(root: Path, *, _close: Callable[[int], None] | None = None):
    parent, _ = _open_verified_parent(
        root / ".bootstrap-reader", missing_leaf_allowed=True, _close=_close
    )
    try:
        yield parent
    finally:
        (_close or os.close)(parent)


def _read(parent: int, name: str) -> dict:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != os.geteuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("unsafe_record")
        data = os.read(fd, MAX_RECORD + 1)
        if len(data) > MAX_RECORD:
            raise ValueError("oversized_record")

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate_key")
                result[key] = value
            return result

        result = json.loads(data, object_pairs_hook=unique)
        if (
            type(result) is not dict
            or type(result.get("version")) is not int
            or result["version"] != 1
        ):
            raise ValueError("record_version")
        return result
    finally:
        os.close(fd)


def _strings(values: object) -> bool:
    return (
        type(values) is list
        and bool(values)
        and len(values) <= MAX_RECORDS
        and all(type(v) is str and 0 < len(v) <= 4096 and "\0" not in v for v in values)
        and len(set(values)) == len(values)
    )


def _paths(values: object) -> bool:
    return _strings(values) and all(
        Path(v).is_absolute() and ".." not in Path(v).parts for v in values
    )


def _key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fingerprint(selector: Path) -> str:
    """Read a bounded ordinary config without parsing or changing its mode."""
    with pinned_directory(selector.parent) as parent:
        fd = os.open(
            selector.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
                raise ValueError("selector_unverified")
            raw = os.read(fd, MAX_RECORD + 1)
            after = os.fstat(fd)
            identity = lambda info: (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
            if len(raw) > MAX_RECORD or identity(after) != identity(before):
                raise ValueError("selector_unverified")
            return hashlib.sha256(raw).hexdigest()
        finally:
            os.close(fd)


def _overlap(left: Path, right: Path) -> bool:
    a, b = left.resolve(), right.resolve()
    if a == b or a in b.parents or b in a.parents:
        return True
    try:
        return a.samefile(b)
    except FileNotFoundError:
        return False


def _activation_witness(record: object) -> None:
    """Validate local generation evidence without importing execution owners."""
    if (
        type(record) is not dict
        or set(record)
        != {"operation_id", "generation", "owners", "namespaces", "store_root"}
        or any(
            type(record[key]) is not str
            or not 0 < len(record[key]) <= 256
            or "\0" in record[key]
            for key in ("operation_id", "generation")
        )
        or not _strings(record["owners"])
        or any(len(owner) > 256 for owner in record["owners"])
        or record["owners"] != sorted(record["owners"])
        or not _strings(record["namespaces"])
        or record["namespaces"] != sorted(record["namespaces"])
        or not _paths([record["store_root"]])
    ):
        raise ValueError("invalid_activation_witness")


def _control_records(
    root: Path, *, activation: bool = True
) -> tuple[list[dict], list[dict], list[dict]]:
    """Read independent fixed evidence; incomplete paired writes remain fenced."""
    try:
        if stat.S_ISLNK(root.lstat().st_mode):
            raise ValueError("bootstrap_linked")
    except FileNotFoundError:
        return [], [], []
    pending, profiles, activations = [], [], []
    with pinned_directory(root) as parent:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("bootstrap_not_private")
        names = os.listdir(parent)
        if len(names) > MAX_RECORDS:
            raise ValueError("too_many_records")
        for name in names:
            if name in ("admission", "unbound-owner", "projection-dependencies"):
                continue
            if name.startswith("activation-") and not name.startswith(
                "activation-update-"
            ):
                digest = name.removeprefix("activation-").removesuffix(".json")
                if (
                    len(digest) != 64
                    or any(c not in "0123456789abcdef" for c in digest)
                    or not name.endswith(".json")
                ):
                    raise ValueError("unknown_activation_record")
                if not activation:
                    # Fixed activation-only evidence cannot authorize execution
                    # here. Its unavailability must still allow safe inspection.
                    continue
            record = _read(parent, name)
            if name.startswith("activation-update-"):
                # This is explicit write-intent evidence, never a record to skip
                # or repair on reads. Even a damaged intent requires recovery.
                raise ValueError("activation_update_pending")
            if name.startswith("activation-"):
                if (
                    set(record) != {"version", "selector", "activation"}
                    or not _paths([record["selector"]])
                    or name != "activation-" + _key(record["selector"]) + ".json"
                ):
                    raise ValueError("invalid_activation_association")
                _activation_witness(record["activation"])
                activations.append(record)
            elif name.startswith("pending-"):
                if (
                    set(record)
                    != {
                        "version",
                        "operation_id",
                        "namespaces",
                        "control_root",
                        "selectors",
                    }
                    or type(record["operation_id"]) is not str
                    or not 0 < len(record["operation_id"]) <= 256
                    or not _strings(record["namespaces"])
                    or not _paths(record["selectors"])
                    or not _paths([record["control_root"]])
                    or name != "pending-" + _key(record["operation_id"]) + ".json"
                ):
                    raise ValueError("invalid_pending")
                pending.append(record)
            elif name.startswith("profile-"):
                if (
                    set(record) - {"activation"}
                    != {"version", "selector", "fingerprint", "namespaces", "roots"}
                    or not _paths([record["selector"]])
                    or not _strings(record["namespaces"])
                    or not _paths(record["roots"])
                    or type(record["fingerprint"]) is not str
                    or len(record["fingerprint"]) != 64
                    or any(c not in "0123456789abcdef" for c in record["fingerprint"])
                    or name != "profile-" + _key(record["selector"]) + ".json"
                ):
                    raise ValueError("invalid_profile")
                if activation and "activation" in record:
                    _activation_witness(record["activation"])
                    if record["activation"]["namespaces"] != record["namespaces"]:
                        raise ValueError("invalid_profile_activation_scope")
                profiles.append(record)
            else:
                raise ValueError("unknown_record")
    return pending, profiles, activations


def _records(root: Path) -> tuple[list[dict], list[dict]]:
    """Read content admission, leaving activation validation to its read gate."""
    pending, profiles, _ = _control_records(root, activation=False)
    return pending, profiles


def _registry(root: Path) -> dict | None:
    authority = root / "admission"
    try:
        info = authority.lstat()
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode) or info.st_mode & 0o077:
        raise ValueError("authority_unsafe")
    marker = root / "unbound-owner"
    marker_info = marker.lstat()
    if (
        not stat.S_ISREG(marker_info.st_mode)
        or marker_info.st_nlink != 1
        or marker_info.st_uid != os.geteuid()
        or marker_info.st_mode & 0o077
    ):
        raise ValueError("enrollment_marker_unsafe")
    with pinned_directory(authority) as parent:
        if "registry.pending.json" in os.listdir(parent):
            raise ValueError("registry_pending")
        result = _read(parent, "registry.json")
        if set(result) != {"version", "entries"} or type(result["entries"]) is not dict:
            raise ValueError("invalid_registry")
        for name, entry in result["entries"].items():
            if (
                not name
                or type(entry) is not dict
                or set(entry) != {"roots", "historical", "pending", "proposed"}
                or not _paths(entry["roots"])
                or type(entry["historical"]) is not list
                or not all(type(v) is str for v in entry["historical"])
                or entry["pending"] is not None
                or entry["proposed"] != []
            ):
                raise ValueError("uncertain_registry")
        return result["entries"]


def _binding(
    selector: Path, profiles: list[dict], registry: dict | None
) -> dict | None:
    match = next((r for r in profiles if r["selector"] == str(selector)), None)
    if match is None:
        return None
    try:
        if _fingerprint(selector) != match["fingerprint"]:
            return None
    except FileNotFoundError:
        return None
    if registry is None or any(n not in registry for n in match["namespaces"]):
        raise ValueError("binding_authority_missing")
    roots = sorted({p for n in match["namespaces"] for p in registry[n]["roots"]})
    if roots != match["roots"]:
        raise ValueError("binding_mapping_changed")
    # Locally reverify roots. Inode replacement does not grant a different namespace.
    for raw in roots:
        path = Path(raw).resolve(strict=True)
        with pinned_directory(path.parent) as parent:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise ValueError("root_unverified")
    return match


def startup_permission(config_selector: Path, bootstrap_root: Path) -> tuple[bool, str]:
    """Return admission without loading config or creating any default state."""
    try:
        selector = lexical_path(config_selector)
        pending, profiles = _records(bootstrap_root)
        registry = _registry(bootstrap_root)
        if not pending:
            # A normal config edit can require re-enrollment but is not recovery.
            if (
                profiles or (bootstrap_root / "unbound-owner").exists()
            ) and registry is None:
                return False, "recovery_scope_uncertain"
            return True, "startup_allowed"
        if any(
            any(_overlap(selector, Path(p)) for p in r["selectors"]) for r in pending
        ):
            return False, "recovery_pending"
        binding = _binding(selector, profiles, registry)
        if binding is None:
            return False, "recovery_scope_uncertain"
        for record in pending:
            if set(binding["namespaces"]) & set(record["namespaces"]):
                return False, "recovery_pending"
            if registry is None or any(n not in registry for n in record["namespaces"]):
                return False, "recovery_scope_uncertain"
            own_tokens = {
                t for n in binding["namespaces"] for t in registry[n]["historical"]
            }
            affected_tokens = {
                t for n in record["namespaces"] for t in registry[n]["historical"]
            }
            if own_tokens & affected_tokens:
                return False, "recovery_pending"
            affected = record["selectors"] + [
                p for n in record["namespaces"] for p in registry[n]["roots"]
            ]
            if any(
                _overlap(Path(a), Path(b))
                for a in [binding["selector"]] + binding["roots"]
                for b in affected
            ):
                return False, "recovery_pending"
        return True, "startup_allowed"
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError):
        return False, "recovery_scope_uncertain"


def require_startup_permission() -> None:
    """Bounded launcher refusal. Custom config cannot relocate this check."""
    allowed, reason = startup_permission(
        effective_config_path(), default_bootstrap_root()
    )
    if not allowed:
        raise SystemExit("Recovery required: " + reason)
