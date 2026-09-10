"""Locally issued fixed recovery associations; never an archive import API."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .admission import Admission, AdmissionTimeout, fcntl
from .bootstrap import (
    _binding,
    _fingerprint,
    _key,
    _overlap,
    _records,
    _registry,
    RecoveryRequired,
)
from .native_files import create_private_directory, flush_directory, pinned_directory
from .profile_paths import lexical_path
from .qualification import qualified_for

UNBOUND_NAMESPACE = "bootstrap.unbound"


class _Pending(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    operation_id: str = Field(min_length=1, max_length=256)
    namespaces: list[str] = Field(min_length=1, max_length=4096)
    control_root: str
    selectors: list[str] = Field(min_length=1, max_length=4096)


class _Profile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    selector: str
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    namespaces: list[str] = Field(min_length=1, max_length=4096)
    roots: list[str] = Field(min_length=1, max_length=4096)


@contextmanager
def _enrollment(authority: Admission):
    try:
        with authority.maintenance((UNBOUND_NAMESPACE,), 0.1):
            yield
    except AdmissionTimeout:
        raise RecoveryRequired("close_unenrolled_clients_and_restart") from None


def _ensure(root: Path) -> None:
    """Registration only: create missing private ancestors, never fix existing ones."""
    if not root.exists():
        _ensure(root.parent)
        try:
            create_private_directory(root)
        except FileExistsError:
            pass
    with pinned_directory(root) as parent:
        if os.fstat(parent).st_uid != os.geteuid() or os.fstat(parent).st_mode & 0o077:
            # Ancestors may be normal trusted home/config dirs; only final bootstrap
            # is checked private by _records, not chmodded during registration.
            return


def _write(root: Path, name: str, data: bytes) -> None:
    if len(data) > 1048576:
        raise ValueError("record_too_large")
    with pinned_directory(root) as parent:
        # Publish exclusively in place: an incomplete/failed write is itself a fence.
        Admission._write_new_record(parent, name, data)
        flush_directory(parent)


def admission_authority(bootstrap_root: Path) -> Admission:
    """Initialize only qualified fixed authority, outside every replacement target."""
    existing = bootstrap_root.parent
    while not existing.exists():
        existing = existing.parent
    allowed, reason = qualified_for("admission", existing)
    if not allowed:
        raise RecoveryRequired(reason)
    _ensure(bootstrap_root)
    _records(bootstrap_root)
    marker = bootstrap_root / "unbound-owner"
    created_marker = False
    if not marker.exists():
        if (bootstrap_root / "admission").exists():
            raise RecoveryRequired("recovery_scope_uncertain")
        try:
            _write(bootstrap_root, marker.name, b"local enrollment owner\n")
            created_marker = True
        except FileExistsError:
            pass
    if created_marker:
        authority = Admission(bootstrap_root / "admission")
        authority.register(UNBOUND_NAMESPACE, (marker,))
        return authority
    try:
        authority = Admission.open_existing(bootstrap_root / "admission")
        with authority._directory() as parent:
            with authority._lock(parent, "registry.lock", fcntl.LOCK_SH):
                registry = authority._read(parent)
                # Validate the marker's private regular-file posture and the exact
                # registered physical identity. Never repair missing/replaced state.
                _registry(bootstrap_root)
                entry = registry.entries.get(UNBOUND_NAMESPACE)
                if (
                    entry is None
                    or entry.roots != [str(marker)]
                    or entry.pending is not None
                    or entry.proposed
                    or not authority._tokens((marker,)) <= set(entry.historical)
                ):
                    raise RecoveryRequired("recovery_scope_uncertain")
        return authority
    except (OSError, ValueError, RuntimeError):
        raise RecoveryRequired("recovery_scope_uncertain") from None


def _verify_pending_selector(selector: Path) -> None:
    """Verify existing bytes or genuine absence for a local pre-publication fence."""
    try:
        selector.lstat()
    except FileNotFoundError:
        pass
    else:
        with pinned_directory(selector.parent):
            _fingerprint(selector)  # Existing bytes stay strict; no TOML parsing.
        return

    ancestor = selector
    missing = None
    while True:
        try:
            ancestor.lstat()
            break
        except FileNotFoundError:
            missing = ancestor.name
            ancestor = ancestor.parent
    # This also rejects links (including dangling links) and non-directories
    # in the existing prefix. Never create destination directories to enroll it.
    with pinned_directory(ancestor) as parent:
        if missing is not None:
            try:
                os.stat(missing, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                return
    raise ValueError("selector_absence_changed")


def register_pending(
    bootstrap_root: Path,
    operation_id: str,
    namespaces: tuple[str, ...],
    control_root: Path,
    selectors: tuple[Path, ...],
) -> None:
    """Durably fence local selectors before publication; never clears evidence."""
    names = Admission._names(namespaces)
    if type(selectors) is not tuple or not selectors:
        raise ValueError("invalid_selectors")
    selected = tuple(lexical_path(p) for p in selectors)
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate_selectors")
    for selector in selected:
        _verify_pending_selector(selector)
        if _overlap(selector, bootstrap_root) or _overlap(selector, control_root):
            raise ValueError("control_root_overlaps_target")
    registry = _registry(bootstrap_root)
    if registry is not None:
        for namespace in names:
            if namespace not in registry:
                raise ValueError("namespace_unregistered")
            for raw in registry[namespace]["roots"]:
                if _overlap(Path(raw), bootstrap_root) or _overlap(
                    Path(raw), control_root
                ):
                    raise ValueError("control_root_overlaps_target")
    record = _Pending(
        operation_id=operation_id,
        namespaces=list(names),
        control_root=str(lexical_path(control_root)),
        selectors=[str(p) for p in selected],
    )
    existing = bootstrap_root.parent
    while not existing.exists():
        existing = existing.parent
    allowed, reason = qualified_for("admission", existing)
    if not allowed:
        raise RecoveryRequired(reason)
    _ensure(bootstrap_root)
    _records(bootstrap_root)
    _write(
        bootstrap_root,
        "pending-" + _key(operation_id) + ".json",
        record.model_dump_json().encode(),
    )


def bind_profile(
    bootstrap_root: Path,
    config_selector: Path,
    namespaces: tuple[str, ...],
    authority_root: Path,
) -> None:
    """Enroll an intact local mapping only after unbound owners have retired.

    Namespaces must already be installed in the fixed authority. This operation
    neither parses an archive nor chooses storage from serialized archive locators.
    Changed mappings require an explicit future reconciliation/refresh operation.
    """
    if lexical_path(authority_root) != lexical_path(bootstrap_root / "admission"):
        raise RecoveryRequired("conflicting_admission_authority")
    authority = admission_authority(bootstrap_root)
    selected = lexical_path(config_selector)
    names = Admission._names(namespaces)
    with _enrollment(authority):
        pending, profiles = _records(bootstrap_root)
        if pending:
            raise RecoveryRequired("recovery_pending")
        registry = _registry(bootstrap_root)
        if registry is None or any(n not in registry for n in names):
            raise ValueError("namespace_unregistered")
        roots = sorted({p for n in names for p in registry[n]["roots"]})
        if any(_overlap(Path(p), bootstrap_root) for p in roots + [str(selected)]):
            raise ValueError("control_root_overlaps_target")
        if not any(_overlap(selected, Path(p)) for p in roots):
            raise ValueError("selector_not_in_admission_scope")
        record = {
            "version": 1,
            "selector": str(selected),
            "fingerprint": _fingerprint(selected),
            "namespaces": list(names),
            "roots": roots,
        }
        if _binding(selected, [record], registry) is None:
            raise ValueError("binding_unverified")
        validated = _Profile.model_validate(record)

        _write(
            bootstrap_root,
            "profile-" + _key(str(selected)) + ".json",
            validated.model_dump_json().encode(),
        )
