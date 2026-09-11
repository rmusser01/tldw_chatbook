"""Private recovery control and persistent, disjoint operation working storage."""

import os
from pathlib import Path

from .bootstrap import _read
from .control_records import _ensure, _write
from .native_files import create_private_directory, pinned_directory
from .profile_paths import default_config_path

_MARKER = {"version": 1, "kind": "backup_recovery_service"}


def default_control_root() -> Path:
    """Keep recovery outside profile data and alongside the default config."""
    return default_config_path().parent / "recovery" / "control"


def work_root(control: Path) -> Path:
    """Pending candidates must survive temporary-directory cleanup and restart."""
    return control.with_name(control.name + "-work")


def _private(root: Path) -> None:
    with pinned_directory(root) as parent:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("recovery_control_not_private")


def verify_default_storage() -> None:
    """Recognize only the fixed installed service namespace without repairing it."""
    control = default_control_root()
    _private(control.parent)
    with pinned_directory(control.parent) as parent:
        if set(os.listdir(parent)) != {control.name, work_root(control).name}:
            raise ValueError("recovery_control_unverified")
    _private(control)
    _private(work_root(control))
    with pinned_directory(control) as parent:
        if _read(parent, "service.json") != _MARKER:
            raise ValueError("recovery_control_unverified")


def is_private_service_work(candidate: Path, preserved_item) -> bool:
    """Recognize internal work beneath the exact excluded default service store."""
    control = default_control_root()
    if (
        preserved_item.owner != "recovery.control"
        or preserved_item.logical_id != "recovery.control:service"
        or preserved_item.status != "intentionally_excluded"
        or preserved_item.path != control.parent
        or not candidate.is_absolute()
        or ".." in candidate.parts
        or work_root(control) not in candidate.parents
    ):
        return False
    try:
        verify_default_storage()
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def ensure_storage(control: Path) -> Path:
    """Create missing private storage; never chmod or repair existing evidence."""
    is_default = control == default_control_root()
    if is_default:
        _ensure(control.parent)
        _private(control.parent)
    _ensure(control.parent)
    try:
        create_private_directory(control)
        created = True
    except FileExistsError:
        created = False
    _private(control)
    _ensure(work_root(control))
    _private(work_root(control))
    if not is_default:
        return work_root(control)
    with pinned_directory(control) as parent:
        try:
            marker = _read(parent, "service.json")
        except FileNotFoundError:
            if not created:
                raise ValueError("recovery_control_unverified") from None
            marker = None
        if marker is not None and marker != _MARKER:
            raise ValueError("recovery_control_unverified")
    if marker is None:
        try:
            _write(
                control,
                "service.json",
                b'{"version":1,"kind":"backup_recovery_service"}\n',
            )
        except FileExistsError:
            pass
        with pinned_directory(control) as parent:
            if _read(parent, "service.json") != _MARKER:
                raise ValueError("recovery_control_unverified")
    return work_root(control)
