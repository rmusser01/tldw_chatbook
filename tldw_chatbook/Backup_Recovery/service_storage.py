"""Private recovery control and persistent, disjoint operation working storage."""

from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from .bootstrap import _bounded_names, _read
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


def _private(root: Path, *, _custody=None):
    with pinned_directory(root, _custody=_custody) as parent:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("recovery_control_not_private")
        return info.st_dev, info.st_ino, info.st_mode


def verify_default_storage(*, _budget=None, _custody=None):
    """Recognize only the fixed installed service namespace without repairing it."""
    control = default_control_root()
    if _budget is not None:
        _budget.reserve(8192 + 64 * len(str(control)))
    before = _private(control.parent, _custody=_custody)
    with pinned_directory(control.parent, _custody=_custody) as parent:
        names = (
            os.listdir(parent)
            if _budget is None
            else _bounded_names(parent, 2, _budget, _custody)
        )
        if set(names) != {control.name, work_root(control).name}:
            raise ValueError("recovery_control_unverified")
    control_identity = _private(control, _custody=_custody)
    work_identity = _private(work_root(control), _custody=_custody)
    with pinned_directory(control, _custody=_custody) as parent:
        if _read(parent, "service.json", _budget=_budget, _custody=_custody) != _MARKER:
            raise ValueError("recovery_control_unverified")
    if (
        _private(control.parent, _custody=_custody) != before
        or _private(control, _custody=_custody) != control_identity
        or _private(work_root(control), _custody=_custody) != work_identity
    ):
        raise ValueError("recovery_control_changed")
    if _budget is not None:
        return before, control_identity, work_identity


def is_private_service_work(
    candidate: Path, preserved_item, *, _budget=None, _custody=None
) -> bool:
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
        verify_default_storage(_budget=_budget, _custody=_custody)
    except (OSError, ValueError, RuntimeError):
        return False
    return True


def is_preserved_service_container(
    candidate: Path, destination: Path, plan, *, synthetic: bool
) -> bool:
    """Allow private service work only inside an untouched existing container.

    A synthetic archive root does not own its existing directory. Its selected
    files may share that parent with the exact excluded default service store,
    provided no planned publication, retirement or new container touches it.
    """
    from .models import Inventory
    from .restore_plan import RestorePlan

    if (
        synthetic is not True
        or type(plan) is not RestorePlan
        or plan.mode != "replace"
        or type(plan.target) is not Inventory
        or destination not in dict(plan.destinations).values()
        or destination.is_symlink()
        or not destination.is_dir()
    ):
        return False
    mutations = (*plan.restore, *plan.retire, *plan.containers)
    return any(
        item.path is not None
        and destination in item.path.parents
        and (item.logical_id, item.path) in plan.preserve
        and is_private_service_work(candidate, item)
        and not any(
            path == item.path or path in item.path.parents or item.path in path.parents
            for _, path in mutations
        )
        for item in plan.target.items
    )


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
