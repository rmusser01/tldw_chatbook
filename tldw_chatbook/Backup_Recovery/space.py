"""Read-only capacity checks, summed independently on each actual volume."""

import shutil
from collections.abc import Mapping
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from .native_files import pinned_directory

_MARGIN = 64 * 1024**2


def _volume(path: Path) -> tuple[int, Path]:
    selected = path
    while not selected.is_dir():
        if selected == selected.parent:
            raise ValueError("capacity_volume_unavailable")
        selected = selected.parent
    with pinned_directory(selected) as descriptor:
        return os.fstat(descriptor).st_dev, selected


def require_capacity(requirements: Mapping[Path, int]) -> None:
    """Sum simultaneously retained bytes by device, including a free-space margin.

    Callers supply each phase's capture, output, decrypt, journal and retained
    rollback requirements. These are local budgets; archive input cannot set them.
    Missing destinations use their existing volume ancestor without creating it.
    """
    volumes = {}
    for path, required in requirements.items():
        if (
            not isinstance(path, Path)
            or not path.is_absolute()
            or type(required) is not int
            or required < 0
        ):
            raise ValueError("invalid_capacity_requirement")
        device, ancestor = _volume(path)
        prior, _ = volumes.get(device, (0, ancestor))
        volumes[device] = prior + required, ancestor
    for required, ancestor in volumes.values():
        if shutil.disk_usage(ancestor).free < required + _MARGIN:
            raise ValueError("insufficient_space")
