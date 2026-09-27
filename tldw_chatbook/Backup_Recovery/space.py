"""Read-only capacity checks, summed independently on each actual volume."""

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from .native_files import pinned_directory

_MARGIN = 64 * 1024**2


def _volume(path: Path) -> tuple[os.stat_result, Path]:
    """Resolve one existing ancestor and stat it through its pinned descriptor."""
    selected = path
    while not selected.is_dir():
        if selected == selected.parent:
            raise ValueError("capacity_volume_unavailable")
        selected = selected.parent
    with pinned_directory(selected) as descriptor:
        return os.fstat(descriptor), selected


def _require(path: Path, required: int) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or type(required) is not int
        or required < 0
    ):
        raise ValueError("invalid_capacity_requirement")


def held_capacity(path: Path) -> Callable[[int], None]:
    """Resolve one destination's volume once, for a loop that re-checks per chunk.

    ``require_capacity`` calls ``_volume``, which enters ``pinned_directory``
    and opens EVERY path component from ``/`` with
    ``O_RDONLY|O_DIRECTORY|O_NOFOLLOW`` plus an ``fstat`` on each, before the
    ``shutil.disk_usage`` it exists to guard -- measured at 51-89x that call's
    cost. The chunked copy paths called it once per 64 KiB, so a 1 GiB payload
    paid ~1.3 s in path walking alone and a 20 GiB restore ~27 s, on the
    wall-clock path the user watches. So resolve once here and keep only the
    free-space read plus an identity re-check in the loop: same margin, same
    errors, same per-chunk cadence, so a volume that fills mid-copy is still
    caught -- and a resolved name that stops being the pinned object is
    refused rather than measured (see ``check`` below).

    Args:
        path: The destination whose volume is checked. Resolved once.

    Returns:
        A callable taking the bytes about to be written.

    Raises:
        ValueError: ``"invalid_capacity_requirement"`` if ``path`` is not an
            absolute ``Path`` -- raised here at resolution time, and again
            from the returned callable for a requirement that is not a
            non-negative ``int``; ``"capacity_volume_unavailable"`` if no
            existing ancestor directory can be resolved, and again from the
            returned callable if the resolved ancestor no longer names the
            object that was pinned; ``"insufficient_space"`` from the returned
            callable when free space is below the requirement plus the margin,
            on the same predicate ``require_capacity`` uses.
    """
    _require(path, 0)
    pinned, ancestor = _volume(path)
    identity = (pinned.st_dev, pinned.st_ino)

    def check(required: int) -> None:
        _require(path, required)
        # `ancestor` is a NAME, and a name is not an identity: renamed,
        # replaced or remounted, it points somewhere else while the caller's
        # open destination descriptor keeps writing to the original
        # filesystem, and `disk_usage` would then report free space for a
        # volume nothing is being written to. One `stat` per chunk (against
        # the walk-every-component pin this function exists to hoist out of
        # the loop) re-proves the name still resolves to the object that was
        # pinned; a mismatch aborts rather than measuring the impostor.
        current = os.stat(ancestor)
        if (current.st_dev, current.st_ino) != identity:
            raise ValueError("capacity_volume_unavailable")
        if shutil.disk_usage(ancestor).free < required + _MARGIN:
            raise ValueError("insufficient_space")

    return check


def require_capacity(requirements: Mapping[Path, int]) -> None:
    """Sum simultaneously retained bytes by device, including a free-space margin.

    Callers supply each phase's capture, output, decrypt, journal and retained
    rollback requirements. These are local budgets; archive input cannot set them.
    Missing destinations use their existing volume ancestor without creating it.
    """
    volumes = {}
    for path, required in requirements.items():
        _require(path, required)
        info, ancestor = _volume(path)
        prior, _ = volumes.get(info.st_dev, (0, ancestor))
        volumes[info.st_dev] = prior + required, ancestor
    for required, ancestor in volumes.values():
        if shutil.disk_usage(ancestor).free < required + _MARGIN:
            raise ValueError("insufficient_space")
