"""Local convenience locators for separately validated restored profiles."""

from __future__ import annotations

import hashlib
import os
import stat
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .admission import Admission, fcntl
from .bootstrap import _read
from .native_files import create_private_directory, flush_directory, pinned_directory
from .native_platform import flush_file
from .qualification import qualified_for


class _Entry(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    profile_id: str
    config: str
    data: str


def _name(profile_id: str) -> str:
    if (
        type(profile_id) is not str
        or not 0 < len(profile_id) <= 256
        or "\0" in profile_id
    ):
        raise ValueError("catalog_id_invalid")
    return hashlib.sha256(profile_id.encode()).hexdigest() + ".json"


def _absolute(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("catalog_absolute_path_required")
    return path


@contextmanager
def _private(root: Path):
    with pinned_directory(root) as parent:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("catalog_not_private")
        yield parent


def _targets(config: Path, data: Path) -> tuple[Path, Path]:
    config, data = _absolute(config), _absolute(data)
    for path, directory in ((config, False), (data, True)):
        with pinned_directory(path.parent) as parent:
            info = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            if info.st_uid != os.geteuid() or not (
                stat.S_ISDIR(info.st_mode)
                if directory
                else stat.S_ISREG(info.st_mode) and info.st_nlink == 1
            ):
                raise ValueError("catalog_target_invalid")
    return config, data


class ProfileCatalog:
    """Persist locators only; launch must separately verify admission/activation.

    Reads never initialize or repair control state. Registration cannot reassign
    an existing ID or import approval metadata, and never modifies target data.
    """

    def __init__(self, control_root: Path):
        self.root = _absolute(control_root) / "profile-catalog"

    def register(self, profile_id: str, config: Path, data: Path) -> None:
        """Durably register exact local locators after executor validation."""
        name = _name(profile_id)
        config, data = _targets(config, data)
        if data == self.root or data in self.root.parents:
            raise ValueError("catalog_overlaps_data")
        entry = _Entry(profile_id=profile_id, config=str(config), data=str(data))
        raw = entry.model_dump_json().encode()
        if len(raw) > 1048576:
            raise ValueError("catalog_record_too_large")
        control = self.root.parent
        existing = control if control.exists() else control.parent
        allowed, reason = qualified_for("admission", existing)
        if not allowed:
            raise ValueError(reason)
        try:
            create_private_directory(control)
        except FileExistsError:
            pass
        with (
            pinned_directory(control.parent) as ancestor,
            _private(control) as control_fd,
        ):
            try:
                create_private_directory(self.root)
            except FileExistsError:
                pass
            with _private(self.root) as parent:
                try:
                    Admission._write_new_record(parent, name, raw)
                except FileExistsError:
                    # A prior complete record may have failed its last barrier.
                    # Recheck its pinned identity before flushing an exact retry.
                    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    if _Entry.model_validate(_read(parent, name)) != entry:
                        raise ValueError("catalog_mapping_changed") from None
                    fd = os.open(
                        name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=parent,
                    )
                    try:
                        info = os.fstat(fd)
                        identity = lambda row: (
                            row.st_dev,
                            row.st_ino,
                            row.st_size,
                            row.st_mtime_ns,
                            row.st_ctime_ns,
                        )
                        if (
                            identity(before) != identity(info)
                            or _read(parent, name) != entry.model_dump()
                        ):
                            raise ValueError("catalog_record_changed")
                        flush_file(fd)
                        after = os.stat(name, dir_fd=parent, follow_symlinks=False)
                        if (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino):
                            raise ValueError("catalog_record_changed")
                    finally:
                        os.close(fd)
                flush_directory(parent)
            # Retry the directory associations too: either mkdir may have
            # completed before a failed parent barrier on an earlier attempt.
            flush_directory(control_fd)
            flush_directory(ancestor)

    def resolve(self, profile_id: str) -> tuple[Path, Path]:
        """Read checked locators, without creating state or granting execution."""
        name = _name(profile_id)
        with _private(self.root.parent), _private(self.root) as parent:
            entry = _Entry.model_validate(_read(parent, name))
            if entry.profile_id != profile_id:
                raise ValueError("catalog_id_mismatch")
            return _targets(Path(entry.config), Path(entry.data))
