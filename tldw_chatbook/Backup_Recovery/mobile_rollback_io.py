"""Bounded physical observations used by the fixed mobile rollback executor."""

import hashlib
import stat
from pathlib import Path

from tldw_chatbook.Utils.platform_files import os

from .archive_reader import _check
from .journal import (
    MAX_EVENTS,
    _artifact_component_limit,
    _ArtifactObservationLimits,
    observe_artifact,
)
from .native_files import _close_owned, create_private_file, pinned_directory


def _physical(info):
    if any(
        type(value) is not int or not 0 <= value < 2**64
        for value in (info.st_dev, info.st_ino)
    ):
        raise ValueError("mobile_rollback_physical_scalar_invalid")
    if info.st_uid != os.geteuid() or not (
        stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
    ):
        raise ValueError("mobile_rollback_object_unverified")


def _directory(path, info):
    _physical(info)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("mobile_rollback_parent_changed")
    return {
        "path": str(path),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
        "ctime_ns": info.st_ctime_ns,
    }


def parent_census(path, budget, custody, cancel):
    """Enumerate a pinned parent with admission before each retained child."""
    budget.value(str(path), text_copies=8)
    budget.reserve(4096)
    with pinned_directory(path, _custody=custody) as parent:
        before = os.fstat(parent)
        state = _directory(path, before)
        component = _artifact_component_limit(parent)
        budget.reserve(component * 16 + 1024)
        children = {}
        iterator = os.scandir(parent)
        custody.directory_iterators.append(iterator)
        try:
            for entry in iterator:
                _check(cancel)
                if (
                    len(children) >= MAX_EVENTS
                    or len(os.fsencode(entry.name)) > component
                ):
                    raise ValueError("mobile_rollback_parent_limit")
                budget.reserve(4096 + 64 * (len(str(path)) + len(entry.name)))
                info = os.stat(entry.name, dir_fd=parent, follow_symlinks=False)
                _physical(info)
                children[entry.name] = {
                    "path": str(path / entry.name),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "mode": info.st_mode,
                    "owner": info.st_uid,
                    "size": info.st_size,
                    "links": info.st_nlink,
                    "mtime_ns": info.st_mtime_ns,
                    "ctime_ns": info.st_ctime_ns,
                }
        finally:
            custody.close_directory_iterator(iterator)
        if _directory(path, os.fstat(parent)) != state:
            raise ValueError("mobile_rollback_parent_changed")
    return {"state": state, "children": children}


def observation(path, budget, custody, cancel, *, metadata=False):
    """Bound the complete native tree observation before hashing any contents."""
    budget.value(str(path), text_copies=16)
    # This first pass observes topology only; it does not allocate read buffers.
    # The native content pass below reserves its own two 64 KiB buffers.
    budget.reserve(8192)
    count = total = characters = longest = depth = 0
    component = 0
    observations = []

    def walk(parent, name, relative):
        nonlocal count, total, characters, longest, depth, component
        _check(cancel)
        if count >= MAX_EVENTS or relative.count("/") > 256:
            raise ValueError("mobile_rollback_observation_limit")
        budget.reserve(8192 + 256 * len(relative))
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        custody.fds.append(fd)
        try:
            info = os.fstat(fd)
            _physical(info)
            count += 1
            characters += len(relative)
            longest = max(longest, len(relative))
            depth = max(depth, relative.count("/"))
            observations.append(
                (
                    relative,
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    info.st_mode,
                )
            )
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1 or not 0 <= info.st_size <= 256 * 1024**3:
                    raise ValueError("mobile_rollback_object_unverified")
                total += info.st_size
                if total > 1024**4:
                    raise ValueError("mobile_rollback_observation_limit")
            else:
                component = max(component, _artifact_component_limit(fd))
                budget.reserve(component * 16 + 1024)
                iterator = os.scandir(fd)
                custody.directory_iterators.append(iterator)
                try:
                    for entry in iterator:
                        budget.reserve(256 + 16 * (len(relative) + len(entry.name)))
                        walk(fd, entry.name, relative + "/" + entry.name)
                finally:
                    custody.close_directory_iterator(iterator)
                after = os.fstat(fd)
                if (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                    after.st_mode,
                ) != (
                    info.st_dev,
                    info.st_ino,
                    info.st_size,
                    info.st_mtime_ns,
                    info.st_ctime_ns,
                    info.st_mode,
                ):
                    raise ValueError("mobile_rollback_object_changed")
        finally:
            _close_owned(fd, custody)

    with pinned_directory(path.parent, _custody=custody) as parent:
        component = _artifact_component_limit(parent)
        walk(parent, path.name, "")
    from .mobile_journal_frame import _local as journal_local

    frame = getattr(journal_local, "frame", None)
    read_buffer = None
    stream_cost = 131072
    if frame is not None:
        frame.check()
        if frame.budget is not budget:
            raise ValueError("mobile_native_budget_changed")
        read_buffer = frame.artifact_buffer()
        # Each file has fixed complete reads, at most one partial view and its
        # final EOF read/view. Short nonfinal reads refuse before further work.
        stream_cost = 512 * (total // 65536 + 3 * count)
    else:
        from .mobile_preview_budget import current

        scope = current()
        if scope is not None:
            if scope.read_budget is not budget:
                raise ValueError("mobile_preview_budget_changed")
            copies = scope.sqlite_copies if hasattr(scope, "sqlite_copies") else scope
            view = copies._buffer()
            budget.reserve(256)
            read_buffer = (view.obj, view, [view])
            stream_cost = 512 * (total // 65536 + 3 * count)
    budget.reserve(
        stream_cost
        + 8192
        + 12 * component
        + 4096 * count
        + 128 * characters
        + 64 * count * (depth + 1)
        + 128 * len(str(path))
    )
    limits = _ArtifactObservationLimits(
        count, total, characters, longest, depth, component
    )
    result = observe_artifact(
        path,
        metadata=metadata,
        _custody=custody,
        _limits=limits,
        _read_buffer=read_buffer,
    )
    first = observations[0]
    if (result["device"], result["inode"]) != first[1:3]:
        raise ValueError("mobile_rollback_object_changed")
    return result


def copy_snapshot(
    candidate, temporary, expected_size, expected_digest, budget, custody, cancel
):
    """Copy the actual owner-validated immutable snapshot with no overwrite."""
    budget.reserve(2 * 65536 + 16384)
    budget.value(str(candidate), text_copies=4)
    budget.value(str(temporary), text_copies=4)
    with pinned_directory(candidate.parent, _custody=custody) as parent:
        source = os.open(
            candidate.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        custody.fds.append(source)
        try:
            before = os.fstat(source)
            _physical(before)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != expected_size
            ):
                raise ValueError("mobile_rollback_snapshot_changed")
            digest, size = hashlib.sha256(), 0
            with create_private_file(temporary, _custody=custody) as destination:
                while True:
                    _check(cancel)
                    chunk = os.read(source, 65536)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > expected_size:
                        raise ValueError("mobile_rollback_snapshot_changed")
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = os.write(destination, view)
                        if written <= 0:
                            raise OSError("mobile_rollback_snapshot_write_failed")
                        view = view[written:]
                if size != expected_size or digest.hexdigest() != expected_digest:
                    raise ValueError("mobile_rollback_snapshot_changed")
                after = os.fstat(source)
                named = os.stat(candidate.name, dir_fd=parent, follow_symlinks=False)
                identity = lambda row: (
                    row.st_dev,
                    row.st_ino,
                    row.st_size,
                    row.st_mtime_ns,
                    row.st_ctime_ns,
                    row.st_mode,
                )
                if identity(after) != identity(before) or identity(named) != identity(
                    before
                ):
                    raise ValueError("mobile_rollback_snapshot_changed")
        finally:
            _close_owned(source, custody)


def move_position(intent, budget, custody, cancel):
    """Accept only the recorded before/after transition and unchanged siblings."""
    source, destination = Path(intent["source"]["path"]), Path(intent["destination"])
    left, right = source.exists(), destination.exists()
    if left == right:
        raise ValueError("mobile_rollback_move_uncertain")
    current = source if left else destination
    if observation(current, budget, custody, cancel) != {
        **intent["source"],
        "path": str(current),
    }:
        raise ValueError("mobile_rollback_move_changed")
    for expected in intent["parents"]:
        path = Path(expected["state"]["path"])
        actual = parent_census(path, budget, custody, cancel)
        keys = ("device", "inode", "mode") if right else tuple(expected["state"])
        if any(actual["state"][key] != expected["state"][key] for key in keys):
            raise ValueError("mobile_rollback_parent_changed")
        excluded = {row.name for row in (source, destination) if row.parent == path}
        if {
            name: row
            for name, row in actual["children"].items()
            if name not in excluded
        } != {
            name: row
            for name, row in expected["children"].items()
            if name not in excluded
        }:
            raise ValueError("mobile_rollback_siblings_changed")
    return "before" if left else "after"


def move_original(intent, budget, custody, cancel, *, _before_effect=None):
    """Use qualified native no-replace while preserving current object metadata."""
    from .native_files import _rename_new
    from .native_platform import flush_directory, flush_file
    from .qualification import _qualified_identity, native_identity

    if move_position(intent, budget, custody, cancel) != "before":
        raise ValueError("mobile_rollback_move_uncertain")
    source, destination = Path(intent["source"]["path"]), Path(intent["destination"])
    with (
        pinned_directory(source.parent, _custody=custody) as left,
        pinned_directory(destination.parent, _custody=custody) as right,
    ):
        identities = {
            row["state"]["path"]: (row["state"]["device"], row["state"]["inode"])
            for row in intent["parents"]
        }
        for path, fd in ((source.parent, left), (destination.parent, right)):
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != identities[str(path)]:
                raise ValueError("mobile_rollback_parent_changed")
        device = os.fstat(left).st_dev
        if device != os.fstat(right).st_dev:
            raise ValueError("cross_volume_retirement_unqualified")
        for operation in (
            "publish_new",
            "publish_directory"
            if intent["source"]["kind"] == "directory"
            else "publish_file",
        ):
            allowed, reason = _qualified_identity(operation, native_identity(right))
            if not allowed:
                raise ValueError(reason)

        def flush(fd, depth=0):
            _check(cancel)
            info = os.fstat(fd)
            _physical(info)
            if info.st_dev != device or depth > 256:
                raise ValueError("mobile_rollback_object_changed")
            if stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise ValueError("mobile_rollback_object_changed")
                flush_file(fd)
                return
            iterator = os.scandir(fd)
            custody.directory_iterators.append(iterator)
            try:
                for entry in iterator:
                    budget.reserve(4096 + 16 * len(entry.name))
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=fd,
                    )
                    custody.fds.append(child)
                    try:
                        flush(child, depth + 1)
                    finally:
                        _close_owned(child, custody)
            finally:
                custody.close_directory_iterator(iterator)
            flush_directory(fd)

        fd = os.open(
            source.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=left
        )
        custody.fds.append(fd)
        try:
            flush(fd)
            remaining = budget._remaining
            if move_position(intent, budget, custody, cancel) != "before" or os.stat(
                source.name, dir_fd=left, follow_symlinks=False
            ) != os.fstat(fd):
                raise ValueError("mobile_rollback_move_uncertain")
            if _before_effect is not None:
                _before_effect(remaining - budget._remaining)
            _rename_new(left, source.name, right, destination.name)
            flush_directory(right)
            flush_directory(left)
        finally:
            _close_owned(fd, custody)


def sync_moved(intent, budget, custody, cancel):
    """Flush only a proven after-position when reconciling a durable move intent."""
    from .native_platform import flush_directory, flush_file

    if move_position(intent, budget, custody, cancel) != "after":
        raise ValueError("mobile_rollback_move_uncertain")
    target = Path(intent["destination"])

    def flush(fd, depth=0):
        info = os.fstat(fd)
        _physical(info)
        if depth > 256:
            raise ValueError("mobile_rollback_object_changed")
        if stat.S_ISREG(info.st_mode):
            flush_file(fd)
            return
        iterator = os.scandir(fd)
        custody.directory_iterators.append(iterator)
        try:
            for entry in iterator:
                _check(cancel)
                budget.reserve(4096 + 16 * len(entry.name))
                child = os.open(
                    entry.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd
                )
                custody.fds.append(child)
                try:
                    flush(child, depth + 1)
                finally:
                    _close_owned(child, custody)
        finally:
            custody.close_directory_iterator(iterator)
        flush_directory(fd)

    with pinned_directory(target.parent, _custody=custody) as parent:
        fd = os.open(
            target.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        custody.fds.append(fd)
        try:
            flush(fd)
        finally:
            _close_owned(fd, custody)
        flush_directory(parent)
    for row in intent["parents"]:
        with pinned_directory(Path(row["state"]["path"]), _custody=custody) as parent:
            flush_directory(parent)


def metadata(path, expected, budget, custody, *, apply=None):
    """Check the pinned restored snapshot; apply only authenticated metadata."""
    from .native_platform import flush_directory, flush_file

    budget.reserve(8192 + 16 * len(str(path)))
    with pinned_directory(path.parent, _custody=custody) as parent:
        fd = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
        )
        custody.fds.append(fd)
        try:
            info = os.fstat(fd)
            _physical(info)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or (info.st_dev, info.st_ino) != (expected["device"], expected["inode"])
            ):
                raise ValueError("mobile_rollback_snapshot_changed")
            if apply is not None:
                os.fchmod(fd, apply["mode"])
                os.utime(fd, ns=(info.st_atime_ns, apply["mtime_ns"]))
                flush_file(fd)
                flush_directory(parent)
            info = os.fstat(fd)
            if info != os.stat(path.name, dir_fd=parent, follow_symlinks=False):
                raise ValueError("mobile_rollback_snapshot_changed")
            return {
                "version": 1,
                "mode": stat.S_IMODE(info.st_mode),
                "mtime_ns": info.st_mtime_ns,
            }
        finally:
            _close_owned(fd, custody)
