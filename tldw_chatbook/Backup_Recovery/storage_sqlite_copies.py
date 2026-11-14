"""Private SQLite main/WAL copying for the original admitted scope objects."""

from __future__ import annotations

from . import storage_admission as host


def _verify_sqlite_target(self, target):
    expected = self.sqlite_targets.get(target)
    if expected is None:
        return
    root = host.os.stat(self.sqlite_directory, follow_symlinks=False)
    if (root.st_dev, root.st_ino) != self.sqlite_directory_identity:
        raise ValueError("capture_sqlite_staging_changed")
    resources = host._CaptureFileDescriptors(self)
    try:
        with resources.pinned_directory(target.parent) as parent:
            info = host.os.fstat(parent)
            if (info.st_dev, info.st_ino) != expected[0]:
                raise ValueError("capture_sqlite_target_changed")
            for suffix, identity in zip(("", "-wal"), expected[1], strict=True):
                try:
                    info = host.os.stat(
                        target.name + suffix, dir_fd=parent, follow_symlinks=False
                    )
                except FileNotFoundError:
                    info = None
                actual = None if info is None else self._sqlite_identity(info)
                if actual != identity:
                    raise ValueError("capture_sqlite_target_changed")
    finally:
        resources.retire()


def _sqlite_identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _capture_sqlite_target(
    self, owner_id, source, progress_guard=None, *, require_capacity
):
    source = host.lexical_path(source)
    if self.staging in source.parents:
        self.verify_sqlite_target(source)
        return None
    if progress_guard is not None:
        progress_guard()
    before = self._sqlite_state(source)
    if source in self.sqlite_snapshots:
        original, target = self.sqlite_snapshots[source]
        if original != before:
            raise ValueError("capture_sqlite_source_changed")
        if target is None:
            raise ValueError("capture_sqlite_copy_incomplete")
        self.verify_sqlite_target(target)
        return target
    required = sum(info[5] for info in before[1][:2] if info is not None)
    if (
        any(
            info is not None and info[5] > self.limits.member_bytes
            for info in before[1][:2]
        )
        or self.copied_bytes + required > self.byte_budget
    ):
        raise ValueError("capture_sqlite_limit")
    require_capacity({self.staging: required * 2})
    resources = host._CaptureFileDescriptors(self)
    try:
        if self.sqlite_directory is None:
            from uuid import uuid4

            name = "sqlite-sources-" + uuid4().hex
            with resources.pinned_directory(self.staging) as parent:
                info = host.os.fstat(parent)
                if (info.st_dev, info.st_ino) != (
                    self.staging_identity.st_dev,
                    self.staging_identity.st_ino,
                ):
                    raise ValueError("capture_sqlite_staging_changed")
                host.os.mkdir(name, mode=0o700, dir_fd=parent)
                self.sqlite_directory = self.staging / name
                created = host.os.stat(name, dir_fd=parent, follow_symlinks=False)
                self.sqlite_directory_identity = created.st_dev, created.st_ino
        target_root = self.sqlite_directory / str(len(self.sqlite_snapshots))
        with resources.pinned_directory(self.sqlite_directory) as parent:
            info = host.os.fstat(parent)
            if (info.st_dev, info.st_ino) != self.sqlite_directory_identity:
                raise ValueError("capture_sqlite_staging_changed")
            host.os.mkdir(target_root.name, mode=0o700, dir_fd=parent)
            created = host.os.stat(
                target_root.name, dir_fd=parent, follow_symlinks=False
            )
            target_identity = created.st_dev, created.st_ino
        target = target_root / "source.sqlite3"
        # Failed attempts remain unqualified until scope retirement.
        self.sqlite_snapshots[source] = before, None
        private_identities = [None, None]
        wal_header = False
        main_header = bytearray()
        with resources.pinned_directory(source.parent) as parent:
            if self._sqlite_identity(host.os.fstat(parent)) != before[0]:
                raise ValueError("capture_sqlite_source_changed")
            for suffix, expected in zip(("", "-wal"), before[1][:2], strict=True):
                if expected is None and not (suffix == "-wal" and wal_header):
                    continue
                if expected is not None:
                    descriptor = host.os.open(
                        source.name + suffix,
                        host.os.O_RDONLY | host.os.O_NOFOLLOW | host.os.O_NONBLOCK,
                        dir_fd=parent,
                    )
                    resources.fds.append(descriptor)
                    if self._sqlite_identity(host.os.fstat(descriptor)) != expected:
                        raise ValueError("capture_sqlite_source_changed")
                with resources.pinned_directory(target_root) as output_parent:
                    info = host.os.fstat(output_parent)
                    if (info.st_dev, info.st_ino) != target_identity:
                        raise ValueError("capture_sqlite_staging_changed")
                    output = host.os.open(
                        target.name + suffix,
                        host.os.O_WRONLY
                        | host.os.O_CREAT
                        | host.os.O_EXCL
                        | host.os.O_NOFOLLOW,
                        0o600,
                        dir_fd=output_parent,
                    )
                    resources.fds.append(output)
                count = 0
                while expected is not None:
                    if progress_guard is not None:
                        progress_guard()
                    self.check()
                    chunk = host.os.read(
                        descriptor, min(1024**2, expected[5] - count + 1)
                    )
                    if not chunk:
                        break
                    if not suffix and len(main_header) < 20:
                        # WAL-header databases need an empty WAL even when
                        # the original was cleanly checkpointed. Create and
                        # bind it now; never adopt a later replacement.
                        main_header.extend(chunk[: 20 - len(main_header)])
                        wal_header = (
                            main_header[:16] == b"SQLite format 3\x00"
                            and main_header[18:20] == b"\x02\x02"
                        )
                    count += len(chunk)
                    if count > expected[5]:
                        raise ValueError("capture_sqlite_source_changed")
                    view = memoryview(chunk)
                    while view:
                        written = host.os.write(output, view)
                        if not written:
                            raise OSError("capture_sqlite_write_failed")
                        view = view[written:]
                if expected is not None and (
                    count != expected[5]
                    or self._sqlite_identity(host.os.fstat(descriptor)) != expected
                ):
                    raise ValueError("capture_sqlite_source_changed")
                current_parent = host.os.stat(target_root, follow_symlinks=False)
                current = host.os.stat(
                    target_root / (target.name + suffix), follow_symlinks=False
                )
                held = host.os.fstat(output)
                if (
                    current_parent.st_dev,
                    current_parent.st_ino,
                ) != target_identity or (current.st_dev, current.st_ino) != (
                    held.st_dev,
                    held.st_ino,
                ):
                    raise ValueError("capture_sqlite_staging_changed")
                private_identities[bool(suffix)] = self._sqlite_identity(held)
    finally:
        try:
            self.verify_sqlite_sources()
        finally:
            resources.retire()
    self.copied_bytes += required
    self.sqlite_targets[target] = target_identity, tuple(private_identities)
    self.sqlite_snapshots[source] = before, target
    return target


def _capture_sqlite_state(self, source):
    resources = host._CaptureFileDescriptors(self)
    try:
        with resources.pinned_directory(source.parent) as parent:
            parent_info = host.os.fstat(parent)
            result = []
            for suffix in ("", "-wal", "-shm", "-journal"):
                try:
                    info = host.os.stat(
                        source.name + suffix, dir_fd=parent, follow_symlinks=False
                    )
                except FileNotFoundError:
                    result.append(None)
                    continue
                if (
                    not host.stat.S_ISREG(info.st_mode)
                    or info.st_nlink != 1
                    or info.st_uid != host.os.geteuid()
                ):
                    raise ValueError("capture_sqlite_source_changed")
                result.append(self._sqlite_identity(info))
            if result[0] is None or (source, *result[0][:2]) not in self.sources:
                raise ValueError("capture_sqlite_source_changed")
            if result[3] is not None and result[3][5]:
                raise ValueError("capture_sqlite_hot_journal")
            return (self._sqlite_identity(parent_info), tuple(result))
    finally:
        resources.retire()


def _preview_sqlite_target(
    self,
    source,
    *,
    shutil,
    tempfile,
    create_private_directory,
    create_private_file,
    require_capacity,
):
    if self.read_budget is not None:
        from .mobile_preview_budget import path_cost

        self.read_budget.reserve(4096 + path_cost(source, copies=5))
    source = host.lexical_path(source)
    if source in self.snapshots:
        previous, target = self.snapshots[source]
        if self.native_scope is not None and self._source_state(source) != previous:
            raise ValueError("preview_sqlite_changed")
        # Owners sharing this source inspect one verified image per preview.
        # Ordinary live commits cannot invalidate that private image; a
        # replaced or unsafe source path still cannot reuse its identity.
        with host.bootstrap.pinned_directory(
            source.parent,
            _close=self.custody.close_descriptor if self.custody else None,
        ) as parent:
            current = host.os.stat(source.name, dir_fd=parent, follow_symlinks=False)
            if (
                not host.stat.S_ISREG(current.st_mode)
                or current.st_nlink != 1
                or current.st_uid != host.os.geteuid()
                or (current.st_dev, current.st_ino) != previous[0][:2]
            ):
                raise ValueError("preview_sqlite_changed")
        return target
    before = self._source_state(source)
    required = sum(state[2] for state in before[:2] if state is not None)
    if (
        any(
            state is not None and state[2] > self.limits.member_bytes
            for state in before[:2]
        )
        or self.copied_bytes + required > self.byte_budget
    ):
        raise ValueError("preview_sqlite_limit")
    temporary_parent = host.Path(tempfile.gettempdir()).resolve()
    require_capacity({temporary_parent: required * 2})
    if self.directory is None:
        self.directory = host.Path(
            tempfile.mkdtemp(prefix="chatbook-preview-", dir=temporary_parent)
        )
    target_root = self.directory / str(len(self.snapshots))
    create_private_directory(target_root)
    target = target_root / "source.sqlite3"
    try:
        with host.bootstrap.pinned_directory(
            source.parent,
            _close=self.custody.close_descriptor if self.custody else None,
        ) as parent:
            for suffix, expected in zip(("", "-wal"), before[:2], strict=True):
                if expected is None:
                    continue
                descriptor = host.os.open(
                    source.name + suffix,
                    host.os.O_RDONLY | host.os.O_NOFOLLOW | host.os.O_NONBLOCK,
                    dir_fd=parent,
                )
                if self.custody is not None:
                    self.custody.fds.append(descriptor)
                try:
                    opened = host.os.fstat(descriptor)
                    if (
                        opened.st_dev,
                        opened.st_ino,
                        opened.st_size,
                        opened.st_mtime_ns,
                        opened.st_ctime_ns,
                    ) != expected:
                        raise ValueError("preview_sqlite_changed")
                    with create_private_file(
                        target.with_name(target.name + suffix),
                        _custody=self.custody,
                    ) as output:
                        if self.read_budget is not None:
                            self._buffer()
                        count = 0
                        while True:
                            size = min(1024**2, expected[2] - count + 1)
                            if self.read_budget is not None:
                                self.read_budget.reserve(768)
                                view = self.copy_buffer[: min(65536, size)]
                                size = host.os.readv(descriptor, [view])
                                chunk = view[:size]
                            else:
                                chunk = host.os.read(descriptor, size)
                            if not chunk:
                                break
                            count += len(chunk)
                            if count > expected[2]:
                                raise ValueError("preview_sqlite_changed")
                            view = memoryview(chunk)
                            while view:
                                if self.read_budget is not None:
                                    self.read_budget.reserve(256)
                                written = host.os.write(output, view)
                                if not written:
                                    raise OSError("preview_sqlite_write_failed")
                                view = view[written:]
                        info = host.os.fstat(descriptor)
                        observed = (
                            info.st_dev,
                            info.st_ino,
                            info.st_size,
                            info.st_mtime_ns,
                            info.st_ctime_ns,
                        )
                        if count != expected[2] or observed != expected:
                            raise ValueError("preview_sqlite_changed")
                finally:
                    if self.custody is None:
                        host.os.close(descriptor)
                    else:
                        from .native_files import _close_owned

                        _close_owned(descriptor, self.custody)
        if self._source_state(source) != before:
            raise ValueError("preview_sqlite_changed")
    except BaseException:
        shutil.rmtree(target_root)
        raise
    self.copied_bytes += required
    self.snapshots[source] = before, target
    return target


def _preview_source_state(self, source):
    if self.read_budget is not None:
        from .mobile_preview_budget import path_cost

        self.read_budget.reserve(2048 + path_cost(source, copies=3))
    result = []
    for suffix in ("", "-wal", "-journal"):
        path = source.with_name(source.name + suffix)
        try:
            info = host.os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            result.append(None)
            continue
        if (
            not host.stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_uid != host.os.geteuid()
        ):
            raise ValueError("preview_sqlite_unavailable")
        result.append(
            (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        )
    if result[0] is None or result[2] is not None and result[2][2]:
        # A hot rollback journal needs recovery; preview never runs that on
        # the source or guesses a committed image from an incomplete copy.
        raise ValueError("preview_sqlite_unavailable")
    return tuple(result)
