"""Committed imported history through actual enrolled leases and retained pins."""

import hashlib
import io
import json
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from tldw_chatbook.Backup_Recovery import native_files, storage_admission
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
from tldw_chatbook.Utils.platform_files import os

_ACTIVE = set()
_QUARANTINED = set()
_MAX_ROOT_BYTES = 16 * 1024 * 1024


def _changed():
    raise ValueError("imported_history_witness_changed")


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
        info.st_uid,
    )


class _PinnedHistory:
    """Only the live native reader issues this owner of actual descriptors."""

    def __init__(self):
        raise TypeError("imported_history_pins_are_native_issued")

    def check(self):
        if type(self) is not _PinnedHistory or self not in _ACTIVE or self.close_failed:
            raise ValueError("imported_history_pins_not_admitted")
        if self.database_lease.execution_context(self.database_path) != self.context:
            _changed()
        for path, lease in self.leases:
            if (
                lease.resource_close_failed
                or lease.execution_context(path) != self.context
            ):
                _changed()

    def close_descriptor(self, fd):
        if self.close_failed:
            raise ValueError("imported_history_retirement_uncertain")
        try:
            os.close(fd)
        except BaseException:  # noqa: BLE001 -- Every uncertain close, including cancellation, retains custody.
            self.close_failed = True
            raise ValueError("imported_history_retirement_uncertain") from None

    def retire(self):
        if self.close_failed:
            _QUARANTINED.add(self)
            raise ValueError("imported_history_retirement_uncertain")
        try:
            while self.fds:
                fd = self.fds[-1]
                self.close_descriptor(fd)
                self.fds.pop()
            for _path, lease in reversed(self.leases):
                lease.close()
        except BaseException:
            self.close_failed = True
            _QUARANTINED.add(self)
            raise
        else:
            _ACTIVE.discard(self)

    def _census(self, relative):
        fd = self.pins[relative]
        expected = self.children[relative]
        self.budget.container(2 * len(expected) + 2)
        found = set()
        with os.scandir(fd) as entries:
            for entry in entries:
                if entry.name not in expected or entry.name in found:
                    _changed()
                found.add(entry.name)
        if found != set(expected):
            _changed()

    def pin(self):
        self.check()
        with native_files.pinned_directory(self.root.parent, _custody=self) as parent:
            fd = os.open(
                self.root.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            self.fds.append(fd)
            self.pins["."] = fd
        for relative in self.order:
            if relative != ".":
                parent, _, leaf = relative.rpartition("/")
                fd = os.open(
                    leaf,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                    dir_fd=self.pins[parent or "."],
                )
                self.fds.append(fd)
                self.pins[relative] = fd
            info = os.fstat(self.pins[relative])
            expected = self.members[relative]["artifact"]
            if (
                info.st_dev != expected["device"]
                or info.st_ino != expected["inode"]
                or info.st_uid != os.geteuid()
                or (not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode))
                or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)
                or (stat.S_ISDIR(info.st_mode) != (expected["kind"] == "directory"))
            ):
                _changed()
            self.stats[relative] = _identity(info)
            if relative in self.children:
                self._census(relative)
        self.verify()

    def verify(self):
        """One bounded digest pass, including every requested-root member."""
        self.check()
        self.passes += 1
        if self.passes > 2:
            raise ValueError("imported_history_digest_pass_limit")
        raw_digests = {}
        for relative in self.order:
            fd = self.pins[relative]
            if _identity(os.fstat(fd)) != self.stats[relative]:
                _changed()
            if relative in self.children:
                self._census(relative)
                continue
            expected = self.members[relative]
            if self.stats[relative][3] != expected["size"]:
                _changed()
            os.lseek(fd, 0, os.SEEK_SET)
            digest = hashlib.sha256()
            remaining = expected["size"]
            while remaining:
                count = os.readv(fd, [self.view[: min(remaining, len(self.buffer))]])
                if not count:
                    _changed()
                digest.update(self.view[:count])
                remaining -= count
            if (
                os.readv(fd, [self.view[:1]])
                or digest.hexdigest() != expected["sha256"]
            ):
                _changed()
            raw_digests[relative] = digest.hexdigest()
        # _Object's SHA hashes its canonical tree census, including for a file;
        # it is deliberately distinct from the raw source file digest.
        for target in self.order:
            rows, total = [], 0
            for relative in self.order:
                if (
                    target != "."
                    and relative != target
                    and not relative.startswith(target + "/")
                ):
                    continue
                suffix = (
                    ""
                    if relative == target
                    else "/"
                    + (relative if target == "." else relative[len(target) + 1 :])
                )
                if relative in self.children:
                    rows.append((suffix, "directory"))
                else:
                    size = self.members[relative]["size"]
                    rows.append((suffix, "file", size, raw_digests[relative]))
                    total += size
            digest = hashlib.sha256(
                json.dumps(rows, separators=(",", ":")).encode()
            ).hexdigest()
            expected = self.members[target]["artifact"]
            if expected["size"] != total or expected["sha256"] != digest:
                _changed()
        if self.members["."]["artifact"] != self.root_witness:
            _changed()
        for relative in self.order:
            if _identity(os.fstat(self.pins[relative])) != self.stats[relative]:
                _changed()
            if relative in self.children:
                self._census(relative)
            if relative != ".":
                parent, _, leaf = relative.rpartition("/")
                named = os.stat(
                    leaf, dir_fd=self.pins[parent or "."], follow_symlinks=False
                )
                if _identity(named) != self.stats[relative]:
                    _changed()
        with native_files.pinned_directory(self.root.parent, _custody=self) as parent:
            if (
                _identity(os.stat(self.root.name, dir_fd=parent, follow_symlinks=False))
                != self.stats["."]
            ):
                _changed()
        self.check()

    def next_segment(self, minimum):
        self.check()
        chosen = None
        for index, relative in self.segments:
            if index >= minimum and (chosen is None or (index, relative) < chosen):
                chosen = index, relative
        return chosen

    @contextmanager
    def open_segment(self, relative):
        self.check()
        if (
            relative not in self.pins
            or relative in self.children
            or not any(name == relative for _, name in self.segments)
        ):
            _changed()
        self.budget.reserve(16384)
        stream = io.BufferedReader(io.FileIO(self.pins[relative], "rb", closefd=False))
        try:
            yield stream
        finally:
            try:
                stream.close()
            except BaseException:
                self.close_failed = True
                _QUARANTINED.add(self)
                raise


def _pins(database, connection, root, members, root_witness, directory, budget):
    from tldw_chatbook.Agents.run_log_paging import _SEGMENT
    from tldw_chatbook.DB.private_sqlite import _ordinary_connections

    lease = _ordinary_connections.get(connection)
    if lease is None or lease.resource_close_failed:
        raise ValueError("imported_history_database_not_admitted")
    context = lease.execution_context(database.db_path)
    if not context[1] or context[1] == ("bootstrap.unbound",):
        raise ValueError("imported_history_enrollment_required")
    budget.container(len(members) * 20)
    # Native empty root spelling remains in receipts; dot is private FD notation.
    by_path = {row["relative_path"] or ".": row for row in members}
    if len(by_path) != len(members) or "." not in by_path:
        _changed()
    if (
        sum(row["size"] for row in members if row["sha256"] is not None)
        > _MAX_ROOT_BYTES
    ):
        raise ValueError("imported_history_root_byte_limit")
    children = {path: [] for path, row in by_path.items() if row["sha256"] is None}
    for path in by_path:
        if path == ".":
            continue
        parent, _, leaf = path.rpartition("/")
        if (parent or ".") not in children:
            _changed()
        children[parent or "."].append(leaf)
    order = []

    def visit(path):
        order.append(path)
        for leaf in sorted(children.get(path, ())):
            visit(leaf if path == "." else path + "/" + leaf)

    visit(".")
    # Reserve both bounded witness passes before allocating pins or reading bytes.
    budget.reserve(65536 + len(members) * 4096)
    for target in order:
        for path in order:
            if target == "." or target == path or path.startswith(target + "/"):
                budget.reserve(2 * (2048 + 64 * len(path)))
    budget.container(len(order) * 6)
    value = object.__new__(_PinnedHistory)
    value.database_lease, value.database_path, value.context = (
        lease,
        database.db_path,
        context,
    )
    value.root, value.root_witness, value.members = root, root_witness, by_path
    value.children, value.order, value.budget = children, order, budget
    value.pins, value.stats, value.leases, value.fds = {}, {}, [], []
    value.close_failed, value.passes = False, 0
    value.buffer = bytearray(65536)
    value.view = memoryview(value.buffer)
    budget.container(len(by_path) * 3)
    value.segments = []
    segment_keys = set()
    for path, row in by_path.items():
        parsed = PurePosixPath(path)
        match = _SEGMENT.fullmatch(parsed.name)
        if row["sha256"] is not None and match:
            key = (str(parsed.parent), int(match[1]))
            if key in segment_keys:
                raise ValueError("imported_history_segment_ambiguous")
            segment_keys.add(key)
            if key[0] == directory:
                value.segments.append((key[1], path))
    _ACTIVE.add(value)
    try:
        for relative in order:
            path = root if relative == "." else root / relative
            acquired = storage_admission.acquire_storage(path)
            value.leases.append((path, acquired))
            if acquired.execution_context(path) != context:
                raise ValueError("imported_history_file_not_admitted")
        value.pin()
        return value
    except BaseException:
        value.retire()
        raise


def _selected_config(database_lease, database_path, budget):
    """Bound the ordinary selected config under its own actual enrolled lease."""
    from tldw_chatbook.Backup_Recovery.bootstrap import MAX_RECORD
    from tldw_chatbook.Backup_Recovery.inventory import _parse_profile_config
    from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

    selector = effective_config_path()
    lease = storage_admission.acquire_storage(selector)
    custody = storage_admission._CaptureFileDescriptors(None)
    try:
        if lease.execution_context(selector) != database_lease.execution_context(
            database_path
        ):
            raise ValueError("imported_history_selector_not_admitted")
        with native_files.pinned_directory(selector.parent, _custody=custody) as parent:
            fd = os.open(
                selector.name,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=parent,
            )
            custody.fds.append(fd)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_RECORD:
                raise ValueError("imported_history_selector_unreadable")
            budget.reserve(128 + 2 * (before.st_size + 1))
            raw = os.read(fd, before.st_size + 1)
            if (
                len(raw) != before.st_size
                or _identity(os.fstat(fd)) != _identity(before)
                or _identity(
                    os.stat(selector.name, dir_fd=parent, follow_symlinks=False)
                )
                != _identity(before)
            ):
                _changed()
            native_files._close_owned(fd, custody)
        return _parse_profile_config(raw, _read_budget=budget)
    finally:
        if custody.close_failed:
            _QUARANTINED.add((lease, custody))
        else:
            custody.retire()
            lease.close()


def _read_page(database, mapped_run_id, operation_id, source_fingerprint, cursor):
    from tldw_chatbook.Agents.run_log_paging import _SEGMENT, _load_page
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
    )
    from tldw_chatbook.Backup_Recovery.profile_paths import (
        canonical_console_paths,
        lexical_path,
        user_data_dir,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.private_sqlite import _ordinary_connections

    from .mobile_consolidation_receipts import _read_owner_receipt

    if (
        type(database) is not AgentRunsDB
        or type(mapped_run_id) is not str
        or not mapped_run_id
    ):
        raise ValueError("imported_history_native_owner_required")
    budget = _TraceReadBudget()
    budget.value((mapped_run_id, operation_id, source_fingerprint, cursor))
    with database.connection() as connection:
        lease = _ordinary_connections.get(connection)
        if (
            lease is None
            or lease.resource_close_failed
            or lease.execution_context(database.db_path)[1]
            in (None, ("bootstrap.unbound",))
        ):
            raise ValueError("imported_history_enrollment_required")
        participant = getattr(lease, "resource_participant", None)
        if (
            participant is not database._maintenance_participant
            or participant.owner_id != "db.agent_runs"
            or participant.repository() is not database
            or connection not in participant.connections
        ):
            raise ValueError("imported_history_database_not_admitted")
        if connection.in_transaction:
            raise ValueError("imported_history_requires_committed_receipt")
        connection.execute("BEGIN DEFERRED")
        try:
            result = _read_owner_receipt(
                connection, operation_id, source_fingerprint, "runs", budget
            )
            if result is None:
                raise ValueError("imported_history_membership_missing")
            _master, entries = result
            budget.container(len(entries) * 12)
            mappings = {
                row["data"]["original"]: row["data"]["mapped"]
                for row in entries
                if row["kind"] == "identity" and row["data"]["domain"] == "run"
            }
            roots = {
                row["data"]["root_id"]: row["data"]
                for row in entries
                if row["kind"] == "runlog_root"
            }
            members = [row["data"] for row in entries if row["kind"] == "runlog"]
            publications = [
                row["data"] for row in entries if row["kind"] == "history_publication"
            ]
            if (
                len(publications) != 1
                or mapped_run_id not in mappings.values()
                or connection.execute(
                    "SELECT count(*) FROM agent_runs WHERE id=?", (mapped_run_id,)
                ).fetchone()[0]
                != 1
            ):
                raise ValueError("imported_history_membership_missing")
            candidates = set()
            for member in members:
                owner = member["owner"]
                framed = member["framed_runs"]
                if owner is not None and mappings.get(owner[0]) != owner[1]:
                    _changed()
                if any(mappings.get(old) != new for old, new in framed) or len(
                    {old for old, _ in framed}
                ) != len(framed):
                    _changed()
                path = PurePosixPath(member["relative_path"])
                if framed and _SEGMENT.fullmatch(path.name) is None:
                    _changed()
                if owner is not None and path.parent.name != owner[0]:
                    _changed()
                if (owner is not None and owner[1] == mapped_run_id) or any(
                    new == mapped_run_id for _, new in framed
                ):
                    candidates.add((member["root_id"], str(path.parent)))
            if len(candidates) != 1:
                raise ValueError("imported_history_membership_missing_or_ambiguous")
            root_id, directory = next(iter(candidates))
            root = roots[root_id]
            original = next(
                old for old, new in mappings.items() if new == mapped_run_id
            )
        finally:
            connection.rollback()
    config = _selected_config(lease, database.db_path, budget)
    stores = CanonicalConsoleStores(*canonical_console_paths(config))
    if lexical_path(database.db_path) != stores.run_path:
        raise ValueError("imported_history_database_selection_changed")
    namespace = (
        user_data_dir(config)
        / "agent_history"
        / "imports"
        / stores.identity
        / source_fingerprint
    )
    publication = publications[0]
    if (
        publication["destination_identity"] != stores.identity
        or publication["source_fingerprint"] != source_fingerprint
        or publication["location"] != str(namespace)
    ):
        _changed()
    ancestor = Path(publication["artifact"]["path"])
    if (ancestor != namespace and ancestor not in namespace.parents) or user_data_dir(
        config
    ) not in ancestor.parents:
        _changed()
    ancestor_lease = storage_admission.acquire_storage(ancestor)
    ancestor_custody = storage_admission._CaptureFileDescriptors(None)
    try:
        if ancestor_lease.execution_context(ancestor) != lease.execution_context(
            database.db_path
        ):
            raise ValueError("imported_history_file_not_admitted")
        with native_files.pinned_directory(ancestor, _custody=ancestor_custody) as fd:
            observed = os.fstat(fd)
            if (observed.st_dev, observed.st_ino) != (
                publication["artifact"]["device"],
                publication["artifact"]["inode"],
            ):
                _changed()
        component = hashlib.sha256(root_id.encode()).hexdigest()
        if root["location"] != component or root["artifact"]["path"] != str(
            namespace / component
        ):
            _changed()
        selected = [row for row in members if row["root_id"] == root_id]
        for member in selected:
            location = (
                component
                if member["relative_path"] == ""
                else component + "/" + member["relative_path"]
            )
            if member["location"] != location or member["artifact"]["path"] != str(
                namespace / location
            ):
                _changed()
        pins = _pins(
            database,
            connection,
            namespace / component,
            selected,
            root["artifact"],
            directory,
            budget,
        )
        try:
            page = _load_page(pins, cursor, original, 50, 256_000, 4_500_000)
            pins.verify()
            return page
        finally:
            pins.retire()
    finally:
        if ancestor_custody.close_failed:
            _QUARANTINED.add((ancestor_lease, ancestor_custody))
        else:
            ancestor_custody.retire()
            ancestor_lease.close()
