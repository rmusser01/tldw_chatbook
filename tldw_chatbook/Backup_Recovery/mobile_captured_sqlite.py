"""Fixed native captured SQLite reads; no semantic plan or import authority."""

import hashlib
import stat
import zipfile
from contextlib import ExitStack, contextmanager
from uuid import uuid4

from tldw_chatbook.DB.private_sqlite import open_recovery_validation
from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, control_records, native_files, sqlite_validation
from .archive_models import ArchiveManifest
from .journal import _mobile_active_owner, _mobile_state
from .limits import ArchiveLimits
from .mobile_consolidation_apply import _CapturedMobile, _MobileCapturePartition
from .models import DiscoverySelections
from .storage_admission import MaintenanceSession, _CaptureFileDescriptors, _local

# These are selectors within an authenticated native declaration, never grants.
_FAMILIES = (
    ("chat", "db.chachanotes.primary", "chat.db", 81, 82),
    ("workspaces", "db.workspaces", "workspaces.db", 8, 9),
    ("runs", "db.agent_runs", "runs.db", 25, 26),
)
_PAGE_ROWS = 1000
_PAGE_BYTES = 16 * 1024**2


class _CapturedSQLite:
    """A live scope owns every connection; callers receive only immutable pages."""

    def __init__(self):
        raise TypeError("mobile_sqlite_is_native_issued")

    def check(self):
        if (
            type(self) is not _CapturedSQLite
            or not self.active
            or self not in self.session._scopes
            or getattr(_local, "mobile_sqlite_scope", None) is not self
        ):
            raise ValueError("mobile_sqlite_inactive")
        captured = self.captured
        if any(
            actual is not expected
            for actual, expected in zip(
                (
                    self.session,
                    captured,
                    self.journal,
                    captured.claim,
                    captured.preview,
                    captured.source,
                    captured.canonical,
                ),
                self.binding,
                strict=True,
            )
        ):
            raise ValueError("mobile_sqlite_binding_changed")
        self.session._check()
        if (
            captured not in self.session._scopes
            or captured.session is not self.session
            or captured.journal is not self.journal
            or self.session._mobile_journal is not self.journal
        ):
            raise ValueError("mobile_sqlite_inactive")
        captured.check()
        with self.journal._locked(exclusive=False) as fd:
            records = self.journal._records(fd)
        prepared, _ = _mobile_state(records)
        control_records._mobile_pending(self.session, self.journal, prepared)
        if (
            self.operation_id != self.journal.operation_id
            or _mobile_active_owner(records) is not None
        ):
            raise ValueError("mobile_sqlite_operation_changed")
        if any(
            getattr(resource, "close_failed", False)
            or getattr(resource, "resource_close_failed", False)
            for resource in self.resources
        ):
            raise ValueError("mobile_sqlite_retirement_uncertain")
        archive_reader._check(self.cancel)

    def page(self, purpose, family, table, *, offset=0, limit=100):
        """Read at most 1000 rows/16 MiB from one validated installed table.

        Returns the exact native column names and tuples of SQLite scalar values.
        No SQL, path, connection, factory, owner adapter, or cursor is accepted.
        """
        self.check()
        if (
            type(limit) is not int
            or not 1 <= limit <= _PAGE_ROWS
            or type(offset) is not int
            or not 0 <= offset <= 2**31 - 1
        ):
            raise ValueError("mobile_sqlite_page_limit")
        member = self._connections.get((purpose, family))
        if member is None:
            raise ValueError("mobile_sqlite_member_absent")
        connection, restrictions, tables, candidate, digest = member
        if type(table) is not str or table not in tables:
            raise ValueError("mobile_sqlite_table_invalid")
        if restrictions.expired():
            raise ValueError("sqlite_resource_limit")
        _verify_candidate(self, restrictions, candidate, digest)
        # Membership in the exact installed catalog precedes identifier quoting.
        cursor = connection.execute(
            'SELECT * FROM "' + table.replace('"', '""') + '" LIMIT ? OFFSET ?',
            (limit, offset),
        )
        try:
            columns = tuple(row[0] for row in cursor.description)
            rows, size = [], 0
            for row in cursor:
                size += sum(
                    len(value.encode("utf-8"))
                    if isinstance(value, str)
                    else len(value)
                    if isinstance(value, bytes)
                    else 8
                    for value in row
                )
                if size > _PAGE_BYTES:
                    raise ValueError("mobile_sqlite_page_limit")
                rows.append(tuple(row))
            result = columns, tuple(rows)
        finally:
            cursor.close()
        self.check()
        _verify_candidate(self, restrictions, candidate, digest)
        return result

    def retire(self):
        self.active = False
        for resource in reversed(tuple(self.resources)):
            resource.retire()
        for root, identity in self.roots:
            if not root.exists() and identity is None:
                continue
            _retire_staging(self, root, identity)
        self.roots.clear()


def _verify_candidate(scope, restrictions, candidate, digest):
    """Authenticate the retained SQLite pin and its strict named binding."""
    fd = restrictions.capture_stream.fileno()
    before = os.fstat(fd)
    identity = archive_reader._identity(before)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
        or before.st_mode & 0o077
    ):
        raise ValueError("mobile_sqlite_candidate_changed")
    with native_files.pinned_directory(
        candidate.parent, _custody=scope.custody
    ) as parent:

        def check_name():
            named = os.stat(candidate.name, dir_fd=parent, follow_symlinks=False)
            if archive_reader._identity(named) != identity:
                raise ValueError("mobile_sqlite_candidate_changed")

        check_name()
        hasher = hashlib.sha256()
        offset = os.lseek(fd, 0, os.SEEK_CUR)
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            remaining = before.st_size
            while remaining:
                archive_reader._check(scope.cancel)
                block = os.read(fd, min(remaining, 1024 * 1024))
                if not block:
                    raise ValueError("mobile_sqlite_candidate_changed")
                hasher.update(block)
                remaining -= len(block)
            if os.read(fd, 1):
                raise ValueError("mobile_sqlite_candidate_changed")
        finally:
            os.lseek(fd, offset, os.SEEK_SET)
        check_name()
        if (
            archive_reader._identity(os.fstat(fd)) != identity
            or hasher.hexdigest() != digest
        ):
            raise ValueError("mobile_sqlite_candidate_changed")
    # Rewalk ancestors too: a held parent alone cannot qualify its current name.
    with native_files.pinned_directory(
        candidate.parent, _custody=scope.custody
    ) as parent:
        check_name()


def _retire_staging(scope, root, identity):
    """Remove only the witnessed private read tree after every reader retired."""
    custody = _CaptureFileDescriptors(scope)

    def remove_children(parent):
        for name in os.listdir(parent):
            before = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if before.st_uid != os.geteuid() or before.st_mode & 0o077:
                raise ValueError("mobile_sqlite_staging_changed")
            if stat.S_ISDIR(before.st_mode):
                child = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                custody.fds.append(child)
                try:
                    actual = os.fstat(child)
                    if (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino):
                        raise ValueError("mobile_sqlite_staging_changed")
                    remove_children(child)
                finally:
                    native_files._close_owned(child, custody)
                named = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino):
                    raise ValueError("mobile_sqlite_staging_changed")
                os.rmdir(name, dir_fd=parent)
            elif stat.S_ISREG(before.st_mode):
                os.unlink(name, dir_fd=parent)
            else:
                raise ValueError("mobile_sqlite_staging_changed")

    try:
        with native_files.pinned_directory(root.parent, _custody=custody) as parent:
            fd = os.open(
                root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            custody.fds.append(fd)
            try:
                actual = os.fstat(fd)
                if identity != (actual.st_dev, actual.st_ino):
                    raise ValueError("mobile_sqlite_staging_changed")
                remove_children(fd)
            finally:
                native_files._close_owned(fd, custody)
            named = os.stat(root.name, dir_fd=parent, follow_symlinks=False)
            if identity != (named.st_dev, named.st_ino):
                raise ValueError("mobile_sqlite_staging_changed")
            os.rmdir(root.name, dir_fd=parent)
    finally:
        custody.retire()


def _members(scope, partition, document):
    """Requalify the actual partition's installed, admitted fixed declarations."""
    partition.check()
    inventory = scope.session._discover_capture_inventory(
        partition.selectors,
        DiscoverySelections(),
        partition.scope_digest,
        _mobile_partition=partition,
    )
    with scope.session._discovery_reads(private_sqlite=True):
        projected = partition.project(scope.session, inventory)
    producers = {row.logical_id: row for row in document.producer_inventory}
    payloads = {row.logical_id: row for row in document.files}
    owner_versions = {row.owner_id: row.schema_version for row in document.owners}
    result = []
    for family, owner, leaf, legacy, canonical in _FAMILIES:
        path = (
            scope.captured.preview.source_dir / leaf
            if partition.purpose == "legacy"
            else dict(
                zip(
                    (row[1] for row in _FAMILIES),
                    scope.captured.preview.destination.paths,
                    strict=True,
                )
            )[owner]
        )
        declarations = [
            row for row in projected.items if row.owner == owner and row.path == path
        ]
        if len(declarations) != 1:
            raise ValueError("mobile_sqlite_declaration_changed")
        declared = declarations[0]
        producer = producers.get(declared.logical_id)
        if producer is None or (
            producer.owner_id,
            producer.status,
            producer.dependencies,
        ) != (declared.owner, declared.status, declared.dependencies):
            raise ValueError("mobile_sqlite_inventory_changed")
        payload = payloads.get(declared.logical_id)
        if declared.status == "unused" and payload is None:
            continue
        expected = legacy if partition.purpose == "legacy" else canonical
        if (
            declared.status != "included"
            or payload is None
            or payload.owner_id != owner
            or owner_versions.get(owner) != expected
        ):
            raise ValueError("mobile_sqlite_member_changed")
        result.append((family, owner, expected, payload))
    return result


def _extract(scope, sealed, members, root):
    """Stream only authenticated fixed members, checking the same size/hash again."""
    with archive_reader._regular(sealed.path, _custody=scope.custody) as stream:
        central = archive_reader._central_preflight(stream, ArchiveLimits())
        with zipfile.ZipFile(stream) as archive:
            offsets = archive_reader._local_ranges(stream, archive.infolist(), central)
            for family, owner, version, payload in members:
                scope.check()
                candidate = root / (family + ".db")
                digest, size = hashlib.sha256(), 0
                with native_files.create_private_file(
                    candidate, _custody=scope.custody
                ) as fd:
                    info = archive.getinfo(payload.payload)
                    for chunk in archive_reader._member_chunks(
                        stream, info, offsets[info.filename], scope.cancel
                    ):
                        size += len(chunk)
                        if size > payload.size:
                            raise ValueError("mobile_sqlite_payload_changed")
                        archive_reader._space(root, len(chunk))
                        digest.update(chunk)
                        view = memoryview(chunk)
                        while view:
                            written = os.write(fd, view)
                            if written <= 0:
                                raise OSError("mobile_sqlite_write_failed")
                            view = view[written:]
                    if size != payload.size or digest.hexdigest() != payload.sha256:
                        raise ValueError("mobile_sqlite_payload_changed")
                yield family, owner, version, payload, candidate
    archive_reader.verify_sealed(sealed, scope.cancel, _custody=scope.custody)


@contextmanager
def _sqlite_reads(captured, *, password, cancel):
    if (
        type(captured) is not _CapturedMobile
        or type(captured.session) is not MaintenanceSession
    ):
        raise ValueError("mobile_sqlite_native_required")
    session = captured.session
    session._check()
    if captured not in session._scopes:
        raise ValueError("mobile_sqlite_inactive")
    captured.check()
    if any(
        getattr(_local, name, None) is not None
        for name in (
            "mobile_scope",
            "capture_scope",
            "discovery_scope",
            "mobile_history_scope",
            "mobile_sqlite_scope",
        )
    ):
        raise ValueError("mobile_sqlite_scope_busy")
    scope = object.__new__(_CapturedSQLite)
    scope.session, scope.captured, scope.journal = session, captured, captured.journal
    scope.operation_id = captured.journal.operation_id
    scope.binding = (
        session,
        captured,
        scope.journal,
        captured.claim,
        captured.preview,
        captured.source,
        captured.canonical,
    )
    scope.active, scope.resources, scope.roots, scope.cancel = True, [], [], cancel
    scope._connections = {}
    scope.versions = ()
    session._scopes.append(scope)
    _local.mobile_sqlite_scope = scope
    scope.custody = _CaptureFileDescriptors(scope)
    try:
        scope.check()
        with ExitStack() as connections:
            versions = []
            for purpose, source in (
                ("legacy", captured.source),
                ("canonical", captured.canonical),
            ):
                if source is None:
                    continue
                partitions = [
                    row
                    for row in session._scopes
                    if type(row) is _MobileCapturePartition
                    and row.purpose == purpose
                    and row.claim is captured.claim
                    and row.journal is captured.journal
                    and row.active
                    and not row.review_only
                ]
                if len(partitions) != 1 or partitions[0].destination != source.path:
                    raise ValueError("mobile_sqlite_partition_changed")
                partition = partitions[0]
                root = source.path.parent / (".mobile-read-" + uuid4().hex)
                scope.roots.append((root, None))
                native_files.create_private_directory(root, _custody=scope.custody)
                info = root.lstat()
                scope.roots[-1] = (root, (info.st_dev, info.st_ino))
                sealed = archive_reader.acquire(
                    source.path,
                    root,
                    ArchiveLimits(),
                    password,
                    cancel,
                    _custody=scope.custody,
                )
                if (
                    sealed.encrypted_source is None
                    or sealed.encrypted_source.digest != source.digest
                    or sealed.manifest_bytes != source.manifest_bytes
                ):
                    raise ValueError("mobile_sqlite_capture_changed")
                document = ArchiveManifest.model_validate_json(sealed.manifest_bytes)
                members = _members(scope, partition, document)
                for family, owner_id, expected, payload, candidate in _extract(
                    scope, sealed, members, root
                ):
                    connection, restrictions = connections.enter_context(
                        open_recovery_validation(
                            owner_id,
                            candidate,
                            writable=False,
                            with_restrictions=True,
                            cancel=cancel,
                            _custody=scope.custody,
                        )
                    )
                    _verify_candidate(scope, restrictions, candidate, payload.sha256)
                    owner = sqlite_validation._installed_owner(owner_id)
                    issues, version = sqlite_validation._check(
                        connection, owner, owner.schema_policy(), restrictions
                    )
                    if issues or version != expected:
                        raise ValueError(
                            issues[0] if issues else "mobile_sqlite_schema_changed"
                        )
                    tables = frozenset(
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_schema WHERE type='table'"
                        )
                    )
                    _verify_candidate(scope, restrictions, candidate, payload.sha256)
                    scope._connections[purpose, family] = (
                        connection,
                        restrictions,
                        tables,
                        candidate,
                        payload.sha256,
                    )
                    versions.append((purpose, family, version))
            scope.versions = tuple(versions)
            scope.check()
            yield scope
    finally:
        try:
            scope.retire()
        finally:
            _local.mobile_sqlite_scope = None
