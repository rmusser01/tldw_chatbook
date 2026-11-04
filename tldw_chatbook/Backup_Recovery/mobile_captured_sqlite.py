"""Fixed native captured SQLite reads; no semantic plan or import authority."""

import hashlib
import sqlite3
import stat
import zipfile
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from uuid import uuid4

from tldw_chatbook.Utils.platform_files import os

from . import archive_reader, control_records, native_files, sqlite_validation
from .archive_models import ArchiveManifest
from .journal import _mobile_active_owner, _mobile_state
from .limits import ArchiveLimits
from .mobile_consolidation_apply import (
    _CapturedMobile,
    _MobileCapturePartition,
    _SourceClaim,
    _verify_capture_archives,
)
from .models import DiscoverySelections, _mobile_pairs, _TraceReadBudget
from .storage_admission import MaintenanceSession, _CaptureFileDescriptors, _local

# These are selectors within an authenticated native declaration, never grants.
_FAMILIES = (
    ("chat", "db.chachanotes.primary", "chat.db", (81, 82, 83, 84, 85), 87),
    ("workspaces", "db.workspaces", "workspaces.db", (8, 9, 10), 11),
    ("runs", "db.agent_runs", "runs.db", (24, 25), 27),
)
_PAGE_ROWS = 1000
_PAGE_BYTES = 16 * 1024**2


def _captured_discovery_budget(session, child=None, *, _read_budget=None):
    """Look up the original coordinator for this reader's held discovery."""
    from .storage_admission import (
        _DiscoveryScope,
        _mobile_admission,
        _PreviewScope,
    )

    session._check()
    reader = getattr(_local, "mobile_sqlite_scope", None)
    receiver = None
    for native in session._scopes:
        if type(native) is _CapturedSQLite and native.active:
            if receiver is not None:
                raise ValueError("mobile_native_reader_changed")
            receiver = native
    admission = _mobile_admission(session)
    if reader is None and receiver is None:
        if (
            child is None
            or _read_budget is not None
            or type(child) is not _DiscoveryScope
            or child not in session._scopes
            or not child.active
            or getattr(_local, "discovery_scope", None) is not child
            or type(admission[3]) is dict
            and child in admission[3]
        ):
            raise ValueError("mobile_native_reader_changed")
        return None
    if (
        type(reader) is not _CapturedSQLite
        or reader is not receiver
        or reader.session is not session
    ):
        raise ValueError("mobile_native_reader_changed")
    native_budget = reader.read_budget
    if (
        admission is not reader.admission
        or admission[2] is not native_budget
        or type(native_budget) is not _TraceReadBudget
        or native_budget._parent is not admission[4]
        or _mobile_pairs.get(native_budget) is not admission[4]
        or admission[3] is not reader.discovery_bindings
        or type(admission[3]) is not dict
    ):
        raise ValueError("mobile_sqlite_admission_changed")
    # The issuer's fixed role is public aggregate or original private generic.
    budget = admission[4]
    if _read_budget is not None and _read_budget is not budget:
        raise ValueError("mobile_preview_budget_changed")
    holders = (reader,)
    if child is not None:
        if type(child) is not _DiscoveryScope:
            raise ValueError("mobile_native_discovery_child_changed")
        copies = child.sqlite_copies
        expected = admission[3].get(child)
        if (
            type(expected) is not tuple
            or len(expected) != 4
            or expected[0] is not reader
            or child.session is not session
            or child not in session._scopes
            or not child.active
            or getattr(_local, "discovery_scope", None) is not child
            or type(copies) is not _PreviewScope
            or not copies.active
            or expected[1] is not copies
            or expected[2] is not child.resources
            or expected[3] is not copies.resources
            or copies.native_scope is not child
            or child.read_budget is not budget
            or copies.read_budget is not budget
        ):
            raise ValueError("mobile_native_discovery_child_changed")
        child.check()
        copies.check()
        holders = (reader, child, copies)
    for holder in holders:
        custody = holder.custody
        if (
            type(custody) is not _CaptureFileDescriptors
            or custody.scope is not holder
            or not any(resource is custody for resource in holder.resources)
            or custody.close_failed
            or getattr(holder, "close_failed", False)
            or getattr(holder, "resource_close_failed", False)
            or any(
                getattr(resource, "close_failed", False)
                or getattr(resource, "resource_close_failed", False)
                for resource in holder.resources
            )
        ):
            raise ValueError("mobile_native_readers_not_retired")
    if child is None:
        reader.check()
    return budget


@contextmanager
def _borrowed_capture_discovery(session, partition, budget):
    """Retain this exact canonical member's native images through fresh project."""
    child = getattr(_local, "discovery_scope", None)
    if child is None:
        raise ValueError("mobile_native_discovery_child_changed")
    if _captured_discovery_budget(session, child) is not budget:
        raise ValueError("mobile_preview_budget_changed")
    reader = getattr(_local, "mobile_sqlite_scope", None)
    if (
        reader._canonical_archive is None
        or partition is not reader._canonical_archive[0]
    ):
        raise ValueError("mobile_native_discovery_child_changed")
    if child.config_sources != session._config_capture_sources:
        raise ValueError("capture_config_scope_changed")
    try:
        yield
    finally:
        # Preserve the original end-of-discovery check, including failure.
        child.sqlite_copies.verify_sources()


_row_data_marker = ContextVar("mobile_plan_row_data", default=None)


def _row_data(reads, budget=None):
    """Require agreement before a retained page or clone can be used."""
    if (
        type(reads) is not _CapturedSQLite
        or not reads.active
        or getattr(_local, "mobile_sqlite_scope", None) is not reads
        or not any(scope is reads for scope in reads.session._scopes)
        or "_row_data_owner" not in reads.__dict__
        or type(reads.resources) is not list
    ):
        raise ValueError("mobile_row_data_changed")
    registration = reads._row_data_owner
    marker = _row_data_marker.get()
    registered = None
    for resource in reads.resources:
        if type(resource) is _RowData:
            if registered is not None:
                raise ValueError("mobile_row_data_changed")
            registered = resource
    if registration is None and marker is None and registered is None:
        return None
    if (
        type(registration) is not tuple
        or len(registration) != 2
        or marker is not registration
        or type(registered) is not _RowData
        or registration[0] is not registered
        or registered.registration is not registration
        or registered.origin is not registration[1]
    ):
        raise ValueError("mobile_row_data_changed")
    registered._check(reads, budget)
    return registered


class _RowData:
    """Native writer's immutable pages; no verdict or read authority."""

    def __init__(self):
        raise TypeError("mobile_row_data_is_native_issued")

    def _check(self, reads, budget):
        from tldw_chatbook.DB.private_sqlite import _SQLiteDescriptorOutcome

        from .storage_admission import _CaptureLease, _mobile_admission

        origin = self.origin
        if type(origin) is not tuple or len(origin) != 18 or origin[0] is not self:
            raise ValueError("mobile_row_data_changed")
        binding = origin[3]
        if (
            self.closed
            or reads is not origin[1]
            or reads.resources is not origin[2]
            or reads.binding is not binding
            or reads.versions is not origin[4]
            or reads.admission is not origin[5]
            or reads.read_budget is not origin[6]
            or type(reads.read_budget) is not _TraceReadBudget
            or reads.read_budget._parent is not origin[7]
            or budget is not None
            and budget is not origin[6]
            or reads._connections is not origin[8]
            or reads.custody is not origin[9]
            or reads._source_archive is not origin[10]
            or reads._canonical_archive is not origin[11]
            or reads.owner_history is not origin[12]
            or reads.discovery_bindings is not origin[13]
            or reads.recovery_anchor is not origin[14]
            or reads.operation_id is not origin[15]
            or self.pages is not origin[17]
            or reads.session is not binding[0]
            or reads.captured is not binding[1]
            or reads.journal is not binding[2]
            or reads.captured.claim is not binding[3]
            or reads.captured.preview is not binding[4]
            or reads.captured.source is not binding[5]
            or reads.captured.canonical is not binding[6]
            or reads.captured.session is not reads.session
            or reads.captured.journal is not reads.journal
            or reads.session._mobile_journal is not reads.journal
            or not any(scope is reads.captured for scope in reads.session._scopes)
        ):
            raise ValueError("mobile_row_data_changed")
        admission = _mobile_admission(reads.session)
        if (
            admission is not origin[5]
            or admission[2] is not origin[6]
            or admission[4] is not origin[7]
            or admission[1] is not origin[12]
            or admission[3] is not origin[13]
            or _mobile_pairs.get(origin[6]) is not origin[7]
            or type(reads.custody) is not _CaptureFileDescriptors
            or reads.custody.scope is not reads
            or not any(resource is reads.custody for resource in reads.resources)
            or any(
                getattr(resource, "close_failed", False)
                or getattr(resource, "resource_close_failed", False)
                for resource in reads.resources
            )
            or len(reads._connections) != len(origin[16])
        ):
            raise ValueError("mobile_row_data_changed")
        for key, member, lease, outcome, stream in origin[16]:
            connection, restrictions, _, _, _ = member
            if (
                reads._connections.get(key) is not member
                or type(lease) is not _CaptureLease
                or lease.scope is not reads
                or lease.connection is not connection
                or lease.resource_closed
                or not any(resource is lease for resource in reads.resources)
                or type(outcome) is not _SQLiteDescriptorOutcome
                or lease.descriptor_outcome is not outcome
                or outcome.connection is not connection
                or outcome.connection_pending
                or outcome.connector_pending
                or outcome.duplicate_pending
                or outcome.duplicate is not None
                and not outcome.duplicate_closed
                or outcome.connection_close_attempted
                or outcome.connection_closed
                or restrictions.capture_scope is not reads
                or restrictions.capture_stream is not stream
                or stream.closed
            ):
                raise ValueError("mobile_row_data_changed")
            sqlite3.Connection.in_transaction.__get__(connection)
            if restrictions.expired():
                raise ValueError("sqlite_resource_limit")
        archive_reader._check(reads.cancel)

    def _selectors(self, purpose, family, table, offset, limit):
        if (
            type(purpose) is not str
            or type(family) is not str
            or type(table) is not str
            or type(offset) is not int
            or not 0 <= offset <= 2**31 - 1
            or type(limit) is not int
            or not 1 <= limit <= _PAGE_ROWS
        ):
            raise ValueError("mobile_sqlite_page_limit")
        for key, member, _, _, _ in self.origin[16]:
            if key[0] == purpose and key[1] == family:
                if table not in member[2]:
                    raise ValueError("mobile_sqlite_table_invalid")
                return
        raise ValueError("mobile_sqlite_member_absent")

    def page(self, purpose, family, table, *, offset, limit):
        reads, budget = self.origin[1], self.origin[6]
        if _row_data(reads, budget) is not self:
            raise ValueError("mobile_row_data_changed")
        self._selectors(purpose, family, table, offset, limit)
        budget.container(5)
        key = purpose, family, table, offset, limit
        result = self.pages.get(key)
        if result is None:
            result = reads.page(purpose, family, table, offset=offset, limit=limit)
            if _row_data(reads, budget) is not self:
                raise ValueError("mobile_row_data_changed")
            budget.value(result[0])
            budget.value(result[1])
            budget.container(2)
            budget.container(2)
            self.pages[key] = result
        return result

    def retire(self):
        # Ambiguous release waits for issuer-local finally; never clear self.pages.
        if self.closed:
            return
        self.closed = True
        reads = getattr(_local, "mobile_sqlite_scope", None)
        registration = _row_data_marker.get()
        if (
            type(reads) is not _CapturedSQLite
            or type(reads.session) is not MaintenanceSession
            or getattr(_local, "maintenance_session", None) is not reads.session
            or not any(scope is reads for scope in reads.session._scopes)
            or type(registration) is not tuple
            or len(registration) != 2
            or registration[0] is not self
            or getattr(reads, "_row_data_owner", None) is not registration
            or self.registration is not registration
            or self.origin is not registration[1]
        ):
            return
        origin = registration[1]
        if (
            type(origin) is tuple
            and len(origin) == 18
            and origin[0] is self
            and origin[1] is reads
            and origin[2] is reads.resources
            and type(reads.resources) is list
            and type(origin[17]) is dict
            and sum(type(resource) is _RowData for resource in reads.resources) == 1
            and any(resource is self for resource in reads.resources)
        ):
            origin[17].clear()


@contextmanager
def _plan_row_data(reads, budget):
    """Retain data only while the native writer derives its one pure plan."""
    from .storage_admission import _CaptureLease, _mobile_admission

    if _row_data(reads, budget) is not None:
        raise ValueError("mobile_row_data_changed")
    _check_plan_candidates(reads)
    admission = _mobile_admission(reads.session)
    if (
        type(budget) is not _TraceReadBudget
        or budget is not admission[2]
        or budget is not reads.read_budget
        or admission is not reads.admission
        or budget._parent is not admission[4]
        or _mobile_pairs.get(budget) is not admission[4]
    ):
        raise ValueError("mobile_row_data_changed")
    budget.container(4)
    budget.container()
    budget.container(len(reads._connections))
    members = []
    for key, member in reads._connections.items():
        lease = None
        for resource in reads.resources:
            if type(resource) is _CaptureLease and resource.connection is member[0]:
                if lease is not None:
                    raise ValueError("mobile_row_data_changed")
                lease = resource
        if lease is None:
            raise ValueError("mobile_row_data_changed")
        budget.container(5)
        members.append(
            (key, member, lease, lease.descriptor_outcome, member[1].capture_stream)
        )
    budget.container(len(members))
    budget.container(18)
    budget.container(4)
    owner = object.__new__(_RowData)
    owner.closed, owner.pages = False, {}
    owner.origin = (
        owner,
        reads,
        reads.resources,
        reads.binding,
        reads.versions,
        admission,
        budget,
        budget._parent,
        reads._connections,
        reads.custody,
        reads._source_archive,
        reads._canonical_archive,
        reads.owner_history,
        reads.discovery_bindings,
        reads.recovery_anchor,
        reads.operation_id,
        tuple(members),
        owner.pages,
    )
    budget.container(2)
    registration = owner, owner.origin
    owner.registration = registration
    resources, pages = reads.resources, owner.pages
    token = None
    try:
        reads._row_data_owner = registration
        resources.append(owner)
        token = _row_data_marker.set(registration)
        if _row_data(reads, budget) is not owner:
            raise ValueError("mobile_row_data_changed")
        yield
        if _row_data(reads, budget) is not owner:
            raise ValueError("mobile_row_data_changed")
        _check_plan_candidates(reads)
        if _row_data(reads, budget) is not owner:
            raise ValueError("mobile_row_data_changed")
    finally:
        owner.closed = True
        pages.clear()
        if getattr(reads, "_row_data_owner", None) is registration:
            reads._row_data_owner = None
        index = len(resources) - 1
        while index >= 0:
            if resources[index] is owner:
                del resources[index]
            index -= 1
        owner.origin = None
        owner.registration = None
        if token is not None and _row_data_marker.get() is registration:
            _row_data_marker.reset(token)


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
        _CapturedMobile._check_binding(captured)
        _SourceClaim._check_binding(captured.claim)
        with self.journal._locked(exclusive=False) as fd:
            records = self.journal._records(fd)
        prepared, witnesses = _mobile_state(records)
        control_records._mobile_pending(self.session, self.journal, prepared)
        _verify_capture_archives(records)
        from .mobile_native_operation import _require_readers_retired
        from .storage_admission import _mobile_admission, _mobile_check_witnesses

        admission = _mobile_admission(self.session)
        if (
            admission is not self.admission
            or admission[1] is not self.owner_history
            or admission[3] is not self.discovery_bindings
        ):
            raise ValueError("mobile_sqlite_admission_changed")
        if self.operation_id != self.journal.operation_id:
            raise ValueError("mobile_sqlite_operation_changed")
        if self.recovery_anchor is None:
            if _mobile_active_owner(records) is not None:
                raise ValueError("mobile_sqlite_operation_changed")
        else:
            if admission[1] or admission[2] is not self.read_budget:
                raise ValueError("mobile_sqlite_operation_changed")
            _require_readers_retired(self.session, _reader=self)
            if _owner_observation(records, self.read_budget) != self.recovery_anchor:
                raise ValueError("mobile_sqlite_operation_changed")
            from .mobile_consolidation import _path_witnesses

            if (
                prepared.source_fingerprint != captured.preview.source_fingerprint
                or _path_witnesses((captured.preview.source_dir,))[0][1]
                != captured.preview.source_witnesses
            ):
                raise ValueError("mobile_source_changed")
            _mobile_check_witnesses(records, witnesses, recovering_mode=True)
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

    def schema_catalog(self):
        """Return only exact fixed-family table/column metadata under live pins."""
        self.check()
        result, entries, size = [], 0, 0
        members = tuple(sorted(self._connections.items()))
        for _key, (_connection, restrictions, _tables, candidate, digest) in members:
            if restrictions.expired():
                raise ValueError("sqlite_resource_limit")
            _verify_candidate(self, restrictions, candidate, digest)
        for (purpose, family), (
            connection,
            restrictions,
            tables,
            candidate,
            digest,
        ) in members:
            for table in sorted(tables):
                cursor = connection.execute(
                    'PRAGMA table_xinfo("' + table.replace('"', '""') + '")'
                )
                try:
                    columns = tuple(row[1] for row in cursor if row[6] != 1)
                finally:
                    cursor.close()
                entries += 1 + len(columns)
                size += len(table.encode("utf-8")) + sum(
                    len(value.encode("utf-8")) for value in columns
                )
                if (
                    not columns
                    or entries > sqlite_validation._CATALOG_LIMIT
                    or size > sqlite_validation._CATALOG_BYTES
                ):
                    raise ValueError("sqlite_catalog_limit")
                result.append((purpose, family, table, columns))
        # No partially verified metadata escapes: every retained object and the
        # full native capture/claim/selector scope are checked before any return.
        for _key, (_connection, restrictions, _tables, candidate, digest) in members:
            if restrictions.expired():
                raise ValueError("sqlite_resource_limit")
            _verify_candidate(self, restrictions, candidate, digest)
        self.check()
        return tuple(result)

    def content_schema_catalog(self, selected):
        """Read a literal C-table subset while retaining every member's fences."""
        from .mobile_plan_manifest import _TABLES

        allowed = {s.table for s in _TABLES if s.family == "chat" and s.handler == "C"}
        if (
            type(selected) is not tuple
            or not 0 < len(selected) <= len(allowed)
            or any(type(table) is not str or table not in allowed for table in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("mobile_content_catalog_selector")
        self.check()
        result, entries, size = [], 0, 0
        members = tuple(sorted(self._connections.items()))
        for _key, (_connection, restrictions, _tables, candidate, digest) in members:
            if restrictions.expired():
                raise ValueError("sqlite_resource_limit")
            _verify_candidate(self, restrictions, candidate, digest)
        for (purpose, family), (
            connection,
            restrictions,
            tables,
            candidate,
            digest,
        ) in members:
            if family != "chat":
                continue
            if not set(selected) <= tables:
                raise ValueError("mobile_content_schema_mismatch")
            for table in sorted(selected):
                cursor = connection.execute(
                    'PRAGMA table_xinfo("' + table.replace('"', '""') + '")'
                )
                try:
                    columns = tuple(row[1] for row in cursor if row[6] != 1)
                finally:
                    cursor.close()
                entries += 1 + len(columns)
                size += len(table.encode("utf-8")) + sum(
                    len(value.encode("utf-8")) for value in columns
                )
                if (
                    not columns
                    or entries > sqlite_validation._CATALOG_LIMIT
                    or size > sqlite_validation._CATALOG_BYTES
                ):
                    raise ValueError("sqlite_catalog_limit")
                result.append((purpose, family, table, columns))
        # No partially verified metadata escapes: every retained object and the
        # full native capture/claim/selector scope are checked before any return.
        for _key, (_connection, restrictions, _tables, candidate, digest) in members:
            if restrictions.expired():
                raise ValueError("sqlite_resource_limit")
            _verify_candidate(self, restrictions, candidate, digest)
        self.check()
        return tuple(result)

    def authority_reference(self, kind, reference_id):
        """Prove one fixed captured raw reference without importing authority."""
        from .mobile_captured_authority import _authority_reference

        return _authority_reference(self, kind, reference_id)

    def identity_plan(self):
        """Reserve declared typed identities from this actual captured snapshot."""
        from .mobile_identity_plan import _plan_identities

        return _plan_identities(self)

    def plan_manifest(self):
        """Check fixed table/column ownership, retaining required pending handlers."""
        from .mobile_plan_manifest import _plan_manifest

        return _plan_manifest(self)

    def workspace_plan(self):
        """Transform Workspace rows without changing any captured or live owner."""
        from .mobile_workspace_plan import _plan_workspaces

        return _plan_workspaces(self)

    def content_graph_plan(self):
        """Transform C1 content fields; later handlers remain required."""
        from .mobile_content_graph import _plan_content_graph

        return _plan_content_graph(self)

    def content_settings_plan(self):
        """Plan C2a settings through fixed native types; successors remain pending."""
        from .mobile_content_settings import _plan_content_settings

        return _plan_content_settings(self)

    def content_workflow_plan(self):
        """Plan native C2b ledgers without replaying imported authority."""
        from .mobile_content_workflows import _plan_content_workflows

        return _plan_content_workflows(self)

    def sync_plan(self):
        """Transform fixed native sync frontiers without owning-write authority."""
        from .mobile_sync_plan import _plan_sync

        return _plan_sync(self)

    def integrated_plan(self):
        """Compose all fixed captured semantic plans; owner effects stay pending."""
        from .mobile_integrated_plan import _plan_integrated

        return _plan_integrated(self)

    def runs_preflight(self):
        """Qualify fixed all-owner history without granting native write authority."""
        from .mobile_runs_preflight import _plan_runs

        return _plan_runs(self)

    def trace_plan(self):
        """Plan fixed normalized history without granting owning-write authority."""
        from .mobile_trace_plan import _plan_traces

        return _plan_traces(self)

    def trace_inputs(self):
        """Read fixed native trace inputs without exposing retained connections."""
        from .mobile_trace_inputs import _read_trace_inputs

        return _read_trace_inputs(self)

    def content_plan(self):
        """Build the complete fixed C projection without import authority."""
        from .mobile_content_memory import _plan_content

        return _plan_content(self)

    def content_actor_plan(self):
        """Compose C2 logical actor history without asset or execution authority."""
        from .mobile_content_actors import _plan_content_actors

        return _plan_content_actors(self)

    def retire(self):
        was_active = self.active
        self.active = False
        custody = self.custody
        if custody is None:
            if self.resources or self.roots:
                raise ValueError("mobile_sqlite_custody_changed")
            return  # Initial custody allocation refused before any acquisition.
        if (
            type(custody) is not _CaptureFileDescriptors
            or custody.scope is not self
            or custody.close_failed
        ):
            raise ValueError("mobile_sqlite_custody_changed")
        if custody not in self.resources:
            if (
                not was_active
                and not self.resources
                and not self.roots
                and not custody.fds
                and not custody.directory_iterators
            ):
                return  # Maintenance rechecks an already physically retired scope.
            raise ValueError("mobile_sqlite_custody_changed")
        for resource in reversed(tuple(self.resources)):
            if resource is not custody:
                resource.retire()
        # Drain existing handles without unregistering the prepaid custodian.
        # The same owner must retain any uncertain cleanup descriptor below.
        while custody.directory_iterators:
            custody.close_directory_iterator(custody.directory_iterators[-1])
        while custody.fds:
            custody.close_descriptor(custody.fds[-1])
            custody.fds.pop()
        for root, identity in self.roots:
            if not root.exists() and identity is None:
                continue
            _retire_staging(self, root, identity)
        custody.retire()
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


def _retire_staging(scope, root, identity, *, _read_budget=None):
    """Remove only the witnessed private read tree after every reader retired."""
    shared_custody = type(scope) is _CapturedSQLite
    if shared_custody:
        custody = scope.custody
        if (
            type(custody) is not _CaptureFileDescriptors
            or custody.scope is not scope
            or custody not in scope.resources
            or custody.close_failed
        ):
            raise ValueError("mobile_sqlite_custody_changed")
    else:
        custody = _CaptureFileDescriptors(scope)

    def remove_children(parent):
        if _read_budget is None:
            names = os.listdir(parent)
        else:
            from .bootstrap import _bounded_names

            names = _bounded_names(
                parent, ArchiveLimits().members, _read_budget, custody
            )
        for name in names:
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
        if not shared_custody:
            custody.retire()


def _legacy_absent(reads, purpose, family, table):
    """Fixed absence proofs from an active, exactly authenticated native catalog."""
    if purpose != "legacy":
        return False
    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_plan_native_snapshot_required")
    versions = {
        ("chat", "character_portrait_performances"): (81, 82),
        ("chat", "character_portrait_performance_frames"): (81, 82),
        ("chat", "character_portrait_regions"): (81, 82, 83),
        ("workspaces", "workspace_binding_authority"): (8,),
        ("runs", "automatic_review_boundary_claims"): (24,),
        ("runs", "console_command_receipts"): (24, 25),
    }.get((family, table), ())
    if not any((purpose, family, version) in reads.versions for version in versions):
        return False
    reads.check()
    _, restrictions, tables, candidate, digest = reads._connections[purpose, family]
    if table in tables:
        raise ValueError("mobile_sqlite_schema_changed")
    if restrictions.expired():
        raise ValueError("sqlite_resource_limit")
    _verify_candidate(reads, restrictions, candidate, digest)
    reads.check()
    return True


def _runs24_receipts_absent(reads, purpose, family, table):
    # Retained private caller name; both authentic native predecessors lack receipts.
    return _legacy_absent(reads, purpose, family, table)


def _native_column_layouts(purpose, family, table, current, version):
    """Fixed semantic projections of already authenticated installed catalogs."""
    expected = current
    if purpose == "legacy":
        missing = ()
        if (family, table) == ("chat", "conversations") and version in (81, 82, 83, 84):
            missing = ("archived_at",)
        elif (family, table) == (
            "chat",
            "console_dispatch_checkpoints",
        ) and version <= 85:
            missing = ("mixin_snapshot_schema_version",)
        elif (family, table) == ("workspaces", "workspace_records") and version in (
            8,
            9,
        ):
            missing = ("archived_at", "archive_revision")
        elif (family, table) == ("runs", "automatic_wake_attempts") and version == 24:
            missing = ("review_boundary_id",)
        elif (family, table) == (
            "chat",
            "character_portrait_performances",
        ) and version == 83:
            missing = ("composition_recipe_json",)
        expected = tuple(c for c in current if c not in missing)
    layouts = [expected]
    if (family, table) == ("runs", "agent_definitions"):
        # Installed upgraded layout appends fallback settings to the old table.
        upgraded = tuple(c for c in expected if c != "fallback_models_json") + (
            "fallback_models_json",
        )
        layouts.append(upgraded)
    elif (family, table) == ("runs", "agent_runs"):
        upgraded = expected[:-3] + (
            "work_chain_id",
            "fallback_targets_json",
            "active_fallback_index",
        )
        layouts.append(upgraded)
    elif (family, table) == ("runs", "automatic_wake_attempts"):
        upgraded = expected[:7] + (
            "state",
            "created_at",
            "accepted_at",
            "completed_at",
            "cause",
            "message_ids_json",
            "attempt_kind",
        )
        if "review_boundary_id" in expected:
            upgraded += ("review_boundary_id",)
        layouts.append(upgraded)
    if (family, table) == (
        "runs",
        "automatic_wake_attempts",
    ) and "review_boundary_id" in expected:
        # Native fresh25 and upgraded25 append the two fields in opposite orders.
        assert expected[-2:] == ("attempt_kind", "review_boundary_id")
        layouts.append(expected[:-2] + ("review_boundary_id", "attempt_kind"))
    # Other exact pre-fallback native upgrade catalogs retained by recovery.
    layouts.extend(
        {
            ("runs", "agent_definitions"): (
                (
                    "id",
                    "name",
                    "description",
                    "instructions",
                    "tool_allowlist",
                    "model",
                    "enabled",
                    "deleted",
                    "created_at",
                    "updated_at",
                    "max_wall_seconds",
                    "provider",
                    "params_json",
                    "fallback_models_json",
                ),
                (
                    "id",
                    "name",
                    "description",
                    "instructions",
                    "tool_allowlist",
                    "model",
                    "enabled",
                    "max_wall_seconds",
                    "deleted",
                    "created_at",
                    "updated_at",
                    "provider",
                    "params_json",
                    "fallback_models_json",
                ),
            ),
            ("runs", "agent_runs"): (
                (
                    "id",
                    "conversation_id",
                    "parent_run_id",
                    "agent_kind",
                    "task",
                    "status",
                    "steps",
                    "result",
                    "budget",
                    "created_at",
                    "updated_at",
                    "assistant_message_id",
                    "agent_definition",
                    "definition_fingerprint",
                    "wake_delivered_at",
                    "resumed_from_run_id",
                    "spawn_event_id",
                    "budget_tokens",
                    "work_chain_id",
                    "resolved_provider",
                    "resolved_model",
                    "resolved_base_url",
                    "resolved_params_json",
                    "fallback_targets_json",
                    "active_fallback_index",
                ),
            ),
        }.get((family, table), ())
    )
    if (family, table) == (
        "runs",
        "automatic_wake_attempts",
    ) and "review_boundary_id" in expected:
        layouts.append(
            expected[:7]
            + (
                "state",
                "created_at",
                "accepted_at",
                "completed_at",
                "cause",
                "message_ids_json",
                "review_boundary_id",
                "attempt_kind",
            )
        )
    return tuple(layouts)


def _native_columns_match(purpose, family, table, actual, current, version):
    """Compare immutable metadata; this pure predicate grants no read authority."""
    return actual in _native_column_layouts(purpose, family, table, current, version)


def _check_plan_candidates(reads):
    """Recheck all fixed captured members before a semantic plan can escape."""
    reads.check()
    for _, restrictions, _, candidate, digest in reads._connections.values():
        if restrictions.expired():
            raise ValueError("sqlite_resource_limit")
        _verify_candidate(reads, restrictions, candidate, digest)
        if restrictions.expired():
            raise ValueError("sqlite_resource_limit")
    reads.check()


_SOURCE_HISTORY_PROBES = (
    (
        "chat",
        "character_portrait_performances",
        "SELECT id FROM character_portrait_performances LIMIT 1",
        "mobile_source_portrait_history_unsupported",
    ),
    (
        "chat",
        "character_portrait_performance_frames",
        "SELECT performance_id FROM character_portrait_performance_frames LIMIT 1",
        "mobile_source_portrait_history_unsupported",
    ),
    (
        "chat",
        "character_portrait_regions",
        "SELECT character_id FROM character_portrait_regions LIMIT 1",
        "mobile_source_portrait_history_unsupported",
    ),
    (
        "runs",
        "automatic_review_boundary_claims",
        "SELECT substr(boundary_id,1,128) FROM automatic_review_boundary_claims LIMIT 1",
        "mobile_source_review_boundary_history_unsupported",
    ),
    (
        "runs",
        "automatic_wake_attempts",
        "SELECT substr(id,1,128) FROM automatic_wake_attempts WHERE review_boundary_id IS NOT NULL LIMIT 1",
        "mobile_source_review_boundary_history_unsupported",
    ),
)


def _refuse_unmapped_source_history(reads):
    """Identity-only bounded probes; portrait blobs and review bodies stay unread."""
    for family, table, sql, reason in _SOURCE_HISTORY_PROBES:
        key = ("legacy", family)
        if key not in reads._connections or _legacy_absent(
            reads, "legacy", family, table
        ):
            continue
        if (
            table == "automatic_wake_attempts"
            and ("legacy", "runs", 24) in reads.versions
        ):
            continue
        reads.check()
        connection, restrictions, tables, candidate, digest = reads._connections[key]
        if table not in tables or restrictions.expired():
            raise ValueError("mobile_sqlite_schema_changed")
        _verify_candidate(reads, restrictions, candidate, digest)
        cursor = connection.execute(sql)
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        _verify_candidate(reads, restrictions, candidate, digest)
        reads.check()
        if row is not None:
            if reads.read_budget is not None:
                reads.read_budget.value(tuple(row))
            raise ValueError(reason + ":" + table)


def _members(scope):
    """One fresh installed discovery for both exact authenticated partitions."""
    scope.check()
    originals = tuple(
        row
        for row in (scope._source_archive, scope._canonical_archive)
        if row is not None
    )
    if not originals or scope._canonical_archive is None:
        raise ValueError("mobile_sqlite_partition_changed")
    if scope.read_budget is not None:
        scope.read_budget.container(len(originals) * 8)
    first = originals[0][0]
    for partition, document, sealed in originals:
        partition.check()
        source = (
            scope.captured.source
            if partition.purpose == "legacy"
            else scope.captured.canonical
        )
        if (
            type(partition) is not _MobileCapturePartition
            or partition.session is not scope.session
            or partition.claim is not scope.captured.claim
            or partition.journal is not scope.journal
            or partition.review_only
            or partition.selectors != first.selectors
            or partition.scope_digest != first.scope_digest
            or source is None
            or partition.destination != source.path
            or sealed.manifest_bytes != source.manifest_bytes
            or sealed.encrypted_source is None
            or sealed.encrypted_source.digest != source.digest
        ):
            raise ValueError("mobile_sqlite_partition_changed")
    inventory = scope.session._discover_capture_inventory(
        first.selectors,
        DiscoverySelections(),
        first.scope_digest,
        _mobile_partition=first,
    )
    schedules = {}
    for partition, document, sealed in originals:
        partition.check()
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
            declarations = sorted(
                (
                    row
                    for row in projected.items
                    if row.owner == owner and row.path == path
                ),
                key=lambda row: row.logical_id,
            )
            if not declarations:
                raise ValueError("mobile_sqlite_declaration_changed")
            selected = []
            expected = owner_versions.get(owner)
            supported = legacy if partition.purpose == "legacy" else (canonical,)
            for declared in declarations:
                producer = producers.get(declared.logical_id)
                if producer is None or (
                    producer.owner_id,
                    producer.status,
                    producer.dependencies,
                    producer.shared_group,
                ) != (
                    declared.owner,
                    declared.status,
                    declared.dependencies,
                    declared.shared_group,
                ):
                    raise ValueError("mobile_sqlite_inventory_changed")
                payload = payloads.get(declared.logical_id)
                if declared.status == "unused" and payload is None:
                    continue
                if (
                    declared.status != "included"
                    or payload is None
                    or payload.owner_id != owner
                    or expected not in supported
                ):
                    raise ValueError("mobile_sqlite_member_changed")
                selected.append(payload)
            if not selected:
                continue
            if len(selected) != len(declarations):
                raise ValueError("mobile_sqlite_declaration_changed")
            if len(declarations) > 1:
                from .mobile_consolidation import _resource_identity

                group = declarations[0].shared_group
                if not group or any(row.shared_group != group for row in declarations):
                    raise ValueError("mobile_sqlite_alias_unproven")
                identities = {_resource_identity(row.path)[1:] for row in declarations}
                if len(identities) != 1 or len(next(iter(identities))) != 2:
                    raise ValueError("mobile_sqlite_alias_unproven")
                if len({(row.size, row.sha256) for row in selected}) != 1:
                    raise ValueError("mobile_sqlite_alias_payload_changed")
            result.append((family, owner, expected, selected[0]))
        schedules[partition.purpose] = result
    scope.check()
    return schedules


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


def _owner_observation(records, budget):
    """Fingerprint exact native events, pre-admitting only new representations."""
    from .journal import _encoded

    if _mobile_active_owner(records) is None:
        return None
    if budget is None:
        raise ValueError("mobile_sqlite_original_budget_required")
    opened = next(
        row for row in reversed(records) if row.event == "mobile_owner_opened"
    )
    tail = records[-1]
    budget.container(5)
    budget.container(2)
    for row in (opened,) if tail is opened else (opened, tail):
        budget.container(12)
        budget.value(row.operation_id, text_copies=4)
        budget.value(row.previous, text_copies=4)
        budget.value(row.event, text_copies=4)
        budget.value(row.evidence, text_copies=4)
        budget.reserve(512)
        fingerprint = hashlib.sha256(_encoded(row)).hexdigest()
        if row is opened:
            opened_hash = fingerprint
        tail_hash = fingerprint
    return (
        opened.sequence,
        opened_hash,
        tail.sequence,
        tail.previous,
        tail_hash,
    )


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
    from .mobile_native_operation import _native_preopen
    from .storage_admission import _mobile_admission

    _native_preopen(captured, cancel)
    admission = _mobile_admission(session)
    if type(admission[2]) is not _TraceReadBudget:
        raise ValueError("mobile_native_budget_required")
    if admission[3] is None:
        admission[2].container()
        admission[3] = {}
    elif type(admission[3]) is not dict:
        raise ValueError("mobile_sqlite_admission_changed")
    with captured.journal._locked(exclusive=False) as fd:
        records = captured.journal._records(fd)
    if _mobile_active_owner(records) is not None and admission[1]:
        raise ValueError("mobile_sqlite_operation_changed")
    anchor = _owner_observation(records, admission[2])
    scope = object.__new__(_CapturedSQLite)
    scope.admission, scope.owner_history = admission, admission[1]
    scope.discovery_bindings = admission[3]
    scope.read_budget, scope.recovery_anchor = admission[2], anchor
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
    scope._row_data_owner = None
    scope._connections = {}
    scope._canonical_archive = None
    scope._source_archive = None
    scope.versions = ()
    session._scopes.append(scope)
    _local.mobile_sqlite_scope = scope
    scope.custody = None
    try:
        scope.custody = _CaptureFileDescriptors(scope)
        scope.check()
        with ExitStack() as connections:
            versions = []
            prepared = []
            authenticated = []
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
                    and row.active
                    and not row.review_only
                    and row.claim is captured.claim
                    and row.journal is captured.journal
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
                if purpose == "canonical":
                    scope._canonical_archive = (partition, document, sealed)
                else:
                    scope._source_archive = (partition, document, sealed)
                authenticated.append((purpose, partition, document, sealed, root))
            # Both original authenticated before-images are available before
            # current discovery; no target member is made a captured source.
            schedules = _members(scope)
            for purpose, partition, document, sealed, root in authenticated:
                members = schedules[purpose]
                # Exhaust the generator, including final sealed-byte verification,
                # before any connection's single native budget starts.
                prepared.extend(
                    (purpose, *member)
                    for member in _extract(scope, sealed, members, root)
                )
            if len(prepared) > 2 * len(_FAMILIES):
                raise ValueError("mobile_sqlite_member_limit")
            for purpose, family, owner_id, expected, payload, candidate in prepared:
                scope.check()
                connection, restrictions, owner = connections.enter_context(
                    sqlite_validation._open_installed_candidate(
                        owner_id,
                        candidate,
                        cancel=cancel,
                        _custody=scope.custody,
                    )
                )
                _verify_candidate(scope, restrictions, candidate, payload.sha256)
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
            _refuse_unmapped_source_history(scope)
            scope.check()
            yield scope
    finally:
        try:
            scope.retire()
        finally:
            _local.mobile_sqlite_scope = None
