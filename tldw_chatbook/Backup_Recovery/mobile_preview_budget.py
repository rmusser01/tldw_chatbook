"""Materialization costs for the actual native first-enrollment read scopes.

These helpers do not issue scopes, admit paths, or relax a SQLite restriction.
An ordinary preview has no allowance; a mobile start carries its one allowance
through the existing preview and maintenance discovery issuers.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from contextvars import ContextVar

_completion_scope = ContextVar("mobile_completion_accounting", default=None)
_uncertain_completion_scopes = set()


class _CompletionObservation:
    """Accounting and retirement only; never installed as storage authority."""

    def __init__(self, budget, custody):
        budget.reserve(2048)
        self.read_budget = budget
        self.custody = custody
        self.pid = os.getpid()
        self.thread = threading.current_thread()
        self.active = True
        self.resources = []  # Existing installed in-memory reference resources.
        self.connections = {}
        self.uncertain_connections = []
        self.copy_buffer = None

    def _buffer(self):
        """Reuse the existing prepaid digest workspace without SQLite copying."""
        from tldw_chatbook.Utils.platform_files import os as native_os

        self.check()
        if not hasattr(native_os, "readv"):
            raise ValueError("preview_copy_buffer_unavailable")
        if self.copy_buffer is None:
            self.read_budget.reserve(65536 + 1024)
            self.copy_buffer = memoryview(bytearray(65536))
        return self.copy_buffer

    def check(self):
        from .storage_admission import _local

        if (
            not self.active
            or _completion_scope.get() is not self
            or self.pid != os.getpid()
            or self.thread is not threading.current_thread()
            or any(
                getattr(_local, name, None) is not None
                for name in (
                    "preview_scope",
                    "discovery_scope",
                    "capture_scope",
                    "maintenance_session",
                )
            )
        ):
            raise ValueError("mobile_completion_scope_inactive")

    def bind(self, connection, path):
        self.check()
        self.read_budget.reserve(512 + path_cost(path))
        lease = self.provenance(connection, path)
        if connection in self.connections:
            raise ValueError("mobile_completion_connection_changed")
        self.connections[connection] = lease

    def provenance(self, connection, path=None):
        from tldw_chatbook.DB.private_sqlite import (
            _ordinary_connections,
            _validated_owner_policy,
        )

        from . import storage_admission as storage
        from .profile_paths import lexical_path

        self.check()
        with storage._lock:
            try:
                lease = _ordinary_connections.get(connection)
            except TypeError:
                lease = None
            if (
                type(lease) is not storage.StorageLease
                or lease not in storage._live_leases
                or lease.resource_policy
                is not _validated_owner_policy("recovery.mobile.completion.read")
                or lease.resource_thread is not self.thread
                or lease.resource_close_failed
                or getattr(connection, "_admission_close_attempted", True)
                or path is not None
                and lease.resource_path != lexical_path(path)
            ):
                raise ValueError("mobile_completion_connection_changed")
            # The native getter refuses closed handles without running SQL.
            sqlite3.Connection.in_transaction.__get__(connection)
            return lease

    def retired(self, connection):
        """Accept only physical native close and ordinary lease retirement."""
        from tldw_chatbook.DB.private_sqlite import _ordinary_connections

        from . import storage_admission as storage

        try:
            sqlite3.Connection.in_transaction.__get__(connection)
        except sqlite3.ProgrammingError:
            pass
        else:
            self.uncertain(connection)
            raise ValueError("mobile_completion_connection_not_retired")
        lease = self.connections.get(connection)
        with storage._lock:
            if connection in _ordinary_connections or lease in storage._live_leases:
                self.uncertain(connection)
                raise ValueError("mobile_completion_connection_not_retired")
        self.connections.pop(connection, None)

    def uncertain(self, connection):
        from tldw_chatbook.DB.private_sqlite import _ordinary_connections

        from . import storage_admission as storage

        with storage._lock:
            lease = self.connections.get(connection) or _ordinary_connections.get(
                connection
            )
            if lease is not None:
                lease.resource_close_failed = True
        if not any(row is connection for row in self.uncertain_connections):
            self.uncertain_connections.append(connection)
        _uncertain_completion_scopes.add(self)


@contextmanager
def _completion_observation(budget, custody):
    """Borrow one original allowance/custody without preview/capture admission."""
    from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget

    from .storage_admission import _CaptureFileDescriptors, _local

    if (
        type(budget) is not _TraceReadBudget
        or type(custody) is not _CaptureFileDescriptors
    ):
        raise TypeError("mobile_completion_accounting_required")
    if _completion_scope.get() is not None or any(
        getattr(_local, name, None) is not None
        for name in (
            "preview_scope",
            "discovery_scope",
            "capture_scope",
            "maintenance_session",
        )
    ):
        raise ValueError("mobile_completion_scope_nested")
    scope = _CompletionObservation(budget, custody)
    token = _completion_scope.set(scope)
    try:
        yield scope
    finally:
        try:
            for resource in tuple(scope.resources):
                resource.retire()  # Only installed in-memory reference handles.
            if scope.connections or scope.uncertain_connections:
                raise ValueError("mobile_completion_connection_not_retired")
        except BaseException:
            _uncertain_completion_scopes.add(scope)
            raise
        finally:
            scope.active = False
            _completion_scope.reset(token)
            # Custody belongs to the complete coordinator, not this read phase.


def _current_completion():
    scope = _completion_scope.get()
    if type(scope) is not _CompletionObservation:
        raise ValueError("mobile_completion_scope_required")
    scope.check()
    return scope


def current():
    from .storage_admission import (
        _DiscoveryScope,
        _local,
        _mobile_coordinator_budget,
        _PreviewScope,
    )

    scope = getattr(_local, "discovery_scope", None) or getattr(
        _local, "preview_scope", None
    )
    completion = _completion_scope.get()
    if completion is not None:
        if scope is not None:
            raise ValueError("mobile_completion_scope_nested")
        return _current_completion()
    if scope is None:
        return None
    if type(scope) not in (_PreviewScope, _DiscoveryScope):
        raise ValueError("mobile_preview_scope_changed")
    scope.check()
    budget = scope.read_budget
    if budget is None:
        return None
    if type(scope) is _DiscoveryScope and (
        scope.sqlite_copies is None
        or scope.sqlite_copies.native_scope is not scope
        or scope.sqlite_copies.read_budget is not budget
        or _mobile_coordinator_budget(scope.session) is not budget
    ):
        raise ValueError("mobile_preview_budget_changed")
    return scope


def reserve(amount):
    scope = current()
    if scope is not None:
        scope.read_budget.reserve(amount)


def read_options(budget=None, custody=None):
    """Propagate this current issuer's allowance to nested native record readers."""
    scope = current()
    if scope is None:
        return budget, custody
    if budget is not None and budget is not scope.read_budget:
        raise ValueError("mobile_preview_budget_changed")
    return scope.read_budget, custody if custody is not None else scope.custody


def json_loads(raw, **options):
    import json

    scope = current()
    if scope is not None:
        scope.read_budget.json(raw)
    return json.loads(raw, **options)


def toml_loads(raw):
    import tomllib

    scope = current()
    if scope is not None:
        from .mobile_runs_selector import _charge_toml

        _charge_toml(raw, scope.read_budget)
    return tomllib.loads(raw.decode("utf-8"))


def children(root):
    """Keep native directory iteration and its close outcome in read custody."""
    scope = current()
    if scope is None:
        return sorted(root.iterdir())
    from .bootstrap import _bounded_names

    scope.read_budget.reserve(512 + path_cost(root, copies=2))
    with scope.custody.pinned_directory(root) as parent:
        names = _bounded_names(parent, 100_000, scope.read_budget, scope.custody)
    scope.read_budget.container(len(names))
    result = []
    for name in names:
        scope.read_budget.reserve(path_cost(root) + 16 * len(name))
        result.append(root / name)
    return result


def path_cost(path, *, copies=1):
    # Path.parts is already retained by the native pathlib object. Do not first
    # stringify it to discover the cost of allocating that string.
    return copies * (256 + 16 * sum(len(part) + 1 for part in path.parts))


def connection_budget(connection):
    scope = current()
    if scope is None:
        return None
    from .storage_admission import _CaptureLease

    if type(scope) is _CompletionObservation and connection in scope.connections:
        if scope.provenance(connection) is not scope.connections[connection]:
            raise ValueError("mobile_completion_connection_changed")
        return scope.read_budget
    if type(scope) is _CompletionObservation:
        # File-backed handles cannot borrow an installed memory reference lease.
        scope.read_budget.reserve(512)
        if not any(
            type(resource) is _CaptureLease
            and resource.connection is connection
            and not resource.resource_closed
            and not resource.resource_close_failed
            for resource in scope.resources
        ):
            raise ValueError("mobile_completion_connection_changed")
        if any(
            row[2] or row[1] not in {"main", "temp"}
            for row in connection.execute("PRAGMA database_list")
        ):
            raise ValueError("mobile_completion_reference_changed")
    if not any(
        type(resource) is _CaptureLease
        and resource.connection is connection
        and not resource.resource_closed
        and not resource.resource_close_failed
        for resource in scope.resources
    ):
        raise ValueError("mobile_preview_connection_changed")
    return scope.read_budget


def catalog_cost(connection):
    """Admit the fixed four-field catalog before any catalog text is fetched."""
    budget = connection_budget(connection)
    if budget is None:
        return
    budget.reserve(2048)
    count, size = connection.execute(
        "SELECT count(*),coalesce(sum(length(CAST(type AS BLOB))+"
        "length(CAST(name AS BLOB))+length(CAST(tbl_name AS BLOB))+"
        "coalesce(length(CAST(sql AS BLOB)),0)),0) FROM sqlite_schema"
    ).fetchone()
    if count > 10_000 or size > 8 * 1024**2:
        raise ValueError("sqlite_resource_limit")
    # Native row/string objects, retained list/tuple references and UTF-8 copies
    # used by the existing catalog byte-limit check.
    budget.reserve(256 + 448 * count + 5 * size)


def literal_catalog(connection, schemas):
    """Read all native fields; reuse installed text only after native equality.

    Candidate literals are representation hints, never accepted catalog records.
    Unknown SQL is fetched under its measured bound and returned unchanged for
    the existing exact validator to refuse. Nothing survives the native scope.
    """
    budget = connection_budget(connection)
    if budget is None:
        raise ValueError("mobile_preview_budget_required")
    budget.reserve(2048)
    count, size, names = connection.execute(
        "SELECT count(*),coalesce(sum(length(CAST(type AS BLOB))+length(CAST(name AS BLOB))+"
        "length(CAST(tbl_name AS BLOB))+coalesce(length(CAST(sql AS BLOB)),0)),0),"
        "coalesce(sum(length(CAST(type AS BLOB))+length(CAST(name AS BLOB))+"
        "length(CAST(tbl_name AS BLOB))),0) FROM sqlite_schema"
    ).fetchone()
    if count > 10_000 or size > 8 * 1024**2:
        raise ValueError("sqlite_resource_limit")
    budget.reserve(768 + 768 * count + 4 * names)
    literals = {}
    seen = set()
    for _, schema in schemas:
        for sql in schema:
            if sql in seen:
                continue
            budget.reserve(256)
            seen.add(sql)
            length = len(sql)
            if length not in literals:
                budget.reserve(128)
                literals[length] = []
            literals[length].append(sql)
    result = []
    for rowid, kind, name, table, length, storage_type, raw_size in connection.execute(
        "SELECT rowid,type,name,tbl_name,length(sql),typeof(sql),length(CAST(sql AS BLOB)) "
        "FROM sqlite_schema ORDER BY type,name"
    ):
        sql = None
        if storage_type != "null":
            matched = False
            if storage_type == "text":
                for literal in literals.get(length, ()):
                    budget.reserve(256)
                    if connection.execute(
                        "SELECT sql IS ? FROM sqlite_schema WHERE rowid=?",
                        (literal, rowid),
                    ).fetchone() == (1,):
                        sql = literal
                        matched = True
                        break
            if not matched:
                budget.reserve(256 + 4 * raw_size)
                sql = connection.execute(
                    "SELECT sql FROM sqlite_schema WHERE rowid=?", (rowid,)
                ).fetchone()[0]
        result.append((kind, name, table, sql))
    return tuple(result)


def scalar_cost(connection, count=1):
    budget = connection_budget(connection)
    if budget is not None:
        budget.reserve(512 + 128 * count)


def matching_schemas(connection, schemas):
    """Compare every ordered catalog SQL value without materializing its text.

    Installed validators already compare this exact non-NULL SQL sequence.
    Only fixed integer projections cross SQLite's boundary on this path; no
    validation is cached between connections or calls.
    """
    budget = connection_budget(connection)
    if budget is None:
        raise ValueError("mobile_preview_budget_required")
    budget.reserve(2048 + 64 * len(schemas))
    cursor = connection.cursor()
    try:
        count, size = cursor.execute(
            "SELECT count(*),coalesce(sum(length(CAST(sql AS BLOB))),0) "
            "FROM sqlite_schema WHERE sql IS NOT NULL"
        ).fetchone()
        if count > 10_000 or size > 8 * 1024**2:
            raise ValueError("sqlite_resource_limit")
        candidates = [row for row in schemas if len(row[1]) == count]
        parameters = [None, 0]
        for offset in range(count):
            if not candidates:
                break
            expected = candidates[0][1][offset]
            parameters[0], parameters[1] = expected, offset
            budget.reserve(256)
            equal = cursor.execute(
                "SELECT sql IS ? FROM sqlite_schema WHERE sql IS NOT NULL "
                "ORDER BY type,name LIMIT 1 OFFSET ?",
                parameters,
            ).fetchone() == (1,)
            if all(row[1][offset] == expected for row in candidates):
                if not equal:
                    candidates.clear()
                continue
            budget.reserve(256 + 64 * len(candidates))
            alternatives = dict.fromkeys(row[1][offset] for row in candidates)
            accepted = expected if equal else None
            if not equal:
                for alternative in alternatives:
                    if alternative == expected:
                        continue
                    parameters[0] = alternative
                    budget.reserve(256)
                    if cursor.execute(
                        "SELECT sql IS ? FROM sqlite_schema WHERE sql IS NOT NULL "
                        "ORDER BY type,name LIMIT 1 OFFSET ?",
                        parameters,
                    ).fetchone() == (1,):
                        accepted = alternative
                        break
            candidates = [row for row in candidates if row[1][offset] == accepted]
        budget.reserve(128 + 32 * len(candidates))
        return tuple(candidates)
    finally:
        cursor.close()


def table_cost(connection, table, columns, *, copies=1):
    """Preprice fixed native column reads, after the caller's catalog check.

    Only actual column identifiers are accepted. This reserves fetched SQLite
    rows, scalar values, payloads and retained references. Typed digest tuples,
    hex strings and JSON encodings have separate reservations at their actual
    construction sites. This helper never returns rows or authorizes a query.
    """
    budget = connection_budget(connection)
    if budget is None:
        return
    from tldw_chatbook.DB.sql_validation import escape_identifier

    budget.reserve(
        2048 + 128 * len(columns) + 32 * (len(table) + sum(map(len, columns)))
    )
    quoted = escape_identifier(table)
    # Membership is checked on the actual immutable candidate, not caller
    # catalog records. The aggregate produces only bounded integer scalars.
    if (
        connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is None
    ):
        raise ValueError("mobile_preview_table_changed")
    expressions = []
    for column in columns:
        name = escape_identifier(column)
        expressions.append(
            f"coalesce(sum(CASE typeof({name}) WHEN 'text' THEN "
            f"length(CAST({name} AS BLOB)) WHEN 'blob' THEN length({name}) ELSE 0 END),0)"
        )
    row = connection.execute(
        "SELECT count(*)," + ",".join(expressions) + f" FROM {quoted}"
    ).fetchone()
    if row[0] > 100_000:
        raise ValueError("mobile_inventory_limit")
    # Four-byte Unicode bounds fetched TEXT; BLOB bytes use the same conservative
    # bound. Scalars include row/retained references, not digest/JSON payloads.
    budget.reserve(
        copies * (256 + row[0] * (128 + 96 * len(columns)) + 4 * sum(row[1:]))
    )


def pragma_cost(connection, kind, name=None):
    """Prefetch only sizes/counts from the fixed native metadata PRAGMAs."""
    budget = connection_budget(connection)
    if budget is None:
        return
    fields = {
        "table_xinfo": (7, ("name", "type", "dflt_value")),
        "index_list": (5, ("name", "origin")),
        "index_xinfo": (6, ("name", "coll")),
        "foreign_key_list": (
            8,
            ("table", "from", "to", "on_update", "on_delete", "match"),
        ),
    }
    width, text = fields[kind]
    budget.reserve(2048 + 16 * len(name))
    projection = "+".join(
        f'coalesce(length(CAST("{field}" AS BLOB)),0)' for field in text
    )
    count, size = connection.execute(
        f"SELECT count(*),coalesce(sum({projection}),0) FROM pragma_{kind}(?)", (name,)
    ).fetchone()
    if count > 10_000:
        raise ValueError("sqlite_resource_limit")
    budget.reserve(256 + count * (128 + 96 * width) + 4 * size)


def integrity_cost(connection, catalog=None):
    """Admit native bounded integrity diagnostics before their fixed PRAGMAs."""
    budget = connection_budget(connection)
    if budget is not None:
        budget.reserve(2048)
        count, size = connection.execute(
            "SELECT count(*),coalesce(sum(length(CAST(quick_check AS BLOB))),0) "
            "FROM pragma_quick_check"
        ).fetchone()
        if count > 100:
            raise ValueError("sqlite_resource_limit")
        budget.reserve(256 + 192 * count + 4 * size)
        row = connection.execute(
            'SELECT length(CAST("table" AS BLOB)),length(CAST(parent AS BLOB)) '
            "FROM pragma_foreign_key_check LIMIT 1"
        ).fetchone()
        budget.reserve(512 + (4 * sum(value or 0 for value in row) if row else 0))
