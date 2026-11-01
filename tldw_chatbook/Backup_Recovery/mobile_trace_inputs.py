"""Fixed retained native trace consumers; immutable evidence, never authority."""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

from tldw_chatbook.Chat.citation_trace_identity import LocalCitationIdentityContext
from tldw_chatbook.Chat.console_trace_legacy import LegacyTraceNormalizer
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader
from tldw_chatbook.Chat.console_trace_projection import ConsoleTraceProjection
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

from .credentials import _citation_context_values, _unused_citation_reference
from .mobile_content_graph import _ContentRow
from .mobile_plan_manifest import _TABLES
from .mobile_trace_authority import _local_characters


@dataclass(frozen=True)
class _CitationContext:
    context_name: str
    identity: LocalCitationIdentityContext = field(repr=False)


@dataclass(frozen=True)
class _CitationSQL:
    context: _CitationContext = field(repr=False)
    no_fingerprint_history: bool


@dataclass(frozen=True)
class _MessageTraceInput:
    message_id: str
    # JSON preserves mapping/list distinctions without mutable native dictionaries.
    native: tuple[str, ...] = field(repr=False)
    legacy: tuple[str, ...] = field(repr=False)
    selected: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True)
class _TraceInputPartition:
    manifest_digest: str
    rows: tuple[_ContentRow, ...] = field(repr=False)
    messages: tuple[_MessageTraceInput, ...] = field(repr=False)
    citation: _CitationSQL = field(repr=False)
    local_characters: tuple = field(repr=False)


@dataclass(frozen=True)
class _TraceInputs:
    source: _TraceInputPartition
    canonical: _TraceInputPartition


class _RetainedTraceDatabase:
    """Native consumer adapter: each real cursor belongs to the retained scope."""

    def __init__(self, connection):
        self._connection = connection

    _local_authority_from_rows = staticmethod(
        CharactersRAGDB._local_authority_from_rows
    )

    def get_local_authority_id(self):
        return CharactersRAGDB.get_local_authority_id(self)

    @contextmanager
    def transaction(self):
        cursor = self._connection.cursor()
        cursor.row_factory = sqlite3.Row
        try:
            yield cursor
        finally:
            cursor.close()


def _call_json(call, budget):
    budget.value(call, text_copies=3)
    return json.dumps(
        asdict(call),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _check_trace_resources(reads):
    reads.check()
    if any(r.expired() for _, r, _, _, _ in reads._connections.values()):
        raise ValueError("sqlite_resource_limit")


def _check_trace_members(reads):
    from .mobile_captured_sqlite import _verify_candidate

    _check_trace_resources(reads)
    for _, restrictions, _, candidate, digest in reads._connections.values():
        _verify_candidate(reads, restrictions, candidate, digest)


def _read_trace_inputs(reads, *, _read_budget=None):
    from .mobile_captured_sqlite import _CapturedSQLite

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_trace_native_snapshot_required")
    reads.check()
    _check_trace_members(reads)
    specs = {
        s.table: s
        for s in _TABLES
        if s.family == "chat"
        and (s.handler == "T" or s.table in {"messages", "message_attachments"})
    }
    catalog = {
        (purpose, table): columns
        for purpose, family, table, columns in reads.schema_catalog()
        if family == "chat"
    }
    if any(
        catalog.get((purpose, table)) != spec.columns
        for purpose in ("legacy", "canonical")
        for table, spec in specs.items()
    ):
        raise ValueError("mobile_trace_schema_mismatch")
    budget = _TraceReadBudget() if _read_budget is None else _read_budget

    partitions = []
    for purpose, captured in (
        ("legacy", reads.captured.source),
        ("canonical", reads.captured.canonical),
    ):
        connection = reads._connections[(purpose, "chat")][0]
        rows = []
        for table, spec in specs.items():
            # This fixed operation already verified the complete installed
            # catalog and all six members. No caller can choose an identifier.
            cursor = connection.execute('SELECT * FROM "' + table + '"')
            try:
                columns = tuple(column[0] for column in cursor.description)
                if columns != spec.columns:
                    raise ValueError("mobile_trace_schema_mismatch")
                while True:
                    _check_trace_resources(reads)
                    batch = cursor.fetchmany(1)
                    if not batch:
                        break
                    for values in batch:
                        for value in values:
                            budget.value(value)
                        budget.container(len(columns) * 2)
                        row = dict(zip(columns, values, strict=True))
                        rows.append(
                            _ContentRow(
                                table,
                                tuple(row[c] for c in spec.identity_columns),
                                tuple(row.items()),
                            )
                        )
            finally:
                cursor.close()
        (key_id,) = _citation_context_values(connection)
        cursor = connection.cursor()
        cursor.row_factory = sqlite3.Row
        try:
            context_rows = cursor.execute(
                "SELECT * FROM rag_identity_context"
            ).fetchall()
        finally:
            cursor.close()
        authority = CharactersRAGDB._local_authority_from_rows(context_rows)
        identity = LocalCitationIdentityContext(
            profile_id=context_rows[0]["profile_id"],
            local_authority_id=authority,
            fingerprint_key_id=key_id,
        )
        citation = _CitationSQL(
            _CitationContext("default", identity),
            _unused_citation_reference(connection, key_id),
        )
        database = _RetainedTraceDatabase(connection)
        local_characters = _local_characters(database, _read_budget=budget)
        native = ConsoleTraceNativeReader(database)
        legacy = LegacyTraceNormalizer(database)
        messages = []
        for row in rows:
            if row.table != "messages":
                continue
            message_id = dict(row.values)["id"]
            _check_trace_resources(reads)
            native_calls = native.read_calls(message_id, _read_budget=budget)
            _check_trace_resources(reads)
            legacy_calls = legacy.read_calls(message_id, _read_budget=budget)
            _check_trace_resources(reads)
            # Both fixed native consumers are needed: native excludes legacy_snapshot.
            budget.container(len(native_calls) + len(legacy_calls))
            calls = native_calls + legacy_calls
            projection = ConsoleTraceProjection(
                legacy_reader=lambda _: (),
                normalized_reader=lambda _, fixed=calls: fixed,
                normalized_reads_enabled=True,
            )
            # Projection sorts/indexes these already bounded records; reserve
            # each possible native/legacy key and selected view before the call.
            budget.value(calls)
            selected = projection.read_calls(message_id)
            result = _MessageTraceInput(
                message_id,
                tuple(_call_json(call, budget) for call in native_calls),
                tuple(_call_json(call, budget) for call in legacy_calls),
                tuple(_call_json(call, budget) for call in selected),
            )
            messages.append(result)
        partitions.append(
            _TraceInputPartition(
                hashlib.sha256(captured.manifest_bytes).hexdigest(),
                tuple(rows),
                tuple(messages),
                citation,
                local_characters,
            )
        )
    _check_trace_members(reads)
    return _TraceInputs(*partitions)
