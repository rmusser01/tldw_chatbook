"""Literal native table/column ownership; required handlers never imply completion."""

import hashlib
import json
from dataclasses import dataclass

from tldw_chatbook.DB.mobile_import_authority import (
    _TABLES,
)
from tldw_chatbook.DB.mobile_import_authority import (
    _TableSpec as _TableSpec,  # noqa: PLC0414 - canonical compatibility export
)


@dataclass(frozen=True)
class _TableDisposition:
    purpose: str
    family: str
    table: str
    columns: tuple[str, ...]
    handler: str
    column_dispositions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _UnsupportedRecord:
    family: str
    table: str
    identity: tuple
    source_digest: str


@dataclass(frozen=True)
class _PlanManifest:
    tables: tuple[_TableDisposition, ...]
    unsupported_records: tuple[_UnsupportedRecord, ...]
    pending_handlers: tuple[str, ...] = ("C", "T", "S", "A-runs")

    def require_complete(self):
        if self.unsupported_records:
            raise ValueError("mobile_plan_unsupported_populated")
        if self.pending_handlers:
            raise ValueError("mobile_plan_required_handlers_pending")


def _disposition(family, table, columns, canonical, *, reads=None):
    matches = [
        row
        for row in _TABLES
        if row.family == family
        and row.table == table
        and (canonical or not row.canonical_only)
    ]
    if len(matches) != 1:
        raise ValueError(
            "mobile_plan_table_disposition_missing:" + family + ":" + table
        )
    spec = matches[0]
    if reads is None:
        matches = columns == spec.columns
    else:
        from .mobile_captured_sqlite import _native_columns_match

        purpose = "canonical" if canonical else "legacy"
        version = next(v for p, f, v in reads.versions if (p, f) == (purpose, family))
        matches = _native_columns_match(
            purpose, family, table, columns, spec.columns, version
        )
    if not matches:
        raise ValueError(
            "mobile_plan_column_disposition_missing:" + family + ":" + table
        )
    return spec


def _plan_manifest(reads):
    from .mobile_captured_sqlite import _CapturedSQLite, _runs24_receipts_absent
    from .mobile_identity_plan import _rows

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_plan_native_snapshot_required")
    reads.check()
    dispositions, unsupported = [], []
    catalog = {
        (purpose, family, table): columns
        for purpose, family, table, columns in reads.schema_catalog()
    }
    for (purpose, family), (_, _, tables, _, _) in sorted(reads._connections.items()):
        canonical = purpose == "canonical"
        expected = {
            row.table
            for row in _TABLES
            if row.family == family
            and (canonical or not row.canonical_only)
            and not _runs24_receipts_absent(reads, purpose, family, row.table)
        }
        if expected != tables:
            raise ValueError("mobile_plan_table_disposition_missing:" + family)
        for table in sorted(tables):
            columns = catalog[purpose, family, table]
            spec = _disposition(family, table, columns, canonical, reads=reads)
            dispositions.append(
                _TableDisposition(
                    purpose,
                    family,
                    table,
                    columns,
                    spec.handler,
                    tuple((column, spec.handler) for column in columns),
                )
            )
            if not canonical and spec.handler == "unsupported-populated":
                for row in _rows(reads, purpose, family, table):
                    # No body or arbitrary recursive JSON is an identity or receipt.
                    identity = tuple(row[column] for column in spec.identity_columns)
                    encoded = json.dumps(
                        tuple(row.items()),
                        ensure_ascii=False,
                        default=lambda value: {
                            "blob_sha256": hashlib.sha256(value).hexdigest()
                        },
                        separators=(",", ":"),
                    ).encode()
                    unsupported.append(
                        _UnsupportedRecord(
                            family, table, identity, hashlib.sha256(encoded).hexdigest()
                        )
                    )
    from .mobile_captured_sqlite import _check_plan_candidates

    _check_plan_candidates(reads)
    return _PlanManifest(tuple(dispositions), tuple(unsupported))
