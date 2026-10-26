"""Private C1 content transformations; partial fields never authorize inserts."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_chatbook.Canvas.repository import CanvasImportBatch

from .mobile_identity_plan import _IdentityPlan, _rows
from .mobile_plan_manifest import _TABLES

Scalar = str | int | float | bytes | None
DependencyIdentity = str | int | tuple[str | int, ...]


@dataclass(frozen=True)
class _ContentRow:
    table: str
    original: tuple[str | int, ...]
    values: tuple[tuple[str, Scalar], ...] = field(repr=False)
    pending_columns: tuple[str, ...] = ()
    required_proofs: tuple[str, ...] = ()
    dependencies: tuple[tuple[str, DependencyIdentity], ...] = ()


@dataclass(frozen=True)
class _ContentBlocker:
    table: str
    original: tuple[str | int, ...]
    reason: str
    dependency_domain: str | None = None
    dependency_id: DependencyIdentity | None = None


@dataclass(frozen=True)
class _ConversationStructure:
    original: str
    mapped: str
    # These are inputs to C2, never completed conversation payloads.
    references: tuple[tuple[str, Scalar], ...]


@dataclass(frozen=True)
class _ContentGraph:
    identities: _IdentityPlan
    rows: tuple[_ContentRow, ...] = field(repr=False)
    source_rows: tuple[_ContentRow, ...] = field(repr=False)
    blockers: tuple[_ContentBlocker, ...]
    conversation_inputs: tuple[_ConversationStructure, ...] = ()
    canvas_batches: tuple["CanvasImportBatch", ...] = field(default=(), repr=False)
    pending_handlers: tuple[str, ...] = ("C2", "C3", "T", "S", "A-runs")

    def require_complete(self) -> None:
        raise ValueError("mobile_content_handlers_pending")


class _UnprovedReference(ValueError):
    def __init__(self, domain, identity):
        super().__init__("content_dependency_unproven")
        self.domain, self.identity = domain, identity


class _NativeConflict(ValueError):
    pass


def _mapped(identities, domain, identity):
    try:
        return identities.mapped(domain, identity)
    except (KeyError, ValueError):
        raise _UnprovedReference(domain, identity) from None


class _GraphBuilder:
    """Fixed handlers share captured data and body-free diagnostics, not SQL."""

    def __init__(self, reads, identities, tables):
        self.identities = identities
        self.specs = {s.table: s for s in _TABLES if s.family == "chat"}
        catalog = {
            (purpose, table): columns
            for purpose, family, table, columns in reads.schema_catalog()
            if family == "chat"
        }
        if any(
            catalog.get((purpose, table)) != self.specs[table].columns
            for purpose in ("legacy", "canonical")
            for table in tables
        ):
            raise ValueError("mobile_content_schema_mismatch")
        self.source = {table: _rows(reads, "legacy", "chat", table) for table in tables}
        self.canonical = {
            table: _rows(reads, "canonical", "chat", table) for table in tables
        }
        self.rows = []
        self.blockers = []
        self.dependencies = []
        self.inputs = []
        self.by_id = {
            table: {r["id"]: r for r in rows}
            for table, rows in self.source.items()
            if "id" in self.specs[table].identity_columns
        }
        for table in tables:
            for row in (*self.source[table], *self.canonical[table]):
                if tuple(row) != self.specs[table].columns:
                    raise ValueError("mobile_content_schema_mismatch")

    def original(self, table, row):
        return tuple(row[c] for c in self.specs[table].identity_columns)

    def ref(self, domain, identity, *, optional=False):
        if identity is None and optional:
            return None
        value = _mapped(self.identities, domain, identity)
        self.dependencies.append((domain, identity))
        return value

    def emit(self, table, source, values, *, pending=(), proofs=()):
        if set(values) != set(self.specs[table].columns) - set(pending):
            raise ValueError("mobile_content_field_ownership_mismatch")
        self.rows.append(
            _ContentRow(
                table,
                self.original(table, source),
                tuple(values.items()),
                tuple(pending),
                tuple(proofs),
                tuple(self.dependencies),
            )
        )

    def handle(self, table, transform):
        for row in self.source[table]:
            self.dependencies = []
            try:
                transform(row)
            except _UnprovedReference as error:
                self.blockers.append(
                    _ContentBlocker(
                        table,
                        self.original(table, row),
                        "dependency_unproven",
                        error.domain,
                        error.identity,
                    )
                )
            except _NativeConflict:
                self.blockers.append(
                    _ContentBlocker(
                        table, self.original(table, row), "native_unique_conflict"
                    )
                )
            except (TypeError, ValueError, KeyError):
                self.blockers.append(
                    _ContentBlocker(
                        table, self.original(table, row), "native_content_invalid"
                    )
                )

    def close_dependencies(self):
        from .mobile_identity_plan import _DOMAINS

        domains = {
            d.table: d.name for d in _DOMAINS if d.family == "chat" and not d.repeated
        }
        composites = {
            "console_response_rule_revisions": "response_rule_revision",
            "console_response_rule_validations": "response_rule_validation",
        }

        def owned(table, original):
            if table in composites:
                return composites[table], original
            if table in domains:
                return domains[table], original[0]
            return None

        blocked = {
            identity
            for b in self.blockers
            if (identity := owned(b.table, b.original)) is not None
        }
        while True:
            retained, removed = [], []
            for row in self.rows:
                missing = next(
                    (ref for ref in row.dependencies if ref in blocked), None
                )
                if missing is None:
                    retained.append(row)
                    continue
                self.blockers.append(
                    _ContentBlocker(
                        row.table, row.original, "dependency_blocked", *missing
                    )
                )
                identity = owned(row.table, row.original)
                if identity is not None:
                    removed.append(identity)
            self.rows = retained
            if not removed:
                break
            blocked.update(removed)

    def retained(self):
        return tuple(
            _ContentRow(table, self.original(table, row), tuple(row.items()))
            for table, rows in self.source.items()
            for row in rows
        )


def _note_body(content, builder):
    from tldw_chatbook.DB.ChaChaNotes_DB import _NOTE_LINK_TARGET_RE

    if type(content) is not str:
        raise ValueError("note_body_invalid")

    def token(match):
        mapped = builder.ref("note", match.group(1))
        tail = match.group(0)
        start, end = match.start(1) - match.start(), match.end(1) - match.start()
        rewritten = tail[:start] + mapped + tail[end:]
        if _NOTE_LINK_TARGET_RE.fullmatch(rewritten) is None:
            raise _UnprovedReference("note", match.group(1))
        return rewritten

    return _NOTE_LINK_TARGET_RE.sub(token, content)


def _notes(builder):
    from tldw_chatbook.DB.ChaChaNotes_DB import extract_note_link_targets

    from .mobile_content_core import _revision

    links = builder.source["note_links"]

    def note(row):
        _revision(row)
        expected = set(
            extract_note_link_targets(row["content"], source_note_id=row["id"])
        )
        recorded = {
            link["target_note_id"]
            for link in links
            if link["source_note_id"] == row["id"]
        }
        if recorded != expected:
            raise ValueError("note_link_body_mismatch")
        # The installed partial UNIQUE index includes every non-NULL locator,
        # including deleted rows. This is exact BINARY DB equality, not a path check.
        locator = row["file_path_on_disk"]
        if locator is not None and any(
            other["file_path_on_disk"] == locator
            for other in builder.canonical["notes"]
        ):
            raise _NativeConflict()
        # File locators/hash/state remain historical DB fields. No file is opened.
        mapped = row | {
            "id": builder.ref("note", row["id"]),
            "content": _note_body(row["content"], builder),
        }
        builder.emit("notes", row, mapped)

    builder.handle("notes", note)

    def link(row):
        source = builder.by_id["notes"].get(row["source_note_id"])
        if source is None or row["target_note_id"] not in extract_note_link_targets(
            source["content"], source_note_id=source["id"]
        ):
            raise ValueError("note_link_body_mismatch")
        builder.emit(
            "note_links",
            row,
            {
                "source_note_id": builder.ref("note", row["source_note_id"]),
                "target_note_id": builder.ref("note", row["target_note_id"]),
            },
        )

    builder.handle("note_links", link)


def _plan_content_graph(reads):
    from .mobile_captured_sqlite import _CapturedSQLite
    from .mobile_content_canvas import _CANVAS_TABLES, _canvas
    from .mobile_content_core import _CORE_TABLES, _core
    from .mobile_content_rules import _RULE_TABLES, _rules

    if type(reads) is not _CapturedSQLite:
        raise ValueError("mobile_content_native_snapshot_required")
    reads.check()
    identities = reads.identity_plan()
    builder = _GraphBuilder(
        reads, identities, _CORE_TABLES + _CANVAS_TABLES + _RULE_TABLES
    )
    _notes(builder)
    _core(builder)
    canvas_batches = _canvas(builder)
    _rules(builder)
    builder.close_dependencies()
    # Native batches remain unavailable if their complete dependency graph failed.
    accepted = {
        table: {dict(row.values)["id"] for row in builder.rows if row.table == table}
        for table in ("canvas_documents", "canvas_revisions", "messages")
    }
    canvas_batches = tuple(
        batch
        for batch in canvas_batches
        if all(doc.canvas_id in accepted["canvas_documents"] for doc in batch.documents)
        and all(
            rev.revision_id in accepted["canvas_revisions"]
            and rev.origin_message_id in accepted["messages"]
            for rev in batch.revisions
        )
    )
    reads.check()
    return _ContentGraph(
        identities,
        tuple(builder.rows),
        builder.retained(),
        tuple(builder.blockers),
        tuple(builder.inputs),
        canvas_batches,
    )
