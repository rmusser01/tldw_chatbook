"""Library pages read display names and exact identities, never full revisions."""

import json
import sqlite3
from contextlib import contextmanager
from uuid import UUID

import pytest

from Tests.Workflows.test_document_complexity import nested_raw
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Workflows.document_service import DocumentService


@pytest.fixture
def documents(tmp_path):
    db = WorkflowsDB(tmp_path / "summaries.sqlite3")
    try:
        yield DocumentService(db)
    finally:
        db.close()


def create_named(documents, index, name):
    return documents.create(
        json.dumps(
            {
                "name": name,
                "steps": [],
                "metadata": {
                    "tldw_workflow": {
                        "format_version": 1,
                        "workflow_id": str(UUID(int=index + 1)),
                        "revision_id": str(UUID(int=index + 1001)),
                        "parent_revision_ids": [],
                    }
                },
                "opaque": "x" * 1024,
            }
        )
    )


def test_summary_search_spans_batches_and_preserves_exact_order(documents, monkeypatch):
    revisions = [create_named(documents, i, f"Straße {i:03}") for i in range(103)]
    descriptions = []
    transaction = documents._db.transaction

    @contextmanager
    def observe(*args, **kwargs):
        with transaction(*args, **kwargs) as cursor:
            yield cursor
            descriptions.append(tuple(item[0] for item in cursor.description))

    monkeypatch.setattr(documents._db, "transaction", observe)
    rows = documents.list_workflow_summaries(page_size=3, offset=99, query="STRASSE")
    assert rows == tuple(
        (f"Straße {i:03}", revisions[i].workflow_id, revisions[i].revision_id)
        for i in (99, 100, 101)
    )
    assert descriptions == [("name", "workflow_id", "revision_id")]
    assert len(documents.list_workflow_summaries()) == 20
    assert len(documents.list_workflow_summaries(page_size=100)) == 100
    assert documents.list_workflow_summaries(offset=103) == ()
    assert documents.list_workflow_summaries(query="' OR 1=1 --") == ()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"page_size": 0}, ValueError),
        ({"page_size": 101}, ValueError),
        ({"page_size": True}, ValueError),
        ({"offset": -1}, ValueError),
        ({"offset": False}, ValueError),
        ({"offset": "1"}, ValueError),
        ({"query": None}, TypeError),
        ({"query": b"text"}, TypeError),
        ({"query": "private-query-" + "x" * 500}, ValueError),
        ({"offset": 2**63}, ValueError),
    ],
)
@pytest.mark.parametrize("method", ["list_workflows", "list_workflow_summaries"])
def test_search_bounds_are_rejected_before_sql(
    documents, monkeypatch, kwargs, error, method
):
    def forbidden(*args, **options):
        pytest.fail("Invalid summary arguments must not reach SQLite")

    monkeypatch.setattr(documents._db, "transaction", forbidden)
    with pytest.raises(error) as caught:
        getattr(documents, method)(**kwargs)
    assert "private-query-" not in str(caught.value)


@pytest.mark.parametrize("method", ["list_workflows", "list_workflow_summaries"])
def test_search_accepts_exact_limit_without_normalizing_input(documents, method):
    name = " " + "ß" * 510 + " "
    revision = create_named(documents, 0, name)
    rows = getattr(documents, method)(query=name)
    assert rows == (
        ((name, revision.workflow_id, revision.revision_id),)
        if method == "list_workflow_summaries"
        else (revision,)
    )
    assert getattr(documents, method)(query=" " * 512) == ()


@pytest.mark.parametrize(
    "raw,name",
    [
        ('{"steps":[]}', "Untitled workflow"),
        ('{"name":null,"steps":[]}', "None"),
        ('{"name":"name survives","steps":[],"opaque":1e9999}', "name survives"),
        (nested_raw(1000), "Preserve me"),
    ],
    ids=[
        "missing-name",
        "null-name",
        "opaque-number",
        "depth1000",
    ],
)
def test_legacy_summary_does_not_admit_or_rewrite_saved_content(documents, raw, name):
    base = create_named(documents, 0, "Original")
    with documents._db.transaction() as cursor:
        cursor.execute(
            "UPDATE workflow_revisions SET definition_json=? WHERE revision_id=?",
            (raw, base.revision_id),
        )
    label = name
    assert documents.list_workflow_summaries() == (
        (label, base.workflow_id, base.revision_id),
    )
    assert documents.list_workflow_summaries(query=label.upper()) == (
        (label, base.workflow_id, base.revision_id),
    )
    assert documents.get_head(base.workflow_id).raw_json == raw


def test_summary_tracks_saved_head_not_pending_draft(documents):
    base = create_named(documents, 0, "Saved name")
    documents.put_draft(
        base.workflow_id,
        base.revision_id,
        documents.edit_field(base.raw_json, "/name", "Pending name"),
        1,
    )
    assert documents.list_workflow_summaries()[0][0] == "Saved name"
    saved = documents.save_revision(base.workflow_id, base.revision_id, 1)
    assert documents.list_workflow_summaries() == (
        ("Pending name", saved.workflow_id, saved.revision_id),
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(chr(0xD800), id="high-surrogate"),
        pytest.param(chr(0xDFFF), id="low-surrogate"),
        "name\x00suffix",
        "🦙",
    ],
)
def test_summary_preserves_admitted_unicode_names_and_saved_bytes(documents, name):
    unusual = create_named(documents, 0, name)
    ordinary = create_named(documents, 1, "Ordinary")
    assert documents.list_workflow_summaries() == (
        (name, unusual.workflow_id, unusual.revision_id),
        ("Ordinary", ordinary.workflow_id, ordinary.revision_id),
    )
    assert documents.list_workflow_summaries(query=name) == (
        (name, unusual.workflow_id, unusual.revision_id),
    )
    assert documents.list_workflows(query=name) == (unusual,)
    assert documents.get_head(unusual.workflow_id).raw_json == unusual.raw_json


@pytest.mark.parametrize("method", ["list_workflows", "list_workflow_summaries"])
@pytest.mark.parametrize(
    "query,offset,page_size,indices,batch_count",
    [
        ("STRASSE", 1, 2, (520, 920), 10),
        ("missing", 0, 2, (), 11),
        ("STRASSE", 0, 1, (20,), 1),
    ],
)
def test_search_scans_once_in_bounded_batches(
    documents, monkeypatch, method, query, offset, page_size, indices, batch_count
):
    revisions = [
        create_named(
            documents, i, f"Straße {i}" if i in (20, 520, 920) else f"Item {i}"
        )
        for i in range(1000)
    ]
    statements, batches, ticks = [], [], []
    transaction = documents._db.transaction

    class ObservedCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            statements.append(sql)
            return super().execute(sql, parameters)

        def fetchall(self):
            rows = super().fetchall()
            batches.append(("fetchall", len(rows)))
            return rows

        def fetchmany(self, size=1):
            rows = super().fetchmany(size)
            batches.append(("fetchmany", len(rows)))
            return rows

    @contextmanager
    def observe(*args, **kwargs):
        with transaction(*args, **kwargs) as cursor:
            connection = cursor.connection
            observed = connection.cursor(factory=ObservedCursor)
            connection.set_progress_handler(lambda: ticks.append(1) or 0, 100)
            try:
                yield observed
            finally:
                connection.set_progress_handler(None, 0)
                observed.close()

    monkeypatch.setattr(documents._db, "transaction", observe)
    rows = getattr(documents, method)(page_size=page_size, offset=offset, query=query)
    assert rows == tuple(
        (f"Straße {i}", revisions[i].workflow_id, revisions[i].revision_id)
        if method == "list_workflow_summaries"
        else revisions[i]
        for i in indices
    )
    scans = [sql for sql in statements if "FROM workflow_heads" in sql]
    print(
        {"query": query, "method": method, "scans": len(scans), "vm_ticks": len(ticks)}
    )
    assert len(scans) == 1, "A search must not restart traversal for each batch"
    assert len(batches) == batch_count, "Stop fetching when the matching page is full"
    assert batches and all(
        kind == "fetchmany" and size <= 100 for kind, size in batches
    )
