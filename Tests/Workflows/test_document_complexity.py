"""Authoring admission bounds must not destroy rejected or older raw content."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Workflows.authoring import WorkflowAuthoring
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.draft_session import DraftSession
from tldw_chatbook.Workflows.models import Draft, InvalidDraft, Revision

IDENTITY = {
    "format_version": 1,
    "workflow_id": "11111111-1111-4111-8111-111111111111",
    "revision_id": "22222222-2222-4222-8222-222222222222",
    "parent_revision_ids": [],
}


def definition(step_count=1):
    return {
        "name": "Preserve me",
        "steps": [
            {"id": f"step_{index}", "type": "prompt", "config": {"template": "Text"}}
            for index in range(step_count)
        ],
        "metadata": {"tldw_workflow": dict(IDENTITY)},
    }


def nested_raw(depth, *, location="metadata", container="array"):
    document = definition()
    target = (
        document["metadata"]
        if location == "metadata"
        else document["steps"][0]["config"]
    )
    target["opaque"] = "NESTED_VALUE"
    # root + metadata = 2 containers; root + steps + step + config = 4.
    levels = depth - (2 if location == "metadata" else 4)
    opening, closing = ("[", "]") if container == "array" else ('{"child":', "}")
    value = opening * levels + '"[{opaque}]"' + closing * levels
    return json.dumps(document).replace('"NESTED_VALUE"', value)


def node_raw(count):
    # Nine containers/values before the array elements: root, steps, metadata,
    # identity, its four values (including the parent array), and opaque array.
    document = {
        "steps": [],
        "metadata": {"tldw_workflow": dict(IDENTITY), "opaque": [None] * (count - 9)},
    }
    return json.dumps(document, separators=(",", ":"))


@pytest.mark.parametrize("count", [499, 500, 501])
def test_step_limit_admits_boundary_and_rejects_next_step(count):
    raw = json.dumps(definition(count))
    if count <= 500:
        assert len(DocumentService.project(raw)["steps"]) == count
    else:
        with pytest.raises(InvalidDraft, match="500"):
            DocumentService.project(raw)


@pytest.mark.parametrize("container", ["array", "object"])
@pytest.mark.parametrize("location", ["metadata", "config"])
@pytest.mark.parametrize("depth", [63, 64, 65, 1000])
def test_container_depth_is_bounded_before_projection_and_dependencies(
    depth, location, container
):
    raw = nested_raw(depth, location=location, container=container)
    for operation in (DocumentService.project, DocumentService.dependency_issues):
        if depth <= 64:
            operation(raw)
        else:
            with pytest.raises(InvalidDraft):
                operation(raw)


@pytest.mark.parametrize("count", [99999, 100000, 100001])
def test_node_limit_counts_opaque_values_and_containers(count):
    raw = node_raw(count)
    if count <= 100000:
        assert len(DocumentService.project(raw)["metadata"]["opaque"]) == count - 9
    else:
        with pytest.raises(InvalidDraft, match="100000"):
            DocumentService.project(raw)


def test_generated_identity_also_counts_toward_node_limit():
    # Exactly 100000 nodes before admission adds the portable identity.
    raw = '{"steps":[],"opaque":[' + ",".join("null" for _ in range(99997)) + "]}"
    with pytest.raises(InvalidDraft, match="100000"):
        DocumentService.project(raw)


@pytest.mark.parametrize("count", [99999, 100000])
def test_save_rechecks_node_budget_after_adding_parent(count, tmp_path):
    db = WorkflowsDB(tmp_path / "workflows.sqlite3")
    try:
        documents = DocumentService(db)
        base = documents.create(node_raw(count))
        draft = documents.put_draft(
            base.workflow_id, base.revision_id, base.raw_json, 1
        )
        if count == 100000:
            with pytest.raises(InvalidDraft, match="100000"):
                documents.save_revision(base.workflow_id, base.revision_id, 1)
            assert documents.get_head(base.workflow_id) == base
            assert documents.list_revisions(base.workflow_id) == (base,)
        else:
            saved = documents.save_revision(base.workflow_id, base.revision_id, 1)
            assert documents.project(saved.raw_json)["metadata"]["tldw_workflow"][
                "parent_revision_ids"
            ] == [base.revision_id]
        assert documents.get_draft(base.workflow_id, base.revision_id) == draft
    finally:
        db.close()


@pytest.mark.parametrize("source_kind", ["revision", "draft"])
@pytest.mark.parametrize("existing_target", [False, True])
def test_copy_rejects_over_budget_lineage_without_changing_target(
    source_kind, existing_target, tmp_path
):
    db = WorkflowsDB(tmp_path / "workflows.sqlite3")
    try:
        documents = DocumentService(db)
        source = documents.create(node_raw(100000))
        documents.put_draft(
            source.workflow_id,
            source.revision_id,
            documents.edit_field(
                source.raw_json, "/metadata/opaque", "[]", as_json=True
            ),
            1,
        )
        head = documents.save_revision(source.workflow_id, source.revision_id, 1)
        if existing_target:
            documents.put_draft(head.workflow_id, head.revision_id, head.raw_json, 1)
        previous = documents.get_draft(head.workflow_id, head.revision_id)
        if source_kind == "draft":
            source = documents.put_draft(
                source.workflow_id, source.revision_id, source.raw_json, 2
            )
        with pytest.raises(InvalidDraft, match="100000"):
            if source_kind == "draft":
                documents.copy_draft_to_head(source, head)
            else:
                documents.copy_revision_to_head(source, head)
        assert documents.get_head(head.workflow_id) == head
        assert documents.get_draft(head.workflow_id, head.revision_id) == previous
        assert documents.project(head.raw_json)["metadata"]["opaque"] == []
    finally:
        db.close()


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(definition(501)),
        nested_raw(1000),
        node_raw(100001),
        '{"steps":[',
    ],
    ids=["steps", "depth", "nodes", "incomplete"],
)
def test_rejected_raw_survives_reopen_with_prior_valid_projection(tmp_path, raw):
    path = tmp_path / "workflows.sqlite3"
    db = WorkflowsDB(path)
    try:
        documents = DocumentService(db)
        base = documents.create(json.dumps(definition()))
        prior_raw = documents.edit_field(base.raw_json, "/name", "Last valid edit")
        prior = documents.put_draft(base.workflow_id, base.revision_id, prior_raw, 1)
        rejected = documents.validate_draft(base, raw, 2, prior)
        assert rejected.error
        assert rejected.raw_text == raw
        assert rejected.last_valid_json == prior_raw
        durable = documents.put_draft(base.workflow_id, base.revision_id, raw, 2)
        assert durable == rejected
        assert documents.get_revision(base.workflow_id, base.revision_id) == base
    finally:
        db.close()
    reopened = WorkflowsDB(path)
    try:
        assert (
            DocumentService(reopened).get_draft(base.workflow_id, base.revision_id)
            == rejected
        )
    finally:
        reopened.close()


def test_near_16_mib_scalar_and_opaque_numbers_remain_lossless():
    document = definition()
    document["metadata"]["vendor"] = {"payload": "", "number": "OPAQUE_NUMBER"}
    raw = json.dumps(document, separators=(",", ":")).replace(
        '"OPAQUE_NUMBER"', "1.234567890123456789e9999"
    )
    payload = "x" * (16 * 1024 * 1024 - 1024 - len(raw))
    raw = raw.replace('"payload":""', '"payload":"' + payload + '"')
    base = Revision(IDENTITY["workflow_id"], IDENTITY["revision_id"], (), raw)
    prior = Draft(base.workflow_id, base.revision_id, 0, raw, raw, None)
    validated = DocumentService.validate_draft(base, raw + " ", 1, prior)
    assert validated.error is None
    assert validated.last_valid_json == raw + " "
    edited = DocumentService.edit_field(raw, "/name", "Updated")
    assert '"number":1.234567890123456789e9999' in edited
    projected = DocumentService.project(edited)
    assert projected["metadata"]["vendor"]["payload"] == payload
    assert projected["metadata"]["tldw_workflow"] == IDENTITY


async def test_existing_over_limit_saved_definition_exports_exact_bytes(tmp_path):
    owner = WorkflowAuthoring(lambda: tmp_path / "workflows.sqlite3")
    try:
        await owner.open()
        base = owner.documents.create(json.dumps(definition()))
        raw = nested_raw(1000)
        # Represent a revision written by the previous admission policy.
        with owner.documents._db.transaction() as cursor:
            cursor.execute(
                "UPDATE workflow_revisions SET definition_json = ? WHERE revision_id = ?",
                (raw, base.revision_id),
            )
        saved = owner.documents.get_revision(base.workflow_id, base.revision_id)
        target = tmp_path / "older.json"
        await owner.export_file(target, saved)
        assert target.read_bytes() == raw.encode("utf-8")
    finally:
        await owner.close()


@pytest.mark.parametrize("modified", [False, True], ids=["unchanged", "invalid-edit"])
async def test_legacy_owner_close_retains_exact_base_projection_as_invalid(modified):
    db = WorkflowsDB(Path(":memory:"))
    documents = DocumentService(db)
    owner = DraftSession(documents)
    try:
        base = documents.create(json.dumps(definition()))
        legacy = nested_raw(1000)
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE workflow_revisions SET definition_json = ? WHERE revision_id = ?",
                (legacy, base.revision_id),
            )
        await owner.select(base.workflow_id, base.revision_id)
        if modified:
            owner.update('{"incomplete":')
        await owner.close()
        durable = documents.get_draft(base.workflow_id, base.revision_id)
        assert durable == owner.current
        assert durable.raw_text == ('{"incomplete":' if modified else legacy)
        assert durable.last_valid_json == legacy
        assert durable.error is not None
        if not modified:
            assert "64 container levels" in durable.error
        with pytest.raises(InvalidDraft):
            documents.save_revision(
                base.workflow_id, base.revision_id, durable.generation
            )
        assert (
            documents.get_revision(base.workflow_id, base.revision_id).raw_json
            == legacy
        )
        reopened = DraftSession(documents)
        assert await reopened.select(base.workflow_id, base.revision_id) == durable
        await reopened.close()
    finally:
        db.close()


def test_legacy_projection_exception_requires_exact_durable_base_bytes():
    db = WorkflowsDB(Path(":memory:"))
    documents = DocumentService(db)
    try:
        base = documents.create(json.dumps(definition()))
        legacy = nested_raw(1000)
        with db.transaction() as cursor:
            cursor.execute(
                "UPDATE workflow_revisions SET definition_json = ? WHERE revision_id = ?",
                (legacy, base.revision_id),
            )
        with pytest.raises(InvalidDraft):
            documents.put_draft(
                base.workflow_id,
                base.revision_id,
                "{",
                0,
                last_valid_json=legacy + " ",
            )
        assert documents.get_draft(base.workflow_id, base.revision_id) is None
        assert (
            documents.get_revision(base.workflow_id, base.revision_id).raw_json
            == legacy
        )
    finally:
        db.close()
