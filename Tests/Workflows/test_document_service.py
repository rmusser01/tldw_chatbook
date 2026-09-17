"""Real-store document tests: loss, stale writes and partial saves are bugs."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from decimal import Decimal
from threading import Barrier
from uuid import UUID, uuid4

import pytest

from Tests.Workflows.helpers import prompt_definition
from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
from tldw_chatbook.Workflows.document_service import DocumentService
from tldw_chatbook.Workflows.models import DraftConflict, InvalidDraft, RevisionConflict

UUID_ALIAS_CASES = [
    pytest.param(
        "BA9E62AA-9859-4D42-98DC-26E3B10CDBA0",
        "B42D45FB-DC91-4645-A6D9-A72C589282BD",
        id="uppercase",
    ),
    pytest.param(
        "ba9e62aa98594d4298dc26e3b10cdba0",
        "b42d45fbdc914645a6d9a72c589282bd",
        id="hex",
    ),
    pytest.param(
        "{ba9e62aa-9859-4d42-98dc-26e3b10cdba0}",
        "{b42d45fb-dc91-4645-a6d9-a72c589282bd}",
        id="braces",
    ),
    pytest.param(
        "urn:uuid:ba9e62aa-9859-4d42-98dc-26e3b10cdba0",
        "urn:uuid:b42d45fb-dc91-4645-a6d9-a72c589282bd",
        id="urn",
    ),
]


@pytest.fixture
def documents(tmp_path):
    db = WorkflowsDB(tmp_path / "workflows.sqlite3")
    try:
        yield DocumentService(db)
    finally:
        db.close()


def test_workflow_pages_and_unicode_search_reach_off_page_heads(documents):
    expected = []
    for index in range(23):
        raw = prompt_definition()
        raw["metadata"]["tldw_workflow"]["workflow_id"] = str(uuid4())
        raw["metadata"]["tldw_workflow"]["revision_id"] = str(uuid4())
        raw["name"] = f"Straße {index}"
        expected.append(documents.create(json.dumps(raw)))
    expected.sort(key=lambda revision: revision.workflow_id)
    assert documents.list_workflows() == tuple(expected[:20])
    assert documents.list_workflows(offset=20) == tuple(expected[20:])
    assert documents.list_workflows(page_size=3, offset=3, query="STRASSE") == tuple(
        expected[3:6]
    )
    assert documents.list_workflows(query="' OR 1=1 --") == ()
    assert documents.get_head(expected[-1].workflow_id) == expected[-1]
    assert documents.get_head("absent") is None


def test_history_and_local_draft_pages_preserve_all_exact_identities(documents):
    base = documents.create(json.dumps(prompt_definition()))
    history = [base]
    drafts = []
    for index in range(5):
        drafts.append(
            documents.put_draft(
                base.workflow_id,
                base.revision_id,
                documents.edit_field(base.raw_json, "/name", f"Revision {index}"),
                1,
            )
        )
        base = documents.save_revision(base.workflow_id, base.revision_id, 1)
        history.append(base)
    assert documents.list_revisions(base.workflow_id, page_size=2, offset=2) == tuple(
        history[2:4]
    )
    ordered = sorted(drafts, key=lambda draft: draft.base_revision_id)
    assert documents.list_drafts(base.workflow_id, page_size=2, offset=2) == tuple(
        ordered[2:4]
    )
    assert documents.list_revisions(base.workflow_id, offset=6) == ()
    assert documents.list_drafts(base.workflow_id, offset=5) == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"page_size": 0},
        {"page_size": 101},
        {"page_size": True},
        {"page_size": 1.5},
        {"offset": -1},
        {"offset": False},
        {"offset": "1"},
    ],
)
def test_collection_pages_reject_invalid_bounds_before_sql(
    documents, monkeypatch, kwargs
):
    def no_transaction(*args, **options):
        pytest.fail("Invalid pagination must not reach SQLite")

    monkeypatch.setattr(documents._db, "transaction", no_transaction)
    for method, args in (
        (documents.list_workflows, ()),
        (documents.list_revisions, ("id",)),
        (documents.list_drafts, ("id",)),
    ):
        with pytest.raises(ValueError):
            method(*args, **kwargs)


def test_legacy_fragment_edit_cannot_adopt_sibling_injection(documents):
    base = documents.create(json.dumps(prompt_definition()))
    with pytest.raises(InvalidDraft):
        documents.edit_field(
            base.raw_json,
            "/steps/0/retry",
            '0,"injected":true',
            as_json=True,
            allow_invalid_fragment=True,
        )


def test_sqlite_integer_maximum_is_accepted_for_generations_and_paging(documents):
    maximum = 2**63 - 1
    base = documents.create(json.dumps(prompt_definition()))
    draft = documents.put_draft(
        base.workflow_id, base.revision_id, base.raw_json, maximum
    )
    assert draft.generation == maximum
    assert documents.get_draft(base.workflow_id, base.revision_id) == draft
    assert documents.list_workflows(offset=maximum) == ()
    assert documents.list_workflow_summaries(offset=maximum) == ()
    assert documents.list_workflow_summaries(offset=maximum, query="missing") == ()
    assert documents.list_revisions(base.workflow_id, offset=maximum) == ()
    assert documents.list_drafts(base.workflow_id, offset=maximum) == ()


@pytest.mark.parametrize(
    "pointer,want",
    [
        ("/nested/items/0/repeat", '"first"'),
        ("/nested/items/1/a~1b~0c/repeat", '"second"'),
        ("/nested/items/1/a~1b~0c/numbers/2", "-0.000"),
    ],
)
def test_json_location_uses_nested_array_and_escaped_pointer_not_member_search(
    documents, pointer, want
):
    raw = '{\n "repeat":"decoy",\n "nested":{"items":[{"repeat":"first"},{"a/b~c":{"repeat":"second","numbers":[1,1e999999999,-0.000]}}]}\n}'
    start, end = documents.json_location(raw, pointer)
    assert start[0] == end[0] == 2
    assert raw.splitlines()[start[0]][start[1] : end[1]] == want
    assert documents.json_location(raw, "/nested/absent") is None


def test_successful_reorder_dependencies_include_consumers_and_explicit_routes(
    documents,
):
    definition = prompt_definition()
    definition["steps"] += [
        {"id": "side", "type": "prompt", "config": {"template": "independent"}},
        {"id": "end", "type": "prompt", "config": {"template": "done"}},
    ]
    definition["steps"][0]["on_success"] = "end"
    base = documents.create(json.dumps(definition))
    moved = documents.edit_steps(base.raw_json, "move", "finish", offset=1)
    references = documents.step_dependencies(moved)
    assert any(
        issue.pointer == "/steps/2/config/template" and "prepare.text" in issue.message
        for issue in references
    )
    assert any(
        issue.pointer == "/steps/0/on_success" and "end" in issue.message
        for issue in references
    )


@pytest.mark.parametrize(
    "token",
    ["9" * 5000, "1e999999999", "-0", "-0.000"],
    ids=["huge", "exponent", "zero", "decimal"],
)
def test_history_copy_is_atomic_clean_target_only_and_preserves_old_draft(
    documents, token
):
    base = documents.create(
        json.dumps(prompt_definition())[:-1] + ',"opaque":' + token + "}"
    )
    old = documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 1)
    head = documents.save_revision(base.workflow_id, base.revision_id, 1)
    old = documents.put_draft(
        base.workflow_id, base.revision_id, '{"old unfinished":', 2
    )
    copied = documents.copy_revision_to_head(base, head)
    assert documents.field_text(copied.raw_text, "/opaque") == token
    assert documents.get_draft(base.workflow_id, base.revision_id) == old
    dirty = documents.put_draft(
        head.workflow_id,
        head.revision_id,
        '{"target unfinished":',
        copied.generation + 1,
    )
    with pytest.raises(DraftConflict):
        documents.copy_revision_to_head(base, head)
    assert documents.get_draft(head.workflow_id, head.revision_id) == dirty
    assert documents.get_draft(base.workflow_id, base.revision_id) == old


@pytest.mark.parametrize("text", ["", "tru", "{", '0,"injected":true'])
def test_fragment_proof_preserves_bytes_projection_and_protected_replays(
    documents, text
):
    from tldw_chatbook.Workflows.models import FieldEdit

    base = documents.create(json.dumps(prompt_definition()))
    source = documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 0)
    edit = FieldEdit(source, "/steps/0/retry", text)
    candidate = documents.edit_fragment(base, edit, 1)
    assert candidate.error
    assert candidate.last_valid_json == source.last_valid_json
    assert '"retry":' + text + "," in candidate.raw_text
    saved = documents.put_draft(
        base.workflow_id,
        base.revision_id,
        candidate.raw_text,
        1,
        last_valid_json=candidate.last_valid_json,
        field_edit=edit,
    )
    assert saved == candidate
    for generation in (1, 2):
        replay = documents.put_draft(
            base.workflow_id, base.revision_id, candidate.raw_text, generation
        )
        assert replay.error == candidate.error
        assert replay.last_valid_json == source.last_valid_json
    with pytest.raises(InvalidDraft):
        documents.save_revision(base.workflow_id, base.revision_id, 2)


@pytest.mark.parametrize(
    "tamper", ["raw", "projection", "source", "pointer", "generation"]
)
def test_fragment_persistence_rejects_forged_or_stale_proof(documents, tamper):
    from dataclasses import replace

    from tldw_chatbook.Workflows.models import FieldEdit

    base = documents.create(json.dumps(prompt_definition()))
    source = documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 0)
    edit = FieldEdit(source, "/steps/0/retry", "")
    candidate = documents.edit_fragment(base, edit, 1)
    if tamper == "raw":
        candidate = replace(candidate, raw_text=source.raw_text)
    elif tamper == "projection":
        candidate = replace(
            candidate,
            last_valid_json=documents.edit_field(source.raw_text, "/name", "forged"),
        )
    elif tamper == "source":
        edit = replace(
            edit,
            source=replace(
                source,
                raw_text=documents.edit_field(source.raw_text, "/name", "forged"),
            ),
        )
    elif tamper == "pointer":
        edit = replace(edit, pointer="/steps/1/retry")
    else:
        documents.put_draft(base.workflow_id, base.revision_id, source.raw_text, 2)
    before = documents.get_draft(base.workflow_id, base.revision_id)
    with pytest.raises((DraftConflict, InvalidDraft)):
        documents.put_draft(
            base.workflow_id,
            base.revision_id,
            candidate.raw_text,
            1,
            last_valid_json=candidate.last_valid_json,
            field_edit=edit,
        )
    assert documents.get_draft(base.workflow_id, base.revision_id) == before


@pytest.mark.parametrize(
    "token",
    ["9" * 5000, "1e999999999999999", "-0", "-0.000"],
    ids=["huge-int", "exponent", "signed-zero", "decimal-zero"],
)
def test_copy_draft_to_head_preserves_source_and_lossless_content(documents, token):
    base = documents.create(
        json.dumps(prompt_definition())[:-1] + ',"opaque":' + token + "}"
    )
    documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 1)
    head = documents.save_revision(base.workflow_id, base.revision_id, 1)
    source = documents.put_draft(
        base.workflow_id,
        base.revision_id,
        documents.edit_field(base.raw_json, "/name", "Newer edits"),
        2,
    )
    copied = documents.copy_draft_to_head(source, head)
    assert copied.base_revision_id == head.revision_id
    assert (
        documents.field_text(copied.raw_text, "/name", as_json=False) == "Newer edits"
    )
    assert documents.field_text(copied.raw_text, "/opaque") == token
    assert documents.get_draft(base.workflow_id, base.revision_id) == source
    assert {d.base_revision_id for d in documents.list_drafts(base.workflow_id)} == {
        base.revision_id,
        head.revision_id,
    }
    saved = documents.save_revision(
        head.workflow_id, head.revision_id, copied.generation
    )
    assert saved.parent_revision_ids == (head.revision_id,)
    assert documents.field_text(saved.raw_json, "/name", as_json=False) == "Newer edits"


@pytest.mark.parametrize("conflict", ["source", "head", "dirty", "invalid"])
def test_copy_draft_to_head_rejects_changed_or_unsafe_inputs_without_mutation(
    documents,
    conflict,
):
    base = documents.create(json.dumps(prompt_definition()))
    documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 1)
    head = documents.save_revision(base.workflow_id, base.revision_id, 1)
    source = documents.put_draft(
        base.workflow_id,
        base.revision_id,
        documents.edit_field(base.raw_json, "/name", "Keep my edits"),
        2,
    )
    if conflict == "source":
        documents.put_draft(base.workflow_id, base.revision_id, '{"unfinished":', 3)
    elif conflict == "invalid":
        source = documents.put_draft(
            base.workflow_id, base.revision_id, '{"unfinished":', 3
        )
    elif conflict == "dirty":
        documents.put_draft(head.workflow_id, head.revision_id, '{"someone_else":', 0)
    elif conflict == "head":
        documents.put_draft(head.workflow_id, head.revision_id, head.raw_json, 0)
        documents.save_revision(head.workflow_id, head.revision_id, 0)
    old_source = documents.get_draft(base.workflow_id, base.revision_id)
    old_target = documents.get_draft(head.workflow_id, head.revision_id)
    with pytest.raises((DraftConflict, RevisionConflict, InvalidDraft)):
        documents.copy_draft_to_head(source, head)
    assert documents.get_draft(base.workflow_id, base.revision_id) == old_source
    assert documents.get_draft(head.workflow_id, head.revision_id) == old_target


def test_two_recovery_copies_cannot_overwrite_the_first_target_draft(documents):
    base = documents.create(json.dumps(prompt_definition()))
    source = documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 1)
    head = documents.save_revision(base.workflow_id, base.revision_id, 1)
    source = documents.put_draft(
        base.workflow_id,
        base.revision_id,
        documents.edit_field(base.raw_json, "/name", "Retained source"),
        2,
    )
    barrier = Barrier(2)

    def copy():
        barrier.wait(timeout=3)
        try:
            return documents.copy_draft_to_head(source, head)
        except DraftConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: copy(), range(2)))
    assert sum(item is not None for item in outcomes) == 1
    assert documents.get_draft(base.workflow_id, base.revision_id) == source
    assert documents.get_draft(head.workflow_id, head.revision_id) in outcomes


def test_recovery_transaction_failure_rolls_back_target_and_keeps_source(
    documents, monkeypatch
):
    base = documents.create(json.dumps(prompt_definition()))
    documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 0)
    head = documents.save_revision(base.workflow_id, base.revision_id, 0)
    source = documents.put_draft(
        base.workflow_id,
        base.revision_id,
        documents.edit_field(base.raw_json, "/name", "Must survive rollback"),
        1,
    )
    transaction = documents._db.transaction

    @contextmanager
    def fail_after_insert(**kwargs):
        with transaction(**kwargs) as cursor:
            yield cursor
            raise OSError("simulated failure before commit")

    with monkeypatch.context() as patcher:
        patcher.setattr(documents._db, "transaction", fail_after_insert)
        with pytest.raises(OSError):
            documents.copy_draft_to_head(source, head)
    assert documents.get_draft(head.workflow_id, head.revision_id) is None
    assert documents.get_draft(base.workflow_id, base.revision_id) == source
    assert documents.list_workflows() == (head,)


@pytest.mark.parametrize(
    "token",
    [
        "9" * 5000,
        "1e999999999999999999999",
        "-0",
        "-0.000",
        "0.12345678901234567890123456789",
    ],
    ids=[
        "huge-int",
        "extreme-exponent",
        "signed-int-zero",
        "signed-decimal-zero",
        "exact-decimal",
    ],
)
def test_public_editor_seams_preserve_opaque_numbers_without_durable_writes(
    documents, token
):
    raw = json.dumps(prompt_definition())[:-1] + ',"opaque":' + token + "}"
    base = documents.create(raw)
    edited = documents.edit_field(raw, "/steps/0/config/template", "A new template")
    assert '"opaque":' + token in edited
    assert documents.field_text(edited, "/opaque") == token
    assert (
        documents.project(edited)["steps"][0]["config"]["template"] == "A new template"
    )
    valid = documents.validate_draft(base, edited, 1, None)
    invalid = documents.validate_draft(base, '{"steps": [', 2, valid)
    assert invalid.error is not None
    assert invalid.last_valid_json == edited
    assert documents.get_draft(base.workflow_id, base.revision_id) is None


def test_editor_identity_and_unsafe_reordering_are_refused(documents):
    base = documents.create(json.dumps(prompt_definition()))
    with pytest.raises(InvalidDraft):
        documents.edit_field(
            base.raw_json, "/metadata/tldw_workflow/workflow_id", "foreign"
        )
    with pytest.raises(InvalidDraft, match="/steps/1/config/template"):
        documents.edit_steps(base.raw_json, "move", "finish", offset=-1)
    with pytest.raises(InvalidDraft, match="/steps/1/config/template"):
        documents.edit_steps(base.raw_json, "delete", "prepare")
    renamed = documents.edit_field(base.raw_json, "/steps/0/name", "Readable name")
    assert documents.project(renamed)["steps"][0]["id"] == "prepare"


def test_step_duplication_preserves_metadata_and_allocates_identity(documents):
    raw = json.dumps(prompt_definition()).replace(
        '"metadata": {', '"metadata": {"vendor": -0,', 1
    )
    duplicate = documents.edit_steps(raw, "duplicate", "prepare")
    steps = documents.project(duplicate)["steps"]
    assert len(steps) == 3
    assert len({step["id"] for step in steps}) == 3
    assert duplicate.count('"vendor":-0') == 1


def test_invalid_field_fragment_is_part_of_recoverable_raw_document(documents):
    from tldw_chatbook.Workflows.models import FieldEdit

    base = documents.create(json.dumps(prompt_definition()))
    source = documents.validate_draft(base, base.raw_json, 0)
    draft = documents.edit_fragment(
        base, FieldEdit(source, "/inputs", '{"source_text":'), 1
    )
    assert draft.error is not None
    assert '"inputs":{"source_text":' in draft.raw_text
    assert draft.last_valid_json == base.raw_json


def test_invalid_draft_survives_reopen_without_replacing_projection(tmp_path):
    path = tmp_path / "workflows.sqlite3"
    db = WorkflowsDB(path)
    documents = DocumentService(db)
    original = prompt_definition()
    original["metadata"]["vendor_extension"] = {"opaque": [1, {"x": True}]}
    revision = documents.create(json.dumps(original))
    draft = documents.put_draft(
        revision.workflow_id, revision.revision_id, '{"steps": [', 1
    )
    assert draft.error is not None
    assert json.loads(draft.last_valid_json) == json.loads(revision.raw_json)
    with pytest.raises(InvalidDraft):
        documents.save_revision(revision.workflow_id, revision.revision_id, 1)
    db.close()
    reopened_db = WorkflowsDB(path)
    try:
        recovered = DocumentService(reopened_db).get_draft(
            revision.workflow_id, revision.revision_id
        )
        assert recovered == draft
    finally:
        reopened_db.close()


@pytest.mark.parametrize("removed", ["namespace", "metadata"])
@pytest.mark.parametrize("has_previous", [False, True])
def test_deleted_identity_stays_invalid_and_exactly_recoverable(
    tmp_path, removed, has_previous
):
    path = tmp_path / "identity.sqlite3"
    db = WorkflowsDB(path)
    documents = DocumentService(db)
    original = prompt_definition()
    # Keep numeric spelling and whitespace observable across refusal/recovery.
    opaque = ', "opaque": [-0.000, 1e999999999999999999999]}'
    try:
        base = documents.create(json.dumps(original)[:-1] + opaque)
        last_valid = base.raw_json
        if has_previous:
            original["name"] = "Last valid edit"
            last_valid = json.dumps(original)[:-1] + opaque
            documents.put_draft(base.workflow_id, base.revision_id, last_valid, 1)
        if removed == "metadata":
            del original["metadata"]
        else:
            del original["metadata"]["tldw_workflow"]
        raw = "\n\t" + json.dumps(original, indent=2)[:-1] + opaque + "\n"
        draft = documents.put_draft(base.workflow_id, base.revision_id, raw, 2)
        assert draft.error is not None
        assert "identity" in draft.error.lower()
        assert draft.raw_text == raw
        assert draft.last_valid_json == last_valid
        with pytest.raises(InvalidDraft, match="identity"):
            documents.save_revision(base.workflow_id, base.revision_id, 2)
        assert documents.list_revisions(base.workflow_id) == (base,)
        assert documents.get_head(base.workflow_id) == base
    finally:
        db.close()

    reopened = WorkflowsDB(path)
    try:
        recovered_documents = DocumentService(reopened)
        assert (
            recovered_documents.get_draft(base.workflow_id, base.revision_id) == draft
        )
        repaired = recovered_documents.put_draft(
            base.workflow_id, base.revision_id, last_valid, 3
        )
        assert repaired.error is None
        saved = recovered_documents.save_revision(base.workflow_id, base.revision_id, 3)
        assert '"opaque":[-0.000,1e999999999999999999999]' in saved.raw_json
        assert saved.parent_revision_ids == (base.revision_id,)
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "location", ["root", "metadata", "namespace", "step", "config"]
)
def test_field_edit_preserves_unknown_json_and_immutable_ancestry(documents, location):
    original = prompt_definition()
    target = {
        "root": original,
        "metadata": original["metadata"],
        "namespace": original["metadata"]["tldw_workflow"],
        "step": original["steps"][0],
        "config": original["steps"][0]["config"],
    }[location]
    target["vendor_extension"] = {"opaque": [None, True, 123, {"雪": "é"}]}
    raw = json.dumps(original)
    base = documents.create(raw)
    edited = json.loads(base.raw_json)
    edited["steps"][0]["config"]["template"] = "Changed"
    assert (
        json.loads(documents.get_revision(base.workflow_id, base.revision_id).raw_json)
        == original
    )
    draft = documents.put_draft(
        base.workflow_id, base.revision_id, json.dumps(edited), 1
    )
    assert draft.error is None
    saved = documents.save_revision(base.workflow_id, base.revision_id, 1)
    assert saved.workflow_id == base.workflow_id
    assert saved.revision_id != base.revision_id
    assert str(UUID(saved.revision_id)) == saved.revision_id
    assert saved.parent_revision_ids == (base.revision_id,)
    edited["metadata"]["tldw_workflow"].update(
        revision_id=saved.revision_id, parent_revision_ids=[base.revision_id]
    )
    assert json.loads(saved.raw_json) == edited
    assert documents.get_revision(base.workflow_id, base.revision_id) == base
    assert documents.list_revisions(base.workflow_id) == (base, saved)
    assert documents.list_workflows() == (saved,)


@pytest.mark.parametrize("metadata", [None, {}, {"vendor": [1, 2]}])
def test_absent_identity_creates_distinct_workflows_without_title_matching(
    documents, metadata
):
    original = prompt_definition()
    if metadata is None:
        del original["metadata"]
    else:
        original["metadata"] = metadata
    first = documents.create(json.dumps(original))
    second = documents.create(json.dumps(original))
    assert first.workflow_id != second.workflow_id
    assert first.revision_id != second.revision_id
    for revision in (first, second):
        identity = json.loads(revision.raw_json)["metadata"]["tldw_workflow"]
        assert identity == {
            "format_version": 1,
            "workflow_id": str(UUID(revision.workflow_id)),
            "revision_id": str(UUID(revision.revision_id)),
            "parent_revision_ids": [],
        }
    if metadata:
        assert json.loads(first.raw_json)["metadata"]["vendor"] == [1, 2]
    assert set(documents.list_workflows()) == {first, second}


@pytest.mark.parametrize("with_metadata", [False, True])
async def test_import_without_identity_synthesizes_it_and_preserves_opaque_content(
    tmp_path, with_metadata
):
    from tldw_chatbook.Workflows.authoring import WorkflowAuthoring

    definition = prompt_definition()
    if with_metadata:
        definition["metadata"] = {"vendor": {"retained": [True, None, "雪"]}}
    else:
        del definition["metadata"]
    raw = json.dumps(definition)[:-1] + ', "opaque": -0.000}'
    source = tmp_path / "incoming.json"
    source.write_text(raw, encoding="utf-8")
    owner = WorkflowAuthoring(lambda: tmp_path / "import.sqlite3")
    try:
        first = await owner.import_file(source)
        second = await owner.import_file(source)
        assert first.workflow_id != second.workflow_id
        assert first.revision_id != second.revision_id
        for revision in (first, second):
            imported = json.loads(revision.raw_json)
            assert imported["metadata"].pop("tldw_workflow") == {
                "format_version": 1,
                "workflow_id": str(UUID(revision.workflow_id)),
                "revision_id": str(UUID(revision.revision_id)),
                "parent_revision_ids": [],
            }
            assert imported.pop("opaque") == 0
            if not with_metadata:
                assert imported.pop("metadata") == {}
            assert imported == definition
            assert '"opaque":-0.000' in revision.raw_json
        assert source.read_text(encoding="utf-8") == raw
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "case", ["empty", "unknown_type", "routing", "101_steps", "over_run_bytes"]
)
def test_structural_save_does_not_apply_run_admission(documents, case):
    original = prompt_definition()
    if case == "empty":
        original["steps"] = []
    elif case == "unknown_type":
        original["steps"][0]["type"] = "future_vendor.parallel"
    elif case == "routing":
        original["steps"][0]["on_success"] = {"future": "opaque"}
    elif case == "101_steps":
        original["steps"] = [
            {"id": f"step_{n}", "type": "unknown", "config": {}} for n in range(101)
        ]
    else:
        original["vendor"] = "x" * (2 * 1024 * 1024 + 1)
    base = documents.create(json.dumps(original))
    documents.put_draft(base.workflow_id, base.revision_id, json.dumps(original), 0)
    saved = documents.save_revision(base.workflow_id, base.revision_id, 0)
    result = json.loads(saved.raw_json)
    assert result["steps"] == original["steps"]
    assert result.get("vendor") == original.get("vendor")


@pytest.mark.parametrize("bad_id", ["prepare", "inputs", "last", "", 1, None])
def test_invalid_step_identity_rejects_revision_but_retains_draft(documents, bad_id):
    original = prompt_definition()
    base = documents.create(json.dumps(original))
    original["steps"][1]["id"] = bad_id
    text = json.dumps(original)
    with pytest.raises(InvalidDraft):
        documents.create(text)
    draft = documents.put_draft(base.workflow_id, base.revision_id, text, 1)
    assert draft.raw_text == text
    assert draft.error is not None
    assert draft.last_valid_json == base.raw_json


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        '{"steps":{}}',
        '{"steps":[{"id":"x","type":1,"config":{}}]}',
        '{"steps":[{"id":"x","type":"prompt","config":[]}]}',
        '{"steps":[],"metadata":null}',
        '{"steps":[],"metadata":{"tldw_workflow":{}}}',
    ],
)
def test_malformed_structure_is_not_silently_normalized(documents, raw):
    with pytest.raises(InvalidDraft):
        documents.create(raw)
    assert documents.list_workflows() == ()


@pytest.mark.parametrize(
    "fragment",
    [
        '"vendor":NaN',
        '"vendor":Infinity',
        '"vendor":-Infinity',
        '"vendor":{"a":1,"a":2}',
        '"steps":[],"steps":[]',
    ],
)
def test_duplicate_keys_and_non_json_numbers_are_recoverable_invalid_text(
    documents, fragment
):
    base = documents.create(json.dumps(prompt_definition()))
    text = '{"steps":[], ' + fragment + "}"
    with pytest.raises(InvalidDraft):
        documents.create(text)
    draft = documents.put_draft(base.workflow_id, base.revision_id, text, 1)
    assert draft.raw_text == text
    assert draft.last_valid_json == base.raw_json
    assert draft.error is not None


def test_unknown_numbers_keep_precision_through_identity_initialization_and_save(
    documents,
):
    raw = '{"steps":[],"vendor":[0.12345678901234567890123456789,1e999,123456789012345678901234567890]}'
    expected = json.loads(raw, parse_float=Decimal)
    base = documents.create(raw)
    documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 1)
    saved = documents.save_revision(base.workflow_id, base.revision_id, 1)
    assert (
        json.loads(saved.raw_json, parse_float=Decimal)["vendor"] == expected["vendor"]
    )


@pytest.mark.parametrize("number", ["1e999999999999999999999999", "7" * 5000, "-0"])
def test_extreme_valid_number_tokens_are_not_lost_to_host_numeric_limits(
    documents, number
):
    base = documents.create('{"steps":[],"opaque":' + number + "}")
    documents.put_draft(base.workflow_id, base.revision_id, base.raw_json, 1)
    saved = documents.save_revision(base.workflow_id, base.revision_id, 1)
    assert (
        json.loads(saved.raw_json, parse_int=str, parse_float=str)["opaque"] == number
    )


def test_generation_replay_stale_writes_and_invalid_projection_recovery(documents):
    base = documents.create(json.dumps(prompt_definition()))
    edited = json.loads(base.raw_json)
    edited["name"] = "Edited"
    valid = documents.put_draft(
        base.workflow_id, base.revision_id, json.dumps(edited), 3
    )
    assert (
        documents.put_draft(base.workflow_id, base.revision_id, valid.raw_text, 3)
        == valid
    )
    for generation, text in [(2, valid.raw_text), (3, "different")]:
        with pytest.raises(DraftConflict):
            documents.put_draft(base.workflow_id, base.revision_id, text, generation)
    invalid = documents.put_draft(base.workflow_id, base.revision_id, "{", 4)
    assert invalid.last_valid_json == valid.last_valid_json
    with pytest.raises(DraftConflict):
        documents.save_revision(base.workflow_id, base.revision_id, 3)
    fixed = documents.put_draft(base.workflow_id, base.revision_id, valid.raw_text, 5)
    assert fixed.error is None
    assert (
        documents.save_revision(base.workflow_id, base.revision_id, 5).revision_id
        != base.revision_id
    )


@pytest.mark.parametrize("generation", [-1, True, 1.5, "1", 2**63])
def test_generation_boundary_rejects_non_sqlite_nonnegative_integers(
    documents, generation
):
    base = documents.create(json.dumps(prompt_definition()))
    with pytest.raises(DraftConflict):
        documents.put_draft(
            base.workflow_id, base.revision_id, base.raw_json, generation
        )
    with pytest.raises(DraftConflict):
        documents.save_revision(base.workflow_id, base.revision_id, generation)
    assert documents.get_draft(base.workflow_id, base.revision_id) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("workflow_id", "not-a-uuid"),
        ("revision_id", "not-a-uuid"),
        ("parent_revision_ids", ["not-a-uuid"]),
        ("format_version", True),
        ("format_version", 2),
    ],
)
def test_invalid_portable_identity_is_not_replaced(documents, field, value):
    original = prompt_definition()
    original["metadata"]["tldw_workflow"][field] = value
    with pytest.raises(InvalidDraft):
        documents.create(json.dumps(original))


@pytest.mark.parametrize("workflow_alias,revision_alias", UUID_ALIAS_CASES)
@pytest.mark.parametrize("field", ["workflow_id", "revision_id", "parent_revision_ids"])
def test_noncanonical_uuid_fields_cannot_create_a_revision(
    documents, workflow_alias, revision_alias, field
):
    original = prompt_definition()
    original["metadata"]["tldw_workflow"][field] = {
        "workflow_id": workflow_alias,
        "revision_id": revision_alias,
        "parent_revision_ids": [workflow_alias],
    }[field]
    with pytest.raises(InvalidDraft):
        documents.create(json.dumps(original))
    assert documents.list_workflows() == ()


@pytest.mark.parametrize("workflow_alias,revision_alias", UUID_ALIAS_CASES)
@pytest.mark.parametrize("violation", ["self_parent", "duplicate_parent"])
def test_uuid_parent_aliases_cannot_hide_invalid_ancestry(
    documents, workflow_alias, revision_alias, violation
):
    original = prompt_definition()
    identity = original["metadata"]["tldw_workflow"]
    identity["parent_revision_ids"] = (
        [revision_alias]
        if violation == "self_parent"
        else [identity["workflow_id"], workflow_alias]
    )
    with pytest.raises(InvalidDraft):
        documents.create(json.dumps(original))
    assert documents.list_workflows() == ()


@pytest.mark.parametrize(
    "workflow_alias,revision_alias",
    [
        *UUID_ALIAS_CASES,
        pytest.param(
            "ba9e62aa-9859-4d42-98dc-26e3b10cdba0",
            "b42d45fb-dc91-4645-a6d9-a72c589282bd",
            id="canonical",
        ),
    ],
)
@pytest.mark.parametrize("collision", ["workflow", "revision", "both"])
def test_uuid_spellings_cannot_duplicate_stored_identities(
    documents, workflow_alias, revision_alias, collision
):
    base = documents.create(json.dumps(prompt_definition()))
    original = prompt_definition()
    identity = original["metadata"]["tldw_workflow"]
    identity["workflow_id"] = (
        workflow_alias
        if collision in {"workflow", "both"}
        else "301c8a65-f77e-4515-b01c-0798cbe9c094"
    )
    identity["revision_id"] = (
        revision_alias
        if collision in {"revision", "both"}
        else "831b36dd-626d-443b-b3c8-18a4e435723a"
    )
    canonical = workflow_alias == "ba9e62aa-9859-4d42-98dc-26e3b10cdba0"
    with pytest.raises(RevisionConflict if canonical else InvalidDraft):
        documents.create(json.dumps(original))
    assert documents.list_workflows() == (base,)
    assert documents.list_revisions(base.workflow_id) == (base,)


@pytest.mark.parametrize("workflow_alias,revision_alias", UUID_ALIAS_CASES)
def test_uuid_alias_draft_text_survives_reopen_without_rewriting_other_json(
    tmp_path, workflow_alias, revision_alias
):
    path = tmp_path / "workflows.sqlite3"
    original = prompt_definition()
    original["metadata"]["vendor_ids"] = [workflow_alias, revision_alias]
    raw = json.dumps(original, indent=2)
    db = WorkflowsDB(path)
    try:
        documents = DocumentService(db)
        base = documents.create(raw)
        assert base.raw_json == raw
        original["metadata"]["tldw_workflow"]["parent_revision_ids"] = [revision_alias]
        invalid_text = json.dumps(original, indent=4)
        draft = documents.put_draft(base.workflow_id, base.revision_id, invalid_text, 1)
        assert draft.error is not None
        assert draft.raw_text == invalid_text
        assert draft.last_valid_json == raw
        with pytest.raises(InvalidDraft):
            documents.save_revision(base.workflow_id, base.revision_id, 1)
    finally:
        db.close()
    reopened = WorkflowsDB(path)
    try:
        documents = DocumentService(reopened)
        assert documents.get_draft(base.workflow_id, base.revision_id) == draft
        assert documents.get_revision(base.workflow_id, base.revision_id) == base
    finally:
        reopened.close()


@pytest.mark.parametrize("field", ["workflow_id", "revision_id", "parent_revision_ids"])
def test_draft_cannot_rebind_base_identity(documents, field):
    base = documents.create(json.dumps(prompt_definition()))
    edited = json.loads(base.raw_json)
    edited["metadata"]["tldw_workflow"][field] = (
        [] if field == "parent_revision_ids" else "301c8a65-f77e-4515-b01c-0798cbe9c094"
    )
    if field == "parent_revision_ids":
        edited["metadata"]["tldw_workflow"][field] = [base.revision_id]
    draft = documents.put_draft(
        base.workflow_id, base.revision_id, json.dumps(edited), 1
    )
    assert draft.error is not None
    with pytest.raises(InvalidDraft):
        documents.save_revision(base.workflow_id, base.revision_id, 1)


def test_missing_or_cross_workflow_base_cannot_own_drafts(documents):
    base = documents.create(json.dumps(prompt_definition()))
    assert documents.get_draft(base.workflow_id, base.revision_id) is None
    with pytest.raises(InvalidDraft):
        documents.save_revision(base.workflow_id, base.revision_id, 0)
    for workflow_id, revision_id in [
        ("missing", base.revision_id),
        (base.workflow_id, "missing"),
    ]:
        with pytest.raises(RevisionConflict):
            documents.get_revision(workflow_id, revision_id)
        with pytest.raises(RevisionConflict):
            documents.put_draft(workflow_id, revision_id, "{", 1)
        assert documents.get_draft(workflow_id, revision_id) is None
    assert documents.list_revisions("missing") == ()
    with pytest.raises(RevisionConflict):
        documents.create(base.raw_json)


def test_invalid_errors_are_bounded_and_do_not_echo_private_content(documents, caplog):
    base = documents.create(json.dumps(prompt_definition()))
    secret = "private-payload-" * 500
    raw = json.dumps({"steps": [{"id": secret, "type": [], "config": {}}]})
    draft = documents.put_draft(base.workflow_id, base.revision_id, raw, 1)
    assert draft.error is not None and len(draft.error) <= 256
    assert "private-payload" not in draft.error
    assert "private-payload" not in caplog.text


def test_editor_projection_does_not_coerce_incompatible_field_shapes(documents):
    base = documents.create(json.dumps(prompt_definition()))
    raw = documents.edit_field(
        base.raw_json, "/steps/0/config/template", '{"future":1}', as_json=True
    )
    assert not documents.field_editable(raw, "/steps/0/config/template", "string")
    assert documents.field_editable(raw, "/name", "string")
    opaque = documents.edit_field(
        base.raw_json, "/steps/0/retry", "1e99999", as_json=True
    )
    assert not documents.field_editable(opaque, "/steps/0/retry", "integer")
    branched = documents.edit_field(
        base.raw_json, "/steps/0/on_failure", '"finish"', as_json=True
    )
    with pytest.raises(InvalidDraft, match="opaque"):
        documents.edit_steps(branched, "duplicate", "prepare")
    assert not documents.field_editable(branched, "/steps/0/config/template", "string")
    configured = documents.edit_field(
        base.raw_json, "/steps/0/config/future_route", "finish"
    )
    with pytest.raises(InvalidDraft, match="opaque"):
        documents.edit_steps(configured, "duplicate", "prepare")


def test_storage_bound_rejects_without_overwriting_recoverable_draft(documents):
    base = documents.create(json.dumps(prompt_definition()))
    saved = documents.put_draft(base.workflow_id, base.revision_id, "{", 1)
    oversized = "é" * (8 * 1024 * 1024 + 1)
    with pytest.raises(InvalidDraft):
        documents.put_draft(base.workflow_id, base.revision_id, oversized, 2)
    with pytest.raises(InvalidDraft):
        documents.create(
            json.dumps({"steps": [], "vendor": oversized}, ensure_ascii=False)
        )
    assert documents.get_draft(base.workflow_id, base.revision_id) == saved


@pytest.mark.parametrize("operation", ["draft", "save"])
def test_two_connections_conflict_without_losing_the_winner(tmp_path, operation):
    path = tmp_path / "workflows.sqlite3"
    stores = [WorkflowsDB(path), WorkflowsDB(path)]
    first, second = [DocumentService(db) for db in stores]
    try:
        base = first.create(json.dumps(prompt_definition()))
        first.put_draft(base.workflow_id, base.revision_id, base.raw_json, 1)
        barrier = Barrier(2)

        def race(service, name):
            barrier.wait(timeout=5)
            try:
                if operation == "save":
                    return service.save_revision(base.workflow_id, base.revision_id, 1)
                edited = json.loads(base.raw_json)
                edited["name"] = name
                return service.put_draft(
                    base.workflow_id, base.revision_id, json.dumps(edited), 2
                )
            except (DraftConflict, RevisionConflict) as error:
                return error

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(race, service, name)
                for service, name in [(first, "one"), (second, "two")]
            ]
            results = [future.result(timeout=10) for future in futures]
        conflict_type = DraftConflict if operation == "draft" else RevisionConflict
        assert sum(isinstance(value, conflict_type) for value in results) == 1
        winner = next(value for value in results if not isinstance(value, Exception))
        if operation == "save":
            assert first.list_revisions(base.workflow_id) == (base, winner)
            assert second.list_workflows() == (winner,)
            assert first.get_draft(base.workflow_id, base.revision_id).generation == 1
        else:
            assert first.get_draft(base.workflow_id, base.revision_id) == winner
            assert (
                second.put_draft(base.workflow_id, base.revision_id, winner.raw_text, 2)
                == winner
            )
    finally:
        for db in stores:
            db.close()
