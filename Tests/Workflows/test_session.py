"""Saved workflow execution against real Notes and permission owners."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from Tests.Workflows.helpers import prompt_definition
from Tests.Workflows.test_session_admission import file_definition, revision_of


@pytest.fixture
async def harness(tmp_path, monkeypatch):
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Chat.console_library_policy import ConsoleLibraryMigrationSeed
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore
    from tldw_chatbook.Notes.Notes_Library import NotesInteropService
    from tldw_chatbook.Notes.notes_scope_service import NotesScopeService
    from tldw_chatbook.runtime_policy import (
        RuntimeSourceState,
        ServicePolicyEnforcer,
    )
    from tldw_chatbook.Workflows.document_service import DocumentService
    from tldw_chatbook.Workflows.session import (
        ModelSelection,
        RunSetup,
        WorkflowSession,
    )
    from tldw_chatbook.Workflows.session_permissions import WorkflowPermissions

    # Session tests own temporary Notes, not the interpreter's config binding.
    monkeypatch.setattr(
        "tldw_chatbook.Notes.Notes_Library.load_console_library_migration_seed",
        lambda: ConsoleLibraryMigrationSeed(auto_retrieve_on_send=False),
    )
    template = CharactersRAGDB(tmp_path / "notes.db", client_id="template")
    owner = NotesInteropService(tmp_path, "application", global_db_to_use=template)
    policy = {"state": RuntimeSourceState(active_source="local")}
    scope = NotesScopeService(
        owner,
        None,
        policy_enforcer=ServicePolicyEnforcer(state_provider=lambda: policy["state"]),
    )
    current = {"scope": scope, "user": "reader"}
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    payload["profiles"]["default"]["servers"] = {
        "agent:builtin": {
            "tools": {
                name: {"state": "allow"}
                for name in (
                    "workflow_read_file",
                    "workflow_local_model",
                    "create_note",
                )
            }
        }
    }
    store.save(payload)
    permissions = WorkflowPermissions(
        BuiltinToolGate(SimpleNamespace(permission_store=store))
    )
    sessions = []

    def new_session():
        session = WorkflowSession(
            permissions,
            notes_scope=lambda: current["scope"],
            notes_user=lambda: current["user"],
        )
        sessions.append(session)
        return session

    source = tmp_path / "source.txt"
    source.write_text("Source canary é", encoding="utf-8")
    source.chmod(0o600)
    database = WorkflowsDB(tmp_path / "workflows.db")
    documents = DocumentService(database)
    setup = RunSetup(
        source=source,
        model=ModelSelection("local-selected", "http://127.0.0.1:9099", "actual-model"),
        review_actor="reader",
        protected_paths=(tmp_path / "notes.db", tmp_path / "workflows.db"),
    )
    requests = []

    async def transport(request):
        requests.append(request)
        assert request.url == "http://127.0.0.1:9099/v1/chat/completions"
        body = json.loads(request.content)
        assert body["model"] == "actual-model"
        assert body["messages"] == [
            {"role": "user", "content": "Summarize in three bullets: Source canary é"}
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "Generated canary"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    monkeypatch.setattr(
        httpx, "AsyncHTTPTransport", lambda **kwargs: httpx.MockTransport(transport)
    )

    def set_permission(name, state):
        data = json.loads(store.path.read_text())
        data["profiles"]["default"]["servers"]["agent:builtin"]["tools"][name][
            "state"
        ] = state
        store.path.write_text(json.dumps(data))

    def rows():
        with template.transaction() as cursor:
            return [
                dict(row)
                for row in cursor.execute(
                    "SELECT id, title, content, deleted FROM notes"
                )
            ]

    try:
        yield SimpleNamespace(
            session=new_session(),
            new_session=new_session,
            setup=setup,
            documents=documents,
            database=database,
            rows=rows,
            owner=owner,
            template=template,
            scope=scope,
            current=current,
            permissions=permissions,
            store=store,
            set_permission=set_permission,
            requests=requests,
            policy=policy,
        )
    finally:
        for session in sessions:
            try:
                await session.close()
            except ValueError:
                pass  # Cleanup-failure tests assert the retained error themselves.
        owner.close_all_user_connections()
        template.close_connection()
        database.close()


async def until(session, predicate):
    changed = asyncio.Event()
    unsubscribe = session.subscribe(changed.set)
    try:
        async with asyncio.timeout(10):
            while not predicate(session.view()):
                assert session.view() is None or session.view().state not in {
                    "failed",
                    "cancelled",
                    "rejected",
                    "uncertain",
                    "completed",
                }, session.view()
                changed.clear()
                await changed.wait()
        return session.view()
    finally:
        unsubscribe()


async def launch(harness, document=None, inputs=None):
    revision = harness.documents.create(json.dumps(document or file_definition()))
    ticket = await harness.session.prepare(revision, inputs or {}, harness.setup)
    return revision, ticket, harness.session.start(ticket)


async def test_captured_actual_notes_path_is_protected_without_caller_hint(
    harness, monkeypatch
):
    from pathlib import Path

    from tldw_chatbook.Workflows import session as module

    h = harness
    revision = h.documents.create(json.dumps(file_definition()))
    ticket = await h.session.prepare(revision, {}, replace(h.setup, protected_paths=()))
    destination = h.session.bindings(ticket).notes
    # Changing the mutable template cannot change the captured cached route.
    h.owner.unified_db_template = None
    reached = []

    def guard(source, *, protected_paths, before_read):
        reached.extend(protected_paths)
        raise ValueError("source_database")

    monkeypatch.setattr(module, "read_local_text", guard)
    h.session.start(ticket)
    await until(h.session, lambda v: v.state == "failed")
    assert Path(destination.db_path) in reached
    assert h.requests == [] and h.rows() == []


async def test_saved_fixture_edited_review_single_note_and_no_execution_rows(harness):
    h = harness

    def historical_rows():
        with h.database.transaction(write=False) as cursor:
            tables = [
                row[0]
                for row in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'workflow_%'"
                )
            ]
            return {
                name: [tuple(row) for row in cursor.execute(f'SELECT * FROM "{name}"')]
                for name in tables
                if name
                not in {"workflow_revisions", "workflow_heads", "workflow_drafts"}
            }

    historical = historical_rows()
    revision, ticket, run_id = await launch(h)
    assert h.session.start(ticket) == run_id
    review = await until(h.session, lambda v: v.state == "review")
    assert review.review_text == "Generated canary"
    changed = json.loads(revision.raw_json)
    changed["steps"][-1]["config"]["content"] = "Changed saved draft"
    h.documents.put_draft(
        revision.workflow_id, revision.revision_id, json.dumps(changed), 1
    )
    h.documents.save_revision(revision.workflow_id, revision.revision_id, 1)
    assert h.session.update_review(run_id, "review", "My exact edit\né")
    assert not h.session.answer_review("stale", "review", accept=True)
    assert h.session.answer_review(run_id, "review", accept=True)
    assert not h.session.answer_review(run_id, "review", accept=True)
    final = await until(h.session, lambda v: v.state == "completed")
    assert final.revision_id == revision.revision_id
    assert len(h.requests) == 1
    assert h.rows() == [
        {
            "id": final.note_id,
            "title": "Reviewed file summary",
            "content": "My exact edit\né",
            "deleted": 0,
        }
    ]
    assert "canary" not in repr(review)
    assert historical_rows() == historical
    fresh = h.new_session()
    assert fresh.view() is None


async def test_binding_owned_inputs_override_imported_and_run_values(harness):
    h = harness
    document = file_definition()
    document["inputs"].update(
        source_uri="/private.txt",
        summary_provider="remote",
        summary_model="remote",
        review_actor="other",
    )
    _, _, run_id = await launch(h, document, {"summary_model": "override"})
    await until(h.session, lambda v: v.state == "review")
    assert h.session.answer_review(run_id, "review", accept=False)
    await until(h.session, lambda v: v.state == "rejected")
    assert len(h.requests) == 1 and h.rows() == []


async def test_tickets_are_single_use_latest_only_and_inputs_detached(harness):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    revision = revision_of(prompt_definition())
    inputs = {"source_text": "captured"}
    old = await h.session.prepare(revision, inputs, h.setup)
    current = await h.session.prepare(revision, inputs, h.setup)
    inputs["source_text"] = []
    with pytest.raises(SessionError):
        h.session.bindings(old)
    with pytest.raises(SessionError):
        h.session.start(old)
    run_id = h.session.start(current)
    assert h.session.start(current) == run_id
    await until(h.session, lambda v: v.state == "completed")
    newer = await h.session.prepare(revision, {}, h.setup)
    h.session.start(newer)
    with pytest.raises(SessionError):
        h.session.start(current)
    await until(h.session, lambda v: v.state == "completed")


@pytest.mark.parametrize("revocation", ["deny", "kill", "missing"])
async def test_exact_ask_identity_and_fresh_revocation(harness, revocation):
    h = harness
    h.set_permission("create_note", "ask")
    _, _, run_id = await launch(h)
    await until(h.session, lambda v: v.state == "review")
    assert h.session.answer_review(run_id, "review", accept=True)
    approval = await until(h.session, lambda v: v.state == "approval")
    effect = approval.pending_effect
    assert effect.kind == "note"
    assert not h.session.answer_effect(run_id, "save", "{}", approve=True)
    assert not h.session.answer_effect("old", "save", effect.payload_json, approve=True)
    if revocation == "deny":
        h.set_permission("create_note", "deny")
    elif revocation == "kill":
        payload = json.loads(h.store.path.read_text())
        payload["kill_switch"] = True
        h.store.path.write_text(json.dumps(payload))
    else:
        h.store.path.unlink()
    assert h.session.answer_effect(run_id, "save", effect.payload_json, approve=True)
    assert not h.session.answer_effect(
        run_id, "save", effect.payload_json, approve=True
    )
    await until(h.session, lambda v: v.state == "failed")
    assert h.rows() == []


async def test_each_ask_consumes_only_current_effect(harness):
    h = harness
    for name in ("workflow_read_file", "workflow_local_model", "create_note"):
        h.set_permission(name, "ask")
    _, _, run_id = await launch(h)
    for step in ("ingest", "summarize", "save"):
        view = await until(
            h.session, lambda v, step=step: v.state == "approval" and v.step_id == step
        )
        assert h.session.answer_effect(
            run_id, step, view.pending_effect.payload_json, approve=True
        )
        assert not h.session.answer_effect(
            run_id, step, view.pending_effect.payload_json, approve=True
        )
        if step == "summarize":
            await until(h.session, lambda v: v.state == "review")
            h.session.answer_review(run_id, "review", accept=True)
    await until(h.session, lambda v: v.state == "completed")
    assert len(h.requests) == len(h.rows()) == 1


@pytest.mark.parametrize(
    "bad", ["x" * 1024 * 1024, "\ud800", None], ids=["oversize", "surrogate", "nontext"]
)
async def test_invalid_review_edit_disables_accept_until_repaired(harness, bad):
    h = harness
    _, _, run_id = await launch(h)
    await until(h.session, lambda v: v.state == "review")
    assert not h.session.update_review(run_id, "review", bad)
    assert h.session.view().message_code == "review_invalid"
    assert not h.session.answer_review(run_id, "review", accept=True)
    assert h.session.update_review(run_id, "review", "repaired")
    assert h.session.answer_review(run_id, "review", accept=True)
    await until(h.session, lambda v: v.state == "completed")
    assert h.rows()[0]["content"] == "repaired"


@pytest.mark.parametrize("change", ["actor", "scope", "owner"])
async def test_current_identity_changes_cannot_redirect_note(harness, change):
    h = harness
    _, _, run_id = await launch(h)
    await until(h.session, lambda v: v.state == "review")
    if change == "actor":
        h.current["user"] = "other"
        assert not h.session.answer_review(run_id, "review", accept=True)
        h.session.cancel(run_id)
        await until(h.session, lambda v: v.state == "cancelled")
    else:
        assert h.session.answer_review(run_id, "review", accept=True)
        if change == "scope":
            h.current["scope"] = object()
        else:
            h.scope.local_notes_service = object()
        await until(h.session, lambda v: v.state == "failed")
    assert h.rows() == []


@pytest.mark.parametrize("seconds", [None, 0, -1, 1])
async def test_review_wait_deadline_and_late_controls(harness, seconds):
    h = harness
    document = file_definition()
    if seconds is None:
        document["steps"][3]["config"].pop("timeout_seconds")
    else:
        document["steps"][3]["config"]["timeout_seconds"] = seconds
    _, _, run_id = await launch(h, document)
    await until(h.session, lambda v: v.state == "review")
    if seconds == 1:
        view = await until(h.session, lambda v: v.state == "failed")
        assert view.message_code == "review_expired"
        assert not h.session.answer_review(run_id, "review", accept=True)
    else:
        assert h.session.update_review(run_id, "review", "persists without subscribers")
        assert h.session.view().review_text == "persists without subscribers"
        h.session.cancel(run_id)
        await until(h.session, lambda v: v.state == "cancelled")
    assert h.rows() == []


async def test_token_budget_refuses_before_post(harness):
    h = harness
    document = file_definition()
    document["steps"][2]["config"]["max_tokens"] = 100_000
    await launch(h, document)
    view = await until(h.session, lambda v: v.state == "failed")
    assert view.message_code == "token_budget"
    assert h.requests == [] and h.rows() == []


@pytest.mark.parametrize("setting", ["timeout", "model", "source", "actor"])
async def test_bad_launch_binding_refused_before_any_effect(harness, setting):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    document = file_definition()
    if setting == "timeout":
        h.setup = replace(
            h.setup, model=replace(h.setup.model, request_timeout_seconds=301)
        )
    elif setting == "model":
        document["steps"][2]["config"]["model"] = "unselected-model"
    elif setting == "source":
        document["steps"][0]["config"]["sources"][0]["uri"] = "/different.txt"
    else:
        document["steps"][3]["config"]["assigned_to_user_id"] = "other"
    with pytest.raises(SessionError):
        await h.session.prepare(revision_of(document), {}, h.setup)
    assert h.requests == [] and h.rows() == []


async def test_capture_request_timeout_can_be_lower_than_default(harness):
    h = harness
    document = file_definition()
    document["steps"][2]["timeout_seconds"] = 60
    h.setup = replace(h.setup, model=replace(h.setup.model, request_timeout_seconds=30))
    _, _, run_id = await launch(h, document)
    await until(h.session, lambda v: v.state == "review")
    h.session.cancel(run_id)


@pytest.mark.parametrize(
    "inputs",
    [
        {"note_title": 1},
        {"note_title": ""},
        {"extra": "x"},
        {"note_title": "x" * (10 * 1024 * 1024)},
    ],
    ids=["type", "minimum", "extra", "limit"],
)
async def test_invalid_inputs_refused_before_capture(harness, inputs, monkeypatch):
    from tldw_chatbook.Workflows import session as module

    def no_capture(*args, **kwargs):
        pytest.fail("invalid inputs must not capture a destination")

    monkeypatch.setattr(module, "capture_local_note_destination", no_capture)
    with pytest.raises(module.SessionError):
        await harness.session.prepare(
            revision_of(file_definition()), inputs, harness.setup
        )


@pytest.mark.parametrize(
    "budget,expected",
    [
        ("_OUTPUT_BYTES", "resolved_limit_or_type"),
        ("_AGGREGATE_BYTES", "output_budget"),
        ("_ACTIVE_SECONDS", "active_budget"),
    ],
)
async def test_output_and_active_budgets_block_advancement(
    harness, monkeypatch, budget, expected
):
    from tldw_chatbook.Workflows import session as module

    monkeypatch.setattr(module, budget, 1)
    if budget == "_ACTIVE_SECONDS":
        monkeypatch.setattr(module, budget, 0)
    ticket = await harness.session.prepare(
        revision_of(prompt_definition()), {}, harness.setup
    )
    harness.session.start(ticket)
    view = await until(harness.session, lambda v: v.state == "failed")
    assert view.message_code == expected
    assert harness.requests == [] and harness.rows() == []


async def test_restart_reopens_documents_and_notes_without_session_state(harness):
    from tldw_chatbook.DB.Workflows_DB import WorkflowsDB
    from tldw_chatbook.Workflows.document_service import DocumentService

    h = harness
    revision, _, run_id = await launch(h)
    await until(h.session, lambda v: v.state == "review")
    h.session.update_review(run_id, "review", "edited before quit")
    await h.session.close()
    h.database.close()
    h.owner.close_all_user_connections()
    reopened = WorkflowsDB(h.setup.protected_paths[1])
    try:
        assert DocumentService(reopened).get_head(revision.workflow_id) == revision
        fresh = h.new_session()
        assert fresh.view() is None
        ticket = await fresh.prepare(revision, {}, h.setup)
        assert fresh.bindings(ticket).notes.user_id == "reader"
        fresh.discard_setup()
        assert h.rows() == []
    finally:
        reopened.close()


async def test_run_binding_getter_only_exposes_current_run_including_terminal(harness):
    from tldw_chatbook.Workflows.session import SessionError

    h = harness
    with pytest.raises(SessionError):
        h.session.run_bindings("not-started")
    revision = revision_of(prompt_definition())
    ticket = await h.session.prepare(revision, {}, h.setup)
    captured = h.session.bindings(ticket)
    run_id = h.session.start(ticket)
    assert h.session.run_bindings(run_id) is captured
    with pytest.raises(SessionError):
        h.session.bindings(ticket)
    with pytest.raises(SessionError):
        h.session.run_bindings("wrong")
    await until(h.session, lambda v: v.state == "completed")
    assert h.session.run_bindings(run_id) is captured
    second = await h.session.prepare(revision, {}, h.setup)
    assert h.session.run_bindings(run_id) is captured
    next_run = h.session.start(second)
    assert h.session.run_bindings(next_run).source == h.setup.source
    with pytest.raises(SessionError):
        h.session.run_bindings(run_id)
    await until(h.session, lambda v: v.state == "completed")


async def test_review_instructions_are_exact_resolved_and_private(harness):
    h = harness
    document = file_definition()
    document["steps"][3]["config"]["instructions"] = (
        "Private instructions: {{ inputs.note_title }} / {{ ingest.text }}"
    )
    _, _, run_id = await launch(h, document)
    view = await until(h.session, lambda v: v.state == "review")
    assert (
        view.review_instructions
        == "Private instructions: Reviewed file summary / Source canary é"
    )
    assert "Private instructions" not in repr(view)
    assert h.session.view().review_instructions == view.review_instructions
    h.session.answer_review(run_id, "review", accept=True)
    final = await until(h.session, lambda v: v.state == "completed")
    assert final.review_instructions is None


async def test_unavailable_app_binding_is_a_payload_free_setup_error(harness):
    from tldw_chatbook.Workflows.session import SessionError, WorkflowSession

    def unavailable():
        raise RuntimeError("private-app-state")

    session = WorkflowSession(
        harness.permissions, notes_scope=unavailable, notes_user=lambda: "reader"
    )
    try:
        with pytest.raises(SessionError, match="setup_invalid"):
            await session.prepare(revision_of(prompt_definition()), {}, harness.setup)
    finally:
        await session.close()


def optional_input_definition(*, nested=False, referenced=True):
    document = prompt_definition()
    properties = {"optional_text": {"type": "string"}}
    if nested:
        properties = {"details": {"type": "object", "properties": properties}}
    document["metadata"]["tldw_workflow"]["input_schema"] = {
        "type": "object",
        "properties": properties,
    }
    path = "inputs.details.optional_text" if nested else "inputs.optional_text"
    document["steps"][0]["config"]["template"] = (
        "Reviewed {{ " + path + " }}" if referenced else "No optional input used"
    )
    document["steps"].insert(
        0,
        {
            "id": "early_save",
            "type": "notes",
            "retry": 0,
            "timeout_seconds": 300,
            "config": {
                "action": "create",
                "title": "Before missing input",
                "content": "No write is allowed before complete input validation",
            },
        },
    )
    return document


@pytest.mark.parametrize(
    "nested,defaults,overrides",
    [
        (False, {}, {}),
        (True, {}, {}),
        (True, {"details": {}}, {}),
        (True, {"details": {"optional_text": "saved"}}, {"details": {}}),
    ],
    ids=["optional-absent", "parent-absent", "child-absent", "override-removes-child"],
)
async def test_missing_optional_input_refused_before_capture_or_effects(
    harness, monkeypatch, nested, defaults, overrides
):
    from tldw_chatbook.Workflows import session as module

    h = harness
    calls = dict.fromkeys(["capture", "file", "model", "note"], 0)

    def observe(name, original):
        def counted(*args, **kwargs):
            calls[name] += 1
            return original(*args, **kwargs)

        return counted

    for name, seam in (
        ("capture", "capture_local_note_destination"),
        ("file", "read_local_text"),
        ("model", "complete_llama_bounded"),
        ("note", "create_local_note"),
    ):
        monkeypatch.setattr(module, seam, observe(name, getattr(module, seam)))
    document = optional_input_definition(nested=nested)
    document["inputs"].update(defaults)
    refusal = None
    try:
        ticket = await h.session.prepare(revision_of(document), overrides, h.setup)
    except module.SessionError as error:
        refusal = error
    else:
        # On the broken implementation, demonstrate the real earlier Note write.
        h.session.start(ticket)
        await until(h.session, lambda v: v.state in {"failed", "completed"})
    assert calls == {"capture": 0, "file": 0, "model": 0, "note": 0}
    assert refusal is not None and refusal.code == "reference"
    assert h.rows() == [] and h.requests == []


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("supply", ["unused", "default", "override"])
async def test_optional_inputs_remain_optional_and_use_final_merged_values(
    harness, nested, supply
):
    h = harness
    document = optional_input_definition(nested=nested, referenced=supply != "unused")
    value = {"optional_text": "available"}
    if nested:
        value = {"details": value}
    if supply == "default":
        document["inputs"].update(value)
    overrides = value if supply == "override" else {}
    ticket = await h.session.prepare(revision_of(document), overrides, h.setup)
    h.session.start(ticket)
    await until(h.session, lambda v: v.state == "completed")
    assert len(h.rows()) == 1
