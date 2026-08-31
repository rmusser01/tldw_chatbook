from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider
from tldw_chatbook.Agents.run_context import (
    CurrentRunActor,
    use_run_actor,
    use_tool_call_id,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Library.library_tool_contract import make_public_id
from tldw_chatbook.Library.local_library_tool_service import LocalLibraryToolService
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository


USER_ID = "lesson-authority-user"
PROFILE_ID = "lesson-authority-profile"
DATASET_ID = "lesson-authority-dataset"
PRIMARY = CurrentRunActor("primary", "run-primary", None)

_MUTATION_TABLES = (
    "notes",
    "note_folders",
    "keywords",
    "note_folder_memberships",
    "note_keywords",
    "note_organization_receipts",
    "notes_organization_adoption_reviews",
    "notes_organization_sync_intents",
    "note_sync_publication_intents",
    "sync_log",
)


@pytest.fixture
def lesson_stack(tmp_path):
    db = CharactersRAGDB(tmp_path / "notes.db", USER_ID)
    notes = NotesInteropService(
        tmp_path,
        "lesson-authority",
        global_db_to_use=db,
    )
    notes._db_instances[USER_ID] = db
    service = LocalLibraryToolService(
        notes_service=notes,
        notes_user_id=USER_ID,
    )
    provider = LibraryToolProvider(service)
    try:
        yield db, notes, service, provider
    finally:
        db.close_connection()


def _checkpoint(db: CharactersRAGDB, *, ready: bool) -> None:
    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES (?, ?, ?, ?, ?, '2026-08-30T00:00:00Z')
            """,
            (
                PROFILE_ID,
                DATASET_ID,
                "ready" if ready else "initializing",
                "ready" if ready else "initializing",
                "complete" if ready else "not_started",
            ),
        )


def _durable_snapshot(db: CharactersRAGDB) -> dict[str, tuple[tuple, ...]]:
    connection = db.get_connection()
    return {
        table: tuple(
            tuple(row)
            for row in connection.execute(
                f"SELECT * FROM {table} ORDER BY rowid"  # noqa: S608 - fixed table allowlist
            ).fetchall()
        )
        for table in _MUTATION_TABLES
    }


def _error_code(result: ToolResult | dict) -> str:
    if isinstance(result, ToolResult):
        assert result.ok is False
        payload = json.loads(result.error)
    else:
        payload = result
    return str(payload["error"]["code"])


def _approve(
    provider: LibraryToolProvider,
    arguments: dict,
    *,
    call_id: str = "call-approved",
):
    preflight = provider.preflight_agent_lesson_save(
        "library_save_note", arguments, call_id
    )
    assert preflight is not None
    with use_run_actor(PRIMARY):
        authority = provider.issue_agent_lesson_approval(PRIMARY.run_id, preflight)
    return preflight, authority


def _provider_invoke(
    provider: LibraryToolProvider,
    arguments: dict,
    *,
    actor: CurrentRunActor | None = PRIMARY,
    call_id: str = "call-approved",
) -> ToolResult:
    if actor is None:
        return provider.invoke("library:library_save_note", arguments)
    with use_run_actor(actor), use_tool_call_id(call_id):
        return provider.invoke("library:library_save_note", arguments)


def _direct_note(
    notes: NotesInteropService,
    *,
    title: str,
    content: str,
    ensure_keywords: tuple[str, ...] = (),
    folder: str | None = None,
    receipt_id: str | None = None,
) -> dict:
    return notes.save_note_with_organization(
        USER_ID,
        title=title,
        content=content,
        ensure_keywords=ensure_keywords,
        folder=folder,
        receipt_id=receipt_id,
        server_profile_id=PROFILE_ID if receipt_id else None,
        dataset_id=DATASET_ID if receipt_id else None,
    )


def _classification_call(
    db: CharactersRAGDB,
    notes: NotesInteropService,
    classification: str,
) -> dict:
    if classification == "requested_marker":
        return {
            "title": "New reviewed lesson",
            "content": "Verified evidence",
            "ensure_keywords": ["agent-lesson"],
        }
    if classification == "current_marker":
        saved = _direct_note(
            notes,
            title="Existing marked lesson",
            content="Original",
            ensure_keywords=("agent-lesson",),
        )
    elif classification == "pending_organization":
        _checkpoint(db, ready=False)
        saved = _direct_note(
            notes,
            title="Pending lesson",
            content="Original",
            ensure_keywords=("agent-lesson",),
            folder="Agent_Lessons",
            receipt_id="receipt-pending",
        )
        assert saved["receipt_state"] == "pending_organization"
    elif classification == "placement_review":
        _checkpoint(db, ready=True)
        LocalNoteFolderRepository(db).create_folder(
            name="agent_lessons", parent_id=None
        )
        saved = _direct_note(
            notes,
            title="Placement lesson",
            content="Original",
            folder="Agent_Lessons",
            receipt_id="receipt-placement",
        )
        assert saved["receipt_state"] == "placement_review"
    else:  # pragma: no cover - parametrization owns this vocabulary
        raise AssertionError(classification)
    return {
        "title": f"Updated {classification}",
        "content": "Updated verified evidence",
        "note_id": make_public_id("note", saved["id"]),
        "expected_version": int(saved["version"]),
        "expected_organization_version": saved["organization_version"],
        "ensure_keywords": [],
    }


@pytest.mark.parametrize(
    "classification",
    (
        "requested_marker",
        "current_marker",
        "pending_organization",
        "placement_review",
    ),
)
@pytest.mark.parametrize(
    ("mode", "expected_code"),
    (
        ("primary_unapproved", "approval_required"),
        ("subagent", "foreground_required"),
        ("fleet", "foreground_required"),
        ("direct_provider", "approval_required"),
        ("direct_service", "approval_required"),
    ),
)
def test_classified_agent_save_requires_exact_foreground_authority(
    lesson_stack,
    classification,
    mode,
    expected_code,
):
    db, notes, service, provider = lesson_stack
    arguments = _classification_call(db, notes, classification)
    before = _durable_snapshot(db)

    if mode == "primary_unapproved":
        result = _provider_invoke(provider, arguments)
    elif mode == "subagent":
        result = _provider_invoke(
            provider,
            arguments,
            actor=CurrentRunActor("subagent", "run-child", "run-primary"),
        )
    elif mode == "fleet":
        result = _provider_invoke(
            provider,
            arguments,
            actor=CurrentRunActor("subagent", "run-fleet", "run-primary"),
        )
    elif mode == "direct_provider":
        result = _provider_invoke(provider, arguments, actor=None)
    else:
        result = service.invoke("library_save_note", arguments)

    assert _error_code(result) == expected_code
    assert _durable_snapshot(db) == before
    assert provider.agent_lesson_approval_count(PRIMARY.run_id) == 0


@pytest.mark.parametrize(
    "classification",
    (
        "requested_marker",
        "current_marker",
        "pending_organization",
        "placement_review",
    ),
)
def test_primary_approved_classified_save_consumes_authority(
    lesson_stack,
    classification,
):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, classification)
    _approve(provider, arguments)

    result = _provider_invoke(provider, arguments)

    assert result.ok is True
    assert provider.agent_lesson_approval_count(PRIMARY.run_id) == 0


def test_ordinary_agent_note_and_internal_note_save_keep_existing_behavior(lesson_stack):
    _db, notes, service, provider = lesson_stack
    credential_like = "api_key=crediblematerial123456"

    provider_result = _provider_invoke(
        provider,
        {"title": "Ordinary agent note", "content": credential_like},
        actor=CurrentRunActor("subagent", "run-child", "run-primary"),
    )
    service_result = service.invoke(
        "library_save_note",
        {"title": "Ordinary MCP note", "content": credential_like},
    )
    internal = notes.save_note_with_organization(
        USER_ID,
        title="Ordinary internal note",
        content=credential_like,
    )

    assert provider_result.ok is True
    assert "error" not in service_result
    assert internal["version"] == 1


def test_marker_removal_after_review_fails_without_restoring_it(lesson_stack):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, "current_marker")
    _approve(provider, arguments)
    note_id = arguments["note_id"]
    from tldw_chatbook.Library.library_tool_contract import parse_public_id

    _, raw_note_id = parse_public_id(note_id, expected_type="note")
    keyword = db.get_connection().execute(
        "SELECT id FROM keywords WHERE keyword = 'agent-lesson' COLLATE BINARY"
    ).fetchone()
    db.unlink_note_from_keyword(raw_note_id, int(keyword["id"]))
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, arguments)

    assert _error_code(result) == "approval_required"
    assert _durable_snapshot(db) == before
    assert db.get_connection().execute(
        "SELECT COUNT(*) FROM note_keywords WHERE note_id = ?", (raw_note_id,)
    ).fetchone()[0] == 0


def test_marker_addition_after_review_fails_closed(lesson_stack):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, "placement_review")
    _approve(provider, arguments)
    from tldw_chatbook.Library.library_tool_contract import parse_public_id

    _, raw_note_id = parse_public_id(arguments["note_id"], expected_type="note")
    keyword_id = db.add_keyword("agent-lesson")
    db.link_note_to_keyword(raw_note_id, keyword_id)
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, arguments)

    assert _error_code(result) == "approval_required"
    assert _durable_snapshot(db) == before


def test_receipt_creation_after_review_fails_closed(lesson_stack):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, "current_marker")
    _approve(provider, arguments)
    from tldw_chatbook.Library.library_tool_contract import parse_public_id

    _, raw_note_id = parse_public_id(arguments["note_id"], expected_type="note")
    with db.transaction() as cursor:
        organization_version = db._library_organization_for_notes(
            cursor, [raw_note_id]
        )[raw_note_id]["organization_version"]
        cursor.execute(
            """
            INSERT INTO note_organization_receipts(
                receipt_id, note_id, requested_folder_name,
                requested_folder_sync_id, requested_keywords_json, review_id,
                collision_ids_json, note_version, organization_version, state,
                created_at, updated_at
            ) VALUES (?, ?, NULL, NULL, '[]', NULL, '[]', ?, ?,
                      'pending_organization', 'now', 'now')
            """,
            (
                "receipt-raced-in",
                raw_note_id,
                arguments["expected_version"],
                organization_version,
            ),
        )
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, arguments)

    assert _error_code(result) in {"approval_required", "organization_changed"}
    assert _durable_snapshot(db) == before


@pytest.mark.parametrize("race", ("delete", "transition"))
def test_receipt_removal_or_transition_after_review_fails_closed(
    lesson_stack,
    race,
):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, "pending_organization")
    _approve(provider, arguments)
    from tldw_chatbook.Library.library_tool_contract import parse_public_id

    _, raw_note_id = parse_public_id(arguments["note_id"], expected_type="note")
    with db.transaction() as cursor:
        if race == "delete":
            cursor.execute(
                "DELETE FROM note_organization_receipts WHERE note_id = ?",
                (raw_note_id,),
            )
        else:
            cursor.execute(
                "UPDATE note_organization_receipts SET state = 'placement_review', "
                "review_id = 'raced-review', collision_ids_json = '[\"folder\"]' "
                "WHERE note_id = ?",
                (raw_note_id,),
            )
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, arguments)

    assert _error_code(result) in {"approval_required", "organization_changed"}
    assert _durable_snapshot(db) == before


def test_content_version_change_after_review_returns_content_changed(lesson_stack):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, "current_marker")
    _approve(provider, arguments)
    from tldw_chatbook.Library.library_tool_contract import parse_public_id

    _, raw_note_id = parse_public_id(arguments["note_id"], expected_type="note")
    with db.transaction() as cursor:
        db._update_note_with_cursor(
            cursor,
            note_id=raw_note_id,
            update_data={"content": "Concurrent user content"},
            expected_version=arguments["expected_version"],
        )
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, arguments)

    assert _error_code(result) == "content_changed"
    assert _durable_snapshot(db) == before


def test_organization_version_change_after_review_returns_organization_changed(
    lesson_stack,
):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, "current_marker")
    _approve(provider, arguments)
    from tldw_chatbook.Library.library_tool_contract import parse_public_id

    _, raw_note_id = parse_public_id(arguments["note_id"], expected_type="note")
    keyword_id = db.add_keyword("concurrent-user-keyword")
    db.link_note_to_keyword(raw_note_id, keyword_id)
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, arguments)

    assert _error_code(result) == "organization_changed"
    assert _durable_snapshot(db) == before


def test_changed_arguments_and_swapped_note_identity_cannot_reuse_authority(
    lesson_stack,
):
    db, notes, _service, provider = lesson_stack
    first = _classification_call(db, notes, "current_marker")
    second_saved = _direct_note(
        notes,
        title="Second marked lesson",
        content="Second original",
        ensure_keywords=("agent-lesson",),
    )
    _approve(provider, first)
    changed = {
        **first,
        "content": "Changed after review",
        "note_id": make_public_id("note", second_saved["id"]),
        "expected_version": second_saved["version"],
        "expected_organization_version": second_saved["organization_version"],
    }
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, changed)

    assert _error_code(result) == "approval_required"
    assert _durable_snapshot(db) == before
    assert provider.agent_lesson_approval_count(PRIMARY.run_id) == 1


def test_reviewed_lesson_create_cannot_be_downgraded_to_ordinary_create(
    lesson_stack,
):
    db, _notes, _service, provider = lesson_stack
    reviewed = {
        "title": "Reviewed lesson",
        "content": "Verified evidence",
        "ensure_keywords": ["agent-lesson"],
    }
    _approve(provider, reviewed)
    downgraded = {**reviewed, "ensure_keywords": []}
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, downgraded)

    assert _error_code(result) == "approval_required"
    assert _durable_snapshot(db) == before
    assert provider.agent_lesson_approval_count(PRIMARY.run_id) == 1


def test_authority_is_single_use_and_replay_mutates_nothing(lesson_stack):
    db, notes, _service, provider = lesson_stack
    arguments = _classification_call(db, notes, "requested_marker")
    _approve(provider, arguments)
    first = _provider_invoke(provider, arguments)
    before_replay = _durable_snapshot(db)

    replay = _provider_invoke(provider, arguments)

    assert first.ok is True
    assert _error_code(replay) == "approval_required"
    assert _durable_snapshot(db) == before_replay


def test_forgeable_authority_fields_do_not_authenticate(lesson_stack):
    from tldw_chatbook.Agents.library_tool_provider import (
        _AgentLessonMutationContext,
    )

    db, notes, service, provider = lesson_stack
    arguments = _classification_call(db, notes, "requested_marker")
    _preflight, authority = _approve(provider, arguments)
    forged = replace(authority)
    before = _durable_snapshot(db)

    result = service._invoke_with_agent_lesson_context(
        "library_save_note",
        arguments,
        _AgentLessonMutationContext(
            issuer=provider,
            authority=forged,
            actor=PRIMARY,
            call_id="call-approved",
        ),
    )

    assert _error_code(result) == "approval_required"
    assert _durable_snapshot(db) == before
    assert provider.agent_lesson_approval_count(PRIMARY.run_id) == 1


def test_fake_noop_issuer_cannot_authenticate_real_authority(lesson_stack):
    from tldw_chatbook.Agents.library_tool_provider import (
        _AgentLessonMutationContext,
    )

    class FakeIssuer:
        def _consume_agent_lesson_approval(self, *_args, **_kwargs):
            return None

    db, _notes, service, provider = lesson_stack
    arguments = _classification_call(db, _notes, "requested_marker")
    _preflight, authority = _approve(provider, arguments)
    before = _durable_snapshot(db)

    result = service._invoke_with_agent_lesson_context(
        "library_save_note",
        arguments,
        _AgentLessonMutationContext(
            issuer=FakeIssuer(),
            authority=authority,
            actor=PRIMARY,
            call_id="call-approved",
        ),
    )

    assert _error_code(result) == "approval_required"
    assert _durable_snapshot(db) == before
    assert provider.agent_lesson_approval_count(PRIMARY.run_id) == 1


def test_classified_credential_refusal_is_content_free_and_atomic(lesson_stack):
    db, _notes, _service, provider = lesson_stack
    secret = "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdef"
    arguments = {
        "title": "Secret-shaped lesson",
        "content": f"Never persist {secret}",
        "ensure_keywords": ["agent-lesson"],
    }
    _approve(provider, arguments)
    before = _durable_snapshot(db)

    result = _provider_invoke(provider, arguments)

    assert _error_code(result) == "credential_material_detected"
    assert secret not in result.error
    assert _durable_snapshot(db) == before
