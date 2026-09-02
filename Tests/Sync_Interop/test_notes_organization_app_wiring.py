from __future__ import annotations

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import pytest

from tldw_chatbook.app import (
    _install_deferred_notes_sync_facades,
    _wire_notes_sync_services,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.notes_scope_service import NotesScopeService, ScopeType
from tldw_chatbook.Sync_Interop.crypto import generate_dataset_key
from tldw_chatbook.Sync_Interop.notes_organization_sync_service import (
    NotesOrganizationSyncService,
)
from tldw_chatbook.Sync_Interop.notes_outbox_producer import NotesSyncV2OutboxProducer
from tldw_chatbook.Sync_Interop.sync_state_repository import SyncStateRepository


class _LocalNotes:
    def __init__(self, db=None):
        self.db = db

    def add_note(self, user_id, title, content, note_id=None):
        if self.db is not None:
            return self.db.add_note(title, content, note_id=note_id)
        del user_id, title, content
        return note_id or "note-a"

    def note_transaction(self, user_id):
        del user_id
        return (
            self.db.transaction(immediate=True)
            if self.db is not None
            else nullcontext()
        )

    def get_keywords_for_note(self, user_id, note_id):
        del user_id
        return self.db.get_keywords_for_note(note_id)

    def get_keyword_by_text(self, user_id, keyword):
        del user_id
        return self.db.get_keyword_by_text(keyword)

    def add_keyword(self, user_id, keyword):
        del user_id
        return self.db.add_keyword(keyword)

    def link_note_to_keyword(self, user_id, note_id, keyword_id):
        del user_id
        return self.db.link_note_to_keyword(note_id, keyword_id)

    def unlink_note_from_keyword(self, user_id, note_id, keyword_id):
        del user_id
        return self.db.unlink_note_from_keyword(note_id, keyword_id)


def test_local_runtime_wiring_seeds_agent_lessons_after_schema_readiness(tmp_path):
    notes = CharactersRAGDB(tmp_path / "local-seed.sqlite", client_id="app-local")
    app = SimpleNamespace(
        active_server_id=None,
        runtime_policy=SimpleNamespace(
            state=SimpleNamespace(active_source="local", active_server_id=None)
        ),
        chachanotes_db=notes,
        sync_state_repository=None,
        notes_scope_service=None,
        local_first_sync_service=None,
        sync_restore_service=None,
        manual_sync_control_service=None,
    )

    _wire_notes_sync_services(app)

    folder = notes.get_connection().execute(
        "SELECT name FROM note_folders WHERE parent_id IS NULL AND deleted = 0"
    ).fetchone()
    seed = notes.get_connection().execute(
        "SELECT profile_id, dataset_id, scope_mode, state FROM agent_lessons_seed_state"
    ).fetchone()
    assert folder["name"] == "Agent_Lessons"
    assert tuple(seed) == ("local", "local", "local_only", "seeded")
    notes.close_connection()


@pytest.mark.asyncio
async def test_production_shaped_wiring_replaces_none_seam_before_note_mutation(
    tmp_path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="app-wiring")
    state = SyncStateRepository(tmp_path / "sync.sqlite", client_id="app-wiring")
    key = generate_dataset_key()
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    shared_keys: dict[str, bytes] = {}
    scope = NotesScopeService(
        local_notes_service=_LocalNotes(),
        server_service=object(),
    )
    app = SimpleNamespace(
        active_server_id="server-a",
        chachanotes_db=notes,
        sync_state_repository=state,
        sync_v2_dataset_keys=shared_keys,
        notes_scope_service=scope,
        local_first_sync_service=SimpleNamespace(
            notes_organization_repository=None,
            notes_organization_sync_service=None,
        ),
        sync_restore_service=SimpleNamespace(notes_organization_repository=None),
        manual_sync_control_service=SimpleNamespace(
            notes_organization_sync_service=None,
            notes_repository=None,
        ),
    )

    _wire_notes_sync_services(app)

    assert isinstance(scope.sync_v2_notes_producer, NotesSyncV2OutboxProducer)
    assert scope.sync_v2_notes_producer.dataset_keys is shared_keys
    assert scope.sync_v2_notes_producer.notes_db is notes
    assert isinstance(scope.organization_sync_service, NotesOrganizationSyncService)
    assert (
        scope.organization_sync_service.notes_producer
        is scope.sync_v2_notes_producer
    )
    assert app.notes_organization_repository.db is notes
    assert app.notes_organization_repository.server_profile_id == "server-a"
    assert app.notes_organization_sync_service.state_repository is state
    assert (
        app.manual_sync_control_service.notes_organization_sync_service
        is app.notes_organization_sync_service
    )
    assert (
        app.manual_sync_control_service.notes_repository
        is app.notes_organization_repository
    )

    shared_keys["dataset-a"] = key
    created = await scope.save_note(
        scope=ScopeType.LOCAL_NOTE,
        title="Ordinary",
        content="Mutation",
        create_note_id="note-a",
        user_id="local-user",
        sync_v2_profile={"server_profile_id": "server-a"},
    )

    rows = state.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id="dataset-a",
    )
    assert created == "note-a"
    assert len(rows) == 1
    assert rows[0]["domain"] == "notes"


@pytest.fixture
def deferred_notes_sync_databases(tmp_path):
    """Yield the real SQLite owners with deterministic failure-path cleanup."""

    with ExitStack() as cleanup:
        notes = CharactersRAGDB(
            tmp_path / "deferred-notes.sqlite",
            client_id="deferred",
        )
        cleanup.callback(notes.close_connection)
        state = SyncStateRepository(
            tmp_path / "deferred-sync.sqlite",
            client_id="deferred",
        )
        cleanup.callback(state.close)
        yield notes, state


@pytest.mark.asyncio
async def test_deferred_notes_sync_facades_wire_on_first_sync_access(
    deferred_notes_sync_databases,
) -> None:
    notes, state = deferred_notes_sync_databases
    scope = NotesScopeService(
        local_notes_service=_LocalNotes(notes),
        server_service=object(),
    )
    app = SimpleNamespace(
        active_server_id="server-a",
        runtime_policy=SimpleNamespace(
            state=SimpleNamespace(
                active_source="server",
                active_server_id="server-a",
            )
        ),
        chachanotes_db=notes,
        sync_state_repository=state,
        sync_v2_dataset_keys={"dataset-a": generate_dataset_key()},
        notes_scope_service=scope,
        local_chat_conversation_service=SimpleNamespace(organization_sync_service=None),
        notes_organization_repository=None,
        notes_organization_sync_service=None,
        local_first_sync_service=SimpleNamespace(
            notes_organization_repository=None,
            notes_organization_sync_service=None,
        ),
        sync_restore_service=SimpleNamespace(notes_organization_repository=None),
        manual_sync_control_service=SimpleNamespace(
            notes_organization_sync_service=None,
            notes_repository=None,
        ),
    )
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )

    assert _install_deferred_notes_sync_facades(app) is True
    deferred_service = app.manual_sync_control_service.notes_organization_sync_service
    assert deferred_service is not None

    created = await scope.save_note(
        scope=ScopeType.LOCAL_NOTE,
        title="Deferred",
        content="First use",
        create_note_id="note-deferred",
        user_id="local-user",
        sync_v2_profile={"server_profile_id": "server-a"},
    )

    assert created == "note-deferred"
    assert deferred_service.notes_repository.db is notes
    assert isinstance(
        app.notes_organization_sync_service,
        NotesOrganizationSyncService,
    )
    assert (
        app.notes_scope_service.organization_sync_service
        is app.notes_organization_sync_service
    )
    assert (
        app.local_first_sync_service.notes_organization_repository
        is app.notes_organization_repository
    )
    assert (
        app.sync_restore_service.notes_organization_repository
        is app.notes_organization_repository
    )
    rows = state.list_pending_sync_v2_outbox_envelopes(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        dataset_id="dataset-a",
    )
    assert len(rows) == 1
    assert rows[0]["domain"] == "notes"


def test_notes_organization_wiring_detaches_local_and_rebinds_next_server(tmp_path):
    notes = CharactersRAGDB(tmp_path / "rebind-notes.sqlite", client_id="app-rebind")
    state = SyncStateRepository(tmp_path / "rebind-sync.sqlite", client_id="app-rebind")
    local_notes = _LocalNotes(notes)
    local_chat = SimpleNamespace(organization_sync_service=None)
    scope = NotesScopeService(local_notes_service=local_notes, server_service=object())
    app = SimpleNamespace(
        active_server_id="server-a",
        runtime_policy=SimpleNamespace(
            state=SimpleNamespace(
                active_source="server",
                active_server_id="server-a",
            )
        ),
        chachanotes_db=notes,
        sync_state_repository=state,
        sync_v2_dataset_keys={},
        notes_scope_service=scope,
        local_chat_conversation_service=local_chat,
        notes_organization_repository=None,
        notes_organization_sync_service=None,
        local_first_sync_service=SimpleNamespace(
            notes_organization_repository=None,
            notes_organization_sync_service=None,
        ),
        sync_restore_service=SimpleNamespace(notes_organization_repository=None),
        manual_sync_control_service=SimpleNamespace(
            notes_organization_sync_service=None,
            notes_repository=None,
        ),
    )

    _wire_notes_sync_services(app)
    first_service = app.notes_organization_sync_service
    assert app.notes_organization_repository.server_profile_id == "server-a"

    app.runtime_policy.state = SimpleNamespace(
        active_source="local",
        active_server_id="server-a",
    )
    _wire_notes_sync_services(app)

    assert app.notes_organization_repository is None
    assert app.notes_organization_sync_service is None
    assert scope.organization_sync_service is None
    assert local_notes.organization_sync_service is None
    assert local_chat.organization_sync_service is None
    assert app.local_first_sync_service.notes_organization_repository is None
    assert app.local_first_sync_service.notes_organization_sync_service is None
    assert app.sync_restore_service.notes_organization_repository is None
    assert app.manual_sync_control_service.notes_organization_sync_service is None
    assert app.manual_sync_control_service.notes_repository is None

    app.active_server_id = "server-b"
    app.runtime_policy.state = SimpleNamespace(
        active_source="server",
        active_server_id="server-b",
    )
    _wire_notes_sync_services(app)

    assert app.notes_organization_repository.server_profile_id == "server-b"
    assert app.notes_organization_sync_service is not first_service
    assert scope.organization_sync_service is app.notes_organization_sync_service
    assert local_notes.organization_sync_service is app.notes_organization_sync_service
    assert local_chat.organization_sync_service is app.notes_organization_sync_service
    notes.close_connection()


@pytest.mark.asyncio
async def test_production_scope_keyword_save_is_atomic_with_organization_intents(
    tmp_path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "keyword-notes.sqlite", client_id="app-keywords")
    state = SyncStateRepository(
        tmp_path / "keyword-sync.sqlite", client_id="app-keywords"
    )
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    with notes.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES ('server-a', 'dataset-a', 'ready', 'ready', 'complete',
                      '2026-08-29T00:00:00+00:00')
            """
        )
    scope = NotesScopeService(
        local_notes_service=_LocalNotes(notes), server_service=object()
    )
    app = SimpleNamespace(
        active_server_id="server-a",
        chachanotes_db=notes,
        sync_state_repository=state,
        sync_v2_dataset_keys={"dataset-a": generate_dataset_key()},
        notes_scope_service=scope,
        local_first_sync_service=SimpleNamespace(
            notes_organization_repository=None, notes_organization_sync_service=None
        ),
        sync_restore_service=SimpleNamespace(notes_organization_repository=None),
    )
    _wire_notes_sync_services(app)

    note_id = "00000000-0000-4000-8000-000000000030"
    result = await scope.save_note(
        scope=ScopeType.LOCAL_NOTE,
        title="Lesson",
        content="Body",
        create_note_id=note_id,
        user_id="local-user",
        keywords=("agent-lesson",),
        sync_v2_profile={"server_profile_id": "server-a"},
    )

    assert result["id"] == note_id
    rows = (
        notes.get_connection()
        .execute(
            "SELECT domain, operation FROM notes_organization_sync_intents ORDER BY domain"
        )
        .fetchall()
    )
    assert [(row["domain"], row["operation"]) for row in rows] == [
        ("notes.keyword", "upsert"),
        ("notes.keyword_link", "upsert"),
    ]


@pytest.mark.asyncio
async def test_production_scope_keyword_save_rolls_back_note_when_group_not_ready(
    tmp_path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "blocked-notes.sqlite", client_id="app-blocked")
    state = SyncStateRepository(
        tmp_path / "blocked-sync.sqlite", client_id="app-blocked"
    )
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    with notes.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO notes_organization_sync_checkpoints(
                server_profile_id, dataset_id, local_state, server_state,
                inventory_phase, updated_at
            ) VALUES ('server-a', 'dataset-a', 'pulling', 'ready', 'complete',
                      '2026-08-29T00:00:00+00:00')
            """
        )
    scope = NotesScopeService(
        local_notes_service=_LocalNotes(notes), server_service=object()
    )
    app = SimpleNamespace(
        active_server_id="server-a",
        chachanotes_db=notes,
        sync_state_repository=state,
        sync_v2_dataset_keys={"dataset-a": generate_dataset_key()},
        notes_scope_service=scope,
        local_first_sync_service=SimpleNamespace(
            notes_organization_repository=None, notes_organization_sync_service=None
        ),
        sync_restore_service=SimpleNamespace(notes_organization_repository=None),
    )
    _wire_notes_sync_services(app)
    note_id = "00000000-0000-4000-8000-000000000031"

    with pytest.raises(ValueError, match="organization group is not ready"):
        await scope.save_note(
            scope=ScopeType.LOCAL_NOTE,
            title="Blocked",
            content="Body",
            create_note_id=note_id,
            user_id="local-user",
            keywords=("agent-lesson",),
            sync_v2_profile={"server_profile_id": "server-a"},
        )

    assert notes.get_note_by_id(note_id) is None
    assert (
        notes.get_connection().execute("SELECT COUNT(*) FROM keywords").fetchone()[0]
        == 0
    )
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_real_per_user_notes_owner_rolls_back_mutation_and_intent_together(
    tmp_path,
) -> None:
    notes = CharactersRAGDB(tmp_path / "real-owner.sqlite", client_id="template")
    state = SyncStateRepository(tmp_path / "real-owner-sync.sqlite", client_id="sync")
    state.set_sync_v2_profile_state(
        server_profile_id="server-a",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id="device-a",
        dataset_id="dataset-a",
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, inventory_phase, updated_at) "
            "VALUES ('server-a', 'dataset-a', 'ready', 'ready', 'complete', '2026-08-29T00:00:00+00:00')"
        )
    local = NotesInteropService(
        base_db_directory=tmp_path,
        api_client_id="app",
        global_db_to_use=notes,
    )
    scope = NotesScopeService(local_notes_service=local, server_service=object())
    app = SimpleNamespace(
        active_server_id="server-a",
        chachanotes_db=notes,
        sync_state_repository=state,
        sync_v2_dataset_keys={"dataset-a": generate_dataset_key()},
        notes_scope_service=scope,
        local_first_sync_service=SimpleNamespace(
            notes_organization_repository=None, notes_organization_sync_service=None
        ),
        sync_restore_service=SimpleNamespace(notes_organization_repository=None),
    )
    _wire_notes_sync_services(app)
    app.notes_organization_sync_service.failure_injector = lambda stage: (
        (_ for _ in ()).throw(RuntimeError("injected"))
        if stage == "after_notes_mutation_and_intent"
        else None
    )

    with pytest.raises(RuntimeError, match="injected"):
        await scope.save_note(
            scope=ScopeType.LOCAL_NOTE,
            title="Rollback",
            content="Body",
            create_note_id="00000000-0000-4000-8000-000000000032",
            user_id="real-user",
            keywords=("agent-lesson",),
        )

    owner = local.notes_db("real-user")
    assert owner is not notes
    assert owner.get_note_by_id("00000000-0000-4000-8000-000000000032") is None
    assert (
        owner.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("profile_mode", [None, "local_only", "server_frontend"])
async def test_implicit_scope_preserves_legacy_local_notes_when_no_profile_is_eligible(
    tmp_path, profile_mode: str | None
) -> None:
    notes, state, scope = _wired_scope(tmp_path, suffix=str(profile_mode or "fresh"))
    if profile_mode is not None:
        state.set_sync_v2_profile_state(
            server_profile_id="inactive-server",
            authenticated_principal_id=None,
            workspace_scope=None,
            profile_mode=profile_mode,
            device_id=None,
            dataset_id=None,
        )
    note_id = f"legacy-{profile_mode or 'fresh'}"

    result = await scope.save_note(
        scope=ScopeType.LOCAL_NOTE,
        title="Legacy local",
        content="No synchronized profile",
        create_note_id=note_id,
        user_id="local-user",
        keywords=("legacy",),
    )

    assert result["id"] == note_id
    assert notes.get_note_by_id(note_id) is not None
    assert notes.get_keywords_for_note(note_id)[0]["keyword"] == "legacy"
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_implicit_scope_routes_the_only_eligible_local_first_profile(
    tmp_path,
) -> None:
    notes, state, scope = _wired_scope(tmp_path, suffix="one-active")
    _persist_ready_profile(notes, state, server_profile_id="server-active")
    state.set_sync_v2_profile_state(
        server_profile_id="server-inactive",
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="server_frontend",
        device_id=None,
        dataset_id=None,
    )
    note_id = "00000000-0000-4000-8000-000000000033"

    result = await scope.save_note(
        scope=ScopeType.LOCAL_NOTE,
        title="Synchronized",
        content="One active profile",
        create_note_id=note_id,
        user_id="local-user",
        keywords=("routed",),
    )

    assert result["id"] == note_id
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 2
    )


@pytest.mark.asyncio
async def test_implicit_scope_rejects_ambiguous_local_first_profiles_without_partial_write(
    tmp_path,
) -> None:
    notes, state, scope = _wired_scope(tmp_path, suffix="ambiguous")
    _persist_ready_profile(notes, state, server_profile_id="server-a")
    _persist_ready_profile(notes, state, server_profile_id="server-b")

    with pytest.raises(ValueError, match="multiple eligible Notes profile scopes"):
        await scope.save_note(
            scope=ScopeType.LOCAL_NOTE,
            title="Ambiguous",
            content="Must choose",
            create_note_id="note-ambiguous",
            user_id="local-user",
            keywords=("blocked",),
        )

    assert notes.get_note_by_id("note-ambiguous") is None
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )


@pytest.mark.asyncio
async def test_explicit_missing_profile_scope_rejects_without_partial_write(
    tmp_path,
) -> None:
    notes, _state, scope = _wired_scope(tmp_path, suffix="missing-explicit")

    with pytest.raises(ValueError, match="persisted Notes profile scope is required"):
        await scope.save_note(
            scope=ScopeType.LOCAL_NOTE,
            title="Missing",
            content="Explicit missing profile",
            create_note_id="note-missing-explicit",
            user_id="local-user",
            keywords=("blocked",),
            sync_v2_profile={"server_profile_id": "missing-server"},
        )

    assert notes.get_note_by_id("note-missing-explicit") is None
    assert (
        notes.get_connection()
        .execute("SELECT COUNT(*) FROM notes_organization_sync_intents")
        .fetchone()[0]
        == 0
    )


def _wired_scope(tmp_path, *, suffix: str):
    notes = CharactersRAGDB(tmp_path / f"notes-{suffix}.sqlite", client_id=suffix)
    state = SyncStateRepository(tmp_path / f"sync-{suffix}.sqlite", client_id=suffix)
    scope = NotesScopeService(
        local_notes_service=_LocalNotes(notes), server_service=object()
    )
    app = SimpleNamespace(
        active_server_id="server-a",
        chachanotes_db=notes,
        sync_state_repository=state,
        sync_v2_dataset_keys={},
        notes_scope_service=scope,
        local_first_sync_service=SimpleNamespace(
            notes_organization_repository=None, notes_organization_sync_service=None
        ),
        sync_restore_service=SimpleNamespace(notes_organization_repository=None),
    )
    _wire_notes_sync_services(app)
    return notes, state, scope


def _persist_ready_profile(
    notes: CharactersRAGDB,
    state: SyncStateRepository,
    *,
    server_profile_id: str,
) -> None:
    dataset_id = f"dataset-{server_profile_id}"
    state.set_sync_v2_profile_state(
        server_profile_id=server_profile_id,
        authenticated_principal_id=None,
        workspace_scope=None,
        profile_mode="local_first",
        device_id=f"device-{server_profile_id}",
        dataset_id=dataset_id,
    )
    with notes.transaction() as cursor:
        cursor.execute(
            "INSERT INTO notes_organization_sync_checkpoints("
            "server_profile_id, dataset_id, local_state, server_state, inventory_phase, updated_at) "
            "VALUES (?, ?, 'ready', 'ready', 'complete', '2026-08-29T00:00:00+00:00')",
            (server_profile_id, dataset_id),
        )
