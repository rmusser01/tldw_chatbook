"""Behavioral recovery evidence for installed operational owners."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_sync_capture_policy_keeps_runtime_authority_inactive():
    from tldw_chatbook.Sync_Interop.recovery import recovery_adapters

    adapters = recovery_adapters()
    assert adapters
    assert all(adapter.activation_required for adapter in adapters)


@pytest.mark.parametrize(
    "package",
    [
        "Sync_Interop",
        "Notifications",
        "Workspaces",
        "Subscriptions",
        "Kanban_Interop",
        "Widgets.Tamagotchi",
        "Scheduling",
        "Notes",
        "Agents",
        "MCP",
        "runtime_policy",
    ],
)
def test_recovery_import_is_declaration_only(package, tmp_path):
    code = f"""import importlib, sys
importlib.import_module("tldw_chatbook.{package}.recovery")
assert "tldw_chatbook.config" not in sys.modules, "recovery imported configuration bootstrap"
assert not any(k.endswith((".server_sync_service", ".notification_dispatch_service", ".registry_service", ".base_tamagotchi")) for k in sys.modules), "recovery imported runtime services"
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


from contextlib import closing
from dataclasses import replace
import importlib
import json
import sqlite3
from threading import Event

from Tests.Backup_Recovery.test_core_owners import application_authority
from tldw_chatbook.Backup_Recovery.models import (
    DiscoveryContext,
    DISCOVERY_CONTEXT_KEY,
    StorageItem,
)

# Actual installed constructors, not hand-built schema/version labels.
STORES = {
    "kanban": (
        "Kanban_Interop.local_kanban_service",
        "LocalKanbanService",
        "Kanban_Interop.recovery",
        "kanban.local",
        None,
    ),
    "workspaces": (
        "DB.Workspace_DB",
        "WorkspaceDB",
        "DB.recovery_operations",
        "db.workspaces",
        "workspaces_db_path",
    ),
    "agent_runs": (
        "DB.AgentRuns_DB",
        "AgentRunsDB",
        "DB.recovery_operations",
        "db.agent_runs",
        None,
    ),
    "subscriptions": (
        "DB.Subscriptions_DB",
        "SubscriptionsDB",
        "DB.recovery_operations",
        "db.subscriptions",
        "subscriptions_db_path",
    ),
    "scheduled_tasks": (
        "Scheduling.db.scheduled_tasks_db",
        "ScheduledTasksDB",
        "Scheduling.recovery",
        "db.scheduled_tasks",
        "scheduled_tasks_db_path",
    ),
    "notifications": (
        "Notifications.client_notifications_db",
        "ClientNotificationsDB",
        "Notifications.recovery",
        "notifications.client",
        "notifications_db_path",
    ),
    "events": (
        "Notifications.event_state_repository",
        "EventStateRepository",
        "Notifications.recovery",
        "runtime.event_state",
        None,
    ),
    "sync": (
        "Sync_Interop.sync_state_repository",
        "SyncStateRepository",
        "Sync_Interop.recovery",
        "runtime.sync_state",
        None,
    ),
    "file_notes": (
        "Notes.file_notes_replica",
        "FileNotesReplica",
        "Notes.recovery",
        "notes.file_notes",
        None,
    ),
}


def operational_adapter(name):
    row = STORES[name]
    return next(
        a
        for a in importlib.import_module("tldw_chatbook." + row[2]).recovery_adapters()
        if a.owner_id == row[3]
    )


@pytest.fixture(params=tuple(STORES))
def operational_store(request, tmp_path):
    name = request.param
    module, symbol, *_ = STORES[name]
    source = tmp_path / (name + ".db")
    constructor = getattr(importlib.import_module("tldw_chatbook." + module), symbol)
    store = constructor(db_path=source) if name == "kanban" else constructor(source)
    if name in {"kanban", "notifications", "events", "sync"}:
        # These installed owners now defer their actual schema until first use.
        store._ensure_schema()
    try:
        yield name, source, store
    finally:
        if hasattr(store, "close"):
            store.close()


def test_operational_schema_matches_actual_constructor(operational_store):
    name, source, store = operational_store
    with closing(sqlite3.connect(source)) as conn:
        schema = tuple(
            row[0]
            for row in conn.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if conn.execute(
            "SELECT 1 FROM sqlite_schema WHERE name='schema_version'"
        ).fetchone():
            version = conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()[0]
    if name == "kanban":
        with closing(sqlite3.connect(source)) as conn:
            version = int(
                conn.execute(
                    "SELECT value FROM local_kanban_schema_meta WHERE key='schema_version'"
                ).fetchone()[0]
            )
    adapter = operational_adapter(name)
    assert (version, schema) in adapter.schema_policy().schema_sql
    assert adapter.validate(source) == ()


def seed_operational(name, source, store):
    if name == "kanban":
        import asyncio

        asyncio.run(
            store.create_board(
                {"name": "Retained board", "client_id": "fixture-client"}
            )
        )
    elif name == "agent_runs":
        store.create_run(
            conversation_id="historical-conversation",
            agent_kind="primary",
            task="Preserve running history",
            budget={"tokens": 13},
        )
    elif name == "workspaces":
        from tldw_chatbook.Workspaces.registry_service import (
            LocalWorkspaceRegistryService,
        )

        service = LocalWorkspaceRegistryService(store)
        service.create_workspace(
            workspace_id="historical-workspace", name="Imported workspace"
        )
        root = source.parent / "external-workspace"
        root.mkdir()
        (root / "untouched.txt").write_bytes(b"External authority is never used")
        service.add_folder_binding("historical-workspace", root, allow_write=True)
    elif name == "subscriptions":
        store.add_subscription(
            "Inert subscription",
            "rss",
            "https://example.invalid/feed",
            auth_config={"keyring_scope": "historical-shared-scope"},
        )
    elif name == "scheduled_tasks":
        store.create_reminder_task(
            "historical-owner",
            "Queued reminder",
            schedule_kind="once",
            run_at="2099-01-01T00:00:00+00:00",
            enabled=True,
        )
    elif name == "notifications":
        store.insert_notification(
            category="automation",
            title="Durable notice",
            message="Retain notification",
            payload={"pending": True, "claim": "historical-device"},
        )
    elif name == "events":
        store.record_observer_status(
            source_authority="server",
            server_profile_id="historical-profile",
            authenticated_principal_id="historical-principal",
            stream_name="notes",
            stream_instance_id="historical-stream",
            status="running",
            details={"cursor": "cursor-23"},
        )
    elif name == "sync":
        store.record_identity_mapping(
            source_authority="server",
            server_profile_id="historical-profile",
            authenticated_principal_id="historical-principal",
            workspace_scope="historical-workspace",
            domain="notes",
            entity_type="note",
            local_entity_id="local-1",
            remote_entity_id="remote-1",
            mapping_status="confirmed",
            details={"device": "historical-device", "lease": "historical-lease"},
        )
    elif name == "file_notes":
        import hashlib

        data = b"---\r\nopaque: yes\r\n---\r\nPrivate recovery bytes\x00"
        store.upsert_file(
            "/historical/external",
            "note.md",
            data,
            content_hash=hashlib.sha256(data).hexdigest(),
            decoded_text=None,
            size=len(data),
            mtime_ns=123,
        )
        store.checkpoint(
            "/historical/external",
            "note.md",
            b"pre-edit\x00recovery",
            content_hash=hashlib.sha256(b"pre-edit\x00recovery").hexdigest(),
            session_key="old-editor-session",
        )
        store.protect("/historical/external", "note.md")
        store.mark_deleted("/historical/external", "note.md")


def dump(path):
    with closing(sqlite3.connect(path)) as conn:
        return tuple(conn.iterdump())


@pytest.mark.parametrize("name", tuple(STORES))
def test_operational_capture_keeps_history_and_does_not_replay(
    name, tmp_path, monkeypatch
):
    # Config-using owners must start with their actual selector. A cached config
    # owner cannot be rebound by the parent suite's per-test environment changes.
    source = tmp_path / (name + ".db")
    code = """import importlib, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from Tests.Backup_Recovery.test_operational_owners import STORES, seed_operational
name, source = sys.argv[1], Path(sys.argv[2])
module, symbol, *_ = STORES[name]
constructor = getattr(importlib.import_module('tldw_chatbook.' + module), symbol)
store = constructor(db_path=source) if name == 'kanban' else constructor(source)
try:
    seed_operational(name, source, store)
finally:
    if hasattr(store, 'close'):
        store.close()
assert not blocked_attempts()
"""
    subprocess.run(  # nosec B603: fixed interpreter/script; fixture values are argv.
        [sys.executable, "-c", code, name, str(source)],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    expected = dump(source)
    before = source.read_bytes()
    adapter = operational_adapter(name)
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    candidate = stage / "capture.db"
    item = StorageItem(
        adapter.owner_id, "profile:fixture:" + adapter.owner_id, source, "included", ()
    )
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(item, candidate, Event())
            adapter.relocate(
                candidate, {"/historical/external": tmp_path / "not-created"}
            )
    assert adapter.activation_required
    assert source.read_bytes() == before
    assert dump(candidate) == expected
    assert not (tmp_path / "not-created").exists()
    assert not any("RecoveryRequired" in line for line in expected)


@pytest.mark.parametrize(
    "case", ["no_authority", "cancelled", "wrong_owner", "existing", "wrong_schema"]
)
def test_operational_capture_refuses_invalid_authority_and_inputs(
    operational_store, tmp_path, monkeypatch, case
):
    name, source, store = operational_store
    if hasattr(store, "close"):
        store.close()
    adapter = operational_adapter(name)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    candidate = stage / "snapshot.db"
    item = StorageItem(
        adapter.owner_id, "profile:fixture:" + adapter.owner_id, source, "included", ()
    )
    cancel = Event()
    if case == "no_authority":
        with pytest.raises(ValueError, match="capture_requires_maintenance"):
            adapter.capture(item, candidate, cancel)
        return
    if case == "cancelled":
        cancel.set()
    if case == "wrong_owner":
        item = replace(item, owner="unknown")
    if case == "existing":
        candidate.write_bytes(b"existing stays")
    if case == "wrong_schema":
        with closing(sqlite3.connect(source)) as connection:
            connection.execute("CREATE TABLE unqualified(value TEXT)")
    authority = application_authority(tmp_path, source, monkeypatch)
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            with pytest.raises((ValueError, InterruptedError, FileExistsError)):
                adapter.capture(item, candidate, cancel)
    if case == "existing":
        assert candidate.read_bytes() == b"existing stays"


@pytest.fixture(params=["local", "targets", "context", "permissions"])
def mcp_store(request, tmp_path):
    name = request.param
    source = tmp_path / (name + ".json")
    if name == "local":
        from tldw_chatbook.MCP.local_store import LocalMCPStore, LocalMCPStoreState

        store = LocalMCPStore(source)
        value = LocalMCPStoreState()
        save = lambda: store.save(value)
    elif name == "targets":
        from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore

        store = ConfiguredServerTargetStore(source)
        save = lambda: store.save_targets([])
    elif name == "context":
        from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore
        from tldw_chatbook.MCP.unified_control_models import UnifiedMCPContext

        store = UnifiedMCPContextStore(source)
        save = lambda: store.save(UnifiedMCPContext())
    else:
        from tldw_chatbook.MCP.permission_store import MCPPermissionStore

        store = MCPPermissionStore(source)
        value = store.load()
        value["profiles"]["default"]["global_default"] = "allow"
        save = lambda: store.save(value)
    save()
    return name, source, store, save


def test_actual_mcp_writers_refuse_ordinary_mutation_during_maintenance(
    mcp_store, tmp_path, monkeypatch
):
    name, source, store, save = mcp_store
    authority = application_authority(tmp_path, source, monkeypatch)
    before = source.read_bytes()
    with authority.maintenance(("core", "bootstrap.unbound"), 1):
        with pytest.raises(RuntimeError):
            save()
    assert source.read_bytes() == before


def test_mcp_capture_retains_permission_history_without_loading_store(
    mcp_store, tmp_path, monkeypatch
):
    from tldw_chatbook.MCP.recovery import recovery_adapters

    name, source, store, save = mcp_store
    expected = source.read_bytes()
    adapter = next(a for a in recovery_adapters() if a.owner_id == "mcp." + name)
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    item = StorageItem(
        adapter.owner_id, "profile:fixture:" + adapter.owner_id, source, "included", ()
    )
    target = stage / "state.json"
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(item, target, Event())
            adapter.relocate(target, {"historical-scope": tmp_path / "new-scope"})
    assert target.read_bytes() == expected
    assert source.read_bytes() == expected
    assert adapter.activation_required
    assert not (tmp_path / "new-scope").exists()


def test_tree_inventory_preserves_empty_dirs_and_refuses_links(tmp_path):
    from tldw_chatbook.Backup_Recovery.file_inventory import inventory_tree

    root = tmp_path / "retained"
    root.mkdir(mode=0o700)
    (root / "empty").mkdir()
    (root / "payload").write_bytes(b"durable")
    external = tmp_path / "external"
    external.mkdir()
    (external / "never-enumerate").write_bytes(b"external")
    (root / "link").symlink_to(external, target_is_directory=True)
    entries = inventory_tree(root, owner="workspaces.change_tracking", external=False)
    assert (
        next(i for i in entries if i.path == root / "empty").status
        == "included_directory"
    )
    assert next(i for i in entries if i.path == root / "link").status == "unsupported"
    assert not any(i.path and i.path.name == "never-enumerate" for i in entries)
    assert all(i.dependencies for i in entries if i.path != root)


@pytest.mark.parametrize(
    "owner", ["workspaces.change_tracking", "agents.history", "subscriptions.assets"]
)
def test_opaque_capture_streams_real_bytes_and_preserves_empty_topology(
    owner, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration

    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    (root / "empty").mkdir()
    source = root / "payload"
    source.write_bytes(b"\x00\xffrecovery\n" * (110000))
    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "fixture")
    }
    adapter = _RawDeclaration(owner)
    entries = adapter._tree(config, root)
    item = next(i for i in entries if i.path == source)
    assert all(
        i.logical_id.startswith("profile:fixture:" + owner + ":") for i in entries
    )
    authority = application_authority(tmp_path, root, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    target = stage / "payload"
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(item, target, Event())
    assert source.read_bytes() == target.read_bytes()
    assert (
        next(i for i in entries if i.path == root / "empty").status
        == "included_directory"
    )


def test_opaque_checker_observes_actual_limit_and_cancel(tmp_path):
    from tldw_chatbook.Backup_Recovery.storage_admission import _check_recovery_file

    source = tmp_path / "payload"
    source.write_bytes(b"x" * 1048577)
    with pytest.raises(ValueError, match="byte_limit"):
        _check_recovery_file("agents.history", source, max_bytes=1048576)
    cancel = Event()
    cancel.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        _check_recovery_file("agents.history", source, max_bytes=1048577, cancel=cancel)


def test_aggregate_declared_tree_accepts_only_explicit_same_profile_topology(tmp_path):
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries
    from tldw_chatbook.Backup_Recovery.recovery_files import _RawDeclaration

    root = tmp_path / "owned"
    root.mkdir(mode=0o700)
    (root / "empty").mkdir()
    (root / "payload").write_bytes(b"keep")
    selector = tmp_path / "config.toml"
    selector.write_text('[general]\nusers_name="fixture"\n')
    config = {DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "fixture")}
    entries = _RawDeclaration("agents.history")._tree(config, root)
    config_item = StorageItem(
        "config", "profile:fixture:config", selector, "included", ()
    )
    assert classify_entries((config_item, *entries)).complete
    child = next(i for i in entries if i.path == root / "payload")
    for damaged in (
        replace(child, dependencies=(config_item.logical_id,)),
        replace(child, owner="other"),
        replace(child, logical_id=child.logical_id.replace("fixture", "other")),
    ):
        assert (
            "overlapping_owner_roots"
            in classify_entries(
                (config_item, *(damaged if i is child else i for i in entries))
            ).issues
        )


def test_subscription_site_config_manager_schema_is_qualified(tmp_path, monkeypatch):
    _run_stable_config_case(tmp_path, "_subscription_site_config_manager_roundtrip")


def _run_stable_config_case(tmp_path, name):
    """Exercise config-owning services before any per-test source retargeting."""
    script = """import sys
from pathlib import Path
import pytest
from Tests.Backup_Recovery import test_operational_owners as cases
with pytest.MonkeyPatch.context() as patches:
    getattr(cases, sys.argv[1])(Path(sys.argv[2]), patches)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, name, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _subscription_site_config_manager_roundtrip(tmp_path, monkeypatch):
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfigManager

    source = tmp_path / "subscriptions.db"
    store = SubscriptionsDB(source)
    store.add_subscription(
        "Retain site definitions", "rss", "https://example.invalid/feed"
    )
    store.close()
    manager = SiteConfigManager(str(source))
    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfig

    assert manager.save_config(
        SiteConfig("example.invalid", {"notes": "Historical site definition"})
    )
    manager.db.close()
    with closing(sqlite3.connect(source)) as connection:
        schema = tuple(
            r[0]
            for r in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
    assert operational_adapter("subscriptions").validate(source) == ()
    adapter = operational_adapter("subscriptions")
    expected = dump(source)
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "hybrid-stage"
    stage.mkdir(mode=0o700)
    candidate = stage / "hybrid.db"
    item = StorageItem(
        adapter.owner_id, "profile:fixture:" + adapter.owner_id, source, "included", ()
    )
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(item, candidate, Event())
    assert dump(candidate) == expected
    with closing(sqlite3.connect(candidate)) as connection:
        connection.execute(
            "UPDATE db_schema_version SET version=41 WHERE schema_name='rag_char_chat_schema'"
        )
        connection.commit()
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            assert adapter.validate(candidate) == ("unsupported_schema_version",)


def test_schema_catalog_alternatives_are_complete_and_ordered():
    from tldw_chatbook.DB.recovery_sqlite import _validate_sqlite

    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("CREATE TABLE a(value TEXT)")
        connection.execute("CREATE TABLE b(value TEXT)")
        schema = tuple(
            r[0]
            for r in connection.execute(
                "SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY type,name"
            )
        )
        assert _validate_sqlite(
            connection, (0,), ((0, schema[:1]), (0, schema[1:]))
        ) == ("unsupported_schema",)
        assert _validate_sqlite(connection, (0,), ((0, tuple(reversed(schema))),)) == (
            "unsupported_schema",
        )
        assert _validate_sqlite(connection, (0,), ((0, schema[:1]), (0, schema))) == ()


def test_core_notes_dependencies_use_qualified_exact_peer_ids(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.Notes.file_notes_replica import FileNotesReplica

    source = tmp_path / "core.db"
    store = CharactersRAGDB(source, "fixture")
    note = store.add_note("Bound note", "Retained")
    with store.transaction() as connection:
        connection.execute(
            "UPDATE notes SET file_path_on_disk='/historical/never-opened.md' WHERE id=?",
            (note,),
        )
    store.close()
    peer = tmp_path / "file_notes.sqlite"
    replica = FileNotesReplica(peer)
    replica.close()
    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "fixture"),
        "database": {"chachanotes_db_path": str(source)},
    }
    adapter = next(a for a in core_adapters() if a.owner_id == "db.chachanotes.primary")
    item = adapter.discover(config)[0]
    key = "profile:fixture:notes.file_notes"
    assert key in item.dependencies
    assert "profile:fixture:notes.sync_bindings" in item.dependencies
    assert adapter.validate_dependencies(item, source, {key: peer}) == ()
    assert adapter.validate_dependencies(
        item, source, {key.replace("fixture", "other"): peer}
    ) == ("dependency_unavailable",)
    assert adapter.validate_dependencies(item, source, {}) == (
        "dependency_unavailable",
    )


def test_note_managed_membership_capture_is_immutable_history(tmp_path, monkeypatch):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
    from Tests.Notes.test_note_folder_repository import _attach_membership
    from tldw_chatbook.Notes.recovery import recovery_adapters

    source = tmp_path / "notes.db"
    store = CharactersRAGDB(source, "fixture")
    repo = LocalNoteFolderRepository(store)
    manual = repo.create_folder(name="Manual", parent_id=None)
    managed = repo.create_folder(name="Managed", parent_id=None)
    note = store.add_note("Historical memberships", "Recovery content")
    _attach_membership(repo, folder_id=manual.folder_id, note_id=note)
    _attach_membership(
        repo,
        folder_id=managed.folder_id,
        note_id=note,
        ownership="managed",
        owner_id="old-device-root",
        owner_active=False,
    )
    store.close()
    expected = dump(source)
    adapter = next(
        a for a in recovery_adapters() if a.owner_id == "notes.sync_bindings"
    )
    authority = application_authority(tmp_path, source, monkeypatch)
    stage = tmp_path / "stage"
    stage.mkdir(mode=0o700)
    candidate = stage / "notes.db"
    item = StorageItem(
        adapter.owner_id, "profile:fixture:" + adapter.owner_id, source, "included", ()
    )
    with authority.maintenance(("core", "bootstrap.unbound"), 1) as session:
        with session.capture_scope((source,), stage):
            adapter.capture(item, candidate, Event())
            adapter.relocate(candidate, {"old-device-root": tmp_path / "never-created"})
    assert dump(candidate) == expected
    assert not (tmp_path / "never-created").exists()
    with closing(sqlite3.connect(candidate)) as connection:
        assert set(
            connection.execute(
                "SELECT ownership,owner_id,owner_active FROM note_folder_memberships"
            )
        ) == {("manual", "", 1), ("managed", "old-device-root", 0)}


def test_briefing_cleanup_cannot_mutate_owned_bytes_in_maintenance(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Subscriptions.briefing_audio import _remove_file_quietly

    source = tmp_path / "audio.wav"
    source.write_bytes(b"not replayed")
    authority = application_authority(tmp_path, source, monkeypatch)
    with authority.maintenance(("core", "bootstrap.unbound"), 1):
        with pytest.raises(RuntimeError):
            _remove_file_quietly(source)
    assert source.read_bytes() == b"not replayed"
    _remove_file_quietly(source)
    assert not source.exists()


def test_optional_pet_default_and_exact_backups_block_unknown_children(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Widgets.Tamagotchi.recovery import recovery_adapters

    monkeypatch.setenv("HOME", str(tmp_path))
    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "fixture")
    }
    adapter = recovery_adapters()[0]
    assert adapter.discover(config)[0].status == "unused"
    parent = tmp_path / ".config" / "tldw_chatbook"
    parent.mkdir(parents=True)
    (parent / "tamagotchi_pets.json").write_bytes(b"historical corrupt evidence")
    (parent / "tamagotchi_pets.backup_20260101_121314.json").write_bytes(
        b"older recovery bytes"
    )
    assert [i.status for i in adapter.discover(config)] == ["included", "included"]
    (parent / "unrecognized.sqlite").write_bytes(b"unknown state")
    assert "unsupported" in {i.status for i in adapter.discover(config)}


def test_dormant_tamagotchi_path_constructors_have_no_installed_call_sites():
    import ast

    root = Path(__file__).resolve().parents[2] / "tldw_chatbook"
    calls = []
    for path in root.rglob("*.py"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            ) in ("SQLiteStorage", "JSONStorage", "ConfigFileStorage", "NotesMirror"):
                calls.append((path.relative_to(root).as_posix(), node.lineno))
    assert calls == [], (
        "New durable pet wiring requires a locator and recovery qualification"
    )


def test_subscriptions_complete_audio_requires_exact_profile_payload(tmp_path):
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.recovery import recovery_adapters
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    base = tmp_path / "data"
    root = base / "fixture"
    root.mkdir(parents=True)
    source = root / "subscriptions.db"
    store = SubscriptionsDB(source)
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )

    watchlist = WatchlistBundleService(store).create(name="Retained watchlist")["id"]
    briefing = store.insert_briefing(watchlist)
    script = store.insert_briefing_script(
        briefing, preset_id=None, preset_name="Historical", roster_snapshot_json="{}"
    )
    audio = store.create_briefing_audio(script, voice_snapshot_json="{}")
    path = root / "briefing_audio" / f"script-{script}-audio-{audio}.wav"
    store.update_briefing_audio(audio, status="complete", file_path=str(path))
    store.close()
    selector = tmp_path / "config.toml"
    selector.write_text("")
    config = {
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(selector, "fixture"),
        "general": {"users_name": "fixture"},
        "paths": {"data_dir": str(base)},
        "database": {"subscriptions_db_path": str(source)},
    }
    db_adapter = operational_adapter("subscriptions")
    raw = recovery_adapters()[0]
    item = db_adapter.discover(config)[0]
    assert item.status == "included"
    config_item = StorageItem(
        "config", "profile:fixture:config", selector, "included", ()
    )
    assert not classify_entries((config_item, item, *raw.discover(config))).complete
    path.parent.mkdir()
    path.write_bytes(b"RIFF retained original audio")
    entries = raw.discover(config)
    assert classify_entries((config_item, item, *entries)).complete
    peer = next(e for e in entries if e.path == path)
    assert db_adapter.validate_dependencies(item, source, {peer.logical_id: path}) == ()
    assert db_adapter.validate_dependencies(
        item, source, {peer.logical_id.replace("fixture", "other"): path}
    ) == ("dependency_unavailable",)
    assert db_adapter.validate_dependencies(item, source, {}) == (
        "dependency_unavailable",
    )


def test_shadow_git_real_process_holds_ordinary_admission_until_exit(
    tmp_path, monkeypatch
):
    import shutil, time, threading
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

    entered = tmp_path / "entered"
    release = tmp_path / "release"
    executable = tmp_path / "git-wrapper"
    executable.write_text(
        "#!"
        + sys.executable
        + "\nimport os,time\nfrom pathlib import Path\nPath("
        + repr(str(entered))
        + ").touch()\nwhile not Path("
        + repr(str(release))
        + ").exists(): time.sleep(0.01)\nos.execv("
        + repr(shutil.which("git"))
        + ",["
        + repr(shutil.which("git"))
        + ']+__import__("sys").argv[1:])\n'
    )
    executable.chmod(0o700)
    work = tmp_path / "external"
    work.mkdir()
    (work / "unchanged").write_bytes(b"not captured")
    owned = tmp_path / "change_review"
    owned.mkdir(mode=0o700)
    repo = ShadowRepoService(
        data_dir=owned, git_executable=str(executable)
    ).repo_for_root(work)
    authority = application_authority(tmp_path, owned, monkeypatch)
    errors = []

    def invoke():
        try:
            repo._run("--version")
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=invoke)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not entered.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert entered.exists()
        with pytest.raises((TimeoutError, RuntimeError)):
            with authority.maintenance(("core", "bootstrap.unbound"), 0.1):
                pass
    finally:
        release.touch()
        worker.join(10)
    assert not worker.is_alive() and not errors
    with authority.maintenance(("core", "bootstrap.unbound"), 1):
        with pytest.raises(RuntimeError):
            repo.ensure_initialized()
    assert not repo.git_dir.exists()
    assert (work / "unchanged").read_bytes() == b"not captured"


@pytest.mark.parametrize(
    "package",
    [
        "Sync_Interop",
        "Notifications",
        "Workspaces",
        "Kanban_Interop",
        "Widgets.Tamagotchi",
    ],
)
def test_lazy_operational_public_exports_preserve_object_identity(package):
    module = importlib.import_module("tldw_chatbook." + package)
    for name, (relative, symbol) in module._EXPORTS.items():
        assert getattr(module, name) is getattr(
            importlib.import_module(relative, module.__name__), symbol
        )


@pytest.mark.parametrize("available", [True, False])
def test_subscriptions_lazy_optional_group_preserves_importerror_contract(available):
    # Only the optional dependency group is replaced; base exports are real.
    code = (
        """import importlib,sys,types
package=importlib.import_module('tldw_chatbook.Subscriptions')
assert 'tldw_chatbook.config' not in sys.modules
original=package.import_module
sentinels={}
def optional(name,base):
    if name in ('.monitoring_engine','.security'):
        if not AVAILABLE: raise ImportError('optional dependency unavailable')
        if name not in sentinels:
            sentinels[name]=types.SimpleNamespace(**{symbol:object() for module,symbol in package._OPTIONAL.values() if module==name})
        return sentinels[name]
    return original(name,base)
package.import_module=optional
assert package._CORE_AVAILABLE is AVAILABLE
assert set(package.__all__)==set(package._BASE)|(set(package._OPTIONAL) if AVAILABLE else set())
for name,(relative,symbol) in package._BASE.items():
    assert getattr(package,name) is getattr(original(relative,package.__name__),symbol)
if AVAILABLE:
    for name,(relative,symbol) in package._OPTIONAL.items():assert getattr(package,name) is getattr(sentinels[relative],symbol)
else:
    assert not hasattr(package,'FeedMonitor')
""".replace("is AVAILABLE", "is " + repr(available))
        .replace("if AVAILABLE", "if " + repr(available))
        .replace("not AVAILABLE", "not " + repr(available))
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=dict(os.environ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_registered_core_and_notes_cohort_preserves_proven_physical_identity(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import inventory, owner_registry
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.recovery_core import core_adapters
    from tldw_chatbook.Notes.recovery import recovery_adapters

    source = tmp_path / "notes.db"
    store = CharactersRAGDB(source, "fixture")
    store.close()
    selector = tmp_path / "config.toml"
    selector.write_text(
        "[database]\nchachanotes_db_path=" + json.dumps(str(source)) + "\n"
    )
    monkeypatch.setattr(owner_registry, "_adapters", {})
    owner_registry.register(core_adapters()[0])
    owner_registry.register(
        next(a for a in recovery_adapters() if a.owner_id == "notes.sync_bindings")
    )
    found = inventory.discover((selector,))
    cohort = [
        i
        for i in found.items
        if i.owner in ("db.chachanotes.primary", "notes.sync_bindings")
    ]
    assert len(cohort) == 2
    assert all(i.path == source and i.status == "included" for i in cohort)
    assert len({i.shared_group for i in cohort}) == 1
    assert not {
        "undeclared_alias",
        "shared_identity_mismatch",
        "invalid_shared_declaration",
    } & set(found.issues)
    damaged = tuple(
        replace(i, shared_group="forged") if i.owner == "notes.sync_bindings" else i
        for i in cohort
    )
    assert inventory._merge_chachanotes_cohort(damaged)[1] == (
        "invalid_shared_declaration",
    )


def test_real_pet_and_execution_history_writers_are_excluded(tmp_path, monkeypatch):
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import ConfigFileStorage
    from tldw_chatbook.MCP.execution_log import MCPExecutionLog
    from Tests.MCP.test_execution_log import _record

    monkeypatch.setenv("HOME", str(tmp_path))
    pet = ConfigFileStorage()
    assert pet.save("pet", {"name": "Retained pet"})
    history = MCPExecutionLog(pet.filepath.parent / "mcp_execution_log.jsonl")
    history.append(_record("historical-tool"))
    before = {p: p.read_bytes() for p in pet.filepath.parent.iterdir() if p.is_file()}
    authority = application_authority(tmp_path, pet.filepath.parent, monkeypatch)
    with authority.maintenance(("core", "bootstrap.unbound"), 1):
        for write in (
            lambda: pet.save("pet", {"name": "Changed"}),
            lambda: history.append(_record("new-tool")),
            lambda: history.read_recent(),
        ):
            with pytest.raises(RuntimeError):
                write()
    assert {p: p.read_bytes() for p in before} == before


def test_run_log_real_bound_writer_is_excluded(tmp_path, monkeypatch):
    _run_log_real_bound_writer_is_excluded(tmp_path, monkeypatch)


def _run_log_real_bound_writer_is_excluded(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run
    from Tests.Backup_Recovery.test_runtime_startup_handoff import _SCRIPT

    # Config imports retain process-startup admission. Only the actual app's
    # settled coordinator can yield it for native maintenance; this fixture
    # therefore uses the existing full handoff instead of retargeting a live
    # config source or manually closing its startup lease.
    setup = '''
    from pathlib import Path
    from tldw_chatbook.Agents.run_log import RunLogWriter
    log_root = Path.home() / "run-log-fixture"
    log_root.mkdir()
    writer = RunLogWriter(root=log_root, dir_name="agent-runs")
    writer.bind("historical-run")
    assert writer.is_active
    assert writer.append(run_id="historical-run", kind="primary", type="model",
                         content="Retained body") == 1
    before = {p: p.read_bytes() for p in log_root.rglob("*") if p.is_file()}
    assert before
'''
    refused = '''
        # The harness has entered real exclusive native maintenance and
        # positively retired startup through the installed runtime owner.
        assert writer.append(run_id="historical-run", kind="primary", type="model",
                             content="Not admitted") is None
        assert not writer.is_active
        writer.close()
        assert {p: p.read_bytes() for p in before} == before
'''
    assert _SCRIPT.count("    runtime = RuntimeMaintenance(app)") == 1
    assert _SCRIPT.count("        assert not storage._startups") == 1
    script = _SCRIPT.replace(
        "    runtime = RuntimeMaintenance(app)",
        setup + "\n    runtime = RuntimeMaintenance(app)",
    ).replace(
        "        assert not storage._startups",
        "        assert not storage._startups\n" + refused,
    )
    _run(tmp_path, "startup", "resume", script=script)


def test_pet_constructor_does_not_create_parent_before_admission(tmp_path, monkeypatch):
    from tldw_chatbook.Widgets.Tamagotchi.tamagotchi_storage import JSONStorage

    owned = tmp_path / "pets"
    owned.mkdir()
    target = owned / "new" / "pet.json"
    authority = application_authority(tmp_path, owned, monkeypatch)
    with authority.maintenance(("core", "bootstrap.unbound"), 1):
        with pytest.raises(RuntimeError):
            JSONStorage(str(target))
    assert not target.parent.exists()


def test_current_notes_device_authority_is_excluded_from_inventory(tmp_path):
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
    from tldw_chatbook.Notes.notes_device_state_store import (
        NotesDeviceStateStore,
        NotesSyncStoreSetting,
    )
    from tldw_chatbook.Notes.recovery import recovery_adapters

    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    config = {"paths": {"data_dir": str(data)}, "general": {"users_name": "fixture"}}
    profile = user_data_dir(config)
    profile.mkdir(mode=0o700)
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
        tmp_path / "config.toml", "fixture"
    )
    database = profile / "tldw_chatbook_notes_sync_state.db"
    store = NotesDeviceStateStore(database)
    try:
        store.initialize()
        store.set_setting(
            NotesSyncStoreSetting("cutover_marker", "captured-device-authority")
        )
        adapter = next(
            a for a in recovery_adapters() if a.owner_id == "notes.sync_state"
        )
        items = adapter.discover(config)
        assert {item.path for item in items} == {
            Path(str(database) + suffix) for suffix in ("", "-wal", "-shm", "-journal")
        }
        assert all(item.status == "intentionally_excluded" for item in items)
        assert adapter.schema_policy() is None
        destination = tmp_path / "copied-authority.db"
        with pytest.raises(ValueError, match="notes_device_state_excluded"):
            adapter.capture(items[0], destination, Event())
        assert not destination.exists()
    finally:
        store.close()


@pytest.mark.parametrize("kind", ["symlink", "directory", "hardlink"])
def test_excluded_notes_device_authority_still_refuses_unsafe_entries(tmp_path, kind):
    from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir
    from tldw_chatbook.Notes.recovery import recovery_adapters

    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    config = {"paths": {"data_dir": str(data)}, "general": {"users_name": "fixture"}}
    profile = user_data_dir(config)
    profile.mkdir(mode=0o700)
    config[DISCOVERY_CONTEXT_KEY] = DiscoveryContext(
        tmp_path / "config.toml", "fixture"
    )
    database = profile / "tldw_chatbook_notes_sync_state.db"
    outside = tmp_path / "outside.db"
    outside.write_bytes(b"keep")
    if kind == "symlink":
        database.symlink_to(outside)
    elif kind == "hardlink":
        os.link(outside, database)
    else:
        database.mkdir()
    adapter = next(a for a in recovery_adapters() if a.owner_id == "notes.sync_state")
    main = next(item for item in adapter.discover(config) if item.path == database)
    assert main.status not in {"included", "intentionally_excluded", "unused"}
    assert outside.read_bytes() == b"keep"
