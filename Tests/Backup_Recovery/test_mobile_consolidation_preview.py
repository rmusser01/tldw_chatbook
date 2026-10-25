"""Read-only physical legacy mobile review before any import is permitted."""


def test_absent_mobile_source_preview_does_not_create_databases(tmp_path):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
        preview_mobile_consolidation,
    )

    target = tmp_path / "canonical"
    target.mkdir(mode=0o700)
    stores = CanonicalConsoleStores(
        chat_path=target / "chat.db",
        workspace_path=target / "workspaces.db",
        run_path=target / "agent_runs.db",
    )
    source = tmp_path / "mobile"
    preview = preview_mobile_consolidation(source, stores)
    assert not preview.has_data
    assert not source.exists()
    assert not stores.chat_path.exists()
    assert not stores.workspace_path.exists()
    assert not stores.run_path.exists()


def test_preview_preserves_native_rows_without_writer_constructors(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import (
        fingerprints,
        seed_legacy_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
        preview_mobile_consolidation,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    source = tmp_path / "mobile"
    expected = seed_legacy_mobile(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("preview invoked a writer constructor")

    for cls in (AgentRunsDB, CharactersRAGDB, WorkspaceDB):
        monkeypatch.setattr(cls, "__init__", forbidden)
    target = tmp_path / "canonical"
    stores = CanonicalConsoleStores(
        target / "chat.db", target / "workspaces.db", target / "runs.db"
    )
    preview = preview_mobile_consolidation(source, stores)
    assert not preview.issues
    assert preview.has_data
    assert fingerprints(source) == expected["fingerprints"]
    assert ("chat.db", "messages", 3) in preview.counts
    assert any(
        expected["artifact_id"] in row
        for leaf, table, rows in preview.mappings
        for row in rows
    )
    assert "private fixture prompt" not in repr(preview)
    assert preview.rollback_scope == stores.paths


def _stores(tmp_path):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
    )

    return CanonicalConsoleStores(
        tmp_path / "canonical" / "chat.db",
        tmp_path / "canonical" / "workspaces.db",
        tmp_path / "canonical" / "agent_runs.db",
    )


def test_destination_identity_survives_creation_with_changed_physical_witnesses(
    tmp_path,
):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import (
        seed_canonical_console,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    stores = _stores(tmp_path)
    before = preview_mobile_consolidation(tmp_path / "absent", stores)
    identity = stores.identity
    seed_canonical_console(stores)
    after = preview_mobile_consolidation(tmp_path / "absent", stores)
    assert stores.identity == identity
    assert before.destination_witnesses != after.destination_witnesses
    assert not (tmp_path / "absent").exists()


def test_preview_refuses_missing_cross_store_reference(tmp_path):
    import sqlite3

    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    with sqlite3.connect(source / "runs.db") as connection:
        connection.execute(
            "UPDATE agent_runs SET parent_run_id='missing' WHERE parent_run_id IS NOT NULL"
        )
    preview = preview_mobile_consolidation(source, _stores(tmp_path))
    assert (
        "mobile_missing_dependency:runs.db.agent_runs.parent_run_id" in preview.issues
    )


def test_preview_refuses_unknown_populated_schema_and_preserves_bytes(tmp_path):
    import sqlite3

    from Tests.Backup_Recovery.mobile_consolidation_fixture import (
        fingerprints,
        seed_legacy_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    with sqlite3.connect(source / "runs.db") as connection:
        connection.execute("CREATE TABLE future_private_data(body TEXT)")
        connection.execute(
            "INSERT INTO future_private_data VALUES ('private unknown body')"
        )
    before = fingerprints(source)
    preview = preview_mobile_consolidation(source, _stores(tmp_path))
    assert "unsupported_schema" in preview.issues
    assert fingerprints(source) == before
    assert "private unknown body" not in repr(preview)


def test_preview_refuses_unsupported_version(tmp_path):
    import sqlite3

    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    with sqlite3.connect(source / "runs.db") as connection:
        connection.execute("INSERT INTO schema_version VALUES (999)")
    assert (
        "unsupported_schema_version"
        in preview_mobile_consolidation(source, _stores(tmp_path)).issues
    )


def test_preview_refuses_source_destination_alias(tmp_path):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    stores = CanonicalConsoleStores(
        source / "chat.db", tmp_path / "ws.db", tmp_path / "runs.db"
    )
    assert "mobile_unsafe_alias" in preview_mobile_consolidation(source, stores).issues
    link = tmp_path / "linked"
    link.symlink_to(source, target_is_directory=True)
    assert (
        "mobile_unsafe_alias"
        in preview_mobile_consolidation(link, _stores(tmp_path)).issues
    )


def test_preview_covers_referenced_artifact_bytes(tmp_path):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seeded = seed_legacy_mobile(source)
    preview = preview_mobile_consolidation(source, _stores(tmp_path))
    assert not preview.issues
    assert seeded["artifact_file"] in preview.capture_scope
    assert any(
        table == "console_trace_surface_nodes" and "artifact_id" in column
        for _, table, column, _, _ in preview.dependencies
    )


def test_unknown_non_database_source_content_blocks_even_without_stores(tmp_path):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    source.mkdir(mode=0o700)
    (source / "unrecognized.bin").write_bytes(b"keep")
    preview = preview_mobile_consolidation(source, _stores(tmp_path))
    assert preview.has_data
    assert "mobile_unknown_source_content" in preview.issues


def _config(tmp_path, name, canonical):
    path = tmp_path / (name + ".toml")
    path.write_text(
        f'[general]\nusers_name="shared"\n[paths]\ndata_dir="{tmp_path / "data"}"\n[database]\nchachanotes_db_path="{canonical / "chat.db"}"\nworkspaces_db_path="{canonical / "workspaces.db"}"\n'
    )
    path.chmod(0o600)
    return path


def test_two_configs_share_physical_source_but_require_destination_review(tmp_path):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    source = tmp_path / "data" / "shared" / "mobile"
    seed_legacy_mobile(source)
    first = _config(tmp_path, "first", tmp_path / "first")
    second = _config(tmp_path, "second", tmp_path / "second")
    service = RecoveryService(tmp_path / "control")
    try:
        preview = service.preview_mobile_consolidation(first, second)
    finally:
        service.close()
    assert preview.source_dir == source
    assert preview.destination.chat_path == tmp_path / "second" / "chat.db"
    assert preview.needs_destination_review
    assert "mobile_destination_review_required" in preview.issues
    assert len(preview.candidate_destinations) == 2


def test_empty_cli_preview_does_not_create_source_or_destination(tmp_path, capsys):
    import json

    from tldw_chatbook.Backup_Recovery.launcher import recovery_main

    config = _config(tmp_path, "selected", tmp_path / "canonical")
    assert (
        recovery_main(
            [
                "--control-root",
                str(tmp_path / "control"),
                "consolidate-mobile",
                "--source-config",
                str(config),
                "--target-config",
                str(config),
                "--preview",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["has_data"] is False
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "canonical").exists()
    assert not (tmp_path / "control").exists()


def test_preview_refuses_duplicate_absent_destinations(tmp_path):
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        CanonicalConsoleStores,
        preview_mobile_consolidation,
    )

    path = tmp_path / "same.db"
    assert (
        "mobile_unsafe_alias"
        in preview_mobile_consolidation(
            tmp_path / "absent", CanonicalConsoleStores(path, path, path)
        ).issues
    )


def test_preview_refuses_missing_referenced_artifact_root(tmp_path):
    import shutil

    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    expected = seed_legacy_mobile(source)
    shutil.rmtree(expected["artifact_file"].parents[2])
    assert (
        "mobile_missing_file_dependency"
        in preview_mobile_consolidation(source, _stores(tmp_path)).issues
    )


def test_unknown_mobile_content_blocks_partial_capture(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import bootstrap, owner_registry
    from tldw_chatbook.Backup_Recovery.capture_service import preview_capture

    config = _config(tmp_path, "selected", tmp_path / "canonical")
    source = tmp_path / "data" / "shared" / "mobile"
    source.mkdir(parents=True, mode=0o700)
    (source / "unknown").write_bytes(b"must retain")
    monkeypatch.setattr(
        bootstrap, "default_bootstrap_root", lambda: tmp_path / "control"
    )
    monkeypatch.setattr(owner_registry, "_adapters", {})
    preview = preview_capture((config,), options={"allow_partial": True})
    assert "mobile_review_required" in preview.issues


def test_source_fingerprint_tracks_content_without_exposing_bodies(tmp_path):
    import sqlite3

    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    before = preview_mobile_consolidation(source, _stores(tmp_path))
    with sqlite3.connect(source / "runs.db") as connection:
        connection.execute(
            "UPDATE agent_runs SET result=? WHERE status=?",
            ("changed private body", "done"),
        )
    after = preview_mobile_consolidation(source, _stores(tmp_path))
    assert not after.issues
    assert before.source_fingerprint != after.source_fingerprint
    assert before.mappings == after.mappings
    assert "changed private body" not in repr(after)


def test_preview_refuses_hardlinked_source_and_preserves_mode(tmp_path):
    import os

    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    os.link(source / "runs.db", tmp_path / "alias.db")
    before = (source / "runs.db").stat().st_mode
    assert (
        "mobile_unsafe_alias"
        in preview_mobile_consolidation(source, _stores(tmp_path)).issues
    )
    assert (source / "runs.db").stat().st_mode == before


def test_preview_uses_installed_workspace_binding_for_artifact_closure(tmp_path):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.models import WorkspaceRuntimeBinding
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    source = tmp_path / "profile" / "mobile"
    seeded = seed_legacy_mobile(source)
    root = tmp_path / "project"
    artifact = root / ".agent-runs" / seeded["child_run_id"] / "events.jsonl"
    artifact.parent.mkdir(parents=True, mode=0o700)
    artifact.write_bytes(b"retained bound workspace artifact")
    artifact.chmod(0o600)
    workspace = WorkspaceDB(source / "workspaces.db", "binding-fixture")
    try:
        LocalWorkspaceRegistryService(workspace).save_runtime_binding(
            WorkspaceRuntimeBinding(
                binding_id="binding",
                workspace_id=seeded["workspace_id"],
                binding_kind="local-filesystem",
                label="Bound folder",
                locator=str(root),
                status="ready",
            )
        )
    finally:
        workspace.close()
    preview = preview_mobile_consolidation(source, _stores(tmp_path))
    assert not preview.issues
    assert artifact in preview.capture_scope


def test_preview_lists_conflicting_destination_ids_without_title_equivalence(tmp_path):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import (
        fingerprints,
        seed_canonical_console,
        seed_legacy_mobile,
    )
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    expected = seed_legacy_mobile(source)
    destination = _stores(tmp_path)
    seed_canonical_console(destination)
    preview = preview_mobile_consolidation(source, destination)
    assert not preview.issues
    assert (
        "workspaces.db",
        "workspace_records",
        tuple(
            sorted((("conflicting-workspace",), (expected["default_workspace_id"],)))
        ),
    ) in preview.identity_conflicts
    assert any(
        leaf == "chat.db" and table == "conversations" and ("conversation",) in ids
        for leaf, table, ids in preview.identity_conflicts
    )
    assert fingerprints(source) == expected["fingerprints"]


def test_explicit_source_path_is_not_reconstructed_from_a_username(tmp_path):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "moved profile with spaces" / "mobile"
    expected = seed_legacy_mobile(source)
    preview = preview_mobile_consolidation(source, _stores(tmp_path))
    assert not preview.issues
    assert preview.source_dir == source
    assert expected["artifact_file"] in preview.capture_scope


def test_source_fingerprint_detects_directory_replacement_with_same_files(tmp_path):
    from Tests.Backup_Recovery.mobile_consolidation_fixture import seed_legacy_mobile
    from tldw_chatbook.Backup_Recovery.mobile_consolidation import (
        preview_mobile_consolidation,
    )

    source = tmp_path / "mobile"
    seed_legacy_mobile(source)
    before = preview_mobile_consolidation(source, _stores(tmp_path))
    old = tmp_path / "old-source"
    source.rename(old)
    source.mkdir(mode=0o700)
    for path in old.iterdir():
        path.rename(source / path.name)
    after = preview_mobile_consolidation(source, _stores(tmp_path))
    assert not after.issues
    assert before.source_fingerprint != after.source_fingerprint
