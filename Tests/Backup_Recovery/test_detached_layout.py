"""Detached import validates acquired bytes without opening historical paths."""

import importlib
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path
from threading import Event

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    import tempfile

    home = tmp_path / "recovery-home"
    config = home / ".config" / "tldw_cli" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text("")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(temporary))


def _layout():
    return importlib.import_module("tldw_chatbook.Backup_Recovery.detached_layout")


def _research(path):
    from tldw_chatbook.Research_Interop.recovery import recovery_adapters

    owner = recovery_adapters()[0]
    with sqlite3.connect(path) as connection:
        for statement in owner.schema_policy().schema_sql[0][1]:
            if not statement.startswith("CREATE TABLE sqlite_sequence"):
                connection.execute(statement)
        connection.execute(
            "INSERT INTO research_runs(id,query,created_at,updated_at) VALUES ('kept','nebula','now','now')"
        )
    return owner


def test_normalization_retains_committed_wal_and_leaves_acquired_set_unchanged(
    tmp_path,
):
    source = tmp_path / "source.sqlite"
    owner = _research(source)
    connection = sqlite3.connect(source)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA wal_autocheckpoint=0")
    connection.execute("UPDATE research_runs SET query='committed WAL'")
    connection.commit()
    acquired = tmp_path / "acquired.sqlite"
    for suffix in ("", "-wal", "-shm"):
        shutil.copyfile(Path(str(source) + suffix), Path(str(acquired) + suffix))
    connection.close()
    before = {p: p.read_bytes() for p in tmp_path.glob("acquired.sqlite*")}
    normalized = tmp_path / "normalized.sqlite"

    _layout()._normalize_sqlite(owner, acquired, normalized, Event())

    with sqlite3.connect(normalized) as restored:
        assert restored.execute("SELECT query FROM research_runs").fetchone() == (
            "committed WAL",
        )
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not Path(str(normalized) + "-wal").exists()


@pytest.mark.parametrize("damage", ["rollback", "wal"])
def test_normalization_refuses_ambiguous_or_damaged_sidecars(tmp_path, damage):
    source = tmp_path / "source.sqlite"
    owner = _research(source)
    Path(str(source) + ("-journal" if damage == "rollback" else "-wal")).write_bytes(
        b"damaged"
    )
    with pytest.raises(ValueError, match="detached_sqlite"):
        _layout()._normalize_sqlite(owner, source, tmp_path / "normalized", Event())


def test_unknown_durable_file_refuses_before_packaging(tmp_path):
    source = tmp_path / "Data"
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('[general]\nusers_name="fixture"\n')
    (source / "portable-location.json").write_text(
        json.dumps({"version": 1, "root": str(tmp_path / "original-unavailable")})
    )
    root = source / "home" / ".local" / "share" / "tldw_cli" / "fixture"
    root.mkdir(parents=True)
    (root / "unknown-durable.bin").write_bytes(b"must remain visible")
    with pytest.raises(ValueError, match="detached_unknown_content"):
        _layout().produce_detached_archive(source, tmp_path / "work", Event())


def _installed_database(owner, path):
    policy = owner.schema_policy()
    version, statements = policy.schema_sql[0]
    with sqlite3.connect(path) as connection:
        from tldw_chatbook.DB.canvas_payload_validation import (
            install_canvas_revision_payload_validator,
        )

        install_canvas_revision_payload_validator(connection)
        for statement in sorted(
            statements,
            key=lambda sql: (
                0 if sql.startswith(("CREATE TABLE", "CREATE VIRTUAL TABLE")) else 1
            ),
        ):
            if statement.startswith("CREATE TABLE sqlite_sequence"):
                continue
            try:
                connection.execute(statement)
            except sqlite3.OperationalError as error:
                if "already exists" not in str(error):
                    raise
        if owner.owner_id == "db.chachanotes.primary":
            connection.execute(
                "INSERT INTO db_schema_version(schema_name,version) VALUES ('rag_char_chat_schema',?)",
                (version,),
            )
            connection.execute(
                "INSERT INTO notes(id,title,content) VALUES ('kept','Nebula','Copied note')"
            )
            connection.execute(
                "INSERT INTO conversations(id,root_id,title,client_id) VALUES ('chat','chat','Copied chat','fixture')"
            )
        elif owner.owner_id.startswith("tts."):
            connection.execute(f"PRAGMA user_version={version}")
        else:
            connection.execute(
                "INSERT INTO schema_version(version) VALUES (?)", (version,)
            )


def _base_profile(tmp_path, *, core_owners=None):
    from tldw_chatbook.DB.recovery_core import core_adapters

    source = tmp_path / "Data"
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text(
        '[general]\nusers_name="fixture"\n[api_settings.openai]\napi_key="synthetic-secret"\n'
    )
    old = tmp_path / "original-unavailable"
    (source / "portable-location.json").write_text(
        json.dumps({"version": 1, "root": str(old)})
    )
    root = source / "home" / ".local" / "share" / "tldw_cli" / "fixture"
    root.mkdir(parents=True)
    from tldw_chatbook.Backup_Recovery.profile_paths import database_leaf

    for owner in core_adapters():
        if core_owners is not None and owner.owner_id not in core_owners:
            continue
        _installed_database(owner, root / database_leaf(owner.setting_name))
    (root / "chat_dicts").mkdir()
    (root / "chat_dicts" / "sample.txt").write_text("representative owned definition")
    return source, root, old


def test_complete_base_payload_forms_partial_archive_without_credentials(
    tmp_path, monkeypatch
):
    source, _root, old = _base_profile(tmp_path)
    original_stat = Path.stat

    def no_old_target(path, *args, **kwargs):
        assert not path.is_relative_to(old), "historical path was observed"
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", no_old_target)
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = _layout().produce_detached_archive(
        source.absolute(), (tmp_path / "work").absolute(), Event()
    )
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["consistency"] == "partial"
        assert manifest["credential_policy"] == "exclude"
        owners = {row["owner_id"] for row in manifest["files"]}
        assert {
            "config",
            "db.chachanotes.primary",
            "db.media.primary",
            "db.prompts.primary",
            "db.library_collections",
            "chat.dictionaries",
        } <= owners
        config = next(row for row in manifest["files"] if row["owner_id"] == "config")
        assert b"synthetic-secret" not in archive.read(config["payload"])
        notes = next(
            row
            for row in manifest["files"]
            if row["owner_id"] == "db.chachanotes.primary"
        )
        restored = tmp_path / "restored.sqlite"
        restored.write_bytes(archive.read(notes["payload"]))
    with sqlite3.connect(restored) as connection:
        assert connection.execute(
            "SELECT content FROM notes WHERE id='kept'"
        ).fetchone() == ("Copied note",)
        assert connection.execute(
            "SELECT title FROM conversations WHERE id='chat'"
        ).fetchone() == ("Copied chat",)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert any("credentials" in line.lower() for line in result.review_lines)
    from tldw_chatbook.Backup_Recovery.archive_reader import acquire
    from tldw_chatbook.Backup_Recovery.destinations import resolve_destinations
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    sealed = acquire(
        result.archive, tmp_path / "inspection", ArchiveLimits(), None, Event()
    )
    profile = manifest["profile_ids"][0]
    plan = resolve_destinations(
        sealed,
        mode="isolated",
        profile_bases={profile: tmp_path / "restored-profile"},
        external_destinations={},
        target=None,
        profile_names={profile: "Restored"},
    )
    assert plan.mode == "isolated"
    assert len(plan.restore) >= len(manifest["files"])


def test_external_database_selector_refuses_without_observing_target(
    tmp_path, monkeypatch
):
    source, _root, _old = _base_profile(tmp_path)
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    with config.open("a") as stream:
        stream.write(
            "[database]\nchachanotes_db_path="
            + json.dumps(str(tmp_path / "external" / "private.db"))
            + "\n"
        )
    with pytest.raises(ValueError, match="detached_external_storage_unsupported"):
        _layout().produce_detached_archive(
            source.absolute(), (tmp_path / "work").absolute(), Event()
        )


@pytest.mark.parametrize("tampered", [False, True, "unlisted_reference"])
def test_only_exact_installed_builtin_asset_support_is_imported(tmp_path, tampered):
    source, root, _old = _base_profile(tmp_path)
    assets = Path(__file__).parents[2] / "tldw_chatbook" / "assets"
    asset = json.loads(
        (assets / "characters/samira/visual_identity_pack.json").read_text()
    )["assets"][0]
    if tampered == "unlisted_reference":
        import hashlib

        license_file = assets / "characters/samira/ASSET_LICENSE.md"
        asset = {
            **asset,
            "storage_relpath": "characters/samira/ASSET_LICENSE.md",
            "bytes": license_file.stat().st_size,
            "sha256": hashlib.sha256(license_file.read_bytes()).hexdigest(),
        }
    digest = "0" * 64 if tampered else asset["sha256"]
    if tampered == "unlisted_reference":
        digest = asset["sha256"]
    with sqlite3.connect(root / "tldw_chatbook_ChaChaNotes.db") as connection:
        connection.execute(
            "INSERT INTO visual_identity_packs(id,owner_user_id,title,source_kind) VALUES (1,0,'Bundled','builtin')"
        )
        connection.execute(
            "INSERT INTO visual_identity_pack_versions(id,pack_id,owner_user_id,version_number,manifest_json) VALUES (1,1,0,1,'{}')"
        )
        connection.execute(
            "INSERT INTO visual_identity_assets(owner_user_id,pack_id,pack_version_id,expression_key,source_filename,storage_relpath,content_type,bytes,sha256,width,height) VALUES (0,1,1,?,?,?,?,?,?,?,?)",
            (
                asset["expression_key"],
                "asset.webp",
                asset["storage_relpath"],
                asset["content_type"],
                asset["bytes"],
                digest,
                asset["width"],
                asset["height"],
            ),
        )
    if tampered:
        with pytest.raises(
            ValueError,
            match="detached_owner_unavailable:persona.visual_identity_builtin",
        ):
            _layout().produce_detached_archive(
                source.absolute(), (tmp_path / "work").absolute(), Event()
            )
        return
    result = _layout().produce_detached_archive(
        source.absolute(), (tmp_path / "work").absolute(), Event()
    )
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        included = [
            row
            for row in manifest["files"]
            if row["owner_id"] == "persona.visual_identity_builtin"
        ]
        assert len(included) == 1
        assert included[0]["sha256"] == asset["sha256"]
    assert any("installed" in line.lower() for line in result.review_lines)


def test_external_workspace_metadata_remains_inert_without_target_io(
    tmp_path, monkeypatch
):
    source, root, old = _base_profile(tmp_path)
    from tldw_chatbook.DB.recovery_operations import recovery_adapters

    owner = next(
        owner for owner in recovery_adapters() if owner.owner_id == "db.workspaces"
    )
    workspace = root / "tldw_chatbook_workspaces.db"
    _installed_database(owner, workspace)
    locator = str(old / "external-workspace")
    with sqlite3.connect(workspace) as connection:
        connection.execute(
            "INSERT INTO workspace_records(workspace_id,name,description,authority,sync_status,active,archived,created_at,updated_at) VALUES ('workspace','Retained','','local','local',0,0,'now','now')"
        )
        connection.execute(
            "INSERT INTO workspace_runtime_bindings(binding_id,workspace_id,binding_kind,label,locator,status,metadata_json,created_at,updated_at) VALUES ('binding','workspace','local_filesystem','External',?,'available','{}','now','now')",
            (locator,),
        )
    original_stat = Path.stat

    def no_old_target(path, *args, **kwargs):
        assert not path.is_relative_to(old), "historical workspace was observed"
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", no_old_target)
    result = _layout().produce_detached_archive(
        source.absolute(), (tmp_path / "work").absolute(), Event()
    )
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        payload = next(
            row for row in manifest["files"] if row["owner_id"] == "db.workspaces"
        )
        detached = tmp_path / "workspace-copy.sqlite"
        detached.write_bytes(archive.read(payload["payload"]))
    with sqlite3.connect(detached) as connection:
        assert connection.execute(
            "SELECT locator FROM workspace_runtime_bindings"
        ).fetchone() == (locator,)


def test_uninstalled_sqlite_trigger_refuses_before_archive_publication(tmp_path):
    source = tmp_path / "source.sqlite"
    owner = _research(source)
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TRIGGER injected AFTER UPDATE ON research_runs BEGIN SELECT load_extension('never'); END"
        )
    with pytest.raises(ValueError, match="unsupported_schema"):
        _layout()._normalize_sqlite(
            owner, source, tmp_path / "normalized.sqlite", Event()
        )
    assert not (tmp_path / "normalized.sqlite").exists()


def test_missing_explicit_custom_store_is_never_replaced_with_an_empty_default(
    tmp_path,
):
    source, _root, old = _base_profile(tmp_path)
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    with config.open("a") as stream:
        stream.write(
            "[database]\nresearch_db_path="
            + json.dumps(str(old / "Data" / "custom" / "missing.sqlite"))
            + "\n"
        )
    with pytest.raises(ValueError, match="detached_required_database_missing"):
        _layout().produce_detached_archive(
            source.absolute(), (tmp_path / "work").absolute(), Event()
        )


@pytest.mark.parametrize("populated", [False, True])
def test_only_empty_installed_embedding_cache_scaffold_is_excluded(tmp_path, populated):
    source, root, _old = _base_profile(tmp_path)
    cache = root / "models" / "embeddings"
    cache.mkdir(parents=True)
    if populated:
        (cache / "unqualified.bin").write_bytes(b"must not disappear")
        with pytest.raises(
            ValueError, match="detached_owner_unavailable:models.artifacts"
        ):
            _layout().produce_detached_archive(
                source.absolute(), (tmp_path / "work").absolute(), Event()
            )
        assert (cache / "unqualified.bin").read_bytes() == b"must not disappear"
        return
    result = _layout().produce_detached_archive(
        source.absolute(), (tmp_path / "work").absolute(), Event()
    )
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert not any(
            row["relative_path"].startswith("embeddings/") for row in manifest["files"]
        )


@pytest.mark.parametrize("artifact", ["notes_sync", "video_capacity", "logs"])
def test_installed_device_and_diagnostic_files_keep_their_exact_exclusions(
    tmp_path, artifact
):
    source, root, _old = _base_profile(tmp_path)
    if artifact == "notes_sync":
        paths = [
            Path(str(root / "tldw_chatbook_notes_sync_state.db") + suffix)
            for suffix in ("", "-wal", "-shm", "-journal")
        ]
        for path in paths:
            path.write_bytes(b"retained device-only evidence")
        excluded_owner = "notes.sync_state"
    elif artifact == "video_capacity":
        path = root / ".generated_videos.capacity.lock"
        path.write_bytes(b"")
        paths = [path]
        excluded_owner = "generation.assets"
    else:
        config = source / "home" / ".config" / "tldw_cli" / "config.toml"
        with config.open("a") as stream:
            stream.write('[logging]\nlog_filename="custom-app.log"\n')
        paths = [
            root / name
            for name in ("custom-app.log", "custom-app.log.1", "custom-app.log.12")
        ]
        for path in paths:
            path.write_bytes(b"retained diagnostic bytes")
        excluded_owner = "diagnostics.logs"
    before = {path: path.read_bytes() for path in paths}
    result = _layout().produce_detached_archive(
        source.absolute(), (tmp_path / "work").absolute(), Event()
    )
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert not any(row["owner_id"] == excluded_owner for row in manifest["files"])
    assert all(path.read_bytes() == data for path, data in before.items())


def test_configured_log_lookalike_is_still_unknown_durable_content(tmp_path):
    source, root, _old = _base_profile(tmp_path)
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    with config.open("a") as stream:
        stream.write('[logging]\nlog_filename="custom-app.log"\n')
    (root / "custom-app.log.backup").write_bytes(b"unqualified")
    with pytest.raises(ValueError, match="detached_unknown_content"):
        _layout().produce_detached_archive(
            source.absolute(), (tmp_path / "work").absolute(), Event()
        )


def test_sqlite_cannot_gain_an_undeclared_raw_file_owner(tmp_path):
    source, _root, old = _base_profile(tmp_path)
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    _research(config.parent / "ui_state.toml")
    selector = old / "Data" / "home" / ".config" / "tldw_cli" / "ui_state.toml"
    with config.open("a") as stream:
        stream.write("[database]\nresearch_db_path=" + json.dumps(str(selector)) + "\n")
    with pytest.raises(ValueError, match="detached_alias_unsupported"):
        _layout().produce_detached_archive(
            source.absolute(), (tmp_path / "work").absolute(), Event()
        )


def test_custom_tts_store_keeps_its_installed_shared_reference_cohort(tmp_path):
    source, _root, old = _base_profile(tmp_path)
    from tldw_chatbook.TTS.recovery import recovery_adapters

    owner = next(
        owner for owner in recovery_adapters() if owner.owner_id == "tts.profile_store"
    )
    custom = source / "custom" / "voice.sqlite"
    custom.parent.mkdir()
    _installed_database(owner, custom)
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    with config.open("a") as stream:
        stream.write(
            "[database]\ntts_profiles_db_path="
            + json.dumps(str(old / "Data" / "custom" / "voice.sqlite"))
            + "\n"
        )
    result = _layout().produce_detached_archive(
        source.absolute(), (tmp_path / "work").absolute(), Event()
    )
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        cohorts = {
            row["owner_id"]: row["shared_group"]
            for row in manifest["producer_inventory"]
            if row["status"] == "included"
            and row["owner_id"] in {"tts.profile_store", "tts.references"}
        }
        assert set(cohorts) == {"tts.profile_store", "tts.references"}
        assert cohorts["tts.profile_store"] == cohorts["tts.references"]


@pytest.mark.parametrize("state", ["empty", "populated", "unexpected", "non_directory"])
def test_restored_canonical_cache_created_by_real_getter_is_narrowly_excluded(
    tmp_path, state
):
    import os
    import subprocess
    import sys

    from Packaging.windows.portable.portable import portable_environment, prepare_data

    source, root, _old = _base_profile(tmp_path)
    cache = root / "embedding_config" / "model_cache_dir"
    config = source / "home" / ".config" / "tldw_cli" / "config.toml"
    with config.open("a") as stream:
        stream.write(
            "[paths]\ndata_dir="
            + json.dumps(str(root.parent))
            + "\n[embedding_config]\nmodel_cache_dir="
            + json.dumps(str(cache))
            + "\n"
        )
    (source / "portable-location.json").write_text(
        json.dumps({"version": 1, "root": str(source.parent)})
    )
    env = portable_environment(source.parent, dict(os.environ))
    prepare_data(source.parent, env)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    child = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import json; from tldw_chatbook.config import get_model_cache_dir; print(json.dumps(str(get_model_cache_dir())))",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == str(cache)
    assert cache.is_dir() and not tuple(cache.iterdir())
    if state == "populated":
        (cache / "retained.bin").write_bytes(b"must not disappear")
    elif state == "unexpected":
        (cache.parent / "unexpected.bin").write_bytes(b"must not disappear")
    elif state == "non_directory":
        cache.rmdir()
        cache.write_bytes(b"must not disappear")
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    if state != "empty":
        with pytest.raises(ValueError, match="detached_restored_cache_unsupported"):
            _layout().produce_detached_archive(
                source.absolute(), (tmp_path / "work").absolute(), Event()
            )
    else:
        result = _layout().produce_detached_archive(
            source.absolute(), (tmp_path / "work").absolute(), Event()
        )
        with zipfile.ZipFile(result.archive) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            exclusions = [
                row
                for row in manifest["exclusions"]
                if "restored_embedding_cache" in row["logical_id"]
            ]
            assert len(exclusions) == 2
            assert all(row["reason"] == "intentionally_excluded" for row in exclusions)
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("notes_present", [True, False])
def test_absent_lazy_default_core_stores_are_reported_and_not_manufactured(
    tmp_path, notes_present
):
    present = {"db.chachanotes.primary"} if notes_present else set()
    source, root, _old = _base_profile(tmp_path, core_owners=present)
    if notes_present:
        from tldw_chatbook.Backup_Recovery.profile_paths import DATABASE_PATHS

        config = source / "home" / ".config" / "tldw_cli" / "config.toml"
        with config.open("a") as stream:
            stream.write("[database]\n")
            for owner, setting, _leaf, legacy in DATABASE_PATHS:
                if owner in {
                    "db.chachanotes.primary",
                    "db.media.primary",
                    "db.prompts.primary",
                    "db.library_collections",
                }:
                    stream.write(
                        setting + '="~/.local/share/tldw_cli/' + legacy + '"\n'
                    )
    before = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    result = _layout().produce_detached_archive(
        source.absolute(), (tmp_path / "work").absolute(), Event()
    )
    with zipfile.ZipFile(result.archive) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        missing = {
            "db.media.primary",
            "db.prompts.primary",
            "db.library_collections",
            "db.library_ingest_jobs",
        }
        if not notes_present:
            missing.add("db.chachanotes.primary")
        inventory = {
            row["owner_id"]: row
            for row in manifest["producer_inventory"]
            if row["logical_id"]
            == "profile:" + manifest["profile_ids"][0] + ":" + row["owner_id"]
        }
        assert all(inventory[owner]["status"] == "unused" for owner in missing)
        assert not any(row["owner_id"] in missing for row in manifest["files"])
        assert all(
            any(owner in line for line in result.review_lines) for owner in missing
        )
        assert manifest["consistency"] == "partial"
    assert not (root / "tldw_chatbook_media_v2.db").exists()
    assert not (root / "tldw_chatbook_prompts.db").exists()
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_missing_default_store_referenced_by_included_collection_still_refuses(
    tmp_path,
):
    source, root, _old = _base_profile(
        tmp_path, core_owners={"db.chachanotes.primary", "db.library_collections"}
    )
    with sqlite3.connect(root / "tldw_chatbook_library_collections.db") as connection:
        connection.execute(
            "INSERT INTO library_collections(collection_id,name,created_at,updated_at) VALUES ('collection','Retained','now','now')"
        )
        connection.execute(
            "INSERT INTO library_collection_items(membership_id,collection_id,source_type,source_id,created_at) VALUES ('member','collection','media','missing','now')"
        )
    with pytest.raises(ValueError, match="dependency_unavailable"):
        _layout().produce_detached_archive(
            source.absolute(), (tmp_path / "work").absolute(), Event()
        )
    assert not (tmp_path / "work.tldw-backup.zip").exists()


@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal", ".lock"])
def test_absent_default_store_with_residual_state_is_never_unused(tmp_path, suffix):
    source, root, _old = _base_profile(tmp_path, core_owners={"db.chachanotes.primary"})
    evidence = root / ("tldw_chatbook_media_v2.db" + suffix)
    evidence.write_bytes(b"retained unresolved state")
    with pytest.raises(
        ValueError, match="detached_(sqlite_orphan_sidecar|unknown_content)"
    ):
        _layout().produce_detached_archive(
            source.absolute(), (tmp_path / "work").absolute(), Event()
        )
    assert evidence.read_bytes() == b"retained unresolved state"
    assert not (tmp_path / "work.tldw-backup.zip").exists()
