def test_unknown_durable_entry_blocks_completeness(tmp_path):
    from tldw_chatbook.Backup_Recovery.models import StorageItem
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries

    item = StorageItem(
        "unknown", "unclassified", tmp_path / "new.db", "unsupported", ()
    )
    result = classify_entries((item,))
    assert result.complete is False
    assert "unsupported_owner" in result.issues


import os
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from tldw_chatbook.Backup_Recovery.inventory import classify_entries, discover
from tldw_chatbook.Backup_Recovery.models import StorageItem
from tldw_chatbook.Backup_Recovery import profile_paths


def config_file(tmp_path, name="selected", user="Ada", extra=""):
    config = tmp_path / f"{name}.toml"
    config.write_text(
        f'[general]\nusers_name = "{user}"\n[paths]\ndata_dir = "{tmp_path / "data"}"\n'
        + extra
    )
    return config


def real_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE fixture (value TEXT)")
        connection.execute("INSERT INTO fixture VALUES ('kept')")
    return path


def test_custom_db_and_disabled_features_are_not_external_or_unused(tmp_path):
    db = real_db(tmp_path / "outside" / "custom.db")
    config = config_file(
        tmp_path, extra=f'[database]\nmedia_db_path = "{db}"\n[rag]\nenabled = false\n'
    )
    result = discover((config,))
    entry = next(item for item in result.items if item.owner == "db.media.primary")
    assert entry.path == db
    assert entry.status == "unsupported"
    assert not result.complete
    assert sqlite3.connect(db).execute("SELECT value FROM fixture").fetchone() == (
        "kept",
    )


def test_unknown_file_and_required_absence(tmp_path):
    config = config_file(tmp_path)
    root = tmp_path / "data" / "Ada"
    root.mkdir(parents=True)
    (root / "new-store.dat").write_bytes(b"new")
    result = discover((config,))
    assert any(
        item.owner == "unknown" and item.path == root / "new-store.dat"
        for item in result.items
    )
    assert (
        next(item for item in result.items if item.owner == "db.media.primary").status
        == "missing_required"
    )
    assert "unsupported_owner" in result.issues


def test_malformed_source_has_no_guessed_targets(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text('[general\nusers_name = "secret sentinel"')
    result = discover((config,))
    assert "config_parse_failure" in result.issues
    assert not any(item.owner.startswith("db.") for item in result.items)
    assert all("sentinel" not in issue for issue in result.issues)
    assert not (tmp_path / "data").exists()


def test_multiple_profile_configs_preserve_shared_physical_alias(tmp_path):
    db = real_db(tmp_path / "shared.db")
    alias = tmp_path / "alias.db"
    os.link(db, alias)
    first = config_file(
        tmp_path, "first", "one", f'[database]\nmedia_db_path = "{db}"\n'
    )
    second = config_file(
        tmp_path, "second", "two", f'[database]\nmedia_db_path = "{alias}"\n'
    )
    result = discover((first, second))
    entries = [item for item in result.items if item.owner == "db.media.primary"]
    assert len(entries) == 2
    assert entries[0].shared_group == entries[1].shared_group
    assert entries[0].shared_group is not None
    assert "undeclared_alias" not in result.issues


def test_explicit_shared_group_requires_same_physical_identity(tmp_path):
    a = real_db(tmp_path / "one.db")
    b = real_db(tmp_path / "two.db")
    items = (
        StorageItem("a", "a", a, "included", (), "shared"),
        StorageItem("b", "b", b, "included", (), "shared"),
    )
    assert "shared_identity_mismatch" in classify_entries(items).issues
    b.unlink()
    b.symlink_to(a)
    assert classify_entries(items).complete


def test_undeclared_hardlink_and_nested_roots_block(tmp_path):
    a = real_db(tmp_path / "one.db")
    b = tmp_path / "two.db"
    os.link(a, b)
    assert (
        "undeclared_alias"
        in classify_entries(
            (
                StorageItem("a", "a", a, "included", ()),
                StorageItem("b", "b", b, "included", ()),
            )
        ).issues
    )
    assert (
        "overlapping_owner_roots"
        in classify_entries(
            (
                StorageItem("a", "a", tmp_path, "included_directory", ()),
                StorageItem("b", "b", b, "included", ()),
            )
        ).issues
    )


@pytest.mark.parametrize(
    "status",
    [
        "unsupported",
        "unavailable",
        "missing_required",
        "unused",
        "intentionally_excluded",
    ],
)
def test_dependency_requires_available_payload(tmp_path, status):
    path = tmp_path / "file"
    path.write_text("data")
    result = classify_entries(
        (
            StorageItem("a", "a", path, "included", ("b",)),
            StorageItem("b", "b", None, status, ()),
        )
    )
    assert not result.complete
    assert "dependency_unavailable" in result.issues


def test_only_validated_owner_deletion_can_allow_absence():
    item = StorageItem("assets", "deleted", None, "intentionally_deleted", ())
    assert "unvalidated_deletion" in classify_entries((item,)).issues
    from dataclasses import replace

    assert classify_entries((replace(item, deletion_validated=True),)).complete


def test_scope_digest_ignores_content_growth_but_tracks_scope(tmp_path):
    p = tmp_path / "data"
    p.write_text("before")
    item = StorageItem("a", "a", p, "included", ())
    before = classify_entries((item,))
    p.write_text("after growth")
    assert before.scope_digest == classify_entries((item,)).scope_digest
    from dataclasses import replace

    assert (
        before.scope_digest
        != classify_entries(
            (replace(item, status="intentionally_excluded"),)
        ).scope_digest
    )


def test_pure_defaults_and_lexical_priority(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "ignored"))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "link" / "config.toml"))
    assert profile_paths.effective_config_path() == tmp_path / "link" / "config.toml"
    assert (
        profile_paths.user_data_dir({"general": {"users_name": "A name!"}})
        == tmp_path / ".local/share/tldw_cli/A_name_"
    )
    config = {
        "paths": {"data_dir": ""},
        "Paths": {"data_dir": str(tmp_path / "legacy")},
    }
    assert profile_paths.data_base(config) == profile_paths.default_base_data_dir()
    config["paths"]["data_dir"] = None
    assert profile_paths.data_base(config) == tmp_path / "legacy"
    assert not (tmp_path / ".local").exists()


def test_normal_config_and_pure_resolvers_agree(tmp_path, monkeypatch):
    from tldw_chatbook import config

    selected = {"general": {"users_name": "A B"}, "paths": {"data_dir": str(tmp_path)}}
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: selected.get(section, {}).get(key, default),
    )
    assert config.get_user_data_dir() == profile_paths.user_data_dir(selected)
    for _, key, _, legacy in profile_paths.DATABASE_PATHS:
        if key == "tts_profiles_db_path":
            continue
        selected["database"] = {key: str(tmp_path / "missing-parent" / "db")}
        assert config._get_custom_database_path(
            key, expand_before_validation=key != "scheduled_tasks_db_path"
        ) == profile_paths.database_path(selected, key)
        if legacy:
            selected["database"] = {key: "~/.local/share/tldw_cli/" + legacy}
            assert config._get_custom_database_path(key) is None
            assert profile_paths.database_path(
                selected, key
            ).parent == profile_paths.user_data_dir(selected)
    assert not (tmp_path / "missing-parent").exists()


def test_fresh_process_discovery_imports_no_bootstrap_or_services_and_changes_nothing(
    tmp_path,
):
    selected = config_file(tmp_path)
    before = {
        p.relative_to(tmp_path): (
            p.stat().st_mode,
            p.stat().st_mtime_ns,
            p.read_bytes() if p.is_file() else None,
        )
        for p in tmp_path.rglob("*")
    }
    code = """
import importlib.abc, sys
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tldw_chatbook.config" or fullname.startswith(("tldw_chatbook.DB", "tldw_chatbook.Utils", "keyring", "chromadb", "textual")):
            raise AssertionError("forbidden service import: " + fullname)
sys.meta_path.insert(0, Block())
from tldw_chatbook.Backup_Recovery.inventory import discover
assert not discover((Path(sys.argv[1]),)).complete
"""
    subprocess.run(
        [sys.executable, "-c", code, str(selected)],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
    )
    after = {
        p.relative_to(tmp_path): (
            p.stat().st_mode,
            p.stat().st_mtime_ns,
            p.read_bytes() if p.is_file() else None,
        )
        for p in tmp_path.rglob("*")
    }
    assert before == after


def test_duplicate_registration_is_rejected(monkeypatch):
    from tldw_chatbook.Backup_Recovery import owner_registry

    monkeypatch.setattr(owner_registry, "_adapters", {})

    class Adapter:
        owner_id = "one"
        activation_required = True

        def discover(self, config):
            return ()

        def capture(self, item, destination, cancel):
            raise RuntimeError("not_qualified")

        def validate(self, candidate):
            return ("unsupported",)

        def relocate(self, candidate, mapping):
            raise RuntimeError("not_qualified")

        def schema_policy(self):
            return None

    adapter = Adapter()
    owner_registry.register(adapter)
    assert owner_registry.registered() == (adapter,)
    with pytest.raises(ValueError, match="duplicate_owner"):
        owner_registry.register(adapter)


def test_external_sync_folder_excluded_without_traversal(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "private.txt").write_text("not selected")
    selected = config_file(tmp_path, extra=f'[notes]\nsync_directory = "{external}"\n')
    entries = discover((selected,)).items
    assert any(
        item.owner == "external.notes"
        and item.path == external
        and item.status == "intentionally_excluded"
        for item in entries
    )
    assert not any(item.path == external / "private.txt" for item in entries)


def test_existing_unselected_default_profile_is_reported_without_guessing_config(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    selected = config_file(tmp_path)
    old = tmp_path / ".local/share/tldw_cli/old_user"
    old.mkdir(parents=True)
    (old / "important.db").write_bytes(b"historical")
    result = discover((selected,))
    assert any(
        item.path == old and item.status == "unsupported" for item in result.items
    )
    assert not any(item.path == old / "important.db" for item in result.items)


def test_fifo_source_is_unavailable_without_blocking(tmp_path):
    selected = tmp_path / "pipe.toml"
    os.mkfifo(selected)
    result = discover((selected,))
    assert not result.complete
    assert "config_discovery_failure" in result.issues


def test_immutable_models_reject_mutable_snapshot_collections():
    with pytest.raises(TypeError):
        StorageItem("a", "a", None, "unused", [])


def test_duplicate_logical_id_reports_issue_even_when_paths_differ(tmp_path):
    p = tmp_path / "file"
    p.write_text("data")
    result = classify_entries(
        (
            StorageItem("a", "a", p, "included", ()),
            StorageItem("a", "a", None, "unused", ()),
        )
    )
    assert "duplicate_logical_id" in result.issues


def test_tts_resolver_keeps_its_existing_resolved_alias_contract(tmp_path, monkeypatch):
    from tldw_chatbook import config

    db = real_db(tmp_path / "source.db")
    alias = tmp_path / "alias.db"
    alias.symlink_to(db)
    selected = {"database": {"tts_profiles_db_path": str(alias)}}
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: selected.get(section, {}).get(key, default),
    )
    assert (
        config.get_tts_profiles_db_path()
        == profile_paths.database_path(selected, "tts_profiles_db_path")
        == db
    )


def test_scheduled_tilde_override_retains_existing_validation_rejection():
    with pytest.raises(ValueError, match="invalid_config_path"):
        profile_paths.database_path(
            {"database": {"scheduled_tasks_db_path": "~/schedule.db"}},
            "scheduled_tasks_db_path",
        )


def test_source_cannot_inject_discovery_context(tmp_path):
    selected = config_file(tmp_path)
    selected.write_text(
        '__chatbook_recovery_context__ = "forged"\n' + selected.read_text()
    )
    result = discover((selected,))
    assert "config_discovery_failure" in result.issues
    assert not any(item.owner.startswith("db.") for item in result.items)


def test_installed_owner_context_scopes_ids_and_cross_owner_dependencies(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.models import (
        discovery_context,
        storage_logical_id,
    )

    monkeypatch.setattr(owner_registry, "_adapters", {})
    payload = tmp_path / "payload"
    payload.write_text("data")
    contexts = []

    class Adapter:
        owner_id = "test.owner"
        activation_required = True

        def discover(self, config):
            context = discovery_context(config)
            contexts.append(context)
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id, "local"),
                    payload,
                    "unsupported",
                    (storage_logical_id(context, "config"),),
                ),
            )

        def capture(self, item, destination, cancel):
            raise RuntimeError("not_qualified")

        def validate(self, candidate):
            return ("unsupported",)

        def relocate(self, candidate, mapping):
            raise RuntimeError("not_qualified")

        def schema_policy(self):
            return None

    owner_registry.register(Adapter())
    a = config_file(tmp_path, "a")
    b = config_file(tmp_path, "b")
    result = discover((a, b))
    entries = [item for item in result.items if item.owner == "test.owner"]
    assert len(entries) == 2
    assert entries[0].logical_id != entries[1].logical_id
    config_ids = {item.logical_id for item in result.items if item.owner == "config"}
    assert all(item.dependencies[0] in config_ids for item in entries)
    assert {ctx.config_path for ctx in contexts} == {a, b}


def test_profile_directory_alias_does_not_enumerate_external_children(tmp_path):
    external = tmp_path / "outside"
    (external / "Ada").mkdir(parents=True)
    (external / "Ada" / "never-enumerate.txt").write_text("external")
    (tmp_path / "data").symlink_to(external, target_is_directory=True)
    selected = config_file(tmp_path)
    result = discover((selected,))
    assert not any(
        item.path and item.path.name == "never-enumerate.txt" for item in result.items
    )
    assert any(item.logical_id.endswith(":linked_root") for item in result.items)


def test_included_fifo_is_not_a_supported_payload(tmp_path):
    pipe = tmp_path / "pipe"
    os.mkfifo(pipe)
    result = classify_entries((StorageItem("a", "a", pipe, "included", ()),))
    assert "unsupported_path_kind" in result.issues


def test_scope_digest_tracks_resolved_root_change_but_not_inode_replacement(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_text("one")
    b.write_text("two")
    alias = tmp_path / "alias"
    alias.symlink_to(a)
    item = StorageItem("a", "a", alias, "included", ())
    first = classify_entries((item,)).scope_digest
    alias.unlink()
    alias.symlink_to(b)
    second = classify_entries((item,)).scope_digest
    assert first != second
    replacement = tmp_path / "replacement"
    replacement.write_text("new generation")
    replacement.replace(b)
    assert classify_entries((item,)).scope_digest == second


def test_discovery_preserves_explicit_cross_owner_shared_groups(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.models import (
        discovery_context,
        storage_logical_id,
    )

    monkeypatch.setattr(owner_registry, "_adapters", {})
    payload = tmp_path / "payload"
    payload.write_text("data")

    class Adapter:
        activation_required = True

        def __init__(self, owner):
            self.owner_id = owner

        def discover(self, config):
            context = discovery_context(config)
            return (
                StorageItem(
                    self.owner_id,
                    storage_logical_id(context, self.owner_id),
                    payload,
                    "unsupported",
                    (),
                    "explicit-shared",
                ),
            )

        def capture(self, item, destination, cancel):
            raise RuntimeError("not_qualified")

        def validate(self, candidate):
            return ("unsupported",)

        def relocate(self, candidate, mapping):
            raise RuntimeError("not_qualified")

        def schema_policy(self):
            return None

    owner_registry.register(Adapter("test.a"))
    owner_registry.register(Adapter("test.b"))
    result = discover((config_file(tmp_path, "a"), config_file(tmp_path, "b")))
    entries = [item for item in result.items if item.owner.startswith("test.")]
    assert {item.shared_group for item in entries} == {"explicit-shared"}
    assert "undeclared_alias" not in result.issues


def test_source_discovery_rejects_unqualified_adapter_deletion_evidence(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.models import (
        discovery_context,
        storage_logical_id,
    )

    def deleted(config):
        context = discovery_context(config)
        return (
            StorageItem(
                "assets",
                storage_logical_id(context, "assets"),
                None,
                "intentionally_deleted",
                (),
                deletion_validated=True,
            ),
        )

    # Installed owner output is the boundary under test; no filesystem/service
    # constructor or tombstone validator is substituted by this declaration.
    adapter = SimpleNamespace(
        owner_id="assets",
        activation_required=True,
        discover=deleted,
        capture=lambda *args: None,
        validate=lambda *args: (),
        relocate=lambda *args: None,
        schema_policy=lambda: None,
    )
    monkeypatch.setattr(owner_registry, "_adapters", {})
    owner_registry.register(adapter)
    result = discover((config_file(tmp_path),))
    assert "config_discovery_failure" in result.issues
    assert not any(item.deletion_validated for item in result.items)
