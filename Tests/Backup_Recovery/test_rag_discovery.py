"""RAG discovery reads retained selectors without constructing runtime owners."""

import json
import sys

import pytest

from tldw_chatbook.Backup_Recovery.models import DISCOVERY_CONTEXT_KEY, DiscoveryContext
from tldw_chatbook.Backup_Recovery.profile_paths import user_data_dir


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    monkeypatch.delenv("RAG_VECTOR_STORE", raising=False)
    return {
        "paths": {"data_dir": str(tmp_path / "data")},
        DISCOVERY_CONTEXT_KEY: DiscoveryContext(tmp_path / "config.toml", "test"),
    }


def adapters():
    from tldw_chatbook.Backup_Recovery.rag_inventory import recovery_adapters

    return {a.owner_id: a for a in recovery_adapters()}


def test_absent_default_roots_are_unused_without_runtime_imports(config):
    before = set(sys.modules)
    for owner in ("rag.definitions", "rag.projections"):
        assert all(
            item.status == "unused" for item in adapters()[owner].discover(config)
        )
    assert not user_data_dir(config).exists()
    assert not any(
        name.startswith(("chromadb", "tldw_chatbook.RAG_Search.config_profiles"))
        for name in set(sys.modules) - before
    )


def test_empty_root_is_metadata_but_nonempty_projection_is_pending(config):
    root = user_data_dir(config) / "chromadb"
    root.mkdir(parents=True)
    adapter = adapters()["rag.projections"]
    assert any(item.status == "included_directory" for item in adapter.discover(config))
    (root / "engine.bin").write_bytes(b"unknown engine bytes")
    rows = adapter.discover(config)
    assert any(item.status == "unsupported" for item in rows)
    assert not any(item.status == "included" for item in rows)


def test_retained_profile_path_is_found_without_migration(config, tmp_path):
    root = user_data_dir(config) / "rag_profiles"
    root.mkdir(parents=True)
    alternate = tmp_path / "old-index"
    profile = root / "renamed.json"
    content = json.dumps(
        {
            "name": "Retained",
            "id": "stale",
            "rag_config": {
                "vector_store": {"type": "chroma", "persist_directory": str(alternate)}
            },
        }
    )
    profile.write_text(content)
    rows = adapters()["rag.projections"].discover(config)
    assert any(item.path == alternate for item in rows)
    assert profile.read_text() == content and sorted(
        p.name for p in root.iterdir()
    ) == ["renamed.json"]
    definitions = adapters()["rag.definitions"].discover(config)
    row = next(item for item in definitions if item.path == profile)
    assert row.status == "unsupported" and row.shared_group is None
    assert row.metadata.relative_path == "renamed.json"


def test_explicit_environment_root_precedes_config(config, tmp_path, monkeypatch):
    explicit = tmp_path / "selected"
    config["AppRAGSearchConfig"] = {
        "rag": {"vector_store": {"persist_directory": str(tmp_path / "other")}}
    }
    monkeypatch.setenv("RAG_PERSIST_DIR", str(explicit))
    rows = adapters()["rag.projections"].discover(config)
    assert any(item.path == explicit for item in rows)
    assert not any(item.path == tmp_path / "other" for item in rows)


def test_malformed_retained_profile_blocks_projection_selection(config):
    root = user_data_dir(config) / "rag_profiles"
    root.mkdir(parents=True)
    (root / "bad.json").write_text('{"rag_config":')
    assert any(
        item.status == "unsupported"
        for item in adapters()["rag.projections"].discover(config)
    )


def test_memory_selection_does_not_hide_retained_default_bytes(config):
    config["AppRAGSearchConfig"] = {"rag": {"vector_store": {"type": "memory"}}}
    root = user_data_dir(config) / "chromadb"
    root.mkdir(parents=True)
    (root / "chroma.sqlite3").write_bytes(b"old projection")
    assert any(
        item.status == "unsupported"
        for item in adapters()["rag.projections"].discover(config)
    )


def test_factory_import_has_no_runtime_bootstrap_in_fresh_process(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(
        tmp_path,
        "inert",
        "success",
        script="""
import sys
from tldw_chatbook.Backup_Recovery.rag_inventory import recovery_adapters
assert {a.owner_id for a in recovery_adapters()} == {'rag.definitions', 'rag.projections', 'db.rag_indexing'}
assert 'tldw_chatbook.RAG_Search' not in sys.modules
assert 'tldw_chatbook.config' not in sys.modules
assert 'chromadb' not in sys.modules
print('retired and reopened')
""",
    )


def test_auto_environment_falls_through_to_configured_engine(config, monkeypatch):
    config["AppRAGSearchConfig"] = {"rag": {"vector_store": {"type": "qdrant"}}}
    monkeypatch.setenv("RAG_VECTOR_STORE", " auto ")
    assert any(
        item.status == "unsupported"
        for item in adapters()["rag.projections"].discover(config)
    )


@pytest.mark.parametrize("populated", [False, True])
def test_public_discovery_preserves_real_rag_file_identities(
    config, monkeypatch, populated
):
    from Tests.Backup_Recovery.test_capture_service import (
        _populate_required_dependencies,
    )
    from tldw_chatbook.Backup_Recovery import owner_registry
    from tldw_chatbook.Backup_Recovery.inventory import classify_entries, discover

    monkeypatch.setattr(owner_registry, "_adapters", {})
    owner_registry.install_adapters()
    source = config[DISCOVERY_CONTEXT_KEY].config_path
    source.write_text(
        "[paths]\ndata_dir=" + json.dumps(config["paths"]["data_dir"]) + "\n"
    )
    root = user_data_dir(config) / "rag_profiles"
    if populated:
        root.mkdir(parents=True)
        for name in ("one", "two"):
            (root / (name + ".json")).write_text(
                json.dumps(
                    {"name": name, "rag_config": {"vector_store": {"type": "memory"}}}
                )
            )
    _populate_required_dependencies(discover((source,)))
    inventory = discover((source,))
    assert not {"shared_identity_unavailable", "shared_identity_mismatch"}.intersection(
        inventory.issues
    )
    classified = classify_entries(inventory.items)
    assert "shared_identity_mismatch" not in classified.issues
    rows = [item for item in inventory.items if item.owner == "rag.definitions"]
    if populated:
        files = [
            item
            for item in rows
            if item.path is not None and item.path.suffix == ".json"
        ]
        assert len(files) == 2
        assert all(item.status == "unsupported" for item in files)
        assert len({item.logical_id for item in files}) == 2
    else:
        assert rows and all(item.status == "unused" for item in rows)
