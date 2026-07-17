"""
Tests for RAG vector store default selection (task-246).

The vector store should default to persistent ChromaDB when the
`embeddings_rag` optional dependencies are installed, with the persist
directory under the app's user data dir, while:
- staying on the in-memory store when the deps are missing, and
- honoring an explicit user override back to `type = "memory"`.

These tests exercise the selection logic with the dependency check
monkeypatched, so they do not require chromadb/torch to be importable
(except the importorskip-gated persistence round-trip at the bottom).
"""

from pathlib import Path

import pytest

from tldw_chatbook.RAG_Search.simplified import config as rag_config_module
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig, VectorStoreConfig


# === Helpers ===

def _patch_environment(monkeypatch, rag_settings=None, user_data_dir="/tmp/tldw-rag-selection-test"):
    """Isolate the selection logic from the host machine's config and env."""
    app_config = {"AppRAGSearchConfig": {"rag": rag_settings or {}}}

    def fake_get_cli_setting(section, key, default=None):
        return app_config.get(section, {}).get(key, default)

    monkeypatch.setattr(rag_config_module, "load_cli_config_and_ensure_existence", lambda: app_config)
    monkeypatch.setattr(rag_config_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(rag_config_module, "get_user_data_dir", lambda: Path(user_data_dir))
    monkeypatch.delenv("RAG_VECTOR_STORE", raising=False)
    monkeypatch.delenv("RAG_PERSIST_DIR", raising=False)
    monkeypatch.delenv("RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("RAG_DEVICE", raising=False)


def _patch_deps(monkeypatch, available: bool):
    """Force the embeddings_rag availability check to a known value."""
    monkeypatch.setattr(rag_config_module, "_embeddings_rag_available", lambda: available)


# === Default selection: deps installed ===

@pytest.mark.unit
class TestDefaultWithEmbeddingsDeps:
    """With embeddings_rag installed, default to persistent ChromaDB."""

    def test_vector_store_config_defaults_to_chroma(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig()

        assert config.type == "chroma"

    def test_persist_directory_defaults_under_user_data_dir(self, monkeypatch):
        _patch_environment(monkeypatch, user_data_dir="/tmp/tldw-rag-user-dir")
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig()

        assert config.persist_directory == Path("/tmp/tldw-rag-user-dir") / "chromadb"

    def test_rag_config_default_is_valid_chroma_config(self, monkeypatch):
        """Plain RAGConfig() (what runtime profiles build) must validate cleanly."""
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)

        config = RAGConfig()

        assert config.vector_store.type == "chroma"
        assert config.vector_store.persist_directory is not None
        assert config.validate() == []

    def test_from_settings_defaults_to_chroma(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)

        config = RAGConfig.from_settings()

        assert config.vector_store.type == "chroma"
        assert config.vector_store.persist_directory == Path("/tmp/tldw-rag-selection-test") / "chromadb"

    def test_hybrid_basic_profile_inherits_persistent_default(self, monkeypatch):
        """The runtime profile used by chat RAG must pick up the persistent default."""
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)

        from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager

        manager = ConfigProfileManager()
        profile = manager.get_profile("hybrid_basic")

        assert profile is not None
        assert profile.rag_config.vector_store.type == "chroma"
        assert profile.rag_config.vector_store.persist_directory is not None


# === Default selection: deps missing ===

@pytest.mark.unit
class TestDefaultWithoutEmbeddingsDeps:
    """Without embeddings deps, behavior is unchanged: in-memory store."""

    def test_vector_store_config_defaults_to_memory(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, False)

        config = VectorStoreConfig()

        assert config.type == "memory"
        assert config.persist_directory is None

    def test_rag_config_default_is_valid_memory_config(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, False)

        config = RAGConfig()

        assert config.vector_store.type == "memory"
        assert config.validate() == []

    def test_from_settings_defaults_to_memory(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, False)

        config = RAGConfig.from_settings()

        assert config.vector_store.type == "memory"

    def test_availability_check_failure_falls_back_to_memory(self, monkeypatch):
        """If the optional-deps probe itself blows up, default must stay memory."""
        _patch_environment(monkeypatch)
        monkeypatch.setattr(rag_config_module, "_EMBEDDINGS_RAG_AVAILABLE", None)

        def broken_probe():
            raise RuntimeError("dependency probe exploded")

        import tldw_chatbook.Utils.optional_deps as optional_deps
        monkeypatch.setattr(optional_deps, "embeddings_rag_deps_installed", broken_probe)

        config = VectorStoreConfig()

        assert config.type == "memory"
        assert config.persist_directory is None

    def test_installed_probe_is_find_spec_based_and_fails_closed(self, monkeypatch):
        """The installed-probe must answer without imports and fail closed."""
        import importlib.util
        from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

        assert isinstance(embeddings_rag_deps_installed(), bool)

        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
        assert embeddings_rag_deps_installed() is False


# === Explicit overrides (AC #3) ===

@pytest.mark.unit
class TestExplicitOverrides:
    """Explicit user configuration must always win over the deps-based default."""

    def test_explicit_memory_in_user_config_wins_over_deps(self, monkeypatch):
        _patch_environment(monkeypatch, rag_settings={"vector_store": {"type": "memory"}})
        _patch_deps(monkeypatch, True)

        config = RAGConfig()

        assert config.vector_store.type == "memory"
        assert config.vector_store.persist_directory is None

    def test_explicit_memory_in_user_config_wins_in_from_settings(self, monkeypatch):
        _patch_environment(monkeypatch, rag_settings={"vector_store": {"type": "memory"}})
        _patch_deps(monkeypatch, True)

        config = RAGConfig.from_settings()

        assert config.vector_store.type == "memory"

    def test_explicit_memory_applies_to_runtime_profiles(self, monkeypatch):
        """User `type = \"memory\"` must also govern profile-built configs."""
        _patch_environment(monkeypatch, rag_settings={"vector_store": {"type": "memory"}})
        _patch_deps(monkeypatch, True)

        from tldw_chatbook.RAG_Search.config_profiles import ConfigProfileManager

        manager = ConfigProfileManager()
        profile = manager.get_profile("hybrid_basic")

        assert profile is not None
        assert profile.rag_config.vector_store.type == "memory"

    def test_env_var_memory_override_wins_over_deps(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)
        monkeypatch.setenv("RAG_VECTOR_STORE", "memory")

        config = VectorStoreConfig()

        assert config.type == "memory"

    def test_explicit_persist_directory_in_user_config_wins(self, monkeypatch):
        _patch_environment(
            monkeypatch,
            rag_settings={"vector_store": {"persist_directory": "/tmp/custom-chroma-home"}},
        )
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig()

        assert config.type == "chroma"
        assert config.persist_directory == Path("/tmp/custom-chroma-home")

    def test_env_var_persist_directory_wins(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)
        monkeypatch.setenv("RAG_PERSIST_DIR", "/tmp/env-chroma-dir")

        config = VectorStoreConfig()

        assert config.persist_directory == Path("/tmp/env-chroma-dir")

    def test_constructor_arguments_bypass_auto_selection(self, monkeypatch):
        """Explicitly constructed configs (tests, fixtures) must be untouched."""
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig(type="memory", persist_directory=None)

        assert config.type == "memory"
        assert config.persist_directory is None

    def test_from_dict_explicit_type_wins(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)

        config = RAGConfig.from_dict({"vector_store": {"type": "memory"}})

        assert config.vector_store.type == "memory"

    def test_explicit_chroma_gets_default_persist_directory(self, monkeypatch):
        """type='chroma' without a persist dir gets the user-data-dir default."""
        _patch_environment(monkeypatch, user_data_dir="/tmp/tldw-rag-user-dir")
        _patch_deps(monkeypatch, False)

        config = VectorStoreConfig(type="chroma")

        assert config.persist_directory == Path("/tmp/tldw-rag-user-dir") / "chromadb"


# === Normalization of explicit values (PR #656 review) ===

@pytest.mark.unit
class TestExplicitValueNormalization:
    """Explicit env/config values are normalized; 'auto'/blank run detection."""

    def test_explicit_auto_env_var_runs_detection(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)
        monkeypatch.setenv("RAG_VECTOR_STORE", "auto")

        config = VectorStoreConfig()

        assert config.type == "chroma"
        assert config.persist_directory is not None

    def test_explicit_auto_in_user_config_runs_detection(self, monkeypatch):
        _patch_environment(monkeypatch, rag_settings={"vector_store": {"type": "auto"}})
        _patch_deps(monkeypatch, False)

        config = VectorStoreConfig()

        assert config.type == "memory"

    def test_auto_env_var_falls_through_to_explicit_config(self, monkeypatch):
        """env 'auto' means automatic behavior, which still honors user config."""
        _patch_environment(monkeypatch, rag_settings={"vector_store": {"type": "memory"}})
        _patch_deps(monkeypatch, True)
        monkeypatch.setenv("RAG_VECTOR_STORE", "auto")

        config = VectorStoreConfig()

        assert config.type == "memory"

    def test_mixed_case_and_whitespace_type_is_canonicalized(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, False)
        monkeypatch.setenv("RAG_VECTOR_STORE", "  Chroma ")

        config = VectorStoreConfig()

        assert config.type == "chroma"
        assert config.persist_directory is not None
        assert RAGConfig(vector_store=config).validate() == []

    def test_uppercase_memory_in_user_config_is_canonicalized(self, monkeypatch):
        _patch_environment(monkeypatch, rag_settings={"vector_store": {"type": "MEMORY"}})
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig()

        assert config.type == "memory"

    def test_blank_type_env_var_runs_detection(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)
        monkeypatch.setenv("RAG_VECTOR_STORE", "   ")

        config = VectorStoreConfig()

        assert config.type == "chroma"

    def test_whitespace_only_persist_dir_env_var_is_ignored(self, monkeypatch):
        _patch_environment(monkeypatch, user_data_dir="/tmp/tldw-rag-user-dir")
        _patch_deps(monkeypatch, True)
        monkeypatch.setenv("RAG_PERSIST_DIR", "   ")

        config = VectorStoreConfig()

        assert config.persist_directory == Path("/tmp/tldw-rag-user-dir") / "chromadb"

    def test_whitespace_only_persist_dir_in_config_is_ignored(self, monkeypatch):
        _patch_environment(
            monkeypatch,
            rag_settings={"vector_store": {"persist_directory": "  "}},
            user_data_dir="/tmp/tldw-rag-user-dir",
        )
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig()

        assert config.persist_directory == Path("/tmp/tldw-rag-user-dir") / "chromadb"

    def test_persist_dir_values_are_stripped(self, monkeypatch):
        _patch_environment(monkeypatch)
        _patch_deps(monkeypatch, True)
        monkeypatch.setenv("RAG_PERSIST_DIR", "  /tmp/padded-chroma-dir  ")

        config = VectorStoreConfig()

        assert config.persist_directory == Path("/tmp/padded-chroma-dir")


# === Legacy [AppRAGSearchConfig.rag.chroma] compatibility (PR #656 review) ===

@pytest.mark.unit
class TestLegacyChromaSection:
    """Profile-built configs must honor the legacy chroma persist location."""

    def test_legacy_chroma_persist_directory_is_honored(self, monkeypatch):
        _patch_environment(
            monkeypatch,
            rag_settings={"chroma": {"persist_directory": "/tmp/legacy-chroma-home"}},
        )
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig()

        assert config.persist_directory == Path("/tmp/legacy-chroma-home")

    def test_legacy_key_matches_from_settings_resolution(self, monkeypatch):
        """Plain RAGConfig() and from_settings() must persist to the same place."""
        _patch_environment(
            monkeypatch,
            rag_settings={"chroma": {"persist_directory": "/tmp/legacy-chroma-home"}},
        )
        _patch_deps(monkeypatch, True)

        plain = RAGConfig()
        loaded = RAGConfig.from_settings()

        assert plain.vector_store.persist_directory == loaded.vector_store.persist_directory

    def test_explicit_vector_store_key_beats_legacy_key(self, monkeypatch):
        _patch_environment(
            monkeypatch,
            rag_settings={
                "vector_store": {"persist_directory": "/tmp/new-style-dir"},
                "chroma": {"persist_directory": "/tmp/legacy-chroma-home"},
            },
        )
        _patch_deps(monkeypatch, True)

        config = VectorStoreConfig()

        assert config.persist_directory == Path("/tmp/new-style-dir")


# === Persistence round-trip (AC #1 evidence; requires real chromadb) ===

@pytest.mark.integration
class TestChromaPersistenceRoundTrip:
    """Documents added to the chroma store survive a store restart."""

    def test_documents_survive_store_recreation(self, tmp_path):
        pytest.importorskip("chromadb")
        pytest.importorskip("numpy")

        from tldw_chatbook.RAG_Search.simplified.vector_store import create_vector_store

        persist_dir = tmp_path / "chromadb"
        collection = "persistence_roundtrip"
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        store = create_vector_store(
            store_type="chroma",
            persist_directory=persist_dir,
            collection_name=collection,
        )
        store.add(
            ids=["doc-1", "doc-2"],
            embeddings=embeddings,
            documents=["first document", "second document"],
            metadata=[{"source": "test"}, {"source": "test"}],
        )
        store.close()
        del store

        # Simulate an app restart: a brand-new store over the same directory.
        reopened = create_vector_store(
            store_type="chroma",
            persist_directory=persist_dir,
            collection_name=collection,
        )
        try:
            results = reopened.search([1.0, 0.0, 0.0], top_k=2)
            result_ids = {r.id for r in results}
            assert "doc-1" in result_ids
            assert len(results) == 2
        finally:
            reopened.close()
