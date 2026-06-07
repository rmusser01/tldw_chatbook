from pathlib import Path

import pytest

from tldw_chatbook.RAG_Search.simplified import config as rag_config_module
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
    SettingsLibraryRagDefaults,
    build_library_rag_save_sections,
    load_library_rag_defaults,
    validate_library_rag_defaults,
)


def _patch_rag_settings(monkeypatch, rag_settings):
    app_config = {"AppRAGSearchConfig": {"rag": rag_settings}}

    def fake_get_cli_setting(section, key, default=None):
        return app_config.get(section, {}).get(key, default)

    monkeypatch.setattr(rag_config_module, "load_cli_config_and_ensure_existence", lambda: app_config)
    monkeypatch.setattr(rag_config_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(rag_config_module, "get_user_data_dir", lambda: Path("/tmp/tldw-rag-test"))
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    monkeypatch.delenv("RAG_SEARCH_MODE", raising=False)


def test_rag_config_loads_settings_controlled_display_defaults(monkeypatch):
    _patch_rag_settings(
        monkeypatch,
        {
            "search": {
                "citation_style": "footnote",
                "snippet_max_chars": 512,
                "max_context_size": 64000,
            }
        },
    )

    config = RAGConfig.from_settings()

    assert config.search.citation_style == "footnote"
    assert config.search.snippet_max_chars == 512
    assert config.search.max_context_size == 64000


def test_load_library_rag_defaults_uses_safe_defaults():
    defaults = load_library_rag_defaults({})

    assert defaults.default_search_mode == "semantic"
    assert defaults.default_top_k == 10
    assert defaults.fts_top_k == 10
    assert defaults.vector_top_k == 10
    assert defaults.hybrid_alpha == 0.5
    assert defaults.score_threshold == 0.0
    assert defaults.include_citations is True
    assert defaults.citation_style == "inline"
    assert defaults.snippet_max_chars == 240
    assert defaults.max_context_size == 16000


def test_load_library_rag_defaults_reads_nested_search_and_retriever_sections():
    defaults = load_library_rag_defaults(
        {
            "AppRAGSearchConfig": {
                "rag": {
                    "search": {
                        "default_search_mode": "hybrid",
                        "default_top_k": 12,
                        "score_threshold": 0.25,
                        "include_citations": False,
                        "citation_style": "footnote",
                        "snippet_max_chars": 480,
                        "max_context_size": 32000,
                    },
                    "retriever": {
                        "fts_top_k": 20,
                        "vector_top_k": 18,
                        "hybrid_alpha": 0.35,
                    },
                }
            }
        }
    )

    assert defaults == SettingsLibraryRagDefaults(
        default_search_mode="hybrid",
        default_top_k=12,
        fts_top_k=20,
        vector_top_k=18,
        hybrid_alpha=0.35,
        score_threshold=0.25,
        include_citations=False,
        citation_style="footnote",
        snippet_max_chars=480,
        max_context_size=32000,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("default_search_mode", "unknown", "Search mode"),
        ("default_top_k", 0, "Default results"),
        ("fts_top_k", 0, "Keyword results"),
        ("vector_top_k", 0, "Vector results"),
        ("hybrid_alpha", 1.5, "Hybrid balance"),
        ("score_threshold", -0.1, "Score threshold"),
        ("citation_style", "mla", "Citation style"),
        ("snippet_max_chars", 49, "Snippet characters"),
        ("max_context_size", 999, "Context budget"),
    ],
)
def test_validate_library_rag_defaults_rejects_invalid_values(field, value, message):
    values = SettingsLibraryRagDefaults()
    values = SettingsLibraryRagDefaults(
        **{**values.__dict__, field: value}
    )

    result = validate_library_rag_defaults(values)

    assert result.valid is False
    assert message in result.message


def test_validate_library_rag_defaults_accepts_valid_values():
    result = validate_library_rag_defaults(
        SettingsLibraryRagDefaults(
            default_search_mode="plain",
            default_top_k=1,
            fts_top_k=50,
            vector_top_k=50,
            hybrid_alpha=1.0,
            score_threshold=1.0,
            include_citations=True,
            citation_style="none",
            snippet_max_chars=50,
            max_context_size=1000,
        )
    )

    assert result.valid is True
    assert "valid" in result.message.lower()


def test_build_library_rag_save_sections_deep_merges_without_dropping_unrelated_rag_config():
    app_config = {
        "AppRAGSearchConfig": {
            "rag": {
                "search": {
                    "cache_size": 200,
                    "semantic_cache_ttl": 7200,
                },
                "retriever": {
                    "media_collection": "existing-media",
                },
                "chunking": {
                    "chunk_size": 400,
                },
            }
        }
    }
    values = SettingsLibraryRagDefaults(
        default_search_mode="hybrid",
        default_top_k=15,
        fts_top_k=25,
        vector_top_k=30,
        hybrid_alpha=0.4,
        score_threshold=0.2,
        include_citations=False,
        citation_style="footnote",
        snippet_max_chars=360,
        max_context_size=24000,
    )

    sections = build_library_rag_save_sections(app_config, values)

    rag = sections["AppRAGSearchConfig"]["rag"]
    assert rag["search"] == {
        "cache_size": 200,
        "semantic_cache_ttl": 7200,
        "default_search_mode": "hybrid",
        "default_top_k": 15,
        "score_threshold": 0.2,
        "include_citations": False,
        "citation_style": "footnote",
        "snippet_max_chars": 360,
        "max_context_size": 24000,
    }
    assert rag["retriever"] == {
        "media_collection": "existing-media",
        "fts_top_k": 25,
        "vector_top_k": 30,
        "hybrid_alpha": 0.4,
    }
    assert rag["chunking"] == {"chunk_size": 400}
