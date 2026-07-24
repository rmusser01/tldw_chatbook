import inspect
from pathlib import Path

import pytest

from tldw_chatbook.RAG_Search.config_profiles import (
    ConfigProfileManager,
    reset_profile_manager_cache,
)
from tldw_chatbook.RAG_Search.simplified import config as rag_config_module
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
    SettingsLibraryRagDefaults,
    validate_library_rag_defaults,
)


@pytest.fixture(autouse=True)
def _hermetic_profile_manager(tmp_path, monkeypatch):
    """Keep validate_library_rag_defaults's profile read inside tmp_path.

    validate_library_rag_defaults routes hard-error checks through the
    adapter's hard_config_errors(), which reads the active profile via
    get_profile_manager(). Left unpatched, that resolves to the real
    ~/.local/share/tldw_cli/.../rag_profiles dir (get_user_data_dir's
    Path.home() is frozen at import time, before any HOME patching a test
    might do) -- reading real files and mkdir-ing a real tree as a side
    effect of running this test file. Point the adapter's manager/active-id
    seams at a tmp-dir-backed manager instead, for every test here.
    """
    mgr = ConfigProfileManager(profiles_dir=tmp_path / "rag_profiles")
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as ad

    monkeypatch.setattr(ad, "_manager", lambda: mgr, raising=False)
    monkeypatch.setattr(ad, "_active_profile_id", lambda: "hybrid_basic", raising=False)
    yield
    reset_profile_manager_cache()


def _patch_rag_settings(monkeypatch, rag_settings):
    app_config = {"AppRAGSearchConfig": {"rag": rag_settings}}

    def fake_get_cli_setting(section, key, default=None):
        return app_config.get(section, {}).get(key, default)

    monkeypatch.setattr(
        rag_config_module, "load_cli_config_and_ensure_existence", lambda: app_config
    )
    monkeypatch.setattr(rag_config_module, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        rag_config_module, "get_user_data_dir", lambda: Path("/tmp/tldw-rag-test")
    )
    monkeypatch.delenv("RAG_TOP_K", raising=False)
    monkeypatch.delenv("RAG_SEARCH_MODE", raising=False)


def test_rag_config_uses_fallbacks_for_invalid_display_default_ints(monkeypatch):
    _patch_rag_settings(
        monkeypatch,
        {
            "search": {
                "snippet_max_chars": "not-an-int",
                "max_context_size": "also-not-an-int",
            }
        },
    )

    config = RAGConfig.from_settings()

    assert config.search.snippet_max_chars == 240
    assert config.search.max_context_size == 16000


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
    values = SettingsLibraryRagDefaults(**{**values.__dict__, field: value})

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


def test_validate_library_rag_defaults_accepts_float_like_integer_values():
    result = validate_library_rag_defaults(
        SettingsLibraryRagDefaults(
            default_top_k="12.0",
            fts_top_k=18.0,
            vector_top_k="19.0",
            snippet_max_chars="512.0",
            max_context_size=64000.0,
        )
    )

    assert result.valid is True


def test_library_rag_public_functions_use_google_style_docstrings():
    for function in (validate_library_rag_defaults,):
        doc = inspect.getdoc(function)
        assert doc is not None
        assert "Args:" in doc
        assert "Returns:" in doc


def test_hard_config_errors_fails_closed_when_the_profile_fetch_raises(monkeypatch):
    """A profile-manager blowup while reading the active profile must not
    escape hard_config_errors as a raised exception -- it must come back as
    a single fail-CLOSED hard error, same as the "no active profile" case.
    """
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as ad

    def _raise():
        raise RuntimeError("profiles dir unreadable")

    monkeypatch.setattr(ad, "_manager", _raise, raising=False)

    errors = ad.hard_config_errors(SettingsLibraryRagDefaults())

    assert errors == ["Could not load the active profile for validation."]
