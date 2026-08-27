import dataclasses
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
    build_library_rag_save_sections,
    load_assistant_library_access_default,
    load_direct_library_tools,
    load_rag_auto_retrieve_on_send,
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


# --- task-1337 Task 7: [console].direct_library_tools -----------------------


def _patch_cli_config(monkeypatch, config):
    """Point the config module's load seam at a fake mapping."""
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda *args, **kwargs: config,
    )


def test_direct_library_tools_field_defaults_true():
    assert SettingsLibraryRagDefaults().direct_library_tools is True


def test_new_console_conversation_policy_defaults_are_safe() -> None:
    defaults = SettingsLibraryRagDefaults()

    assert defaults.rag_auto_retrieve_on_send is False
    assert defaults.assistant_library_access_default is False


@pytest.mark.parametrize("value", [True, False])
def test_new_console_policy_default_loaders_read_their_distinct_sections(value):
    config = {
        "chat_defaults": {"rag_auto_retrieve_on_send": value},
        "console": {"assistant_library_access_default": value},
    }

    assert load_rag_auto_retrieve_on_send(config) is value
    assert load_assistant_library_access_default(config) is value


@pytest.mark.parametrize(
    "app_config",
    [
        {},
        {"console": {}},
        {"console": "not-a-mapping"},
        {"console": {"direct_library_tools": "maybe"}},
        {"console": {"direct_library_tools": "enabled"}},
        {"console": {"direct_library_tools": 42}},
        {"console": {"direct_library_tools": ["true"]}},
        {"console": {"direct_library_tools": None}},
    ],
)
def test_load_direct_library_tools_defaults_true_when_missing_or_malformed(
    app_config,
):
    assert load_direct_library_tools(app_config) is True


def test_load_direct_library_tools_reads_live_config_when_no_mapping_given(
    monkeypatch,
):
    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": False}})
    assert load_direct_library_tools() is False

    _patch_cli_config(monkeypatch, {})
    assert load_direct_library_tools() is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        (" True ", True),
        ("1", True),
        ("yes", True),
        ("t", True),
        ("false", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("f", False),
    ],
)
def test_load_direct_library_tools_coerces_bool_and_string_forms(raw, expected):
    app_config = {"console": {"direct_library_tools": raw}}

    assert load_direct_library_tools(app_config) is expected


def test_build_library_rag_save_sections_deep_merges_without_dropping_keys():
    app_config = {
        "console": {"max_parallel_runs": 3, "rail_state": {"open": True}},
        "AppRAGSearchConfig": {"rag": {"top_k": 5}},
        "general": {"default_tab": "chat"},
    }
    values = SettingsLibraryRagDefaults(
        direct_library_tools=False,
        rag_auto_retrieve_on_send=True,
        assistant_library_access_default=True,
    )

    sections = build_library_rag_save_sections(app_config, values)

    assert sections["console"]["direct_library_tools"] is False
    assert sections["console"]["assistant_library_access_default"] is True
    assert sections["chat_defaults"]["rag_auto_retrieve_on_send"] is True
    # Unrelated Console keys survive the merge.
    assert sections["console"]["max_parallel_runs"] == 3
    assert sections["console"]["rail_state"] == {"open": True}
    # The RAG section rides along verbatim (profile system remains the RAG
    # writer; this keeps the two-section save atomic).
    assert sections["AppRAGSearchConfig"] == {"rag": {"top_k": 5}}
    # Untouched sections are not part of the payload at all.
    assert "general" not in sections
    # Deep merge: mutating the returned sections must not reach back into the
    # caller's app_config.
    sections["console"]["rail_state"]["open"] = False
    assert app_config["console"]["rail_state"]["open"] is True


def test_build_library_rag_save_sections_handles_missing_sections():
    values = SettingsLibraryRagDefaults(direct_library_tools=True)

    sections = build_library_rag_save_sections({}, values)

    assert sections == {
        "console": {
            "direct_library_tools": True,
            "assistant_library_access_default": False,
        },
        "chat_defaults": {"rag_auto_retrieve_on_send": False},
        "AppRAGSearchConfig": {},
    }


def test_direct_library_tools_absent_from_console_session_settings():
    """The toggle is global app config -- never serialized per session."""
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings

    field_names = {field.name for field in dataclasses.fields(ConsoleSessionSettings)}
    assert "direct_library_tools" not in field_names


def test_rag_defaults_load_overlays_live_console_setting(monkeypatch):
    """Both the active-profile load and the profile-picker preview read the
    global [console] toggle fresh, so the checkbox never shows a stale value."""
    import tldw_chatbook.UI.Screens.settings_rag_profile_adapter as ad

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": False}})
    assert ad.load_rag_defaults_from_active_profile().direct_library_tools is False
    preview = ad.get_profile_defaults("hybrid_basic")
    assert preview is not None
    assert preview.direct_library_tools is False

    _patch_cli_config(monkeypatch, {"console": {"direct_library_tools": True}})
    assert ad.load_rag_defaults_from_active_profile().direct_library_tools is True


# --- TASK-3502 AC#1: the reranker-provider Select's option list is
# ENUMERATED from the dispatch table `chat_api_call` actually resolves the
# reranker's `model_provider` against -- never hand-listed, so a provider
# this build cannot dispatch can never be offered, and a newly registered
# one cannot go missing. ---


def test_reranker_provider_options_are_enumerated_from_the_dispatch_table():
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        DEFAULT_RERANKER_PROVIDER,
        library_rag_reranker_provider_options,
    )

    options = library_rag_reranker_provider_options()

    # The default row is labelled explicitly and carries that provider's own
    # NAME -- picking it must really write openai back over a profile
    # currently set to another provider (a blank sentinel there would leave
    # the old provider in place: blank means "leave the default alone").
    assert options[0] == (f"{DEFAULT_RERANKER_PROVIDER} (default)", DEFAULT_RERANKER_PROVIDER)
    assert {value for _label, value in options[1:]} == (
        set(API_CALL_HANDLERS) - {DEFAULT_RERANKER_PROVIDER}
    )
    assert [value for _label, value in options[1:]] == sorted(
        set(API_CALL_HANDLERS) - {DEFAULT_RERANKER_PROVIDER}
    )
    assert all(label == value for label, value in options[1:])


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Blank resolves to the provider a blank field really bills at run
        # time, so the control names it instead of showing nothing.
        ("", "openai"),
        ("   ", "openai"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        (" anthropic ", "anthropic"),
        # A hand-edited profile naming a provider this build cannot dispatch
        # must not reach Select(value=...) -- that raises
        # InvalidSelectValueError out of compose().
        ("not-a-real-provider", "openai"),
        (None, "openai"),
    ],
)
def test_normalise_reranker_provider_falls_back_to_the_default_row(raw, expected):
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        normalise_library_rag_reranker_provider,
    )

    assert normalise_library_rag_reranker_provider(raw) == expected
