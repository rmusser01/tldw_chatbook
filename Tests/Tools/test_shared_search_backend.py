"""Shared backend routing through real Settings saves and local tool handlers."""

import tomllib

import pytest

from tldw_chatbook import config
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.Web_Scraping import WebSearch_APIs


@pytest.fixture
def search_profile(tmp_path, monkeypatch):
    """Keep persistence and dispatch real; replace external search/LLM phases."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    for name in (
        "_CONFIG_CACHE",
        "_CONFIG_CACHE_SOURCE",
        "_SETTINGS_CACHE",
        "_SETTINGS_CACHE_SOURCE",
        "_CONFIG_GENERATION",
        "settings",
    ):
        monkeypatch.setattr(config, name, getattr(config, name))
    config_path.write_text(
        "[tools]\nweb_deep_search_enabled = true\n"
        '[SearchSettings]\nrelevance_analysis_llm = "openai"\n'
        'final_answer_llm = "openai"\n',
        encoding="utf-8",
    )
    adapter = SettingsConfigAdapter()
    adapter.load(force_reload=True)
    calls = []

    def search(**kwargs):
        calls.append(("basic", kwargs["search_engine"]))
        return {
            "results": [
                {
                    "title": "Result",
                    "url": "https://example.com/",
                    "content": "Evidence",
                }
            ]
        }

    def generate(question, params):
        calls.append(("deep", params["engine"]))
        return {
            "web_search_results_dict": {
                "results": [{"title": "Result", "url": "https://example.com/"}],
                "warnings": [],
            },
            "sub_query_dict": {"sub_questions": [], "main_goal": question},
        }

    async def analyze(results, subqueries, params, cancel_event=None):
        return {
            "final_answer": {
                "text": "Answer [1].",
                "evidence": [
                    {"id": 1, "title": "Result", "url": "https://example.com/"}
                ],
                "confidence": 0.8,
                "chunks": [],
            },
            "relevant_results": {"1": {}},
            "web_search_results_dict": results,
        }

    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", search)
    monkeypatch.setattr(WebSearch_APIs, "generate_and_search", generate)
    monkeypatch.setattr(WebSearch_APIs, "analyze_and_aggregate", analyze)
    web_tool_impls._reset_state_for_tests()
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda hub: EffectiveToolState(
            state="allow", origin="tool_override"
        ),
        kill_switch=lambda: False,
    )
    yield adapter, provider, calls, config_path
    web_tool_impls._reset_state_for_tests()


def _invoke_both(provider, **overrides):
    basic = provider.invoke(
        "local:web_search", {"query": "routing", **overrides.get("basic", {})}
    )
    deep = provider.invoke(
        "local:web_deep_search", {"question": "routing", **overrides.get("deep", {})}
    )
    assert basic.ok, basic.content
    assert deep.ok, deep.content
    return basic.content, deep.content


def test_settings_save_changes_next_basic_and_deep_call_without_rebuilding_provider(
    search_profile,
):
    adapter, provider, calls, config_path = search_profile
    assert adapter.save_values("SearchSettings", {"search_provider_default": "serper"})
    first = _invoke_both(provider)
    assert calls == [("basic", "serper"), ("deep", "serper")]
    assert all("Engine: serper (saved default)" in text for text in first)

    assert adapter.save_values("SearchSettings", {"search_provider_default": "exa"})
    second = _invoke_both(provider)
    assert calls[-2:] == [("basic", "exa"), ("deep", "exa")]
    assert all("Engine: exa (saved default)" in text for text in second)
    assert (
        tomllib.loads(config_path.read_text())["SearchSettings"][
            "search_provider_default"
        ]
        == "exa"
    )


@pytest.mark.parametrize("saved", [None, "google"])
def test_missing_preference_uses_shared_fallback_and_preserves_existing_google(
    search_profile, saved
):
    adapter, provider, calls, config_path = search_profile
    if saved is not None:
        assert adapter.save_values("SearchSettings", {"search_provider_default": saved})
    before = config_path.read_bytes()
    output = _invoke_both(provider)
    expected = "google" if saved else "duckduckgo"
    source = "saved default" if saved else "application default"
    assert calls == [("basic", expected), ("deep", expected)]
    assert all(f"Engine: {expected} ({source})" in text for text in output)
    assert config_path.read_bytes() == before
    assert (
        config.load_settings(force_reload=True)["search_settings_general"][
            "default_search_provider"
        ]
        == expected
    )
    assert web_tool_impls.deep_search_pipeline_params()["engine"] == expected


def test_explicit_override_is_temporary_and_cache_hit_uses_current_provenance(
    search_profile,
):
    adapter, provider, calls, config_path = search_profile
    assert adapter.save_values("SearchSettings", {"search_provider_default": "serper"})
    before = config_path.read_bytes()
    explicit = _invoke_both(
        provider, basic={"search_engine": " EXA "}, deep={"engine": " EXA "}
    )
    assert calls == [("basic", "exa"), ("deep", "exa")]
    assert all("Engine: exa (call override)" in text for text in explicit)
    assert config_path.read_bytes() == before
    inherited = _invoke_both(provider)
    assert calls[-2:] == [("basic", "serper"), ("deep", "serper")]
    assert all("Engine: serper (saved default)" in text for text in inherited)

    assert adapter.save_values("SearchSettings", {"search_provider_default": "exa"})
    cached = provider.invoke("local:web_search", {"query": "routing"})
    assert cached.ok and "Engine: exa (saved default)" in cached.content
    assert "call override" not in cached.content
    assert calls == [
        ("basic", "exa"),
        ("deep", "exa"),
        ("basic", "serper"),
        ("deep", "serper"),
    ]


@pytest.mark.parametrize("invalid", ["brvae", "", "  ", 123, False])
def test_invalid_saved_backend_fails_before_dispatch_but_explicit_override_still_works(
    search_profile, invalid
):
    adapter, provider, calls, config_path = search_profile
    assert adapter.save_values("SearchSettings", {"search_provider_default": invalid})
    before = config_path.read_bytes()
    for name, args in [
        ("web_search", {"query": "routing"}),
        ("web_deep_search", {"question": "routing"}),
    ]:
        result = provider.invoke(f"local:{name}", args)
        assert not result.ok
        assert "search_provider_default" in result.error
        assert "invalid-args" not in result.error
    assert calls == []
    _invoke_both(provider, basic={"search_engine": "serper"}, deep={"engine": "serper"})
    assert calls == [("basic", "serper"), ("deep", "serper")]
    assert config_path.read_bytes() == before


@pytest.mark.parametrize("invalid", ["unknown", "", 123])
def test_invalid_explicit_backend_does_not_fall_back_to_saved_preference(
    search_profile, invalid
):
    adapter, provider, calls, _ = search_profile
    assert adapter.save_values("SearchSettings", {"search_provider_default": "serper"})
    for name, args in [
        ("web_search", {"query": "routing", "search_engine": invalid}),
        ("web_deep_search", {"question": "routing", "engine": invalid}),
    ]:
        result = provider.invoke(f"local:{name}", args)
        assert not result.ok and "invalid-args" in result.error
    assert calls == []


def test_cached_result_reserves_space_for_longer_application_default_label(
    search_profile, monkeypatch
):
    _, provider, calls, _ = search_profile
    monkeypatch.setattr(web_tool_impls, "SEARCH_TOTAL_MAX_BYTES", 200)

    def long_results(**kwargs):
        calls.append(("basic", kwargs["search_engine"]))
        return {
            "results": [{"title": "R", "url": "https://e.com", "content": "x" * 78}] * 2
        }

    monkeypatch.setattr(WebSearch_APIs, "perform_websearch", long_results)
    explicit = provider.invoke(
        "local:web_search", {"query": "routing", "search_engine": "duckduckgo"}
    )
    inherited = provider.invoke("local:web_search", {"query": "routing"})
    assert explicit.ok and inherited.ok
    assert calls == [("basic", "duckduckgo")]
    assert "application default" in inherited.content and "omitted" in inherited.content
    assert len(explicit.content.encode("utf-8")) <= 200
    assert len(inherited.content.encode("utf-8")) <= 200
