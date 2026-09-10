"""Shared saved search settings, proven through HTTP dispatch rather than provider mocks."""

import importlib
from unittest.mock import Mock

import pytest

from tldw_chatbook import config
from tldw_chatbook.Web_Scraping import WebSearch_APIs as api


def test_field_catalog_and_legacy_resolution(monkeypatch):
    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    assert len(settings.BACKENDS) == 10
    raw = {
        "SearchEngines": {
            "bing_search_api_key": "",
            "search_engine_api_key_bing": "legacy",
        }
    }
    monkeypatch.delenv("BING_SEARCH_API_KEY", raising=False)
    assert (
        settings.resolve_backend_fields("bing", raw)["bing_search_api_key"] == "legacy"
    )
    monkeypatch.setenv("BING_SEARCH_API_KEY", "environment")
    assert (
        settings.resolve_backend_fields("bing", raw)["bing_search_api_key"]
        == "environment"
    )
    assert (
        settings.field_source(settings.BACKENDS["bing"].fields[0], raw)
        == "Environment: BING_SEARCH_API_KEY"
    )
    assert raw["SearchEngines"]["bing_search_api_key"] == ""


def test_saved_brave_web_key_reaches_next_dispatch(monkeypatch):
    raw = {
        "SearchEngines": {
            "brave_search_api_key": "first",
            "brave_search_ai_api_key": "wrong",
        }
    }
    monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", lambda: raw)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    response = Mock()
    response.json.return_value = {
        "web": {
            "results": [
                {"title": "Found", "url": "https://example.org", "description": "Text"}
            ]
        }
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(api.requests, "get", get)
    first = api.perform_websearch("brave", "query", "DE", "de", "de-DE", 1)
    assert first.get("processing_error") is None
    assert get.call_args.kwargs["headers"]["X-Subscription-Token"] == "first"
    raw["SearchEngines"]["brave_search_api_key"] = "second"
    api.perform_websearch("brave", "query", "DE", "de", "de-DE", 1)
    assert get.call_args.kwargs["headers"]["X-Subscription-Token"] == "second"
    assert get.call_args.kwargs["params"]["country"] == "DE"


def test_kagi_dispatch_uses_documented_endpoint_and_numeric_limit(monkeypatch):
    monkeypatch.setenv("KAGI_API_KEY", "test-kagi")
    response = Mock()
    response.json.return_value = {
        "data": [
            {"t": 0, "title": "Found", "url": "https://example.org", "snippet": "Text"}
        ]
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(api.requests, "get", get)
    result = api.perform_websearch("kagi", "query", "US", "en", "en", 1)
    assert result.get("processing_error") is None
    assert get.call_args.args[0] == "https://kagi.com/api/v0/search"
    assert get.call_args.kwargs["params"] == {"q": "query", "limit": 1}


def test_searx_dispatch_preserves_endpoint_options_and_parses_json_object(monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    monkeypatch.setenv(
        "SEARX_URL", "http://localhost:8080/search?format=json&engines=google"
    )
    response = Mock()
    response.headers = {"Content-Type": "application/json"}
    response.json.return_value = {
        "results": [{"title": "Found", "url": "https://example.org", "content": "Text"}]
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(api.requests.Session, "get", get)
    monkeypatch.setattr(api.time, "sleep", lambda _: None)
    result = api.perform_websearch("searx", "query", "US", "en", "en", 1)
    assert result.get("processing_error") is None
    assert result["results"][0]["url"] == "https://example.org"
    query = parse_qs(urlsplit(get.call_args.args[0]).query)
    assert query["format"] == ["json"]
    assert query["engines"] == ["google"]


@pytest.mark.parametrize(
    "value",
    [
        "https://a:b@host/search",
        "https://host/#fragment",
        "ftp://host",
        "https://searx.example.com/search",
        "http://host:bad",
    ],
)
def test_setup_rejects_invalid_searx_endpoint(monkeypatch, value):
    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    monkeypatch.delenv("SEARX_URL", raising=False)
    assert settings.setup_issues(
        "searx", {"SearchEngines": {"searx_search_api_url": value}}
    )


@pytest.mark.parametrize(
    "value",
    [
        "http://localhost:8080/search",
        "http://192.168.1.5/search",
        "https://search.example.org/search?format=json",
    ],
)
def test_setup_allows_configured_local_or_public_searx(monkeypatch, value):
    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    monkeypatch.delenv("SEARX_URL", raising=False)
    assert (
        settings.setup_issues(
            "searx", {"SearchEngines": {"searx_search_api_url": value}}
        )
        == []
    )


@pytest.mark.parametrize(
    "status,kind,expected",
    [
        (401, "auth", "Authentication"),
        (403, "auth", "Authentication"),
        (429, "rate_limit", "quota"),
        (500, "http", "failed"),
    ],
)
def test_probe_classifies_http_failures_without_secret_text(
    monkeypatch, status, kind, expected
):
    import requests

    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    monkeypatch.setenv("SERPER_API_KEY", "sentinel-secret")
    response = requests.Response()
    response.status_code = status
    response.url = "https://provider.example/search?key=sentinel-secret"
    error = requests.HTTPError(
        "sentinel-secret untrusted provider body", response=response
    )
    monkeypatch.setattr(api.requests, "post", Mock(side_effect=error))
    result = api.perform_websearch("serper", "query", "US", "en", "en", 1)
    assert result["error_kind"] == kind
    assert "sentinel-secret" not in str(result)
    probe = settings.probe_saved_backend("serper")
    assert not probe.ok and expected in probe.message
    assert "sentinel-secret" not in str(probe)


@pytest.mark.parametrize(
    "backend,field,env_var,header",
    [
        (
            "brave",
            "brave_search_api_key",
            "BRAVE_SEARCH_API_KEY",
            "X-Subscription-Token",
        ),
        ("serper", "serper_search_api_key", "SERPER_API_KEY", "X-API-KEY"),
        ("exa", "exa_search_api_key", "EXA_API_KEY", "x-api-key"),
        ("kagi", "kagi_search_api_key", "KAGI_API_KEY", "Authorization"),
        ("yandex", "yandex_search_api_key", "YANDEX_SEARCH_API_KEY", "Authorization"),
        ("google", "google_search_api_key", "GOOGLE_SEARCH_API_KEY", None),
        ("tavily", "tavily_search_api_key", "TAVILY_API_KEY", "Authorization"),
        (
            "bing",
            "bing_search_api_key",
            "BING_SEARCH_API_KEY",
            "Ocp-Apim-Subscription-Key",
        ),
    ],
)
def test_real_settings_save_and_environment_override_reach_http(
    monkeypatch, backend, field, env_var, header
):
    """Only outgoing HTTP is replaced; Settings persistence, resolution and dispatch run."""
    import json

    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    adapter = SettingsConfigAdapter()
    adapter.load(force_reload=True)
    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.delenv("GOOGLE_SEARCH_ENGINE_ID", raising=False)
    monkeypatch.delenv("YANDEX_FOLDER_ID", raising=False)
    response = Mock()
    response.headers = {}
    response.json.return_value = {
        "results": [],
        "data": [],
        "items": [],
        "organic": [],
        "webPages": {"value": []},
    }
    send = Mock(return_value=response)
    monkeypatch.setattr(api.requests, "get", send)
    monkeypatch.setattr(api.requests, "post", send)
    monkeypatch.setattr(api.requests.Session, "get", send)
    values = {
        field: "first-saved",
        "google_search_engine_id": "cx-saved",
        "yandex_search_folder_id": "folder-saved",
    }
    assert adapter.save_values("SearchEngines", values)

    def check(expected):
        api.perform_websearch(backend, "query", "US", "en", "en", 1)
        kwargs = send.call_args.kwargs
        if header:
            prefix = {"kagi": "Bot ", "yandex": "Api-Key ", "tavily": "Bearer "}.get(
                backend, ""
            )
            assert kwargs["headers"][header] == prefix + expected
        elif backend == "google":
            assert kwargs["params"]["key"] == expected
            assert kwargs["params"]["cx"] == "cx-saved"
        else:
            assert json.loads(kwargs["data"])["api_key"] == expected
        assert 0 < kwargs["timeout"] <= 30

    check("first-saved")
    assert adapter.save_values("SearchEngines", {field: "second-saved"})
    check("second-saved")
    monkeypatch.setenv(env_var, "environment")
    check("environment")
    assert (
        config.load_cli_config_and_ensure_existence()["SearchEngines"][field]
        == "second-saved"
    )


@pytest.mark.parametrize(
    "canonical,legacy,value",
    [
        ("bing_search_api_key", "search_engine_api_key_bing", "configured-bing"),
        (
            "searx_search_api_url",
            "search_engine_searx_api",
            "http://localhost:8080/search",
        ),
    ],
)
def test_loader_preserves_shipped_keys_and_legacy_aliases(
    monkeypatch, canonical, legacy, value
):
    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    for backend in settings.BACKENDS.values():
        for field in backend.fields:
            monkeypatch.delenv(field.env_var, raising=False)
    for saved in ({canonical: value}, {canonical: "", legacy: value}):
        assert config.save_settings_to_cli_config({"SearchEngines": saved})
        assert config.load_settings()["search_engines_keys"][canonical] == value
        assert api.initialize_config()["search_engines"][canonical] == value


@pytest.mark.parametrize(
    "backend", ["brave", "serper", "exa", "kagi", "google", "tavily", "searx"]
)
def test_probe_network_failure_is_closed_and_does_not_invoke_llm(monkeypatch, backend):
    import requests

    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    for field in settings.BACKENDS[backend].fields:
        monkeypatch.setenv(
            field.env_var,
            "http://localhost:8080/search" if backend == "searx" else "sentinel-secret",
        )
    fail = Mock(
        side_effect=requests.ConnectionError(
            "sentinel-secret private-host untrusted-body"
        )
    )
    monkeypatch.setattr(api.requests, "get", fail)
    monkeypatch.setattr(api.requests, "post", fail)
    monkeypatch.setattr(api.requests.Session, "get", fail)
    monkeypatch.setattr(
        api, "chat_api_call", Mock(side_effect=AssertionError("no LLM allowed"))
    )
    result = settings.probe_saved_backend(backend)
    assert not result.ok
    assert (
        result.message
        == "Could not connect. Check network access and the configured endpoint."
    )
    assert "sentinel" not in str(result)


@pytest.mark.parametrize("value", ["https://@host/search", "https://host\\path/search"])
def test_setup_rejects_empty_userinfo_and_backslash(monkeypatch, value):
    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    monkeypatch.delenv("SEARX_URL", raising=False)
    assert settings.setup_issues(
        "searx", {"SearchEngines": {"searx_search_api_url": value}}
    )


def test_probe_does_not_log_untrusted_provider_payload(monkeypatch):
    from loguru import logger

    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "sentinel-secret")
    monkeypatch.setenv("GOOGLE_SEARCH_ENGINE_ID", "engine")
    response = Mock()
    response.json.return_value = {
        "items": [],
        "unexpected": "sentinel-secret untrusted-body",
    }
    monkeypatch.setattr(api.requests, "get", Mock(return_value=response))
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        result = settings.probe_saved_backend("google")
    finally:
        logger.remove(sink)
    assert result.ok
    assert "sentinel-secret" not in "".join(messages)


@pytest.mark.parametrize(
    "country,expected",
    [
        ("US", "countryUS"),
        ("de", "countryDE"),
        ("GB", "countryUK"),
        ("countryNZ", "countryNZ"),
        ("countryUS|countryCA", "countryUS|countryCA"),
    ],
)
def test_google_dispatch_uses_country_restriction_values(
    monkeypatch, country, expected
):
    monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_SEARCH_ENGINE_ID", "test-engine")
    response = Mock()
    response.json.return_value = {"items": []}
    get = Mock(return_value=response)
    monkeypatch.setattr(api.requests, "get", get)
    result = api.perform_websearch("google", "query", country, "en", "en", 1)
    assert result["processing_error"] is None
    assert get.call_args.kwargs["params"]["cr"] == expected


@pytest.mark.parametrize(
    "backend",
    ["google", "brave", "bing", "kagi", "serper", "exa", "tavily", "searx", "yandex"],
)
def test_malformed_http_payload_never_leaks_through_parser_diagnostics(
    monkeypatch, backend
):
    """Real response JSON reaches real parsers; failures discard partial payloads."""
    import base64
    import json

    import requests
    from loguru import logger

    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    sentinel = "synthetic-secret-provider-body"
    hit = {
        "title": sentinel,
        "url": "https://example.org",
        "link": "https://example.org",
        "snippet": sentinel,
    }
    payloads = {
        "google": {"items": [], "searchInformation": {"totalResults": sentinel}},
        "brave": {"query": {"original": sentinel}, "web": {"results": [hit, None]}},
        "bing": {
            "queryContext": {"originalQuery": sentinel},
            "webPages": {"value": [hit, None]},
        },
        "kagi": {"meta": {"ms": 1, "id": sentinel}, "data": [{"t": 0, **hit}, None]},
        "serper": {"organic": [hit, None]},
        "exa": {"results": [hit, None]},
        "tavily": {"results": [hit, None]},
        "searx": {"results": [hit, None]},
        "yandex": {
            "rawData": base64.b64encode(
                f'<yandexsearch><error code="32">{sentinel}</error></yandexsearch>'.encode()
            ).decode()
        },
    }
    for field in settings.BACKENDS[backend].fields:
        monkeypatch.setenv(
            field.env_var,
            "http://localhost:8080/search" if backend == "searx" else "test-key",
        )
    response = requests.Response()
    response.status_code = 200
    response.headers["Content-Type"] = "application/json"
    response._content = json.dumps(payloads[backend]).encode()
    # Session.request is the external boundary for top-level requests and SearX/Bing sessions.
    monkeypatch.setattr(requests.Session, "request", Mock(return_value=response))
    messages = []
    sink = logger.add(lambda message: messages.append(str(message)))
    try:
        result = api.perform_websearch(backend, "query", "US", "en", "en", 1)
    finally:
        logger.remove(sink)
    assert result.get("processing_error")
    assert sentinel not in "".join(messages)
    assert sentinel not in str(result)
    assert result.get("results") == []


@pytest.mark.parametrize("status,kind", [(429, "rate_limit"), (503, "http")])
def test_duckduckgo_http_error_is_a_failed_search_and_probe(monkeypatch, status, kind):
    import requests

    settings = importlib.import_module(
        "tldw_chatbook.Web_Scraping.search_backend_settings"
    )
    response = requests.Response()
    response.status_code = status
    response._content = b"<html><body>provider-secret-error-page</body></html>"
    monkeypatch.setattr(requests.Session, "request", Mock(return_value=response))
    result = api.perform_websearch("duckduckgo", "query", "US", "en", "en", 1)
    assert result.get("error_kind") == kind
    probe = settings.probe_saved_backend("duckduckgo")
    assert not probe.ok
    assert "provider-secret" not in str(probe)
