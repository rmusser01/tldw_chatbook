"""Exa/Serper/Yandex backends (task-1355): request-shape + parser pins, live smoke."""

import base64
import json

import pytest

from tldw_chatbook.Web_Scraping import WebSearch_APIs


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise WebSearch_APIs.requests.exceptions.HTTPError(f"status {self.status_code}")


class _FakeRequests:
    """Stands in for the module's `requests` import; records the call."""

    exceptions = None  # filled in _patch_requests with the real exceptions module

    def __init__(self, payload, status_code=200):
        self.calls = []
        self._payload = payload
        self._status = status_code

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self._payload, self._status)

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self._payload, self._status)


def _patch_requests(monkeypatch, payload, status_code=200):
    fake = _FakeRequests(payload, status_code)
    fake.exceptions = WebSearch_APIs.requests.exceptions  # keep real exception types
    monkeypatch.setattr(WebSearch_APIs, "requests", fake)
    return fake


def _set_key(monkeypatch, key, value):
    monkeypatch.setitem(WebSearch_APIs.loaded_config_data["search_engines"], key, value)


# ---------------------------------------------------------------------------
# Serper
# ---------------------------------------------------------------------------

_SERPER_PAYLOAD = {
    "organic": [
        {"title": "Result One", "link": "https://one.example/", "snippet": "first snippet",
         "position": 1, "date": "2026-01-01"},
        {"title": "Result Two", "link": "https://two.example/", "snippet": "second snippet",
         "position": 2},
    ]
}


def test_serper_request_shape(monkeypatch):
    _set_key(monkeypatch, "serper_search_api_key", "test-serper-key")
    fake = _patch_requests(monkeypatch, _SERPER_PAYLOAD)
    WebSearch_APIs.search_web_serper("cherry cake", "US", "en", 7)
    call = fake.calls[0]
    assert call["url"] == "https://google.serper.dev/search"
    assert call["headers"]["X-API-KEY"] == "test-serper-key"
    assert call["headers"]["Content-Type"] == "application/json"
    body = json.loads(call["data"]) if "data" in call else call["json"]
    assert body == {"q": "cherry cake", "gl": "us", "hl": "en", "num": 7}


def test_serper_request_defaults(monkeypatch):
    _set_key(monkeypatch, "serper_search_api_key", "test-serper-key")
    fake = _patch_requests(monkeypatch, _SERPER_PAYLOAD)
    WebSearch_APIs.search_web_serper("q", None, None, None)
    body = fake.calls[0].get("json") or json.loads(fake.calls[0]["data"])
    assert body["gl"] == "us" and body["hl"] == "en" and body["num"] == 10


def test_serper_missing_key_raises(monkeypatch):
    _set_key(monkeypatch, "serper_search_api_key", "")
    _patch_requests(monkeypatch, _SERPER_PAYLOAD)
    with pytest.raises(ValueError, match="[Ss]erper"):
        WebSearch_APIs.search_web_serper("q", "US", "en", 5)


def test_serper_parser_standard_shape():
    out = {}
    WebSearch_APIs.parse_serper_results(_SERPER_PAYLOAD, out)
    assert len(out["results"]) == 2
    first = out["results"][0]
    assert first["title"] == "Result One"
    assert first["url"] == "https://one.example/"
    assert first["content"] == "first snippet"
    assert first["metadata"]["snippet"] == "first snippet"
    assert first["metadata"]["position"] == 1
    assert first["metadata"]["date_published"] == "2026-01-01"
    assert first["metadata"]["relevance_score"] is None
    # absent organic key tolerated
    out2 = {}
    WebSearch_APIs.parse_serper_results({}, out2)
    assert out2["results"] == []


def test_serper_end_to_end_through_process(monkeypatch):
    _set_key(monkeypatch, "serper_search_api_key", "test-serper-key")
    _patch_requests(monkeypatch, _SERPER_PAYLOAD)
    raw = WebSearch_APIs.search_web_serper("q", "US", "en", 5)
    result = WebSearch_APIs.process_web_search_results(raw, "serper")
    assert result["processing_error"] is None
    assert [r["url"] for r in result["results"]] == ["https://one.example/", "https://two.example/"]


# ---------------------------------------------------------------------------
# Exa
# ---------------------------------------------------------------------------

_EXA_PAYLOAD = {
    "results": [
        {"title": "Exa One", "url": "https://exa-one.example/", "publishedDate": "2026-02-02",
         "author": "Ada", "highlights": ["highlight text one", "second highlight"]},
        {"title": "Exa Two", "url": "https://exa-two.example/", "author": None, "highlights": []},
    ]
}


def test_exa_request_shape(monkeypatch):
    _set_key(monkeypatch, "exa_search_api_key", "test-exa-key")
    fake = _patch_requests(monkeypatch, _EXA_PAYLOAD)
    WebSearch_APIs.search_web_exa("cherry cake", 5)
    call = fake.calls[0]
    assert call["url"] == "https://api.exa.ai/search"
    assert call["headers"]["x-api-key"] == "test-exa-key"
    body = call.get("json") or json.loads(call["data"])
    assert body == {"query": "cherry cake", "numResults": 5, "type": "auto",
                    "contents": {"highlights": True}}


def test_exa_missing_key_raises(monkeypatch):
    _set_key(monkeypatch, "exa_search_api_key", "")
    _patch_requests(monkeypatch, _EXA_PAYLOAD)
    with pytest.raises(ValueError, match="[Ee]xa"):
        WebSearch_APIs.search_web_exa("q", 5)


def test_exa_parser_standard_shape():
    out = {}
    WebSearch_APIs.parse_exa_results(_EXA_PAYLOAD, out)
    assert len(out["results"]) == 2
    first, second = out["results"]
    assert first["content"] == "highlight text one"
    assert first["metadata"]["snippet"] == "highlight text one"
    assert first["metadata"]["date_published"] == "2026-02-02"
    assert first["metadata"]["author"] == "Ada"
    assert second["content"] == ""  # empty highlights tolerated


def test_exa_end_to_end_through_process(monkeypatch):
    _set_key(monkeypatch, "exa_search_api_key", "test-exa-key")
    _patch_requests(monkeypatch, _EXA_PAYLOAD)
    raw = WebSearch_APIs.search_web_exa("q", 5)
    result = WebSearch_APIs.process_web_search_results(raw, "exa")
    assert result["processing_error"] is None
    assert len(result["results"]) == 2
