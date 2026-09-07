"""Exa/Serper/Yandex backends (task-1355): request-shape + parser pins, live smoke.
Also google/brave/duckduckgo/kagi/tavily/searx timeout pins (task-3060)."""

import base64
import json

import pytest

from tldw_chatbook.Web_Scraping import WebSearch_APIs


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        # searx's response is inspected for Content-Type before .json()/.text
        # is chosen -- every OTHER engine's fake response never reads this,
        # so a JSON-flavored default is a safe, additive default for them.
        self.headers = headers if headers is not None else {"Content-Type": "application/json"}
        self.text = ""

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
# task-3060: HTTP timeouts for the six older backends (google, brave,
# duckduckgo, kagi, tavily, searx). serper/exa/yandex already got timeout=30
# in task-1355 (pinned above); bing already carries its own timeout=10
# (untouched, out of scope); baidu is a bare `pass` stub with no HTTP call
# (dropped from scope, AC #1 amended -- see the task file).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

_GOOGLE_PAYLOAD = {"items": [{"title": "G Title", "link": "https://g.example/", "snippet": "g snippet"}]}


def test_google_request_carries_timeout(monkeypatch):
    _set_key(monkeypatch, "google_search_api_key", "test-google-key")
    _set_key(monkeypatch, "google_search_engine_id", "test-cx")
    fake = _patch_requests(monkeypatch, _GOOGLE_PAYLOAD)
    WebSearch_APIs.search_web_google("cherry cake")
    assert fake.calls, "search_web_google made no request"
    for call in fake.calls:
        assert call["timeout"] == 30


def test_google_timeout_raises_existing_error_contract(monkeypatch):
    """AC #2: a simulated hang must surface as a bounded-time error, not
    block indefinitely. search_web_google's existing contract (unchanged by
    this task) is to log and re-raise a RequestException unmodified -- a
    requests.Timeout IS a RequestException, so this proves the SAME
    behavior the code already had for any other network failure, now
    reachable because the call finally carries a timeout at all."""
    _set_key(monkeypatch, "google_search_api_key", "test-google-key")
    _set_key(monkeypatch, "google_search_engine_id", "test-cx")

    class _TimeoutRequests:
        exceptions = WebSearch_APIs.requests.exceptions

        def get(self, *a, **k):
            raise WebSearch_APIs.requests.exceptions.Timeout("simulated hang")

    monkeypatch.setattr(WebSearch_APIs, "requests", _TimeoutRequests())
    with pytest.raises(WebSearch_APIs.requests.exceptions.Timeout):
        WebSearch_APIs.search_web_google("cherry cake")


# ---------------------------------------------------------------------------
# Brave
# ---------------------------------------------------------------------------

_BRAVE_PAYLOAD = {"web": {"results": [{"title": "Br Title", "url": "https://br.example/",
                                        "description": "br snippet"}]}}


def test_brave_request_carries_timeout(monkeypatch):
    _set_key(monkeypatch, "brave_search_ai_api_key", "test-brave-key")
    fake = _patch_requests(monkeypatch, _BRAVE_PAYLOAD)
    WebSearch_APIs.search_web_brave("cherry cake", "US", "en", "en", 10)
    assert fake.calls, "search_web_brave made no request"
    for call in fake.calls:
        assert call["timeout"] == 30


# ---------------------------------------------------------------------------
# DuckDuckGo
# ---------------------------------------------------------------------------


class _FakeDDGResponse:
    def __init__(self, content: bytes):
        self.content = content


class _FakeDDGRequests:
    """DuckDuckGo's response is read via `.content` (raw bytes), not
    `.json()` -- a distinct fake shape from `_FakeRequests` above."""

    def __init__(self, content: bytes):
        self.calls = []
        self._content = content

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeDDGResponse(self._content)


def test_duckduckgo_request_carries_timeout(monkeypatch):
    if not WebSearch_APIs.LXML_AVAILABLE:
        pytest.skip("lxml not available")
    # "No  results." (DDG's own literal, two spaces) short-circuits the
    # function immediately after the request -- avoids needing a realistic
    # lxml-parseable HTML fixture just to pin the request kwargs.
    fake = _FakeDDGRequests(b"No  results.")
    monkeypatch.setattr(WebSearch_APIs, "requests", fake)
    WebSearch_APIs.search_web_duckduckgo("cherry cake")
    assert fake.calls, "search_web_duckduckgo made no request"
    for call in fake.calls:
        assert call["timeout"] == 30


# ---------------------------------------------------------------------------
# Kagi
# ---------------------------------------------------------------------------

_KAGI_PAYLOAD = {"data": [{"t": 0, "title": "K Title", "url": "https://k.example/", "snippet": "k snippet"}]}


def test_kagi_request_carries_timeout(monkeypatch):
    _set_key(monkeypatch, "kagi_search_api_key", "test-kagi-key")
    fake = _patch_requests(monkeypatch, _KAGI_PAYLOAD)
    WebSearch_APIs.search_web_kagi("cherry cake", 10)
    assert fake.calls, "search_web_kagi made no request"
    for call in fake.calls:
        assert call["timeout"] == 30


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
    assert call["timeout"] == 30
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


def test_serper_http_error_raises(monkeypatch):
    _set_key(monkeypatch, "serper_search_api_key", "test-serper-key")
    _patch_requests(monkeypatch, {}, status_code=429)
    with pytest.raises(WebSearch_APIs.requests.exceptions.HTTPError):
        WebSearch_APIs.search_web_serper("q", "US", "en", 5)


def test_serper_http_error_via_perform_websearch(monkeypatch):
    """perform_websearch's dispatch try/except (spec 5) must turn a backend
    HTTPError into its {"processing_error": ...} envelope, not crash the
    caller -- see the `except Exception as e` at the end of perform_websearch,
    which returns {"processing_error": f"Error performing web search: {e}"}."""
    _set_key(monkeypatch, "serper_search_api_key", "test-serper-key")
    _patch_requests(monkeypatch, {}, status_code=401)
    result = WebSearch_APIs.perform_websearch("serper", "q", "US", "en", "en", 5)
    assert isinstance(result, dict)
    assert result.get("processing_error") is not None
    assert "401" in result["processing_error"]


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
    assert call["timeout"] == 30
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
    assert first["title"] == "Exa One"
    assert first["url"] == "https://exa-one.example/"
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


def test_exa_http_error_raises(monkeypatch):
    _set_key(monkeypatch, "exa_search_api_key", "test-exa-key")
    _patch_requests(monkeypatch, {}, status_code=401)
    with pytest.raises(WebSearch_APIs.requests.exceptions.HTTPError):
        WebSearch_APIs.search_web_exa("q", 5)


# ---------------------------------------------------------------------------
# Yandex
# ---------------------------------------------------------------------------

_YANDEX_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
  <response date="20260806T000000">
    <results>
      <grouping>
        <group>
          <doc>
            <url>https://ya-one.example/</url>
            <title>Ya <hlword>One</hlword> Title</title>
            <passages>
              <passage>First <hlword>passage</hlword> text.</passage>
              <passage>Second passage.</passage>
            </passages>
          </doc>
        </group>
        <group>
          <doc>
            <url>https://ya-two.example/</url>
            <title>Ya Two</title>
          </doc>
        </group>
      </grouping>
    </results>
  </response>
</yandexsearch>"""

_YANDEX_ERROR_XML = b"""<?xml version="1.0" encoding="utf-8"?>
<yandexsearch version="1.0">
  <response><error code="32">Quota exhausted for this key</error></response>
</yandexsearch>"""


def _yandex_payload(xml_bytes):
    return {"rawData": base64.b64encode(xml_bytes).decode("ascii")}


def test_yandex_request_shape(monkeypatch):
    _set_key(monkeypatch, "yandex_search_api_key", "test-ya-key")
    _set_key(monkeypatch, "yandex_search_folder_id", "test-folder")
    fake = _patch_requests(monkeypatch, _yandex_payload(_YANDEX_XML))
    WebSearch_APIs.search_web_yandex("cherry cake", 5)
    call = fake.calls[0]
    assert call["url"] == "https://searchapi.api.cloud.yandex.net/v2/web/search"
    assert call["headers"]["Authorization"] == "Api-Key test-ya-key"
    assert call["timeout"] == 30
    body = call.get("json") or json.loads(call["data"])
    assert body == {
        "query": {"searchType": "SEARCH_TYPE_COM", "queryText": "cherry cake"},
        "folderId": "test-folder",
        "responseFormat": "FORMAT_XML",
    }


def test_yandex_missing_key_or_folder_raises(monkeypatch):
    _patch_requests(monkeypatch, _yandex_payload(_YANDEX_XML))
    _set_key(monkeypatch, "yandex_search_api_key", "")
    _set_key(monkeypatch, "yandex_search_folder_id", "test-folder")
    with pytest.raises(ValueError, match="[Yy]andex"):
        WebSearch_APIs.search_web_yandex("q", 5)
    _set_key(monkeypatch, "yandex_search_api_key", "test-ya-key")
    _set_key(monkeypatch, "yandex_search_folder_id", "")
    with pytest.raises(ValueError, match="folder"):
        WebSearch_APIs.search_web_yandex("q", 5)


def test_yandex_parser_flattens_hlwords_and_passages():
    out = {}
    WebSearch_APIs.parse_yandex_results(_yandex_payload(_YANDEX_XML), out)
    assert len(out["results"]) == 2
    first, second = out["results"]
    assert first["url"] == "https://ya-one.example/"
    assert first["title"] == "Ya One Title"                # hlword flattened
    assert first["content"] == "First passage text. Second passage."
    assert first["metadata"]["snippet"] == "First passage text. Second passage."
    assert second["content"] == ""                          # passage-less doc tolerated


def test_yandex_error_through_process_sets_processing_error():
    """In-XML <error> (quota/auth inside HTTP 200) must surface via the
    processing_error seam — never a silent empty result list."""
    result = WebSearch_APIs.process_web_search_results(_yandex_payload(_YANDEX_ERROR_XML), "yandex")
    assert result["processing_error"] is not None
    assert "32" in result["processing_error"] or "Quota" in result["processing_error"]
    assert result["results"] == []


def test_yandex_end_to_end_through_process(monkeypatch):
    _set_key(monkeypatch, "yandex_search_api_key", "test-ya-key")
    _set_key(monkeypatch, "yandex_search_folder_id", "test-folder")
    _patch_requests(monkeypatch, _yandex_payload(_YANDEX_XML))
    raw = WebSearch_APIs.search_web_yandex("q", 5)
    result = WebSearch_APIs.process_web_search_results(raw, "yandex")
    assert result["processing_error"] is None
    assert [r["url"] for r in result["results"]] == ["https://ya-one.example/", "https://ya-two.example/"]


def test_yandex_http_error_raises(monkeypatch):
    _set_key(monkeypatch, "yandex_search_api_key", "test-ya-key")
    _set_key(monkeypatch, "yandex_search_folder_id", "test-folder")
    _patch_requests(monkeypatch, {}, status_code=429)
    with pytest.raises(WebSearch_APIs.requests.exceptions.HTTPError):
        WebSearch_APIs.search_web_yandex("q", 5)


# ---------------------------------------------------------------------------
# Tavily (task-2990)
# ---------------------------------------------------------------------------

# Real Tavily API response shape: search_web_tavily returns response.json()
# unmodified on success.
_TAVILY_PAYLOAD = {
    "query": "cherry cake",
    "results": [
        {"title": "Tavily One", "url": "https://tavily-one.example/", "content": "first content",
         "score": 0.87, "published_date": "2026-03-03"},
        {"title": "Tavily Two", "url": "https://tavily-two.example/", "content": "second content",
         "score": 0.42},
    ],
}

_TAVILY_ERROR_STRING = "There was an error searching for content. boom"


def test_tavily_parser_standard_shape():
    out = {}
    WebSearch_APIs.parse_tavily_results(_TAVILY_PAYLOAD, out)
    assert len(out["results"]) == 2
    first, second = out["results"]
    assert first["title"] == "Tavily One"
    assert first["url"] == "https://tavily-one.example/"
    assert first["content"] == "first content"
    assert first["metadata"]["snippet"] == "first content"
    assert first["metadata"]["relevance_score"] == 0.87
    assert first["metadata"]["date_published"] == "2026-03-03"
    # Tavily's score is a real 0-1 relevance score -- correctly directioned,
    # unlike serper's rank-based "position".
    assert second["metadata"]["relevance_score"] == 0.42
    assert second["metadata"]["date_published"] is None


def test_tavily_parser_absent_results_tolerated():
    out = {}
    WebSearch_APIs.parse_tavily_results({}, out)
    assert out["results"] == []


def test_tavily_error_string_raises_and_surfaces_as_processing_error():
    """search_web_tavily returns a plain error STRING (not a dict) on
    request failure. The parser must raise ValueError with that text
    directly, AND process_web_search_results (which must not choke on the
    non-dict input) must surface it as processing_error rather than
    silently producing zero results (task-2990)."""
    with pytest.raises(ValueError, match="boom"):
        WebSearch_APIs.parse_tavily_results(_TAVILY_ERROR_STRING, {})

    result = WebSearch_APIs.process_web_search_results(_TAVILY_ERROR_STRING, "tavily")
    assert result["processing_error"] is not None
    assert "boom" in result["processing_error"]
    assert result["results"] == []


def test_tavily_end_to_end_through_process(monkeypatch):
    _set_key(monkeypatch, "tavily_search_api_key", "test-tavily-key")
    _patch_requests(monkeypatch, _TAVILY_PAYLOAD)
    raw = WebSearch_APIs.search_web_tavily("q")
    result = WebSearch_APIs.process_web_search_results(raw, "tavily")
    assert result["processing_error"] is None
    assert [r["url"] for r in result["results"]] == [
        "https://tavily-one.example/", "https://tavily-two.example/",
    ]


def test_tavily_request_carries_timeout(monkeypatch):
    """task-3060."""
    _set_key(monkeypatch, "tavily_search_api_key", "test-tavily-key")
    fake = _patch_requests(monkeypatch, _TAVILY_PAYLOAD)
    WebSearch_APIs.search_web_tavily("cherry cake")
    assert fake.calls, "search_web_tavily made no request"
    for call in fake.calls:
        assert call["timeout"] == 30


def test_tavily_parser_non_dict_item_raises_value_error():
    """Per-item shape validation: a list element that is not a dict must raise ValueError
    with the index, RED-first test to prove the index is reported."""
    bad_payload = {
        "results": [
            {"title": "Good", "url": "https://good.example/", "content": "ok"},
            "not a dict",  # Bad: string at index 1
            {"title": "Also Good", "url": "https://also.example/", "content": "ok"},
        ]
    }
    with pytest.raises(ValueError, match="index 1"):
        WebSearch_APIs.parse_tavily_results(bad_payload, {})


def test_tavily_non_dict_item_surfaces_as_processing_error():
    """Non-dict items in tavily results must surface as processing_error via the seam."""
    bad_payload = {
        "results": [
            "not a dict",  # Bad: string at index 0
        ]
    }
    result = WebSearch_APIs.process_web_search_results(bad_payload, "tavily")
    assert result["processing_error"] is not None
    assert "index 0" in result["processing_error"]
    assert result["results"] == []


# ---------------------------------------------------------------------------
# Searx (task-2990)
# ---------------------------------------------------------------------------

# Real SearX/SearXNG shape as search_web_searx hands it to the parser: it
# always returns a JSON-encoded STRING (not a dict), of either a hit list
# or an error dict -- unlike every other backend in this file.
_SEARX_HITS = [
    {"title": "Searx One", "link": "https://searx-one.example/", "snippet": "first snippet",
     "publishedDate": "2026-04-04"},
    {"title": "Searx Two", "link": "https://searx-two.example/", "snippet": "second snippet"},
]
_SEARX_PAYLOAD = json.dumps(_SEARX_HITS)
_SEARX_ERROR_PAYLOAD = json.dumps({"error": "No information was found online for the search query."})


class _FakeSearxSession:
    """search_web_searx breaks the standard `requests.get/post` idiom every
    other engine here uses: it calls `searx_create_session()` ->
    `requests.Session()` -> `session.get(...)`. `_FakeRequests` above has no
    `.Session()` stub, so this test patches `searx_create_session` directly
    (task-3060, Important 5) rather than extending the shared fake."""

    def __init__(self, payload):
        self.calls = []
        self._payload = payload

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self._payload)


def test_searx_request_carries_timeout(monkeypatch):
    """task-3060. Also mocks `random.uniform` (Minor 8): search_web_searx
    calls `time.sleep(random.uniform(2, 5))` before every request --
    precedent: test_deep_search_pipeline.py:667."""
    _set_key(monkeypatch, "searx_search_api_url", "https://searx.example.com/search")
    fake_session = _FakeSearxSession(_SEARX_HITS)
    monkeypatch.setattr(WebSearch_APIs, "searx_create_session", lambda: fake_session)
    monkeypatch.setattr(WebSearch_APIs.random, "uniform", lambda a, b: 0.0)
    WebSearch_APIs.search_web_searx("cherry cake")
    assert fake_session.calls, "search_web_searx made no request"
    for call in fake_session.calls:
        assert call["timeout"] == 30


def test_searx_parser_loads_json_string():
    out = {}
    WebSearch_APIs.parse_searx_results(_SEARX_PAYLOAD, out)
    assert len(out["results"]) == 2
    first, second = out["results"]
    assert first["title"] == "Searx One"
    assert first["url"] == "https://searx-one.example/"
    assert first["content"] == "first snippet"
    assert first["metadata"]["snippet"] == "first snippet"
    assert first["metadata"]["date_published"] == "2026-04-04"
    assert first["metadata"]["relevance_score"] is None
    assert second["metadata"]["date_published"] is None


def test_searx_parser_accepts_already_parsed_list():
    """Defensive: direct/test callers may hand parse_searx_results an
    already-decoded list instead of the JSON string search_web_searx
    actually returns."""
    out = {}
    WebSearch_APIs.parse_searx_results(_SEARX_HITS, out)
    assert len(out["results"]) == 2


def test_searx_parser_falls_back_to_raw_searxng_shape_keys():
    """A pre-parsed list whose items use raw SearXNG API keys (url/content)
    rather than search_web_searx's own hit keys (link/snippet) must still
    populate url/content -- not silently yield "" rows (port reference's
    `item.get("link") or item.get("url")` / `... or item.get("content")`
    fallback pair)."""
    out = {}
    WebSearch_APIs.parse_searx_results(
        [{"title": "Raw SearXNG Hit", "url": "https://raw.example/", "content": "raw content"}],
        out,
    )
    assert out["results"][0]["url"] == "https://raw.example/"
    assert out["results"][0]["content"] == "raw content"
    assert out["results"][0]["metadata"]["snippet"] == "raw content"


def test_searx_empty_list_tolerated():
    out = {}
    WebSearch_APIs.parse_searx_results(json.dumps([]), out)
    assert out["results"] == []


def test_searx_unparseable_json_raises_value_error():
    with pytest.raises(ValueError):
        WebSearch_APIs.parse_searx_results("not json {", {})


def test_searx_non_error_dict_raises_value_error():
    """A decoded payload that's a dict WITHOUT an "error" key (or any other
    non-list shape) is not silently tolerated as "no results" -- it raises,
    matching the docstring's stated contract."""
    with pytest.raises(ValueError, match="[Uu]nexpected"):
        WebSearch_APIs.parse_searx_results(json.dumps({"query": "cherry cake"}), {})


def test_searx_json_scalar_raises_value_error():
    with pytest.raises(ValueError, match="[Uu]nexpected"):
        WebSearch_APIs.parse_searx_results(json.dumps(None), {})


def test_searx_error_dict_raises_and_surfaces_as_processing_error():
    """search_web_searx encodes its error as `json.dumps({"error": ...})`
    -- still a plain STRING. The parser must decode it, raise ValueError
    with the error text directly, AND process_web_search_results must
    surface it as processing_error rather than silently producing zero
    results (task-2990)."""
    with pytest.raises(ValueError, match="No information"):
        WebSearch_APIs.parse_searx_results(_SEARX_ERROR_PAYLOAD, {})

    result = WebSearch_APIs.process_web_search_results(_SEARX_ERROR_PAYLOAD, "searx")
    assert result["processing_error"] is not None
    assert "No information" in result["processing_error"]
    assert result["results"] == []


def test_searx_parser_non_dict_item_raises_value_error():
    """Per-item shape validation: a list element that is not a dict must raise ValueError
    with the index, RED-first test to prove the index is reported."""
    bad_payload = [
        {"title": "Good", "link": "https://good.example/", "snippet": "ok"},
        "not a dict",  # Bad: string at index 1
        {"title": "Also Good", "link": "https://also.example/", "snippet": "ok"},
    ]
    with pytest.raises(ValueError, match="index 1"):
        WebSearch_APIs.parse_searx_results(bad_payload, {})


def test_searx_non_dict_item_surfaces_as_processing_error():
    """Non-dict items in searx results must surface as processing_error via the seam."""
    bad_payload = json.dumps([
        "not a dict",  # Bad: string at index 0
    ])
    result = WebSearch_APIs.process_web_search_results(bad_payload, "searx")
    assert result["processing_error"] is not None
    assert "index 0" in result["processing_error"]
    assert result["results"] == []


# ---------------------------------------------------------------------------
# process_web_search_results type-guard scoping (task-2990 review round)
# ---------------------------------------------------------------------------

def test_type_guard_rejects_string_payload_for_non_string_engines():
    """The (dict, str-for-tavily/searx) type guard in
    process_web_search_results is scoped to just those two engines -- their
    local backends are the only ones that can hand back a string. A string
    payload for any OTHER engine must still raise TypeError, not silently
    reach that engine's parser: e.g. parse_brave_results does membership
    checks like `"query" in raw_results`, which a stray string would
    satisfy character-by-character and could silently produce zero results
    with no error at all -- exactly the defect class this task exists to
    close, just relocated to a different engine."""
    with pytest.raises(TypeError, match="dictionary"):
        WebSearch_APIs.process_web_search_results("some error text", "brave")


def test_type_guard_still_rejects_non_dict_non_str_even_for_tavily():
    """The str allowance is specifically `isinstance(x, str)`, not "not a
    dict" -- None (or any other non-dict, non-str value) must still raise
    TypeError even for tavily/searx."""
    with pytest.raises(TypeError, match="dictionary"):
        WebSearch_APIs.process_web_search_results(None, "tavily")


# ---------------------------------------------------------------------------
# Parity: every engine SEARCH_ENGINES advertises must produce real results
# ---------------------------------------------------------------------------

# Emptied by task-2990: parse_tavily_results / parse_searx_results were
# `pass` stubs, so a real API response parsed to zero results and the
# caller rendered that as "No results found" for an engine advertised as
# working. Both now have real implementations (see WebSearch_APIs.py);
# every engine goes through the normal (>=1 result) assertion branch below.
_KNOWN_BROKEN_PARSERS = set()  # see task-2990

# One minimal, realistic non-empty payload per engine, shaped like that
# engine's own real API response (read from each search_web_*/parse_*
# function above) -- not a generic placeholder. exa/serper/yandex reuse
# the payload constants already defined above for their dedicated tests.
_ENGINE_SAMPLE_PAYLOADS = {
    "google": {
        "items": [{"title": "G Title", "link": "https://g.example/", "snippet": "g snippet"}]
    },
    "bing": {
        "webPages": {"value": [{"name": "B Title", "url": "https://b.example/", "snippet": "b snippet"}]}
    },
    "duckduckgo": {
        "results": [{"title": "D Title", "href": "https://d.example/", "body": "d snippet"}]
    },
    "brave": {
        "web": {"results": [{"title": "Br Title", "url": "https://br.example/", "description": "br snippet"}]}
    },
    "kagi": {
        "data": [{"t": 0, "title": "K Title", "url": "https://k.example/", "snippet": "k snippet"}]
    },
    "exa": _EXA_PAYLOAD,
    "serper": _SERPER_PAYLOAD,
    "yandex": _yandex_payload(_YANDEX_XML),
    "tavily": _TAVILY_PAYLOAD,
    "searx": _SEARX_PAYLOAD,
}


def test_agent_enum_engines_all_dispatchable():
    """Every engine the agent tool advertises must reach a real backend that
    parses a realistic, non-empty payload of ITS OWN shape into at least one
    standardized result. _KNOWN_BROKEN_PARSERS (task-2990) is now empty --
    tavily and searx both have real parsers -- so every engine goes through
    this one assertion branch; a regression back to a `pass` stub would fail
    it immediately.
    """
    assert _KNOWN_BROKEN_PARSERS == set(), "task-2990 allowlist should stay empty"
    from tldw_chatbook.Tools.web_tool_impls import SEARCH_ENGINES
    for engine in SEARCH_ENGINES:
        payload = _ENGINE_SAMPLE_PAYLOADS.get(engine)
        assert payload is not None, f"no sample payload registered for advertised engine {engine!r}"
        result = WebSearch_APIs.process_web_search_results(payload, engine)
        assert result["processing_error"] is None, (
            f"{engine}: unexpected processing_error parsing a minimal {engine} payload: "
            f"{result['processing_error']}"
        )
        assert len(result["results"]) >= 1, (
            f"agent enum advertises {engine!r} but its parser produced zero results "
            f"from a minimal, realistic {engine} payload"
        )


# ---------------------------------------------------------------------------
# Live smoke (triple-gated: --run-live + TLDW_LIVE_SEARCH_TESTS=1 + key files)
# ---------------------------------------------------------------------------

import os
import sys
from pathlib import Path

# Key files at the checkout root are covered by the tracked .gitignore rules
# (*-api-key.txt, yandex-folder-id.txt), so they're read from there without
# risk of an accidental commit; when running from a worktree, copy key files
# there before running with --run-live.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_ENABLED = os.environ.get("TLDW_LIVE_SEARCH_TESTS") == "1"


def _key_file(name):
    """Read a key file. Only called inside gated test bodies, never at collection time."""
    p = _REPO_ROOT / name
    return p.read_text().strip() if p.exists() else ""


# ---------------------------------------------------------------------------
# _usable_key pins (RED-first: the gate runs at import/decoration time, so an
# in-test env-flag monkeypatch can't retrigger it -- unit-test the predicate
# directly instead of the decorator machinery, and never touch a real key.)
# ---------------------------------------------------------------------------

def test_usable_key_missing_file_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_REPO_ROOT", tmp_path)
    assert _usable_key("does-not-exist.txt") is False


def test_usable_key_empty_file_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_REPO_ROOT", tmp_path)
    (tmp_path / "empty-key.txt").write_text("")
    assert _usable_key("empty-key.txt") is False


def test_usable_key_whitespace_only_file_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_REPO_ROOT", tmp_path)
    (tmp_path / "whitespace-key.txt").write_text("   \n\t  \n")
    assert _usable_key("whitespace-key.txt") is False


def test_usable_key_unreadable_binary_file_is_unusable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_REPO_ROOT", tmp_path)
    # Invalid UTF-8 byte sequence: read_text() raises UnicodeDecodeError,
    # which _usable_key must treat as "unusable", not let propagate.
    (tmp_path / "binary-key.txt").write_bytes(b"\xff\xfe\x00\x01\x02\xfd\x80\x81")
    assert _usable_key("binary-key.txt") is False


def test_usable_key_normal_content_is_usable(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_REPO_ROOT", tmp_path)
    (tmp_path / "real-key.txt").write_text("sk-not-a-real-key\n")
    assert _usable_key("real-key.txt") is True


def _usable_key(name):
    """True if the key file exists, is readable as text, and has non-blank
    content after stripping whitespace.

    A guarded read: only ever called from inside `_live_gate` after the
    `_LIVE_ENABLED` short-circuit, so this never runs on an ordinary test
    pass. Unreadable content (invalid encoding, OS-level read error) is
    treated the same as a missing file -- never lets the exception escape.
    """
    try:
        return bool(_key_file(name))
    except (OSError, UnicodeDecodeError):
        return False


def _live_gate(*key_names):
    """Double-gated marker: env flag first (short-circuit), then usable-key check.

    Returns pytest.mark.skip if env flag off or any key file is missing,
    empty/whitespace-only, or unreadable; pytest.mark.live otherwise. This
    marker is then subject to the --run-live CLI backstop in conftest.
    """
    if not _LIVE_ENABLED:
        return pytest.mark.skip(reason="TLDW_LIVE_SEARCH_TESTS != 1")
    missing = [n for n in key_names if not _usable_key(n)]
    if missing:
        return pytest.mark.skip(reason=f"missing/unusable key file(s): {', '.join(missing)}")
    return pytest.mark.live


@_live_gate("serper-api-key.txt")
def test_live_serper(monkeypatch):
    _set_key(monkeypatch, "serper_search_api_key", _key_file("serper-api-key.txt"))
    result = WebSearch_APIs.process_web_search_results(
        WebSearch_APIs.search_web_serper("python programming language", "US", "en", 3), "serper"
    )
    assert result["processing_error"] is None
    assert result["results"] and result["results"][0]["url"]


@_live_gate("exa-api-key.txt")
def test_live_exa(monkeypatch):
    _set_key(monkeypatch, "exa_search_api_key", _key_file("exa-api-key.txt"))
    result = WebSearch_APIs.process_web_search_results(
        WebSearch_APIs.search_web_exa("python programming language", 3), "exa"
    )
    assert result["processing_error"] is None
    assert result["results"] and result["results"][0]["url"]


@_live_gate("yandex-api-key.txt", "yandex-folder-id.txt")
def test_live_yandex(monkeypatch):
    _set_key(monkeypatch, "yandex_search_api_key", _key_file("yandex-api-key.txt"))
    _set_key(monkeypatch, "yandex_search_folder_id", _key_file("yandex-folder-id.txt"))
    result = WebSearch_APIs.process_web_search_results(
        WebSearch_APIs.search_web_yandex("python programming language", 3), "yandex"
    )
    assert result["processing_error"] is None
    assert result["results"] and result["results"][0]["url"]
