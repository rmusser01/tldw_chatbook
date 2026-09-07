# Search Backends: Exa, Serper, Yandex — Implementation Plan (task-1355)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make engine choice real — implement the serper/yandex stubs and the absent exa backend in `Web_Scraping/WebSearch_APIs.py`, wire every surface that enumerates engines, per spec `Docs/superpowers/specs/2026-08-06-search-backends-exa-serper-yandex-design.md`.

**Architecture:** Three sync `requests`-based backends + standardized-shape parsers following `search_web_brave`/`parse_brave_results`; dispatch fixes in `perform_websearch` + `process_web_search_results`; config keys in loader + in-file TOML template; enum sweep across six surfaces; mocked unit tests + double-gated live tests.

**Tech Stack:** requests, defusedxml-with-fallback (Yandex XML), pytest with monkeypatched module attributes.

## Global Constraints

- **NEVER print key-file contents** — not even head/tail fragments — into logs, test output, or reports (spec §1; a prior programme leaked key bytes into a transcript and forced a rotation).
- Missing key/config → `ValueError` with a clear "Please provide a valid <engine> …" message (brave's pattern); HTTP failures → `response.raise_for_status()` — never swallowed.
- Standardized result shape (every parser): items with `title`/`url`/`content` strings + `metadata` dict containing at least `snippet`; appended to `output_dict["results"]` (initialize the list if absent) — `parse_brave_results` is the reference.
- Yandex request body: ONLY the verified fields `{"query": {"searchType": "SEARCH_TYPE_COM", "queryText": …}, "folderId": …, "responseFormat": "FORMAT_XML"}` — no `groupSpec` (spec §2).
- Exa body includes `"contents": {"highlights": true}` — a recorded paid trade for snippets (spec §2).
- `--strict-markers` is active: the `live` marker MUST be registered in `pyproject.toml` before any test uses it (unregistered = collection error).
- Live tests double-gated: per-engine key file AND `TLDW_LIVE_SEARCH_TESTS=1` (spec §5). Live tests never run in the standard gates.
- Tests run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/test_search_backends.py -v -p no:randomly` (foreground; venv only). Never `git stash`.
- TDD: failing tests first (RED for the stated reason), then implement.

---

### Task 1: Tracked ignore rules for credential files (security prerequisite)

**Files:**
- Modify: `.gitignore` (the block containing `/openai-api-key.txt` at ~line 209)

**Interfaces:** none — but this MUST be the branch's first implementation commit (spec §1): it lands before anything invites key files into the tree.

- [ ] **Step 1: Edit `.gitignore`** — below the two existing key lines, add:

```
*-api-key.txt
/yandex-folder-id.txt
```

(Keep `/openai-api-key.txt` and `/moonshot-api-key.txt`; the glob subsumes them but they stay for history clarity.)

- [ ] **Step 2: Verify with git plumbing** (no test suite for this):

Run: `git check-ignore -v exa-api-key.txt serper-api-key.txt yandex-api-key.txt yandex-folder-id.txt anthropic-api-key.txt`
Expected: every path reported ignored **by `.gitignore`** (not `.git/info/exclude`).

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore(security): track ignore rules for credential files (*-api-key.txt, yandex-folder-id.txt)"
```

---

### Task 2: Serper backend

**Files:**
- Modify: `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` (stub `search_web_serper` ~line 2829, stub `parse_serper_results` ~line 2837, dispatch branch ~line 1199)
- Modify: `tldw_chatbook/config.py` (loader `search_engines` dict ~line 2182 beside `tavily_search_api_key`; TOML template `[search_engines]` block ~line 3695)
- Test: Create `Tests/Web_Scraping/test_search_backends.py`

**Interfaces:**
- Produces: `search_web_serper(search_query, content_country, search_lang, result_count) -> dict` (raw Serper JSON); `parse_serper_results(raw_results, output_dict) -> None`; config key `serper_search_api_key`.
- Test scaffolding produced for Tasks 3/4: `_FakeResponse`, `_patch_requests`, `_set_key` helpers as written below.

- [ ] **Step 1: Write the failing tests** (create `Tests/Web_Scraping/test_search_backends.py`)

```python
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
```

- [ ] **Step 2: Run to verify RED**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/test_search_backends.py -v -p no:randomly -k serper`
Expected: FAIL — `search_web_serper()` takes 0 arguments (stub), parser returns None output/`KeyError: 'results'` (stub `pass`), missing-key test fails (no ValueError raised).

- [ ] **Step 3: Implement**

Replace the serper stubs (~line 2829):

```python
def search_web_serper(search_query, content_country=None, search_lang=None, result_count=None):
    """Query the Serper google-search API and return its raw JSON.

    Args:
        search_query: The query string.
        content_country: 2-letter country code for `gl` (lowercased; default "us").
        search_lang: Interface language for `hl` (default "en").
        result_count: Number of organic results (default 10).

    Returns:
        dict: Raw Serper response JSON (organic results under "organic").

    Raises:
        ValueError: when no Serper API key is configured.
        requests.exceptions.HTTPError: on non-2xx responses.
    """
    serper_api_key = loaded_config_data["search_engines"].get("serper_search_api_key", "")
    if not serper_api_key:
        raise ValueError("Please provide a valid Serper API key ([search_engines] serper_search_api_key)")
    headers = {"X-API-KEY": serper_api_key, "Content-Type": "application/json"}
    payload = {
        "q": search_query,
        "gl": (content_country or "us").lower(),
        "hl": search_lang or "en",
        "num": int(result_count) if result_count else 10,
    }
    response = requests.post("https://google.serper.dev/search", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def parse_serper_results(serper_search_results, web_search_results_dict):
    """Parse Serper organic results into the standardized shape.

    answerBox/knowledgeGraph blocks are deliberately ignored — organic web
    results only, like every sibling parser (spec 2026-08-06 §2). `position`
    is stored as-is under metadata.position; relevance_score stays None
    (mapping rank into a "relevance" field would invert its meaning).
    """
    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []
    for result in (serper_search_results or {}).get("organic", []):
        web_search_results_dict["results"].append({
            "title": result.get("title", ""),
            "url": result.get("link", ""),
            "content": result.get("snippet", ""),
            "metadata": {
                "date_published": result.get("date", None),
                "author": None,
                "source": None,
                "language": None,
                "relevance_score": None,
                "position": result.get("position", None),
                "snippet": result.get("snippet", None),
            },
        })
```

Delete the dead `def test_search_serper(): pass` stub beside them. Fix the dispatch branch (~line 1199):

```python
        elif search_engine.lower() == "serper":
            web_search_results = search_web_serper(
                search_query, content_country, search_lang, result_count
            )
```

In `config.py`: add to the loader dict beside `tavily_search_api_key` (~line 2182):

```python
            "serper_search_api_key": _get_typed_value(
                search_engines_section, "serper_search_api_key", ""
            ),
```

and to the TOML template `[search_engines]` block (~line 3695):

```toml
# Serper (google.serper.dev) API key
serper_search_api_key = ""
```

- [ ] **Step 4: Run to verify GREEN**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/test_search_backends.py -v -p no:randomly`
Expected: all serper tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Web_Scraping/WebSearch_APIs.py tldw_chatbook/config.py Tests/Web_Scraping/test_search_backends.py
git commit -m "feat(search): implement Serper backend + parser, wire dispatch and config (task-1355)"
```

---

### Task 3: Exa backend

**Files:**
- Modify: `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` (new section before the Tavily section ~line 2844; `perform_websearch` dispatch — add an `exa` branch after `kagi` ~line 1196; `process_web_search_results` — add an `exa` branch after `duckduckgo` ~line 1536)
- Modify: `tldw_chatbook/config.py` (loader + template, same blocks as Task 2)
- Test: `Tests/Web_Scraping/test_search_backends.py` (append)

**Interfaces:**
- Consumes: Task 2's test scaffolding (`_FakeRequests`, `_patch_requests`, `_set_key`).
- Produces: `search_web_exa(search_query, result_count) -> dict`; `parse_exa_results(raw_results, output_dict) -> None`; config key `exa_search_api_key`.

- [ ] **Step 1: Write the failing tests** (append)

```python
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
```

- [ ] **Step 2: RED**

Run: `… -k exa`
Expected: FAIL — `AttributeError: module … has no attribute 'search_web_exa'`.

- [ ] **Step 3: Implement**

New section (before Tavily's, with the sibling-style banner comment):

```python
######################### Exa Search #########################
#
# https://exa.ai/docs/reference/search
def search_web_exa(search_query, result_count=None):
    """Query the Exa search API and return its raw JSON.

    Requests `contents.highlights` — billed as contents retrieval on top of
    the search call; a deliberate paid trade for snippet text (spec
    2026-08-06 §2), since a result without a snippet is nearly useless to
    the model.

    Args:
        search_query: The query string.
        result_count: numResults (default 10).

    Returns:
        dict: Raw Exa response JSON (results under "results").

    Raises:
        ValueError: when no Exa API key is configured.
        requests.exceptions.HTTPError: on non-2xx responses.
    """
    exa_api_key = loaded_config_data["search_engines"].get("exa_search_api_key", "")
    if not exa_api_key:
        raise ValueError("Please provide a valid Exa API key ([search_engines] exa_search_api_key)")
    headers = {"x-api-key": exa_api_key, "Content-Type": "application/json"}
    payload = {
        "query": search_query,
        "numResults": int(result_count) if result_count else 10,
        "type": "auto",
        "contents": {"highlights": True},
    }
    response = requests.post("https://api.exa.ai/search", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def parse_exa_results(exa_search_results, web_search_results_dict):
    """Parse Exa results into the standardized shape (first highlight = snippet)."""
    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []
    for result in (exa_search_results or {}).get("results", []):
        highlights = result.get("highlights") or []
        snippet = highlights[0] if highlights else ""
        web_search_results_dict["results"].append({
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": snippet,
            "metadata": {
                "date_published": result.get("publishedDate", None),
                "author": result.get("author", None),
                "source": None,
                "language": None,
                "relevance_score": None,
                "snippet": snippet or None,
            },
        })
```

Dispatch branches: in `perform_websearch` after the `kagi` branch:

```python
        elif search_engine.lower() == "exa":
            web_search_results = search_web_exa(search_query, result_count)
```

and in `process_web_search_results` after the `duckduckgo` branch:

```python
        elif search_engine.lower() == "exa":
            parse_exa_results(search_results, web_search_results_dict)
```

Config loader + template gain `exa_search_api_key` (same shapes as Task 2, comment `# Exa (exa.ai) API key`).

- [ ] **Step 4: GREEN** — full file run, all serper+exa tests pass.

- [ ] **Step 5: Commit** — `feat(search): add Exa backend + parser, wire dispatch and config (task-1355)`

---

### Task 4: Yandex backend

**Files:**
- Modify: `tldw_chatbook/Web_Scraping/WebSearch_APIs.py` (stubs `search_web_yandex`/`parse_yandex_results` ~lines 2900/2908; dispatch ~line 1217; module imports — add the defusedxml-with-fallback block and `import base64` beside the existing imports)
- Modify: `tldw_chatbook/config.py` (loader + template: `yandex_search_folder_id`)
- Test: `Tests/Web_Scraping/test_search_backends.py` (append)

**Interfaces:**
- Consumes: Task 2's scaffolding.
- Produces: `search_web_yandex(search_query, result_count) -> dict` (standardized-items dict, see below); `parse_yandex_results(yandex_search_results, web_search_results_dict) -> None`; config key `yandex_search_folder_id`.
- Design note the implementer must honor: the raw HTTP response is JSON-wrapping-base64-XML. `search_web_yandex` returns the RAW response JSON (house contract: backends return raw API responses); ALL decoding/parsing lives in `parse_yandex_results`, so `process_web_search_results`'s try/except is the single error seam.

- [ ] **Step 1: Write the failing tests** (append)

```python
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
```

- [ ] **Step 2: RED** — `… -k yandex`: stub signature/`pass` failures as in Task 2.

- [ ] **Step 3: Implement**

Module imports (top of file, beside existing stdlib imports): `import base64`, plus the repo's guarded XML import (mirror `Subscriptions/security.py`):

```python
try:
    import defusedxml.ElementTree as _yandex_ET
except ImportError:
    import xml.etree.ElementTree as _yandex_ET

    logger.warning(
        "defusedxml not available, using standard xml.etree for Yandex result parsing. "
        "Install defusedxml for better security."
    )
```

(The file already imports `xml.etree.ElementTree as xET` for sitemaps elsewhere? It does NOT — `guarded_fetch_requests` XML use lives in Article_Extractor_Lib. The alias `_yandex_ET` avoids colliding with anything; check imports before adding.)

Replace the yandex stubs:

```python
def search_web_yandex(search_query, result_count=None):
    """Query Yandex Cloud Search API v2 (synchronous REST) and return raw JSON.

    The response wraps a base64-encoded XML document in "rawData"
    (proto: yandex/cloud/searchapi/v2/search_service.proto — WebSearchService.Search,
    POST /v2/web/search). Decoding and parsing live in parse_yandex_results so
    process_web_search_results' try/except is the single error seam.

    Args:
        search_query: The query string.
        result_count: Unused by the request (Yandex returns its default page,
            ~10 groups; the agent layer trims client-side) — accepted for
            dispatch-signature parity.

    Returns:
        dict: Raw response JSON ({"rawData": "<base64 XML>"}).

    Raises:
        ValueError: when the API key or folder id is not configured.
        requests.exceptions.HTTPError: on non-2xx responses.
    """
    yandex_api_key = loaded_config_data["search_engines"].get("yandex_search_api_key", "")
    if not yandex_api_key:
        raise ValueError("Please provide a valid Yandex Search API key ([search_engines] yandex_search_api_key)")
    folder_id = loaded_config_data["search_engines"].get("yandex_search_folder_id", "")
    if not folder_id:
        raise ValueError("Please provide the Yandex Cloud folder id ([search_engines] yandex_search_folder_id)")
    headers = {"Authorization": f"Api-Key {yandex_api_key}", "Content-Type": "application/json"}
    payload = {
        "query": {"searchType": "SEARCH_TYPE_COM", "queryText": search_query},
        "folderId": folder_id,
        "responseFormat": "FORMAT_XML",
    }
    response = requests.post(
        "https://searchapi.api.cloud.yandex.net/v2/web/search", headers=headers, json=payload
    )
    response.raise_for_status()
    return response.json()


def parse_yandex_results(yandex_search_results, web_search_results_dict):
    """Decode rawData base64 XML and parse docs into the standardized shape.

    Raises on an in-XML <error> element (quota/auth/malformed-query arrive
    inside HTTP 200): a quota error must never render as "No results found"
    for a query that was never searched (spec 2026-08-06 §2). The raise is
    caught by process_web_search_results and lands in processing_error.
    """
    if "results" not in web_search_results_dict:
        web_search_results_dict["results"] = []
    raw_b64 = (yandex_search_results or {}).get("rawData", "")
    if not raw_b64:
        raise ValueError("Yandex response had no rawData field")
    xml_bytes = base64.b64decode(raw_b64)
    root = _yandex_ET.fromstring(xml_bytes)
    error_el = root.find(".//error")
    if error_el is not None:
        code = error_el.get("code", "?")
        text = "".join(error_el.itertext()).strip()
        raise ValueError(f"Yandex API error (code {code}): {text}")
    for doc in root.findall(".//group/doc"):
        url_el = doc.find("url")
        title_el = doc.find("title")
        passages = [
            " ".join("".join(p.itertext()).split())
            for p in doc.findall(".//passage")
        ]
        content = " ".join(passages).strip()
        web_search_results_dict["results"].append({
            "title": "".join(title_el.itertext()).strip() if title_el is not None else "",
            "url": url_el.text.strip() if url_el is not None and url_el.text else "",
            "content": content,
            "metadata": {
                "date_published": None,
                "author": None,
                "source": None,
                "language": None,
                "relevance_score": None,
                "snippet": content or None,
            },
        })
```

Delete the dead `def test_search_yandex(): pass` stub. Fix dispatch (~line 1217):

```python
        elif search_engine.lower() == "yandex":
            web_search_results = search_web_yandex(search_query, result_count)
```

Config loader + template gain `yandex_search_folder_id` (comment `# Yandex Cloud folder id for Search API v2`).

- [ ] **Step 4: GREEN** — full file, all tests pass.
- [ ] **Step 5: Commit** — `feat(search): implement Yandex Cloud v2 backend + XML parser with honest error surfacing (task-1355)`

---

### Task 5: The engine-surface sweep

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py` (`SEARCH_ENGINES` ~line 515)
- Modify: `tldw_chatbook/Utils/Utils.py` (`global_search_engines` ~line 144)
- Modify: `tldw_chatbook/Research_Interop/local_research_search_service.py` (`LOCAL_SUPPORTED_WEBSEARCH_ENGINES` ~line 20) — ONLY after the routing check below passes
- Test: `Tests/Agents/test_local_tool_provider.py` (extend `test_web_search_spec_schema`), `Tests/Web_Scraping/test_search_backends.py` (enum-parity test)

**Interfaces:** consumes Tasks 2–4 (all three engines must be dispatchable before enums advertise them).

- [ ] **Step 1: Investigate and record (before any edit).** For each surface, answer "does it feed `perform_websearch`?" with a grep-backed one-liner in your report:
  - `Utils.py global_search_engines` — find its consumers (`grep -rn "global_search_engines" tldw_chatbook/`); update only if a consumer routes to `perform_websearch` or presents engine choices the dispatcher must honor.
  - `Research_Interop/local_research_search_service.py LOCAL_SUPPORTED_WEBSEARCH_ENGINES` — confirm it routes to `perform_websearch` (it already lists `serper`; if it routes, add `exa` — serper/yandex are present).
  - `tldw_api/research_search_schemas.py SUPPORTED_WEBSEARCH_ENGINES` — this is the remote server API contract (it already lists `exa` and `firecrawl`, engines the local dispatcher doesn't have): expected verdict is RECORDED SKIP (a server contract, not local dispatch) — verify and record.
  - `Tools/web_search_tool.py` — confirm liveness: `grep -rn "web_search_tool" tldw_chatbook/ --include=*.py | grep -v "Tools/web_search_tool.py"`. If nothing registers/imports it, RECORDED SKIP (retired file); if something does, add the three engines to its enum.

- [ ] **Step 2: Failing tests.** Extend `test_web_search_spec_schema` in `Tests/Agents/test_local_tool_provider.py`:

```python
    for engine in ("exa", "serper", "yandex"):
        assert engine in props["search_engine"]["enum"]
```

Add to `Tests/Web_Scraping/test_search_backends.py`:

```python
def test_agent_enum_engines_all_dispatchable():
    """Every engine the agent tool advertises must reach a real backend."""
    from tldw_chatbook.Tools.web_tool_impls import SEARCH_ENGINES
    for engine in SEARCH_ENGINES:
        result = WebSearch_APIs.process_web_search_results({}, engine)
        # a real engine parses an empty payload to an empty result list;
        # an unknown engine sets processing_error ("Invalid Search Engine Name")
        assert result["processing_error"] is None or "Invalid" not in str(result["processing_error"]), (
            f"agent enum advertises {engine!r} but process_web_search_results rejects it"
        )
```

Run both — RED (enum lacks the three engines; parity test fails on "exa" until the enum change... note: parity test passes for engines already wired; RED comes from the spec-schema test until Step 3).

- [ ] **Step 3: Implement.** `SEARCH_ENGINES = ("google", "bing", "duckduckgo", "brave", "kagi", "tavily", "searx", "exa", "serper", "yandex")`. Apply the Step-1 verdicts: `Utils.py` list gains `"exa", "serper"` (yandex present) if its consumer check said update; research-interop local set gains `"exa"` if it routes; record skips otherwise. NOTE: `bing`/`baidu`/`searx` in other lists are NOT this task's concern — only the three new engines.

- [ ] **Step 4: GREEN** — `Tests/Agents/test_local_tool_provider.py` + `Tests/Web_Scraping/test_search_backends.py` both fully green. CAUTION: if the parity test exposes that `bing` (in the agent enum today) hits a dead parser, do NOT expand scope — report it as a finding for a follow-up task.

- [ ] **Step 5: Commit** — `feat(search): advertise exa/serper/yandex on every live engine surface (task-1355)`

---

### Task 6: Live tests + registration of the `live` marker

**Files:**
- Modify: `pyproject.toml` (markers list ~line 504)
- Test: `Tests/Web_Scraping/test_search_backends.py` (append)

**Interfaces:** consumes Tasks 2–4. The live run itself happens at programme verification, not inside this task's gate.

- [ ] **Step 1: Register the marker** (`--strict-markers` makes an unregistered marker a collection error):

```toml
    "live: marks tests that call real paid external APIs (require key files + TLDW_LIVE_SEARCH_TESTS=1)",
```

- [ ] **Step 2: Append the live tests:**

```python
# ---------------------------------------------------------------------------
# Live smoke (double-gated: key file AND TLDW_LIVE_SEARCH_TESTS=1 — spec §5)
# ---------------------------------------------------------------------------

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIVE_ENABLED = os.environ.get("TLDW_LIVE_SEARCH_TESTS") == "1"


def _key_file(name):
    p = _REPO_ROOT / name
    return p.read_text().strip() if p.exists() else ""


def _live_gate(*key_names):
    missing = [n for n in key_names if not _key_file(n)]
    if not _LIVE_ENABLED:
        return pytest.mark.skip(reason="TLDW_LIVE_SEARCH_TESTS != 1")
    if missing:
        return pytest.mark.skip(reason=f"missing key file(s): {', '.join(missing)}")
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
```

(Note `_live_gate` returns a skip marker OR `pytest.mark.live` — when live, the test carries the registered marker; when gated off, it skips with the precise reason. NEVER echo key contents anywhere.)

- [ ] **Step 3: Verify gating** — plain run of the file shows the three live tests SKIPPED with the env-flag reason; `TLDW_LIVE_SEARCH_TESTS=1` run WITHOUT key files shows the missing-file reason. Both outputs quoted in the report.
- [ ] **Step 4: Commit** — `test(search): double-gated live smoke tests + live marker registration (task-1355)`

---

## Final verification (whole branch)

- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/ Tests/Tools/ Tests/Agents/test_local_tool_provider.py -q -p no:randomly` — green, count read.
- Collection sweep: `… -m pytest Tests/ --collect-only -q -p no:randomly | tail -3` — no new errors.
- **Live run (controller, once, foreground):** for each key file the owner has provided: `TLDW_LIVE_SEARCH_TESTS=1 … -m pytest Tests/Web_Scraping/test_search_backends.py -v -p no:randomly -k live` — results reported honestly per engine (a 4xx from a bad key is a finding, not a pass).
- Backlog: task-1355 In Progress → Done with Implementation Notes (incl. the sweep's recorded skips).
