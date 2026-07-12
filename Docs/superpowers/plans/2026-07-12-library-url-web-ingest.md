# URL / web source local ingest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-12-library-url-web-ingest-design.md`. Branch `claude/followups-url-ingest` off dev `00d08308`. Anchors exact at branch point; grep symbols, line numbers drift.

**Goal:** Let a user ingest a URL into the Library — web pages via a new sync `httpx`+`trafilatura` article extractor, audio/video URLs via the existing (URL-accepting) transcription processors — all through the existing parse-pool → persist pipeline.

**Architecture:** A unified `classify_ingest_source` sets `detected_type` at submit (so the heavy-lane cap applies to media URLs for free). `parse_local_file_for_ingest` detects a URL, skips the file-path machinery, and routes `article` → the new extractor, `audio`/`video` → the existing processors (URL as input). Both produce the shared `result`/payload dict; the payload `url` is the canonical post-redirect URL.

**Tech Stack:** Python ≥3.11, `httpx` (core dep), `trafilatura` (optional extra, installed), multiprocessing spawn pool, pytest.

## Global Constraints

- **Spawn-safe worker module:** `ingest_parse_worker.py` module scope stays stdlib-only; ALL heavy imports (`httpx`, `trafilatura`, `local_file_ingestion`) are deferred inside functions. The new `web_article_ingestion.py` may only be imported from inside a function on the worker path.
- **No behavior change for file paths:** a local file path classifies + parses exactly as today; every existing ingest test stays green.
- **Classifier:** URL (`urlparse(s).scheme in ("http","https")`) → video (known host youtube.com/youtu.be/vimeo.com/dailymotion.com OR video-ext path) / audio (audio-ext path) / else `"article"`; non-URL → `detect_file_type`.
- **URL-aware tail:** the payload `url` is the canonical post-redirect URL for a URL source (NOT `file://…`, never `.absolute()` on a URL).
- **Article extractor:** `httpx.Client` GET `timeout=30.0`, `follow_redirects=True`, browser UA, **≤10 MB** streamed body guard, non-HTML content-type → permanent, `trafilatura` extract + `_strip_boilerplate` + free metadata (`sitename`/`hostname`/`language`), canonical url from `resp.url`, empty extraction → permanent, missing `trafilatura` → permanent. No Playwright.
- **Retryable/permanent taxonomy:** permanent (`PermanentIngestError`) = invalid URL, other-4xx, DNS-resolution failure, non-HTML, empty extraction, missing `trafilatura`; retryable (any other exception) = network/timeout, 5xx, 408/429. `classify_parse_failure` recognizes `PermanentIngestError` via a DEFERRED import.
- **Security:** the UI gates URLs through `validate_url` (rejects non-`https?://` schemes).
- **Staging:** explicit paths only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> \
    -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
  ```

---

### Task 1: Pure helpers — `classify_ingest_source`, `canonicalize_url`, `PermanentIngestError`, `classify_parse_failure` seam

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
- Modify: `tldw_chatbook/Local_Ingestion/ingest_parse_worker.py` (`classify_parse_failure`)
- Test: `Tests/Local_Ingestion/test_url_ingest_helpers.py` (create)

**Interfaces:**
- Produces: `classify_ingest_source(source: str) -> str`; `canonicalize_url(url: str) -> str`; `class PermanentIngestError(FileIngestionError)`; `classify_parse_failure` returns `True` for `PermanentIngestError`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Local_Ingestion/test_url_ingest_helpers.py`:
```python
import pytest

from tldw_chatbook.Local_Ingestion.local_file_ingestion import (
    classify_ingest_source, canonicalize_url, PermanentIngestError,
)
from tldw_chatbook.Local_Ingestion.ingest_parse_worker import classify_parse_failure


@pytest.mark.parametrize("url,expected", [
    ("https://youtube.com/watch?v=abc", "video"),
    ("https://youtu.be/abc", "video"),
    ("https://vimeo.com/123", "video"),
    ("https://cdn.example.com/clip.mp4", "video"),
    ("https://cdn.example.com/talk.mp3", "audio"),
    ("https://example.com/blog/post", "article"),
    ("http://example.com/a", "article"),
])
def test_classify_urls(url, expected):
    assert classify_ingest_source(url) == expected


def test_classify_files_use_detect_file_type():
    assert classify_ingest_source("/tmp/a.pdf") == "pdf"
    assert classify_ingest_source("/tmp/a.mp3") == "audio"


def test_classify_non_http_scheme_is_not_url():
    # a file:// or bare path must NOT be treated as an http URL
    with pytest.raises(Exception):
        classify_ingest_source("file:///etc/passwd")   # no known extension -> detect_file_type raises


def test_canonicalize_url_strips_tracking_and_normalizes():
    got = canonicalize_url("HTTPS://Example.com:443/Path/?utm_source=x&b=2&a=1#frag")
    assert got == "https://example.com/Path?a=1&b=2"
    assert canonicalize_url("https://example.com/") == "https://example.com/"
    assert canonicalize_url("https://example.com/p/") == "https://example.com/p"


def test_classify_parse_failure_permanent_ingest_error():
    assert classify_parse_failure(PermanentIngestError("bad url")) is True
    assert classify_parse_failure(RuntimeError("transient")) is False
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Local_Ingestion/test_url_ingest_helpers.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `cannot import name 'classify_ingest_source'` / `PermanentIngestError`.

- [ ] **Step 3: Add helpers to `local_file_ingestion.py`**

Near `FileIngestionError` (`local_file_ingestion.py:30`), add:
```python
class PermanentIngestError(FileIngestionError):
    """A parse/fetch failure that will fail identically on retry (bad URL,
    4xx, non-HTML content, empty extraction, missing extractor dependency).
    ``classify_parse_failure`` maps this to a permanent (non-retryable) job.
    """


_VIDEO_URL_HOSTS = ("youtube.com", "youtu.be", "vimeo.com", "dailymotion.com")
_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg")
_AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".wma", ".opus")
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "igshid", "mc_cid", "mc_eid", "ref", "ref_src",
})


def _is_http_url(source: str) -> bool:
    from urllib.parse import urlparse
    try:
        return urlparse(source).scheme in ("http", "https")
    except Exception:
        return False


def classify_ingest_source(source: str) -> str:
    """Classify an ingest source into a media type.

    For an http/https URL: a known video host or a video-extension path ->
    ``"video"``; an audio-extension path -> ``"audio"``; otherwise
    ``"article"``. For any non-URL source, delegate to ``detect_file_type``
    (extension-based; raises ``FileIngestionError`` for unknown types).
    """
    from urllib.parse import urlparse
    source = str(source)
    if _is_http_url(source):
        parsed = urlparse(source)
        host = (parsed.hostname or "").lower()
        path = parsed.path.lower()
        if any(host == h or host.endswith("." + h) for h in _VIDEO_URL_HOSTS) or path.endswith(_VIDEO_EXTS):
            return "video"
        if path.endswith(_AUDIO_EXTS):
            return "audio"
        return "article"
    return detect_file_type(source)


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL for a clean stored value: lowercase scheme/host,
    drop default port + fragment, strip trailing slash (except root), remove
    tracking params, and sort remaining query params for stability.
    """
    from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    netloc = host
    if parsed.port and not ((scheme == "https" and parsed.port == 443) or (scheme == "http" and parsed.port == 80)):
        netloc = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = urlencode(sorted(
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
    ))
    return urlunparse((scheme, netloc, path, "", query, ""))
```

- [ ] **Step 4: Extend `classify_parse_failure` (deferred import — worker module stays stdlib-only)**

In `ingest_parse_worker.py`, prepend to `classify_parse_failure`'s body (before the `FileNotFoundError` check):
```python
    try:
        from .local_file_ingestion import PermanentIngestError
        if isinstance(exc, PermanentIngestError):
            return True
    except Exception:
        pass
```
(Deferred import: `classify_parse_failure` only runs AFTER `parse_local_file_for_ingest` has already imported `local_file_ingestion`, so this is cheap and keeps the module's spawn-safe stdlib-only scope. `PermanentIngestError` subclasses `FileIngestionError`, so its "Unsupported file type"-style messages also still classify correctly if the import ever fails.)

- [ ] **Step 5: Run to verify it passes + the existing ingest suites**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Local_Ingestion/test_url_ingest_helpers.py Tests/Local_Ingestion/test_ingest_parse_worker.py \
  -q -p no:cacheprovider -o addopts="" --timeout=180
```
Expected: PASS (new + existing worker tests).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Local_Ingestion/local_file_ingestion.py tldw_chatbook/Local_Ingestion/ingest_parse_worker.py Tests/Local_Ingestion/test_url_ingest_helpers.py
git commit -m "feat(ingest): url source classifier + canonicalize_url + PermanentIngestError seam (162)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Article extractor — `web_article_ingestion.py`

**Files:**
- Create: `tldw_chatbook/Local_Ingestion/web_article_ingestion.py`
- Test: `Tests/Local_Ingestion/test_web_article_ingestion.py` (create)

**Interfaces:**
- Consumes: `PermanentIngestError`, `canonicalize_url` (Task 1).
- Produces: `extract_article_for_ingest(url: str, options: dict) -> dict` returning `{content, title, author, keywords, chunks, analysis, metadata, url}` (a `result` dict for the shared tail; `url` = canonical post-redirect URL).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Local_Ingestion/test_web_article_ingestion.py`:
```python
from unittest.mock import patch, MagicMock
import pytest

from tldw_chatbook.Local_Ingestion.web_article_ingestion import extract_article_for_ingest
from tldw_chatbook.Local_Ingestion.local_file_ingestion import PermanentIngestError


def _resp(html, *, status=200, ctype="text/html; charset=utf-8", final_url=None):
    r = MagicMock()
    r.status_code = status
    r.headers = {"content-type": ctype, "content-length": str(len(html))}
    r.url = final_url or "https://example.com/post"
    r.text = html
    r.iter_bytes = lambda chunk_size=65536: iter([html.encode("utf-8")])
    r.raise_for_status = MagicMock()
    return r


def _client_returning(resp):
    client = MagicMock()
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    client.get.return_value = resp
    return client


def test_extract_success_shape(monkeypatch):
    html = "<html><head><title>Hello</title></head><body><article><p>" + ("word " * 80) + "</p></article></body></html>"
    with patch("httpx.Client", return_value=_client_returning(_resp(html, final_url="https://example.com/post?utm_source=x"))):
        result = extract_article_for_ingest("https://example.com/post?utm_source=x", {})
    assert "word word" in result["content"]
    assert result["url"] == "https://example.com/post"            # canonical, tracking stripped
    assert result["title"]
    assert result["chunks"] == [] and result["analysis"] == ""


def test_non_html_content_type_is_permanent():
    with patch("httpx.Client", return_value=_client_returning(_resp("%PDF-1.4", ctype="application/pdf"))):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/x.pdf", {})


def test_empty_extraction_is_permanent(monkeypatch):
    html = "<html><body><script>window.__NEXT_DATA__={}</script></body></html>"
    # trafilatura returns None on a JS shell
    with patch("httpx.Client", return_value=_client_returning(_resp(html))), \
         patch("trafilatura.extract", return_value=None):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/spa", {})


def test_permanent_on_4xx():
    resp = _resp("nope", status=404)
    import httpx
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock(status_code=404)))
    with patch("httpx.Client", return_value=_client_returning(resp)):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/missing", {})


def test_retryable_on_503():
    resp = _resp("busy", status=503)
    import httpx
    resp.raise_for_status = MagicMock(side_effect=httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock(status_code=503)))
    with patch("httpx.Client", return_value=_client_returning(resp)):
        with pytest.raises(Exception) as ei:
            extract_article_for_ingest("https://example.com/busy", {})
        assert not isinstance(ei.value, PermanentIngestError)     # retryable


def test_missing_trafilatura_is_permanent(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def _fail(name, *a, **k):
        if name == "trafilatura":
            raise ImportError("no trafilatura")
        return real_import(name, *a, **k)
    html = "<html><body><p>x</p></body></html>"
    with patch("httpx.Client", return_value=_client_returning(_resp(html))), \
         patch.object(builtins, "__import__", _fail):
        with pytest.raises(PermanentIngestError):
            extract_article_for_ingest("https://example.com/a", {})
```

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Local_Ingestion/test_web_article_ingestion.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Local_Ingestion.web_article_ingestion`.

- [ ] **Step 3: Implement the extractor**

Create `tldw_chatbook/Local_Ingestion/web_article_ingestion.py`:
```python
"""Fetch a URL and extract its article text for Library ingest.

Sync + dependency-light (httpx + trafilatura, no browser) so it runs inside
the spawn-based parse pool worker exactly like the file-parsing branches.
Heavy imports (httpx/trafilatura) are deferred inside the function.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from loguru import logger

from .local_file_ingestion import PermanentIngestError, canonicalize_url

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB body guard
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
_BOILERPLATE = re.compile(
    r"^\s*(subscribe now|sign up|share on \w+|follow us|newsletter|read more|"
    r"thanks for reading|advertisement)\b.*$",
    re.IGNORECASE,
)


def _strip_boilerplate(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _BOILERPLATE.match(ln)]
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


def extract_article_for_ingest(url: str, options: Dict[str, Any]) -> Dict[str, Any]:
    import httpx

    try:
        with httpx.Client(follow_redirects=True, timeout=30.0,
                          headers={"User-Agent": _UA, "Accept": "text/html,*/*"}) as client:
            resp = client.get(url)
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _RETRYABLE_STATUS:
                    raise                                    # retryable
                raise PermanentIngestError(f"URL fetch failed ({status}) for {url}") from exc
            ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if ctype and "html" not in ctype and "xml" not in ctype:
                raise PermanentIngestError(f"URL is not a web page (content-type {ctype!r}): {url}")
            body = resp.text
            if len(body.encode("utf-8", errors="ignore")) > _MAX_BYTES:
                raise PermanentIngestError(f"URL response too large (>{_MAX_BYTES} bytes): {url}")
            final_url = str(resp.url)
    except httpx.HTTPError as exc:
        # network/timeout -> retryable; a DNS resolution failure -> permanent.
        import socket
        if isinstance(getattr(exc, "__cause__", None), socket.gaierror):
            raise PermanentIngestError(f"URL host could not be resolved: {url}") from exc
        raise  # retryable transport error

    try:
        import trafilatura
    except ImportError as exc:
        raise PermanentIngestError(
            "Web article ingest requires 'trafilatura' -- install with "
            "pip install tldw_chatbook[websearch]"
        ) from exc

    content = trafilatura.extract(body, include_comments=False, include_tables=False)
    content = _strip_boilerplate(content or "")
    if not content:
        raise PermanentIngestError(
            f"Couldn't extract article content -- the page may require JavaScript (not supported): {url}"
        )

    meta = trafilatura.extract_metadata(body)
    md = {}
    if meta is not None:
        md = {k: getattr(meta, k, None) for k in ("sitename", "hostname", "language", "date")}
    md["ingestion_method"] = "web_article"

    return {
        "content": content,
        "title": (getattr(meta, "title", None) or options.get("title") or url),
        "author": (getattr(meta, "author", None) or options.get("author") or "Unknown"),
        "keywords": [],
        "chunks": [],
        "analysis": "",
        "metadata": md,
        "url": canonicalize_url(final_url),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run the Step-1 tests. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Local_Ingestion/web_article_ingestion.py Tests/Local_Ingestion/test_web_article_ingestion.py
git commit -m "feat(ingest): sync httpx+trafilatura article extractor for URL ingest (162)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: URL-aware `parse_local_file_for_ingest` routing

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` (`parse_local_file_for_ingest`)
- Test: `Tests/Local_Ingestion/test_parse_url_routing.py` (create)

**Interfaces:**
- Consumes: `classify_ingest_source`, `extract_article_for_ingest`, `PermanentIngestError` (Tasks 1-2).

- [ ] **Step 1: Write the failing test**

Create `Tests/Local_Ingestion/test_parse_url_routing.py`:
```python
from unittest.mock import patch
import pytest

from tldw_chatbook.Local_Ingestion.local_file_ingestion import parse_local_file_for_ingest


def test_article_url_routes_to_extractor_and_sets_real_url():
    fake = {"content": "Body text here", "title": "T", "author": "A", "keywords": [],
            "chunks": [], "analysis": "", "metadata": {}, "url": "https://example.com/post"}
    with patch("tldw_chatbook.Local_Ingestion.web_article_ingestion.extract_article_for_ingest",
               return_value=fake) as ex:
        payload = parse_local_file_for_ingest("https://example.com/post?utm_source=x", {})
    ex.assert_called_once()
    assert payload["media_type"] == "article"
    assert payload["content"] == "Body text here"
    assert payload["url"] == "https://example.com/post"          # NOT file://, NOT .absolute()
    assert not payload["url"].startswith("file://")


def test_video_url_routes_to_audio_video_branch():
    # the audio/video branch calls the processor with the URL as input;
    # mock the processor to return a transcript result.
    fake_result = {"content": "transcript", "title": "Vid", "author": "U", "chunks": [], "analysis": ""}
    with patch("tldw_chatbook.Local_Ingestion.local_file_ingestion.LocalVideoProcessor") as VP:
        VP.return_value.process_videos.return_value = [{**fake_result, "status": "Success"}]
        payload = parse_local_file_for_ingest("https://youtube.com/watch?v=abc", {})
    assert payload["media_type"] == "video"
    assert payload["url"] == "https://youtube.com/watch?v=abc"    # the URL, not file://
    args, kwargs = VP.return_value.process_videos.call_args
    assert kwargs.get("inputs") == ["https://youtube.com/watch?v=abc"]


def test_article_permanent_error_propagates_unwrapped():
    # CRITICAL: a PermanentIngestError from the extractor must NOT be re-wrapped
    # into a plain FileIngestionError by the outer `except Exception` -- else
    # classify_parse_failure sees a bare FileIngestionError and marks it retryable.
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import PermanentIngestError
    from tldw_chatbook.Local_Ingestion.ingest_parse_worker import classify_parse_failure
    with patch("tldw_chatbook.Local_Ingestion.web_article_ingestion.extract_article_for_ingest",
               side_effect=PermanentIngestError("page requires JavaScript")):
        with pytest.raises(PermanentIngestError) as ei:
            parse_local_file_for_ingest("https://example.com/spa", {})
    assert classify_parse_failure(ei.value) is True     # stays permanent through the wrapper
```
(If the video branch's result-shape/`process_videos` kwargs differ, match the real branch — grep the `elif file_type == 'video'` block; the assertions on `media_type`/`url`/`inputs` are the point.)

- [ ] **Step 2: Run to verify it fails**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Local_Ingestion/test_parse_url_routing.py -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — a URL currently raises `FileNotFoundError` (the `.exists()` gate) before any routing.

- [ ] **Step 3: Add URL detection + article branch + `source_url` in the tail**

In `parse_local_file_for_ingest` (`local_file_ingestion.py:186`), replace the top:
```python
    file_path = Path(file_path)
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    # Detect file type
    try:
        file_type = detect_file_type(file_path)
    except FileIngestionError as e:
        logger.error(f"Unsupported file type: {file_path} - {e}")
        raise
```
with:
```python
    raw_source = str(file_path)
    is_url = _is_http_url(raw_source)
    if is_url:
        # URL source: skip the file-path machinery entirely.
        file_type = classify_ingest_source(raw_source)   # "article" | "audio" | "video"
        source_url = raw_source                            # article branch overrides w/ canonical
        # keep file_path as the raw URL string so the audio/video branches'
        # `str(file_path)` passes the URL straight to the URL-accepting processor.
    else:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            file_type = detect_file_type(file_path)
        except FileIngestionError as e:
            logger.error(f"Unsupported file type: {file_path} - {e}")
            raise
        source_url = f"file://{file_path.absolute()}"
```
Make the title default URL-safe. At the title default (`local_file_ingestion.py:211-212`), `file_path.stem` runs unconditionally BEFORE the branch dispatch and would raise `AttributeError` on a URL `str`. Replace:
```python
    if title is None:
        title = file_path.stem
```
with:
```python
    if title is None:
        title = raw_source if is_url else file_path.stem
```
(A URL's extracted title comes from trafilatura / the processor via `result.get('title', title)`; this default is only a fallback.)

Add the article branch alongside the other `elif file_type == ...:` branches (mirror the plaintext branch's `result` shape):
```python
        elif file_type == 'article':
            from .web_article_ingestion import extract_article_for_ingest
            result = extract_article_for_ingest(raw_source, options)
            source_url = result.get('url', source_url)     # canonical post-redirect URL
```
**CRITICAL — preserve the permanent classification through the outer wrapper.** The function's outer `except Exception` (`local_file_ingestion.py:512-514`) re-raises EVERY exception as a bare `FileIngestionError(f"Failed to ingest {file_type} file: …")`. A `PermanentIngestError` (a `FileIngestionError` subclass) caught here would be re-wrapped into a plain `FileIngestionError` whose message does not start with "Unsupported file type", so `classify_parse_failure` would mark it **retryable** — wrong. Insert an unwrapped re-raise immediately BEFORE the `except Exception`:
```python
    except PermanentIngestError:
        # keep the permanent classification intact for classify_parse_failure
        raise
    except Exception as e:
        logger.error(f"Error parsing {file_type} file {file_path}: {e}")
        raise FileIngestionError(f"Failed to ingest {file_type} file: {str(e)}")
```
(Only `PermanentIngestError` is re-raised unwrapped — a retryable exception, e.g. an httpx timeout, is still wrapped into `FileIngestionError`, whose message doesn't match the "Unsupported file type" prefix, so it correctly stays retryable. This does NOT change any file-path behavior: no file branch raises `PermanentIngestError`.)

In the shared tail (`local_file_ingestion.py:497`), change the payload `'url'` and the metadata file_path to use `source_url`/the raw source, and DO NOT call `.absolute()` on a URL:
```python
        media_metadata['file_path'] = raw_source if is_url else str(file_path)
        media_metadata['file_type'] = file_type
        return {
            'media_type': file_type,
            'file_type': file_type,
            'title': extracted_title,
            'author': extracted_author,
            'content': content,
            'keywords': all_keywords,
            'url': source_url,
            'analysis_content': analysis,
            ...  # remaining keys unchanged (chunks/chunk_options/metadata/file_path)
            'file_path': raw_source if is_url else str(file_path),
        }
```
(The `f"file://{file_path.absolute()}"` literal and `str(file_path)` uses in the tail become the `source_url`/`raw_source`-vs-`str(file_path)` conditional above.)

- [ ] **Step 4: Run to verify it passes + the full local-ingestion suite**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Local_Ingestion/ -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: PASS (URL routing + every existing file-path test unchanged).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Local_Ingestion/test_parse_url_routing.py
git commit -m "feat(ingest): route URL sources through article/media branches with canonical url (162)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Submit + UI wiring + backlog Done

**Files:**
- Modify: `tldw_chatbook/app.py` (`submit_library_ingest_job` classifier)
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (`_submit_library_ingest_form` validation branch)
- Modify: `tldw_chatbook/Widgets/Library/library_ingest_canvas.py` (placeholder copy)
- Modify: `backlog/tasks/task-162 - URL-web-source-local-ingest.md`
- Test: `Tests/UI/test_library_url_ingest_submit.py` (create)

**Interfaces:**
- Consumes: `classify_ingest_source` (Task 1); `validate_url` (`Utils/input_validation.py`).

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_library_url_ingest_submit.py` (unit-test the classifier→detected_type path + the URL-vs-file validation decision as pure logic where possible):
```python
from tldw_chatbook.Local_Ingestion.local_file_ingestion import classify_ingest_source
from tldw_chatbook.Utils.input_validation import validate_url


def test_submit_detected_type_for_url_is_article_or_video():
    assert classify_ingest_source("https://example.com/post") == "article"
    assert classify_ingest_source("https://youtube.com/watch?v=z") == "video"


def test_url_validation_accepts_http_rejects_scheme_tricks():
    assert validate_url("https://example.com/post") is True
    assert validate_url("file:///etc/passwd") is False
    assert validate_url("javascript:alert(1)") is False
```
(These lock the seams the submit path depends on; the app-level `submit_library_ingest_job` change is covered by Task 1's classifier swap + the existing ingest runner tests, which must stay green.)

- [ ] **Step 2: Run to verify it fails/passes**

Run the file. `classify_ingest_source`/`validate_url` exist (Tasks 1 + pre-existing), so these should PASS as regression guards; the RED for this task is behavioral (the app/UI still reject URLs until Step 3-4). Confirm they pass, then proceed.

- [ ] **Step 3: Submit path uses the classifier**

In `app.py:1368`, replace:
```python
            detected_type = detect_file_type(source_path) or ""
```
with:
```python
            detected_type = classify_ingest_source(source_path) or ""
```
Ensure `classify_ingest_source` is imported (extend the existing `from tldw_chatbook.Local_Ingestion import detect_file_type, FileIngestionError` line to include `classify_ingest_source`, and confirm it's re-exported from the package `__init__` — if not, import from `.Local_Ingestion.local_file_ingestion`). The surrounding `try/except FileIngestionError → ""` stays (an unclassifiable file still degrades to light).

- [ ] **Step 4: UI validation branch + placeholder copy**

In `library_screen.py`'s `_submit_library_ingest_form` (~:6180), replace the file-only validation with a URL-vs-file branch:
```python
        from urllib.parse import urlparse
        from tldw_chatbook.Utils.input_validation import validate_url
        if urlparse(raw_path).scheme in ("http", "https"):
            if not validate_url(raw_path):
                self._notify_library_ingest_warning("That doesn't look like a valid http(s) URL.")
                return
            submitted_source = raw_path
        else:
            try:
                validated_path = validate_path_simple(Path(raw_path).expanduser(), require_exists=True)
            except ValueError:
                self._notify_library_ingest_warning("Could not find that file.")
                return
            submitted_source = str(validated_path)
        ...
        submit(source_path=submitted_source, ...)          # rest of the call unchanged
```
In `library_ingest_canvas.py` (~:68-135), change the `#library-ingest-path` Input placeholder from "Path to a local file…" to "Path to a local file or a URL…".

- [ ] **Step 5: Run to verify it passes + import smoke + mark backlog Done**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_library_url_ingest_submit.py Tests/Library/test_library_ingest_runner.py Tests/UI/test_library_shell.py \
  -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
HOME=/private/tmp/tldw-chatbook-test-home PYTHONPATH=$(pwd) \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('import ok')"
```
Expected: PASS + `import ok`. Then mark backlog Done:
```bash
perl -0pi -e 's/- \[ \] (#\d)/- [x] $1/g' "backlog/tasks/task-162 - URL-web-source-local-ingest.md"
perl -0pi -e 's/^status: .*/status: Done/m' "backlog/tasks/task-162 - URL-web-source-local-ingest.md"
```
Add a short `## Implementation Notes` (classifier + heavy-lane reuse; sync httpx+trafilatura article extractor; media-URL passthrough; canonical URL; server-lesson borrows).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/Widgets/Library/library_ingest_canvas.py \
  Tests/UI/test_library_url_ingest_submit.py "backlog/tasks/task-162 - URL-web-source-local-ingest.md"
git commit -m "feat(ingest): accept URLs in the Library ingest form (classify + validate); task 162 done (162)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 4)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Local_Ingestion/ Tests/Library/ Tests/UI/test_library_url_ingest_submit.py Tests/UI/test_library_shell.py \
  -q -p no:cacheprovider -o addopts="" --timeout=600 --timeout-method=thread
```
Plus `python -c "import tldw_chatbook.app"`. Then the whole-branch review (opus) and finishing-a-development-branch. Served-TUI visual QA (paste an article URL → it ingests as an article row; paste a YouTube URL → it transcribes) is worthwhile but optional (network-dependent).
