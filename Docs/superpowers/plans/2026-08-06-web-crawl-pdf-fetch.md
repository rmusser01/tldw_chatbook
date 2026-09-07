# Web-tools v2 (`web_crawl` + PDF fetch) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a budgeted same-host `web_crawl` agent tool and ephemeral PDF text-extraction inside `web_fetch` (backlog tasks 1357/1358), per `Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md`.

**Architecture:** Everything lands in `tldw_chatbook/Tools/web_tool_impls.py` beside the v1 `web_fetch`/`web_search` cores, reusing their SSRF guard (`_validate_hop`), rate limiter, transport/test seams, and cache. Registration is one `LocalToolSpec` in `Agents/local_tool_provider.py` — MCP exposure is automatic via the existing generic `_register_local_agent_tools` path (no MCP code).

**Tech Stack:** Python ≥3.11, httpx (sync client), stdlib `html.parser.HTMLParser` + `xml.etree.ElementTree`, pymupdf via local import (optional `pdf` extra), pytest + `httpx.MockTransport`.

## Global Constraints

Copied from the spec; every task's requirements include these.

- **No new dependencies.** Link/title parsing via stdlib `HTMLParser`; sitemap via stdlib `ElementTree`; pymupdf imported *locally* inside the extraction function with try/except (v1's trafilatura pattern).
- **No DB imports** anywhere in `web_tool_impls.py` — both features are ephemeral by construction.
- **Use the module-level `time` import** (`time.monotonic()`, `time.sleep()`) — never `from time import monotonic`. Tests monkeypatch `web_tool_impls.time` with a fake clock; a direct import silently escapes it.
- All fetch paths go through `_validate_hop` (egress guard) on the initial URL **and every redirect hop**; clients are constructed with `follow_redirects=False, transport=_transport, trust_env=False`.
- Errors are `LocalToolError` with bracketed structured reasons: existing `[invalid-url]`, `[ssrf]`, `[rate-limited]`, `[http-<status>]`, `[timeout]`, `[redirect-limit]`, `[empty-content]`, `[fetch-failed]`; new `[too-large]`, `[pdf-error]`, `[missing-dep]`, `[invalid-args]`, `[crawl-failed]`.
- Exact constants (spec §1/§2): `PDF_MAX_BYTES = 20 * 1024 * 1024`, `FETCH_CACHE_MAX_ENTRIES = 256`, `CRAWL_DEFAULT_MAX_PAGES = 20`, `CRAWL_MAX_PAGES_CEILING = 40`, `CRAWL_DEFAULT_MAX_DEPTH = 2`, `CRAWL_MAX_DEPTH_CEILING = 5`, `CRAWL_DEADLINE_SECONDS = 120.0`, `CRAWL_PAGE_TIMEOUT_SECONDS = 10.0`, `CRAWL_EXCERPT_MAX_CHARS = 200`, `CRAWL_RESULT_MAX_BYTES = 24 * 1024`, `CRAWL_BLOCK_MAX_BYTES = 1024`, `SITEMAP_MAX_BYTES = 5 * 1024 * 1024`, `_CRAWL_USER_AGENT = "tldw-chatbook-web-crawl/1.0"`.
- Run tests with the repo venv, foreground only: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <file> -v -p no:randomly`. Never use `git stash` (repo-wide across all worktrees).
- Host scope = exact host, lowercased, leading `www.` folded — applied to *both* scope checks and visited-dedup.
- The crawl **writes** the fetch cache (key `(final_url, FETCH_MAX_BYTES)`) but never **reads** it.

---

### Task 1: Cache key fix + entry cap

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py` (cache declaration ~line 125, cache read/write inside `web_fetch` ~lines 267–272 and 318)
- Test: `Tests/Tools/test_web_tool_impls.py`

**Interfaces:**
- Consumes: existing `_fetch_cache: dict`, `FETCH_CACHE_TTL_SECONDS`, `web_fetch`.
- Produces: cache keyed by `(url: str, max_bytes: int)`; `FETCH_CACHE_MAX_ENTRIES = 256`; `_cache_put(key: tuple[str, int], text: str) -> None` used for every cache write. Later tasks (2, 4) call `_cache_put`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Tools/test_web_tool_impls.py`)

```python
def test_cache_keyed_by_max_bytes(fetch_env):
    """A small-cap fetch must not poison a later full-cap fetch (spec §1)."""
    body = b"y" * 600
    fetch_env.routes["http://example.com/sized"] = _text_page(body)
    small = web_fetch("http://example.com/sized", max_bytes=100)
    assert "truncated" in small
    fetch_env.clock.now += 2.0  # clear the rate-limit interval
    full = web_fetch("http://example.com/sized")
    assert "truncated" not in full
    # Two distinct requests: different caps are different cache entries.
    assert fetch_env.calls.count("http://example.com/sized") == 2


def test_cache_entry_cap_evicts_earliest_expiry(fetch_env):
    from tldw_chatbook.Tools.web_tool_impls import FETCH_CACHE_MAX_ENTRIES

    for i in range(FETCH_CACHE_MAX_ENTRIES):
        url = f"http://example.com/p{i}"
        fetch_env.routes[url] = _text_page(f"page {i}".encode())
        web_fetch(url)
        fetch_env.clock.now += 2.0
    assert len(web_tool_impls._fetch_cache) == FETCH_CACHE_MAX_ENTRIES
    # One more insert evicts the earliest-expiry entry (p0).
    fetch_env.routes["http://example.com/extra"] = _text_page(b"extra")
    web_fetch("http://example.com/extra")
    assert len(web_tool_impls._fetch_cache) == FETCH_CACHE_MAX_ENTRIES
    assert ("http://example.com/p0", FETCH_MAX_BYTES) not in web_tool_impls._fetch_cache
    assert ("http://example.com/extra", FETCH_MAX_BYTES) in web_tool_impls._fetch_cache
```

- [ ] **Step 2: Run to verify both fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_tool_impls.py -v -p no:randomly -k "cache"`
Expected: `test_cache_keyed_by_max_bytes` FAILS (call count is 1 — the old URL-only key returns the 100-byte text) and `test_cache_entry_cap_evicts_earliest_expiry` FAILS with ImportError on `FETCH_CACHE_MAX_ENTRIES`.

- [ ] **Step 3: Implement**

In `web_tool_impls.py`, change the cache declaration and add the bounded writer:

```python
FETCH_CACHE_MAX_ENTRIES = 256

# Keyed by (url, effective max_bytes): a small-cap fetch must not poison a
# later full-cap call. Bounded because web_crawl bulk-loads it (spec §1).
_fetch_cache: dict[tuple[str, int], tuple[float, str]] = {}


def _cache_put(key: tuple[str, int], text: str) -> None:
    if key not in _fetch_cache and len(_fetch_cache) >= FETCH_CACHE_MAX_ENTRIES:
        oldest = min(_fetch_cache, key=lambda k: _fetch_cache[k][0])
        _fetch_cache.pop(oldest)
    _fetch_cache[key] = (time.monotonic() + FETCH_CACHE_TTL_SECONDS, text)
```

In `web_fetch`, replace the cache read `cached = _fetch_cache.get(url)` with `cached = _fetch_cache.get((url, max_bytes))` (and the stale-pop with `_fetch_cache.pop((url, max_bytes), None)`), and replace the final write `_fetch_cache[url] = (...)` with `_cache_put((url, max_bytes), text)`. Note the cache read happens *after* the `max_bytes` clamp — keep it that way so the key always holds the effective cap.

- [ ] **Step 4: Run the full file to verify green**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_tool_impls.py -v -p no:randomly`
Expected: all PASS (the existing `test_fetch_caches_within_ttl` exercises same-URL-same-cap and must still pass).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/web_tool_impls.py Tests/Tools/test_web_tool_impls.py
git commit -m "fix: fetch cache keyed by (url, max_bytes) + 256-entry bound"
```

---

### Task 2: PDF fetch inside `web_fetch`

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py` (`_fetch_once` ~line 169, `web_fetch` body ~lines 245–320, `_extract_text` untouched)
- Test: `Tests/Tools/test_web_tool_impls.py`

**Interfaces:**
- Consumes: Task 1's `_cache_put`.
- Produces:
  - `_fetch_once(client, url, max_bytes, *, pdf_max_bytes: int | None = None, html_only: bool = False) -> tuple[int, httpx.Headers, bytes, bool, bool]` — returns `(status, headers, body, truncated, is_pdf)`. `pdf_max_bytes` raises the read ceiling mid-stream when the response is a PDF; `html_only=True` (consumed by Task 4) stops reading the body after the first buffered chunk when the declared main type is non-empty and not HTML.
  - `PDF_MAX_BYTES = 20 * 1024 * 1024`.
  - `_extract_pdf_text(body: bytes, max_bytes: int) -> str`.
  - Error copy (exact): `[too-large] PDF exceeds 20 MB — use media ingestion for large documents`; `[missing-dep] PDF support requires pymupdf — pip install tldw_chatbook[pdf]`; `[pdf-error] PDF is encrypted`; `[pdf-error] could not parse PDF: <reason>`; `[empty-content] PDF contains no extractable text (scanned document?) — use media ingestion with OCR`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Tools/test_web_tool_impls.py`)

```python
# ---------------------------------------------------------------------------
# PDF fetch (spec 2026-08-06 §1)
# ---------------------------------------------------------------------------

pymupdf = pytest.importorskip("pymupdf")


def _make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_response(body: bytes, content_type: str = "application/pdf") -> httpx.Response:
    headers = {"content-type": content_type} if content_type else {}
    return httpx.Response(200, content=body, headers=headers)


def test_fetch_extracts_pdf_text(fetch_env):
    body = _make_pdf(["alpha page one text", "beta page two text"])
    fetch_env.routes["http://example.com/doc.pdf"] = _pdf_response(body)
    result = web_fetch("http://example.com/doc.pdf")
    assert "alpha page one text" in result
    assert "beta page two text" in result


def test_fetch_pdf_sniff_beats_mislabeled_content_type(fetch_env):
    body = _make_pdf(["sniffed content"])
    for ctype in ("application/octet-stream", "text/html", ""):
        fetch_env.routes["http://example.com/mislabeled"] = _pdf_response(body, ctype)
        web_tool_impls._reset_state_for_tests()
        result = web_fetch("http://example.com/mislabeled")
        assert "sniffed content" in result, f"failed for content-type {ctype!r}"


def test_fetch_pdf_reads_past_html_cap(fetch_env):
    """Mid-stream cap raise: a >max_bytes PDF must be read in full (spec §1)."""
    filler = "lorem ipsum dolor sit amet " * 40
    body = _make_pdf([f"page {i} {filler}" for i in range(400)])
    assert len(body) > 64 * 1024
    fetch_env.routes["http://example.com/big.pdf"] = _pdf_response(body)
    result = web_fetch("http://example.com/big.pdf", max_bytes=64 * 1024)
    assert "page 0" in result  # parsed => the full byte stream was read


def test_fetch_pdf_over_ceiling_refused(fetch_env, monkeypatch):
    monkeypatch.setattr(web_tool_impls, "PDF_MAX_BYTES", 1024)
    body = _make_pdf(["x" * 500] * 20)
    assert len(body) > 1024
    fetch_env.routes["http://example.com/huge.pdf"] = _pdf_response(body)
    with pytest.raises(LocalToolError, match=r"too-large.*media ingestion"):
        web_fetch("http://example.com/huge.pdf")


def test_fetch_pdf_extracted_text_truncated_with_page_count(fetch_env):
    body = _make_pdf([f"page {i} " + "words " * 200 for i in range(30)])
    fetch_env.routes["http://example.com/long.pdf"] = _pdf_response(body)
    result = web_fetch("http://example.com/long.pdf", max_bytes=2048)
    assert "truncated: extracted text exceeded max_bytes=2048" in result
    assert "of 30 pages" in result
    # early stop: not every page was processed
    assert "processed 30 of 30" not in result


def test_fetch_pdf_encrypted_refused(fetch_env):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "secret")
    body = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="hunter2")
    doc.close()
    fetch_env.routes["http://example.com/locked.pdf"] = _pdf_response(body)
    with pytest.raises(LocalToolError, match=r"pdf-error.*encrypted"):
        web_fetch("http://example.com/locked.pdf")


def test_fetch_pdf_textless_points_at_ocr(fetch_env):
    doc = pymupdf.open()
    doc.new_page()  # one blank page, no text layer
    body = doc.tobytes()
    doc.close()
    fetch_env.routes["http://example.com/scan.pdf"] = _pdf_response(body)
    with pytest.raises(LocalToolError, match=r"empty-content.*OCR"):
        web_fetch("http://example.com/scan.pdf")


def test_fetch_pdf_damaged_bytes_error(fetch_env):
    # pymupdf is sometimes lenient with garbage after a valid header: it may
    # raise at open (-> pdf-error) or yield a zero-page doc (-> empty-content).
    # Either way the caller gets a structured refusal, never a crash.
    fetch_env.routes["http://example.com/junk.pdf"] = _pdf_response(b"%PDF-1.7 garbage not a real pdf")
    with pytest.raises(LocalToolError, match=r"pdf-error|empty-content"):
        web_fetch("http://example.com/junk.pdf")


def test_fetch_pdf_missing_dep_message(fetch_env, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_pymupdf(name, *a, **k):
        if name == "pymupdf":
            raise ImportError("nope")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pymupdf)
    fetch_env.routes["http://example.com/doc.pdf"] = _pdf_response(_make_pdf(["hi"]))
    with pytest.raises(LocalToolError, match=r"missing-dep.*tldw_chatbook\[pdf\]"):
        web_fetch("http://example.com/doc.pdf")


def test_fetch_pdf_result_is_cached(fetch_env):
    body = _make_pdf(["cache me"])
    fetch_env.routes["http://example.com/doc.pdf"] = _pdf_response(body)
    first = web_fetch("http://example.com/doc.pdf")
    second = web_fetch("http://example.com/doc.pdf")
    assert first == second
    assert fetch_env.calls.count("http://example.com/doc.pdf") == 1


def test_fetch_html_types_unaffected(fetch_env):
    """Regression guard: ordinary HTML still goes through trafilatura/tag-strip."""
    fetch_env.routes["http://example.com/page"] = _html_page()
    assert "main article body sentence" in web_fetch("http://example.com/page")
```

- [ ] **Step 2: Run to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_tool_impls.py -v -p no:randomly -k "pdf"`
Expected: FAIL — `test_fetch_extracts_pdf_text` and siblings raise `LocalToolError: [empty-content] unsupported content type: application/pdf` (current behavior); the ceiling test fails on missing `PDF_MAX_BYTES`.

- [ ] **Step 3: Implement `_fetch_once` sniffing + the PDF branch**

Replace `_fetch_once` with:

```python
_PDF_MAGIC = b"%PDF-"


def _fetch_once(
    client: httpx.Client,
    url: str,
    max_bytes: int,
    *,
    pdf_max_bytes: "int | None" = None,
    html_only: bool = False,
) -> tuple[int, httpx.Headers, bytes, bool, bool]:
    """One GET with a bounded streaming read; redirects are NOT followed.

    Returns (status, headers, body, truncated, is_pdf). The read cap is
    decided MID-STREAM (spec §1): a response identified as PDF — by header
    or by a %PDF- prefix sniff on the first >=5 buffered bytes — reads up
    to ``pdf_max_bytes`` instead of ``max_bytes``, because a byte-truncated
    PDF is unparseable. ``html_only`` (web_crawl) stops the body read after
    the sniff buffer when the declared main type is non-empty and not HTML.
    """
    with client.stream("GET", url) as response:
        status = response.status_code
        if status in _REDIRECT_STATUSES:
            return status, response.headers, b"", False, False
        declared = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        chunks: list[bytes] = []
        downloaded = 0
        is_pdf: "bool | None" = True if declared == "application/pdf" else None
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            downloaded += len(chunk)
            if is_pdf is None and downloaded >= len(_PDF_MAGIC):
                is_pdf = b"".join(chunks)[: len(_PDF_MAGIC)] == _PDF_MAGIC
            if html_only and declared and declared not in _HTML_TYPES:
                break  # crawl only needs the type (PDFs included); don't drain the body
            cap = pdf_max_bytes if (is_pdf and pdf_max_bytes is not None) else max_bytes
            if downloaded > cap:
                break  # overshoot by at most one chunk; sliced below
        if is_pdf is None:  # body shorter than the magic prefix
            is_pdf = b"".join(chunks)[: len(_PDF_MAGIC)] == _PDF_MAGIC
        body = b"".join(chunks)
        cap = pdf_max_bytes if (is_pdf and pdf_max_bytes is not None) else max_bytes
        truncated = len(body) > cap
        if truncated:
            body = body[:cap]
        return status, response.headers, body, truncated, is_pdf
```

Add after the fetch constants (~line 100):

```python
PDF_MAX_BYTES = 20 * 1024 * 1024  # refusal threshold, never a truncation (spec §1)
```

Add the extractor (after `_extract_text`):

```python
def _extract_pdf_text(body: bytes, max_bytes: int) -> str:
    """Ephemeral PDF text extraction: bytes in, text out, nothing on disk.

    Stops the page loop as soon as accumulated text passes ``max_bytes`` —
    a 20 MB PDF can be thousands of pages and the tail is about to be
    thrown away anyway (spec §1).
    """
    try:
        import pymupdf  # local import: optional heavy dep (pdf extra)
    except ImportError as exc:
        raise LocalToolError(
            "[missing-dep] PDF support requires pymupdf — pip install tldw_chatbook[pdf]"
        ) from exc
    try:
        doc = pymupdf.open(stream=body, filetype="pdf")
    except Exception as exc:
        raise LocalToolError(f"[pdf-error] could not parse PDF: {exc}") from exc
    try:
        if doc.needs_pass and not doc.authenticate(""):
            raise LocalToolError("[pdf-error] PDF is encrypted")
        total_pages = doc.page_count
        parts: list[str] = []
        size = 0
        processed = 0
        for page in doc:
            text = page.get_text()
            processed += 1
            if text.strip():
                parts.append(text.strip())
                size += len(text.encode("utf-8"))
            if size > max_bytes:
                break
    except LocalToolError:
        raise
    except Exception as exc:  # damaged page trees surface mid-iteration
        raise LocalToolError(f"[pdf-error] could not parse PDF: {exc}") from exc
    finally:
        doc.close()
    joined = "\n\n".join(parts)
    if not joined:
        raise LocalToolError(
            "[empty-content] PDF contains no extractable text (scanned document?) "
            "— use media ingestion with OCR"
        )
    if size > max_bytes or processed < total_pages:
        raw = joined.encode("utf-8")[:max_bytes]
        joined = raw.decode("utf-8", errors="ignore") + (
            f"\n\n[... truncated: extracted text exceeded max_bytes={max_bytes}; "
            f"processed {processed} of {total_pages} pages ...]"
        )
    return joined
```

Wire `web_fetch`: the call site becomes `status, headers, body, truncated, is_pdf = _fetch_once(client, current_url, max_bytes, pdf_max_bytes=PDF_MAX_BYTES)`, and the post-loop block becomes:

```python
    if status >= 400:
        raise LocalToolError(f"[http-{status}] upstream returned status {status} for {url!r}")

    if is_pdf:
        if truncated:  # hit the 20 MB PDF ceiling: refuse, never truncate bytes
            raise LocalToolError("[too-large] PDF exceeds 20 MB — use media ingestion for large documents")
        text = _extract_pdf_text(body, max_bytes)
    else:
        text = _extract_text(body, headers.get("content-type", ""))
        if truncated:
            text += f"\n\n[... truncated: response exceeded max_bytes={max_bytes} ...]"
    _cache_put((url, max_bytes), text)
    return text
```

- [ ] **Step 4: Run the whole file**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_tool_impls.py -v -p no:randomly`
Expected: all PASS, including every pre-existing v1 test (redirects, rate limit, byte cap, SSRF).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/web_tool_impls.py Tests/Tools/test_web_tool_impls.py
git commit -m "feat: ephemeral PDF text extraction in web_fetch (task-1358)"
```

---

### Task 3: Crawl pure helpers (parser, normalization, sitemap XML, formatting)

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py` (new section after `web_fetch`)
- Test: Create `Tests/Tools/test_web_crawl.py`

**Interfaces:**
- Consumes: `_truncate_to_bytes` (exists, v1).
- Produces (Task 4/5 consume these exactly):
  - `_CrawlLinkParser(HTMLParser)` with attributes `links: list[str]`, `base_href: str | None`, `title: str` after `feed()`.
  - `_normalize_crawl_url(url: str) -> str` — lowercased www-folded host, fragment dropped, query kept, path defaults to `/`.
  - `_crawl_host(url: str) -> str` — lowercased host with leading `www.` folded, `""` if absent.
  - `_coerce_budget(value, default: int, ceiling: int) -> int` — int coercion, garbage→default, clamp to `[1, ceiling]`.
  - `_parse_sitemap(xml_bytes: bytes) -> tuple[list[str], list[str]]` — `(page_urls, child_sitemap_urls)`; `urlset` fills the first, `sitemapindex` the second.
  - `_format_crawl_result(pages: list[dict], failed: int, blocked: int, stop_reason: str) -> str` — page dicts `{"url": str, "title": str, "excerpt": str, "marker": str | None}`.
  - Constants: all `CRAWL_*`, `SITEMAP_MAX_BYTES`, `_CRAWL_USER_AGENT` from Global Constraints.

- [ ] **Step 1: Write the failing tests** (create `Tests/Tools/test_web_crawl.py`)

```python
"""web_crawl: pure-helper unit tests (no transport) + crawl behavior tests."""

import socket
from types import SimpleNamespace

import httpx
import pytest

from tldw_chatbook.Tools import web_tool_impls
from tldw_chatbook.Tools.web_tool_impls import (
    LocalToolError,
    _CrawlLinkParser,
    _coerce_budget,
    _crawl_host,
    _format_crawl_result,
    _normalize_crawl_url,
    _parse_sitemap,
)


def _parse(html: str) -> _CrawlLinkParser:
    parser = _CrawlLinkParser()
    parser.feed(html)
    parser.close()
    return parser


def test_parser_collects_links_title_and_base():
    p = _parse(
        "<html><head><title>My &amp; Page</title><base href='/de/'></head>"
        "<body><a href='a.html'>A</a><a href='/b'>B</a>"
        "<a>no href</a><a href='c.html'>C</a></body></html>"
    )
    assert p.title == "My & Page"
    assert p.base_href == "/de/"
    assert p.links == ["a.html", "/b", "c.html"]


def test_parser_survives_malformed_html():
    p = _parse("<a href='x'><b><title>t</<><a href='y'>")
    assert "x" in p.links


def test_normalize_folds_www_case_and_fragment():
    assert (
        _normalize_crawl_url("HTTP://WWW.Example.COM/Path?q=1#frag")
        == "http://example.com/Path?q=1"
    )
    assert _normalize_crawl_url("http://example.com") == "http://example.com/"


def test_crawl_host_folds_www():
    assert _crawl_host("https://www.Example.com/x") == "example.com"
    assert _crawl_host("https://example.com/x") == "example.com"
    assert _crawl_host("not a url") == ""


def test_coerce_budget_clamps_and_defaults():
    assert _coerce_budget(3, 20, 40) == 3
    assert _coerce_budget(999, 20, 40) == 40
    assert _coerce_budget(0, 20, 40) == 1
    assert _coerce_budget(-5, 20, 40) == 1
    assert _coerce_budget("garbage", 20, 40) == 20
    assert _coerce_budget(None, 20, 40) == 20
    assert _coerce_budget("7", 20, 40) == 7


_URLSET = b"""<?xml version="1.0"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a</loc></url>
  <url><loc> https://example.com/b </loc></url>
</urlset>"""

_INDEX = b"""<?xml version="1.0"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>https://example.com/s1.xml</loc></sitemap>
  <sitemap><loc>https://example.com/s2.xml</loc></sitemap>
</sitemapindex>"""


def test_parse_sitemap_urlset_and_index():
    pages, children = _parse_sitemap(_URLSET)
    assert pages == ["https://example.com/a", "https://example.com/b"]
    assert children == []
    pages, children = _parse_sitemap(_INDEX)
    assert pages == []
    assert children == ["https://example.com/s1.xml", "https://example.com/s2.xml"]


def test_parse_sitemap_garbage_raises():
    with pytest.raises(LocalToolError, match="crawl-failed"):
        _parse_sitemap(b"this is not xml at all")


def _page(url, title="T", excerpt="ex", marker=None):
    return {"url": url, "title": title, "excerpt": excerpt, "marker": marker}


def test_format_blocks_footer_and_marker():
    out = _format_crawl_result(
        [_page("http://e.com/1", "One", "first words"),
         _page("http://e.com/m.pdf", "", "", marker="[application/pdf]")],
        failed=2, blocked=1, stop_reason="page budget reached",
    )
    assert "1. One\n   URL: http://e.com/1\n   first words" in out
    assert "2. [application/pdf]\n   URL: http://e.com/m.pdf" in out
    assert out.endswith("Crawled 2 pages (2 failed, 1 blocked). Stopped: page budget reached.")


def test_format_total_cap_omits_pages():
    from tldw_chatbook.Tools.web_tool_impls import CRAWL_RESULT_MAX_BYTES

    pages = [_page(f"http://e.com/{i}", f"T{i}", "x" * 190) for i in range(200)]
    out = _format_crawl_result(pages, failed=0, blocked=0, stop_reason="page budget reached")
    assert "further pages omitted" in out
    assert len(out.encode("utf-8")) < CRAWL_RESULT_MAX_BYTES + 2048  # footer + omission slack
    assert "Crawled 200 pages" in out  # footer reports the crawl, not the capped list


def test_format_empty_crawl_is_just_footer():
    out = _format_crawl_result([], failed=1, blocked=0, stop_reason="no more links within depth")
    assert out == "Crawled 0 pages (1 failed, 0 blocked). Stopped: no more links within depth."
```

- [ ] **Step 2: Run to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_crawl.py -v -p no:randomly`
Expected: FAIL at import (`_CrawlLinkParser` etc. undefined).

- [ ] **Step 3: Implement the helpers**

Add to `web_tool_impls.py` (new section after the `web_fetch` function; extend the module imports with `from collections import deque`, `from html.parser import HTMLParser`, and `import xml.etree.ElementTree as xET`):

```python
# ---------------------------------------------------------------------------
# web_crawl (spec 2026-08-06 §2)
# ---------------------------------------------------------------------------

CRAWL_DEFAULT_MAX_PAGES = 20
CRAWL_MAX_PAGES_CEILING = 40
CRAWL_DEFAULT_MAX_DEPTH = 2
CRAWL_MAX_DEPTH_CEILING = 5
CRAWL_DEADLINE_SECONDS = 120.0
CRAWL_PAGE_TIMEOUT_SECONDS = 10.0   # per page; a hung page must not eat the crawl
CRAWL_EXCERPT_MAX_CHARS = 200
CRAWL_RESULT_MAX_BYTES = 24 * 1024
CRAWL_BLOCK_MAX_BYTES = 1024
SITEMAP_MAX_BYTES = 5 * 1024 * 1024

_CRAWL_USER_AGENT = "tldw-chatbook-web-crawl/1.0"

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class _CrawlLinkParser(HTMLParser):
    """Collect <a href>, <base href>, and <title> text from one page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.base_href: "str | None" = None
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        elif tag == "base" and self.base_href is None:
            href = dict(attrs).get("href")
            if href:
                self.base_href = href
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def _crawl_host(url: str) -> str:
    """Lowercased host with a leading ``www.`` folded; '' when absent/bad."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _normalize_crawl_url(url: str) -> str:
    """Visited-set identity: scheme+folded host+path+query, no fragment."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{parts.port}" if parts.port else ""
    path = parts.path or "/"
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{host}{port}{path}{query}"


def _coerce_budget(value, default: int, ceiling: int) -> int:
    """v1 argument style: garbage degrades to the default, range clamps."""
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(result, ceiling))


def _parse_sitemap(xml_bytes: bytes) -> tuple[list[str], list[str]]:
    """Return (page_urls, child_sitemap_urls) from a urlset/sitemapindex."""
    try:
        root = xET.fromstring(xml_bytes)
    except xET.ParseError as exc:
        raise LocalToolError(f"[crawl-failed] sitemap could not be parsed: {exc}") from exc
    locs = [
        loc.text.strip()
        for loc in root.findall(f".//{_SITEMAP_NS}loc")
        if loc.text and loc.text.strip()
    ]
    if root.tag == f"{_SITEMAP_NS}sitemapindex":
        return [], locs
    return locs, []


def _format_crawl_result(pages: list[dict], failed: int, blocked: int, stop_reason: str) -> str:
    blocks: list[str] = []
    total = 0
    for i, page in enumerate(pages, 1):
        if page["marker"]:
            block = f"{i}. {page['marker']}\n   URL: {page['url']}"
        else:
            block = f"{i}. {page['title'] or 'No title'}\n   URL: {page['url']}"
            if page["excerpt"]:
                block += f"\n   {page['excerpt']}"
        block = _truncate_to_bytes(block, CRAWL_BLOCK_MAX_BYTES)
        block_bytes = len(block.encode("utf-8"))
        if total + block_bytes > CRAWL_RESULT_MAX_BYTES:
            blocks.append("… [further pages omitted: total size cap reached]")
            break
        blocks.append(block)
        total += block_bytes
    footer = f"Crawled {len(pages)} pages ({failed} failed, {blocked} blocked). Stopped: {stop_reason}."
    return "\n\n".join(blocks + [footer]) if blocks else footer
```

- [ ] **Step 4: Run to verify green**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_crawl.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/web_tool_impls.py Tests/Tools/test_web_crawl.py
git commit -m "feat: web_crawl pure helpers — link parser, URL identity, sitemap XML, formatting"
```

---

### Task 4: `web_crawl` BFS core

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py` (below the Task 3 helpers)
- Test: `Tests/Tools/test_web_crawl.py`

**Interfaces:**
- Consumes: Task 2's `_fetch_once(..., html_only=True)` five-tuple; Task 3's helpers; Task 1's `_cache_put`; v1's `_validate_hop`, `_enforce_rate_limit`, `_decode_body`, `_extract_text`, `_HTML_TYPES`, `_REDIRECT_STATUSES`, `FETCH_MAX_REDIRECTS`, `FETCH_MAX_BYTES`, `_transport`.
- Produces: `web_crawl(url: str, *, max_pages=CRAWL_DEFAULT_MAX_PAGES, max_depth=CRAWL_DEFAULT_MAX_DEPTH) -> str` (Task 5 adds `sitemap_url`); internal `_crawl_fetch_page(client, url, deadline, *, max_bytes=FETCH_MAX_BYTES, html_only=True) -> tuple[str, httpx.Headers, bytes, bool]` returning `(final_url, headers, body, truncated)`; internal `class _CrawlDeadline(Exception)`. Stop-reason strings (exact): `"page budget reached"`, `"no more links within depth"`, `"deadline reached"`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Tools/test_web_crawl.py`)

```python
# ---------------------------------------------------------------------------
# web_crawl BFS (transport-level tests, v1 fetch_env conventions)
# ---------------------------------------------------------------------------

from tldw_chatbook.Tools.web_tool_impls import web_crawl

_PUBLIC_IP = "93.184.216.34"


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def crawl_env(monkeypatch):
    """MockTransport + fake DNS + fake clock, mirroring test_web_tool_impls."""
    routes: dict[str, object] = {}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        item = routes.get(url)
        if item is None:
            return httpx.Response(404)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(request)
        return item

    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (_PUBLIC_IP, 80))]
    )
    monkeypatch.setattr(web_tool_impls, "_transport", httpx.MockTransport(handler))
    clock = _FakeClock()
    monkeypatch.setattr(web_tool_impls, "time", clock)
    web_tool_impls._reset_state_for_tests()
    yield SimpleNamespace(routes=routes, calls=calls, clock=clock)
    web_tool_impls._reset_state_for_tests()


def _html(body_text: str, links: list[str] = (), title: str = "Page") -> httpx.Response:
    anchors = "".join(f"<a href='{href}'>l</a>" for href in links)
    html = (
        f"<html><head><title>{title}</title></head>"
        f"<body><p>{body_text}</p>{anchors}</body></html>"
    )
    return httpx.Response(200, content=html.encode(), headers={"content-type": "text/html"})


def _site(env, spec: dict) -> None:
    """spec: url -> (body_text, links) tuples or ready Responses."""
    for url, item in spec.items():
        if isinstance(item, tuple):
            body_text, links = item
            env.routes[url] = _html(body_text, links, title=f"Title {url.rsplit('/', 1)[-1]}")
        else:
            env.routes[url] = item


def test_crawl_lists_pages_with_titles_and_excerpts(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("home page words", ["/a", "/b"]),
        "http://example.com/a": ("alpha page words", []),
        "http://example.com/b": ("beta page words", []),
    })
    out = web_crawl("http://example.com/")
    assert "Title a" in out and "alpha page words" in out
    assert "Title b" in out and "beta page words" in out
    assert "Crawled 3 pages (0 failed, 0 blocked)" in out
    assert out.endswith("Stopped: no more links within depth.")


def test_crawl_respects_max_pages_budget(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", [f"/p{i}" for i in range(10)]),
        **{f"http://example.com/p{i}": (f"page {i}", []) for i in range(10)},
    })
    out = web_crawl("http://example.com/", max_pages=4)
    assert len(crawl_env.calls) == 4
    assert "Stopped: page budget reached." in out


def test_crawl_respects_max_depth(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("d0", ["/d1"]),
        "http://example.com/d1": ("d1", ["/d2"]),
        "http://example.com/d2": ("d2", ["/d3"]),
        "http://example.com/d3": ("d3", []),
    })
    web_crawl("http://example.com/", max_depth=1)
    assert "http://example.com/d2" not in crawl_env.calls
    assert "http://example.com/d1" in crawl_env.calls


def test_crawl_stays_on_host_and_folds_www(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", [
            "http://other.com/x",
            "http://www.example.com/a",     # same host after www fold
            "http://example.com/a",         # duplicate of the above
            "http://example.com/a#section", # fragment variant: same identity
        ]),
        "http://www.example.com/a": ("alpha", []),
    })
    web_crawl("http://example.com/")
    assert "http://other.com/x" not in crawl_env.calls
    # www/apex/fragment variants collapse to one fetch.
    assert crawl_env.calls.count("http://www.example.com/a") == 1
    assert not any(c.startswith("http://example.com/a") for c in crawl_env.calls)


def test_crawl_honors_base_href(crawl_env):
    crawl_env.routes["http://example.com/dir/"] = httpx.Response(
        200,
        content=(
            b"<html><head><title>B</title><base href='/other/'></head>"
            b"<body><a href='rel.html'>r</a></body></html>"
        ),
        headers={"content-type": "text/html"},
    )
    _site(crawl_env, {"http://example.com/other/rel.html": ("resolved", [])})
    web_crawl("http://example.com/dir/")
    assert "http://example.com/other/rel.html" in crawl_env.calls


def test_crawl_blocked_link_counted_not_fatal(crawl_env, monkeypatch):
    """DNS flips private mid-crawl (rebinding): the later same-host URL is
    guard-blocked, counted, and the crawl continues.

    Same-host crawls share one hostname, so per-URL DNS is impossible — the
    deterministic construction flips the answer inside an earlier page's
    handler. BFS order: / -> /ok (flips DNS) -> /blocked-later (resolves
    private, refused before any request)."""
    state = {"private": False}

    def dns(host, *a, **k):
        ip = "10.0.0.5" if state["private"] else _PUBLIC_IP
        return [(2, 1, 6, "", (ip, 80))]

    monkeypatch.setattr(socket, "getaddrinfo", dns)
    _site(crawl_env, {
        "http://example.com/": ("root", ["/ok", "/blocked-later"]),
        "http://example.com/blocked-later": ("never seen", []),
    })

    def ok_then_flip(request):
        state["private"] = True
        return _html("fine words", [])

    crawl_env.routes["http://example.com/ok"] = ok_then_flip
    out = web_crawl("http://example.com/")
    assert "1 blocked" in out
    assert "fine words" in out
    assert "http://example.com/blocked-later" not in crawl_env.calls
```

```python
def test_crawl_failed_page_counted_not_fatal(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/dead", "/ok"]),
        "http://example.com/ok": ("fine", []),
    })
    crawl_env.routes["http://example.com/dead"] = httpx.Response(500)
    out = web_crawl("http://example.com/")
    assert "1 failed" in out
    assert "fine" in out


def test_crawl_start_url_failure_raises_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/"] = httpx.Response(500)
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/")


def test_crawl_nonhtml_listed_with_marker_not_expanded(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/doc.pdf"]),
    })
    crawl_env.routes["http://example.com/doc.pdf"] = httpx.Response(
        200, content=b"%PDF-1.7 pretend", headers={"content-type": "application/pdf"}
    )
    out = web_crawl("http://example.com/")
    assert "[application/pdf]" in out
    assert "http://example.com/doc.pdf" in out


def test_crawl_offhost_redirect_listed_not_expanded(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/moved"]),
        "http://elsewhere.com/final": ("away content", ["/next-on-elsewhere"]),
    })
    crawl_env.routes["http://example.com/moved"] = httpx.Response(
        302, headers={"location": "http://elsewhere.com/final"}
    )
    out = web_crawl("http://example.com/")
    assert "http://elsewhere.com/final" in out       # listed at final URL
    assert "http://elsewhere.com/next-on-elsewhere" not in crawl_env.calls


def test_crawl_redirect_into_private_space_blocked(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/trap"]),
    })
    crawl_env.routes["http://example.com/trap"] = httpx.Response(
        302, headers={"location": "http://169.254.169.254/latest/meta-data"}
    )
    out = web_crawl("http://example.com/")
    assert "1 blocked" in out
    assert "http://169.254.169.254/latest/meta-data" not in crawl_env.calls


def test_crawl_deadline_stops(crawl_env):
    from tldw_chatbook.Tools.web_tool_impls import CRAWL_DEADLINE_SECONDS

    def slow_then_page(request):
        crawl_env.clock.now += CRAWL_DEADLINE_SECONDS + 1
        return _html("slow", [])

    _site(crawl_env, {"http://example.com/": ("root", ["/slow", "/never"])})
    crawl_env.routes["http://example.com/slow"] = slow_then_page
    _site(crawl_env, {"http://example.com/never": ("never", [])})
    out = web_crawl("http://example.com/")
    assert "Stopped: deadline reached." in out
    assert "http://example.com/never" not in crawl_env.calls


def test_crawl_rate_limits_between_pages(crawl_env):
    _site(crawl_env, {
        "http://example.com/": ("root", ["/a"]),
        "http://example.com/a": ("alpha", []),
    })
    web_crawl("http://example.com/")
    assert crawl_env.clock.sleeps  # second same-domain fetch waited


def test_crawl_uses_crawl_user_agent(crawl_env):
    seen_agents: list[str] = []

    def capture(request):
        seen_agents.append(request.headers.get("user-agent", ""))
        return _html("root", [])

    crawl_env.routes["http://example.com/"] = capture
    web_crawl("http://example.com/")
    assert seen_agents == ["tldw-chatbook-web-crawl/1.0"]


def test_crawl_warm_writes_fetch_cache(crawl_env):
    from tldw_chatbook.Tools.web_tool_impls import web_fetch

    _site(crawl_env, {"http://example.com/": ("cache me please words", [])})
    web_crawl("http://example.com/")
    n_calls = len(crawl_env.calls)
    result = web_fetch("http://example.com/")
    assert "cache me please words" in result
    assert len(crawl_env.calls) == n_calls  # served from cache, no new request


def test_crawl_invalid_args(crawl_env):
    with pytest.raises(LocalToolError, match="invalid-args"):
        web_crawl("")
    with pytest.raises(LocalToolError, match="invalid-args"):
        web_crawl("   ")
```

- [ ] **Step 2: Run to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_crawl.py -v -p no:randomly`
Expected: FAIL at import (`web_crawl` undefined); Task 3's helper tests still PASS.

- [ ] **Step 3: Implement `web_crawl`**

Add below the Task 3 helpers in `web_tool_impls.py`:

```python
class _CrawlDeadline(Exception):
    """Internal: the wall-clock budget expired mid-fetch."""


def _crawl_fetch_page(
    client: httpx.Client,
    url: str,
    deadline: float,
    *,
    max_bytes: int = FETCH_MAX_BYTES,
    html_only: bool = True,
) -> tuple[str, "httpx.Headers", bytes, bool]:
    """Guarded, rate-limited GET with the crawl's redirect loop.

    Returns (final_url, headers, body, truncated). Checks the deadline
    between redirect hops — one page's full chain must not overshoot the
    crawl budget by minutes (spec §2).
    """
    current = url
    for _hop in range(FETCH_MAX_REDIRECTS + 1):
        if time.monotonic() >= deadline:
            raise _CrawlDeadline()
        _validate_hop(current)
        _enforce_rate_limit(urlsplit(current).hostname or "unknown")
        try:
            status, headers, body, truncated, _is_pdf = _fetch_once(
                client, current, max_bytes, html_only=html_only
            )
        except httpx.TimeoutException as exc:
            raise LocalToolError(f"[timeout] fetch timed out: {current!r}") from exc
        except httpx.InvalidURL as exc:
            raise LocalToolError(f"[invalid-url] {exc}") from exc
        except httpx.HTTPError as exc:
            raise LocalToolError(f"[fetch-failed] {exc}") from exc
        if status in _REDIRECT_STATUSES:
            location = headers.get("location")
            if not location:
                raise LocalToolError(f"[http-{status}] redirect without a Location header")
            current = urljoin(current, location)
            continue
        if status >= 400:
            raise LocalToolError(f"[http-{status}] upstream returned status {status} for {current!r}")
        return current, headers, body, truncated
    raise LocalToolError(f"[redirect-limit] exceeded {FETCH_MAX_REDIRECTS} redirects for {url!r}")


def web_crawl(
    url: str,
    *,
    max_pages: int = CRAWL_DEFAULT_MAX_PAGES,
    max_depth: int = CRAWL_DEFAULT_MAX_DEPTH,
) -> str:
    """Same-host breadth-first crawl returning a bounded page list.

    Each listed page carries URL, title, and a short excerpt; the model is
    expected to follow up with web_fetch on pages that matter (spec §2).
    Every URL is egress-guarded; budgets bound fetch ATTEMPTS; a wall-clock
    deadline bounds the whole crawl. Ephemeral: no database writes.

    Raises:
        LocalToolError: [invalid-args] for a bad url/host; [crawl-failed]
            when the START url cannot be fetched (per-page failures inside
            the crawl are results, counted in the footer).
    """
    if not isinstance(url, str) or not url.strip():
        raise LocalToolError("[invalid-args] url must be a non-empty string")
    url = url.strip()
    max_pages = _coerce_budget(max_pages, CRAWL_DEFAULT_MAX_PAGES, CRAWL_MAX_PAGES_CEILING)
    max_depth = _coerce_budget(max_depth, CRAWL_DEFAULT_MAX_DEPTH, CRAWL_MAX_DEPTH_CEILING)
    scope_host = _crawl_host(url)
    if not scope_host:
        raise LocalToolError(f"[invalid-args] url has no host: {url!r}")

    deadline = time.monotonic() + CRAWL_DEADLINE_SECONDS
    queue: "deque[tuple[str, int]]" = deque([(url, 0)])
    visited = {_normalize_crawl_url(url)}
    pages: list[dict] = []
    failed = blocked = 0
    attempts = 0
    stop_reason = "no more links within depth"

    client = httpx.Client(
        follow_redirects=False,
        timeout=CRAWL_PAGE_TIMEOUT_SECONDS,
        headers={"User-Agent": _CRAWL_USER_AGENT},
        transport=_transport,
        trust_env=False,
    )
    try:
        while queue:
            if attempts >= max_pages:
                stop_reason = "page budget reached"
                break
            if time.monotonic() >= deadline:
                stop_reason = "deadline reached"
                break
            current, depth = queue.popleft()
            is_start = attempts == 0
            attempts += 1
            try:
                final_url, headers, body, _truncated = _crawl_fetch_page(client, current, deadline)
            except _CrawlDeadline:
                stop_reason = "deadline reached"
                break
            except LocalToolError as exc:
                if is_start:
                    raise LocalToolError(f"[crawl-failed] start URL could not be fetched: {exc}") from exc
                if "[ssrf]" in str(exc):
                    blocked += 1
                else:
                    failed += 1
                continue
            visited.add(_normalize_crawl_url(final_url))

            ctype = (headers.get("content-type") or "").split(";", 1)[0].strip().lower()
            if ctype and ctype not in _HTML_TYPES:
                pages.append({"url": final_url, "title": "", "excerpt": "", "marker": f"[{ctype}]"})
                continue

            html = _decode_body(body, headers.get("content-type", ""))
            parser = _CrawlLinkParser()
            try:
                parser.feed(html)
                parser.close()
            except Exception:  # noqa: BLE001 — keep whatever was collected
                pass
            try:
                full_text = _extract_text(body, headers.get("content-type", ""))
            except LocalToolError:
                full_text = ""
            if full_text:
                _cache_put((final_url, FETCH_MAX_BYTES), full_text)
            pages.append({
                "url": final_url,
                "title": parser.title.strip(),
                "excerpt": full_text[:CRAWL_EXCERPT_MAX_CHARS].strip(),
                "marker": None,
            })

            # Expansion: same-host pages only, within the depth budget. A page
            # that redirected off-host is listed but its links are not followed.
            if depth >= max_depth or _crawl_host(final_url) != scope_host:
                continue
            base = urljoin(final_url, parser.base_href) if parser.base_href else final_url
            for href in parser.links:
                absolute = urljoin(base, href)
                try:
                    scheme = urlsplit(absolute).scheme.lower()
                except ValueError:
                    continue
                if scheme not in _ALLOWED_SCHEMES:
                    continue
                if _crawl_host(absolute) != scope_host:
                    continue
                norm = _normalize_crawl_url(absolute)
                if norm in visited:
                    continue
                visited.add(norm)
                queue.append((absolute, depth + 1))
    finally:
        client.close()

    return _format_crawl_result(pages, failed, blocked, stop_reason)
```

- [ ] **Step 4: Run both web-tool test files**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_crawl.py Tests/Tools/test_web_tool_impls.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/web_tool_impls.py Tests/Tools/test_web_crawl.py
git commit -m "feat: web_crawl same-host BFS with budgets, deadline, cache warm-write (task-1357)"
```

---

### Task 5: Sitemap mode

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py` (`web_crawl` signature + a seeding helper)
- Test: `Tests/Tools/test_web_crawl.py`

**Interfaces:**
- Consumes: Task 4's `web_crawl`, `_crawl_fetch_page`; Task 3's `_parse_sitemap`, `_crawl_host`, `_normalize_crawl_url`; `SITEMAP_MAX_BYTES`.
- Produces: `web_crawl(url, *, max_pages=..., max_depth=..., sitemap_url: str | None = None) -> str`; stop reason `"sitemap exhausted"`; internal `_seed_from_sitemap(client, sitemap_url, scope_host, max_pages, deadline) -> list[str]`.

- [ ] **Step 1: Write the failing tests** (append to `Tests/Tools/test_web_crawl.py`)

```python
# ---------------------------------------------------------------------------
# sitemap mode (spec §2)
# ---------------------------------------------------------------------------

def _sitemap_response(xml: bytes) -> httpx.Response:
    return httpx.Response(200, content=xml, headers={"content-type": "application/xml"})


def test_sitemap_mode_seeds_pages_and_skips_expansion(crawl_env):
    xml = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://example.com/a</loc></url>"
        b"<url><loc>http://example.com/b</loc></url>"
        b"</urlset>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {
        "http://example.com/a": ("alpha words", ["/should-not-follow"]),
        "http://example.com/b": ("beta words", []),
        "http://example.com/should-not-follow": ("nope", []),
    })
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "alpha words" in out and "beta words" in out
    # sitemap IS the discovery: links on seeded pages are not expanded
    assert "http://example.com/should-not-follow" not in crawl_env.calls
    assert "Stopped: sitemap exhausted." in out


def test_sitemap_index_one_level(crawl_env):
    index = (
        b'<?xml version="1.0"?>'
        b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<sitemap><loc>http://example.com/s1.xml</loc></sitemap>"
        b"<sitemap><loc>http://cdn-other.com/s2.xml</loc></sitemap>"
        b"</sitemapindex>"
    )
    child = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://example.com/from-child</loc></url>"
        b"</urlset>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    crawl_env.routes["http://example.com/s1.xml"] = _sitemap_response(child)
    _site(crawl_env, {"http://example.com/from-child": ("child page words", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "child page words" in out
    # off-host child sitemap (different host than sitemap_url) never fetched
    assert "http://cdn-other.com/s2.xml" not in crawl_env.calls


def test_sitemap_offhost_page_urls_filtered(crawl_env):
    xml = (
        b'<?xml version="1.0"?>'
        b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        b"<url><loc>http://example.com/ok</loc></url>"
        b"<url><loc>http://other.com/evil</loc></url>"
        b"<url><loc>http://10.0.0.5/internal</loc></url>"
        b"</urlset>"
    )
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {"http://example.com/ok": ("fine", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "fine" in out
    assert "http://other.com/evil" not in crawl_env.calls
    assert "http://10.0.0.5/internal" not in crawl_env.calls  # off-host AND private


def test_sitemap_respects_max_pages(crawl_env):
    urls = "".join(
        f"<url><loc>http://example.com/p{i}</loc></url>".encode().decode()
        for i in range(10)
    )
    xml = (
        '<?xml version="1.0"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls}</urlset>"
    ).encode()
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(xml)
    _site(crawl_env, {f"http://example.com/p{i}": (f"page {i}", []) for i in range(10)})
    web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml", max_pages=3)
    page_calls = [c for c in crawl_env.calls if "/p" in c]
    assert len(page_calls) == 3


def test_sitemap_unfetchable_raises_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/sitemap.xml"] = httpx.Response(500)
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")


def test_sitemap_garbage_xml_raises_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(b"not xml")
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
```

- [ ] **Step 2: Run to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_crawl.py -v -p no:randomly -k "sitemap"`
Expected: the new tests FAIL with `TypeError: web_crawl() got an unexpected keyword argument 'sitemap_url'`. (`test_parse_sitemap_*` from Task 3 still PASS.)

- [ ] **Step 3: Implement**

Add the seeding helper next to `_crawl_fetch_page`:

```python
def _seed_from_sitemap(
    client: httpx.Client,
    sitemap_url: str,
    scope_host: str,
    max_pages: int,
    deadline: float,
) -> list[str]:
    """Collect up to max_pages same-host page URLs from a sitemap.

    Sitemap fetches are discovery overhead — they do NOT consume the page
    budget; the deadline bounds a pathological index (spec §2). Host rules:
    child sitemaps must share sitemap_url's host; page URLs must share the
    crawl scope host.
    """
    final_url, _headers, body, truncated = _crawl_fetch_page(
        client, sitemap_url, deadline, max_bytes=SITEMAP_MAX_BYTES, html_only=False
    )
    if truncated:
        raise LocalToolError(f"[crawl-failed] sitemap exceeds {SITEMAP_MAX_BYTES} bytes: {sitemap_url!r}")
    page_urls, children = _parse_sitemap(body)
    sitemap_host = _crawl_host(final_url)

    urls: list[str] = []
    seen: set[str] = set()

    def take(candidates: list[str]) -> None:
        for candidate in candidates:
            if len(urls) >= max_pages:
                return
            if _crawl_host(candidate) != scope_host:
                continue
            norm = _normalize_crawl_url(candidate)
            if norm in seen:
                continue
            seen.add(norm)
            urls.append(candidate)

    take(page_urls)
    for child in children:
        if len(urls) >= max_pages or time.monotonic() >= deadline:
            break
        if _crawl_host(child) != sitemap_host:
            continue
        try:
            _f, _h, child_body, child_truncated = _crawl_fetch_page(
                client, child, deadline, max_bytes=SITEMAP_MAX_BYTES, html_only=False
            )
        except (LocalToolError, _CrawlDeadline):
            continue
        if child_truncated:
            continue
        try:
            child_pages, _nested = _parse_sitemap(child_body)  # one level: nested indexes ignored
        except LocalToolError:
            continue
        take(child_pages)
    return urls
```

In `web_crawl`: add `sitemap_url: "str | None" = None` to the signature and docstring ("sitemap mode: the page list comes from the sitemap; `max_depth` is ignored and links are not expanded"). After the client is constructed (inside the `try`), seed the queue:

```python
        expand_links = sitemap_url is None
        if sitemap_url is not None:
            if not isinstance(sitemap_url, str) or not sitemap_url.strip():
                raise LocalToolError("[invalid-args] sitemap_url must be a non-empty string")
            try:
                seeded = _seed_from_sitemap(client, sitemap_url.strip(), scope_host, max_pages, deadline)
            except _CrawlDeadline:
                seeded = []
            except LocalToolError as exc:
                if "[crawl-failed]" in str(exc):
                    raise
                raise LocalToolError(f"[crawl-failed] sitemap could not be fetched: {exc}") from exc
            queue = deque((u, 0) for u in seeded)
            visited = {_normalize_crawl_url(u) for u in seeded}
            stop_reason = "sitemap exhausted"
```

and change the start-URL special case + expansion guard to respect the mode: the `is_start` → `[crawl-failed]` escalation applies only when `sitemap_url is None` (in sitemap mode a failing first page is an ordinary per-page failure), and the expansion block runs only `if expand_links`. When the queue drains in sitemap mode the default stop reason is `"sitemap exhausted"` (set above); BFS mode keeps `"no more links within depth"`.

- [ ] **Step 4: Run both files**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_crawl.py Tests/Tools/test_web_tool_impls.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/web_tool_impls.py Tests/Tools/test_web_crawl.py
git commit -m "feat: web_crawl sitemap mode — urlset + one-level index, host-scoped, budget-free discovery"
```

---

### Task 6: Registration + tool descriptions

**Files:**
- Modify: `tldw_chatbook/Agents/local_tool_provider.py` (imports ~line 535, `web_fetch` spec ~line 787, append `web_crawl` spec after `web_search` ~line 822)
- Modify: `tldw_chatbook/MCP/server.py` (module docstring line 13: tool list mention)
- Test: `Tests/Agents/test_local_tool_provider.py`

**Interfaces:**
- Consumes: Task 5's final `web_crawl(url, *, max_pages, max_depth, sitemap_url)` + `CRAWL_DEFAULT_MAX_PAGES`, `CRAWL_MAX_PAGES_CEILING`, `CRAWL_DEFAULT_MAX_DEPTH`, `CRAWL_MAX_DEPTH_CEILING` from `web_tool_impls`.
- Produces: catalog id `local:web_crawl`; MCP exposure is automatic via `_register_local_agent_tools` (no MCP code change — docstring only).

- [ ] **Step 1: Write the failing tests**

In `Tests/Agents/test_local_tool_provider.py`, update the pinned catalog list in `test_catalog_lists_default_specs_with_local_ids` — add `"local:web_crawl"` after `"local:web_search"` — and append:

```python
def test_web_crawl_spec_schema(tmp_path):
    p = make_provider(root=tmp_path)
    schema = p.load_schema("local:web_crawl")
    assert schema.parameters["required"] == ["url"]
    props = schema.parameters["properties"]
    assert props["url"]["type"] == "string"
    assert props["max_pages"]["type"] == "integer"
    assert props["max_depth"]["type"] == "integer"
    assert props["sitemap_url"]["type"] == "string"
    for optional in ("max_pages", "max_depth", "sitemap_url"):
        assert optional not in schema.parameters["required"]
    # network-classed: default ask comes from the global permission default.
    assert p.hub_tool_for("web_crawl").tags == ()


def test_web_crawl_description_states_contract(tmp_path):
    p = make_provider(root=tmp_path)
    desc = p.hub_tool_for("web_crawl").description
    assert "web_fetch" in desc          # points the model at the follow-up tool
    assert "sitemap_url" in desc
    assert "max_depth" in desc          # documents the sitemap-mode exception


def test_web_fetch_description_mentions_pdf(tmp_path):
    p = make_provider(root=tmp_path)
    assert "PDF" in p.hub_tool_for("web_fetch").description
```

- [ ] **Step 2: Run to verify they fail**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py -v -p no:randomly -k "catalog or web_crawl or web_fetch_description"`
Expected: the catalog test FAILS (list mismatch) and the three new tests FAIL (`hub_tool_for("web_crawl")` unknown / "PDF" absent).

- [ ] **Step 3: Implement registration**

In `local_tool_provider.py`, extend the `web_tool_impls` import block (~line 535) with `web_crawl, CRAWL_DEFAULT_MAX_PAGES, CRAWL_MAX_PAGES_CEILING, CRAWL_DEFAULT_MAX_DEPTH, CRAWL_MAX_DEPTH_CEILING`. Update `web_fetch`'s description to:

```python
            description=(
                "Fetch a web page and return its extracted text; PDFs are "
                "text-extracted too (up to 20 MB, ephemeral — nothing is "
                "ingested). SSRF-guarded (public http(s) only), "
                "redirect-capped, byte-capped, cached."
            ),
```

Append after the `web_search` spec:

```python
        LocalToolSpec(
            name="web_crawl",
            description=(
                "Crawl a website breadth-first from a start URL and return a "
                "bounded page list (URL, title, short excerpt per page) — "
                "follow up with web_fetch on pages that matter. Same-host "
                "only, SSRF-guarded, rate-limited (~1 page/sec), wall-clock "
                "capped (120s). Optional sitemap_url seeds the page list "
                "from a sitemap instead of link discovery (max_depth is "
                "ignored in that mode)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Start URL; its host defines the crawl scope."},
                    "max_pages": {"type": "integer", "default": CRAWL_DEFAULT_MAX_PAGES, "minimum": 1, "maximum": CRAWL_MAX_PAGES_CEILING, "description": "Fetch-attempt budget."},
                    "max_depth": {"type": "integer", "default": CRAWL_DEFAULT_MAX_DEPTH, "minimum": 1, "maximum": CRAWL_MAX_DEPTH_CEILING, "description": "Link depth from the start URL (start = 0)."},
                    "sitemap_url": {"type": "string", "description": "Optional sitemap.xml URL to seed pages from instead of link discovery."},
                },
                "required": ["url"],
            },
            handler=lambda args: web_crawl(
                args["url"],
                max_pages=args.get("max_pages", CRAWL_DEFAULT_MAX_PAGES),
                max_depth=args.get("max_depth", CRAWL_DEFAULT_MAX_DEPTH),
                sitemap_url=args.get("sitemap_url"),
            ),
            # network-classed: default ask from the permission store's global
            # default; read-only, so no risk tags.
            tags=(),
        ),
```

In `MCP/server.py` line 13, change the exposed-tools mention to `(`fs_*`, `fs_patch`, `git_*`, `web_fetch`, `web_search`, `web_crawl`)`.

- [ ] **Step 4: Run the provider suite + both web files**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_local_tool_provider.py Tests/Tools/test_web_crawl.py Tests/Tools/test_web_tool_impls.py -v -p no:randomly`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py tldw_chatbook/MCP/server.py Tests/Agents/test_local_tool_provider.py
git commit -m "feat: register web_crawl agent tool; web_fetch description covers PDFs (tasks 1357/1358)"
```

---

## Final verification (whole-branch)

- Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/ Tests/Agents/test_local_tool_provider.py Tests/Web_Scraping/test_web_fetch_wiring.py -v -p no:randomly` — expect green.
- Collection sweep: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/ --collect-only -q -p no:randomly 2>&1 | tail -5` — expect no new collection errors vs the base commit.
- Grep gate (ephemerality): `grep -n "Client_Media_DB\|ChaChaNotes\|Local_Ingestion" tldw_chatbook/Tools/web_tool_impls.py` — expect no output.
