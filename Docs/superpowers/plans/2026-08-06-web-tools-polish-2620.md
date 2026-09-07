# Web-tools polish (task-2620) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close backlog task-2620 — every deferred/parked finding from the merged web-tools v2 branch (PR #1376), each fixed or explicitly re-ruled.

**Architecture:** All code changes stay in `tldw_chatbook/Tools/web_tool_impls.py` + its two test files; one docstring touch in `Agents/local_tool_provider.py`. No new behavior surfaces — this is correctness/honesty/coverage polish on shipped behavior.

**Tech Stack:** Same as parent plan (`Docs/superpowers/plans/2026-08-06-web-crawl-pdf-fetch.md`): httpx MockTransport seam, fake clock via module-level `time`, venv pytest.

## Global Constraints

- Use the module-level `time` import exclusively; tests monkeypatch `web_tool_impls.time`.
- No new dependencies; no DB imports in `web_tool_impls.py`.
- Error strings that change must stay structured (`[reason] …`); every changed behavior gets a RED-first test; pure-coverage tests may be GREEN-on-arrival if stated.
- Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_web_crawl.py Tests/Tools/test_web_tool_impls.py -v -p no:randomly` (foreground; venv only). Never `git stash`.
- The spec (`Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md`) is updated in the same commit as any behavior it pins (stop reasons, error copy).

---

### Task 1: Behavioral fixes (task-2620 items 1, 2, 4, 5, 8, 9 + minors)

**Files:**
- Modify: `tldw_chatbook/Tools/web_tool_impls.py`
- Modify: `Docs/superpowers/specs/2026-08-06-web-crawl-pdf-fetch-design.md` (stop-reason + error-copy pins)
- Test: `Tests/Tools/test_web_crawl.py`, `Tests/Tools/test_web_tool_impls.py`

**Interfaces:**
- Consumes: everything shipped in PR #1376 (current dev tip).
- Produces: `_seed_from_sitemap(...) -> tuple[list[str], bool]` (urls, children_capped); new stop reason string `"sitemap child budget reached"`; `_pymupdf_available() -> bool` module helper.

Nine sub-items, each RED-first unless noted:

**(a) Sitemap parse failures of ANY kind → structured.** `_parse_sitemap`'s except becomes `except (xET.ParseError, ValueError) as exc:` — defusedxml's refusals (`EntitiesForbidden` etc.) subclass `ValueError`, not `ParseError`, and today escape as raw exceptions (root) or abort the crawl (child; the `except LocalToolError: continue` was written for exactly that case and now heals automatically). Tests (skipif defusedxml absent, since the stdlib fallback parses internal entities without complaint):

```python
_ENTITY_SITEMAP = (
    b'<?xml version="1.0"?><!DOCTYPE urlset [<!ENTITY x "y">]>'
    b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    b"<url><loc>http://example.com/&x;</loc></url></urlset>"
)

@requires_defusedxml
def test_sitemap_entity_declaration_root_is_crawl_failed(crawl_env):
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(_ENTITY_SITEMAP)
    with pytest.raises(LocalToolError, match="crawl-failed"):
        web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")

@requires_defusedxml
def test_sitemap_entity_declaration_child_is_skipped(crawl_env):
    index = (b'<?xml version="1.0"?>'
             b'<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
             b"<sitemap><loc>http://example.com/bad.xml</loc></sitemap>"
             b"<sitemap><loc>http://example.com/good.xml</loc></sitemap>"
             b"</sitemapindex>")
    good = (b'<?xml version="1.0"?>'
            b'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
            b"<url><loc>http://example.com/page</loc></url></urlset>")
    crawl_env.routes["http://example.com/sitemap.xml"] = _sitemap_response(index)
    crawl_env.routes["http://example.com/bad.xml"] = _sitemap_response(_ENTITY_SITEMAP)
    crawl_env.routes["http://example.com/good.xml"] = _sitemap_response(good)
    _site(crawl_env, {"http://example.com/page": ("still works", [])})
    out = web_crawl("http://example.com/", sitemap_url="http://example.com/sitemap.xml")
    assert "still works" in out
```

with `requires_defusedxml = pytest.mark.skipif(...)` following the file's `requires_pymupdf` pattern (guarded `try: import defusedxml`).

**(b) Child-cap honesty.** `_seed_from_sitemap` returns `(urls, children_capped)`; `children_capped = True` when the `SITEMAP_MAX_CHILDREN` break fires. `web_crawl`'s sitemap branch:

```python
seeded, children_capped = _seed_from_sitemap(...)
...
if time.monotonic() >= deadline:
    stop_reason = "deadline reached"
elif children_capped:
    stop_reason = "sitemap child budget reached"
else:
    stop_reason = "sitemap exhausted"
```

Spec §2 gains the third sitemap stop reason. Test: monkeypatch `SITEMAP_MAX_CHILDREN` to 1, index with 2 same-host children (both empty urlsets) → footer ends `Stopped: sitemap child budget reached.` (RED: currently "sitemap exhausted").

**(c) Unspoofable blocked-vs-failed.** `web_crawl`'s classifier becomes `if str(exc).startswith("[ssrf]"):` — `_validate_hop` puts the reason at position 0; interpolated URLs cannot be. Test: same-host link `/x[ssrf]y` that 404s → footer counts `1 failed, 0 blocked` (RED: currently counted blocked).

**(d) html_only aborts sniffed PDFs too.** In `_fetch_once`, the early-break becomes:

```python
if html_only and is_pdf is not None and (is_pdf or (declared and declared not in _HTML_TYPES)):
    break  # crawl only needs the type; don't drain the body
```

(A declared-empty non-PDF body still reads on for the later `<html` sniff; genuine HTML never breaks.) Test: crawl route serving `text/html` whose body iterator yields `%PDF-` then RAISES if pulled more than 4 chunks — crawl must complete (abort happened) and list `[application/pdf]`.

**(e) `[too-large]` copy derives from the constant.** `f"[too-large] PDF exceeds {PDF_MAX_BYTES // (1024 * 1024)} MB — use media ingestion for large documents"`. Update `test_fetch_pdf_over_ceiling_refused` to match `r"too-large.*media ingestion"` only (it monkeypatches the constant, so the number must not be pinned). Spec §1's exact-copy line gains "(the number renders from `PDF_MAX_BYTES`)".

**(f) No 20 MB download when pymupdf is absent.** Add:

```python
def _pymupdf_available() -> bool:
    """Cheap availability probe (no import): the 20 MB PDF read ceiling and
    the [missing-dep] refusal must be decided before downloading, and
    optional_deps.check_dependency() eagerly imports the module — wrong cost
    for the fetch hot path."""
    return importlib.util.find_spec("pymupdf") is not None
```

(`import importlib.util` at top.) In `web_fetch`: pass `pdf_max_bytes=PDF_MAX_BYTES if _pymupdf_available() else None`, and in the `if is_pdf:` branch raise the existing `[missing-dep]` copy BEFORE the `[too-large]` check when `not _pymupdf_available()`. `_extract_pdf_text` keeps its own import guard (belt and braces; the existing `builtins.__import__` test still passes through it). New test: monkeypatch `web_tool_impls._pymupdf_available` to `lambda: False`, serve a `%PDF-` body larger than the caller cap via a guard-iterator that fails past the cap → `[missing-dep]` raised, iterator guard never tripped.

**(g) Title accumulation bounded.** `_CrawlLinkParser.handle_data`: `self.title = (self.title + data)[:512]`. Test: unclosed `<title>` followed by 100 KB of text → `len(parser.title) <= 512`.

**(h) Namespace-less sitemaps parse.** In `_parse_sitemap`: fall back to un-namespaced `.//loc` when the namespaced findall is empty, and detect the index via `root.tag.rsplit("}", 1)[-1] == "sitemapindex"`. Test: namespace-free `<urlset><url><loc>…</loc></url></urlset>` seeds pages (RED: currently zero pages, "sitemap exhausted").

**(i) Redirect-duplicate targets listed once.** In `web_crawl` after a successful fetch:

```python
final_norm = _normalize_crawl_url(final_url)
if final_norm in visited and final_norm != _normalize_crawl_url(current):
    continue  # a previously-fetched URL already redirected here
visited.add(final_norm)
```

(The start page has `final == current`, so it always lists.) The duplicate still consumed its attempt slot — that is the budget contract. Test: `/one` and `/two` both 302 → `/target` → `/target` listed once, footer `Crawled 2 pages` (RED: currently 3 with a duplicate row).

Steps: write all failing tests → verify each RED for its stated reason → implement → both files green → commit `fix: web-tools polish — sitemap refusal contract, honest stop reasons, sniff-abort, dedup (task-2620)`.

---

### Task 2: Coverage + docs + closure (items 3, 7, remaining minors, rulings)

**Files:**
- Modify: `Tests/Tools/test_web_crawl.py`, `Tests/Tools/test_web_tool_impls.py`
- Modify: `tldw_chatbook/Tools/web_tool_impls.py` (docstrings only), `tldw_chatbook/Agents/local_tool_provider.py` (module docstring line ~63)
- Modify: `backlog/tasks/task-2620 - Web-crawl-PDF-fetch-deferred-review-findings.md`

**Interfaces:** Consumes Task 1's final state. Produces nothing new — assertions and prose.

**(a) Ephemerality static guard** (pins task-1358's headline claim):

```python
def test_module_never_imports_persistence():
    import inspect
    import re
    from tldw_chatbook.Tools import web_tool_impls
    src = inspect.getsource(web_tool_impls)
    assert re.search(r"Client_Media_DB|ChaChaNotes|Local_Ingestion|RAG_Indexing|sqlite3", src) is None
```

**(b) Frontier-cap test pins both directions:** change `assert len(crawl_env.calls) <= 6` to `== 6` in `test_crawl_caps_links_enqueued_per_page` (verify the exact expected count from the fixture first; adjust the literal to the true value, not to make it pass).

**(c) Between-hops deadline coverage:** a page whose redirect handler advances the fake clock past `CRAWL_DEADLINE_SECONDS` mid-chain → crawl stops `deadline reached`, the redirect target is never fetched (exercises `_CrawlDeadline` raise/catch — currently zero coverage).

**(d) Docstring truth:** `web_fetch`'s docstring gains the PDF branch (detection, 20 MB refusal, extracted-text truncation), the `missing-dep`/`pdf-error`/`too-large` reasons, and the `(url, max_bytes)`/256-entry cache facts. `local_tool_provider.py`'s module docstring tool list (~line 63) gains `web_crawl`.

**(e) Close task-2620:** tick all three ACs; Implementation Notes summarizing fixes + these explicit won't-fix rulings:
- **optional_deps centralization (Qodo rule 497159):** availability probing uses `find_spec` because `optional_deps.check_dependency()` *imports the module eagerly* — the wrong cost profile for a hot fetch path; the heavy imports stay local per the module's v1 trafilatura precedent. WON'T-FIX beyond Task 1(f).
- **Rate-limit bucket www-folding:** v1 `web_fetch` parity; diverging the two tools' buckets is worse than the 2-req/s worst case. WON'T-FIX (was never in the task's AC; recorded for completeness).

Steps: tests first (a–c; (a) and (c) may be GREEN-on-arrival — state it), then docs (d), then closure (e) → both files green → commit `test/docs: web-tools polish coverage + docstring truth; close task-2620`.
