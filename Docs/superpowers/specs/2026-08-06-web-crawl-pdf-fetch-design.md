# Web-tools v2: `web_crawl` + PDF fetch — design

- Date: 2026-08-06
- Backlog: task-1357 (recursive web crawling as a tool), task-1358 (PDF fetch without permanent ingestion)
- Builds on: `Docs/superpowers/specs/2026-08-05-web-search-fetch-tools-design.md` (v1, merged via PR #1363) — its egress guard, caps, rate limiter, cache, and registration surfaces are the foundation; nothing in v1 is redesigned here.

## Owner rulings (brainstorm 2026-08-06)

1. `web_crawl` returns a **page list with a short excerpt per page** (URL + title + ~200-char excerpt), not full content. The model follows up with `web_fetch` on pages it cares about.
2. The crawler is a **new lightweight httpx BFS built on the v1 core**, plus a **sitemap mode** — NOT a wrapper around `Article_Extractor_Lib`'s Playwright crawlers. Task-1357's "wrap them" premise predates v1's merged design and is overridden: those crawlers launch Chromium per call, write a `scrape_progress.json` resume file into the cwd, use the older egress guard, and their link collector has no page budget. JS-rendered sites therefore won't crawl — same posture as v1's light fetch path.
3. PDF support lives **inside `web_fetch`** (content-type/magic detection), not a separate tool. The model never needs to know the content type in advance.
4. PDFs get their **own 20 MB download ceiling** (refuse above — a byte-truncated PDF is unparseable, so the cap is a refusal threshold, never a truncation). HTML/text keeps the existing 1 MiB default / 5 MiB ceiling untouched. Truncation for PDFs applies to the **extracted text**, satisfying task-1358's "truncated like HTML" AC.

## §1 PDF support in `web_fetch` (task-1358)

All changes are in `tldw_chatbook/Tools/web_tool_impls.py`; no new module.

**Detection.** A response is a PDF iff the content-type main type is `application/pdf` **or** the body starts with `%PDF-` (no valid HTML/text starts with those bytes; servers mislabel PDFs as `application/octet-stream`, `text/html`, or nothing constantly, so the sniff wins over the declared type). The sniff buffers until at least 5 body bytes have arrived before deciding — a server dribbling one byte per chunk must not defeat it.

**Mid-stream cap selection.** The read cap is decided while the stream is open, because the caller's `max_bytes` (default 1 MiB) is chosen before the content type is known and a PDF cut at 1 MiB is garbage:

- `_fetch_once` reads the first chunk, checks content-type header + `%PDF-` prefix.
- PDF → read ceiling becomes `PDF_MAX_BYTES = 20 * 1024 * 1024` for this response, regardless of the caller's `max_bytes`. One request; no re-fetch, no second rate-limit hit.
- Not PDF → the caller's capped `max_bytes` applies exactly as today.
- A PDF whose body exceeds 20 MB → `LocalToolError("[too-large] PDF exceeds 20 MB — use media ingestion for large documents")`. Never returned truncated.

**Extraction (ephemeral).** `pymupdf.open(stream=body, filetype="pdf")` → per-page `page.get_text()` joined with blank lines. Nothing touches disk; nothing writes to the media DB. Import is local to the function with try/except, matching v1's trafilatura pattern.

- **Early stop:** the page loop stops as soon as accumulated text exceeds the caller's effective `max_bytes` — a 20 MB / several-thousand-page PDF must not burn seconds extracting text that is about to be thrown away. Truncation marker records progress: `[... truncated: extracted text exceeded max_bytes=N; processed P of T pages ...]` (byte-capped via the existing `_truncate_to_bytes` helper, never splitting a codepoint).
- Encrypted: if `doc.is_encrypted`, try `doc.authenticate("")`; on failure → `[pdf-error] PDF is encrypted`.
- Damaged/unopenable → `[pdf-error] could not parse PDF: <reason>`.
- Opens fine but zero extractable text (scanned images) → `[empty-content] PDF contains no extractable text (scanned document?) — use media ingestion with OCR`.
- pymupdf not installed → `[missing-dep] PDF support requires pymupdf — pip install tldw_chatbook[pdf]`.

**Cache + rate limiting** are inherited: the cached value is the extracted text, keyed like every other fetch. Two cache fixes ride along, both motivated by this work:

- **Key fix (v1 quirk):** the cache is currently keyed by URL alone, so a fetch at `max_bytes=100` poisons a later full-cap call with 100 bytes of text. Key becomes `(url, effective_max_bytes)`.
- **Entry cap:** the cache is unbounded and swept only on access, which was tolerable when entries trickled in one fetch at a time — a 40-page crawl bulk-loads it. Cap at `FETCH_CACHE_MAX_ENTRIES = 256`, evicting the entry with the earliest expiry when full.

## §2 `web_crawl` tool (task-1357)

New sync core `web_crawl(...)` in `web_tool_impls.py`, reusing `_validate_hop` (egress guard), `_enforce_rate_limit`, `_fetch_once`, `_extract_text`, and the module transport/test seam. No Playwright, no resume files, no `Article_Extractor_Lib` import, no new dependencies (link extraction via stdlib `html.parser.HTMLParser`).

**Parameters.**

| param | type | default | ceiling | meaning |
|---|---|---|---|---|
| `url` | str, required | — | — | start URL; defines the host scope in both modes |
| `max_pages` | int | 20 | 40 | budget of **fetch attempts** (see semantics) |
| `max_depth` | int | 2 | 5 | BFS depth; start URL is depth 0. Ignored in sitemap mode (documented in the tool description) |
| `sitemap_url` | str | none | — | when given, the queue is seeded from the sitemap instead of BFS link discovery |

Out-of-range budgets are clamped into `[1, ceiling]`; garbage values coerce to the default — the v1 argument-handling style (`web_fetch` clamps `max_bytes`, `web_search` coerces `result_count`), not an error.

Internal constants (not parameters — YAGNI): `CRAWL_DEADLINE_SECONDS = 120.0` wall-clock bound, `CRAWL_PAGE_TIMEOUT_SECONDS = 10.0` per-page HTTP timeout (a single hung page must not eat the deadline; `web_fetch`'s own 30 s stays as-is), `CRAWL_EXCERPT_MAX_CHARS = 200`, per-page read cap = `FETCH_MAX_BYTES` (1 MiB), `CRAWL_RESULT_MAX_BYTES = 24 * 1024` total / `CRAWL_BLOCK_MAX_BYTES = 1024` per page block (byte caps in the `web_search` style, same truncate helper, omission marker when the total cap cuts the list).

**BFS mode.**

1. Queue starts at `(url, depth 0)`; visited set holds normalized URLs (fragment stripped, host lowercased **and leading `www.` folded — for dedup, not just scope**, query kept, non-http(s) schemes dropped).
2. Per dequeued URL: deadline check → egress guard (`_validate_hop`) → per-domain rate limit → GET via `_fetch_once` semantics with per-hop redirect validation exactly like `web_fetch`; the deadline is also checked **between redirect hops** (otherwise one page's full redirect chain could overshoot the deadline by up to 6 × page timeout). Both the requested and the final URL are marked visited. Requests identify as `tldw-chatbook-web-crawl/1.0` so site operators can distinguish crawls from single fetches.
3. HTML responses: extract `<title>` (regex, entities unescaped), full text via `_extract_text` (trafilatura with tag-strip fallback), excerpt = first 200 chars; collect `<a href>` links via `HTMLParser`, resolve with `urljoin` **honoring `<base href>` when present** (ignoring it resolves relative links against the wrong origin), keep same-host only, enqueue at depth+1 if depth allows.
4. Non-HTML responses (PDF, images, …): the body read is aborted after the first chunk; the URL is listed with a type marker (e.g. `[application/pdf]`) and no excerpt; its links are (trivially) not expanded.
5. A page whose redirects land **off-host** is listed at its final URL but its links are **not** expanded (host scope: exact host with a leading `www.` folded on both sides; registrable-domain matching needs a public-suffix list — out of scope).
6. Stop when the queue empties, `max_pages` fetch attempts are spent, or the deadline passes.

**Budget semantics.** `max_pages` counts **fetch attempts** — discovering that a link is a PDF, an error, or blocked costs a request, so attempts are what bound total traffic. Failed fetches are omitted from the list but counted; guard-blocked URLs likewise (never fetched, still an attempt slot — a hostile page must not use link spam to make the crawler probe unboundedly).

**Sitemap mode.** `sitemap_url` is egress-guarded and fetched (cap `SITEMAP_MAX_BYTES = 5 MiB`); `<loc>` entries parsed from a `urlset`, or — for a `sitemapindex` — one level of child sitemaps is fetched (each guarded, within the deadline) until `max_pages` URLs are collected. Host rules, precisely: `sitemap_url` itself may live on any public host (sitemaps sometimes sit on a CDN); **child sitemaps must share `sitemap_url`'s host** (that's where indexes point); **page URLs must share `url`'s host** — `url` defines the crawl scope in both modes. Sitemap fetches themselves are discovery overhead and do **not** consume `max_pages`; the deadline bounds a pathological index (e.g. thousands of empty child sitemaps). Discovered URLs are then processed exactly like BFS step 2–4 **except no link expansion** — the sitemap *is* the discovery. Every URL still passes the same-host filter and the egress guard individually: a sitemap listing private/internal addresses must not become an SSRF path.

**Cache interplay.** The crawler **writes** the fetch cache (full extracted text per HTML page, keyed `(url, FETCH_MAX_BYTES)`) so the model's follow-up `web_fetch` of a listed page is instant. It does **not read** the cache — cached text has no link structure — so a re-crawl inside the TTL honestly re-fetches.

**Output.** Numbered blocks in the `web_search` style:

```
1. Page Title
   URL: https://example.com/docs/intro
   First two hundred characters of extracted text…

2. [application/pdf]
   URL: https://example.com/manual.pdf

Crawled 12 pages (2 failed, 1 blocked). Stopped: page budget reached.
```

The status footer is always present and states the stop reason (`page budget reached` / `no more links within depth` / `deadline reached` / `sitemap exhausted`).

## §3 Registration

- **Agent tool** (`Agents/local_tool_provider.py`): `web_crawl` `LocalToolSpec` beside `web_fetch`/`web_search` — network-classed, `tags=()`, read-only; the Ask default comes from the permission store's global default, satisfying task-1357's "Ask default" AC by the same mechanism v1 uses. Schema: `url` required; `max_pages`, `max_depth`, `sitemap_url` optional with the defaults/ceilings above.
- **Tool descriptions are part of the contract.** `web_fetch`'s agent-tool description currently implies text-only ("Fetch a web page and return its extracted text") — the model reads that and won't try PDF links. It gains "; PDFs are text-extracted (≤20 MB, ephemeral)". `web_crawl`'s description states the page-list-plus-excerpt contract, the budgets, and that `max_depth` is ignored in sitemap mode.
- **MCP: no server change at all.** Verified against shipped code (this differs from v1's spec text, which described per-tool nested registration): `MCP/server.py::_register_local_agent_tools()` exposes local agent tools *generically*, opt-in behind `[mcp] expose_local_tools` (default false), routed through `LocalToolProvider`'s permission gate — and ask-state tools **fail closed** for external callers (no approval card exists outside the Console; an operator grants a tool externally via Console "Always allow" or by editing `mcp_permissions.json`). Registering the `LocalToolSpec` is therefore the *only* registration work; `web_crawl` inherits that exposure and that fail-closed external posture automatically, as does `web_fetch`'s PDF support.
- **No new config.** Budgets and ceilings are module constants like v1's `FETCH_*` (verified: no `[webfetch]` config section shipped either); a config surface can come later if anyone asks.

## §4 Error contract

All failures follow v1: `LocalToolError` with a structured bracket reason at the agent layer, `{"error": ...}` at MCP. New reasons: `[too-large]` (PDF over ceiling), `[pdf-error]`, `[missing-dep]`, plus crawl-level `[invalid-args]` (bad url/budgets) and `[crawl-failed]` (start URL unfetchable — per-page failures inside a crawl are *results*, reported in the footer, not exceptions). Existing reasons (`[ssrf]`, `[invalid-url]`, `[rate-limited]`, `[http-<status>]`, `[timeout]`, `[empty-content]`) are reused unchanged.

## §5 Testing

Same seams as v1 (`Tests/Tools/test_web_tool_impls.py` conventions: `httpx.MockTransport` via the module `_transport` hook, `_reset_state_for_tests`, hermetic DNS by using literal-public-IP or patched resolution).

- **PDF:** fixture PDFs generated in-test with pymupdf (valid multi-page; encrypted; image-only/textless) plus handcrafted bytes (damaged; `%PDF-` body under `text/html` and `application/octet-stream` mislabels; oversized body → refusal; mid-stream cap raise actually reads past 1 MiB on a >1 MiB fixture). Extracted-text truncation marker byte-exact, early page-loop stop asserted (pages-processed count in the marker). `[missing-dep]` path via import patching. Ephemerality: no media-DB import anywhere in the module (static assertion), no temp files (tmp-path watch).
- **Crawl:** a fixture site as a MockTransport routing table — link graph with depth >5, off-host links, `www.`-variant links (deduped, not double-fetched), a `<base href>` page (relative links resolve against the base), private-address links (guard refusal counted as blocked), a redirect-to-off-host page (listed, not expanded), a redirect-into-private-space page (blocked), non-HTML nodes (listed with marker, budget consumed), a hanging route (deadline stop, monkeypatched clock), page/depth budget stops, visited dedup across fragments, the crawl User-Agent asserted on requests, cache warm-write asserted (`web_fetch` after crawl hits cache without a new request) and entry-cap eviction asserted, sitemap `urlset` + one-level `sitemapindex`, sitemap page URL off-host filtered, footer text exact per stop reason, total/block byte caps.
- **Registration:** schema/handler wiring for `web_crawl` in the provider spec list, and the two updated tool descriptions asserted. No MCP-side test: exposure is the existing generic `_register_local_agent_tools` path, already covered.

## §6 Non-goals

- robots.txt honoring — task-1361, unchanged; the 1 s/domain rate limiter is the politeness floor here.
- Binary (non-PDF) fetch — task-1359. The PDF magic-sniff is deliberately not generalized.
- Response-cache redesign — task-1360 (the `(url, cap)` key fix in §1 is a correctness fix, not the redesign).
- Playwright/JS-rendered crawling, registrable-domain host scoping, OCR for scanned PDFs, crawl resume/incremental state, per-crawl config surface.
- Any write to any database. Both features are ephemeral by construction; media ingestion remains the path for keeping documents.
