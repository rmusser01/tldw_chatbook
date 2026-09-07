# robots.txt enforcement for web tools — design (task-2833)

- Date: 2026-08-07
- Backlog: task-2833 (Enforce robots.txt for web-tools)
- Owner directive: "finish the web-tools arc" — rulings below made by the controller and recorded for review; the spec's adversarial review gate replaces the interactive design Q&A.
- Reference: tldw_server2's `Web_Scraping/filters.py` `RobotsFilter` (stdlib `urllib.robotparser.RobotFileParser`, per-host TTL cache incl. negative caching, fail-open "compat" default). Chatbook has NO robots code today and NO `[webfetch]` config section — both are introduced here.

## Rulings (decisions an interactive brainstorm would have surfaced)

1. **Default ON.** `[webfetch] respect_robots_txt` defaults to `true` for both `web_fetch` and `web_crawl`. These are agent-driven, automated fetchers — exactly what robots.txt governs — and this arc's own spec called the 1 s/domain rate limiter "the politeness floor". The off switch exists for the user who needs it.
2. **Fail-open on unreachable robots.txt** (server "compat" semantics): a robots.txt that can't be fetched or parsed (network error, non-2xx, garbage) is cached as `None` = no restrictions, same TTL. A "strict" fail-closed mode is a recorded non-goal. Failure to fetch robots must be caught with a broad `except Exception` — in the shipped code this is what makes the feature backward-compatible (a route-less robots fetch in existing tests fails → fail-open → old behavior), and in production a robots outage must not brick fetching.
3. **Per-hop checks.** Robots is consulted at the same position as `_validate_hop`, for every hop of every content fetch — a redirect into a disallowed path is a disallowed fetch. Cache-hit path in `web_fetch` re-checks robots exactly like it re-runs `_validate_hop` (rules may have changed since the body was cached).
4. **Everything content-shaped is checked**: `web_fetch` hops, `web_crawl` BFS pages, sitemap XML fetches (root + children — the server checks them like any URL; a site that disallows `/sitemap-secret.xml` means it). The robots.txt fetch itself is exempt (never self-checked) and uses a dedicated byte cap, not the page cap.
5. **The robots.txt fetch is rate-limited** through the existing `_enforce_rate_limit` like any other request to the host — politeness applies to the politeness probe too. Cost: up to +1 s on the first fetch from a new domain while enforcement is on. Recorded, not hidden.
6. **User agent**: `can_fetch()` is called with the exact User-Agent string the tools already send (module constant; the implementation reads it from the existing request-header code, not a new literal). RobotFileParser's own specific-UA-then-`*` fallback does the rest.

## Mechanism

New module-level machinery in `Tools/web_tool_impls.py` (beside the existing cache idiom; NOT in `Utils/egress.py` — web_tool_impls deliberately owns its own guard stack pending task-609 consolidation):

- `ROBOTS_MAX_BYTES = 64 * 1024` (dedicated cap for the robots.txt body), `ROBOTS_CACHE_TTL_SECONDS = 1800.0` (server parity), `ROBOTS_CACHE_MAX_ENTRIES = 128`.
- `_robots_cache: dict[str, tuple[float, RobotFileParser | None]]` keyed by `scheme://netloc` — value `(expires_at_monotonic, parser_or_None)`; `None` = unreachable → allow (ruling 2). Earliest-expiry eviction at capacity (mirror `_cache_put`). Registered in `_reset_state_for_tests()`.
- `_robots_allows(client, url, ua) -> bool`: cache lookup → on miss, `_enforce_rate_limit(host)` + fetch `{scheme}://{netloc}/robots.txt` via `_fetch_once` with `ROBOTS_MAX_BYTES` (broad try/except → cache `None`) → `RobotFileParser.parse(text.splitlines())` → `can_fetch(ua, url)`. Synchronous, no locks — matches the module's idiom (single-threaded per tool call; a cross-call stampede costs one duplicate robots fetch, accepted). The UA is a PARAMETER (spec review, Important 3): the two tools send different agents (`_USER_AGENT` = tldw-chatbook-web-fetch/1.0 for web_fetch; `_CRAWL_USER_AGENT` = tldw-chatbook-web-crawl/1.0 for web_crawl and all sitemap fetches), and `can_fetch()` must be called with the requesting tool's actual string. The cached parser is shared per-host; only the query is per-UA.
- Injection sites, named (spec review, Minor 8): `web_fetch`'s inline redirect loop (web_tool_impls.py ~:454-463, beside `_validate_hop`) and `_crawl_fetch_page`'s hop loop (~:801-829) — the latter is also the path every sitemap fetch takes, which is how ruling 4 falls out for free. The constructed robots.txt URL itself goes through `_validate_hop` before being fetched (Minor 9 — symmetry with every other request).
- A robots.txt body that comes back TRUNCATED at `ROBOTS_MAX_BYTES` is treated as a fetch failure → cached `None` → fail-open (spec review, Minor 7): a half-file could silently drop trailing Disallow lines and parse permissively; refusing to trust a truncated policy is the honest reading.
- Gate read: `_webfetch_settings()` reader following the `_deep_search_settings()` idiom (local import of config helpers, strict `_bool` coercion — `"false"` string must disable; the deep-search gate's fail-open lesson applies here in the opposite direction: this flag defaults TRUE, and a `"true"` string must not read as disabled either). Read once per tool invocation, not per hop.

## Behavior

- `web_fetch`: a disallowed hop returns the structured refusal `[robots-disallowed] {url} — {host}/robots.txt disallows this path for this tool's user agent; set [webfetch] respect_robots_txt = false to override` (error contract style of `[ssrf]`/`[invalid-url]`). Cache hits re-check (ruling 3).
- `web_crawl`: a disallowed page/sitemap is SKIPPED, not fatal — the existing two-way exception dispatch (`[ssrf]` → blocked, else → failed) gains an explicit THIRD branch (`str(exc).startswith("[robots-disallowed]")` → `robots_disallowed` counter; spec review, Important 4), surfaced by `_format_crawl_result` as its own clause alongside failed/blocked. The crawl continues with allowed URLs. A disallowed SEED (start URL or root sitemap) flows through the existing unconditional wrap and therefore returns the DOUBLE-WRAPPED string `[crawl-failed] start URL could not be fetched: [robots-disallowed] …` (or `sitemap could not be fetched:` — amended per spec review, Critical 2: the wrap is how every other seed-failure type reads today; tests assert the `[robots-disallowed]` substring inside the `[crawl-failed]` message, not a bare prefix).
- Config template: new commented `[webfetch]` block in `CONFIG_TOML_CONTENT` with `# respect_robots_txt = true` and a comment stating scope (web_fetch + web_crawl), default, and the fail-open semantics.
- Tool descriptions (`local_tool_provider.py`): one clause added to web_fetch's and web_crawl's descriptions — "honors robots.txt (configurable)". Static text; accurate for the default.

## Testing (MockTransport route idiom; `_reset_state_for_tests` isolation)

**Existing-suite compatibility (spec review, Critical 1 — the original "backward compatible" claim was FALSE):** with the default ON, robots pre-fetches add transport-call entries and an extra `_enforce_rate_limit` hit that break at least seven existing exact-list/count/sleep assertions across `test_web_tool_impls.py` and `test_web_crawl.py` (reviewer traced them individually; e.g. `test_fetch_rate_limits_per_domain`'s `sleeps == []` becomes `[1.0]`). RULING: the shared `fetch_env`/`crawl_env` fixtures monkeypatch the `_webfetch_settings` seam to `respect_robots_txt = False` as part of their existing isolation duties, preserving every old test's semantics; robots tests opt in explicitly. This is a TEST-FIXTURE default only — the shipped config default remains true.

- Disallowed path refused (web_fetch) with the exact `[robots-disallowed]` prefix; allowed path proceeds.
- Specific-UA vs `*` precedence honored (fixture robots.txt with both groups). Fixture-authoring caveat (spec review, Minor 6): stdlib `RobotFileParser` is FIRST-MATCH-WINS in file order, not longest-path — `Disallow: /search` before `Allow: /search/public` refuses `/search/public`; write fixtures (and expected results) with that ordering rule in mind.
- Unreachable robots.txt (missing route / 500 / garbage) → fetch proceeds (fail-open) — and this is also the existing-suite-compat proof.
- Robots cache: second fetch to same host does NOT re-fetch robots.txt (route hit counter); TTL expiry re-fetches (fake clock); negative cache holds for TTL.
- Redirect into a disallowed path refused mid-chain.
- web_crawl: disallowed pages skipped + counted in footer; disallowed child sitemap skipped; disallowed seed → structured refusal.
- Toggle off (`respect_robots_txt = false`, monkeypatched settings seam) → no robots fetch at all (route counter zero).
- Cache-hit re-check: cached body + newly-disallowing robots → refusal on the cached path.
- Mutation checks at implementation time: disable the `can_fetch` consult → allowed/disallowed tests discriminate; drop the footer count → crawl-count test red.

## Non-goals (recorded)

Strict fail-closed mode; per-call `respect_robots` argument (config-only — keeps the `_fetch_cache` key robots-free; a per-call override would have to join the cache key, the exact poisoning class PR #1376's review caught); Crawl-delay/Sitemap directive honoring; robots enforcement for `web_search` provider APIs (fixed HTTPS endpoints — robots is inapplicable) or `web_deep_search`'s scrape path — recorded HONESTLY (spec review, Important 5): this leaves an inconsistency where a path that web_fetch/web_crawl refuse can still be scraped by web_deep_search, and the codebase's own precedent (`is_public_http_url` was bolted onto `scrape_article` for exactly this kind of parity) shows the right fix is a follow-up patch on that path, not an architectural exemption — follow-up task to be filed at implementation; consolidating with `Utils/egress.py` (task-609).
