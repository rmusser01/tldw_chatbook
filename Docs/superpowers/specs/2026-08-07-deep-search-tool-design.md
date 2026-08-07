# Deep-search tool: port + expose `generate_and_search` — design (task-1356)

- Date: 2026-08-07
- Backlog: task-1356 (Expose LLM-summarized deep search as an opt-in tool)
- Reference codebase (owner-directed): `tldw_server2` at `/Users/macbook-dev/Documents/GitHub/tldw_server2` — its **live** module `tldw_Server_API/app/core/Web_Scraping/WebSearch_APIs.py` is a hardened superset of chatbook's pipeline (its `WebSearch/Web_Search.py` is labeled legacy by its own README and is NOT the reference).
- Context: chatbook's `Web_Scraping/WebSearch_APIs.py` holds the full pipeline (sub-query generation → fan-out search → per-result relevance + scrape + summarize → LLM synthesis) reachable today only through `Research_Interop/local_research_search_service.py`, which no UI or tool invokes — wired but dead. The local pipeline is the *older* fork with real defects: three different final-answer key names across return branches (`Report`/`Report`/`summary`), hardcoded confidence (0.9/0.0), one unbounded synthesis prompt, no timeouts anywhere, a blocking `input()` review path, and an LLM-failure fallback that re-searches the original question twice.

## Owner rulings (brainstorm 2026-08-07)

1. **Port the server's hardening first**, then expose the tool on top — the tool ships production-grade rather than wrapping known defects.
2. **Double opt-in**: a `[tools]` config boolean (default OFF — the tool is absent from the catalog, Console AND MCP, until enabled) plus the permission store's Ask default per call once registered.
3. **Revive the dead config**: TOML section `[SearchSettings]` → loader dict `search_settings_general` (`config.py:1130`, `:2039-2088`) is already parsed with defaults but has zero consumers — it becomes the tool's default surface, gaining the new timeout/cap keys beside the existing ones.

## §1 The port (phase-2 hardening in `Web_Scraping/WebSearch_APIs.py`, source: tldw_server2's live module)

Ported, with chatbook adaptations:

- **`FinalAnswerDict` typed contract** `{text: str, evidence: list[dict], confidence: float, chunks: list[dict]}` — `aggregate_results` rewritten to return it on ALL branches (replacing `Report`/`Report`/`summary`). Callers checked: the only consumer is `local_research_search_service.py`'s passthrough, and the client schema (`tldw_api/research_search_schemas.py` `WebSearchAggregateResponse.final_answer`) already models `{text, evidence, confidence, chunks}` — the port *aligns* local output with the schema that exists.
- **Chunked map-reduce aggregation**: `_build_chunk_infos(items, max_chars=6000)` greedy packing, per-chunk summarization with per-chunk try/except degradation (failed chunks fall back to truncated raw text, `failed_chunks` counted), then one synthesis call over the chunk summaries — replacing the unbounded single prompt.
- **Computed confidence**: `_estimate_confidence(relevant_count, chunk_count, failed_chunks, has_llm)` (server's formula verbatim; clamped [0.1, 0.99], 0.0 only for zero relevant results) — replacing hardcoded 0.9.
- **`_sanitize_sub_questions`**: normalize str/dict items, case-insensitive dedup, drop sub-queries equal to the original question; LLM-failure fallback becomes `[]` (NOT `[original_query]` — chatbook's current fallback searches the original question twice).
- **Timeouts**: `asyncio.wait_for` on the relevance LLM call and the scrape, from the revived config (`relevance_llm_timeout_s`, `relevance_scrape_timeout_s`, defaults 30/30).
- **Scrape-failure fallback**: `_build_result_fallback_content(result)` synthesizes Title/Snippet/URL content so a failed or unavailable scrape keeps the result instead of dropping it. Side benefit worth stating: the tool degrades gracefully when Playwright (websearch extra) is absent — snippets-only analysis instead of failure.
- **Non-interactive review**: `review_and_select_results` replaced with the server's pure version (no `selector` → all candidates pass; optional callable filter). The blocking `input()` is deleted.
- **Provider warnings surfaced**: per-query provider errors accumulate into `web_search_results_dict["warnings"]` (and `error` when zero results) instead of silent `continue`.
- **`cancel_event` support** (server parity): threaded through `analyze_and_aggregate`/`search_result_relevance`; the tool's overall deadline sets it, so timeout means *partial synthesis over what was analyzed* with an honest count — never discarded paid work.
- **Citation integrity (beyond the server — both codebases share this gap):** evidence renumbered **1..N over the relevant results**, the synthesis prompt shown a clean numbered list (`[1] Title — content…`), and each evidence entry extended with `id`, `url`, and `title` (captured at relevance time; the server's evidence has `id` but NOT url/title, which a Sources rendering requires). The prompt's `[N]` instruction finally references numbers that exist.

**Deliberately NOT ported** (recorded): the per-provider circuit breaker (`Infrastructure.circuit_breaker` has no chatbook analogue), `_enforce_provider_outbound_policy` (chatbook's search-provider calls are fixed HTTPS endpoints per the task-1355 ruling; the *scrape* path guard is §4), the server's extra engines (firecrawl/boards/sogou/etc.), and Bing's deprecation.

**Backend HTTP timeouts (defect found in review):** NO search backend sets `timeout=` on its HTTP call — including serper/exa/yandex shipped in task-1355 — so one unresponsive API hangs the fan-out forever. This spec adds `timeout=30` to the three backends this series owns (serper, exa, yandex). The seven older engines' missing timeouts are filed as a follow-up task at implementation time, not silently changed here.

## §2 The tool

`web_deep_search(question: str, engine: Optional[str] = None, max_results: Optional[int] = None) -> str` — sync core in `Tools/web_tool_impls.py` beside its siblings.

- **Loop-safe async runner** (crash risk found in review): phase 2 is async, and the handler runs from two contexts — the agent worker thread (no loop) and the MCP path (loop presence unproven). The core uses `asyncio.run` when `asyncio.get_running_loop()` raises, otherwise executes the coroutine in a dedicated thread with its own loop and joins. A test covers the running-loop case explicitly.
- **Overall deadline**: `deep_search_timeout_s` (default 300) — checked between phases; on expiry mid-relevance, sets `cancel_event` and synthesizes from partial analysis. Phase-1 sync searches are bounded by the new per-request timeouts; the residual "an in-flight search may overrun the deadline by up to one request timeout" is documented, not hidden.
- **Output** (byte-capped, house truncation helper):

```
<synthesized answer with [N] citations>

Sources:
[1] Page Title — https://…
[2] …

Confidence: 0.78 · Engine: google · Sub-queries: 3 · Analyzed 8 of 10 results (2 scrape fallbacks)
```

  The footer is always present and states partial/timeout/warning facts (provider errors, deadline hits, fallback counts). Zero relevant results → an explicit statement listing the sub-queries tried. Per-phase LLM failures → structured `[deep-search-failed] <phase>: <reason>`.
- **Cost transparency**: the tool DESCRIPTION states the spend shape — "makes ~2×results+3 LLM calls and up to `max_results` page fetches per invocation (≈25 LLM calls at defaults); every call costs real money on paid providers". The Ask approval is informed, not blind.
- **Params**: `question` required; `engine` optional (validated against the wired engines, default from revived config `default_search_provider`); `max_results` optional, clamped to `[1, search_result_max]`.
- **Registration (double opt-in)**: the `LocalToolSpec` is appended to the provider's spec list ONLY when `get_cli_setting("tools", "web_deep_search_enabled", False)` is true — absent from Console catalog and MCP exposure otherwise. Once registered: network-classed `tags=()` → permission store Ask default per call. Enabling requires an app restart (provider builds specs at construction) — documented in the config comment and the tool docs.

## §3 Config (the revival)

`[SearchSettings]` (TOML) → `search_settings_general` (loader — already parsed). Existing keys wired as tool defaults: `search_provider_default` (engine), `relevance_analysis_llm`, `final_answer_llm`, `search_enable_subquery` (+ the cap: generated sub-queries are truncated to `search_default_max_queries - 1`, so total fan-out queries ≤ `search_default_max_queries`, default 5), `search_result_max` (cap for `max_results`, default 10). New keys beside them: `relevance_llm_timeout_s = 30`, `relevance_scrape_timeout_s = 30`, `deep_search_timeout_s = 300`. `[tools] web_deep_search_enabled = false` in the tools section. The config template's commented-out `[search_settings]` placeholder is replaced by a real, commented `[SearchSettings]` block. A Settings-UI toggle row is a recorded non-goal (that UI derives from `_GATEABLE_BUILTINS`, a different registry).

## §4 Security & resource posture

- Search-provider calls: fixed HTTPS endpoints, no egress guard needed (1355 precedent), now with per-request timeouts.
- **Relevance-phase scrapes fetch arbitrary result URLs** — the plan must VERIFY `scrape_article`'s Playwright navigation path enforces the egress guard (`Utils/egress.py` is imported by `Article_Extractor_Lib`; coverage of the browser path must be proven, and a pre-scrape `validate`/guard check added in the pipeline if it is not inherent). This is a blocking plan-time verification, not an assumption.
- LLM keys resolve via `chat_api_call`'s existing provider resolution; never logged. First invocation may pay a lazy-import latency (`Summarization_General_Lib` pulls heavy deps) — documented.

## §5 Testing (all LLM calls mocked — per the AC)

Established seam: fake ONLY `chat_api_call` (module attribute) + `scrape_article`, run the real pipeline (`Tests/Web_Scraping/test_websearch_internal_prompts.py` precedent; `Summarization_General_Lib.analyze` patched at its source module for the summarize path).

- Ported-layer unit tests: chunk packing (oversize entry split), per-chunk failure degradation + `failed_chunks` in confidence, confidence formula values, sub-question sanitization/dedup/empty-fallback, non-interactive review, scrape-fallback content, warnings accumulation, evidence carries id/url/title with 1..N numbering matching the prompt payload, `FinalAnswerDict` shape on ALL branches (success/empty/LLM-failure).
- Tool-level: full mocked run → answer + numbered Sources + footer counts; zero-relevant honesty; per-phase failure errors; deadline → partial synthesis with honest footer (fake clock); the running-event-loop runner case; gate absent-by-default (catalog excludes the tool) and present-when-enabled (monkeypatched setting); byte caps.
- Backend timeout additions: per-backend `timeout=30` pinned in the existing request-shape tests.
- No live tests (AC: mocked). The existing triple-gated live infrastructure is not extended here.

## §6 Non-goals

Circuit breaker; outbound-policy port for provider calls; server-mode routing via `ResearchSearchScopeService` (natural follow-up once the tool exists); streaming/progress events; Settings-UI toggle row; the server's extra engines; fixing the seven older backends' missing HTTP timeouts (follow-up task, filed at implementation); reviving `search_settings_general` keys the tool doesn't use (`search_language_*` etc. stay parsed-but-unread, recorded).
