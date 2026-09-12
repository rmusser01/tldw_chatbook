---
id: TASK-19557
title: >-
  API-key headers survive cross-origin redirects in two clients
status: Done
assignee:
  - '@claude'
created_date: '2026-08-21 20:07'
labels:
  - security
  - credentials
  - http
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 1
#3 and #4**, filed together because they are one bug shape in two clients.
Re-verified at this branch base.

Both `httpx` and `requests` strip only `Authorization` (and `Cookie`) when a
redirect crosses origins. **Custom API-key headers are not stripped** — they
are forwarded to whatever host the redirect names.

**Client 1 — the tldw server client.**
`tldw_chatbook/tldw_api/client.py:1144` sets `headers["X-API-KEY"] =
self.token` on a client constructed at line 1149 with
`follow_redirects=True`. `api-key` is the **default auth mode**, and this
client backs Notes, Chatbooks, Character, Sync and MCP traffic. **No test pins
this.**

**Client 2 — Anthropic.** `LLM_Calls/LLM_API_Calls.py:1434` sets
`"x-api-key": final_api_key` with the same exposure (also
`LLM_API_Calls.py:1570`, and the same shape in
`LLM_Calls/Summarization_General_Lib.py`).

The sharpest detail: **the fix is already present one function over.**
`LLM_API_Calls.py:3528-3549` handles this correctly and carries a comment
naming the hazard. This is, again, a seam that never adopted a primitive the
repo already owns — not a missing idea.

A user pointed at a hostile or compromised endpoint (or a legitimate one that
starts redirecting) hands over their API key.

## Acceptance Criteria

- [x] Neither `X-API-KEY` nor `x-api-key` is sent to an origin other than the
      one the request was authorized for — on redirect, the credential header
      is dropped or the redirect is refused
- [x] The repair reuses the existing correct pattern at
      `LLM_API_Calls.py:3528-3549` rather than introducing a second mechanism
- [x] Applied to both clients: `tldw_api/client.py` and the Anthropic call
      sites in `LLM_API_Calls.py` / `Summarization_General_Lib.py`
- [x] A regression test drives an actual cross-origin redirect and asserts the
      credential header is absent on the second hop, for each client
- [x] The test is mutation-checked: restoring the header makes it red
- [x] The remaining custom-credential headers in the codebase are swept for the
      same shape and either fixed or recorded as clean

## Implementation Plan

1. Read `tldw_api/client.py` (~1144-1150), `LLM_API_Calls.py` (~1433,
   ~1570, and the reference fix at ~3528-3549), `Summarization_General_Lib.py`
   (~1053, ~1125), and `Utils/egress.py`'s `_STRIP_HEADERS`/guarded-fetch
   helpers to confirm the exact defect shape and the repo's own established
   fix pattern.
2. Decide the httpx-client fix shape (refuse-outright vs. hop-loop vs.
   egress-routing) against AC #2 (reuse the existing pattern, don't invent a
   second mechanism).
3. Fix `tldw_api/client.py`: construct the shared client with
   `follow_redirects=False` and refuse any 3xx response before it's
   processed, across all five request-issuing methods.
4. Fix the two Anthropic call sites (`chat_with_anthropic` in
   `LLM_API_Calls.py`, `summarize_with_anthropic` in
   `Summarization_General_Lib.py`) by mirroring the Google
   `allow_redirects=False` + explicit 3xx-refuse pattern.
5. Sweep the codebase for sibling custom-auth-header call sites with
   redirects enabled; fix any found in the two target files, report the rest.
6. Write born-red regression tests for both clients using an in-process
   mocked transport (httpx.MockTransport / a patched
   `requests.adapters.HTTPAdapter.send`) — never real network egress.
7. Mutation-check: revert the fix via Edit, confirm the tests fail showing
   the credential reaching the cross-origin host; restore the fix via Edit,
   confirm green.
8. Run the tldw_api and LLM_Calls suites plus a repo-wide collect-only sweep;
   baseline any failure against origin/dev before attributing it.

## Implementation Notes

**Fix shape (both clients: refuse-outright, matching AC #2).** Mirrored
`chat_with_google`'s already-shipped pattern exactly rather than inventing a
hop-loop or routing through `Utils/egress.py`'s guarded fetch:

- `tldw_api/client.py`: the shared `httpx.AsyncClient` is now constructed
  with `follow_redirects=False` (was `True`, undocumented, present since the
  client's very first commit). A new `TLDWAPIClient._raise_if_redirected`
  static helper raises `APIConnectionError` on any 3xx response; it's called
  in all five request-issuing methods (`_request`, `_binary_request`,
  `_headers_request`, `_stream_request`, `_sse_request`) right after the
  response is obtained, before `raise_for_status()`/body processing.
- `LLM_API_Calls.py` (`chat_with_anthropic`): both `session.post` calls (the
  initial POST and the cache-control-rejected retry) now pass
  `allow_redirects=False`; an explicit `if 300 <= response.status_code <
  400` check raises `ChatProviderError` before `raise_for_status()`.
- `Summarization_General_Lib.py` (`summarize_with_anthropic`): the
  `requests.post` call now passes `allow_redirects=False`; a 3xx response
  returns an error string (`"...refusing to follow with credentials."`),
  matching this function's own established "return a string, don't raise"
  failure convention (see its "API Key Not Provided"/"Network error"
  sibling returns).

**Why refuse-outright, not a hop-loop or egress-routing.** AC #2 mandates
reusing the existing pattern rather than a second mechanism, and the
repo's own reference fix (`chat_with_google`, `LLM_API_Calls.py:3528-3549`)
already refuses unconditionally (no same-origin exception) rather than
stripping-and-following. A hop-loop would reinvent httpx/requests' own
redirect engine with real risk of missing an edge case; routing the httpx
client through `Utils/egress.py`'s guarded fetch would be strictly better
(it also closes SSRF on the same seam) but is scoped for GET-only fetches
today and would be a much larger, riskier change across a client with five
distinct request shapes (JSON, binary, headers-only, NDJSON stream, SSE)
and dozens of callers — and would still deviate from AC #2. Verified there
is no legitimate reason for either client to follow a redirect in normal
operation (`base_url`/`api_url` are the server/endpoint the caller
explicitly configured).

**No-new-diagnostic-call constraint (a real trap, not hypothetical).** Both
`LLM_API_Calls.py` and `Summarization_General_Lib.py` participate in
`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`'s pinned,
content-hashed diagnostic-call inventory (`Docs/security/production-
diagnostic-inventory.json`, `Tests/fixtures/summarization_diagnostic_
review.json`). An initial draft added `logger.error`/`logging.error` calls
in the new redirect-refuse branches (mirroring Google's own `logger.error`)
and broke three `manifest_boundary` tests by changing pinned content
digests -- confirmed via an Edit-based revert-and-rerun that these three
tests are **not** pre-existing-red (they pass on origin/dev). The existing
in-file precedent (a comment a few lines above the Anthropic redirect fix,
about sampling-param dropping) already documents avoiding new log lines in
this module for exactly this reason. Final fix carries **no new logging
calls**; the caller-visible signal is the raised exception message /
returned string only.

**Sibling sweep.** Swept every custom-auth-header call site in the
codebase (`x-api-key`, `x-goog-api-key`, `Ocp-Apim-Subscription-Key`,
`X-Subscription-Token`, `xi-api-key`, bearer-in-custom-header, etc.).
Clean (verified, no fix needed): `LLM_Provider_Catalog/
openai_compatible_model_discovery.py` (explicit `follow_redirects=False`
on the actual credentialed request), `Research_Interop/
academic_providers.py`'s `search_semantic_scholar` (plain `httpx.Client()`
— httpx's own default is `follow_redirects=False`; the file's one
`follow_redirects=True` site, `search_osf`, carries no auth header),
`Image_Generation/adapters/gemini_image_adapter.py` (routes through
`Image_Generation/http_client.py`'s own hop loop using `egress.
_hop_headers`/`_STRIP_HEADERS`, which already lists `x-goog-api-key`),
`UI/Screens/settings_image_gen_defaults.py` probes (explicit
`follow_redirects=False`), `TTS/backends/elevenlabs.py` and `Widgets/
Settings_Widgets/server_switch_modal.py` (plain `httpx.AsyncClient()` with
no `follow_redirects=True` override), and Kagi/Yandex web-search (their
custom scheme rides the `Authorization` header name, which both libraries
already strip cross-host).

Genuine siblings found but left unfixed (same bug shape, outside this
task's two named clients — reported for separate filing, ranked by
severity):
1. **`Subscriptions/monitoring_engine.py` (watchlists/feed custom auth) +
   `Subscriptions/site_config_manager.py` (`SiteConfig.get_headers`,
   default header name `X-API-Key`, user-configurable to any name) —
   HIGHEST PRIORITY.** Both feed into `Utils/egress.py`'s
   `guarded_fetch_httpx_async` — the repo's shared "already-fixed" guarded
   transport — whose `_STRIP_HEADERS` list is `("authorization", "cookie",
   "proxy-authorization", "x-goog-api-key")` and does **not** include
   `x-api-key`. This is a live gap in the shared primitive itself,
   default-config-reachable via the Subscriptions/Watchlists feature (six
   scraper modules route through the same site-config header path). Adding
   `"x-api-key"` to `_STRIP_HEADERS` closes the common/default case; the
   fully-arbitrary user-named-header case is a harder, separate problem.
2. `LLM_Calls/LLM_API_Calls_Local.py` (KoboldAI native backend, `X-Api-Key`,
   `requests.Session().post(current_api_base_url, ...)` with default
   `allow_redirects=True`; `current_api_base_url` is user-configured).
3. `Web_Scraping/WebSearch_APIs.py`: Bing (`Ocp-Apim-Subscription-Key`,
   user-configurable `bing_search_api_url`), Brave (`X-Subscription-Token`),
   Serper (`X-API-KEY`), Exa (`x-api-key`) — all `requests.get/post` with no
   `allow_redirects=False`; Bing carries the highest risk since its URL is
   user-configurable.

**Born-red evidence.** Two new test files, both using an in-process
transport double (never real sockets — no `@pytest.mark.loopback_network`/
`allow_network` needed):
- `Tests/tldw_api/test_client_redirect_credential_leak.py` — wraps
  `httpx.AsyncClient` construction to inject `httpx.MockTransport` while
  keeping `_get_client()`'s real kwargs (`follow_redirects`) live.
- `Tests/LLM_Calls/test_anthropic_redirect_credential_leak.py` — patches
  `requests.adapters.HTTPAdapter.send` (the layer just above the socket),
  so `requests`' own `Session`/redirect machinery runs for real for both
  `chat_with_anthropic` and `summarize_with_anthropic`.

Both suites assert header-absence on the cross-origin host *before*
asserting the raise/return outcome, so a regression is reported as "the
credential leaked" rather than merely "no exception was raised". Verified
via Edit-based revert/restore (saved as a `.patch` file first, `git apply
-R` / `git apply`, diffed byte-identical afterward): at base (flags flipped
back to `True`/`allow_redirects` default), all 3 new tests fail with the
sentinel key `sentinel-test-x-api-key-must-never-leak` shown present in the
`evil.example` request headers; with the fix restored, all 3 pass.

**Fixture fallout fixed (pre-existing rigid test doubles).** Adding
`allow_redirects=`/checking `.status_code` broke five pre-existing rigid
test doubles whose `post()`/response stubs didn't anticipate the new
kwarg/attribute; widened each to accept it (no behavioral test changes):
`Tests/tldw_api/test_media_reading_client.py` (`FakeStreamResponse` needed
`status_code`), `Tests/LLM_Calls/test_debug_log_fstring_hygiene.py`
(`_CapturedSession.post`), `Tests/Chat/test_console_provider_gateway.py`
(`_CapturedURLSession.post`), `Tests/Chat/test_chat_functions.py`
(`_CapturedSession.post`).

**Files changed:**
`tldw_chatbook/tldw_api/client.py`,
`tldw_chatbook/LLM_Calls/LLM_API_Calls.py`,
`tldw_chatbook/LLM_Calls/Summarization_General_Lib.py`,
`Tests/tldw_api/test_client_redirect_credential_leak.py` (new),
`Tests/LLM_Calls/test_anthropic_redirect_credential_leak.py` (new),
`Tests/tldw_api/test_media_reading_client.py`,
`Tests/LLM_Calls/test_debug_log_fstring_hygiene.py`,
`Tests/Chat/test_console_provider_gateway.py`,
`Tests/Chat/test_chat_functions.py`.

**Verification (venv `../../.venv/bin/python`, `PYTHONPATH` + cwd pinned to
this worktree, `tldw_chatbook.__file__` asserted before every run):**
- New born-red tests: 3 passed (green with fix); all 3 fail at base showing
  the leaked header (verified via Edit-based patch revert/restore).
- `Tests/tldw_api/`: 489 passed.
- `Tests/LLM_Calls/`: 1049 passed, 3 failed -- all 3 confirmed
  pre-existing on origin/dev (unrelated `test_summarization_diagnostic_
  privacy.py::test_manifest_boundary_*` failures; reproduced identically
  with this task's changes fully reverted).
- `Tests/Chat/` (files touching `chat_with_anthropic`/Anthropic paths) +
  `Tests/Agents/test_agent_budget_cache_aware.py` +
  `Tests/RAG_Search/test_reranker_system_prompt.py` +
  `Tests/Chat/test_chat_conversation_scope_service.py` +
  `Tests/RuntimePolicy/test_boundary_guards.py`: all passed (694 + 16 + 18).
- Repo-wide `pytest --collect-only -q`: 53824 tests collected, 0 collection
  errors.

## Lessons

- The pinned diagnostic-call inventory
  (`Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`) is repo-wide
  (`tldw_chatbook/**/*.py`), not scoped to the two "owned" summarization
  files it names in its docstring -- a new `logger`/`logging` call ANYWHERE
  in a TASK-492/TASK-494-classified file changes the pinned
  `manifest_boundary` sha256, even in an unrelated file touched by the same
  change. Caught here only because the task's own instructions demanded
  baselining every failure against origin/dev before attributing it; the
  in-file precedent comment (sampling-params, same function) was the
  correct signal to follow from the start.
