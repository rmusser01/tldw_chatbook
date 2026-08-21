---
id: TASK-19552
title: >-
  Google Search API key is logged at INFO on the default search engine
status: Done
assignee: []
created_date: '2026-08-21 20:02'
labels:
  - security
  - websearch
  - logging
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 1
#2**. CONFIRMED by the lane, **independently re-confirmed by the review
controller**, and re-verified at this branch base.

`tldw_chatbook/Web_Scraping/WebSearch_APIs.py`:

```
3402:        params["key"] = google_search_api_key
3428:        logger.info(f"Prepared parameters for Google Search: {params}")
```

The whole `params` dict — including the API key just written into it — is
interpolated into an INFO-level log line. This is not a DEBUG-only path:

```
296:        "search_engine": search_params.get("engine", "google"),
```

**Google is the default engine**, so this fires on the ordinary `web_search`
tool path and on deep search, at the default log level, for any user who has
configured a Google CSE key.

Two things make this worse than a single leaky line:

- It is a **recurrence of a class a previous remediation wave already
  repaired**. The fix pattern exists in the repo; this call site never adopted
  it. (Bing on the same surface is clean — it passes the credential as a
  header and never formats it.)
- It compounds with TASK-19555: the in-app log buffer has no sanitizing filter
  and the Logs screen offers "Copy all" to the system clipboard, so this key
  does not merely sit in a file — it is on the share path the UI actively
  invites users to use.

## Acceptance Criteria

- [x] No search-provider credential reaches any log record, at any level, on
      any engine — the Google path in particular never formats a dict that
      contains `key`
- [x] The fix redacts at the point of formatting rather than relying on the
      caller's log level, so raising the log level for troubleshooting cannot
      re-expose it
- [x] A regression test asserts that a configured Google CSE key does not
      appear in emitted log records for a search call, and that the test would
      fail if the redaction were removed (mutation-checked, not merely green)
- [x] The other search-provider paths in `WebSearch_APIs.py` are swept for the
      same shape and any further instances are fixed in the same change
- [x] Owner is advised whether the affected key needs rotating, given the key
      may already be in existing local log files

## Implementation Plan

1. Read `Utils/log_sanitizer.py` (denylist-based) and `Utils/sensitive_llm_logging.py`
   (allowlist-based, with its documented rationale: a denylist has already
   failed twice on the analogous LLM-request logging path in this repo).
   Prefer the allowlist shape for the Google `params` dict per the
   controller's brief.
2. Sweep every `logger.*` call in `WebSearch_APIs.py` for a credential-bearing
   interpolation: params dicts, headers dicts, auth values, or exception
   text that could embed a URL built with `params=`. Cross-reference each
   engine's request shape (header-based auth vs. URL-query auth) to know
   which sites are even capable of leaking a credential.
3. Empirically verify (small script against the real `requests` exception
   classes) whether an HTTPError/ConnectionError's `str()` embeds the full
   request URL for a GET request that carries the key as a query param —
   this determines whether the Google function's exception-log site
   (`logger.error(f"Error during API request: {str(re)}")`) is a second,
   independent leak vector beyond the already-identified params-dict log.
4. Fix the Google `search_web_google` function:
   a. Add a module-level allowlist of safe-to-log Google CSE param keys
      (everything currently set except `key`) and a small helper that
      projects `params` through it before formatting into the INFO log.
   b. If step 3 confirms the exception-text vector, redact exception text
      at the formatting point (reusing `Utils/log_sanitizer.sanitize_string`,
      which already has a Google `AIza...` key pattern) rather than
      widening the allowlist helper to unstructured text.
5. Write born-red regression tests: capture loguru records (the
   `Tests/Chat/test_chat_api_call_endpoint_redaction.py` bridge idiom) while
   invoking the Google parameter-preparation path and the exception path
   with a sentinel API key; assert the sentinel is absent from every log
   record while the non-credential params are still present.
6. Run the targeted web-search test suites plus a repo-wide
   `--collect-only -q` sweep to confirm no collection regressions.
7. Update task file: AC checkboxes, Implementation Notes, status Done.

## Implementation Notes

Approach: fixed two independent credential-log vectors in
`search_web_google` (`tldw_chatbook/Web_Scraping/WebSearch_APIs.py`), swept
every other `logger.*` call in the file and every sibling
`search_web_*`/`test_search_*` function for the same shape, and added
born-red regression coverage.

**Vector 1 (the filed defect):** `logger.info(f"Prepared parameters for
Google Search: {params}")` at line 3428 formatted the whole `params` dict,
including `params["key"]`. Fixed with an explicit ALLOWLIST —
`SAFE_GOOGLE_SEARCH_PARAM_KEYS` (`q, c2coff, cr, cx, num, dateRestrict,
exactTerms, excludeTerms, filter, gl, hl, lr, safe, sort` — i.e. every key
`search_web_google` sets except `key`) plus a `_safe_search_params_for_log`
helper that projects `params` through it before formatting. Any future
param this function starts setting is dropped from the log by default
unless someone explicitly adds it to the allowlist — matching the "unknown
key is safe by default, not exposed by default" property
`Utils/sensitive_llm_logging.py` documents and that the controller's brief
asked to follow, in preference to a denylist (`Utils/log_sanitizer.py`
documents two prior denylist failures on the analogous LLM-request path).

**Vector 2 (found during the sweep, not in the original filing):**
`logger.error(f"Error during API request: {str(re)}")` (now line 3490).
Empirically verified against the real `requests` library (a `requests.
models.Response` with `.url` set to a URL carrying a sentinel `AIza...`
key, `.raise_for_status()` called) that both `HTTPError` and
`ConnectionError` embed the full request URL — including `key=<value>` —
in their `str()` when the key was passed as a GET query param
(`requests.get(search_url, params=params, ...)`, as Google's call does).
This fires on exactly the failure modes a user hits while debugging a
bad/expired key: 401/403/429/connection failure. This is why the Bing
branch is clean by comparison — Bing sends its key as a header, so
`response.url`/the connection-pool message never contains it. Fixed by
wrapping the `RequestException` handler text through `Utils/log_sanitizer.
sanitize_string`, which already carries a dedicated
`AIza[A-Za-z0-9_-]{35}` pattern for Google API keys (plus generic
Bearer/URL-userinfo/other vendor-key patterns) — pattern-based redaction is
the right tool here specifically because exception text is unstructured (no
"keys" to allowlist), unlike the params dict in vector 1. Disclosed
limitation: this only catches known credential *shapes*; Google's real CSE
keys are uniformly `AIza`-prefixed, so this is adequate coverage for this
engine, but the fix would not catch an arbitrarily-shaped secret in free
text.

**Vector 3 (also found during the sweep):** the `except ValueError as ve:`
branch (line 3481), which sits *before* `RequestException` in the chain.
`requests.exceptions.JSONDecodeError` (raised by `response.json()` on
malformed content) is itself a `ValueError` subclass, so a bad-JSON
response is caught here, not by the `RequestException`/`Exception`
handlers vector 2 fixed — meaning those two fixes alone would have left one
branch that can still receive a `ValueError` unprotected. This function's
own two hardcoded config-error `ValueError`s never carry the key, and a
real JSON-decode message doesn't naturally embed the URL either, but the
branch is sanitized defensively anyway (via the same `sanitize_string`
call) so the "at any level, on any engine" AC holds unconditionally rather
than "in the cases I happened to think of." The catch-all
`except Exception as e:` (line 3494) got the same treatment for the same
reason.

**Sweep results — every `logger.*` call in the file, by engine:**
- Bing (2638-2760): clean, unchanged — key travels via header
  (`Ocp-Apim-Subscription-Key`), never formatted; `params` logged at DEBUG
  contains no key. Confirmed as the reference shape.
- Brave (2923-3096): clean — key in header (`X-Subscription-Token`); no
  logger call touches `params`/`headers`.
- Kagi (3616-3640): clean re: credentials — key in header
  (`Authorization: Bot ...`); `logger.debug(response.json())` logs
  *response* content, not the request/key. Flagged (not fixed — different
  task) as a content-logging concern for TASK-19555.
- SearX (3730-3838): no API key at all for this engine; `logger.info(f"Search
  URL: {search_url}")` embeds the user's query text in a URL, which is a
  content concern for TASK-19555, not a credential.
- Serper (3919-3952), Exa (3991-4022), Yandex (4159-4196): clean — key in
  header, POST body (not URL query params), and **no logger call in any of
  the three functions**.
- Tavily (4057-4093): key is embedded directly in the JSON POST `payload`
  dict (`payload["api_key"] = tavily_api_key`) rather than a header — the
  same structural smell as Google, but **no logger call in the function
  formats `payload`** today, so nothing currently leaks. Flagged as a
  latent risk for a future maintainer (a debug log of `payload` here would
  leak immediately) but not fixed — no live defect, and reshaping Tavily's
  auth transport is outside this task's scope (logging, not transport).
- Google (3317-3499): the three vectors fixed above.
- Every other `logger.*` call in the file (result counts, `raw_results`
  response-content dumps, generic error-message text with no request
  object embedded, sub-query/relevance-pipeline logging) was checked and
  contains no provider credential — several ARE response/query content and
  are reported (not fixed) as TASK-19555 scope per the controller's brief.

**Tests added:** `Tests/Web_Scraping/test_websearch_credential_logging.py`
(4 tests), using the repo's standard loguru→caplog bridge
(`Logging_Config._forward_loguru_to_standard`, per
`Tests/Chat/test_chat_api_call_endpoint_redaction.py`):
- `test_safe_search_params_for_log_drops_key_keeps_allowlisted_and_unknown`
  — unit-level: the allowlist helper drops `key` AND an unlisted/future key.
- `test_google_success_path_never_logs_the_key` — vector 1: baseline proves
  the actual sent `params` dict (containing the real sentinel) would leak
  if formatted directly; asserts the emitted INFO log doesn't carry it
  while `cherry cake` and `cx123` still appear.
- `test_google_http_error_never_logs_the_key_via_exception_text` — vector 2:
  baseline proves `requests`'s own `HTTPError.__str__()` embeds the
  sentinel via the URL; asserts the emitted ERROR log doesn't, while
  "403"/"Forbidden" still appear.
- `test_google_value_error_never_logs_the_key_via_exception_text` — vector 3
  (the `except ValueError` branch), same shape.

Each is mutation-checked, not merely green: I reverted each of the three
`sanitize_string(...)`/allowlist call sites in turn (via Edit, back to the
original `{params}`/`{str(re)}`/`{str(ve)}` formatting) and reran the
corresponding test — all three failed red with the sentinel key visibly
present in the captured log record (e.g. `...&key=AIzaTASK19552SENTINEL...`
for vector 2) — then restored the fix and reran the full file green again.
Full 4/4 file run: `4 passed, 1 warning in 0.57s`.

**Owner advisory:** no real API key was used anywhere in this branch's
tests or logs — only synthetic sentinels shaped like `AIza<35 chars>`.
Whether to rotate any Google CSE key that may already be sitting in an
existing user's local log file from before this fix is an owner decision;
this task cannot inspect anyone's local logs to answer that, so it is
surfaced here rather than assumed either way.

**Verification run (venv `../../.venv/bin/python`, `PYTHONPATH`+cwd pinned
to this worktree, confirmed via a `tldw_chatbook.__file__` path assert):**
- New file: `Tests/Web_Scraping/test_websearch_credential_logging.py` —
  4 passed.
- `Tests/Web_Scraping/` (full dir, incl. `test_search_backends.py`'s
  existing google/bing/brave/kagi/serper/exa/tavily/yandex/searx request-
  shape pins) — 222 passed, 3 skipped (`TLDW_LIVE_SEARCH_TESTS` live tests,
  unrelated).
- `Tests/Tools/test_web_search_tool.py` (the `web_search` local-tool entry
  point) — 2 passed.
- `Tests/Internal_Prompts/test_websearch_prompt_parity.py` — 5 passed,
  1 pre-existing failure (`test_result_relevance_eval_parity`, a Jinja2
  whitespace assertion unrelated to logging/credentials). Confirmed
  pre-existing and unrelated to this change by running the identical test
  against a throwaway `origin/dev` worktree at the same base commit
  (`2a15a72bb`) — it fails there too, byte-for-byte the same assertion.
  Not touched; out of this task's scope.
- `Tests/Utils/test_log_sanitizer.py` (the reused `sanitize_string` utility,
  unmodified by this change) — 57 passed.
- Repo-wide `pytest --collect-only -q` — 53633 tests collected, zero
  collection errors.
- Mutation check (see above): all 3 fixed log sites individually reverted
  and reran red, then restored and reran green.

Also noted, not fixed (pre-existing, out of scope): `Tests/Web_Scraping/
test_security.py::TestAPIKeySecurity::test_api_keys_not_logged` and
`test_api_key_masking` are vacuous — they patch `logging.info` in isolation
and never call any real search function, so they could not have caught
this defect and don't exercise the fix either. Left as-is; not this task's
scope to rewrite a pre-existing test.

**Diagnostic-inventory restamp (`Docs/security/production-diagnostic-
inventory.json`):** the 3 log-site edits change the AST source text of 3
diagnostic calls (content, not count — still 104 calls for this file), so
the `WebSearch_APIs.py` row's `diagnostic_digest` needed restamping per
`scripts/check_persistent_diagnostic_inventory.py`. Recomputed the digest
for just this file (`63f3ddaba4b6557c0714` → `37972e1ac99fc6d3d304`,
`call_count` unchanged at 104) and hand-edited only that one row, rather
than running `--write` (which regenerates the whole file).

**Owner-relevant finding, NOT caused by this task:** running the checker's
full `--write` regeneration surfaced that the checked-in inventory is
*already stale on `origin/dev`* for two unrelated files —
`tldw_chatbook/DB/Client_Media_DB_v2.py` (338→339 calls) and
`tldw_chatbook/UI/Screens/library_screen.py` (109→111 calls) — meaning
`Tests/Architecture/test_persistent_diagnostic_inventory.py::
test_production_diagnostic_inventory_and_sink_topology_are_unchanged`
already fails on a pristine `origin/dev` checkout at this branch's base
commit (`2a15a72bb`), confirmed by running the identical check in a
throwaway `origin/dev` worktree before touching any source. This task
deliberately did not touch those two files or restamp their rows — I
haven't reviewed what diagnostic calls changed there, and bundling an
unreviewed restamp into this PR would blur the credential-logging fix with
unrelated content. Surfacing this for the controller/owner: someone needs
to review those two files' diagnostic changes and restamp their rows
separately; until then this one Architecture test stays red for a reason
unrelated to TASK-19552.

**Files changed:**
- `tldw_chatbook/Web_Scraping/WebSearch_APIs.py`
- `Tests/Web_Scraping/test_websearch_credential_logging.py` (new)
