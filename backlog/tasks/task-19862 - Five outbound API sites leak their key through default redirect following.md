---
id: TASK-19862
title: >-
  Five outbound API sites leak their key through default redirect following
status: Done
assignee:
  - rmusser01
created_date: '2026-08-22'
labels:
  - security
  - credentials
  - websearch
  - llm-providers
priority: medium
dependencies:
  - TASK-19557
  - TASK-19733
---

## Description

Source: unfiled residue in **TASK-19557**'s implementation notes, with exact
lines and per-site severities confirmed by **TASK-19733**'s reviewer.
Re-verified at `3605bd52d`: neither
`Web_Scraping/WebSearch_APIs.py` nor `LLM_Calls/LLM_API_Calls_Local.py`
contains the string `allow_redirects` anywhere, so all five sites take the
`requests` default of `allow_redirects=True`.

Five outbound call sites send an API key in a **custom header** and follow
redirects automatically. `requests` strips only `Authorization` and `Cookie`
when a redirect crosses origins; a vendor-specific header such as
`X-Api-Key`, `Ocp-Apim-Subscription-Key`, `X-Subscription-Token` or
`X-API-KEY` survives the hop and is delivered to whatever host the redirect
names.

None of these sites route through `Utils/egress.py`, so the per-hop
re-validation and cross-origin allowlist that **TASK-19733** built cannot reach
them — that fix hardened the shared primitive, and these five bypass it
entirely.

Sites, with the severities the reviewer assigned:

| Site | Header | Request | Severity |
| --- | --- | --- | --- |
| KoboldAI | `LLM_API_Calls_Local.py:1011` (`X-Api-Key`) | `:1061` `session.post` | **medium** |
| Bing | `WebSearch_APIs.py:2711` (`Ocp-Apim-Subscription-Key`) | `:2728` `session.get` | **medium** |
| Brave | `:2964` (`X-Subscription-Token`) | `:2980` `requests.get` | low |
| Serper | `:3985` (`X-API-KEY`) | `:3992` `requests.post` | low |
| Exa | `:4055` (`x-api-key`) | `:4062` `requests.post` | low |

Kobold is the worst of the five on two counts. Its destination,
`current_api_base_url`, is **user-configured** — a redirect-serving endpoint is
reachable by ordinary misconfiguration or by a hostile local-server URL, not
only by a compromised vendor. And it is a POST: on a 302 or 303 `requests`
converts the method to GET and re-issues to the new host, still carrying the
key header.

Bing is medium because `bing_search_api_url` is likewise read from config
(`WebSearch_APIs.py:2672`) and the credential is a paid Azure subscription key.
Brave, Serper and Exa are low because their endpoints are hard-coded vendor
URLs — only a vendor-side compromise or DNS attack reaches them.

Kagi and Yandex were checked and are genuinely clean.

Minimal fix is `allow_redirects=False` at each site: none of these APIs
legitimately redirects, so refusing a redirect is a behaviour-preserving change
that turns a silent credential disclosure into a visible error.

## Acceptance Criteria

- [x] None of the five sites follows a redirect while carrying its API key
      header
- [x] A redirect response at each site produces a clear, non-silent failure
      rather than a second request to the redirect target
- [x] A test per site drives a redirect from a stubbed server and asserts the
      key header is never sent to the second host, and each is mutation-checked
      (restoring default redirect following makes it red)
- [x] The Kobold POST case is covered specifically for a 302/303, where the
      method converts to GET — a test that only exercises a 307 does not
      demonstrate the fix
- [x] Every redirect response that is refused is closed, so the connection is
      not leaked (the defect TASK-19557's Qodo round found on its own refusal
      paths)
- [x] The refusal message does not echo the raw attacker-controlled `Location`
      header (consistent with the exception-text rule applied in TASK-19321 /
      TASK-19552 / TASK-19557)
- [x] The remaining `requests` call sites in both modules are swept for the same
      shape and the result recorded, so this residue does not have to be
      rediscovered a third time

## Implementation Plan

Re-verified at branch base `ef831d9f38` (2026-09-30): the four WebSearch
sites named below were fixed incidentally by **TASK-32894** (commit
`f25f707e6e`), which introduced `_credentialed_search_request` in
`Web_Scraping/WebSearch_APIs.py` (allow_redirects=False, capped body,
close-on-refusal) and routed all eight credentialed backends through it —
but with recorder-fake tests that assert the kwarg, not stubbed-server
redirects. The KoboldAI site in `LLM_Calls/LLM_API_Calls_Local.py`
(`chat_with_kobold`, `session.post` carrying `X-Api-Key` to a
user-configured URL) is still unpatched.

1. Kobold fix, test-first: write a born-red stubbed-server redirect test
   for `chat_with_kobold` (302 and 303 — method converts POST to GET —
   plus 307), watch it leak, then pass `allow_redirects=False` at the
   `session.post`, explicitly refuse any 3xx with `response.close()`
   before raising, and watch it green.
2. Add per-site stubbed-server redirect tests for all five sites in the
   TASK-19557 idiom (`HTTPAdapter.send` patched one layer above the
   socket, so `resolve_redirects`/`rebuild_auth`/`rebuild_headers` run
   for real): a first host serves the 30x, a second host records what it
   receives; assert the key header never reaches the second host, the
   refusal raises, the refused response is explicitly closed, and the
   error text does not echo the `Location` value.
3. Mutation-check every new test: temporarily restore default redirect
   following at the site under test (and separately remove the refusal
   close), confirm the test goes red, restore, confirm green.
4. Sweep every remaining `requests`/`session` call site in both modules
   for the same shape (credential in a custom header + default redirect
   following) and record the outcome in Implementation Notes.
5. Targeted test runs only: the new test files plus the two modules'
   existing test files.

ADR required: no — applies an already-decided per-site refusal pattern
(TASK-19557's `allow_redirects=False` convention and TASK-32894's shared
transport) to one remaining site; no schema, sync, contract or boundary
decision is being made.

## Notes

This residue has now been reported by two separate reviewers (TASK-19557's
notes, then TASK-19733's) without ever being filed. That is the reason for the
sweep criterion: the point is to close the class, not the five known members.

## Implementation Notes

### Approach

Branch `fix/task-19862-redirect-key-leaks` off `origin/dev` `ef831d9f38`.
On arrival, four of the five sites were already fixed incidentally by
**TASK-32894** (commit `f25f707e6e`, landed after this task was filed):
its `_credentialed_search_request` transport passes
`allow_redirects=False`, refuses a 30x with `EgressFetchError`, caps the
body, and closes the response in a `finally`. Bing, Brave, Serper and Exa
all route through it. Its own tests, however, assert the kwarg against a
recording fake — they never drive a real redirect, so this task's
"stubbed server + the key never reaches the second host" evidence gap
remained open, and the fifth site (KoboldAI) was entirely unpatched.

This task therefore delivered:

1. **The KoboldAI fix** (`chat_with_kobold` in
   `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py`): `allow_redirects=False`
   on the key-bearing `session.post`, an explicit `response.close()` and a
   `ChatProviderError` refusal of any 3xx before the body is parsed — the
   TASK-19557 convention. The message names the status code and points at
   the configured URL; it never echoes `Location`.
2. **Live-transport redirect tests for all five sites** (the TASK-19557
   idiom: `HTTPAdapter.send` patched one layer above the socket, so
   `resolve_redirects`/`rebuild_method`/`rebuild_auth`/`rebuild_headers`
   run for real; a first host serves the 30x, a second host records what
   it receives): `Tests/LLM_Calls/test_kobold_redirect_credential_leak.py`
   (302/303/307) and
   `Tests/Web_Scraping/test_search_backend_redirect_credential_leak.py`
   (4 sites x 302/303/307). Each case asserts: the first hop carried the
   sentinel key; **no request at all** reached the second host; a loud
   refusal mentioning "redirect"; the refusal text contains neither the
   `Location` host nor its path; and the refused response was closed
   exactly twice (requests' own redirect-peek close + the refusal path's
   own — `== 2`, not `>= 1`, per the verified rationale in
   `test_anthropic_redirect_credential_leak.py`).

### Per-site evidence (exact commands and results)

All runs in the task worktree with `uv venv` (Python 3.12) +
`pip install -e ".[dev]"`.

- **KoboldAI (born red → green).**
  `python -m pytest Tests/LLM_Calls/test_kobold_redirect_credential_leak.py -q`
  - Before the fix: `3 failed` — on every status the redirect was
    followed and the attacker host's body (`"stolen"`) became the model's
    generation.
  - After the fix: `3 passed`.
- **Bing/Brave/Serper/Exa (green on arrival; pinned by the new
  live-transport tests).**
  `python -m pytest Tests/Web_Scraping/test_search_backend_redirect_credential_leak.py -q`
  → `12 passed` (4 sites x 302/303/307).

### Mutation checks (restoring default redirect following → red)

1. Helper mutated (`WebSearch_APIs.py:909` `allow_redirects=False` →
   `True`; edit applied with a python replace, reverted with
   `git checkout HEAD --`): the 12 search cases → `12 failed`, each
   failure message showing the sentinel key delivered to
   `evil.example` — e.g. serper/exa on 302/303 arrive **as GET**
   (`'x-api-key': 'sentinel-…'`, method converted from POST), on 307 as
   the original POST; Bing's `ocp-apim-subscription-key` and Brave's
   `x-subscription-token` likewise. Restored → `12 passed`.
2. Kobold mutated (`allow_redirects=False` → `True` via python replace;
   fixed file preserved with `cp`, never `git stash`): `3 failed`, with
   `KoboldAI request was re-issued to the redirect target 'evil.example'
   as 'GET' carrying headers {'x-api-key': 'sentinel-…'}` on 302/303 and
   `as 'POST'` on 307 — the exact method-conversion case the AC calls
   out. Restored → `3 passed`.
3. Closure mutations: removing the refusal path's own `response.close()`
   (Kobold) → `3 failed` with `observed 1 close() call(s)`; disabling the
   helper's `finally` close → `12 failed` with the same signature. Both
   restored → green. The `== 2` assertions detect a missing
   production-owned close, not just the library's own.

### Sweep: every remaining outbound call site in both modules

`LLM_Calls/LLM_API_Calls_Local.py` (3 dispatch sites, 2 header builders):

- `:302` and `:377` — the shared local-OpenAI-compatible POST (streaming
  and non-streaming; llama.cpp/vllm/ollama/mlx/ooba/tabby/aphrodite/
  custom-openai). Only credential header is `Authorization: Bearer`
  (`:163`); `requests`' own `rebuild_auth` strips `Authorization` on any
  host/scheme/port change, so these are **not** in the custom-header leak
  class. Same-host redirects keep it, but a same-host hop is not a new
  recipient. Deliberately unchanged (minimal fix; altering redirect
  behavior of the shared transport for 7+ local providers is outside
  this task's AC). Noted wart: a silent 302 POST→GET conversion would
  still break streaming without a credential leak — follow-up material,
  not a member of this class.
- `:1113` Kobold `X-Api-Key` (`:1054`) — **fixed here**.

`Web_Scraping/WebSearch_APIs.py`:

- 8 sites through `_credentialed_search_request` (bing `:2795`, brave
  `:3037`, google `:3533`, kagi `:3745`, serper `:4024`, exa `:4102`,
  tavily `:4203`, yandex `:4309`) — redirect-refusing by construction
  (TASK-32894), pinned by its AST census test plus this task's
  live-transport tests for the four custom-header members. Kagi/Yandex/
  Tavily carry `Authorization` (double-protected); Google's key rides in
  the query string, which `requests` does not re-attach to a `Location`
  target — and the helper refuses the hop regardless.
- `:3201` DuckDuckGo HTML POST — **uncredentialed** (data-only, no key
  header): named exemption already on record in TASK-32894's census
  (`_UNCREDENTIALED_CALLERS`).
- `:3898` SearX GET — **uncredentialed** user-configured instance
  (`Accept` header only): same recorded exemption.

No other `requests`/`session` HTTP dispatch (including `.request(`,
`httpx`) exists in either module. **The custom-header +
default-redirect-following class is closed in both modules.**

### Test-suite impact

Targeted runs (new files + both modules' existing test files +
kobold-driving chat tests): all new/edited tests green
(`3 + 12` new; `test_search_backend_redirect_refusal.py` +
`test_websearch_credentialed_transport.py` +
`test_kobold_tabby_config.py` +
`test_custom_openai_credential_resolution.py` +
`test_chat_model_capability_predicates.py` +
`test_local_llm_provider_config.py` +
`test_chat_unit_mocked_APIs.py` all green). Pre-existing local failures
proven by baseline A/B (`git checkout HEAD --` swap of the one changed
production file, never `git stash`): `Tests/Web_Scraping/` shows 34
failures and `Tests/Chat/test_streaming_stop_closes_transport.py` 7
failures **identically with and without this change** (name-diffed
failure lists; the `RecoveryRequired` config-admission signature the
testing-evidence lessons document for this machine's profile state).
This change adds zero new failures. `uvx ruff check --select F,E9` on
all changed/new files: `All checks passed!`.

### Files changed

- `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` — KoboldAI redirect
  refusal (+31/-2).
- `Tests/LLM_Calls/test_kobold_redirect_credential_leak.py` — new.
- `Tests/Web_Scraping/test_search_backend_redirect_credential_leak.py` —
  new.
- This task file.

ADR required: no — one remaining site brought under the already-decided
TASK-19557 refusal convention and TASK-32894's shared transport; no
schema, sync, contract, or boundary decision. (Restated from the plan.)
