---
id: TASK-32188
title: Make saved search backend setup reach real requests
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 22:31'
updated_date: '2026-09-09 23:01'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Search setup must use current credentials and correct provider requests so guided setup can report honest readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings saves and environment overrides reach the next search without shared mutable request configuration.
- [x] #2 Shipped Bing and SearX fields remain compatible with legacy aliases and resolve consistently.
- [x] #3 Brave web search uses its web key; Kagi and SearX requests and result parsing match current provider contracts.
- [x] #4 A shared backend field catalog supports all ten Console engines with required fields and secret-safe setup checks.
- [x] #5 Offline tests exercise real dispatch and HTTP request construction; retired or restricted providers are labeled accurately.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce stale credentials and provider request failures with offline HTTP-boundary tests.
2. Add a shared backend field catalog and pure effective-value/setup-check helpers for all ten Console engines.
3. Resolve current per-request config and environment values; support shipped fields and legacy aliases; repair Brave, Kagi and SearX request/result contracts.
4. Verify current vendor contracts from primary documentation, targeted request tests, import provenance and scoped static checks. UI work consumes the catalog in a separate task.

ADR required: yes
ADR path: backlog/decisions/012-provider-credential-settings-boundary.md
Reason: extend the existing credential boundary to the existing web-search config tables and explicit saved-settings probes. ADR-032 continues to govern query destinations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Backend requests now resolve current config-owner values and environment overrides
at each invocation; the import-time mutable snapshot is removed. The new
`Web_Scraping/search_backend_settings.py` owns all ten backend field definitions,
source labels, required-field/dependency checks, SearX endpoint validation and the
explicit saved-settings probe. `WebSearch_APIs.py` uses the same effective values.
Bing/SearX normalization accepts shipped names and legacy aliases; the default
SearX placeholder is now empty so it cannot mask an older configured endpoint.

Brave uses the ordinary web key by default and passes its documented web-search
parameters, preserving explicitly requested AI-key behavior. Dispatcher arguments
no longer mistake excluded domains for credentials. Kagi uses the documented
legacy v0 URL once and receives a numeric result limit. SearX retains endpoint
query options, requests JSON explicitly, reads the `results` list, closes its
session and treats an empty list as a successful search. Tavily dispatch sends
excluded domains as exclusions and retains raised HTTP failures for classification.

The probe performs only the visible sample search, with no synthesis or result
cache, and returns closed success/auth/quota/network/timeout/failure messages.
Provider bodies, credential-bearing exception text and Google raw result dumps
are excluded from probe diagnostics. Standard workflows keep their existing cache.
Bing retirement, Google new-customer restrictions and Kagi v0 deprecation are
shown explicitly while configured choices remain available.

ADR: [ADR-012](../decisions/012-provider-credential-settings-boundary.md), including
its Web Search extension; [ADR-032](../decisions/032-local-agent-tool-permission-boundary.md)
continues to govern query destinations. No new credential store or provider migration.

Verification: `PYTHONPATH=$PWD /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/test_search_backend_settings.py Tests/Web_Scraping/test_search_backends.py Tests/Web_Scraping/test_websearch_credential_logging.py Tests/test_websearch_config.py Tests/test_probe_import_provenance.py -q --tb=short`
reported **91 passed, 3 skipped** (explicit live-provider tests), with the existing
Requests dependency-version warning. New regressions were observed failing before
fixes. Eight keyed engines are exercised through real Settings adapter saves,
config publication, dispatch and HTTP construction; only external HTTP is replaced.
Tests also cover repeat saves, environment precedence, aliases, SearX JSON results,
local/invalid endpoints, safe probe errors and provider-payload logging.

New metadata/test files pass Ruff lint and formatting checks. Existing backend and
config files pass focused syntax/undefined-name checks and `git diff --check`;
broad style lint on the legacy backend still reports pre-existing style findings.
The old config-printing diagnostic was replaced with a secret-safe key-presence test.
No full suite or live provider call was run. Per-request provider timeouts remain in
force; SearX retries can make total probe latency longer than one request timeout.
Independent review and final acceptance-criteria/Done hygiene remain with the parent task.

Vendor contract sources checked on 2026-09-09:

- [Brave web-search contract](https://api-dashboard.search.brave.com/api-reference/web/search/post)
- [Brave authentication](https://api-dashboard.search.brave.com/documentation/guides/authentication)
- [Kagi legacy v0 contract and deprecation](https://help.kagi.com/kagi/api/search-legacy.html)
- [Kagi current v1 overview](https://help.kagi.com/kagi/api/search.html) (full v1 reference was unavailable; no migration was guessed)
- [SearXNG search API](https://github.com/searxng/searxng/blob/master/docs/dev/search_api.rst)
- [Google restricted availability](https://developers.google.com/custom-search/v1/overview)
- [Microsoft Bing retirement](https://learn.microsoft.com/en-us/lifecycle/announcements/bing-search-api-retirement)

### Independent review repairs

Review reproduced a malformed Google `searchInformation.totalResults` string
being interpolated into an integer-conversion exception log, plus partial parsed
provider data surviving error responses. Added nine real HTTP-response/JSON/parser
regressions, observed failures, then replaced parser diagnostics with closed text
and cleared partial metadata/results on failure. Removed the remaining raw Bing
body log. The Google case now proves the conversion failure is reached without
leaking its synthetic secret through either logs or returned output.

Google's official country-parameter table confirms `cr=countryUS`, not `cr=US`.
Bare two-letter inputs now become country-prefixed tokens (GB maps to countryUK);
already-prefixed and boolean expressions are preserved. Five dispatch/request
cases cover that behavior. [Google country parameter values](https://developers.google.com/custom-search/docs/json_api_reference).

DuckDuckGo now checks HTTP status before interpreting the HTML body. Two real
429/503 response regressions prove neither dispatch nor the saved-settings probe
reports an error page as successful empty search results. The new public
`saved_field_value` helper lets Settings display saved legacy endpoints separately
from effective environment overrides.

The same targeted command now reports **107 passed, 3 skipped**, with the same
existing Requests warning. Full new-file Ruff/format, scoped existing-file
syntax/undefined-name checks and `git diff --check` pass.

Reviewed each changed diagnostic statement before using the inventory script's
`--write` workflow. `WebSearch_APIs.py` goes from 104 to 91 diagnostic calls: raw
bodies/headers/URLs/exceptions are removed or replaced with fixed labels and
closed status categories. Inventory regeneration also revealed existing HEAD
inventory drift: Media DB summary-search logging adds one fixed-format diagnostic
with validated pagination/sort values; Library adds fixed-text tool-catalog and
lifecycle-worker-unmount warnings. Those statements were inspected in their
source/history, retained under their existing TASK-494 owner, and no new sink or
exclusion was added. The unmount warning's existing exception attachment remains
outside this search change. The final checker verifies **519 owners, 1213 TASK-492
calls, 7138 TASK-494 calls, 7 sink files**. Inventory file:
`Docs/security/production-diagnostic-inventory.json`.

Final integration verification: 448 targeted tests passed, three live tests skipped. Independent review confirmed all four findings resolved; diagnostic inventory verified. Guided Web Search consumes the shared metadata and saved-field resolver. Runtime service availability is not proven by offline tests; Kagi v0 deprecation remains explicitly disclosed. Changes remain uncommitted in the isolated worktree.

PR integration: moved onto dev 86a8054edb, preserving current TLS, profile/footer, privacy and config publication behavior. Final evidence and baseline limitations: Docs/superpowers/reviews/2026-09-09-search-settings-pr-integration.md (525 targeted cases passed across two runs, three live cases skipped).
<!-- SECTION:NOTES:END -->
