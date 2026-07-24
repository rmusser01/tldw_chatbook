---
id: TASK-367
title: Make endpoint URL entry forgiving - normalize v1 and completion-path variants
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-23 09:00'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three contradictory URL shapes in one flow: (1) selecting llama.cpp prefills 'http://localhost:8080/completion'; (2) Discover models with a working server at bare-origin http://127.0.0.1:9099 was refused with 'This endpoint is not OpenAI-compatible for v1 discovery. Configure a /v1 endpoint to discover models.' — meaning the app's own prefilled '/completion' default would also fail discovery; (3) with a realistic typo 'ttp://127.0.0.1:9099/v1' (dropped h), the app returned the SAME '/v1' message even though the URL already ends in /v1 — the real problem (invalid scheme) is never surfaced, and no inline validation flags the malformed URL. Only after typing the exact 'http://.../v1' form did discovery succeed (~0.8s, j1-24).

**Repro:** Provider=llama.cpp (note /completion default) -> Endpoint=http://127.0.0.1:9099 -> Discover models -> '/v1' refusal; set Endpoint='ttp://127.0.0.1:9099/v1' -> Discover -> same '/v1' refusal; fix to http://127.0.0.1:9099/v1 -> succeeds.

**Verifier note:** Code-confirmed on a post-ledger surface (Settings discovery shipped 1fd5f5f0c, 2026-07-11, tasks 188/191-console). _discovery_status_from_error flattens every unsupported_endpoint kind into the single '/v1' copy (settings_screen.py:4958 → MODEL_DISCOVERY_UNSUPPORTED_ENDPOINT_COPY:218), discarding the client's distinct messages; supports_openai_compatible_model_discovery (openai_compatible_model_discovery.py:224-249) returns False both for a malformed scheme (parse→None, e.g. 'ttp://…/v1') and for llama.cpp's native /completion path — so the app's own prefilled default fails its own discovery and a scheme typo is misdiagnosed as a /v1-path problem. No inline URL validation on the field. Nothing in the ledger covers discovery UX. P2: first-run guidance actively contradicts itself, but manual model entry and the Console-modal discover path remain.

**Source:** Console UX expert review 2026-07-20 (finding j1-endpoint-v1-guessing; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J1 new-user cold start journey. Evidence: `j1-12-after-provider-switch-settle.png`, `j1-21-discover-good-result.png`, `j1-23-endpoint-check.png`, `j1-24-discover-success.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prefill a default that works with the app's own features
- [x] #2 Auto-append or auto-probe /v1 instead of demanding the user reformat
- [x] #3 Distinguish 'malformed URL' from 'missing /v1 path'
- [x] #4 Validate the URL inline on blur
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**#1/#2 already resolved on dev since the review.** `_models_path_for_endpoint_path`
now maps a bare origin (`/`) AND llama.cpp's native `/completion`/`/completions`
to `/v1/models`, so the prefilled default works and a bare origin is auto-probed
at `/v1` without the user reformatting (locked by the existing discovery tests
`test_llamacpp_completion_url_maps_to_v1_models` etc.).

**#3 malformed vs unsupported — implemented.** `discover_openai_compatible_models`
previously returned a single `unsupported_endpoint` error for BOTH a malformed
URL (bad scheme/host, e.g. a dropped 'h' → `ttp://…`) and a valid-but-unsupported
path, so `settings_screen._discovery_status_from_error` could only show the one
generic "configure a /v1 endpoint" copy — misdiagnosing a scheme typo as a path
problem. Now a malformed endpoint (`_parse_endpoint(...) is None`) returns a
distinct `malformed_endpoint` kind with its own message, and
`_discovery_status_from_error` surfaces the client's DISTINCT message/recovery
for both kinds instead of flattening them.

**#4 inline validation on blur — implemented.** New `ProviderEndpointURLValidator`
(passes empty/well-formed http(s) URLs, fails a malformed one) is attached to the
endpoint `SettingsURLInput` with `validate_on={"blur", "submitted"}`, so a
malformed URL is flagged at the field on blur (Textual `-invalid` state) rather
than only when the user later saves or runs Discover.

Verified RED→GREEN: `test_malformed_endpoint_reports_distinct_error_kind`
(discovery module), `test_discovery_status_distinguishes_malformed_from_unsupported`
+ `test_provider_endpoint_url_validator_flags_malformed_only` (settings); 67
discovery + provider-test tests green.
<!-- SECTION:NOTES:END -->
