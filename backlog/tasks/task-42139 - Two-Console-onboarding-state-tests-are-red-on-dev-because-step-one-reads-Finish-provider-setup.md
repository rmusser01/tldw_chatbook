---
id: TASK-42139
title: "Two Console onboarding-state tests are red on dev because step one reads Finish provider setup"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 18:45
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/Chat/test_console_onboarding_state.py
- tldw_chatbook/Chat/console_onboarding_state.py
- tldw_chatbook/Chat/console_session_settings.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (round 18). At d85e90bc9e two nodes in `Tests/Chat/test_console_onboarding_state.py` fail (through the test wrapper): `test_missing_key_renders_card_with_provider_step_active` (line 51) and `test_endpoint_problems_relabel_step_one` (line 73). Step one's label is "Finish provider setup" where they expect "Connect a provider (API key or local server)" and "Save the provider's server address (endpoint)".

The likely cause: `build_console_setup_card_state` picks the label from `readiness.blocker` (`tldw_chatbook/Chat/console_onboarding_state.py:355`, map at `:47-57`), but the test's `_readiness` helper builds `ConsoleSettingsReadiness` with only a `label`, so `blocker` is `None` and the fallback wins. The lookup by `readiness.blocker` came with c7417cc518 (2026-10-31). One other node of this file runs in a PR lane (`.github/workflows/derived-artifacts.yml:479`); these two do not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named and fixed, so both nodes pass and still pin the label production shows for a missing key and for an endpoint problem
- [x] #2 Both nodes run in a PR lane
- [x] #3 A negative control (a temporary wrong label in `_STEP_ONE_LABELS`) turns a fixed node red; the notes record it
<!-- AC:END -->

## Implementation Plan

1. Name the cause: since #2365 (939dee8dc2, 2026-09-04) step one's label is keyed by `readiness.blocker`; the test helper builds a label-only `ConsoleSettingsReadiness`, whose legacy normalization sets every blocked label to `blocker="readiness_unknown"` ("Finish provider setup").
2. Fix the tests: build the blocked readiness with the production builder (`build_console_settings_readiness`) for a missing key, an invalid endpoint, an unsaved endpoint, no provider and an unsupported provider, so the nodes pin the labels production shows.
3. Lane: replace the file's single-node entry in the PR Fast Lane with the whole file.
4. Negative control: a temporary wrong label in `_STEP_ONE_LABELS` turns a fixed node red.

## Implementation Notes

**Cause.** Test drift, not a product fault. #2365 (939dee8dc2, 2026-09-04) keyed `_STEP_ONE_LABELS` by `readiness.blocker` instead of `readiness.label`, and gave `ConsoleSettingsReadiness` a legacy normalization: a readiness built from a label alone with `native_send_supported=False` gets `blocker="readiness_unknown"`, whatever the label says. The file's `_readiness("Missing key")` / `_readiness("Invalid URL")` helpers therefore all read "Finish provider setup". The test file was not touched by #2365, and only one of its nodes ran in a lane, so the two nodes stayed red for 2.5 months.

**Fix (tests only).** A `_blocked(setup)` helper builds the readiness with the production builder (`build_console_settings_readiness`) for five named setups: missing key (openai, no key), invalid endpoint (llama.cpp `api_url = "not a url"`), unsaved endpoint (vLLM session URL differing from the saved one), no provider, unsupported provider. A probe through the wrapper confirmed the builder's labels/blockers for each (`Missing key/credential_missing`, `Invalid URL/endpoint_invalid`, `Endpoint not saved/endpoint_not_saved`, `Unknown/provider_missing`, `Unknown/provider_unsupported`). `_build()`'s default and the two `console_setup_is_blocking` missing-key cases use `_blocked("missing_key")`; `_readiness` keeps only ready readiness and its docstring says why.

**Evidence.** Base dc371f0842 (`-n 4`, load ~12): the two nodes fail ("Finish provider setup"). After: 19 passed. Negative control: `credential_missing` and `endpoint_invalid` set to "WRONG LABEL" in `_STEP_ONE_LABELS` turned both fixed nodes red (`assert 'WRONG LABEL' == 'Connect a pr...local server)'`, `... == 'Save the pro...ss (endpoint)'`), 2 failed / 17 passed; reverted.

**Lane.** The PR Fast Lane's single-node entry for this file (`derived-artifacts.yml`) is now the whole file (pure, 19 tests, about 3 s at `-n 4`), mirrored in `Tests/CI/test_ci_queue_pressure_contract.py`'s target list.

Files: `Tests/Chat/test_console_onboarding_state.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
