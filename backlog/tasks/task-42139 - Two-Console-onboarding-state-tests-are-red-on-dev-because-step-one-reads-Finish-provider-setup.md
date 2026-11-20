---
id: TASK-42139
title: "Two Console onboarding-state tests are red on dev because step one reads Finish provider setup"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:06
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
- [ ] #1 The cause is named and fixed, so both nodes pass and still pin the label production shows for a missing key and for an endpoint problem
- [ ] #2 Both nodes run in a PR lane
- [ ] #3 A negative control (a temporary wrong label in `_STEP_ONE_LABELS`) turns a fixed node red; the notes record it
<!-- AC:END -->
