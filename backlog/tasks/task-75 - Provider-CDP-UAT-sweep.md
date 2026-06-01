---
id: TASK-75
title: Provider CDP UAT sweep
status: Done
assignee: []
created_date: 2026-06-01 00:46
updated_date: 2026-06-01 00:47
labels:
- qa
- providers
- console
- cdp
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify every testable Chatbook Console provider through rendered Textual-web/CDP using isolated config and redacted provider credentials.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime provider inventory is extracted from Chatbook code
- [x] #2 Inventory records model source and local endpoint reachability
- [x] #3 Hosted providers with usable keys are tested through CDP
- [x] #4 Local/custom providers are skipped unless endpoint is reachable
- [x] #5 Each passed provider receives a second assistant reply in the same Console session
- [x] #6 External failures are classified separately from Chatbook defects
- [x] #7 Raw API keys do not appear in evidence
- [x] #8 QA report and residual risks are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Build redacted provider inventory and isolated CDP launch helpers.
2. Generate provider inventory and QA report skeleton.
3. Launch Chatbook through Textual-web/CDP with isolated HOME/XDG config/data.
4. Run a manual two-turn provider sweep through the rendered app.
5. Fix and rerun only Chatbook-caused provider defects.
6. Record evidence, residual risks, and task closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Built and ran a redacted Textual-web/CDP provider UAT sweep against the isolated QA profile using keys from the adjacent tldw_server2 env file. Added deterministic UAT model config, dropdown-backed Console Settings model selection, expanded provider options to all Console-sendable handlers, and fixed Mistral readiness fallback to `MISTRAL_API_KEY`.

The final full sweep attempted 11 hosted providers with usable keys: 7 reached the second assistant reply and 4 were classified as external/provider failures or timeouts. Local/custom providers were skipped from inventory because their endpoints were unreachable, oobabooga lacked an explicit model, and zai lacked a usable key. Evidence and residual risks are recorded in `Docs/superpowers/qa/provider-cdp-uat/2026-05-31-provider-cdp-uat.md` and `provider-sweep-results.json`.

Verification: targeted Chat/UI/QA pytest suite passed, JS harness syntax checks passed, `git diff --check` passed, and text secret scan over QA artifacts found no raw key patterns outside test fixtures/redaction regexes.
<!-- SECTION:NOTES:END -->
