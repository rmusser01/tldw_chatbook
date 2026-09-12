---
id: TASK-22303
title: Restore priced Console cost-chip harness readiness
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 04:36'
updated_date: '2026-08-26 04:41'
labels: []
dependencies:
  - TASK-22527
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the priced Console cost-chip integration harness persist its fake Anthropic credential through the same configuration source that mounted readiness refreshes, so the tests exercise cost tracking instead of blocking at provider setup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Anthropic cost-chip helper remains send-ready after mount-time configuration refreshes.
- [x] #2 Priced, cache-warm, cache-expiry, modal, and session-isolation cost-chip tests reach their fake gateways without real network or credential access.
- [x] #3 The complete Console cost-chip screen test module passes without changing production behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the priced-send test failure as RED evidence and trace the provider-readiness config source used after mount.
2. Persist the fake Anthropic credential through the sandboxed config API while retaining the existing app snapshot provider/model setup.
3. Run the first priced-send test, the complete cost-chip screen module, the original 174-test baseline slice, and static/format checks.
4. Record implementation evidence and complete task hygiene.

ADR required: no
ADR path: N/A
Reason: This is a test-only configuration-source correction using an established config API; no production contract or architecture changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated the Anthropic cost-chip test helper to atomically persist its fake provider/model defaults and fake API key into the per-test sandbox config while retaining the app snapshot values used during mount. This matches Console's established disk-loaded readiness refresh behavior, so mount-time config writes no longer erase the harness's apparent credential. Removed one pre-existing unused import and normalized the touched file with Ruff; production code, real credentials, and network paths are unchanged. ADR required: no; ADR path: N/A; Reason: test-only use of the existing sandboxed configuration API. TDD evidence: the priced-send screen test failed before the repair with `Missing key` and never reached the fake reply, then passed after persistence was added. Verification: first priced send 1 passed; complete cost-chip screen module 15 passed; original focused baseline slice 174 passed; Ruff lint and format checks passed; `git diff --check` passed.
<!-- SECTION:NOTES:END -->
