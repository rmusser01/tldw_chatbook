---
id: TASK-16077
title: Reconcile Console request test contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 05:19'
updated_date: '2026-08-14 05:23'
labels:
  - testing
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair stale Console integration fixtures that still treat prepared requests as raw message lists or return partial provider-resolution doubles, so the current request and wake-approval contracts are exercised rather than timing out.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prepared request assertions inspect the canonical message payload
- [x] #2 Wake approval tests use the complete provider-resolution contract
- [x] #3 Focused Console request and wake-safety tests pass without production changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the three isolated RED failures and identify the current request/resolution contracts.
2. Update only the stale test assertions and shared gateway double.
3. Re-run the named regressions and adjacent focused modules, then complete static/diff checks.

ADR required: no
ADR path: N/A
Reason: test-only reconciliation to existing provider-request and approval contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

- Updated the private-history assertion to inspect `PreparedProviderRequest.messages_payload`, matching the canonical request object already used by adjacent tests.
- Replaced the shared scripted gateway's partial ad-hoc resolution object with `ConsoleProviderResolution`; this restored the real wake approval path instead of ending the turn before review.
- Verified the three isolated regressions and the complete 498-test Console agent/fleet feature gate. No production file changed for this task.
