---
id: TASK-31810
title: Restore guarded outbox corruption fixture
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06'
updated_date: '2026-09-06 02:56'
labels:
  - tests
  - sync
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The illegal non-assistant state fixture stops at the semantic mutation guard
before testing rejection of corrupt persisted source proof. Restore deliberate
fixture corruption without disabling or changing the production guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A valid source proof is readable before fixture corruption; the persisted illegal user state is rejected after reopening the database.
- [x] #2 Direct mutation is rejected outside a narrowly scoped fixture-owned authorization, including after the injection.
- [x] #3 Complete outbox and semantic-guard files pass with unchanged production code; static checks and independent review qualify the change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the outbox file and trace the existing test-only corruption authorization pattern. The two-file baseline is 3 failed/51 passed; only one failure belongs to this outbox repair.
2. Keep the original malformed-state assertion, prove the valid precondition, and use the existing private message-update authorization only to inject the fixture corruption. Verify persisted corrupt state and rejection after reopening, then check the mutation guard is still active. Keep production validation and triggers unchanged except the fixture's existing sync-trigger removal.
3. Run the complete outbox and semantic-mutation-guard files, lint/format/diff checks and independent review. Record the two retention-delete failures separately; do not mask them or claim full-suite completion.

ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Test-only corruption injection restores coverage of an existing fail-closed contract without changing mutation authority or persistence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The fixture now verifies a valid source proof, proves ordinary SQL mutation is
rejected, and injects the deliberately illegal user-row state under the existing
private authorization for one message and `message_update` only. A rejected
write after leaving that scope proves authorization cleanup on the same handle.
Closing and reopening verifies that the illegal state really persisted at
version 1 before the original source-proof rejection assertion runs. The
fixture's pre-existing sync-trigger removal preserves the original valid log.
Production code and semantic guard triggers remain unchanged; ADR-097 applies
without a new architectural decision.

The original two-file baseline recorded 3 failures/51 passes, one in this outbox
fixture and two unrelated hard-delete retention probes
(`/private/tmp/tldw-guarded-sync-baseline.xml`). The repaired complete outbox and
v57 semantic-guard files pass 45 tests with two existing dependency warnings
(`/private/tmp/tldw-outbox-guard-final.xml`). Scoped Ruff formatting, full-file
lint and diff checks pass. Whole-file formatting reports pre-existing unrelated
drift; only the modified test was formatted. Independent review found no
actionable issues. The two retention-delete failures and the broader review
remain open; no production or full-suite fix is claimed.
<!-- SECTION:NOTES:END -->
