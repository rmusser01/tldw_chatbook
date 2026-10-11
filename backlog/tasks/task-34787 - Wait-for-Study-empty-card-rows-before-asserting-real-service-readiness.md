---
id: TASK-34787
title: Wait for Study empty card rows before asserting real service readiness
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-11 00:18'
updated_date: '2026-10-11 00:21'
labels: []
dependencies: []
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required PR UI lane can observe a newly selected deck while its queued real deck-change worker is still rebuilding card rows. The local real-service contract case must observe the expected rendered empty state before asserting it, so its existing positive service and database checks qualify completed UI work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The existing local real-service case waits for the actual empty card-list label while the natural deck-change worker may still be pending, before retaining its empty-label assertion.
- [x] #2 All existing deck and card database, service-signature, workspace-scope and painted-row assertions remain intact; the two-case gated module passes without production changes, extra test nodes, retries or longer timeouts.
- [ ] #3 The same held real-worker control records the original mounted-but-empty assertion failure and passes after the readiness correction; scoped static checks and independent review qualify the test-only change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the required UI4 failure at published 9d178607 and compare the immutable 449c2c205e baseline. Use one outside-repository scheduling control that holds the natural Select.Changed worker after the actual StudyScopeService/SQLite read; confirm selected deck and mounted-but-empty rows precede the original assertion failure.
2. Make the existing gated local real-service case wait through its existing _wait_until helper for the expected empty card-list label before retaining the original assertion. Keep its DB, card creation, rendering and service-contract assertions unchanged; add no sleep, timeout, retry, production guard or new test node.
3. Run the identical held-worker control GREEN and the two-case gated real-service module with unique temporary data. Compare scoped static diagnostics to the immutable parent, record evidence and limitations, and obtain independent review before closeout.
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md (existing)
Reason: test-only readiness correction within the existing profile-admitted real-service harness; no storage, runtime, UI or ownership boundary changes. Read TASK34000.6 and the existing mounted-row lesson in backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The existing local real-service case now waits for the expected empty card-list label through its existing _wait_until helper before retaining the original assertion. The six added lines preserve all eight existing assertions, both gated test nodes, the real StudyScopeService/SQLite checks, card creation and painted-row checks. No production, helper, timeout, retry or CI-census change was needed.

The actual required UI4 log first fails the empty-row assertion; its worker MountError occurs during assertion-driven run_test teardown. An outside-repository scheduling control holds the natural Select.Changed worker after the actual service/DB read: the deck is selected and the list mounted/attached, but labels are temporarily empty. Original published9d control: 1 FAIL, 0 ERROR, 3.84s. Genuine immutable449 baseline with the same control: 1 FAIL, 0 ERROR, 4.13s; Study source/test modes and blobs match9d. Repaired identical control: 1 PASS, 3.73s, with the same initial empty observation. Ordinary two-case module: 2 PASS, 0 ERROR/SKIP, 5.13s. Receipts: /private/tmp/pr2882-study-pending-{red,baseline449,green}.{log,xml,json}, /private/tmp/pr2882-study-module-green.{log,xml}, /private/tmp/pr2882-study-results.json and /private/tmp/pr2882-study-source-static-proof.json.

Scoped fatal Ruff, formatter and diff checks pass. Full Ruff retains exactly the parent's three diagnostics (I001/S110/BLE001); no new diagnostic. The incident matches the existing mounted-row readiness lesson for TASK14904 in backlog/docs/lessons-testing-evidence.md. ADR required: no; existing backlog/decisions/126-complete-local-backup-and-recovery.md profile admission and ownership remain intact. No Study navigation lifecycle or full-suite claim is made. Independent review and root preflight/closeout remain pending; status stays In Progress and AC3 unchecked.
<!-- SECTION:NOTES:END -->
