---
id: TASK-31714
title: Fix Console second-send trace surface rejection found during snapshot UAT
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:24'
updated_date: '2026-09-05 19:46'
labels: []
dependencies: []
documentation:
  - backlog/docs/validation-31714-console-trace-snapshot-handoff.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore ordinary multi-turn Chatbook sends with capture enabled after live validation found a pre-dispatch trace failure even without snapshot operations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Two ordinary sends succeed through the real preparation and trace boundary with capture enabled.
- [x] #2 Historical trace remains reconstructable and mismatched or unauthorized surface changes remain fail-closed.
- [x] #3 Targeted regressions and disposable live snapshot and tool-approval handoff UAT are recorded without opening production snapshot support.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the exact synthetic incoming versus durable trace-surface mismatch on current dev and trace it to its preparation/settlement owner. 2. Add a real SQLite and provider-boundary regression that fails before the smallest correction; retain negative provenance and ownership controls. 3. Run targeted trace/Console tests, lint/format and scoped security checks. 4. Re-run disposable two-send control, then Admin save/restart/restore and pending-tool approval UAT; record measured native reuse and cleanup. ADR required: no new ADR. ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Reason: restore existing reference-backed, fail-closed trace contract; do not widen its ownership/privacy boundaries or open production snapshot support.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renumbered this newly created record from TASK-31701 before implementation: all-ref/worktree sweep found older unrelated owners of 31701 and maximum allocated ID 31713.

Implemented the minimal visible-value correction in ConsoleChatController: omit native/persisted identity annotations from trace comparison and frozen provider rows, retaining native-owner lookup and exact content matching under ADR-097. Added a production SQLite/store/controller/gateway/agent trace regression with RED/GREEN evidence, equal-text distinct owners, changed-history fail-closed control, and native-reader before/after equality. Independent review has no blocking findings; its reconstruction suggestion was implemented. Targeted final339passed1deselected1warning; one catalog count assertion independently fails unchanged dev and remains out of scope. Ruff has zero introduced diagnostics against192 inherited findings; changed ranges are format-clean and git diff check passes. Actual mounted two-send control and save/restart/restore now demonstrate6031 cached/27processed versus cold0/6058, unchanged messages/durable state/settings, no autosends and unchanged Pause/Resume behavior. Tool approval survives restoration but continuation hits a separate trace_turn_unavailable error also reproduced without lifecycle operations; no tool-loop success or production enablement claimed. Evidence and exact limitations: backlog/docs/validation-31714-console-trace-snapshot-handoff.md. Added testing lesson about real factory coverage. Owned test runtime stopped and two disposable synthetic copies deleted; ten receipts retained. Shared dirty checkout untouched. No PR/push requested.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Ordinary capture-on multi-turn sends fixed and measured snapshot reuse verified. Production factory regression, negative controls and immutable reconstruction pass. Separate tool-chain ownership blocker documented; server snapshot acceptance and production allowlist remain gated.
<!-- SECTION:FINAL_SUMMARY:END -->
