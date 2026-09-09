---
id: TASK-32040
title: Preserve selected private history when Console trace capture is enabled
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 06:18'
updated_date: '2026-09-08 07:41'
labels:
  - console
  - trace
  - bug
dependencies: []
references:
  - backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Capture On currently discards matching saved provider continuation history before dispatch even though the same conversation replays it with Capture Off. Preserve provider behavior and exact saved ownership across capture modes, including affected successor requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Matching retained continuation and thinking history reaches the provider with the same values under Capture On and Capture Off.
- [x] #2 Trace reconstruction retains the exact saved owner and selected private history without exposing it in diagnostics or admitting foreign, changed or unsupported history.
- [x] #3 Ordinary and transformed successor sends remain valid with retained private history, including cold reconstruction; targeted real-controller and real-database tests cover the behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: Repair the existing selected-sidecar replay and exact saved-owner provenance contract; preserve the storage, privacy and provider boundaries.

1. Add failing real-controller and SQLite capture-on/off parity tests for matching saved private continuation history, then cover thinking replay and rejected owners.
2. Preserve the selected owner bindings while preparing captured semantics, reuse existing private-history selection and attachment behavior, and prevent a prepared request from silently dropping selected sidecars.
3. Verify provider payloads and trace reconstruction agree, and repair any demonstrated bounded successor transition involving unchanged sidecars without broadening artifact authority.
4. Run targeted gateway, preparation, trace and controller regressions plus baseline-relative lint, changed-range format, inventory and diff checks; update task notes and user documentation.

Implementation location: /private/tmp/tldw-trace-fix-32029, based on dev 3cccd9326c556a245fc87ab5013c192242e389cf. The older main workspace holds this task record for tracking; the complete Console patch stays in the isolated dev checkout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Captured requests now retain eligible saved continuation and thinking history with the same provider values as Capture Off. Preparation binds the selected private history to its exact saved revision; admission verifies raw values before credential filtering. Canonical checkpoint decoding restores the existing typed representation without weakening issuance identity, and thinking rewrites retain their exact saved owner.

Warm and cold successors preserve selected history across ordinary and transformed turns. Supported local displayable and hosted proprietary thinking responses use a frozen response projection to verify their complete saved response. A process-local keyed proof compares raw values before filtering and never enters durable storage or diagnostics. New response revision and artifact writes honor frozen PII policy, including older unprofiled headers; failed PII detection omits affected response data. Historical records retain their original interpretation, and request and response mask domains remain distinct.

The controller, preparation, trace runtime/service, semantic revision, settlement and reader changes are covered by real SQLite/controller/gateway regressions in test_console_trace_sidecar_replay.py, test_console_trace_sidecar_redaction.py, test_console_trace_thinking_settlement.py and test_console_trace_proprietary_settlement.py. Positive capture-mode, transformed, cold reconstruction and actual typed-provider response cases accompany negative owner, credential-equivalent forgery, malformed profile and PII controls. ADR required: no new ADR; amended backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md documents repairs to the existing replay, response and privacy contracts. The source-pin schema migration is recorded by TASK-32032. Console documentation and incident-based testing lessons were updated.

Final verification: 372 passed in 13 targeted trace modules, with four exact failures independently reproduced on unchanged dev and excluded (one provider-shape, two oversized-response and one cold-recovery fixture case). Adjacent preparation, gateway, thinking and continuation checks: 517 passed, including the two localhost checks rerun with listener access. All six derived preflight checks passed. Baseline-relative Ruff across 34 changed Python files reports 1,015 inherited/current findings and zero new findings; all ten new test files pass formatting and git diff --check HEAD passes. Independent review found no remaining actionable defect. Tests use real application boundaries with inference replaced; no live profile or full repository suite was used.

Implementation is in /private/tmp/tldw-trace-fix-32029 at dev base 3cccd9326c556a245fc87ab5013c192242e389cf; the older main workspace holds the tracking record. No commit, merge or push was performed. This task was renumbered from TASK-32039 to TASK-32040 after a final cross-ref/worktree audit found concurrent claims to the earlier ID.
<!-- SECTION:NOTES:END -->
