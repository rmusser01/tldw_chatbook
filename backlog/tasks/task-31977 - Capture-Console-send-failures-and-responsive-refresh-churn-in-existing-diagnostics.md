---
id: TASK-31977
title: >-
  Capture Console send failures and responsive refresh churn in existing
  diagnostics
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-07 21:58'
updated_date: '2026-09-07 22:26'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reports of blocked sends and flickering can reach support with no useful diagnostic evidence. Extend the existing metadata-only logs and responsiveness monitor so failures before trace setup and excessive refresh activity can be identified from the normal shareable logs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh Console sends record metadata-only stages and outcomes before trace setup, with safe failure categories.
- [x] #2 Rapid refresh or recompose activity is diagnosed even when the event loop remains responsive, without unbounded logging or loop-side file writes.
- [x] #3 The existing persistent log and Logs Copy all export retain diagnostic events, runtime version and capture state without private content.
- [x] #4 Targeted real-sink, mounted-flow, privacy and failure-isolation checks pass; recovery behavior is unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Amend ADR-029 for bounded Console lifecycle and UI churn metadata; reuse the current diagnostic sink, collector and monitor drain. ADR required: yes (amend existing backlog/decisions/029-local-private-data-boundary.md). Reason: admit explicit operational event fields and document off-loop delivery, privacy and retention.
2. Add failing real-file and Copy all tests for pre-trace send failure, trace failure categorization and responsive refresh churn; verify privacy and failure isolation.
3. Add a lazy per-send diagnostic scope, stage breadcrumbs and typed failure summaries; extend the existing monitor with bounded off-loop diagnostic delivery and churn detection.
4. Verify mounted sends, existing recovery and responsiveness behavior, privacy, inventory and boot-budget checks; update troubleshooting documentation and task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented metadata-only Console send-stage events from the visible action through controller admission, durable commit, trace reservation/dispatch and provider entry. Failure summaries preserve safe categories, allowlisted trace reason codes, SQLite codes, runtime versions and resolved capture state at WARNING logging thresholds. Each subsequent queued controller submission receives its own diagnostic token and event allowance.

Extended the existing UI responsiveness monitor with bounded off-loop delivery and edge-triggered Console-sync / screen-recompose churn events, including the last send stage. Added bounded teardown draining. The normal persistent file and Logs Copy all use their existing privacy boundary and retention; no new export surface or capture policy change.

ADR required: yes, fulfilled by the TASK-31977 amendment to backlog/decisions/029-local-private-data-boundary.md. Added backlog/docs/console-send-diagnostics.md and an incident-backed testing lesson. No new ADR file, database schema or dependencies.

Validation: 133 diagnostic, mounted Console, trace-recovery/runtime, responsiveness and share-path tests passed; 73 gateway trace/capture tests passed; 12 startup/import/CSS budget checks passed (218 total across those runs). New files and the monitor pass Ruff; changed ranges format cleanly; baseline-relative lint found zero introduced findings (917 prior, 915 current in the legacy files). Diagnostic inventory verifies unchanged without regeneration; git diff --check passes. Independent review findings were reproduced RED and fixed; follow-up review found no remaining actionable issue.

Known baseline verification limitation: Tests/test_persistent_diagnostic_boundary.py::test_persona_workspace_diagnostics_do_not_interpolate_private_values fails on an unchanged MCP logging call. Reproduced against the original dev archive; the remaining 47 tests in that targeted run passed. The repository also has pre-existing whole-file lint debt. Task remains In Progress rather than claiming the all-green Definition of Done. The original reporter-specific trace failure/flicker has not been reproduced or fixed by this diagnostic extension.

Prepared together with the previously verified TASK-31976 recovery fix for the user-requested PR against dev. Full suite not run per repository policy; targeted validation and the baseline limitations above accompany the PR.
<!-- SECTION:NOTES:END -->
