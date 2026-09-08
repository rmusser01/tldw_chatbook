---
id: TASK-32048
title: Preserve Console trace capture when tool discovery changes the system prompt
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 17:24'
updated_date: '2026-09-08 19:29'
labels:
  - console
  - trace
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The llama.cpp agent protocol embeds loaded tool schemas in the leading system message. A successful find_tools and load_tools sequence changes that message while appending tool traffic, which the trace surface planner rejects as unsupported_surface_change before the third provider request. Preserve the exact per-call system content and trace history while admitting this valid continuation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A real SQLite/controller/agent/gateway run discovers and loads a tool, completes the third llama.cpp request with Capture On, and preserves every captured provider request.
- [x] #2 Subsequent sends and owned retries remain usable; earlier captured calls remain immutable and credentials stay filtered.
- [x] #3 The trace ledger continues to reject unowned or changed saved history and keeps its bounded reference storage contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, amendment to existing ADR097. ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Reason: keep provider-owned rendered system content in per-call header components without changing provider input, existing surface nodes, storage schema or disclosure policy.
1. Preserve the failing real-factory discovery regression and add Capture Off, cold continuation and successive-send controls.
2. Admit only an existing leading rendered_system artifact slot as a per-call header override. Recheck slot identity, exact incoming row and frozen policy through preparation, final-value verification and atomic binding; saved revisions and other history retain exact matching.
3. Store the verified credential-filtered row in a rendered_system_row header component and reconstruct only that eligible slot from the call header. Older calls without the component retain current reads.
4. Verify original provider bytes, filtered trace reconstruction, unchanged historical calls, rollback/retry ownership and rejected history/provenance changes.
5. Run targeted trace and logging regressions, format/lint changed code and record evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserve provider-owned leading system messages in each call header when tool discovery changes loaded schemas. The positional source artifact and previous captures remain immutable; final row values, source identity and frozen policy are revalidated before atomic binding. Native reads overlay only the eligible first system slot and preserve older headers. No schema or provider-wire changes.

Reproduced the reported third-request unsupported_surface_change through real SQLite, Console controller, agent and gateway with mocked HTTP: find_tools then successful load_tools changed the system prompt. Regression coverage includes Capture On/Off, project instructions on/off, cold factories, following sends, rollback/owned retry, credential/PII filtering, malformed headers and rejected saved-history changes. The exact reporter tool sequence remains unconfirmed. Follow-up investigation also reproduced a later generic validation failure after reopening and discarding an interrupted tool run, including already-failed follow-up sends.

Validation: 295 targeted trace tests passed; 167 sanitizer/provider-log tests passed; all six scripts/preflight.sh checks passed. Modified Python ranges formatted and git diff --check clean. No introduced Ruff findings against HEAD; existing production-file lint debt remains unchanged. Independent review found no trace correctness issues; its suggested empty-delta case was added and passes. No full suite run.

ADR: amended backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Added the successful-discovery verification incident to backlog/docs/lessons-testing-evidence.md. Main code changes are console_trace_service.py and console_trace_native_reader.py, with real pipeline and focused header regressions.

Combined verification with discarded-run recovery on dev 7a7493e529: 407 targeted trace tests and 177 logging/UI tests passed. All six preflight checks pass; no introduced Ruff findings.
<!-- SECTION:NOTES:END -->
