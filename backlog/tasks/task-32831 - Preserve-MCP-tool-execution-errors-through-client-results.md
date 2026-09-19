---
id: TASK-32831
title: Preserve MCP tool execution errors through client results
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 02:30'
updated_date: '2026-09-19 02:45'
labels:
  - mcp
  - ui
  - correctness
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A server-reported tool failure must appear as a failed invocation in Test Tool and Audit, while leaving the connection available for a corrected retry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP isError true results reach the existing failure path with text-only details or a generic fallback.
- [x] #2 Absent or false isError preserves successful result shapes, and a tool error does not disconnect the server.
- [x] #3 The real control plane records failed tool outcomes without persisting error-body secrets and accepts a subsequent successful invocation.
- [x] #4 Targeted regressions and private native dark/light execution at 170x48 verify visible failure and successful retry; compact inspector reachability failures remain explicitly unqualified in the review ledger.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the native false-OK/false-success-audit reproduction and add isolated real-stdio client/control-plane regressions.
2. Preserve the protocol error flag and map it to the existing client error shape using only text content with a generic fallback; retain success and connection behavior.
3. Run targeted transport/service neighbors, independent review, and native Test Tool failure/retry at 170x48 in dark/light. Verify audit metadata and shutdown. Retain the observed 80x24 inspector reachability failure for a separate UI slice.
4. Save evidence, update review ledgers, and create a separate draft PR against dev.

ADR required: no
ADR path: N/A (existing ADR-111 and ADR-161 apply)
Reason: Corrects propagation of an existing MCP protocol result through established client failure and UI/audit paths; no transport, service interface, permission, storage or application-structure change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved MCP isError through the stdio result and translated true flags into the existing client error shape, extracting only nonblank text and using a generic fallback. Successful shapes and connection lifetime remain unchanged; the existing inspector/audit path now reports failures accurately. Real stdio tests cover text/mixed/empty results, absent/false flags, safe logging/audit metadata and retry.

Verification: 11 targeted cases pass; six regressions failed on unchanged dev. Two older control-plane setup failures reproduce unchanged on dev and are retained as debt. Four inspected native dark/light captures at170x48 show failure then success on one connection; app/child exit, ten DBs, lock, default fingerprints and source hashes pass. Seven preflight guards pass; new files lint/format and changed production ranges pass, with no added baseline Ruff diagnostics. Independent read-only review found no actionable issue.

Plan deviation: 80x24 execution qualification failed because the argument field could not be revealed, even after explicit scrolling. Narrowed this client-only slice to wide execution; retained both failed attempts and made compact inspector reachability the next review. No UI layout changes were bundled. Evidence: Docs/superpowers/qa/2026-09-18-mcp-tool-errors/README.md; both design-system review ledgers updated. ADR required: no; existing ADR-111 and ADR-161 apply. Separate draft PR against dev; owner visual approval remains required before merge.
<!-- SECTION:NOTES:END -->
