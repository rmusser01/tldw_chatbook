---
id: TASK-32831
title: Preserve MCP tool execution errors through client results
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 02:30'
updated_date: '2026-09-20 23:21'
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
ADR required: no. ADR path: N/A; existing ADR-111 and ADR-161 apply. Reason: restore existing MCP tool-result failure semantics without a new API, transport, permission, storage or UI boundary. Resume saved PR2716 on merged PR2714 dev e4096e2059. 1. Reproduce the saved real-stdio failures on merged dev, then integrate the minimal client repair. 2. Validate fixture paths and malformed requests using current shared boundary helpers and focused regressions. 3. Replace the obsolete native executable with a supported current-dev runner; use private roots, actual import provenance, terminal warmup, network guard, exclusive evidence and owned cleanup. 4. Requalify failure/audit/retry in dark/light wide views; attempt compact reachability and retain any limitation without UI scope expansion. 5. Run focused tests, artifact/static checks and independent review, update the existing draft PR with fresh evidence and owner gallery. Current-head CI, accumulated review and fresh owner visual approval gate merge.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved MCP isError through the stdio result and translated true flags into the existing client error shape, extracting only nonblank text and using a generic fallback. Successful shapes and connection lifetime remain unchanged; the existing inspector/audit path now reports failures accurately. Real stdio tests cover text/mixed/empty results, absent/false flags, safe logging/audit metadata and retry.

Verification: 11 targeted cases pass; six regressions failed on unchanged dev. Two older control-plane setup failures reproduce unchanged on dev and are retained as debt. Four inspected native dark/light captures at170x48 show failure then success on one connection; app/child exit, ten DBs, lock, default fingerprints and source hashes pass. Seven preflight guards pass; new files lint/format and changed production ranges pass, with no added baseline Ruff diagnostics. Independent read-only review found no actionable issue.

Plan deviation: 80x24 execution qualification failed because the argument field could not be revealed, even after explicit scrolling. Narrowed this client-only slice to wide execution; retained both failed attempts and made compact inspector reachability the next review. No UI layout changes were bundled. Evidence: Docs/superpowers/qa/2026-09-18-mcp-tool-errors/README.md; both design-system review ledgers updated. ADR required: no; existing ADR-111 and ADR-161 apply. Separate draft PR against dev; owner visual approval remains required before merge.

Integrated saved PR2716 on merged PR2714 dev e4096e2059. Six execution regressions reproduce before repair; 67 distinct targeted cases pass, including isolated malformed-content, real stdio/audit/retry, both fixture CLI boundaries, runner ownership and transport neighbors. Eleven fixture boundary failures reproduced before explicit-root/path and strict request validation. All eight artifact guards pass, no added baseline Ruff diagnostics; independent production/fixture and native-runner reviews clear. Native dark/light 170x48 passes with eight inspected SVGs (four identical feedback/settled pairs), same live connection, actual failure/success audit, clean exit and healthy private databases. Fresh 80x24 attempt fails earlier than historical evidence: Test Tool is offscreen despite focus/scroll; exit1 and clean owned cleanup are retained, compact execution unqualified. User defaults and fixture sentinels unchanged, source/runner hashes match, zero network. Existing ADR-111/161 apply. Current QA: Docs/superpowers/qa/2026-09-18-mcp-tool-errors/current-dev/README.md. Keep In Progress pending current-head CI/review, fresh owner visual approval and merge.
<!-- SECTION:NOTES:END -->
