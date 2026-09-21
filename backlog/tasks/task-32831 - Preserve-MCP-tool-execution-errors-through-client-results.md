---
id: TASK-32831
title: Preserve MCP tool execution errors through client results
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-19 02:30'
updated_date: '2026-09-21 00:18'
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
- [x] #5 Malformed non-boolean isError values are rejected as bounded failures, never successful invocations, without logging response bodies.
- [x] #6 The required CI lifecycle regression finishes Home startup with Canvas unwarmed before testing shutdown, without depending on splash duration or changing production behavior.
- [x] #7 The serial required Fast Lane retains its test targets and required-gate relationship with a bounded job budget sufficient for both measured pytest steps.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-111 and ADR-161 apply. Reason: restore existing MCP tool-result failure semantics without a new API, transport, permission, storage or UI boundary. Resume saved PR2716 on merged PR2714 dev e4096e2059. 1. Reproduce the saved real-stdio failures on merged dev, then integrate the minimal client repair. 2. Validate fixture paths and malformed requests using current shared boundary helpers and focused regressions. 3. Replace the obsolete native executable with a supported current-dev runner; use private roots, actual import provenance, terminal warmup, network guard, exclusive evidence and owned cleanup. 4. Requalify failure/audit/retry in dark/light wide views; attempt compact reachability and retain any limitation without UI scope expansion. 5. Run focused tests, artifact/static checks and independent review, update the existing draft PR with fresh evidence and owner gallery. Current-head CI, accumulated review and fresh owner visual approval gate merge.
Qodo follow-up: reproduce non-boolean isError success fallthrough with real stdio cases; validate the existing flag using a strict private boundary model and discard validator details before logging. Preserve current content/result behavior, prove a corrected retry works, and repeat affected tests/native evidence. No new ADR or transport interface.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved MCP isError through the stdio result and translated true flags into the existing client error shape, extracting only nonblank text and using a generic fallback. Successful shapes and connection lifetime remain unchanged; the existing inspector/audit path now reports failures accurately. Real stdio tests cover text/mixed/empty results, absent/false flags, safe logging/audit metadata and retry.

Verification: 11 targeted cases pass; six regressions failed on unchanged dev. Two older control-plane setup failures reproduce unchanged on dev and are retained as debt. Four inspected native dark/light captures at170x48 show failure then success on one connection; app/child exit, ten DBs, lock, default fingerprints and source hashes pass. Seven preflight guards pass; new files lint/format and changed production ranges pass, with no added baseline Ruff diagnostics. Independent read-only review found no actionable issue.

Plan deviation: 80x24 execution qualification failed because the argument field could not be revealed, even after explicit scrolling. Narrowed this client-only slice to wide execution; retained both failed attempts and made compact inspector reachability the next review. No UI layout changes were bundled. Evidence: Docs/superpowers/qa/2026-09-18-mcp-tool-errors/README.md; both design-system review ledgers updated. ADR required: no; existing ADR-111 and ADR-161 apply. Separate draft PR against dev; owner visual approval remains required before merge.

Integrated saved PR2716 on merged PR2714 dev e4096e2059. Six execution regressions reproduce before repair; 67 distinct targeted cases pass, including isolated malformed-content, real stdio/audit/retry, both fixture CLI boundaries, runner ownership and transport neighbors. Eleven fixture boundary failures reproduced before explicit-root/path and strict request validation. All eight artifact guards pass, no added baseline Ruff diagnostics; independent production/fixture and native-runner reviews clear. Native dark/light 170x48 passes with eight inspected SVGs (four identical feedback/settled pairs), same live connection, actual failure/success audit, clean exit and healthy private databases. Fresh 80x24 attempt fails earlier than historical evidence: Test Tool is offscreen despite focus/scroll; exit1 and clean owned cleanup are retained, compact execution unqualified. User defaults and fixture sentinels unchanged, source/runner hashes match, zero network. Existing ADR-111/161 apply. Current QA: Docs/superpowers/qa/2026-09-18-mcp-tool-errors/current-dev/README.md. Keep In Progress pending current-head CI/review, fresh owner visual approval and merge.

Qodo identified malformed non-boolean isError values falling through as success. Extending the existing tool-result boundary to reject those values with a fixed malformed-response failure; targeted wire regressions and fresh native qualification will be recorded before merge.

Qodo malformed-flag follow-up complete: shared MCPToolResultInput validates the existing isError flag strictly; the client suppresses validator/body details behind a fixed error and preserves the session. Eight real-stdio regressions fail before repair; 22 affected cases pass afterward, bringing the distinct targeted inventory to 75. Eight artifact guards and baseline-relative static checks pass; independent review is clear. Final wide-002 native runs pass in both themes, with all eight PNG renders identical to the inspected prior captures. Compact-002 again fails before execution at offscreen Test Tool; the fresh capture was inspected. Both app/fixture lifecycles, private databases, locks, unchanged defaults and exact source hashes verify. Final exports and manifest replace current qualification while the previous run remains immutable at 8b7be87978. No new ADR. Await current-head CI, accumulated Qodo disposition and fresh owner visual approval before merge.

Current-head CI run35545242345 passed1152 Fast Lane cases but failed the sync-constructed canvas watcher test: splash closure queued ChatScreen mounting during runtime disposal, leaving its store absent. The same124-case admission suite passed on the preceding head and the individual test passes locally; fixed startup timing is required rather than a blind rerun. Plan: prove the missing mounted-Console precondition, disable splash in this test via persisted private config and await the existing Console selector before watcher assertions; run the isolated regression and its admission-sensitive module, independently review the test-only correction, then rerun current-head CI. ADR required:no; test harness repair only, existing lifecycle behavior unchanged. Owner visual approval recorded for19ceab4dc3; MCP code and gallery remain unchanged.

CI budget plan: current run measured885.51s main tests plus299.84s admission tests and roughly75s setup, exceeding20min; preceding run passed18m19s. Raise the existing serial job bound to30min and update its exact contract/spec timeout, preserving every target, per-test timeout, runner count, dependency boundary and required gate. ADR required:no new ADR; existingADR103 applies because only the operational timeout changes. Test startup correction uses a delegated getter override rather than persisted config because this module shares its bootstrap profile; await the actual startup task and Console selector before watcher assertions.

CI follow-up verified: select real Home startup, scoped splash getter override (no shared-profile writes), await actual startup task and Home header, retain every unwarmed-Canvas/watch/disposal assertion.123 admission-sensitive passes plus1 existing expected failure;26CI contracts pass; isolated case passes. Job bound30min preserves all targets, per-test deadlines, serial runner and required gate. Seven artifact guards pass in sandbox; Mermaid succeeds separately with network-enabled pinned-input fetch. Zero introduced lint diagnostics; changed ranges format clean. Independent review clear after fixing profile isolation and all spec timeout literals. ExistingADR103 applies alongside111/161; no new ADR. Owner approved19ceab4dc3 gallery; all production/native hashes unchanged. Evidence: current-dev/ci-followup. Await new current-head CI/review, then merge without another visual prompt.
<!-- SECTION:NOTES:END -->
