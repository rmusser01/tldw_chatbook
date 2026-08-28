---
id: TASK-23019
title: Close Library adaptive-reader programme with cross-destination UAT
status: Done
assignee:
  - '@codex'
created_date: '2026-08-27 13:58'
updated_date: '2026-08-28 17:33'
labels:
  - library
  - ui
  - qa
dependencies:
  - TASK-22034
  - TASK-22857
references:
  - >-
    Docs/superpowers/specs/2026-08-27-library-adaptive-reader-programme-closeout-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close the Library adaptive-reader programme with reproducible cross-destination evidence that the shared shell and destination contracts remain correct together after all migrations, while allowing only localized repairs to already-approved behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media, Conversations, Notes, Prompts, and Skills pass one production-shaped automated cross-reader matrix covering retained pane identity, collapse, preferences, focus, selection/loading truth, stale settlement, and resize purity.
- [x] #2 All five destinations pass live containment, collapse, restoration, mode reachability, and selection checks at 160x50, 120x35, 100x30, and 80x24.
- [x] #3 Every stable ID in the bounded TASK-23019 closeout catalogue maps to at least one fresh automated result and one live journey from the exact recorded subject revision; earlier destination evidence is lineage rather than a substitute.
- [x] #4 Every declared writable config, profile, XDG, database, temporary, and raw-evidence path resolves inside scratch before application import; phase-scoped tripwires permit only declared read-only subject-checkout and resolved Python-runtime resources, scratch runtime writes, and the validated evidence-promotion destination; they record no prohibited filesystem or network attempt or checkout/runtime mutation; all harness-created database and host-worker owners close; and the raw scratch root is removed without reading or hashing real user-owned content.
- [x] #5 A sequential single-app route cycle proves destination preferences, drafts, selection, modes, focus, and asynchronous workers do not leak across readers.
- [x] #6 Any same-PR repair is a localized regression against ADR-086 or an approved destination contract, has a focused failing regression test, and introduces no new schema, ADR, service authority, capability, or redesign.
- [x] #7 The final automated and live matrices, required derived-artifact checks, targeted static checks, capability ledger, and bounded evidence README pass; the manifest identifies the exact tested subject revision, and the final branch differs only by retained evidence, task, ledger, and any incident-derived lessons documentation required by the repository Definition of Done.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add the bounded manifest and hermetic runner contracts
2. Add pre-import filesystem/network tripwires
3. Add the missing production-shaped live matrix and sequential route cycle
4. Run the curated automated matrix and classify any failures
5. Freeze and verify the exact subject revision
6. Promote normalized evidence and close the programme

ADR required: no
ADR path: N/A
Reason: verifies ADR-086 without changing its storage, service-authority, security, or application-structure boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed the cross-reader verification matrix against subject
`07efa181097f04de0594572e3bf29ec514c8cf4b`, tree
`07e22bec1c27ffa7ec7d1546f02f735fd4f623c9`. The retained TASK-23019 bundle records 60
automated and 32 live PASS results, 92 fact files, and 16 captures across all five readers and
the 160x50 / 120x35 / 100x30 / 80x24 live matrix. `hashes.json` covers 114 files with SHA-256;
`manifest.json` hashes to `ee23ff3c75b40223d3e75d2cb878e0cd619a5cd21612ce457ce7ee91beb3d0c3`
and `summary.json` to `b9fdf4e2f9ed1a4a8b79aaf4e3412f16811a225231f16ba00eaeb34c402c7866`.

The task-local runner, child containment boundary, live scenarios, and closeout tests provide the
catalogue, scratch/network/process tripwires, route cycle, capability journeys, cleanup checks,
normalization, and atomic promotion. Review repairs were localized to normalization, settled Work
focus before capture, asynchronous visible-row/control reacquisition, and named bounded diagnostics.
The final 490-test closeout module, the exact 60-result automated matrix, and all 32 live results
passed after the focused RED/GREEN repairs on the frozen subject. The exact verifier also passed
twice without creating bytecode or changing the evidence inventory. No product schema, reader
capability, service authority, or shell design changed.

Generated compositor text/SVG captures retain terminal-column padding by design and are therefore
excluded from whitespace-only diff checking; their exact bytes remain covered by `hashes.json`.

ADR required: no new ADR. ADR-086 is verified as written; the work changes evidence and test
reliability, not an architectural boundary. Evidence is retained under
`Docs/superpowers/reviews/evidence/task-23019/` and is reproducible with the verifier command in
its README. The incident-derived lesson is recorded in
`backlog/docs/lessons-testing-evidence.md`: a PASS label is not evidence until focus and mounted
identity have settled and failures name their live root.
<!-- SECTION:NOTES:END -->
