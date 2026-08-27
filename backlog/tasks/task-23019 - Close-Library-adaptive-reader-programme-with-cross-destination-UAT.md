---
id: TASK-23019
title: Close Library adaptive-reader programme with cross-destination UAT
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-27 13:58'
updated_date: '2026-08-27 14:24'
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
- [ ] #1 Media, Conversations, Notes, Prompts, and Skills pass one production-shaped automated cross-reader matrix covering retained pane identity, collapse, preferences, focus, selection/loading truth, stale settlement, and resize purity.
- [ ] #2 All five destinations pass live containment, collapse, restoration, mode reachability, and selection checks at 160x50, 120x35, 100x30, and 80x24.
- [ ] #3 Every stable ID in the bounded TASK-23019 closeout catalogue maps to at least one fresh automated result and one live journey from the exact recorded subject revision; earlier destination evidence is lineage rather than a substitute.
- [ ] #4 Every declared writable config, profile, XDG, database, temporary, and raw-evidence path resolves inside scratch before application import; phase-scoped tripwires permit only declared read-only subject-checkout and resolved Python-runtime resources, scratch runtime writes, and the validated evidence-promotion destination; they record no prohibited filesystem or network attempt or checkout/runtime mutation; all harness-created database and host-worker owners close; and the raw scratch root is removed without reading or hashing real user-owned content.
- [ ] #5 A sequential single-app route cycle proves destination preferences, drafts, selection, modes, focus, and asynchronous workers do not leak across readers.
- [ ] #6 Any same-PR repair is a localized regression against ADR-086 or an approved destination contract, has a focused failing regression test, and introduces no new schema, ADR, service authority, capability, or redesign.
- [ ] #7 The final automated and live matrices, required derived-artifact checks, targeted static checks, capability ledger, and bounded evidence README pass; the manifest identifies the exact tested subject revision, and the final branch differs only by retained evidence, task, and ledger documentation.
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
