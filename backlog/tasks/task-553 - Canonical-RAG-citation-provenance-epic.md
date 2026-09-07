---
id: TASK-553
title: Canonical RAG citation provenance epic
status: Done
assignee: []
created_date: '2026-07-24 00:42'
updated_date: '2026-07-27 19:05'
labels:
  - rag
  - citations
  - provenance
  - epic
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-27-citation-evidence-inspector-design.md
  - Docs/superpowers/plans/2026-07-27-minimal-console-rag-citations.md
  - Docs/superpowers/qa/2026-07-27-task-553-rag-citation-uat.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console users generate answers from selected local RAG evidence, retain the
exact cited chunks across restart, inspect them from the answer, and open the
supported original Library item. Server, sync, export, import, and new
provenance subsystems are outside this epic's approved closeout scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can search for and select exact local Library evidence, generate an answer from it, and receive the correct `[S#]` marker.
- [x] #2 A fresh profile using production service composition persists the cited-answer trace and shows `Sources (N)` after restart.
- [x] #3 Users can inspect the exact cited chunks and open supported originals through Library in the rendered end-to-end flow.
- [x] #4 The closeout remains local-only and focused automated citation, Console, and Library checks pass without expanding into server, sync, export, import, or baseline repair.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Complete the local prompt-evidence, terminal-answer, and minimal Console
   citation slices through TASK-553.13 to TASK-553.16.
2. Run scoped automated verification and rendered UAT against a fresh isolated
   profile.
3. Repair only the production key-provisioning blocker tracked by TASK-553.17.
4. Repeat the same rendered flow through `Sources (N)`, exact chunk inspection,
   Library open, and restart before closing the epic.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: ADR-024 and its existing foundation plan already require first-use key
provisioning only when no fingerprint-bearing rows exist.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the minimal local citation workflow across existing boundaries:
Library evidence selection, Console answer generation, canonical local trace
persistence, answer-level Sources inspection, and exact Library item opening.
The only closeout repair provisions the existing fingerprint key on first
enabled use when the canonical store has no fingerprint-bearing rows; restored
or populated databases with a missing key still fail closed.

Scoped citation/Console tests (437), filtered Library checks (8), touched-file
Ruff, and diff validation passed. Rendered UAT with a real local model and
fresh isolated data passed through app restart. Server, sync, export, import,
and additional provenance systems remain outside the completed scope.
<!-- SECTION:NOTES:END -->

## UAT Closeout Evidence

- The initial rendered UAT exposed a missing first-use key provisioning
  boundary. TASK-553.17 repaired that existing production-composition path
  without changing the citation architecture or adding a subsystem.
- The rerun passed real note search, evidence handoff, live llama.cpp
  generation, `[S1]`, one persisted canonical trace/owner, `Sources (1)`,
  literal chunk inspection, exact Library open, and an application restart.
- Detailed row counts, scoped verification, and screenshots are recorded in
  `Docs/superpowers/qa/2026-07-27-task-553-rag-citation-uat.md`.
