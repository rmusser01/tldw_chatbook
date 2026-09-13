---
id: TASK-19801
title: 'Chunking template parity PR 0: #1 residue — fail-closed rolling-summarize, deletions, inventory'
status: Done
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-21'
labels:
  - chunking
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 0 (independently mergeable) of the Chunking Template Parity sub-project (#2 of the parity program, ADR-078): clear the residue accepted by sub-project #1 — rolling-summarize becomes fail-closed everywhere (shim markers were data corruption with a friendly face), the dead `table_serializer`/`template_events` residue is deleted, and the persistent diagnostic inventory is regenerated — plus the Task 1 governance filings (ADR-078 itself, the §11 follow-ups, the upstream defects ledger).

Spec: `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` (§8.3, §11, §12 PR-0 ACs 1-5 and process ACs 48-49). Plan: `Docs/superpowers/plans/2026-08-21-chunking-template-parity.md` (PR 0, Tasks 1-3). Board companion: TASK-19641-19648 (§11 filings).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rolling-summarize fails closed through both the engine and the shim — provider failure raises on every path, **both** marker prefixes (`[Summarization failed…]` and `[Summarization error…]`) are un-persistable, and each guard was shown red under a seeded mutation before acceptance (spec ACs 1-3)
- [x] #2 The #1 residue is deleted — `table_serializer.py`, its `serialize_tables` kwarg threading, its doc sections, `template_events.py`, and the orphan SQL — with the named suites green without them (spec AC 4)
- [x] #3 The persistent diagnostic inventory is regenerated (`scripts/check_persistent_diagnostic_inventory.py`) with its row diff hand-reviewed in the same commit (spec AC 5; corrected en route — the spec's "two deleted modules" was factually one: `template_events` had zero logger calls and never had a row)
- [x] #4 Process ACs 48-49 landed with this PR: ADR-078 created before implementation with its number swept across remote refs and untracked worktree files, and every §11 chatbook item re-verified live on `origin/dev` then filed (TASK-19641-19648) with the five upstream items recorded in `Chunking/engine/UPSTREAM_DEFECTS.md`
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Governance first (plan Task 1): ADR-078, §11 follow-up filings, upstream defects ledger
2. Fail-closed rolling-summarize through engine + shim, mutation-proven (plan Task 2)
3. Delete the residue and regenerate the diagnostic inventory with a reviewed row diff (plan Task 3)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: closed #1's accepted residue before any #2 feature code — fail-closed rolling-summarize, residue deletions, inventory regeneration — plus the Task 1 governance filings (ADR-078; §11 items filed as TASK-19641-19648; UPSTREAM_DEFECTS.md entries 9-14 + #15 appended later in PR A).

- Commits `a4cdae5e4..d628afe6b` (PR-0 marker `d628afe6b`), incl. straggler fix `801ac910b`; SDD tasks 1-3.
- Deviations-with-rulings: spec §13.1 and `.superpowers/sdd/2026-08-21-chunking-template-parity/progress.md` — inventory-drift absorption ruled correct (gate proven already red at parent; script byte-exact by design; every row attributed in the commit body), the except-`ChunkingError` passthrough accepted as structurally necessary to prevent double-wrap, and §11 item 3b (embedding_template_selector) not re-filed — TASK-16472 AC#2 already owns fix-vs-retire with the same evidence.
- Suites at PR-0 close: Chunking-minus-sync 430p/29s/1xf (baseline+4); both marker prefixes source-pinned; 3/3 mutations recorded.
