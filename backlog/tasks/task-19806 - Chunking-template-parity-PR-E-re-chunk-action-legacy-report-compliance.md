---
id: TASK-19806
title: 'Chunking template parity PR E: re-chunk action, legacy-chunk report, compliance'
status: Done
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-21'
labels:
  - chunking
dependencies: [TASK-19805]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR E (re-chunk & report) of the Chunking Template Parity sub-project (ADR-078): ship the Library-surface legacy-chunk report line and the re-chunk action with **forced** re-indexing (the `needs_reindexing` path would silently skip every item), a worker mutual-exclusion guard with Backfill (never `exclusive=True` — Textual cancels, it doesn't refuse), policy-id registration, design-system compliance, and the closing docs/CHANGELOG/user-guide pass.

Spec: `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` (§10, §12 PR-E ACs 41-47 plus process ACs 51-52). Plan: `Docs/superpowers/plans/2026-08-21-chunking-template-parity.md` (PR E, Tasks 12-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library surface renders the legacy-chunk line only when non-zero, omits it when zero, and consumes only `legacy_chunk_report` (spec AC 41)
- [x] #2 Re-chunk replaces chunk rows **stamped** and the legacy count drops by exactly the number reported re-chunked (remainder explained by skipped/failed counts); a post-re-chunk RAG search returns the new chunk text (proving the forced re-index), the owning service's query cache is cleared, and an interrupted run leaves the item re-indexable next time (spec ACs 42-44)
- [x] #3 Re-chunk and Backfill cannot run simultaneously and neither cancels the other — the second press is refused with a notice; the action's policy id is registered and pinned in `Tests/RuntimePolicy/` (spec ACs 45-46)
- [x] #4 New controls define rest/hover/focus/disabled states with `$ds-*` tokens (no raw hex), the new `Select` is colors-only styled, classes are styled or registered, the CSS bundle is rebuilt with `build_css.py`, and the CSS/token guards pass; CHANGELOG and `Docs/User_Guide/` (picker + Library controls, offset-basis caveat) updated with re-verified stamps (spec ACs 47, 51-52)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. LibrarySearchRagPanel legacy-chunk report line — display-gated, race-free (plan Task 12)
2. Re-chunk action: forced re-index (§10.2.1), separate worker group + mutual guard, policy id (plan Task 13)
3. Compliance sweep (design tokens, CSS bundle), CHANGELOG, user guides, targeted final sweep (plan Task 14)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: report line and re-chunk on `LibrarySearchRagPanel` (the ADR-003-correct surface) with display-gated Static idiom, per-item forced re-index, a separate worker group with a both-directions in-flight guard against Backfill, and the closing compliance/docs pass.

- Commits `78c0844ab..219278d95` (PR-E marker `219278d95`) plus final-review fix `79d7e05e7` (inventory regenerated for PR A-E drift); SDD tasks 12-14 + final review.
- Deviations-with-rulings: spec §13.1 and `.superpowers/sdd/2026-08-21-chunking-template-parity/progress.md` — epoch-sentinel pre-add mark is the only crash-safe §10.2.1 reading (ruling, reviewer-verified from source); `index_batch_optimized` has no per-doc chunk override so vector rebuild uses the service chunker (template boundaries in the chunk table only; boundary parity = future spec-level API change, noted for #3+); the no-op trap was proven with a control test.
- Final review verdict READY-WITH-LISTED-FOLLOWS (47 satisfied / 7 satisfied-with-nuance / 0 not satisfied, merge-base `e31a18d45..219278d95`): I-1 fixed in `79d7e05e7`; **I-2 remains a merge-PR obligation** (v7 collision re-sweep + record in the PR description); M-1 recommends one full `Tests/UI/` run on the merge PR.
