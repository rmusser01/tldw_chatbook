---
id: TASK-21135
title: 'Student workflow (sub-project #5): library_save_note closes the write loop'
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-22'
labels:
  - chunking
  - notes
dependencies: [TASK-20939]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-project #5 of the chunking parity program (single PR, stacked on #4/TASK-20939's four chunk tools — no new ADR; the spec's §8 rulings are the long-form record): the write side of the student story — one tool, `library_save_note`, that lets a Console/MCP agent land what #4's read path gathered. Create-default with version-locked update (`note_id` + `expected_version`, `ConflictError` → `content_changed`), one-level folder grouping through the notes scope service (idempotent ensure, race-tolerant), the `library.notes/save` policy action with denial preceding every backend call, the provenance-header convention (documented, never enforced), and the QA-in-notes flashcard target. Plus the upgraded end-to-end student story (save → re-read → search-based re-run → update-not-duplicate), the study-notes fan-out pattern in the user guide, and final review.

Spec: `Docs/superpowers/specs/2026-08-23-student-workflow-design.md` (§4 design, §5 conventions, §6 flashcards, §7 testing, §8's 11 rulings). Plan: `Docs/superpowers/plans/2026-08-23-student-workflow.md` (two tasks, one PR).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] The write tool: `library_save_note` as the 24th descriptor (note/save) — create-default returning id+version, version-locked update via `note_id`+`expected_version` with stale-version `content_changed`; input bounds schema-level and now invoke-time (title 512 / content 100_000 / folder 256, named limits); provenance-header convention in the description (spec §4.1-§4.2, §8.2, §8.6, §8.8)
- [x] The folder seam: rows via the legacy notes interop, folders/placements via `NotesScopeService` pinned to the notes screen's own `local_note` scope; idempotent ensure (lookup → create-on-miss → re-query-on-collision), order validate → policy → folder → row → attach so a folder failure never orphans a note (spec §4.3, §8.5, §8.10)
- [x] Policy: `library.notes/save.local` registered and wired at both construction sites (chat_screen factory, MCP build); denial precedes every backend call, mutation-pinned on both seams (spec §6)
- [x] Story + conventions: the #4 read path now closes the loop through the shared dispatcher — Chapter-7 note saved with provenance header, re-read whole, re-run leg search-based (`library_search_notes(query=title)`, the third-review ruling — the list tool has no folder filter) updating instead of duplicating, QA-flashcard leg saved and re-read; fan-out pattern + save-note contract documented, CHANGELOG (spec §5, §7.6)
- [x] Close-out: both follow-ups filed (TASK-20979 flashcards viewing/SRS surface; TASK-20980 folder-filtered list-notes candidate); final review READY-WITH-LISTED-FOLLOWS with both follows fixed in-branch (invoke-time length guards incl. the spec-save twin; prose/count/spec-errata sweep); targeted suites green (spec §7.7)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Descriptor + schema + dispatch + handler + policy wiring (plan Task 1)
2. Student-story upgrade, docs, follow-ups, close-out (plan Task 2)
3. Final review + the two listed follows fixed in-branch (this task's close-out)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: one PR stacked on TASK-20939 (#4) — the save handler lives in `local_library_tool_service` (ruling §8.10: the note-backend dispatch, not the media-chunk service), everything descriptor-derived so the Console and MCP runtimes cannot drift.

- Commits `fe457e4e8`/`c34c5d578` (spec + plan) through HEAD: `6e04d3199` (tool + policy), `015a5015d` (exclusion-set pin 23→29), `63c8d5bf6` (story + docs + follow-ups), `0b2737827` (Task-2 review minors), `8affabbdb` (invoke-time length guards), `a7fe38563` (final-review prose sweep).
- Key rulings honored (§8, 11 total): one write tool, no structured fan-out (§8.1); create-default + optimistic locking (§8.2); flashcards are Q/A-in-notes, real-rows follow-up filed (§8.3, TASK-20979); conventions as affordances not prompt presets (§8.4); the rows/folders seam split with the scope service (§8.5); schema maxLength bounds (§8.6, made invoke-time in `8affabbdb`); duplicate-window accepted with the re-run convention search-based per the third review (§8.7 + §5, superseding list-before-rerun; TASK-20980 holds the conditional list-filter candidate); provenance header carries `revision:` (§8.8); `ConflictError` → `content_changed` (§8.9); handler placement (§8.10); follow-ups at close-out (§8.11).
- The scope pin: folder writes go to `ScopeType.LOCAL_NOTE` (the notes UI's own scope — any other scope makes folders invisible there), reached only through the scope-exposed children-list seam (`parent_id=None`); the repository's path-getter is not scope-exposed and the ledger's STOP rule forbids reaching the repository from the tool layer.
- Final review: READY-WITH-LISTED-FOLLOWS, both follows fixed in-branch (`8affabbdb` length guards incl. the spec-save name/description twin — constants shared with the schema so the surfaces cannot drift; `a7fe38563` shadowed-count arithmetic 29, spec errata annotations on the superseded list-before-rerun phrasing, storage-error copy "read"→"operation", guide punctuation). Knowingly open: TASK-20979 (flashcard UX — the deliberate QA-in-notes ruling); the fan-out stays docs-only by non-goal (§8.1).
- Long-form record: spec §8 (11 rulings) + `Docs/superpowers/plans/2026-08-23-student-workflow.md` (two-task plan; no separate sdd ledger this round). Depends on TASK-20939 (#4's chunk tools and spec-save precedent).
