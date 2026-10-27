---
id: TASK-37031
title: Preserve newer Evals bench input when queued Save completes
status: In Progress
created_date: 2026-10-27 18:00
labels:
- textual-patterns
- evals
- correctness
priority: medium
references:
- tldw_chatbook/UI/Evals/bench_editor.py
- tldw_chatbook/UI/Screens/evals_screen.py
- Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md
assignee:
- '@codex'
updated_date: 2026-10-27 20:21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep text accepted after a successful Evals bench Save from disappearing when its queued completion rebuilds the editor. The submitted SQLite value can be correct while the user loses a newer unsaved draft.

A mounted word-bench observation holds the parent pump, dispatches the real Save button event, then sends focused-child keys. The child accepts `Saved-new` (caret 9, dirty); consuming queued `BenchEditor.Saved` replaces it with `Saved` (caret 5). Ordinary Save passes. This controlled interleave establishes the mechanism, not natural frequency; character benches were source-only.

**Scope and related owners:** [TASK-1482](<task-1482 - Bench-authoring-targets-probes-and-top-K-editable-with-Duplicate-and-Delete.md>) delivered bench authoring. [TASK-1610](<task-1610 - Run-completion-must-not-destroy-a-dirty-bench-editor.md>) protects dirty editors from run completion, whereas this trigger is Save publication. [TASK-36037](<task-36037 - Preserve-unfinished-Evals-dataset-names-when-snippet-imports-complete.md>) protects dataset names during snippet import. [TASK-15475](<task-15475 - Panel-scoped-updates-for-Settings,-Evals,-and-speech-settings;-dedupe-mount-resume-double-work.md>) scopes region replacement, which can still discard a newer child draft. Preserve these completed outcomes; no existing open task owns this observed Save trigger.

**ADR assessment:** No new ADR is expected for a bounded repair preserving existing ownership and service contracts. Existing [ADR-011](<../decisions/011-chatbook-workbench-ui-system.md>), [ADR-033](<../decisions/033-application-session-state-ownership.md>), [ADR-150](<../decisions/150-design-token-system-and-design-language.md>), [ADR-161](<../decisions/161-component-pattern-library.md>) apply; reassess if implementation changes those boundaries.

Evidence, source pins, book pages, controls and limitations: [post-merge Textual audit](../../Docs/superpowers/qa/2026-10-27-textual-book-after-pr3251.md). Audited merged dev: `416feb50ee667fdc95cb269bf50f8b1d5b3ff8ff`. The supplied book is reference data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A successful word-bench Save followed by later accepted name input retains that exact newer draft and its editing focus, caret and selection when queued Save publication is consumed.
- [x] #2 The submitted snapshot remains correctly persisted and reflected in the Catalog/Inspector; later input stays unsaved until an explicit subsequent Save.
- [x] #3 Ordinary Save, validation and write failures, Revert, and existing run-completion dirty-editor protections retain their established behavior.
- [x] #4 Mounted real-SQLite tests cover ordinary Save and a queued parent Save completion after actual focused-child input, asserting persisted data, live draft and editing context after native work drains.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce ordinary, queued focused-child input and departure Save publication with mounted real-SQLite tests. 2. Make the smallest existing-owner applicability fix while retaining persisted cleanup and Catalog/Inspector refresh. 3. Run targeted runtime, static and governance checks; document evidence and self-review. ADR required: no. ADR path: N/A. Reason: bounded repair of existing owner/service/UI contracts. Existing ADR paths: backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Preserved later word-bench drafts across successful Save publication using an originating-editor raw form snapshot. The existing serialized swap checks applicability after awaited Catalog replacement before touching the detail; Save refresh shares the current selection revision so pending user navigation still mounts. Catalog/Inspector reflect persisted data; ordinary unchanged Save reloads cleaned values, and successful submission advances the dirty baseline. Added five mounted real-SQLite private-profile cases covering ordinary cleanup, queued later input, input during awaited rail refresh, settled departure and pending departure. Final five actual children PASS with DB Saved versus live Saved-new, retained child/focus/caret 9/selection (9,9), dirty state and subsequent explicit Save. Existing validation, conflict/deleted-write failure, Revert and dirty run/sample completion controls PASS. Scoped screen batch: 97 PASS, 2 FAIL across 99 cases (including five passing private children); scoped bench batch: 31 PASS, 3 FAIL across 34 cases. A serial exact-five control comparison against original product bytes and final bytes gives the same two persistent native legacy failures and three passes; no unrelated product fix applied. All exact failures, non-reproducing broad-run failures and source qualification are recorded in the report. Root explicitly authorized bootstrap_profile only in the two legacy Evals test modules after confirming no deliberate profile switching; DB fixtures and all 123/107 function ASTs remain unchanged. Evals owner is 4211 lines; only its size row lowered from 4236. All six touched Python files pass focused Ruff lint/format, compilation and whitespace checks after authorized unused-import/mechanical normalization. Final size/import controls: 4 PASS; preimport 558/558 modules and first-send 93/94. Token/component governance: 25 PASS, 5 inherited native Windows path failures with unchanged guard/CSS inputs. Governing ADR paths: backlog/decisions/011-chatbook-workbench-ui-system.md; backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md. No new ADR required: bounded existing-owner repair. Architecture docs updated. Full evidence, hashes, exclusions and concerns: .superpowers/sdd/2026-10-27-textual-37031-37034/task-1-report.md. Keep In Progress for independent root review and current-head hosted CI.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->