---
id: TASK-37266
title: Preserve later Character Bench drafts and selection when queued Save publishes
status: In Progress
created_date: 2026-10-28 22:07
labels:
- textual-patterns
- evals
- data-retention
priority: medium
references:
- tldw_chatbook/UI/Evals/character_bench_editor.py
- tldw_chatbook/UI/Screens/evals_screen.py
assignee:
- '@codex'
updated_date: 2026-10-28 23:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep a successful Character Bench Save from replacing later accepted edits or newer supported shell selection. A saved snapshot and the current editing context can differ while their separate Textual message pumps and awaited region refreshes complete.

On merged dev `7884e11a496a677c8180cc5d67234a57b7d659b9`, mounted real-SQLite controls preserve the submitted `Saved` value but replace later native child input `Saved-new` with a fresh `Saved` Input. Caret/selection change from 9/(9,9) to 5/(0,5); focus is restored by ID. This occurs before publication and during Catalog refresh. Completed and lock-pending selections through the documented public shell API are also displaced. Ordinary Save passes. These controlled interleavings establish possibility, not natural frequency or actual later rail-event ordering.

**Existing owners:** [TASK-37142](<task-37142 - Preserve-newer-Evals-bench-input-when-queued-Save-completes.md>) (renumbered from audit TASK-37031) fixes the word-bench producer and explicitly leaves character benches unchanged. [TASK-1610](<task-1610 - Run-completion-must-not-destroy-a-dirty-bench-editor.md>) protects dirty editors from run completion; [TASK-1484](<task-1484 - Rename-affordances-for-Evals-benches-and-datasets.md>) owns rename refresh; [TASK-1912](<task-1912 - Duplicate-a-character-probe-bench.md>) and [TASK-1913](<task-1913 - Character-probe-Phase-2-review-hygiene-batch.md>) retain their character-authoring scopes. This is the separate CharacterBenchEditor Saved publication, not a reopening of those completed tasks.

**ADR assessment:** New ADR expected: no; ADR path: N/A. A bounded repair should preserve [ADR-011](<../decisions/011-chatbook-workbench-ui-system.md>), [ADR-033](<../decisions/033-application-session-state-ownership.md>), [ADR-150](<../decisions/150-design-token-system-and-design-language.md>) and [ADR-161](<../decisions/161-component-pattern-library.md>); reassess only if ownership/service contracts change.

Book Chapters 7/9 and the actual controls, source pins, ownership screen and limits are in the [post-merge audit](../../Docs/superpowers/qa/2026-10-28-textual-book-after-pr3283.md). The supplied book is reference data.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Successful Character Bench Save followed by accepted later field input retains the exact later draft and its editing context when publication completes, including input accepted during the awaited Catalog refresh.
- [ ] #2 The submitted snapshot remains correctly persisted and reflected in Catalog/Inspector; later draft edits remain unsaved until a subsequent explicit Save, which persists them normally.
- [ ] #3 Newer supported public shell selection remains authoritative whether already mounted or awaiting the existing region-swap lock when an older character Save publishes.
- [ ] #4 Ordinary Save/rename refresh, validation/write failures, Revert and existing run-completion dirty-editor guards retain their established outcomes; mounted real-SQLite controls exercise the actual Save producer, native focused-child input and selection interleaves.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A (existing ADR-011/033/150/161).
Reason: bounded Character Bench publication fix preserving Evals service/state and separate message namespaces.
1. Read TASK-37266, existing word/character Save paths, task-specific plan, audit controls and applicable UI/testing docs.
2. Promote real SQLite/native Save controls; prove intended RED for later input, post-rail input and newer completed/pending selection, including raw sampler/card/description state.
3. Capture raw submitted editor applicability and conditionally publish through the existing serialized region swap; retain submitted persistence and normal canonical Save.
4. Verify GREEN and targeted original character/word/dirty-run controls plus scoped static/size/governance checks; update Evals architecture evidence.
5. Root commits exact owned changes, obtains independent spec/quality review, enrolls gates, and closes ACs/notes/status only after verified completion.
Full plan: Docs/superpowers/plans/2026-10-28-textual-37266-37268.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
