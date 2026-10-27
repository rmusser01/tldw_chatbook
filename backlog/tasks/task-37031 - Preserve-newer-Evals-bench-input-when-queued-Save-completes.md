---
id: TASK-37031
title: Preserve newer Evals bench input when queued Save completes
status: To Do
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
- [ ] #1 A successful word-bench Save followed by later accepted name input retains that exact newer draft and its editing focus, caret and selection when queued Save publication is consumed.
- [ ] #2 The submitted snapshot remains correctly persisted and reflected in the Catalog/Inspector; later input stays unsaved until an explicit subsequent Save.
- [ ] #3 Ordinary Save, validation and write failures, Revert, and existing run-completion dirty-editor protections retain their established behavior.
- [ ] #4 Mounted real-SQLite tests cover ordinary Save and a queued parent Save completion after actual focused-child input, asserting persisted data, live draft and editing context after native work drains.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
