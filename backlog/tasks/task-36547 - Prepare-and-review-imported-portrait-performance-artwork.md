---
id: TASK-36547
title: Prepare and review imported portrait performance artwork
status: Done
created_date: 2026-10-23 22:56
dependencies:
- TASK-36542
- TASK-36546
labels:
- personas
- character-chat
- ui
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
- Docs/superpowers/reviews/2026-10-23-task-36547-review.md
assignee:
- '@codex'
updated_date: 2026-10-24 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users assign ready-made mouth/blink frames to a persisted local character, inspect them at Console size and apply or discard the complete set safely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preparation admits only saved local characters and persisted active artwork through exact local profile/service/database and editor/source authority; dirty artwork/bindings require explicit source Save/Apply without saving unrelated text.
- [x] #2 All six still-image slots share one disposable draft; import/replace/remove status, per-role validation and matching dimensions are visible before Apply.
- [x] #3 Offline preview cycles all six roles at the Console image-box size, compares against the source and offers idle/blink playback without requiring image generation or TTS.
- [x] #4 Apply publishes a complete validated set atomically and emits committed revision invalidation; Cancel and source/character/profile changes discard the draft and keep current artwork.
- [x] #5 Removing an applied performance restores ordinary artwork and invalidates predecessor caches; reopening the editor shows the stored set and its profile-local portability limit.
- [x] #6 Keyboard-accessible mounted tests cover unsaved artwork, read-only server selection with matching IDs, profile replacement, mismatched/imported frames, cancellation, transaction failure and apply-to-Console handoff.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/242-console-radio-portrait-performance.md
Reason: Implement approved source-bound local authoring, preview and atomic publication using existing storage and Personas authority.

1. Read saved-character/bound-pack fixtures, editor dirty artwork/binding state, exact local service/database ownership and existing file-picker/publication flows.
2. Add failing pure draft and actual screen-handler tests for source gates, six slots, cancellation, conflicts and imported frames.
3. Add a disposable source-bound draft and inline editor using existing form patterns and avatar renderer. Keep heavy authoring orchestration in a focused Persona module with thin screen entry/event methods to respect screen-size governance.
4. Capture/revalidate actual DB/service/editor/source identities before and after worker work; normalize imports off-thread and keep duplicate work guarded with finally cleanup.
5. Provide deterministic six-role/source comparison and idle preview; Apply/Remove through existing CAS APIs, publish committed actor invalidation, and keep Cancel non-mutating. Show local portability and explicit source-save guidance.
6. Run pure/mounted import-to-Console and affected expression/visual tests, rebuild CSS, check tokens/lint/screen limits, self-review, document and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the source-bound inline Living portrait import/review workflow under ADR-242. One disposable six-role draft validates normalized geometry/limits and survives partial replacement failures. Exact local DB/service/editor/session and persisted source/local authority are checked around native work; dirty artwork/bindings require explicit Save/Apply, while unrelated text stays unsaved. Imports use bounded file reads and existing finite callback draining/operation-owned connections. Apply/Remove use revision CAS and committed Console invalidation; Cancel/source/profile changes fence late results. Offline six-role/idle preview reuses Console rendering and the existing timer, respects motion preferences, restores invoking keyboard focus after pickers and stacks narrow panes. Late preview failure reports an already committed result truthfully. Existing thumbnail helpers moved unchanged to a focused module, preserving screen APIs and lowering PersonasScreen to 16,542 lines below its 16,564 ceiling.

Validation: 32 focused pure/mounted authoring, playback and affected expression tests passed (radio-36547-last-ui). Four additional held-import/duplicate/narrow cases passed (radio-36547-compact). The real mounted Personas-to-Console test verifies committed red fitted pixels and original yellow artwork after Remove. 77 CSS/token/build and relevant/slack module-budget checks passed, 31 unrelated ceiling cases deselected (radio-36547-guards-final). New code/tests pass Ruff/format; HEAD-source lint comparison shows zero added diagnostics in all four legacy modules; git diff --check passes. The earlier broad size run had 103 passes and six unchanged baseline ceiling violations, verified byte-identical to HEAD. No full suite, provider/device qualification, push or merge performed.

Review: Docs/superpowers/reviews/2026-10-23-task-36547-review.md. Updated architecture, user guidance, implementation plan and concrete fixture lesson. ADR required: yes; backlog/decisions/242-console-radio-portrait-performance.md applies. Main files: Character_Chat/portrait_performance_authoring.py, UI/Persona_Modules/{portrait_performance,character_avatar_helpers}.py, portrait_performance_editor.py, Personas editor/screen, Console avatar offline API, focused tests and generated token CSS.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
