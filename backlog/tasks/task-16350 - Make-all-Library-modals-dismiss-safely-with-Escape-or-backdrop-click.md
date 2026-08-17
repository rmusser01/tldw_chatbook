---
id: TASK-16350
title: Make all Library modals dismiss safely with Escape or backdrop click
status: Done
assignee: []
created_date: '2026-08-15 04:49'
updated_date: '2026-08-15 09:20'
labels:
  - library
  - ui
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every modal transitively reachable from Library safely dismissible from the keyboard or backdrop without bypassing transient layers, mutation guards, destructive confirmations, trust boundaries, or typed cancellation semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every ModalScreen transitively reachable from Library can be safely cancelled with Escape and with a primary-button click on its backdrop, except an explicitly documented non-dismissible gate.
- [x] #2 Escape gives a focused descendant overlay first refusal, then peels shared-picker path editing, search, and recent locations in that deterministic order before terminal cancellation; visible Cancel and backdrop remain terminal safe-cancel requests.
- [x] #3 Terminal Escape, backdrop, and visible Cancel return each modal's exact safe negative result and never confirm, save, delete, install, authorize, trust, or discard by themselves.
- [x] #4 Clicks inside modal content, descendant overlays, non-primary clicks, and inputs with unknown provenance do not dismiss a modal.
- [x] #5 Prompt collection create and rename mutations cannot be bypassed by queued Escape, backdrop, or visible Cancel input while a mutation is active; Cancel remains enabled as a guarded request, the first rejected close updates the existing status line once, and later requests do not stack feedback.
- [x] #6 Cancellation is single-shot, top-screen-only, mount-generation-safe, and restores only an eligible recorded opener or its single eligible stable-ID replacement.
- [x] #7 Focused tests mount and behavior-test every concrete reachable modal class, exercise real Textual key and overlay dispatch, verify MRO handlers run once, and enforce an explicit bidirectional launch inventory across supported direct, controller-injected, nested-widget, and modal-to-modal owner edges.
- [x] #8 Shared file picker changes preserve typed results and existing behavior for representative non-Library callers, including EnhancedFileOpen/EnhancedFileSave smart dismissal, handler suppression, persistence, and application import compatibility.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed executable plan:
`Docs/superpowers/plans/2026-08-14-task-16350-library-modal-dismissal.md`.

1. Extend the shared safe-dismiss boundary to restore only an eligible opener or its single eligible stable-ID replacement.
2. Adopt the shared file-picker base, reconcile Textual full-MRO lifecycle/navigation dispatch, and preserve EnhancedFileOpen/EnhancedFileSave compatibility.
3. Add exact typed safe dismissal to ordinary Library skill, model, Prompt-delete, and Note-folder modals.
4. Add the same contract to File Notes and Git detail/trust/authorization surfaces without changing trust or push policy.
5. Move Prompt collection create/rename/retry callbacks to a screen worker, keep visible Cancel enabled as a guarded request, and reject stale same-instance remount completions.
6. Enforce a narrow bidirectional launch-edge inventory and independently mount every concrete reachable modal for visible Cancel, Escape, and backdrop behavior.
7. Run only the named touched/related test and static matrices, record mutation RED/GREEN evidence, complete review/documentation hygiene, and close the task.

ADR required: yes; amend the existing ADR.

ADR path: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`.

Reason: Library adoption changes the established cross-module modal cancellation, shared file-picker, and focus-restoration contracts. ADR-031 already owns this interaction grammar and was amended by the approved design work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The Library modal surface now uses one safe-dismissal grammar while preserving each modal's typed negative result and positive path. The shared boundary restores only an eligible exact opener or one eligible stable-ID replacement; shared file pickers own Escape precedence and terminal visible/backdrop cancellation; ordinary, File Notes, Git, and Prompt collection modals adopt the same one-shot contract. Prompt collection create/rename/retry work runs in a screen worker guarded by mutation epoch and mount generation, so close requests stay responsive without bypassing or queueing behind work. The launch inventory remains an explicit 32-edge table rather than a general call-graph abstraction.

Implementation commits: `2852111d6`, `dc8c920f9`, `47ca159f9`, `93c2f5ab4`, `ce5d7c493`, `cf6fadab7`, `2dc2d299d`, `b714aedd5`, `0b610655f`, `a3e06734d`, `6b0df89a2`, `f068e25b6`, `84db99aad`, and `151924515`. Production changes are bounded to the shared dismissal helper, the base/enhanced file pickers, the two skill trust modals, model installation, Prompt deletion and collection management, Note folder dialogs, and File Notes workspace/Git dialogs. Related coverage is bounded to the twelve named test files in the implementation plan.

Evidence:

- Settled-HEAD exact twelve-file pytest matrix: `899 passed, 3 warnings in 945.87s`.
- Targeted Ruff format check (two files), compileall (eleven production files), and `git diff --check` passed.
- Targeted Ruff check reports one pre-existing `F401` at `Tests/UI/test_library_file_notes_git.py:16`; the exact diagnostic reproduces at task base `6696c6fe9`.
- Targeted MyPy reports 37 errors in four files versus 38 errors in five files at `6696c6fe9`; every current diagnostic is base-equivalent apart from line movement, and the branch removes the base Prompt collection `Literal` diagnostic. No new static-analysis debt was introduced.
- Mutation RED/GREEN proofs pinned: shared backdrop dispatch (`0 results` versus `1`), exact Prompt deletion negative (`None` versus fingerprinted decision), Prompt collection in-flight guard (default screen versus guarded modal), stable-ID focus (policy input versus recomposed opener), and the modal-to-modal inventory edge (`32 discovered` versus `31 declared`). Every mutation was restored byte-for-byte before the GREEN rerun.

ADR check: ADR required; existing `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md` was amended for TASK-16350, so no new ADR was created.

Independent cumulative review: approved after `84db99aad` bound focus restoration to the revealed screen and `151924515` made cancelled Prompt collection changes retryable; no P0-P2 findings remain.

Plan deviations: the final inventory suite consolidated the planned mutation oracle names into stronger concrete parameterized and bidirectional fixed-point nodes; the same five behaviors were mutation-proven with their exact current nodes. Whole-file Ruff/MyPy baseline debt was documented rather than repaired outside scope. No new lessons entry was added because the encountered test-name drift, mutation discipline, five-digit task handling, and base comparison traps are already covered by existing lessons.

Closeout state: acceptance evidence, documentation, and independent cumulative review are complete; the closeout documentation commit and final branch-state verification are recorded by the final plan steps.
<!-- SECTION:NOTES:END -->

## References

- Design: `Docs/superpowers/specs/2026-08-14-task-16350-library-modal-dismissal-design.md`
- ADR: `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`
