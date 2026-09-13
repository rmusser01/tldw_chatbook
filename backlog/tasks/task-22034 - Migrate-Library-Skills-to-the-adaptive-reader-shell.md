---
id: TASK-22034
title: Migrate Library Skills to the adaptive reader shell
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 23:28'
updated_date: '2026-08-26 22:03'
labels:
  - library
  - ui
dependencies:
  - TASK-22033
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move Skills into the shared Library adaptive reader structure while preserving local-store import editing trust review supporting-file and recovery boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Skills list remains mounted beside a permanent work pane with independent list collapse and destination-specific geometry
- [x] #2 Overview is the default mode and Edit Trust and Files are explicit destination-owned modes
- [x] #3 Trust identifies the reviewed revision or fingerprint and existing policy marks prior review stale after applicable changes
- [x] #4 Supporting files remain read-only unless an existing capability explicitly permits editing
- [x] #5 Create import trust review recovery and destructive actions remain reachable without unmounting the list
- [x] #6 Selection loading draft navigation stale workers trust changes deletion and retry follow the approved identity and recovery contracts
- [x] #7 Automated list editor trust files geometry focus and capability tests pass with a representative live TUI walkthrough
<!-- AC:END -->

## Implementation Plan

1. Inventory the current Skills browse, editor, import, trust, supporting-file, conflict, deletion, and recovery capabilities before changing production UI.
2. Add one revision-aware Skills reader presentation model for Overview, Edit, Trust, and Files while leaving LocalSkillsService and the existing trust service authoritative.
3. Keep the Skills list mounted beside a permanent destination-owned work pane using the shared Library adaptive reader shell and independent Library/Items collapse preferences.
4. Preserve all existing workflows and identity fences, then verify targeted state, UI, trust, file, geometry, focus, and live TUI journeys.

ADR required: yes

ADR path: `backlog/decisions/086-library-adaptive-reader-shell.md`

Reason: This task implements the accepted Library structural boundary and Skills mode contract without changing Skills storage, import, trust, execution, or supporting-file authority.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated Library Skills to the shared adaptive reader with permanent Items and Work panes, independent Library/Items collapse preferences, a protected 48-column Work floor, and stable F6 regions. Added Overview as the default projection plus explicit Edit, Trust, and read-only Files modes while retaining one existing `SkillEditorState` draft and the existing Basic/Advanced editor submode. Trust now labels the exact reviewed manifest generation and full fingerprint; detail, trust-status, review, approval, revoke, bootstrap, and script-grant settlements are fenced by the retained Work generation, with every verdict still delegated to `SkillTrustService`.

Preserved create, file/directory/URL import, filtering, sorting, optimistic save/conflict recovery, deletion, trust setup/reset/review/approve, script grants, supporting-file metadata, and retry without adding storage or file-write authority. A final retained-identity review found legacy whole-screen recomposes after import browsing, Back/discard, delete, trust reset, and trust setup; those paths now use destination-scoped synchronization, and exact list/work identity is covered across import, setup, and delete.

Verification: 310 targeted Skills reader/editor/import/service/trust/file tests passed; 153 shared-shell and Media/Conversations/Notes/Prompts/Skills cross-reader tests passed; the production-shaped Textual matrix passed at 160x50, 120x35, 100x30, and 80x24 while cycling all four modes; 17 focused post-format reader/state tests passed. Ruff lint, py_compile, git diff checks, and the required Impeccable detector passed with no findings. The three new files are Ruff-formatted; the four modified legacy files retain their identical `origin/dev` whole-file formatter drift to avoid an unrelated broad rewrite. Per repository policy, no full-suite sweep was run without explicit user opt-in.

ADR required: yes. ADR path: `backlog/decisions/086-library-adaptive-reader-shell.md`. Reason: this directly implements the accepted long-lived Library reader boundary while leaving Skills storage, trust, execution, and supporting-file ownership unchanged; no new ADR was required.

Post-rebase verification: rebased cleanly onto `origin/dev` at `c6218918d1e70c1938f7e11df592d0c70ca60383`. Fresh integration gates passed all 310 targeted Skills tests and all 153 shared-shell/cross-reader tests with the same two dependency warnings and environmental pytest temporary-cleanup noise only. No integration correction was required.

Pre-PR review hardening: fenced delete settlement and interlock cleanup by the exact Work generation, reserved the one-shot post-create scroll receipt for the Work pane, projected selected identity into the retained Items row with a persistent marker, and invalidated active trust-review receipts after content saves. The focused 203-test Skills state/canvas/reader/lifecycle set passed after these corrections; Ruff, py_compile, and diff checks remained clean.

Protected-check inventory review: the statement-level diagnostic comparison showed one intentional replacement in `library_screen.py`: the removed `info` call interpolated `skill_name`, while the new retained Work-pane failure diagnostic is fixed-text `debug` output with exception context and no user content, secret, path, or URL. Regenerated `Docs/security/production-diagnostic-inventory.json`; all six Derived Artifacts checks then passed under Python 3.11.

Final pre-merge rebase: rebased cleanly onto `origin/dev` at `a8c7241744f76c50ddc15f6ad01da32f3dd245d6`. All six Derived Artifacts checks, `git diff --check`, and the 206-test Skills reader/state/canvas/lifecycle suite passed; output contained only the two known dependency warnings and environmental pytest temporary-cleanup noise.

Qodo follow-up: completed all three requested Google-style docstring remediations for the public reader helpers, retained Work pane, and trust header, including constructor and `sync_state()` arguments plus `compose()` results. Ruff, Python 3.11 compilation, `git diff --check`, and 26 focused Skills state tests passed after the documentation-only change.
<!-- SECTION:NOTES:END -->
