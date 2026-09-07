---
id: TASK-202
title: 'Library Prompts editor: group the action row as it grows'
status: Done
assignee:
  - '@codex'
created_date: '2026-07-12 22:21'
updated_date: '2026-08-09 00:56'
labels:
  - ux
  - library
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Second-pass UX review of Library Prompts (2026-07-12): the editor action row now carries six flat buttons (Save, Use in Console, Export, Copy text, Duplicate prompt, Delete). As Console-injection actions land in Phase 2 it will crowd; group into primary (Save) / content actions / lifecycle (Duplicate, Delete).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Normal editor actions retain stable IDs and keyboard order: Save, Use in Console, Export, Copy Markdown, Duplicate, Delete.
- [x] #2 Actions are grouped by primary, content, and lifecycle purpose; Save is visually distinguishable and Delete uses the existing danger treatment.
- [x] #3 Conflict actions Save as new and Reload render in the same always-visible action area.
- [x] #4 At 200x50 the action area is visible and nonzero; at shorter sizes it remains scroll-reachable without obscuring the final editor field.
- [x] #5 Copy Markdown copies the live unsaved working copy through the application clipboard seam and only reports success after copying succeeds.
- [x] #6 Unavailable or failed clipboard support is reported honestly without logging Prompt bodies.
- [x] #7 Single delete requires confirmation and warns that both the saved artifact and unsaved working copy are discarded when the editor is dirty.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement grouped, always-visible editor actions; wire live Copy Markdown; add shared delete confirmation; verify TCSS geometry. ADR required: no; ADR path: N/A; reason: UI-only change under ADR-011/ADR-040.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Grouped the Prompt editor actions into persistent primary, content, and lifecycle sections around a single scroll owner, preserving action IDs/order and the existing primary/danger treatments. Wired canonical live-working-copy Markdown for legacy and structured Prompt/Recipe artifacts, truthful clipboard outcomes, lossless/fail-closed compatibility handling, and a reusable dirty-aware Prompt/Recipe delete confirmation that retains the existing soft-delete service path and guards duplicate or stale decisions. Updated the Library guide and compositor screenshot; verified the supported, dirty, conflict, confirmation, compatibility, narrow-terminal, and clipboard-unavailable states.

ADR required: no. ADR path: N/A. This is a UI composition and defect-repair change within ADR-011/ADR-040; it introduces no storage, service, provider, security-policy, or ownership boundary.

Verification after rebasing onto `origin/dev` (`3023578c0`): affected UI/CSS suite `145 passed`; Ruff and `py_compile` passed for all changed Python files; CSS integrity and source/bundle sync passed; both diff checks passed; Impeccable layout scan returned no findings. The ADR-029 privacy hardening of the existing export warning was explicitly reviewed; the diagnostic inventory changes only the `library_screen.py` content digest with its 82-call count and sink topology unchanged, and the architecture inventory suite passed 8 tests. Independent final spec and code/security/UX reviews approved the implementation with no findings after that update. A complete repository run recorded `32341 passed, 210 skipped, 136 failed`; none of the 136 current failures were TASK-202 feature tests. Identical isolated replay on the pre-rebase branch and clean `origin/dev` showed 128 common unrelated failures, two old-base-only failures fixed upstream, and one origin-only Notes failure; five TASK-202-looking cache entries were stale node IDs rather than collected failures.

No new lesson was added: the incidents encountered are already captured by the existing lessons to compare identical failure sets, assert geometry rather than display text, and regenerate generated CSS instead of hand-merging it.
<!-- SECTION:NOTES:END -->
