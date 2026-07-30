---
id: TASK-1411
title: Restore File Notes entry and compact-terminal Prepare usability
status: Done
assignee:
  - '@codex'
created_date: '2026-07-30 16:06'
updated_date: '2026-07-30 17:03'
labels:
  - notes
  - git
  - library
  - ux
  - accessibility
dependencies:
  - TASK-1350
references:
  - Docs/superpowers/qa/file-notes-full-app-uat-2026-07-30/README.md
  - >-
    .impeccable/critique/2026-07-30T16-13-58Z__tldw-chatbook-ui-screens-library-screen-py.md
documentation:
  - backlog/decisions/038-file-notes-guarded-session-commit.md
  - backlog/decisions/035-file-notes-session-git-index-controls.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Full-app acceptance testing found that File Notes cannot be reached through the visible Library Notes source switch at normal terminal widths and that Prepare session for commit hides essential status and actions at 40x20 when the linked-root path is realistically long. Restore a discoverable path into File Notes and make the existing guarded staging/commit workflow operable at the supported compact viewport without weakening its session-only Git safety promises.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library > Notes visibly exposes both Database and Files as focusable, keyboard-operable source choices at 120x40 and 160x45, and selecting Files reaches the retained File Notes workspace.
- [x] #2 At 40x20 with a realistic long linked-root and repository path, Prepare session for commit keeps current status and every required staging/commit action reachable through a visible or keyboard-scrollable layout without clipping labels or depending on pointer input.
- [x] #3 Wide and compact layouts preserve the existing count-and-promise copy, editor/Navigator switching, focus return, and unrelated-change safety behavior defined by ADR-035 and ADR-038.
- [x] #4 A real full-app disposable-repository walkthrough covers source selection, root selection, process-only trust, body editing with exact frontmatter preservation, autosave, session-only staging, review, commit success, and unrelated-staged blocking.
- [x] #5 Focused mounted regressions assert rendered reachability and keyboard operation at 160x45, 120x40, and 40x20 rather than relying only on direct widget method calls; targeted lint and diff checks pass.
- [x] #6 Database and Files retain an explicit visible selected-source state instead of relying only on a disabled control, and keyboard focus is never moved to an off-viewport source choice.
- [x] #7 The complete-staged-state mismatch keeps its exact fail-closed safety statement and adds a concise, accurate recovery instruction to commit or unstage unrelated staged changes, refresh, and review the session again.
- [x] #8 Keyboard users can open the exact unelided linked-root and runtime-warning text, and can scroll actionless Prepare states to their complete status and guidance at 40x20.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; conform to backlog/decisions/035-file-notes-session-git-index-controls.md and backlog/decisions/038-file-notes-guarded-session-commit.md.
Reason: this is a presentation, keyboard-reachability, and recovery-copy repair within the existing File Notes ownership and guarded-commit contracts.

1. Add failing mounted Textual regressions for visible source choices and selected state at normal widths, compact linked-root behavior, keyboard-scrollable Prepare actions, and actionable unrelated-staging recovery copy.
2. Give the source separator bounded geometry and preserve a visible selected-source marker.
3. Bound the compact linked-root summary, expose the exact detail in a keyboard-operated read-only dialog, and make actionable and actionless Prepare states vertically keyboard-scrollable.
4. Add the concise safe recovery instruction to the existing complete-staged-state refusal without weakening its exact-match block.
5. Run only focused affected tests, targeted static checks, the layout detector, and a real full-app wide/40x20 UAT; then self-review and record the result.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Bounded the Library Notes source separator, kept both choices enabled, and added explicit selected-source labels so Database and Files stay visible and keyboard-operable.
- Kept the linked-root summary to one row, added a visible keyboard-operated Details dialog for the exact root/runtime warning, and made Prepare a keyboard-scrollable surface in both actionable and actionless states.
- Preserved the fail-closed complete-stage proof while adding accurate recovery steps for unrelated staged content.
- Added rendered geometry and real-key regressions at 160x45, 120x40, and 40x20. The affected UI files passed 169 tests; the exact guarded-commit integration regression, targeted Ruff, compilation, diff/JSON checks, and layout detector passed.
- Repeated the real full-app PTY walkthrough with no runtime bypass and recorded the accepted remediation evidence under `Docs/superpowers/qa/file-notes-full-app-uat-2026-07-30/`.
- ADR required: no. The implementation conforms to ADR-035 and ADR-038 without changing storage, synchronization, or Git authority.
<!-- SECTION:NOTES:END -->
