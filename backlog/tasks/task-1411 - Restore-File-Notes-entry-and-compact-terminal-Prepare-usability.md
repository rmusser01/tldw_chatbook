---
id: TASK-1411
title: Restore File Notes entry and compact-terminal Prepare usability
status: To Do
assignee:
  - '@codex'
created_date: '2026-07-30 16:06'
updated_date: '2026-07-30 16:14'
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
- [ ] #1 Library > Notes visibly exposes both Database and Files as focusable, keyboard-operable source choices at 120x40 and 160x45, and selecting Files reaches the retained File Notes workspace.
- [ ] #2 At 40x20 with a realistic long linked-root and repository path, Prepare session for commit keeps current status and every required staging/commit action reachable through a visible or keyboard-scrollable layout without clipping labels or depending on pointer input.
- [ ] #3 Wide and compact layouts preserve the existing count-and-promise copy, editor/Navigator switching, focus return, and unrelated-change safety behavior defined by ADR-035 and ADR-038.
- [ ] #4 A real full-app disposable-repository walkthrough covers source selection, root selection, process-only trust, body editing with exact frontmatter preservation, autosave, session-only staging, review, commit success, and unrelated-staged blocking.
- [ ] #5 Focused mounted regressions assert rendered reachability and keyboard operation at 160x45, 120x40, and 40x20 rather than relying only on direct widget method calls; targeted lint and diff checks pass.
<!-- AC:END -->

## ADR Check

ADR required: no

ADR path: N/A; conform to
`backlog/decisions/035-file-notes-session-git-index-controls.md` and
`backlog/decisions/038-file-notes-guarded-session-commit.md`.

Reason: this task repairs presentation and keyboard reachability within the
existing File Notes ownership, staging, and guarded-commit contracts. It does
not change storage, schema, sync policy, Git authority, service boundaries,
security policy, dependencies, or long-lived application structure.
