---
id: TASK-15506
title: Lead File Notes push review with outcomes
status: Done
assignee: []
created_date: '2026-08-11 22:30'
updated_date: '2026-08-11 22:49'
labels:
  - notes
  - git
  - ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-08-11T20-58-28Z__ok-widgets-library-library-file-notes-workspace-py.md
  - backlog/decisions/011-chatbook-workbench-ui-system.md
modified_files:
  - tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py
  - Tests/UI/test_library_file_notes_git_push.py
  - backlog/docs/lessons-testing-evidence.md
  - backlog/tasks/task-15506 - Lead-File-Notes-push-review-with-outcomes.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the final Session Git push decision scannable by presenting what will change, where it will go, the exact session scope, and side effects before technical provenance while retaining complete operational detail on demand.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The push review leads with plain-language blocks for what changes and where they go.
- [x] #2 The push review shows the exact included note scope before confirmation.
- [x] #3 The push review states local and remote side effects before confirmation.
- [x] #4 Object identity, refs, lease, endpoint, transport, hooks, authentication, and provenance remain complete and keyboard-reachable under collapsed Technical details.
- [x] #5 The review and its disclosure remain usable and scrollable at 40x20 and normal layouts with focused UI regressions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused regressions that require the final push review to expose four outcome-first sections while keeping technical provenance collapsed by default and keyboard-reachable.

2. Restructure the presentation-only push review into visible change, destination, exact-scope, and side-effect summaries followed by a standard Technical details disclosure with complete provenance and endpoint inspection.

3. Add scoped semantic styling for the disclosure that preserves content-safe focus and compact 40x20 scrolling.

4. Run the focused Session Git push and File Notes layout matrix, Ruff, compile checks, and diff checks; then self-review and complete the task notes.

ADR required: no

ADR path: backlog/decisions/011-chatbook-workbench-ui-system.md

Reason: This is a bounded information-hierarchy and progressive-disclosure refinement inside the existing presentation widget. ADR-011 already requires visible workflow controls, contextual detail, stable composition, and responsive behavior; no service, storage, ownership, security, or long-lived application boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reordered the immutable Session Git push review into four visible decision sections: What changes, Where it goes, Exact scope, and Side effects.

Moved candidate identity, branch and ref data, sanitized endpoint, lease, transport, authentication limits, local-hook behavior, and Git-object provenance into a collapsed Technical details disclosure. The existing selectable endpoint dialog remains available inside the disclosure.

Added content-safe semantic disclosure styling and an exact descendant-focus scroll repair so the nested endpoint action is actually painted and reachable at 40x20 rather than merely holding offscreen focus.

Expanded mounted regressions for outcome copy, collapsed-by-default behavior, complete technical detail, endpoint modal return focus, compositor visibility, keyboard expansion, scrolling, and fixed-footer geometry. The guarded-push module passed 56 tests with four dev-baseline copy-mismatch cases deselected; origin/dev itself renders Checking push while those four tests expect Push checking.

Ruff, compileall, and git diff --check passed. CSS bundle generation was not applicable because the scoped styling lives in the widget's DEFAULT_CSS and was parsed by mounted Textual tests. Added the nested-focus compositor incident to lessons-testing-evidence.md.

ADR required: no. ADR-011 already defines the applicable visible-workflow, progressive-detail, stable-composition, and responsive behavior contract.
<!-- SECTION:NOTES:END -->
