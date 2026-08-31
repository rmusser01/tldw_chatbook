---
id: TASK-26042
title: Console Workspace Files read-only inspector
status: In Progress
assignee: []
created_date: '2026-08-31 16:24'
updated_date: '2026-08-31 17:06'
labels: []
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-31-workspace-files-inspector-design.md
  - >-
    backlog/decisions/079-workspace-file-inspector-direct-user-authority-and-save-publication.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give Console users a safe way to inspect files from any visible named workspace without activating it or changing the current task, session, conversation, composer, approvals, or workspace context. Deliver the non-activating modal, explicit binding scope, bounded tree/filter, safe viewer, large-file paging, attention return path, and lifecycle behavior as a useful read-only slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either Console entry point opens exactly one inspector visit for the selected named workspace while every active Console context fingerprint remains unchanged.
- [x] #2 Every current local-folder binding is represented with explicit identity, access mode, and availability; no unavailable or changed binding silently falls back or retargets the modal.
- [x] #3 The modal provides bounded directory paging and selected-binding-only literal filtering with visible loading, progress, partial, empty, cancelled, failed, and truncated states.
- [x] #4 Safe UTF-8 files are viewable, files over 200,000 decoded characters through 8 MiB use revision-pinned pages of at most 100,000 characters, and files over 8 MiB remain metadata-only.
- [x] #5 Hostile filesystem names and unsafe file control text render as visible escaped text without markup or terminal injection while raw path identity remains separate and revalidated.
- [x] #6 Back to Console, Escape, backdrop dismissal, resize, duplicate activation, generic Console attention, and graceful quit preserve the specified focus, state, privacy, and teardown behavior.
- [ ] #7 Production-shaped Textual and live scratch evidence covers 80x24, 100x30, 120x40, and 160x50 layouts, non-active context preservation, paging, hostile text, privacy, and prohibited side effects.
- [x] #8 List, read, and filter work is bounded, coalesced, cancellable where safe, stale-result resistant, and leaves no workers or transient modal resources after graceful teardown.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes\nADR path: backlog/decisions/079-workspace-file-inspector-direct-user-authority-and-save-publication.md\nReason: Implements ADR-079's non-activating Console modal, direct-user read authority, revalidation, privacy, and bounded lifecycle boundaries.\n\n1. Build the revalidating read-only filesystem service with bounded list/filter/read paging and hostile-text safety.\n2. Build the Console safe modal with bounded worker lanes, responsive layouts, focus/dismiss behavior, and viewer states.\n3. Wire both typed non-activating entry points, single-visit admission, privacy-minimized attention, and graceful lifecycle behavior.\n4. Complete production-shaped and isolated live scratch evidence, documentation, independent whole-slice review, and task closure.\n\nDetailed plan: Docs/superpowers/plans/2026-08-31-task-26042-workspace-files-read-only.md
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked only after their behavior and prohibited side effects are evidenced.
- [x] #2 The task is moved to In Progress before an Implementation Plan is added, and that plan records ADR required: yes, ADR-079, and the reason.
- [x] #3 Targeted automated tests, relevant static checks, and git diff --check pass; a full suite is run only after explicit user approval.
- [ ] #4 Production-shaped Textual evidence and an isolated live scratch verification cover the user-facing path and preserve unrelated Console and profile state.
- [x] #5 Relevant documentation and concise Implementation Notes identify the approach, trade-offs, files changed, verification, and any plan deviation.
- [ ] #6 A self-review confirms security, privacy, accessibility, performance, licensing, task dependencies, and no unrelated regression before the task is set to Done.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 4 added four-size production-CSS evidence using the real Textual Console
screen and isolated temporary workspace/profile/database roots. The 80x24
route follows the existing single-pane shell contract through the typed
controller admission seam; the 100x30, 120x40, and 160x50 routes use visible,
scrolled, real pointer clicks on both active and non-active entry controls.
The evidence fingerprints Console context, approvals, bindings, temporary
files, and redirected profile/config/data/registry-database content before
and after modal navigation/dismissal. It revealed and fixed a compact-mount
status-query race in the modal. ADR-079 remains applicable; no new ADR or
lesson is required. Targeted verification is recorded in the Task 4 report.
The Task 4 Textual harness evidence is not a substitute for the remaining true
live-TUI scratch run, so AC #7 and Definition of Done #1/#4 are intentionally
unchecked. The task remains In Progress pending that evidence and the required
independent whole-slice review; Definition of Done #6 and the Done status are
intentionally unchanged.
<!-- SECTION:NOTES:END -->
