---
id: TASK-26044
title: Console Workspace Files isolated Git decoration
status: To Do
assignee: []
created_date: '2026-08-31 16:26'
updated_date: '2026-08-31 16:28'
labels: []
dependencies:
  - TASK-26043
references:
  - Docs/superpowers/specs/2026-08-31-workspace-files-inspector-design.md
  - >-
    backlog/decisions/079-workspace-file-inspector-direct-user-authority-and-save-publication.md
  - task-26042
  - task-26043
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add useful Git status to the completed Workspace Files inspector without allowing repository state, configuration, hooks, environment redirects, malformed paths, or Git failure to influence filesystem authority or editing. Deliver accessible, fail-soft decoration through a dedicated bounded read-only reader.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The inspector shows accessible tracked, modified, untracked, added, deleted, renamed, conflicted, ignored, unavailable, and truncated Git states without relying on color alone or hiding the selected file's full state.
- [ ] #2 Git status runs through an absolute resolved executable with no-replace-objects, literal pathspecs, fsmonitor/rename/maintenance/gc suppression, NUL-delimited porcelain-v2 output, optional-lock suppression, no lazy fetch, and isolated system/global configuration.
- [ ] #3 The adapter never stages, refreshes or writes the index, runs hook-capable behavior, prompts, opens a pager/editor, accepts caller-supplied Git redirects, mutates repository configuration, or feeds parsed output back into command arguments.
- [ ] #4 Raw Git path bytes are parsed losslessly for filesystem identity, rendered through the hostile-text formatter, and revalidated inside the explicitly selected binding before decoration.
- [ ] #5 Git work is lazy per binding and nested repository, permits one active plus one coalesced latest request per binding under the modal-wide concurrency cap, and terminates subprocess groups cleanly on cancellation, dismissal, and graceful quit.
- [ ] #6 Git absence, timeout, malformed output, truncation, nested-repository complexity, or cancellation remains a local decoration state and cannot alter browsing, authority, editability, Save outcomes, or Console context.
- [ ] #7 Known durable Save and explicit Refresh update Git decoration without confusing Git state with Unsaved, Conflict, or Edited this visit precedence.
- [ ] #8 Real-repository tests cover status/path parsing, hostile fsmonitor and caller environment isolation, timeout/output bounds, subprocess teardown, nested repositories, and before/after index and repository fingerprints proving no mutation.
- [ ] #9 Production-shaped Textual and live scratch evidence verifies decoration, legend/focus behavior, fail-soft operation, responsive layouts, and unchanged authority when Git is missing or hostile.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 All acceptance criteria are checked only after their behavior and prohibited side effects are evidenced.
- [ ] #2 The task is moved to In Progress before an Implementation Plan is added, and that plan records ADR required: yes, ADR-079, and the reason.
- [ ] #3 Targeted automated tests, relevant static checks, and git diff --check pass; a full suite is run only after explicit user approval.
- [ ] #4 Production-shaped Textual evidence and an isolated live scratch verification cover the user-facing path and preserve unrelated Console and profile state.
- [ ] #5 Relevant documentation and concise Implementation Notes identify the approach, trade-offs, files changed, verification, and any plan deviation.
- [ ] #6 A self-review confirms security, privacy, accessibility, performance, licensing, task dependencies, and no unrelated regression before the task is set to Done.
<!-- DOD:END -->
