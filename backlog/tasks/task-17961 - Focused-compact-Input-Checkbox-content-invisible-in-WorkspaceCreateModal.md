---
id: TASK-17961
title: >-
  Focused compact Input/Checkbox content invisible in
  WorkspaceCreateModal
status: To Do
assignee: []
created_date: '2026-08-18 03:09'
labels:
  - workspaces
  - ui
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live-verifying task-18704 (shared workspace creation modal) found that the Name Input, folder-path Input, and the "Switch to this workspace" Checkbox in tldw_chatbook/Widgets/workspace_create_modal.py all render as an empty bordered box -- top and bottom border rows with zero content rows, hiding the value/label entirely -- whenever that specific widget has keyboard focus. Blurred, each renders correctly as a single-line compact row with its value/placeholder visible. The underlying data is unaffected (confirmed: typed values persist across focus changes, folder Add/Remove works, and Create/Cancel/toast behavior is functionally correct end to end), so this is a pure rendering defect, but a severe one: a user tabbing through the form or actively typing into the Name or folder-path field sees nothing on screen while doing so. A structurally similar collapsed-content-row appearance was also observed on the pre-existing, unrelated "Show archived" Checkbox in Settings ▸ Workspaces (which does not use compact=True), suggesting the root cause may be a broader pre-existing Textual/CSS rendering characteristic (possibly version 8.2.8's interaction between a widget's compact/tall border styles and its :focus state) rather than something newly introduced only by this modal -- needs root-causing across both cases before concluding scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce headlessly (Pilot test or compositor render_strips check) that a focused compact Input/Checkbox in WorkspaceCreateModal shows its label/value
- [ ] #2 Root-cause whether this is scoped to WorkspaceCreateModal's CSS or a broader Textual/app-CSS interaction also affecting Settings' non-compact Show archived checkbox
- [ ] #3 Fix the rendering so a focused field/checkbox in the modal always shows its current value or label
- [ ] #4 Add a regression test asserting focused-state content is visible via Screen._compositor.render_strips(), not terminal-capture text alone
<!-- AC:END -->
