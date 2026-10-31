---
id: TASK-34781
title: >-
  SmartContentTree crashes on a conversation or note title containing
  markup-like brackets
status: To Do
assignee: []
created_date: '2026-10-10 18:13'
labels:
  - bug
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
SmartContentTree builds tree node labels from user titles (item.title plus subtitle) as plain str, and Textual's Tree runs every str label through Text.from_markup. A conversation or note titled with a stray closing tag such as [/b] raises MarkupError the same way the TASK-34780 toasts did, and a title with [tag] text loses it from the label. Found by the TASK-34780 review (SmartContentTree.py around the parent_node.add(label) call). The markup census does not check Tree labels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A conversation or note whose title contains [/b] or other markup-like brackets renders its title literally in SmartContentTree and never raises
- [ ] #2 A test on the real widget fails on dev with MarkupError and passes with the fix
<!-- AC:END -->
