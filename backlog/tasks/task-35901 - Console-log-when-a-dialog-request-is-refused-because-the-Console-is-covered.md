---
id: TASK-35901
title: 'Console: log when a dialog request is refused because the Console is covered'
status: To Do
assignee: []
created_date: '2026-10-21 01:24'
labels:
  - console
  - diagnostics
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Since TASK-34720 the Console's dialog openers (Library access, the system prompt editor, the character picker, the retrieval-scope picker and the Conversation Inspector, alongside Switch model's earlier check) drop a request that finds another screen over the Console. That is right for a burst of activations. But a single deliberate activation is dropped too when an unrelated dialog is pushed while its opener awaits a read or runs as a worker: the startup model-list consent dialog, an approval card, or the Ctrl+Q prompt. Nothing records the drop, so a report that a segment or menu entry did nothing cannot be diagnosed from the app log (found by the TASK-34720 review).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each refused open writes one debug-level log line naming the dialog that was refused and the screen that covered the Console
- [ ] #2 The log line carries no conversation content, prompt text or credential
- [ ] #3 The production diagnostic inventory lists the new log statements
- [ ] #4 A test asserts the line is written on a refusal and not on a successful open
<!-- AC:END -->
