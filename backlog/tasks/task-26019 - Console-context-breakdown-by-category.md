---
id: TASK-26019
title: 'Console: context breakdown by category'
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
labels:
  - console
  - context
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user cannot see what is consuming the context window. Verified on origin/dev: Widgets/Console/console_context_controls.py:105,118 shows a request row and a conversation row plus overhead, and a named grep for breakdown across Chat/ and Widgets/Console/ returns only cost-modal rows - so when a window fills, there is no way to tell whether tool schemas, RAG results, attachments or history are responsible. Hermes splits the window into eight named categories with a glyph grid. Chatbook's PreparedConsoleRequest accounting already separates memory, compactable and overhead, so the data is partly assembled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The context surface reports usage split by named category covering at minimum: system prompt, tool schemas, retrieved context, attachments, memory summary and live conversation
- [ ] #2 Category figures are derived from the same accounting used to build the request, not estimated separately - a mismatch between the two is impossible by construction
- [ ] #3 Categories that cannot be attributed are shown as an explicit unattributed bucket rather than silently folded into another
- [ ] #4 The existing model-window honesty is preserved: an unverified window is still labeled as estimated
- [ ] #5 The breakdown updates without a model call
- [ ] #6 Where a category is large enough to act on, the surface names the action that would reduce it
<!-- AC:END -->
