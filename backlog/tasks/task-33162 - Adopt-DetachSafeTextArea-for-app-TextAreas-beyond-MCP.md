---
id: TASK-33162
title: Adopt DetachSafeTextArea for app TextAreas beyond MCP
status: To Do
assignee: []
created_date: '2026-09-28 02:11'
labels:
  - ci-throughput-3-candidate
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33115 (PR #2866) swapped every TextArea construction site under tldw_chatbook/UI/MCP_Modules/ for DetachSafeTextArea, closing the text-area--gutter detach-race KeyError there. The spec scoped that fix (D) to the five MCP sites only. About 127 other TextArea( constructions across the app still use the stock widget and remain exposed to the same detach race (see also TASK-32456/TASK-32298, the Library Notes canvas instance). Candidate for CI throughput sub-project 3.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every remaining stock TextArea( construction site that can be detached with a pending repaint is inventoried
- [ ] #2 Each site in scope is swapped for DetachSafeTextArea or an equivalent guard
- [ ] #3 A regression test demonstrates the detach race no longer raises at each swapped site
<!-- AC:END -->
