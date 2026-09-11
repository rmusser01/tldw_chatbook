---
id: TASK-32281
title: >-
  Exact-input allow rules: review and remove UI, honoured on every row that
  offers it
status: To Do
assignee: []
created_date: '2026-09-10 19:13'
labels:
  - mcp
  - permissions
  - approvals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'Always allow this exact input' persists argument rules that no UI lists or removes (no references under UI/ or Widgets/), and Virtual CLI rows offer it while their verdict path recognises only once, session, always and deny. User decision 2026-09-10: keep the option on the card and build the rules list. Source: live approval-card / MCP-permissions UX review 2026-09-10 against dev 3315241674 (snapshot .impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tool inspector lists each stored exact-input allow with its arguments and a Remove action, and removing it makes the next matching call ask again.
- [ ] #2 The Permissions matrix marks tools that carry argument rules.
- [ ] #3 Every producer that offers the exact-input decision honours it (Virtual CLI included) or narrows its options so it is not offered.
- [ ] #4 The user guide describes the rule and where to remove it.
<!-- AC:END -->
