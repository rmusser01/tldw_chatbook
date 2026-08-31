---
id: TASK-26009
title: 'Secret redaction by value shape, not just key name'
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
labels:
  - security
  - mcp
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Redaction can be defeated by a value that does not sit under a recognized key. Verified on origin/dev: MCP/redaction.py:1-114 matches on key names, CLI argument names and URL query parameters only, and line 64 documents its own bypass - a secret value beginning with a dash survives. The exposure surfaces are the approval card (Widgets/Chat_Widgets/chat_approval_card.py:44) and the execution log. Hermes matches value shapes: provider key prefixes, JWTs, private-key blocks, database connection strings and bearer headers. This is a pure function with no caller changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A value matching a known secret shape is redacted regardless of the key or argument name it appears under
- [ ] #2 Shapes covered include at minimum: common provider key prefixes, JWTs, PEM private-key blocks, database connection URIs and Authorization header values
- [ ] #3 The documented dash-prefixed bypass at MCP/redaction.py:64 no longer applies
- [ ] #4 Redaction is applied on both the display path (approval card) and the stored path (execution log) - verified separately for each
- [ ] #5 False positives are bounded: ordinary prose, file paths and git SHAs are not redacted, asserted by tests
- [ ] #6 Redaction remains a pure function with no new I/O or configuration dependency
<!-- AC:END -->
