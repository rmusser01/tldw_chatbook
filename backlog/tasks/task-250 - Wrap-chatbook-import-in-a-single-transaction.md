---
id: TASK-250
title: Wrap chatbook import in a single transaction
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, chatbooks]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
chatbook_importer._import_conversations commits per add_conversation/add_message/set_message_attachments — ~1,500 commits for a 50-conversation × 30-message import. TransactionContextManager is reentrant (ChaChaNotes_DB.py:9703-9722), so one outer transaction per chatbook (or per conversation) is a pure win. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import runs under an outer transaction; commit count measured before/after on a synthetic chatbook
- [ ] #2 Round-trip + conflict-resolution tests green; partial-failure semantics documented
<!-- AC:END -->
