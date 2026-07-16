---
id: TASK-249
title: Use messages_fts in search_conversations_page instead of correlated LIKE
status: To Do
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, db]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ChaChaNotes_DB.search_conversations_page (4924-4936) matches message content via correlated EXISTS + leading-wildcard LIKE per candidate row; the schema already maintains messages_fts and search_messages_by_content (6336-6360) uses it correctly. Measured 1.4→7.8ms per scope with an active query; multiplied by Console's per-tick scope loop this reaches ~70ms/tick during streams (see task-251). Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P0 A4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Content matching goes through messages_fts MATCH; results equivalent for representative queries (incl. no-match, special chars)
- [ ] #2 Measured per-scope query time with an active search string drops by >2x on a seeded large DB
- [ ] #3 Existing conversation-search tests green
<!-- AC:END -->
