---
id: TASK-33087
title: SQLite store kernel ADR and connection-transaction consolidation
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, db]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The DB layer carries 21 independent transaction() context managers and 9 verbatim copies of the held thread-local connection plus liveness-ping idiom, because base_db.py provides almost none of what a store actually needs. The copies know they are copies: RAG_Indexing_DB.py documents copying the Workspace_DB idiom, AgentRuns_DB.py says it follows the Workspace_DB pattern, and three classes carry close() aliases to feel like BaseDB without inheriting it. One kernel in base_db (held connection with liveness ping, transaction with defined nesting semantics, column-exists helpers) makes the base class real and deletes roughly 2k+ LOC of copied plumbing. ChaChaNotes' richer transaction semantics (depth tracking, quiescence) become the canonical implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ADR records the kernel decision with ChaChaNotes transaction semantics as canonical.
- [ ] #2 The held connection plus liveness-ping idiom is implemented once and consumed by every store that currently copies it.
- [ ] #3 Transaction nesting semantics are documented and covered by tests.
- [ ] #4 No store retains a private copy of the consolidated machinery.
- [ ] #5 Targeted DB test suites for every touched store pass.
<!-- AC:END -->
