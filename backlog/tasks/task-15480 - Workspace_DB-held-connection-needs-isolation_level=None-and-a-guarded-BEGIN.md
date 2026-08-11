---
id: TASK-15480
title: Workspace_DB held connection needs isolation_level=None and a guarded BEGIN
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - bug
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the latency audit: the task-3011 held-connection port of Workspace_DB omitted the lesson task-3012 later documented in AgentRuns (`DB/AgentRuns_DB.py:77-85`): Python sqlite3's default isolation mode auto-BEGINs on DML, so any future bare DML issued through `connection()` accumulates an implicit transaction that makes an explicit `BEGIN` raise ("cannot start a transaction within a transaction") and silently rolls back on close. `DB/Workspace_DB.py:37-40` lacks `isolation_level=None`, and its `transaction()` (`:78-89`) runs a bare `BEGIN` without checking `in_transaction` (Research/Writing check it; Subscriptions documents its reasoning). Latent today — audit the call sites — but a correctness fuse for the next contributor. Also fix the stale comment at `Tools/workspace_file_roots.py:43-45` still describing Workspace_DB as fresh-connection-per-op.

Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Workspace_DB holds its connection with isolation_level=None; the bare-DML call-site audit is recorded in the notes
- [ ] #2 transaction() is safe when a transaction is already open (test)
- [ ] #3 The stale workspace_file_roots comment is corrected; existing Workspace surfaces green
<!-- AC:END -->
