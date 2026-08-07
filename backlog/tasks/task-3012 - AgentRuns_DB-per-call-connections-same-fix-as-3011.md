---
id: TASK-3012
title: AgentRuns_DB opens per-call connections — apply the task-3011 held-connection fix
status: To Do
assignee: []
created_date: '2026-08-07 06:00'
labels:
  - db
  - agents
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while gating task-3011: `AgentRuns_DB`'s module docstring says it "follows the Workspace_DB pattern: BaseDB, per-call connections" — the exact anti-pattern task-3011 just removed from WorkspaceDB (which measured ~60% of the Console push). AgentRuns is hot during agent runs (per-step persistence), so each step currently pays full private-SQLite connection setup. Apply the same thread-local held-connection idiom (idle-gated liveness ping, per-thread isolation, close() teardown) with the same three-test shape (reuse pin watched RED, rollback+usable guard, per-thread guard). Note the agent service runs on a worker thread — the thread-local design covers it, but the audit should confirm no connection is shared across the service/main boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] Repeated AgentRuns reads/writes on one thread open no new connections after warm-up (spy-pinned, watched RED first).
- [ ] Transaction rollback semantics and post-failure usability preserved; per-thread isolation pinned.
- [ ] Existing AgentRuns/Agents test files green.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
