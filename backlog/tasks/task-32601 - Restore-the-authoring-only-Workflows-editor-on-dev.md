---
id: TASK-32601
title: Restore the authoring-only Workflows editor on dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-15 04:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved portable workflow editor on current dev while keeping unfinished execution and SQLite ownership changes isolated.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can create, edit, save, import and export workflow definitions in the canonical Workflows screen.
- [ ] #2 The library, step navigator, overview and collapsed continuous form remain usable with keyboard and at supported terminal sizes.
- [ ] #3 Drafts survive navigation and restart, failed persistence stays recoverable, and opaque server fields survive round trips.
- [ ] #4 Only existing SQLite safety utilities are used; new execution and process-ownership infrastructure remain absent.
<!-- AC:END -->

## Implementation Plan

ADR required: no new decision.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md; ADR-125 and ADR-150 also apply.
Reason: implement only the previously approved authoring contract on current dev; runtime and the unapproved helper-owned-lock proposal are excluded.

1. Preserve the original and integration branches; use the clean codex/workflows-authoring-dev worktree based on 77eb2601a6.
2. Follow Docs/superpowers/plans/2026-09-14-workflows-authoring-dev.md. Reuse the reviewed editor-only checkpoint b34eda3d64 and the lossless document/draft code.
3. Register workflow document storage through current private_sqlite utilities. Retain existing migrations for file compatibility, but omit runtime locks and execution APIs.
4. Wire the editor into the real app, including durable draft flushing before navigation/quit and explicit local JSON import/export. Preserve current Console follow behavior.
5. Check the current tldw_server dev definition contract; verify targeted storage, authoring, lifecycle, keyboard and production-CSS behavior. Review actual captures at 160x48, 110x36 and 60x20.
6. Record exact verification and independent review. Do not mark Done with unresolved failures or unwaived static debt.

## ID provenance

The CLI offered TASK-32591; the all-ref object-path and 38-worktree scans already contained IDs through 32600. Only this newly created file/header was moved to the checked-free TASK-32601 before implementation. No existing task was renumbered.
