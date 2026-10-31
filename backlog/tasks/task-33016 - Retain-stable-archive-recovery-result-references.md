---
id: TASK-33016
title: Retain stable archive recovery result references
status: Done
assignee:
  - '@codex'
created_date: '2026-09-27 15:26'
updated_date: '2026-09-27 16:51'
labels: []
dependencies:
  - TASK-33015
documentation:
  - Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md
  - backlog/decisions/166-agent-assisted-archive-recovery.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep displayed recovery choices attached to exact identities across searches and retries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Session-owned result references enforce the approved retention and expiry limits.
- [x] #2 Manual refresh and imported transcript content cannot create model evidence or trusted controls.
- [x] #3 Wrong-owner and expired handles are refused; cursors retain their original normalized filters and result handles grant no restoration permission.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md. Reason: approved session-owned immutable recovery references. Follow Task 3 in Docs/superpowers/plans/2026-09-27-agent-archive-recovery.md: deterministic ownership/expiry red tests, bounded random references and cursor registry, model-visible/manual provenance, final UTF-8 response fitting, scoped validation and commit. Native call settlement is implemented by the coordinator delivery unit, not this registry. Use /tmp/archive-recovery-venv/bin/python; targeted tests only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented immutable session-owned archive results, opaque cursor/ref lookup, manual-only provenance, expiry/FIFO retention and UTF-8 response fitting with lifecycle identities preserved. ADR: backlog/decisions/166-agent-assisted-archive-recovery.md. Independent review found mutable supplied cursor anchors; fixed and re-reviewed clean. 95 targeted tests passed; scoped Ruff lint/format and diff checks clean. Runtime/card enforcement remains assigned Tasks 5/6; handles confer no restore permission.
<!-- SECTION:NOTES:END -->

## Filing provenance

Created with Backlog CLI in a temporary Git workspace using the project configuration, after the production CLI repeatedly scanned local branch history. Copied into this project with a unique ID above the refreshed all-ref/worktree ceiling and current local task IDs; dependencies and contents were verified.
