---
id: TASK-1971
title: 'Change review: B/E turn snapshots around agent runs + change_snapshots schema'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
  - agents
  - db
dependencies:
  - TASK-1970
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the turn protocol: baseline snapshot B at run start (skipped when clean — B = previous tip), end snapshot E at run end; B == E means no changes. Records go in a new AgentRunsDB `change_snapshots` table (run_id, root, baseline_sha, end_sha, files_changed, adds, dels, reverted, tracking_error; schema bump per repo discipline). One row per (run, root) for multi-root workspaces. The FIRST snapshot of a root happens at root-registration time on a background worker — never as first-send latency. Failure posture: tracking never blocks the agent; a failed snapshot logs, stores tracking_error, and the run proceeds.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run that writes a file yields a change_snapshots row whose diff(B,E) matches disk truth exactly (property test against real git)
- [ ] #2 A run that touches nothing yields NO row and no card
- [ ] #3 A change made by a SCRIPT the agent ran (not write_file) appears in diff(B,E)
- [ ] #4 Snapshot failure (e.g. git binary removed mid-session) stores tracking_error and the agent reply still completes
- [ ] #5 Registering a root triggers its initial snapshot in the background; the first send performs no full-tree add
- [ ] #6 Schema migration applies cleanly to an existing AgentRunsDB
<!-- AC:END -->
