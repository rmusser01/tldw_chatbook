---
id: TASK-32462
title: 'Design: per-tool persona-floor display in MCP Permissions'
status: To Do
assignee: []
created_date: '2026-09-11 23:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from Wave C (TASK-32454): the Permissions matrix shows stored tool policy only; Console personas layer additional floors (Agents/persona_policy.py persona_floor_state) on top at runtime, so a user auditing a ws-* profile cannot see which tools stay Ask regardless of Allow. The open design question: which persona context resolves the floors when the MCP screen views a workspace profile (the screen has no persona selection; the workspace may have several). Candidate approaches: (a) show floors for the workspace's currently-active assistant persona, (b) a persona picker in the matrix, (c) a read-only 'floors apply' aggregate badge per tool derived from ALL personas for that workspace. v1 shipped an informational hint line only. Needs a product decision before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design decided (which persona context + which surface),ADR written if it changes permission-display semantics,Implemented with tests showing effective-after-floor states for a selected workspace profile
<!-- AC:END -->
