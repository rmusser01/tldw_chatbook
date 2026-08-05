---
id: TASK-1342
title: 'Local agent tools phase 3b-i: fs_patch (unified-diff apply)'
status: In Progress
assignee: []
created_date: '2026-08-05 17:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md §2.4. Plan: Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-i.md. ADR-032. Port of tldw_server filesystem_diff.py @ 5605b9d9.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 fs_patch applies multi-file multi-hunk unified diffs confined to the workspace root
- [ ] #2 Context mismatches, deletes, renames, and malformed diffs return model-actionable errors without writing
- [ ] #3 dry_run returns the would-be result and writes nothing
- [ ] #4 Diff size/file/hunk limits enforced; writes are encode-before-write and newline-preserving
- [ ] #5 All new tests pass
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-08-05-local-agent-tools-phase3b-i.md
<!-- SECTION:PLAN:END -->
