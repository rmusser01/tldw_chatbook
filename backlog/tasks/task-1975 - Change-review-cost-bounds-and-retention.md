---
id: TASK-1975
title: 'Change review: root budgets, oversize excludes, snapshot retention/GC'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - workspaces
  - change-review
  - performance
dependencies:
  - TASK-1970
  - TASK-1971
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the substrate bounded. Registration scan enforces max_files/max_total_bytes per root — over budget disables tracking for that root with honest copy ('narrow the root or add excludes'), never a silent half-track. Git cannot exclude by size, so the scan appends files over max_file_bytes to info/exclude dynamically and the review discloses 'N oversized files untracked'. Retention: prune change_snapshots rows past retention_days (default 30), then reflog expire + git gc --prune on the shadow repo, on the existing maintenance path; orphaned shadow repos (root removed) GC'd by age.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An over-budget root registers with tracking disabled and the stated copy; an in-budget root tracks
- [ ] #2 A file exceeding max_file_bytes lands in info/exclude, is absent from diffs, and the untracked count is disclosed in the review
- [ ] #3 Retention run prunes rows older than the cutoff AND shrinks the shadow repo (object count measured before/after)
- [ ] #4 A shadow repo whose root no longer exists is removed by GC
- [ ] #5 All knobs read from [workspaces.change_review] with env-var overrides per repo convention
<!-- AC:END -->
