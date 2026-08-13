---
id: TASK-15742
title: Make POSIX-only tests collect safely on Windows
status: In Progress
assignee: []
created_date: '2026-08-13 20:15'
labels:
  - testing
  - windows
  - portability
dependencies: []
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore repository-wide pytest collection on Windows by gating POSIX-only signal and file-lock test contracts without changing production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The media playback test module collects on Windows without evaluating unavailable SIGSTOP or SIGCONT constants
- [ ] #2 POSIX signal behavior remains covered where supported and documented non-POSIX no-op behavior is covered
- [ ] #3 TTS profile materialization tests skip clearly when fcntl is unavailable while retaining POSIX coverage
- [ ] #4 The repository-wide suite advances beyond the two prior collection errors
- [ ] #5 Focused tests and static checks pass
- [ ] #6 No production behavior schema dependency or ADR changes are introduced
<!-- AC:END -->

## References

- Design: `Docs/superpowers/specs/2026-08-13-task-15742-windows-posix-test-collection-design.md`
- ADR required: no; this is a test-only portability correction that preserves existing runtime boundaries.
