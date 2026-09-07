---
id: TASK-15742
title: Make POSIX-only tests collect safely on Windows
status: Done
assignee: []
created_date: '2026-08-13 20:15'
updated_date: '2026-08-13 21:04'
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
- [x] #1 The media playback test module collects on Windows without evaluating unavailable SIGSTOP or SIGCONT constants
- [x] #2 POSIX signal behavior remains covered where supported and documented non-POSIX no-op behavior is covered
- [x] #3 TTS profile materialization tests skip clearly when fcntl is unavailable while retaining POSIX coverage
- [x] #4 The repository-wide suite advances beyond the two prior collection errors
- [x] #5 Focused tests and static checks pass
- [x] #6 No production behavior schema dependency or ADR changes are introduced
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture both Windows collection failures as RED evidence and gate media signal tests by capability.
2. Gate the POSIX TTS materialization suite when fcntl is unavailable.
3. Run focused and repository-wide collection verification plus static checks.
4. Complete TASK-15742 notes and acceptance criteria, rebase on current dev, then push and create the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a test-only Windows collection repair. Root causes were import-time evaluation of unavailable SIGSTOP/SIGCONT constants in the media parameter list and an unconditional fcntl import in the POSIX TTS suite. Media tests now gate only signal-delivery contracts by capability, resolve signal names inside supported tests, and cover the existing no-signal clock/no-kill behavior. The TTS module now skips explicitly when fcntl is absent while retaining all POSIX contracts.

Verification: repository-wide collect-only exited 0 with 42,293 tests collected in 100.99s and seven explicit module skips, including the TTS fcntl reason; neither prior collection error remains. The combined media+TTS matrix exited 0 with 33 passed and 4 skipped in 2.58s. Running the TTS module alone intentionally returns pytest exit 5 because its explicit module skip leaves no selected tests. Scoped Ruff, py_compile, and git diff --check passed. The repository-wide suite was stopped at the mandated 300.3s ceiling after reaching 3%; at least ten failure markers appeared without identities or tracebacks. Those failures remain unclassified, and the available evidence does not attribute them to TASK-15742.

No production behavior, schema, dependency, configuration, migration, or runtime boundary changed. ADR required: no; this test-only portability correction preserves the existing runtime architecture. Modified task files: Tests/Media_Playback/test_player_pipeline.py and Tests/TTS/test_profile_reference_materialization.py, plus design, plan, and Backlog documentation. No generalizable new lesson was added.
<!-- SECTION:NOTES:END -->

## References

- Design: `Docs/superpowers/specs/2026-08-13-task-15742-windows-posix-test-collection-design.md`
- ADR required: no; this is a test-only portability correction that preserves existing runtime boundaries.
