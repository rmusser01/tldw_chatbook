---
id: TASK-21251
title: Keep Windows startup alive when Actor Pack cleanup is unsupported
status: In Progress
assignee: []
created_date: '2026-08-24 00:09'
updated_date: '2026-08-24 00:43'
labels:
  - bug
  - actor-packs
  - windows
dependencies: []
references:
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent Actor Pack startup housekeeping from terminating the application on platforms where private staging authority cannot be verified, while preserving the existing fail-closed cleanup boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Constructing the Actor Pack import service does not raise when private staging is usable but platform verification is unavailable.
- [x] #2 Startup cleanup does not enumerate, modify, or delete staged candidates unless private staging authority is verified.
- [x] #3 Authenticated startup cleanup behavior on supported platforms remains unchanged.
- [x] #4 Automated regression coverage exercises the unsupported-platform startup path and the supported cleanup path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved fail-closed startup behavior and link the governing Actor Pack ADR.
2. Add a regression test proving unsupported-platform cleanup is a non-destructive no-op during service construction.
3. Implement the smallest guard that skips startup sweeping when platform verification is unavailable while preserving errors for unusable staging.
4. Run focused Actor Pack tests, relevant startup coverage, and static checks.

ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: This routine boot bug fix preserves ADR-074’s existing fail-closed authority boundary and introduces no new architecture or security policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the minimal fail-closed startup guard: usable but unverified private staging now makes Actor Pack startup sweeping a non-destructive no-op, while unusable staging still fails and verified cleanup behavior remains unchanged. Added one focused regression test proving unverified staging is neither enumerated nor read and its candidate remains intact.

Verification on 2026-08-23: PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs/test_actor_pack_import.py -q -> 24 passed, 0 skipped, 0 deselected, pytest exit 0; PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs -q -> 202 passed, 0 skipped, 0 deselected, pytest exit 0. Each run reported 1 RequestsDependencyWarning from the existing requests dependency version mismatch. Each run also emitted 94 post-summary PytestWarning cleanup messages while retrying immutable temporary directories; these teardown warnings were non-failures and both commands exited 0.

Static and patch checks: compileall -q for tldw_chatbook/Actor_Packs/importer.py and Tests/Actor_Packs/test_actor_pack_import.py exited 0; git diff --check origin/dev...HEAD exited 0. Scope review confirmed the functional diff is one focused test plus the minimal early-return guard, with no app.py, archive, schema, dependency, or config changes.

ADR required: no; ADR-074 remains governing. Lessons: no new general lesson; the incident is specific to the documented platform split.
<!-- SECTION:NOTES:END -->
