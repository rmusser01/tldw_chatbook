---
id: TASK-21251
title: Keep Windows startup alive when Actor Pack cleanup is unsupported
status: In Progress
assignee: []
created_date: '2026-08-24 00:09'
updated_date: '2026-08-24 03:00'
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
- [x] #5 Attempting to inspect a valid Actor Pack when private staging is usable but unverified fails with actor_pack_import_disk_unavailable before any staging candidate is created.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved fail-closed startup behavior and link the governing Actor Pack ADR.
2. Add a regression test proving unsupported-platform cleanup is a non-destructive no-op during service construction.
3. Implement the smallest guard that skips startup sweeping when platform verification is unavailable while preserving errors for unusable staging.
4. Run focused Actor Pack tests, relevant startup coverage, and static checks.
5. Address verified PR review feedback with a red-green regression that preserves the stable disk-unavailable category when private staging cannot be verified.

ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: This routine boot and error-classification fix preserves ADR-074’s existing fail-closed authority boundary and introduces no new architecture or security policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the minimal fail-closed startup guard: usable but unverified private staging now makes Actor Pack startup sweeping a non-destructive no-op, while unusable staging still fails and verified cleanup behavior remains unchanged. Added one focused regression test proving unverified staging is neither enumerated nor read and its candidate remains intact.

Prior passing evidence on 2026-08-23: the Actor Pack import module run (`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs/test_actor_pack_import.py -q`) passed 24/24 tests; the Actor Pack package run (`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs -q`) passed 202/202 tests. Both commands exited 0. Each run reported one existing RequestsDependencyWarning and non-failing post-summary cleanup warnings for leftover pytest temporary directories.

Fresh current-HEAD evidence on 2026-08-23: startup selection (`PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs/test_actor_pack_import.py -q -k startup_sweep`) passed 2 tests with 22 deselected; Ruff format check reported `2 files already formatted`; Ruff lint reported `All checks passed!`; exact command `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q tldw_chatbook/Actor_Packs/importer.py Tests/Actor_Packs/test_actor_pack_import.py` exited 0; `git diff --check origin/dev...HEAD` exited 0. The startup selection also emitted the existing RequestsDependencyWarning and non-failing post-summary cleanup warnings for leftover pytest temporary directories.

Repository-wide verification limitation: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q` produced no terminal result after roughly 100 minutes and was interrupted. This is neither a pass nor a failure and is the sole reason TASK-21251 remains In Progress.

Scope review confirmed the functional diff is one focused test plus the minimal early-return guard, with no app.py, archive, schema, dependency, or config changes. ADR required: no; ADR-074 remains governing. Lessons: no new general lesson; the incident is specific to the documented platform split.

PR review remediation (Qodo): verified that unverified private staging was converted through inspect_archive() to actor_pack_import_invalid. _preflight_space() now raises the existing actor_pack_import_disk_unavailable category directly, which propagates unchanged, preserves fail-closed behavior, and matches existing UI recovery copy. A red-green regression proves a valid pack receives disk-unavailable before any staging candidate is created. Verification: targeted regression passed 1/1; `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Actor_Packs -q` passed 203/203; Ruff format reported both changed Python files already formatted; Ruff lint and git diff --check passed. TASK-21251 remains In Progress solely because the previously recorded repository-wide pytest run has no terminal result.
<!-- SECTION:NOTES:END -->
