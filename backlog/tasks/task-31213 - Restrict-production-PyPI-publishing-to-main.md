---
id: TASK-31213
title: Restrict production PyPI publishing to main
status: Done
assignee: []
created_date: '2026-09-03 22:34'
updated_date: '2026-09-03 22:40'
labels:
  - packaging
  - ci
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure production PyPI publishing cannot be triggered from dev or release tags, and only runs from protected main branch pushes when the built version is not already on PyPI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Production PyPI publishing is triggered only by pushes to main.
- [x] #2 The production publish job requires refs/heads/main and a protected ref.
- [x] #3 Tag pushes do not trigger production PyPI publishing.
- [x] #4 A version-exists guard prevents duplicate production uploads on main pushes.
- [x] #5 The GitHub pypi environment deployment policy permits main only.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add workflow policy regression tests for the PyPI trigger and publish job guards.
2. Change .github/workflows/publish-pypi.yml to trigger production on protected main pushes instead of v* tag pushes.
3. Add a PyPI version-exists check before the production publish job so ordinary main pushes do not attempt duplicate uploads.
4. Verify the live GitHub pypi environment policy allows only the main branch.
5. Run targeted packaging tests and diff checks.

ADR required: no
ADR path: N/A
Reason: This is CI/release policy wiring for an existing packaging workflow, not a storage, runtime, API, or architectural boundary change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed the PyPI workflow so production publishing is triggered by protected `main` branch pushes instead of `v*` tag pushes. The build job now exports the project version through a validated GitHub Actions output writer, a separate PyPI status job checks whether that version already exists, and the production publish job runs only when that check reports a new version. TestPyPI remains a manual dispatch from protected `dev`.

Extracted the PyPI version-exists guard into `Packaging/check_pypi_release.py`, added executable coverage for existing releases, missing releases, unexpected HTTP errors, validated output-path writes, workflow-facing output emission, and serialized same-ref workflow runs. Updated PyPI release documentation and the release build wrapper output so maintainers no longer follow the old tag-driven production path. Verified the live GitHub `pypi` environment deployment policy now permits only the `main` branch.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Packaging/test_release_metadata.py -q` -> 15 passed, 1 existing dependency warning
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile Packaging/check_pypi_release.py Tests/Packaging/test_release_metadata.py` -> passed
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Packaging/check_pypi_release.py Tests/Packaging/test_release_metadata.py` -> passed
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "from pathlib import Path; import yaml; yaml.safe_load(Path('.github/workflows/publish-pypi.yml').read_text()); print('yaml ok')"` -> yaml ok
- `bash -n Packaging/build_release.sh` -> passed
- `git diff --check` -> passed
- `gh api repos/rmusser01/tldw_chatbook/environments/pypi/deployment-branch-policies` -> only `main` branch policy
<!-- SECTION:NOTES:END -->
