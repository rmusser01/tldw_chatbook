---
id: TASK-31214
title: Prevent main PyPI workflow from publishing stale versions
status: Done
assignee: []
created_date: '2026-09-03 23:11'
updated_date: '2026-09-03 23:14'
labels:
  - packaging
  - ci
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the main-driven PyPI workflow cannot publish an older package version that is absent from PyPI but lower than the latest published release.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The PyPI release-status helper reports the latest published version.
- [x] #2 Production publishing is allowed only when the candidate version is absent and newer than the latest published version.
- [x] #3 A stale lower version on main skips production upload instead of publishing.
- [x] #4 Focused executable tests cover latest-version comparison and stale-version skip behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the PyPI release-status helper to read the package-level PyPI JSON and identify the latest valid published version.
2. Emit a publish_release output that is true only for absent versions newer than the latest published release.
3. Change the workflow publish condition to use publish_release rather than release_exists alone.
4. Add targeted tests for newer, existing, missing-package, and stale lower-version behavior.
5. Run focused tests, lint, syntax checks, and environment-policy verification.

ADR required: no
ADR path: N/A
Reason: This is a defensive CI release guard inside the existing packaging workflow, not a new architecture or release boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Extended `Packaging/check_pypi_release.py` so it checks both exact candidate-version existence and the latest package-level PyPI release. The workflow now consumes `publish_release`, which is true only when the candidate version is absent and greater than the latest valid published version. This prevents a stale `main` checkout at `0.1.8.0` from publishing after `0.1.8.1` already exists.

Updated PyPI release docs to state that protected `main` publishes only absent versions newer than the latest PyPI release. Added focused tests for newer absent versions, existing versions, stale lower versions, missing-project first releases, workflow output wiring, and a subprocess CLI run against a local PyPI-shaped HTTP server. Addressed review feedback by validating CLI inputs with Pydantic through the shared input-validation module, validating the PyPI JSON response shape, centralizing the PyPI request timeout, reporting the real latest release for existing stale candidates, rejecting custom output names that would overwrite fixed metadata, and keeping the selected Python interpreter consistent through the local release scripts.

Verification:
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Packaging/test_release_metadata.py -q` -> 25 passed
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Web_Scraping/test_input_validation.py -q` -> 9 passed
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m py_compile Packaging/check_pypi_release.py Packaging/common/dist_path.py Packaging/common/version.py Tests/Packaging/test_release_metadata.py tldw_chatbook/Utils/input_validation.py` -> passed
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check Packaging/check_pypi_release.py Packaging/common/dist_path.py Packaging/common/version.py Tests/Packaging/test_release_metadata.py tldw_chatbook/Utils/input_validation.py` -> passed
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "from pathlib import Path; import yaml; yaml.safe_load(Path('.github/workflows/publish-pypi.yml').read_text()); print('yaml ok')"` -> yaml ok
- `bash -n Packaging/build_release.sh` -> passed
- `bash -n Packaging/build_dist.sh` -> passed
- `git diff --check` -> passed
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Packaging/check_pypi_release.py 0.1.8.0` -> `release_exists=false`, `latest_version=0.1.8.1`, `publish_release=false`
- `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Packaging/check_pypi_release.py 0.1.8.2` -> `release_exists=false`, `latest_version=0.1.8.1`, `publish_release=true`
<!-- SECTION:NOTES:END -->
