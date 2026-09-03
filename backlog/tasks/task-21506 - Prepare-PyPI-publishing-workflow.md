---
id: TASK-21506
title: Prepare PyPI publishing workflow
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 05:42'
updated_date: '2026-08-24 06:05'
labels:
  - packaging
  - release
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the package publishing path ready for TestPyPI and PyPI by aligning release metadata, documentation, and automation with the installed-distribution contract already defined by ADR-032.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release version metadata is derived from or matches the package version used by PyPI artifacts.
- [x] #2 PyPI release documentation names the real console scripts, uses fresh artifact gates, and no longer teaches long-lived token upload as the default path.
- [x] #3 A GitHub Actions workflow can build checked artifacts, publish to TestPyPI on manual dispatch, and publish to PyPI from protected version tags using trusted publishing.
- [x] #4 Targeted packaging verification passes without running the full test suite.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/032-immutable-installed-distribution-assets.md`

Reason: ADR-032 already defines the installed distribution and artifact-content boundary. This task aligns release metadata, docs, and CI publishing mechanics with that accepted boundary; it does not introduce a new package format, runtime owner, storage model, or dependency boundary.

1. Derive native packaging version metadata from `pyproject.toml` so PyPI and native packaging cannot drift silently.
2. Update PyPI release docs/scripts to use fresh artifact output, the existing `Packaging/check_manifest.py` gate, real `tldw-cli`/`tldw-serve` smoke commands, and trusted publishing as the default release path.
3. Add a scoped GitHub Actions publish workflow with a build/check job and separate least-privileged trusted-publishing jobs for TestPyPI and PyPI.
4. Add or update focused packaging tests for version consistency if existing coverage does not pin it.
5. Run targeted packaging verification, `git diff --check`, complete task notes, and mark acceptance criteria only after fresh evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepared the PyPI publishing path around trusted publishing and the existing
installed-distribution contract from ADR-032.

- Made `Packaging/common/version.py` derive `VERSION` from `pyproject.toml`
  and added a focused metadata test for `pyproject.toml`,
  `tldw_chatbook/__init__.py`, packaging helper metadata, and the real
  `tldw-cli`/`tldw-serve` scripts.
- Added `.github/workflows/publish-pypi.yml` with a build/check artifact job,
  manual TestPyPI publishing through the `testpypi` environment, and PyPI
  publishing from protected `v*` tags through the `pypi` environment. The build
  job fails if artifact versions drift from `pyproject.toml` or a tag does not
  match `v<project-version>`. Only publish jobs receive `id-token: write`.
- Reworked `Packaging/PYPI_RELEASE.md`, `Packaging/PYPI_README.md`,
  `Packaging/build_dist.sh`, `Packaging/build_release.sh`, and
  `Packaging/PACKAGING_CHECKLIST.md` around fresh artifacts, trusted
  publishing, real smoke commands, and explicit release-tool prerequisites.
- Tightened native packaging release version handling: macOS DMG packaging now
  reads the root `pyproject.toml` without importing release helpers, and the
  Windows NSIS installer fails unless `Packaging/windows/build_windows.py`
  provides `PRODUCT_VERSION`.
- Kept the current `dev` migration packaging hardening: ChaChaNotes SQL
  migrations ship via `migrations/*.sql`, while `Packaging/check_manifest.py`
  and the installed-distribution regression derive the required migration set
  from source/runtime reads instead of hand-maintained lists.
- Reused the existing `backlog/docs/lessons-testing-evidence.md` packaging
  lesson that now covers the TASK-21506 migration-package-data incident.

Verification:

- Red metadata test first failed on stale `Packaging.common.version.VERSION`
  (`0.1.6.2` vs `0.1.8.0`); after the fix,
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Packaging/test_release_metadata.py -q`
  passed on the clean `dev`-based branch: 2 passed, 1 environment warning in
  1.08s.
- `PYTHON=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Packaging/build_dist.sh`
  passed on the clean `dev`-based branch, building
  `tldw_chatbook-0.1.8.0.tar.gz` and
  `tldw_chatbook-0.1.8.0-py3-none-any.whl`; `twine check` passed for both
  artifacts and `Packaging/check_manifest.py` validated both artifacts.
- Installed-distribution regression in an outside-checkout Python 3.12 runner
  first failed on the missing v40-to-v41 migration; after rebasing onto current
  `dev`'s generalized migration packaging and artifact-derived checks,
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py -m integration -q -p no:cacheprovider`
  passed on the clean `dev`-based branch: 163 passed, 1 environment warning in
  558.64s after rebasing onto current `dev`.
- `DIST_DIR=/private/tmp/tldw-should-refuse Packaging/build_dist.sh` failed
  closed with `Refusing DIST_DIR outside repository root`, confirming the
  cleanup script no longer deletes arbitrary external artifact directories.
- `bash -n Packaging/build_dist.sh`, `bash -n Packaging/build_release.sh`,
  `bash -n Packaging/macos/scripts/package_dmg.sh`, workflow YAML parse,
  host-Python macOS version extraction, and targeted `git diff --check` passed.

Full test suite was not run, per repo guidance to use targeted verification
unless a full sweep is explicitly requested.
<!-- SECTION:NOTES:END -->
