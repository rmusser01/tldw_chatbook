---
id: TASK-903
title: Verify installed distributions and immutable packaged assets
status: Done
assignee:
  - '@codex'
created_date: '2026-07-24 17:40'
updated_date: '2026-07-24 20:13'
labels:
  - packaging
  - reliability
  - security
dependencies: []
references:
  - backlog/decisions/032-immutable-installed-distribution-assets.md
documentation:
  - Docs/superpowers/specs/2026-07-24-installed-distribution-integrity-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prove the built distributions outside the source checkout, include every runtime-owned asset and vendored license, and prevent installed commands from rebuilding generated package files at runtime.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exactly one fresh sdist and wheel pass automated required-content and forbidden-content verification from an isolated temporary source and build tree
- [x] #2 The wheel contains the compiled CSS bundle, RAG pipeline config, exactly the built-in chunking JSON templates, eval configuration, and vendored Apache and MIT license files while sdist-only, example (including namespace-discovered Python), test, cache, and OS files remain excluded
- [x] #3 An isolated wheel installation resolves tldw_chatbook from the installed target and runtime loaders consume the packaged configs and templates
- [x] #4 Installed tldw-cli --help and tldw-serve --help exit successfully with private state contained under a temporary root
- [x] #5 Installed entry points and the application factory leave the complete installed-target file inventory and content hashes unchanged and do not attempt CSS rebuilds
- [x] #6 Regression coverage runs through the integration test gate without relying on source-checkout imports or network dependency resolution
- [x] #7 Built metadata uses the current AGPL-3.0-or-later SPDX license expression, declares LICENSE, and avoids the legacy license-table deprecation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes

ADR path: `backlog/decisions/032-immutable-installed-distribution-assets.md`

Reason: ADR-032 already defines the explicit distribution-content and installed-runtime immutability boundary implemented by this task.

Full plan: `Docs/superpowers/plans/2026-07-24-installed-distribution-integrity.md`

1. Add a fresh-tree artifact regression, then make the root manifest, package discovery, package data, test dependencies, and SPDX license metadata explicit.
2. Make `Packaging/check_manifest.py` enforce the same required, forbidden, entry-point, and license contract against exactly one new sdist and wheel.
3. Install the wheel outside the checkout, exercise packaged loaders and installed entry points under private state, hash the complete target, and guard all three CSS bootstrap sites with one source-tree predicate.
4. Update the packaging checklist, run focused and cross-task sentinels, build from committed source, and reconcile TASK-903 only after every acceptance criterion has fresh evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the explicit distribution and installed-runtime immutability
boundary defined by
[ADR-032](../decisions/032-immutable-installed-distribution-assets.md).

- Moved the canonical sdist manifest to the repository root, made wheel data
  ownership and namespace exclusions explicit, raised the setuptools floor,
  and emitted current SPDX/Core Metadata 2.4 license metadata.
- Replaced the warning-oriented manifest script with an executable checker for
  exactly one sdist and wheel, required/forbidden content, exact templates,
  entry points, project/vendored licenses, and parsed metadata.
- Added a temporary-source build and isolated `pip --no-deps --target`
  regression that loads packaged RAG/chunking/eval resources, runs
  `tldw-cli --help` and `tldw-serve --help` under private state, and compares
  complete installed-target hashes before and after startup.
- Kept the existing source-checkout CSS freshness behavior while guarding
  direct module execution, `get_app()`, and `main_cli_runner()` with one
  adjacent-`pyproject.toml` source-tree predicate.
- Updated `MANIFEST.in`, `pyproject.toml`, `requirements-test.txt`,
  `Packaging/check_manifest.py`, `Packaging/PACKAGING_CHECKLIST.md`,
  `tldw_chatbook/app.py`, and focused packaging/CI/startup tests. No packaging
  framework or application-state decomposition was added.

Verification:

- Focused packaging/startup gate: 32 passed, 1 existing dependency warning.
- Installed-distribution integration gate: 6 passed.
- Eval sentinel: 321 passed, 13 skipped, 3 existing warnings.
- Tool/agent sentinel: 173 passed, 1 existing dependency warning.
- Ruff, Python compilation, and `git diff --check`: passed.
- A `git archive HEAD` build under `/private/tmp` produced exactly one sdist
  and wheel; `Packaging/check_manifest.py dist` accepted both.
<!-- SECTION:NOTES:END -->
