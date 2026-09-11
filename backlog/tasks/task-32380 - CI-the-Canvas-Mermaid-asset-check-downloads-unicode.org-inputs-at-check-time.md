---
id: TASK-32380
title: 'CI: the Canvas Mermaid asset check downloads unicode.org inputs at check time'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - ci
  - infra
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`scripts/check_canvas_mermaid_assets.py` calls `acquire_declared_inputs()`, which downloads the Unicode UCD files and the mermaid tarball unless `--input-dir` is passed. Neither caller passes one: `.github/workflows/derived-artifacts.yml:137` and `scripts/preflight.sh:108` both invoke it bare. The required `Derived artifacts reproduce from their sources` check therefore fails whenever unicode.org is unreachable -- observed returning HTTP 522 and connection resets on 2026-09-11 between roughly 08:00 and 08:40 UTC, red-lighting PRs whose diffs touched nothing related. A derived-artifact check should depend only on the repository.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run of `scripts/check_canvas_mermaid_assets.py` with no network access succeeds for an unchanged tree
- [ ] #2 The CI workflow and `scripts/preflight.sh` obtain the declared inputs from a vendored or cached location rather than fetching them per run
- [ ] #3 Refreshing the vendored inputs is a deliberate, documented step, and a stale-input condition is reported distinctly from a genuine artifact mismatch
<!-- AC:END -->
