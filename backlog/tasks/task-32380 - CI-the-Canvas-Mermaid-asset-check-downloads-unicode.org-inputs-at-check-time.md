---
id: TASK-32380
title: 'CI: the Canvas Mermaid asset check downloads unicode.org inputs at check time'
status: Done
assignee:
- '@codex'
created_date: 2026-09-11 10:30
labels:
- ci
- infra
dependencies: []
priority: medium
updated_date: 2026-10-18 04:57
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`scripts/check_canvas_mermaid_assets.py` calls `acquire_declared_inputs()`, which downloads the Unicode UCD files and the mermaid tarball unless `--input-dir` is passed. Neither caller passes one: `.github/workflows/derived-artifacts.yml:137` and `scripts/preflight.sh:108` both invoke it bare. The required `Derived artifacts reproduce from their sources` check therefore fails whenever unicode.org is unreachable -- observed returning HTTP 522 and connection resets on 2026-09-11 between roughly 08:00 and 08:40 UTC, red-lighting PRs whose diffs touched nothing related. A derived-artifact check should depend only on the repository.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A run of `scripts/check_canvas_mermaid_assets.py` with no network access succeeds for an unchanged tree
- [x] #2 The CI workflow and `scripts/preflight.sh` obtain the declared inputs from a vendored or cached location rather than fetching them per run
- [x] #3 Refreshing the vendored inputs is a deliberate, documented step, and a stale-input condition is reported distinctly from a genuine artifact mismatch
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/238-offline-authenticated-canvas-build-inputs.md
Reason: make complete hash-pinned source inputs repository-owned build dependencies rather than a live network requirement; preserve immutable runtime identities.
1. Vendor the eight exact declared source inputs outside installed package data, retaining licenses and the unchanged input manifest.
2. Add a failing offline-default test and use the existing acquisition and builder validation to consume the repository inputs without fetching or skipping.
3. Provide an explicit fetch-only maintenance command; keep explicit offline overrides and missing/mismatched input failures.
4. Update build/preflight documentation, run real offline reproduction and tamper controls, affected CI tests and static checks; close both overlapping tasks with the same evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Vendored all eight original declared source inputs (19,636,664 bytes) outside application package data; their original manifest and runtime/profile bytes are unchanged. Normal checker, required CI and preflight now authenticate and reproduce offline. Explicit cache overrides retain precedence and fail closed; --fetch-inputs NEW_DIRECTORY is a separate validated maintenance action that refuses existing destinations and does not rebuild/repin. Offline-default and missing-command regressions failed before implementation. Final affected checker/workflow/membership tests: 137 passed, exit 0, including real reproduction with network access refused; actual standalone reproduction and all preflight checks pass. Independent review cleared the pin, cache/CLI, licensing and packaging boundaries. Build/test lint and format pass; pre-existing queue-contract formatting debt is unchanged. ADR-238 applies and is linked in build docs and vendor README. Resolves the same missing-input mechanism as TASK-32377. No full suite or fresh hosted CI claimed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
