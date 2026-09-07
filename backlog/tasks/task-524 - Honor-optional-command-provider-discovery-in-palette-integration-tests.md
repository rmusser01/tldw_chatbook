---
id: TASK-524
title: Honor optional command-provider discovery in palette integration tests
status: Done
assignee: []
created_date: '2026-07-24 18:55'
updated_date: '2026-07-24 19:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the aggregate command-palette provider test with Textual's contract: providers may inherit the default discover implementation, which yields NotImplemented while still supporting search.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Providers that override discover return only Hit objects
- [x] #2 Providers that inherit Textual's optional discover implementation are accepted only with the NotImplemented sentinel
- [x] #3 The full command-palette provider module passes
- [x] #4 The merge-base failure and no-ADR decision are documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the aggregate discovery failure on the feature branch and merge base.
2. Distinguish providers that override discover from providers using Textual's permitted default implementation.
3. Assert Hit objects for implemented discovery and the exact NotImplemented sentinel for inherited discovery.
4. Run the full provider module, Ruff, diff checks, and review before completion.

ADR required: no
ADR path: N/A
Reason: This corrects a test assumption to match the installed framework contract; production provider behavior and application architecture remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned the aggregate discovery test with Textual Provider.discover: providers overriding discover must yield non-empty Hit objects, while providers inheriting the framework default must yield exactly the NotImplemented sentinel. Renamed the test and docstring to describe that two-branch contract instead of preserving the stale all-Hits assumption.

The feature branch and merge base both fail because ImageGenCommandProvider validly inherits Textual optional discovery. Verification: the full provider module passes (60 passed); the provider + latest-dev smoke batch passes 63 tests; Ruff and diff checks pass. test_command_palette_providers.py retains unrelated pre-existing whole-file format debt that is identical on the merge base. Independent re-review approved with no findings.

ADR required: no. This corrects a test assumption to the installed framework contract and changes no production provider behavior or architecture.
<!-- SECTION:NOTES:END -->
