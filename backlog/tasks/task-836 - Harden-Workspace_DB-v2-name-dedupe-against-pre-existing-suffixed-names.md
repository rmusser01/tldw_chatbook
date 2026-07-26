---
id: TASK-836
title: Harden Workspace_DB v2 name-dedupe against pre-existing suffixed names
status: Done
assignee: []
created_date: '2026-07-26 16:20'
labels:
  - workspaces
  - db
  - hardening
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The v2 migration's duplicate-name dedupe only collision-checks names it renames within the same pass, not pre-existing '(n)'-suffixed names already in the table. A contrived legacy DB holding both a real 'Foo (2)' workspace and a foo/Foo duplicate pair can have the migration rename into a still-colliding name, making CREATE UNIQUE INDEX raise and abort _initialize_schema for that DB. Seed the dedupe's seen-set with ALL existing non-archived names (not just renamed ones) and add a regression test for the contrived layout.

Source: workspace folder-roots train final review (spec 2026-07-26-settings-workspaces-category-design.md), deferred-minor triage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Migration succeeds on a DB containing both case-duplicates and pre-existing '(n)' names
- [x] #2 A regression test covers the contrived layout
- [x] #3 Index creation can no longer abort schema initialization for any dedupe outcome
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Absorbed into the PR-review fix wave on feat/workspace-folder-bindings (commit 97dc1ea43): the v2 dedupe now pre-seeds a reserved set with ALL non-archived casefolded names before any rename (covering cross-group collisions AND pre-existing '(n)' names), strips names before suffixing, and the whole migration runs inside the shared transaction() for atomic rollback. Regression tests: cross-group and pre-existing-suffix layouts in test_workspace_registry_service.py.
<!-- SECTION:NOTES:END -->
