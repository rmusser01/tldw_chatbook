---
id: TASK-16202
title: Align current-schema tests with ChaChaNotes v37
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 06:04'
updated_date: '2026-08-14 06:11'
labels:
  - database
  - migrations
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align current-schema evidence after provider-continuation migration V37 shipped, including the wheel/sdist ownership needed for an installed V36 database to reach V37, while preserving deliberate historical V36 fixtures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current-schema assertions follow V37 in source and installed-distribution coverage.
- [x] #2 Historical V36 migration fixtures remain frozen and unchanged.
- [x] #3 Wheel, sdist, and manifest ownership include the V36-to-V37 migration SQL, and an installed wheel upgrades a V35 database through V37.
- [x] #4 Affected database, packaging, static, diff, and mutation gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: tests and package-data ownership are aligned to the already-shipped V37 migration; no storage or migration-policy change.

1. Inventory exact V36 literals and separate current-version claims from deliberate historical V36 fixtures.
2. Replace only current-version claims with the live schema owner or V37, and add the already-shipped migration SQL to the explicit wheel/sdist manifests.
3. Prove restoring V36 or omitting the migration artifact fails, then run the affected database and installed-package/static gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Aligned current-schema tests and distribution ownership with shipped ChaChaNotes V37. Current-version assertions now follow V37/live schema ownership while deliberate V36 fixtures remain unchanged. Added the V36→V37 provider-continuation migration SQL to wheel and sdist package data and completed the release manifest checker's previously incomplete V32→V37 migration inventory. RED evidence: the V34 backfill asserted 36 against live 37; the installed-wheel probe failed because the V36→V37 SQL was absent; the full packaging file then exposed eight fail-open manifest checks for V32→V36 artifacts. GREEN: 58 adjacent dictionary/migration tests and 23 full installed-distribution tests passed; Ruff check/format and diff-check passed. ADR required: no; this packages the existing migration without changing storage policy.
<!-- SECTION:NOTES:END -->
