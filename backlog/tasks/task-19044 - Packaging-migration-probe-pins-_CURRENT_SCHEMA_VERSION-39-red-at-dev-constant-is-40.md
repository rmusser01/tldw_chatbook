---
id: TASK-19044
title: >-
  Packaging migration probe pins _CURRENT_SCHEMA_VERSION == 39 — red at dev
  (constant is 40)
status: To Do
assignee: []
created_date: '2026-08-20 08:40'
labels:
  - test-health
  - db
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16840's investigation flagged two schema-version equality pins that were
already red at the wave's base `cef56efaf` (constant 39, tests asserting 38):
`Tests/DB/test_chachanotes_trajectory_metadata_migration.py:44` and `:220`.
Those two were fixed on dev by commit `46945ebbe` (v39→v40,
`transcript_annotations`), which converted them to
`== CharactersRAGDB._CURRENT_SCHEMA_VERSION`.

But the same commit bumped the constant to 40 and left a third literal pin
behind: `Tests/Packaging/test_installed_distribution.py:807`
(`INSTALLED_MIGRATION_PROBE`) reads the constant into
`current_schema_version` and then asserts `current_schema_version == 39` —
red whenever the integration-marked packaging suite runs at dev `1bf7f234e`
(the constant is 40). History shows the literal being hand-bumped each time
(37→39 at `4a2d48046`) and missed on this bump — the exact
equality-pin-on-a-moving-constant fragility the trajectory fix just removed.
The probe's real job (monkeypatch to v35, reopen, prove the installed
migration chain reaches the current version) does not need the literal at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The installed-migration probe passes at current dev with no hand-maintained schema-version literal remaining in the file (owner ruling: durable over re-pinning the new number)
- [ ] #2 The probe still proves what the literal stood in for: a v35 database migrates to whatever `_CURRENT_SCHEMA_VERSION` is, under the installed distribution
- [ ] #3 The affected packaging test is demonstrated green at dev (run evidence, not collection)
<!-- AC:END -->
