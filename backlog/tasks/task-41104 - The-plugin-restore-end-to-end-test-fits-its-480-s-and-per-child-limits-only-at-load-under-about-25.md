---
id: TASK-41104
title: "The plugin restore end-to-end test fits its 480 s and per-child limits only at load under about 25"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- tests
- flaky
priority: low
references:
- Tests/Plugins/test_recovery.py
- .github/workflows/plugin-qualification.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** `test_plugin_backup_restores_in_place_and_waits_for_authority_recovery` is in the Managed plugin qualification workflow (`plugin-qualification.yml:131`, a PR path filter on `tldw_chatbook/Plugins/**` and the backup files). A slow runner fails it on time, not on behaviour.

**Evidence.** `test_recovery.py:1843` sets `@pytest.mark.timeout(480)`; the restore children get 360 s (`:1872`) and 300 s (`:1879`). The comment above it (`:1840-1842`) records 4.0 min at load 15-20 and 5.4-6.9 min at load 25-40. TASK-40308's notes: at load 30-50 its restore children exceed the limits; with relaxed limits it passed in 777 s at load about 40 (children 401 s and 348 s); with the committed limits it passed in 326 s at load 19-41.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test's time limits come from measured CI times of its children (recorded with run ids), with at least 2x headroom
- [ ] #2 Or the slow part (the two Tools-group restores, mostly journal revalidation per the comment) is made faster, with before/after times recorded
- [ ] #3 The test passes 3 of 3 locally at load 40 or more
<!-- AC:END -->
