---
id: TASK-40308
title: "After a restore, plugins whose data roots or authority changed stay off with no way back in the app: add an authority reset that keeps and re-attaches data, and a reviewed rebind of restored roots"
status: To Do
created_date: 2026-11-12 19:36
labels:
- follow-up
- plugins
- backup
- recovery
- feature
priority: medium
references:
- tldw_chatbook/Plugins/coordinator.py
- tldw_chatbook/Plugins/data_cleanup.py
- tldw_chatbook/Plugins/acquisition.py
- Docs/Plugins/operations.md
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (plugins stream, TASK-40132; owner decision D2, recorded there: "Truthful copy now, feature next"). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** After a restore where plugins changed, or where their data roots were recreated, the plugins stay off with their data on disk, and the app offers no way out. TASK-40132 shipped the truthful copy; this is the feature the owner asked for next.

**Evidence.**
- Reset keeps data roots but never re-attaches them: `Plugins/coordinator.py:131` (`root_reset_retained_custody`). `Docs/Plugins/operations.md:125-126`: "Resetting plugin authority does not help... setup refuses while the restored registry still lists installed plugins."
- A recreated root or fetch record fails identity checks: `Plugins/data_cleanup.py:209`, `:213`, `:884` (`root_anchor_changed`), `:232` (`root_incarnation_changed`); `acquisition.py:131` `_qualify` fails recreated custody. `operations.md:139-144`: "Chatbook has no way to rebind restored folders yet". TASK-40132 lists it under Known limits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 From the Plugins screen, a user recovers previously installed plugins after such a restore, with their saved data re-attached; an end-to-end test covers it
- [ ] #2 A restored data root or fetch record with a new identity can be rebound after a review step, and the plugin runs again
- [ ] #3 Docs/Plugins/operations.md describes the recovery instead of the current limit
<!-- AC:END -->
