---
id: TASK-41953
title: "ChaChaNotes index census misses five indexes from v81..v86, and v81->v82 dropped UNIQUE from idx_canvas_revisions_canvas_sequence"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- db
- migrations
priority: medium
references:
- scripts/index_plan_pin_census.tsv
- tldw_chatbook/DB/ChaChaNotes_DB.py
dependencies:
- TASK-41653
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41653 found five indexes created in v81..v86 with no plan pin, and the v81->v82 migration dropped `UNIQUE` from `idx_canvas_revisions_canvas_sequence`. Whether that drop was intended is not recorded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each of the five indexes has a plan-pin row captured with sqlite_stat1 absent
- [ ] #2 The UNIQUE drop is confirmed intended (recorded) or restored by a migration with a test
<!-- AC:END -->
