---
id: TASK-34929
title: Gate-Watchlists-per-visit-section-loads-on-staleness
status: To Do
assignee: []
created_date: '2026-10-13 14:52'
labels:
  - performance
  - ui-perf
  - wave-3-screen-switch
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Watchlists on_mount fires five loaders on every visit — _refresh_local_wc_snapshot, _load_active_section_data, _load_tree_data, _refresh_overview_data, _resolve_daily_report_banner (watchlists_collections_screen.py:1349-1361). They run off-pump in workers so nothing freezes, but every visit re-queries all sections with no unchanged-since-last-visit gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Re-entering Watchlists with unchanged data skips redundant section re-queries,Forced refresh action still reloads everything,Overview and tree stay correct after data changes (tests)
<!-- AC:END -->
