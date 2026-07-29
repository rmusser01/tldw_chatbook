---
id: TASK-1360
title: Decide the fate of baseline_manager, a 697-line change detector nothing imports
status: To Do
assignee: []
created_date: '2026-07-29 23:10'
labels:
  - watchlists
  - dead-code
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`tldw_chatbook/Subscriptions/baseline_manager.py` is 697 lines and 20 methods of change-detection
machinery with **no importers anywhere in the repo and no tests**. The only reference to it is a
comment in `content_pane.py`. Verified: a grep for `baseline_manager`/`BaselineManager` across every
`.py` under `tldw_chatbook/` returns only that comment, and an import statement must name the module.

It matters because it is **better than the code actually in use**. `ChangeReport` carries a
`change_type` vocabulary of `'content'/'structural'/'semantic'/'new'/'removed'` — the spec's site
mockup shows `structural` — plus a `summary` field that is precisely what the `diff_summary` column
wants, and it compares key elements structurally rather than only by text ratio.

The live path (`monitoring_engine.py`) instead computes `change_percentage` from
`difflib.SequenceMatcher` and hardcodes `change_type: "content"`. TASK-1343 extends that live path
rather than adopting this module, deliberately, because pulling 697 untested lines into the fetch
path is its own risk and deserves its own decision.

This is the sixth instance in this codebase of a construct that is built, wired and carrying
nothing while reading as live to a grep — see TASK-1220 (`content_processor` orphaned) and
TASK-1221 (Watchlists gated on three packages nothing imports).

Note also: a comment at `content_pane.py:160-161` cites "`baseline_manager.py`/`monitoring_engine.py`"
as the producers of `change_percentage`. Half of that is false and is corrected by TASK-1343.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded on whether baseline_manager is adopted into the live change-detection path, kept as a documented future component, or deleted
- [ ] #2 If adopted, it has test coverage before it enters the fetch path, and monitoring_engine's duplicate detection is retired rather than left as a second implementation
- [ ] #3 If deleted, the change_type vocabulary and structural comparison worth keeping are named in the task notes so the capability is not silently lost
- [ ] #4 A liveness check is run from both ends (what imports it, and what it imports) rather than a single outward grep
<!-- AC:END -->
