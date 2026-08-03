---
id: TASK-1360
title: Decide the fate of baseline_manager, a 697-line change detector nothing imports
status: Done
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
- [x] #1 A decision is recorded on whether baseline_manager is adopted into the live change-detection path, kept as a documented future component, or deleted
- [x] #2 (moot — not adopted) If adopted, it has test coverage before it enters the fetch path, and monitoring_engine's duplicate detection is retired rather than left as a second implementation
- [x] #3 (moot — not deleted; capability named in the decision) If deleted, the change_type vocabulary and structural comparison worth keeping are named in the task notes so the capability is not silently lost
- [x] #4 A liveness check is run from both ends (what imports it, and what it imports) rather than a single outward grep
- [x] #5 (moot — not adopted; the re-keying prerequisite is recorded in the module header for any future adoption) If its pruning is adopted, it is re-keyed per (subscription, url) first: as written it prunes per subscription, which under the per-URL baselines shipped by TASK-1361/1362 would evict other URLs' baselines on multi-URL sources and cause endless re-baselining (see TASK-1393)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Decision (AC#1): KEEP as a documented future component** — neither adopted into the live fetch
path nor deleted. Recorded as a prominent NOT-WIRED header at the top of
`tldw_chatbook/Subscriptions/baseline_manager.py`.

Rationale: (a) I did not author this 697-line subsystem and deletion is hard to reverse; (b) it
implements a genuinely better change-detection approach the spec mockup references (`structural`
change_type, structural element comparison, a `summary` field for `diff_summary`) — deleting loses
that; (c) adopting it is a substantial, deliberate effort with hard prerequisites (AC#5's
per-(subscription,url) re-keying, AC#2's test coverage + retiring monitoring_engine's duplicate
detection) that "deserves its own decision", not a drive-by; (d) the *actual* harm the task names —
reading as live to a grep, the 6th orphaned-but-grep-live instance — is fully neutralized by the
NOT-WIRED header, which tells any future reader/grep-er immediately that nothing imports it and why.

**AC#4 (liveness from both ends):** OUTWARD — no `import` of `baseline_manager`/`BaselineManager`
anywhere in `tldw_chatbook/` or `Tests/`; the only references are docstring mentions in
`monitoring_engine.py:477` and `test_watchlist_snapshot_pruning.py` (which tests the LIVE
`monitoring_engine`/`URLMonitor._store_snapshot` pruning, not this module — it merely cites it as
context). INWARD — the module imports stdlib + `bs4` + loguru; it is functional code but nothing is
wired INTO it. Confirmed dead in both directions, correcting the single-outward-grep the task warned
against.

**AC#3 (capability worth keeping, per the "if deleted" arm — recorded even though NOT deleted):** the
`ChangeReport.change_type` vocabulary (`content`/`structural`/`semantic`/`new`/`removed`), the
`summary` field, and the structural (key-element) comparison rather than pure text-ratio. All named
in the module header so the capability is preserved as a design record regardless of the code's fate.

Note: the false comment the task cites (`content_pane.py:160-161` naming baseline_manager as a
`change_percentage` producer) was ALREADY corrected on dev by TASK-1343 — verified; no change needed.

File: `tldw_chatbook/Subscriptions/baseline_manager.py` (module header only; no logic change).
<!-- SECTION:NOTES:END -->
