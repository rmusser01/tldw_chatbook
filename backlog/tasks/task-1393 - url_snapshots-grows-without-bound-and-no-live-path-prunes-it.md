---
id: TASK-1393
title: url_snapshots grows without bound, and no live path prunes it
status: Done
assignee: []
created_date: '2026-07-30 05:20'
labels:
  - watchlists
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No live code path ever deletes from `url_snapshots` — verified during TASK-1362's whole-branch
review: the only `DELETE` in the repo is `baseline_manager.py:677-681`, and that module has zero
importers (TASK-1360). Every significant change stores a full row including `raw_html`.

TASK-1362 makes this newly load-bearing twice over: the default `change_threshold` of `0.0` means
every real change persists a snapshot (probed: 6 changes → 6 rows), and TASK-1361/1362's per-URL
baselines multiply storage by a source's URL count. Steady state is monotonic growth in the user's
private database.

**Constraint for the fix, established in the same review:** any pruning must be keyed
**per (subscription, url)**, never per subscription — the baseline SELECT is now
`WHERE subscription_id = ? AND url = ?` (`monitoring_engine.py`), so per-subscription pruning on a
multi-URL source would evict other URLs' baselines and cause endless re-baselining on rotation.
The orphaned `baseline_manager` pruning has exactly that defect; see the matching note in
TASK-1360.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Snapshots are pruned on a live path, keyed per (subscription, url), keeping at least the newest N per URL so every URL always retains its baseline
- [x] #2 A multi-URL source cycling through checks never loses another URL's baseline to pruning, pinned by a test that fails under per-subscription pruning
- [x] #3 The reader's [previous snapshot] affordance still works after pruning (the second-newest per URL survives, or the affordance degrades honestly)
<!-- AC:END -->

## Implementation Plan
<!-- SECTION:PLAN:BEGIN -->
1. Confirm `URLMonitor._store_snapshot` is the single live write chokepoint and that it already
   holds a transaction; confirm the shadow-mode guard sits before the INSERT.
2. Add `_SNAPSHOTS_KEPT_PER_URL = 3` with its rationale, and prune inside that same transaction
   immediately after the INSERT, keyed per `(subscription_id, url)` and selecting survivors by the
   same `ORDER BY created_at DESC, id DESC` the baseline SELECT uses.
3. New `Tests/Subscriptions/test_watchlist_snapshot_pruning.py` driving the real producer through
   the existing end-to-end harness, plus direct `_store_snapshot` tests.
4. Mutation-test the three load-bearing choices: per-subscription keying, N=1, inverted ordering.
5. Run `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/`.
<!-- SECTION:PLAN:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
**One DELETE, in the transaction that already existed.** `URLMonitor._store_snapshot`
(`Subscriptions/monitoring_engine.py`) is the single live write into `url_snapshots` and already
opens `db.transaction()`. The prune goes immediately after the INSERT, so the two commit together
and the table is never observably over the cap; it sits after the `persist_snapshots` guard, so a
shadow run neither writes nor deletes.

**Keyed per `(subscription_id, url)` on both halves of the statement** — the constraint this task
was filed with. Survivors are chosen by `ORDER BY created_at DESC, id DESC`, the *same* ordering
`check_url`'s baseline SELECT uses (TASK-1361's tie-break), which makes it an invariant rather than
a coincidence that the row the next check reads is the first survivor and can never be pruned. The
existing `idx_url_snapshots_lookup(subscription_id, url, created_at)` covers the subquery.

**N = 3** (`_SNAPSHOTS_KEPT_PER_URL`, rationale in the constant's comment): the live baseline; the
second-newest, which the design spec's Content-pane mockup promises to a `[previous snapshot]`
affordance that is **not built yet** (no reference anywhere in `UI/` — filed separately, and
pruning must not foreclose it); and one row of slack for the same-second `created_at` tie window.
Deliberately no config surface — YAGNI. `baseline_manager.retention_days` is orphaned code and was
left untouched (TASK-1360). Over-sized existing databases self-heal on the next write per URL.

**AC#2 needed the dispositions, not a row count.** The first draft asserted "the quiet URL ends
with exactly one row" — which *passes* under per-subscription pruning, because the evicted URL is
immediately re-baselined by its own next check. It still has a row; it just never reports a change
again. The test now asserts the full disposition-count dict of every run.

**Mutation testing** (all reverted): per-subscription keying -> RED, the quiet URL re-baselining
at run 3 (`{'baseline': 1, 'unchanged': 0}`); `N = 1` -> RED, 4 tests incl. AC#3; inverted survivor
`ORDER BY` -> RED, 5 tests, one of them reporting a phantom `changed` against the oldest kept text.

**Files:** `tldw_chatbook/Subscriptions/monitoring_engine.py`;
`Tests/Subscriptions/test_watchlist_snapshot_pruning.py` (new, 10 tests).
Suites: `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/` -> **601 passed** in 193.82s.
Report: `task-1393-report.md`.
<!-- SECTION:NOTES:END -->
