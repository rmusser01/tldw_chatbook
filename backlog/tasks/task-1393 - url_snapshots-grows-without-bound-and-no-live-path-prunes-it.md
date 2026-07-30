---
id: TASK-1393
title: url_snapshots grows without bound, and no live path prunes it
status: To Do
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
- [ ] #1 Snapshots are pruned on a live path, keyed per (subscription, url), keeping at least the newest N per URL so every URL always retains its baseline
- [ ] #2 A multi-URL source cycling through checks never loses another URL's baseline to pruning, pinned by a test that fails under per-subscription pruning
- [ ] #3 The reader's [previous snapshot] affordance still works after pruning (the second-newest per URL survives, or the affordance degrades honestly)
<!-- AC:END -->
