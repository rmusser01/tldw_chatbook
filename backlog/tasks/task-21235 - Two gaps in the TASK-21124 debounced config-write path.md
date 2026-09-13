---
id: TASK-21235
title: >-
  Two gaps in the TASK-21124 debounced config-write path
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - config
  - console
  - ux
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; two review findings
from TASK-21124 (PR #2005, `8e949873e`) — one confirmed, one suspected. Filed together because
they are the same seam: the write side of that change.

TASK-21124 removed the global config-file lock from the warm read path (100 warm reads: 100 →
**0** lock acquisitions; worst reader stall **18.2 → 3.7 ms**; write path 4 → 2 parses) and
debounced the Logs and dictation settings writes. Two gaps were left open in the write path:

1. **Confirmed.** `UI/Logs_Window.py` gates a persist on `snapshot ==
   self._persisted_filter_state` (`:351`, `:377`, `:401`), but `_persisted_filter_state` is
   only advanced when a debounced write completes (`:288`, `:332`). A Logs filter-chip click
   landing while a debounced write is in flight compares against a **stale** baseline, is
   judged a no-op, and is swallowed — the user's filter change is not persisted until unmount
   repairs it. The dictation `_settings_dirty` shape does not have this gap; the two should
   agree.
2. **Suspected.** `config.py:5339` `_install_bootstrap_cache_from_raw` recomputes the config
   path at publish time rather than using the path its caller read from. A `TLDW_CONFIG_PATH`
   flip between read and publish would label path-A data as path-B cache.

## Acceptance Criteria

- [ ] A Logs filter change made while a debounced write is in flight is persisted without waiting for unmount
- [ ] A test drives that interleaving and fails on the swallow
- [ ] The Logs and dictation debounce gates use one shape, or the divergence is documented in code with the reason it is safe
- [ ] The bootstrap cache is published under the path its data was read from, and a test with a config-path flip between read and publish fails if the cache is mislabelled
- [ ] TASK-21124's warm-read measurement (0 lock acquisitions across 100 warm reads) does not regress
