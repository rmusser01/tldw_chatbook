---
id: TASK-21235
title: >-
  Two gaps in the TASK-21124 debounced config-write path
status: Done
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

## Implementation Plan

1. Gap 1: replace the Logs scheduling gate's baseline comparison with the
   dictation module's task-15470 dirty-flag shape (arm unconditionally;
   clear at worker snapshot time so mid-flight edits re-dirty); keep the
   worker/unmount baseline comparisons so click-away-and-back still writes
   nothing and filter-TEXT edits still persist at unmount.
2. Gap 2: thread the caller's `config_path` through
   `_publish_runtime_config_unlocked` into `_install_bootstrap_cache_from_raw`
   so the cache is published under the path its data was read from; the
   recompute stays only as a fallback for path-less callers.
3. Tests: an interleaving test (blocked first write + revert mid-flight,
   asserting the revert persists with NO unmount; mutation-verified red
   against the old gate) and a path-flip test on the cache install.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Gap 1** (`UI/Logs_Window.py`): `_persist_filter_state` arms
  `_filter_state_dirty` unconditionally; `_persist_filter_state_off_loop`
  clears it at snapshot time (an edit landing mid-write re-dirties and gets
  its own timer -- the dictation "edit 2 LOST" reasoning);
  `on_unmount` re-snapshots after awaiting any in-flight worker and keeps
  the baseline write-gate (so text-only edits, which never arm the
  debounce, still flush; no-op teardown still writes nothing).
- **Gap 2** (`config.py`): `_install_bootstrap_cache_from_raw(raw, config_path)`
  and `_publish_runtime_config_unlocked(raw_config, config_path)`; all 8
  write-path call sites now pass the `config_path` their read/write used.
  A `TLDW_CONFIG_PATH` flip between read and publish can no longer label
  path-A data as the path-B `_CONFIG_CACHE_SOURCE`.
- **Tests**: `Tests/UI/test_logs_filter_persist_debounce.py::
  test_revert_during_in_flight_write_is_persisted_without_unmount`
  (private-profile wrapped -- the file's harness is locally-red under the
  shared-profile sandbox, verified at HEAD; mutation check: restoring the
  old stale-baseline gate fails it with "swallowed against the stale
  baseline"); `Tests/test_config_read_fastpath_task21124.py` gains the
  path-flip test and a publish-threading pin (11/11 pass, including
  TASK-21124's 0-lock-acquisitions warm-read guard -- AC5 holds).
