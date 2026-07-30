---
id: TASK-1454
title: >-
  Narrow the per-test double gc.collect() autouse fixture to periodic/marked collection and add an fd-leak sentinel
status: Done
assignee: []
created_date: '2026-07-30 09:05'
labels:
  - testing
  - performance
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/conftest.py`'s autouse `cleanup_file_descriptors` ran **two full `gc.collect()` passes after every test** — ~23,000 full-heap collections per ~11,600-test run, in a process whose heap includes Textual and (in CI) torch/transformers/chromadb. The 2026-07-30 audit ranks this the #3 wall-clock driver (est. 10–40 min/run). The fixture exists because of real FD-leak incidents (loguru handlers; fds under Textual's redirected streams), so it must be narrowed with a replacement leak-detection story, not deleted: the targeted `cleanup_loguru_handlers` fixture stays untouched, collection becomes periodic (every N tests) or per-test via the already-registered-but-unused `requires_cleanup` marker, and a session-scoped fd-count sentinel warns on suspicious growth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] gc.collect() runs every N tests (`TLDW_TEST_GC_EVERY`, default 25) or immediately after tests marked `requires_cleanup`; `TLDW_TEST_GC_EVERY=1` restores per-test collection
- [x] `cleanup_loguru_handlers` is unchanged
- [x] A session-scoped fd sentinel warns (warn-only initially) when open-fd count grows past a configurable threshold, with actionable guidance in the message
- [x] Full-suite run completes without fd exhaustion; junit outcome diff vs baseline shows no regressions

## Implementation Plan

1. Replace the double per-test collect with a counter-gated single collect (ResourceWarnings suppressed during the pass) + `requires_cleanup` marker path
2. Add `fd_leak_sentinel` (session, autouse): count `/dev/fd` (darwin) / `/proc/self/fd` entries at session start/end, warn past `TLDW_TEST_FD_GROWTH_LIMIT` (default 200)
3. Verify: full run, junit diff, no ResourceWarning storms

## Implementation Notes

Counter-gated single `gc.collect()` (the second bare collect added nothing the
warnings-suppressed one doesn't do), `requires_cleanup` marker honored per test,
env escape hatch for CI conservatism, and the fd sentinel as the leak-detection
replacement — it produces an actionable signal (bisect with TLDW_TEST_GC_EVERY=1,
mark offenders) instead of silently papering over leaks per test.
Modified: `Tests/conftest.py`.
