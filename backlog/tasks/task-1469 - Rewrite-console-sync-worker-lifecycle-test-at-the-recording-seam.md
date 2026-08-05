---
id: TASK-1469
title: >-
  Rewrite test_console_sync_records_worker_lifecycle at the worker-recording seam (xfail-quarantined by task-1457)
status: Done
assignee: []
created_date: '2026-07-30 19:20'
labels:
  - testing
  - ui
priority: medium
dependencies: [task-1457]
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_ui_responsiveness.py::test_console_sync_records_worker_lifecycle` was dormant from the repo root (pytest-asyncio strict mode silently skipped its unmarked coroutine) until task-1457 unified the config. Once it ran, it failed: it drives `ChatScreen._sync_native_console_chat_ui` through a hand-built `ChatScreen.__new__` skeleton that stubs 12 delegation stages, but the production method now has ~25 — each newly-grown stage breaks the skeleton with another missing attribute or stale stub signature (three layers were peeled during triage before capping). It is quarantined `xfail(strict=False)`. The test's actual subject — the responsiveness monitor records worker start/finish around the core-state sync — deserves a test that doesn't re-rot on every delegation change: assert at the recording seam (e.g. monkeypatch only `_record_ui_worker_started/_finished`'s caller boundary or run the real screen via the app harness) instead of stubbing the whole delegation graph.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

- [x] The worker-lifecycle recording behavior is covered by a test that passes on current dev
- [x] The test does not enumerate `_sync_native_console_chat_ui`'s delegation stages (adding a stage must not break it)
- [x] The `xfail` quarantine on the old test is removed (rewritten or deleted with the coverage moved)

## Implementation Plan

1. Study the recording bracket (`_record_ui_worker_started/finished` around the stage pipeline in `_sync_native_console_chat_ui`)
2. Replace stage enumeration with `MagicMock(spec=ChatScreen)` (auto-stubs every current stage, async defs become AsyncMocks, tracks the class as it evolves); bind the REAL recording helpers + a real monitor
3. Cover the bracket's three behaviors: active-during-stages, finished-even-on-stage-failure, re-entry-defers-without-recording
4. Mutation-check: break the started-recording and confirm the assertion fails

## Implementation Notes

By implementation time a foreign train had already removed the xfail by
RE-ENUMERATING the stages (15 hand-stubs, updated to the current pipeline) —
green today, and back on the same rot treadmill this task exists to end.
Replaced with a spec-mock probe (`_make_sync_probe_screen`): only one
deliberate anchor remains (`_sync_console_chat_core_state`, the semantic core,
used as the mid-flight sampling point), so adding a stage cannot break these
tests. Added the two behaviors the old test never covered: a raising stage
must not leak an active-worker record (the try/finally), and re-entry must
defer without recording. Mutation-verified: with the started-recording broken,
the mid-flight sample reads 0 and the test fails. File: 14 passed.
Modified: `Tests/UI/test_ui_responsiveness.py`.
