---
id: TASK-1469
title: >-
  Rewrite test_console_sync_records_worker_lifecycle at the worker-recording seam (xfail-quarantined by task-1457)
status: To Do
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

- [ ] The worker-lifecycle recording behavior is covered by a test that passes on current dev
- [ ] The test does not enumerate `_sync_native_console_chat_ui`'s delegation stages (adding a stage must not break it)
- [ ] The `xfail` quarantine on the old test is removed (rewritten or deleted with the coverage moved)
