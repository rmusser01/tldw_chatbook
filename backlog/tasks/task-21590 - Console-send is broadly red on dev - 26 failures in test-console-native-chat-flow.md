---
id: TASK-21590
title: >-
  Console send is broadly red on dev   26 failures in test console native chat flow
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - testing
  - dev-red
  - console
priority: high
---
## Description

`Tests/UI/test_console_native_chat_flow.py` fails 26 tests on pristine dev. The failures span
generic-provider send, the retry/regenerate/continue family, and the first-send-flag pair — the
core Console send path. This is either a stale harness or a real break in sending, and until
someone determines which, every Console branch inherits a red baseline that hides regressions in
exactly the app's most-used flow.

A scratch probe on pristine dev shows the draft is **neither sent nor cleared after Enter**, and
a single-line control fails the same way — so it is not shift+enter specific.

## Acceptance Criteria

- [ ] A determination is recorded, with evidence, of whether Console send is genuinely broken in the shipped app or only in the test harness
- [ ] If the app is broken, the send path is fixed and a test pins the behaviour that regressed
- [ ] If the harness is stale, the harness is repaired so the tests exercise the real send path again — not deleted, and not relaxed until they pass
- [ ] `Tests/UI/test_console_native_chat_flow.py` is green on dev
- [ ] The fix is verified by mutation: breaking send makes these tests fail again

## Evidence (verified first-hand on dev 33ff5b754, 2026-08-23)

```
pytest Tests/UI/test_console_native_chat_flow.py -q -p no:randomly
  -> 26 failed, 271 passed  (7m 33s)
```

Surfaced by the TASK-21501/21123 implementer, which classified rather than waved through the
composer-suite reds it inherited: 9 of its 12 composer-suite failures trace to this same root
cause. Independently reproduced here before filing.
