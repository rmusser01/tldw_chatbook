---
id: TASK-15270
title: >-
  Console test apps mount with a config that silently defaults every turn-context setting
status: To Do
assignee: []
created_date: '2026-08-11 09:00'
labels:
  - tests
  - console
  - test-infrastructure
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while triaging task-15210, and raised rather than absorbed because the blast radius (90+ modules) is far larger than that task's scope.

`_build_test_app` patches `load_settings` to a small synthetic dict that carries no `[console]` or `[chat_defaults]` section, and production *correctly* refuses to refresh a settings snapshot that lacks the disk-load markers. So every mounted Console test sees a `ConsoleTurnExecutionContext` whose `rag_defaults` are frozen at their defaults, no matter what the test believes it configured. Measured directly during 15210: `get_cli_setting=True` while the app_config key was `MISSING` and the context read `auto_retrieve_on_send: False`.

The live app is unaffected — `app.py` does `self.app_config = load_settings()`, whose result carries both the toggle and the markers (verified) — so this is a test-harness defect, not a product one.

**Why it is worth its own task: it can hollow out a passing test.** `test_send_proceeds_when_auto_retrieve_fails` was green **vacuously** for exactly this reason — auto-retrieval never fired, so the deliberately-exploding backend was never called, and the test only ever asserted that an ordinary send works. It was repaired in 15210 (`exploding_search.await_count == 1`), but any other test whose subject reads through the turn-context snapshot has the same exposure and would look green while asserting nothing.

The fix is presumably to make `_build_test_app` produce a config the snapshot will accept (markers included), so a test that sets a `[chat_defaults]` value gets that value. That change will likely flip some currently-green tests to red — those are the ones that were never really testing their subject, and each needs the 15210 treatment rather than a revert.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A Console test that configures a `[chat_defaults]`/`[console]` setting sees that value through the turn-context snapshot, instead of a silent default
- [ ] #2 Tests that turn red once the config is honoured are triaged individually (real regression vs assertion that was never exercised), not reverted wholesale
- [ ] #3 A guard makes the vacuous-pass shape detectable: a test whose subject is "X still works when Y fails" asserts that Y was actually attempted
<!-- AC:END -->
