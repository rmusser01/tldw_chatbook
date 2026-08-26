---
id: TASK-21133
title: >-
  Retire the dead 10s token-count timer - its entire consumer surface was removed
status: Done
assignee:
  - '@claude'
created_date: '2026-08-22'
labels:
  - cleanup
  - performance
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21133).

The app-global 10 s interval (app.py:11746) resolves the active footer and, on the chat tab,
attempts four widget queries that ALL fail - `#chat-log` no longer exists anywhere, footers
compose `show_token_count=False` since task-17653, and the estimator result is only
debug-logged (chat_token_events.py:103-181; the file's own comments say the counter is
retired). The producer ticks forever for nothing.

## Acceptance Criteria

- [x] The interval, update_token_count_display, and the periodic path are deleted
- [x] ~~the estimator remains for on-demand callers (input-changed / model-changed)~~
      **AMENDED 2026-08-24 before implementing** (see Implementation Notes, "What the
      filing got wrong"): the two named on-demand callers,
      `chat_token_events.handle_chat_input_changed` and
      `handle_model_or_provider_changed`, have **zero** references anywhere in the
      repository. They are not wired to any event, so preserving the estimator "for"
      them would preserve dead code for dead callers. The whole module goes.
- [x] No footer or token-display regression - existing tests green
- [x] Nothing consumes what the timer produced: proved by inspection of every layer
      (reactive, widget, event, test) and by a live mounted-app measurement, before
      anything was deleted

## Implementation Plan

1. Verify the filing before deleting anything: enumerate every consumer of what the
   timer produces (reactive, widget, posted event, test, dynamic dispatch), not just
   grep for the producer's name.
2. Measure the base arm live -- a mounted app, isolated profile -- so "what the fire
   did" is a number, not a description.
3. Delete the interval, the one-shot, the handle, and every layer of the periodic
   path that the deletion makes unreachable.
4. Pin the retirement with tests at each layer it could come back, each
   mutation-checked.
5. Walk quit/unmount for dangling handles and unretrieved tasks.
6. Regenerate the production diagnostic inventory after reading every changed row.

## Implementation Notes

The 10 s timer is gone, and so is everything that existed only to feed it.

**What was verified before deleting (the filing is a hypothesis).** Every layer was
checked, not just the producer's name:

- `#chat-log` exists nowhere in the package. The only `chat-log` hits are a CSS *class*
  (`#console-session-surface .chat-log`), an unrelated comment, and a test fixture.
- `AppFooterStatus(show_token_count=...)` has exactly **one** construction site in the
  package, `UI/Navigation/base_app_screen.py:239`, and it passes `False`. So
  `update_token_count()` returns early everywhere -- there is no screen on which a
  write can reveal the chip. `Tests/UI/test_footer_token_counter_retired.py` already
  pinned that and still passes untouched.
- No reactive, no posted event, no widget render reads the estimate. The producer's
  last statement is a `logger.debug` f-string.
- `chat_token_events`'s only importer was `Utils/db_status_manager.py`, reached only
  from `TldwCli.update_token_count_display`, reached only from the two timers.
- The two functions the AC wanted preserved for "on-demand callers"
  (`handle_chat_input_changed`, `handle_model_or_provider_changed`) have **zero**
  references in the repo -- no `@on`, no registry, no string dispatch. With the timer
  gone, all four public functions and both private helpers in the module are
  unreachable, so the module and its two dedicated test files go with it.

**Measured, base arm (merge-base `ceb4196a4`, mounted ChatScreen, isolated profile,
`current_tab = chat`, 20 timed fires after one warm-up).** Per fire: **7**
`DOMNode.query_one` calls -- 1 hit (resolving the active screen's `AppFooterStatus`)
and 6 misses across 3 distinct selectors that no live screen composes
(`#chat-api-provider`, `#chat-log`, `#chat-custom-token-limit`), each raising
`NoMatches` after a subtree walk; then the estimator over the empty history the
missing `#chat-log` guarantees; then a `logger.debug` f-string that is built whether
or not DEBUG is enabled. Median **0.046 ms** per fire (min 0.045, max 0.065) on a fast
M-series with a small harness DOM -- the walk cost scales with real DOM size. First
fire **1.806 ms**. Chip state after 21 fires: `display=False`, renderable `''`.

**Removed per minute: 6 timer fires** (the 10 s interval), plus the one-shot at 0.5 s
after footer setup, plus the `call_after_refresh` callback each fire posted to the
message pump. After: the app arms no interval of its own in
`_schedule_footer_status_updates` at all (the 120 s DB-size interval belongs to
`DBStatusManager`).

**Checked and NOT claimed:** no import-closure win. The deleted module's two
module-scope imports (`Utils.token_counter`, `Widgets/Chat_Widgets/chat_message`) are
both already resident at footer-timer time on a real boot (measured: 1068 package
modules resident), so the 0.5 s one-shot was not dragging anything in.

**Quit / unmount walk.** `on_shutdown_request` and `on_unmount` both call
`db_status_manager.stop_periodic_updates()` then `_stop_footer_status_timers()`. The
latter no longer owns a handle, so there is nothing to leave dangling; it is now a
diagnostics clear only, idempotent, and every call inside it is exception-guarded. The
change *removes* a teardown hazard rather than adding one: the retired interval's
callback was `lambda: self.call_after_refresh(self.update_token_count_display)`, which
posted a coroutine to the message pump every 10 s including during teardown. The pin
suite calls the shutdown pair twice on a booted app and the app then exits its
`run_test` context cleanly.

**Tests.** New `Tests/Performance/test_footer_token_timer_retired.py` (4). Updated
`Tests/UI/test_ui_responsiveness.py` (3 tests: the doubles no longer supply
`update_token_count_display`, and their `set_interval` now *raises* so a resurrected
interval is a hard failure rather than an unread count). Deleted
`Tests/Chat/test_token_display_limit.py` (3) and
`Tests/Chat/test_footer_token_dirty_gate.py` (9) with their subject. Net -8 tests.

Mutation results, 4 deliberate defects, 4 caught:

| mutant | caught by |
|---|---|
| re-arm the 0.5 s one-shot + 10 s interval + `footer-token-periodic` diagnostic | 4 tests (2 new, 2 updated) |
| restore `TldwCli.update_token_count_display` | `test_no_token_count_display_entry_points_remain` |
| restore a `chat_token_events` module file | `test_periodic_token_producer_module_is_gone` |
| drop the `footer-db-size-periodic` stop from `_stop_footer_status_timers` | `test_app_stops_footer_status_timers_and_diagnostics` |

Suites run green: `test_footer_token_timer_retired` + `test_ui_responsiveness` +
`test_footer_token_counter_retired` = 21 passed; `test_db_status_manager` +
`test_app_footer_shortcut_context` + `test_ui_responsiveness_stall_persist` +
`test_legacy_entrypoints_retired` + `test_app_startup_performance` +
`test_app_import_weight` = 59 passed.

**Modified/added files:** `tldw_chatbook/app.py`,
`tldw_chatbook/Utils/db_status_manager.py`,
`tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py` (deleted),
`Tests/Chat/test_token_display_limit.py` (deleted),
`Tests/Chat/test_footer_token_dirty_gate.py` (deleted),
`Tests/UI/test_ui_responsiveness.py`,
`Tests/Performance/test_footer_token_timer_retired.py` (new),
`Docs/security/production-diagnostic-inventory.json` (three removed diagnostics --
one `error`, one `debug`, one `info` -- all reviewed with `--statements` first; no
additions, so no new interpolation surface).
