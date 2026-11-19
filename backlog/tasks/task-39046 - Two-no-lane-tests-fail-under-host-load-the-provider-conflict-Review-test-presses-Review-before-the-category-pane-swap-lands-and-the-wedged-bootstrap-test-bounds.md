---
id: TASK-39046
title: "Two no-lane tests fail under host load: the provider-conflict Review test presses Review before the category pane swap lands, and the wedged-bootstrap test bounds its second call at 10 ms"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 10:09
updated_date: 2026-11-06 22:16
labels:
- follow-up
- tests
- flaky
- settings
- console
priority: low
references:
- Tests/UI/test_settings_configuration_hub.py
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/Chat/test_console_chat_controller.py
- tldw_chatbook/Chat/console_chat_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the round-10 Console PRs (send #3433, head e3106a313c; Console tests #3434, head 16c99d2d9f), run stacked with storage #3435. Line numbers are dev's; ones marked #3434 are from that head.

**`test_provider_navigation_conflict_requires_review_discard_or_return`** (`Tests/UI/test_settings_configuration_hub.py`, a private-profile test). No PR lane runs it: the census lists other nodes of this file, not this one.
- 10 runs at `-n 4`: 1 red (run 9, load 28) with `SettingsURLInput(id='settings-provider-endpoint-value').has_focus` False at `:8720`. The others took 14.6-27.8 s at load 16-31.
- A probe (20 iterations, load about 23, 2 misses) shows the cause. In a miss, `apply_navigation_context`'s asynchronous pane swap consumes its pending category focus about 40 ms after the Review press. It calls `_focus_category` at `settings_screen.py:4774-4777`, which takes focus to the rail button, and the endpoint never gets focus (not even after 3 s). In passing runs that focus lands 0.1-0.6 s before the press.
- So the single `pilot.pause()` at `:8700` does not wait for the swap, and waiting longer for `has_focus` would not help.
- A pending navigation focus overriding a focus the user just set is also a minor product edge, reachable only by a press within about 40 ms of the navigation.

**`test_a_wedged_bootstrap_does_not_park_a_second_worker`** (`Tests/Chat/test_console_chat_controller.py`). No PR lane runs the file.
- It asserts `second_elapsed < 0.01` (`:11758`, #3434 :11789, from 027a136111, task-32275), while the budget it guards is patched to 0.05 s. It failed at 11 ms in a whole-file run at load 15-63.
- Alone, 10 runs at `-n 4`: 10 green, 4.1-6.4 s, load 15-26.
- A probe of 200 repetitions at load about 24 gave median 0.1 ms, p99 4.9 ms and max 6.0 ms, so the margin is about 1.7x.
- Also, a released earlier worker's `_bootstrap` finally clears whichever `_PERSONAL_CONTEXT_BOOTSTRAP_IN_FLIGHT` is patched in at that moment (`console_chat_controller.py:20818-20826`). A repeated body without per-iteration isolation saw `calls == [1, 1]`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Review test waits until the pane swap's pending category focus is consumed before it presses Review; under a temporary delay before `_focus_category` (`settings_screen.py:4777`), the old test fails at the `has_focus` check every time and the new one passes
- [x] #2 The Review test passes 20 runs at `-n 4`, with the load recorded
- [x] #3 The wedged-bootstrap test's bound stays below the patched budget, so it still fails when the second call waits the budget out; this is shown red under a temporary edit that removes the in-flight guard
- [x] #4 The wedged-bootstrap test passes 10 whole-file runs at `-n 4`, with the load recorded
<!-- AC:END -->

## Implementation Plan

1. Review test: wait for the navigation's pane swap to land (`_after_category_panes`, the screen's own post-swap hook) before reading the conflict panes and pressing Review. Show the old test failing under a temporary delay before the swap's `_focus_category`, and the new one passing; then run it 20 times at `-n 4`.
2. Wedged-bootstrap test: give the second call a budget far above its bound (patched after the first call), and join the wedged worker inside the test, so its `finally` clears the patched guard and not the next test's. Show it red with the in-flight guard removed; then run the whole file 10 times at `-n 4`.
3. No lane change: the criteria ask for none.

## Implementation Notes

**`test_provider_navigation_conflict_requires_review_discard_or_return`** (`Tests/UI/test_settings_configuration_hub.py`):
- Right after `apply_navigation_context`, the test registers `screen._after_category_panes(event.set)` and awaits the event (bounded at 10 s). The screen drains that queue after the swap's own `_focus_category` and after the conflict widgets' update. So the test reads the rebuilt panes, and Review's deferred focus queues behind the swap's. The wait sits before the reads, not only before Review: under a delay the reads raced the same swap (below).
- Two more waits in the same test, found by the 20-run check at load 60-95 and both deterministic:
  - Review's focus travels press -> handler -> the app's `call_later`, and one `pilot.pause()` returned before it landed. A probe showed the only `set_focus` in a failing run was the swap's, 1.4-2.5 s before the press, and the endpoint was not rebuilt. The test now waits for `endpoint.has_focus`, bounded at 5 s. Nothing can steal focus at that point, so a wait is the right fix here, unlike in the filed case.
  - The Return press's `NavigateToScreen` reaches the harness a hop after the screen. One pause gave `assert 0 == 1` (1 of 8). The test waits for the message to arrive, bounded at 10 s.
- AC#1 negative control (temporary `await asyncio.sleep(...)` before the swap's `_focus_category`, then reverted; serial, run.sh, load 66-104):
  - 20 ms delay: the old test failed 3 of 3 and 2 of 2, but at `assert Select.NULL == 'openai'`, its read of the rebuilt provider pane, one assertion before `has_focus`. The new test passed 3 of 3 and 3 of 3.
  - 0.5 s delay: the old test passed 3 of 3, because the swap's focus then lands after its check. The new test passed 2 of 2.
  - With no delay at load about 80, the old test failed 2 of 2 at the `has_focus` check, the filed failure.
  - So the criterion's "fails at the has_focus check every time" holds only without the delay; with a delay the old test fails every time one read earlier, on the same race.
- AC#2: 20 of 20 passed at `-n 4` (a temporary `parametrize("_rep", range(20))`, removed), load 72-95. Before the two extra waits it was 15 of 20 and then 19 of 20 at load 60-73.
- The product edge stays: a navigation's pending category focus can still take focus from a control focused within about 40 ms after the navigation. Listed as a follow-up, not changed.

**`test_a_wedged_bootstrap_does_not_park_a_second_worker`** (`Tests/Chat/test_console_chat_controller.py`):
- The first call keeps the 0.05 s budget, so it gives up quickly. Before the second call the budget is patched to 2.0 s, and the bound is `second_elapsed < 1.0`: a 100x margin over the 11 ms load failure, yet below the budget. In `finally`, after `released.set()`, the test runs `loop.run_until_complete(loop.shutdown_default_executor())`. That joins the wedged worker while the patched `_PERSONAL_CONTEXT_BOOTSTRAP_IN_FLIGHT` is still in place, so the worker's `finally` cannot clear another test's guard.
- AC#3, with the in-flight check removed from `_personal_context_service` (temporary, reverted): red at `assert [1, 1] == [1]`. With the elapsed bound checked first: red at `assert 2.0025744170002326 < 1.0`. The bound still fails when the second call waits the budget out.
- AC#4: 10 whole-file runs at `-n 4` (run.sh), load 114 falling to 13. The wedged test passed in all 10. Each run was 206 failed, 159 passed, 313-336 s. The same 206 nodes fail in a whole-file run at `fu-base-r11` (load 15-19), mostly `RecoveryRequired: raw_source_selection_changed` and Console session `KeyError`s, none of them this test.

Files: `Tests/UI/test_settings_configuration_hub.py`, `Tests/Chat/test_console_chat_controller.py`.
