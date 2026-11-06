---
id: TASK-39046
title: "Two no-lane tests fail under host load: the provider-conflict Review test presses Review before the category pane swap lands, and the wedged-bootstrap test bounds its second call at 10 ms"
status: To Do
created_date: 2026-11-06 10:09
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
Found in round 10 of the 2026-10-23 burn-down; checked on dev b2c618185f, 2026-11-06, and on the state after the round-10 Console PRs merge. Both files are the same in the two trees at these lines; line numbers are from the post-merge state.

**`test_provider_navigation_conflict_requires_review_discard_or_return`** (`Tests/UI/test_settings_configuration_hub.py`, a private-profile test). No PR lane runs it: the census lists other nodes of this file, not this one.
- 10 runs at `-n 4`: 1 red (run 9, load 28) with `SettingsURLInput(id='settings-provider-endpoint-value').has_focus` False at `:8720`. The others took 14.6-27.8 s at load 16-31.
- A probe (20 iterations, load about 23, 2 misses) shows the cause. In a miss, `apply_navigation_context`'s asynchronous pane swap consumes its pending category focus about 40 ms after the Review press. It calls `_focus_category` at `settings_screen.py:4774-4777`, which takes focus to the rail button, and the endpoint never gets focus (not even after 3 s). In passing runs that focus lands 0.1-0.6 s before the press.
- So the single `pilot.pause()` at `:8700` does not wait for the swap, and waiting longer for `has_focus` would not help.
- A pending navigation focus overriding a focus the user just set is also a minor product edge, reachable only by a press within about 40 ms of the navigation.

**`test_a_wedged_bootstrap_does_not_park_a_second_worker`** (`Tests/Chat/test_console_chat_controller.py`). No PR lane runs the file.
- It asserts `second_elapsed < 0.01` (`:11789`, from 027a136111, task-32275), while the budget it guards is patched to 0.05 s. It failed at 11 ms in a whole-file run at load 15-63.
- Alone, 10 runs at `-n 4`: 10 green, 4.1-6.4 s, load 15-26.
- A probe of 200 repetitions at load about 24 gave median 0.1 ms, p99 4.9 ms and max 6.0 ms, so the margin is about 1.7x.
- Also, a released earlier worker's `_bootstrap` finally clears whichever `_PERSONAL_CONTEXT_BOOTSTRAP_IN_FLIGHT` is patched in at that moment (`console_chat_controller.py:20818-20826`). A repeated body without per-iteration isolation saw `calls == [1, 1]`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Review test waits until the pane swap's pending category focus is consumed before it presses Review; under a temporary delay before `_focus_category` (`settings_screen.py:4777`), the old test fails at the `has_focus` check every time and the new one passes
- [ ] #2 The Review test passes 20 runs at `-n 4`, with the load recorded
- [ ] #3 The wedged-bootstrap test's bound stays below the patched budget, so it still fails when the second call waits the budget out; this is shown red under a temporary edit that removes the in-flight guard
- [ ] #4 The wedged-bootstrap test passes 10 whole-file runs at `-n 4`, with the load recorded
<!-- AC:END -->
