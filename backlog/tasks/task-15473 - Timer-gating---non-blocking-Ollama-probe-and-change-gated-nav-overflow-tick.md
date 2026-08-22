---
id: TASK-15473
title: 'Timer gating: non-blocking Ollama probe and change-gated nav overflow tick'
status: Done
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two verified steady-state loop-stealers from the audit. (1) The Models screen's 3 s status timer ends in a BLOCKING `socket.create_connection(("127.0.0.1", 11434), timeout=0.25)` on the event loop when no app-owned Ollama process exists (`UI/LLM_Management_Window.py:525` -> `UI/Screens/llm_screen.py:90-98`) — instant on ECONNREFUSED but up to 250 ms of frozen UI per probe if the port blackholes (firewalled/container setups). (2) The nav bar re-measures overflow hints every 0.5 s forever on the active screen and schedules two extra callbacks per tick (`UI/Navigation/main_navigation.py:396/:445/:598`) — scroll math, hint toggles, re-center, ghost-button geometry — with no change detection, on every screen, app-lifetime.

Fix direction: probe via `asyncio.open_connection` (same 0.25 s cap) or a thread; nav tick gets a cheap change signature (scroll_x, container width, button count) and skips no-op ticks — or becomes resize/scroll-event-driven if that stays simple. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No blocking connect ever runs on the event loop (evidence); Models availability UX unchanged
- [x] #2 The nav tick performs no measurement/ghost work when nothing changed (evidence); hints still correct after resize, overflow, and scroll (tests)
<!-- AC:END -->

## Implementation Plan

1. Ollama probe (`UI/Screens/llm_screen.py`'s `_probe_local_server`): convert to an
   `async def` using `asyncio.wait_for(asyncio.open_connection(host, port), timeout=0.25)`,
   catching `(OSError, asyncio.TimeoutError)` -> down, success -> up (close the writer).
   Same semantics, same 0.25s cap. Its one caller, `LLMManagementWindow._ollama_api_available`
   (`UI/LLM_Management_Window.py:613`), becomes async and awaits it; its one caller,
   `_update_ollama_api_state` (`:622`), becomes async too. `_update_ollama_api_state` is
   driven from two places: `set_interval(3.0, ...)` in `on_mount` (Textual timers natively
   support async callbacks via `invoke()`, no change needed there) and a synchronous
   `step()` loop in `_finish_deferred_mount` (`:541-551`) -- that loop gets an
   `inspect.isawaitable(result)` check so it can `await` whichever steps are coroutines
   without disturbing the per-step try/except isolation or ordering.
2. Nav overflow tick (`UI/Navigation/main_navigation.py`): add
   `self._overflow_signature: tuple | None = None` in `__init__`. In
   `_update_overflow_hints`, after the existing early-return guards, compute a cheap
   signature `(scroll_x, strip.region.width, strip.virtual_size.width, button ids tuple)`
   from data already being queried; compare to the stored signature and return early
   (no hint toggle, no `_refresh_overflow_hint_visibility`, no `_recenter_strip`
   scheduling) when unchanged, else store it and run the existing pipeline. First call
   after mount always runs (`None` never equals a real tuple). Keep the 0.5s timer
   (stability preference) rather than switching to event-driven.
3. Tests: (a) a counting-seam test proving a repeated no-op tick calls zero downstream
   work (patch `_refresh_overflow_hint_visibility`, assert uncalled on the second,
   unchanged tick); (b) a "content changed" test (mount an extra button into the strip)
   proving the gate does not falsely skip; (c) confirm existing resize/scroll/overflow
   coverage in `test_master_shell_navigation.py`/`test_chrome_ux_fixes.py` still passes
   unmodified; (d) Ollama probe up/refused/timeout parity tests against a real localhost
   listener/closed port (`@pytest.mark.allow_network`, no mocking of asyncio itself);
   (e) a loop-responsiveness test: run the real probe against a genuinely unresponsive
   address (10.255.255.1, a private/non-routed "black hole" -- verified in this sandbox to
   hang for the full timeout rather than fail fast) concurrently with a fast heartbeat
   task, asserting many heartbeat ticks land during the probe's ~0.25s wait (a blocking
   equivalent measurably starves the heartbeat to ~1 tick in the same window).
4. Run nav bar suites + LLM screen suites + new tests; compare failure sets against the
   pre-change baseline (already captured: `Tests/UI/test_llm_screen_lab_adoption.py` has
   20 pre-existing teardown-only network-guard errors unrelated to blocking, and the nav
   suites are clean) to confirm no regressions.

## Implementation Notes

Both halves implemented as planned; no deviations.

**Ollama probe.** `_probe_local_server` (`UI/Screens/llm_screen.py`) is now
`async def`, using `await asyncio.wait_for(asyncio.open_connection(host, port),
timeout=0.25)` in place of the blocking `socket.create_connection(..., timeout=0.25)`.
Same up/refused/timeout semantics, same 0.25s cap. Its sole caller,
`LLMManagementWindow._ollama_api_available` (`UI/LLM_Management_Window.py`), and
*its* sole caller, `_update_ollama_api_state`, are now coroutines too. `set_interval`
needed no change (Textual's `Timer._tick` already awaits a coroutine callback via
`invoke()`); the synchronous `step()` loop in `_finish_deferred_mount` gained an
`inspect.isawaitable(result): await result` branch so the one now-async step is
awaited without disturbing the other two sync steps' independent try/except
isolation. A repo-wide grep confirmed both functions have exactly one caller each --
nothing else needed converting.

**Nav overflow tick.** `MainNavigationBar._update_overflow_hints` now computes a
signature `(scroll_x, strip.region.width, strip.virtual_size.width, tuple(button
ids))` from data it already queries, and returns before any hint toggle /
`_refresh_overflow_hint_visibility` / `_recenter_strip` scheduling when the
signature matches the stored one from the last full pass (`self._overflow_signature`,
seeded `None` so the first pass after mount always runs). Kept the 0.5s timer
(stability preference, per instructions) rather than moving to resize/scroll events.

**Evidence.** Mutation-tested every guard by temporarily breaking it and confirming
the corresponding test goes red, then restored:
- Reverting the probe to the old blocking `socket.create_connection` turned red both
  `test_probe_never_calls_the_blocking_socket_primitives` and
  `test_event_loop_stays_responsive_during_an_unresponsive_probe` (heartbeat count
  0 vs >=10 required) -- the async version measured dozens of heartbeat ticks during
  a real ~0.25s wait against `10.255.255.1` (a non-routed "black hole" address,
  empirically verified in this sandbox to hang for the full timeout rather than fail
  fast) where the blocking equivalent measured zero.
- Inverting the availability gate's boolean in `_update_ollama_api_state` turned red
  all three tests in `test_llm_screen_ollama_ux_unchanged.py`.
- Disabling the nav signature gate (`if False and signature == ...`) turned red
  `test_a_settled_no_op_tick_does_no_measurement_or_toggle_work` (the "born red"
  counting-seam test).
- Dropping the button-ids component from the nav signature turned red
  `test_a_tick_after_a_destination_is_added_still_does_the_work` (the nav-content-
  change case the task called out as needing its own coverage).

**New test files:**
- `Tests/UI/test_llm_screen_ollama_probe_nonblocking.py` -- probe up/refused/timeout
  parity against real sockets (a real listener, a closed port, and the black-hole
  address), a structural guard that the blocking primitive is never called, and the
  loop-responsiveness heartbeat test. Marked `allow_network` (module-level
  `pytestmark`) per `Tests/conftest.py`'s autouse network-egress guard (task-15111).
- `Tests/UI/test_llm_screen_ollama_ux_unchanged.py` -- end-to-end button-gating
  behavior (disabled + tooltip when down, enabled when up, flips on the next tick)
  with the probe patched to a fixed result -- no real socket, no network marker
  needed.
- `Tests/UI/test_nav_overflow_tick_gating.py` -- the no-op-tick counting seam and the
  content-change case.

**Existing coverage re-verified unmodified:** `Tests/UI/test_master_shell_navigation.py`
(the interval/resize/restore-active focus-stranding tests already cover scroll and
resize), `Tests/UI/test_chrome_ux_fixes.py` (overflow-indicator correctness at
several widths), `Tests/UI/test_settings_nav_active_scroll.py`. All pass unchanged
after the gate.

**Pre-existing gap found, not fixed (out of scope).**
`Tests/UI/test_llm_screen_lab_adoption.py` has 20 tests that ERROR at teardown --
`AssertionError: test attempted network egress (blocked): socket.create_connection ->
127.0.0.1:11434` -- because mounting `LLMScreen`/`LLMManagementWindow` runs
`_finish_deferred_mount`'s post-mount step list unconditionally, which reaches the
real Ollama probe with nothing marking the test `allow_network`. Confirmed
pre-existing on a clean worktree (identical `58 passed, 20 errors` before and after
this task's change) and NOT introduced or fixed by converting the probe to async --
`asyncio.open_connection` routes through the same guarded `socket.connect_ex`, so
network egress is denied identically either way. One test's own docstring even
claims "mounting Models reaches no network at all" (task-887), which is no longer
true. A similar, larger-surface version likely also affects
`Tests/ProductionApp/test_llm_destination_actions.py` (same mount path, no
`allow_network` marker found there either) -- not verified in depth, out of this
task's scope. Worth a follow-up task to either mark the affected tests
`allow_network` or patch `_probe_local_server` in their shared fixture.

**Files modified:** `tldw_chatbook/UI/Screens/llm_screen.py`,
`tldw_chatbook/UI/LLM_Management_Window.py`,
`tldw_chatbook/UI/Navigation/main_navigation.py`.
**Files added:** `Tests/UI/test_llm_screen_ollama_probe_nonblocking.py`,
`Tests/UI/test_llm_screen_ollama_ux_unchanged.py`,
`Tests/UI/test_nav_overflow_tick_gating.py`.

## Review fix (post-approval minor)

Reviewer flagged a real race the async conversion introduced:
`_update_ollama_api_state` checked `is_attached`/`screen.is_active` only BEFORE the
now-awaited (up to ~0.25s) probe call; the old synchronous version was atomic
end-to-end, so a widget that detached or whose screen went inactive WHILE the probe
was in flight used to still get its buttons mutated once the probe resolved, since
nothing re-checked after the `await`. Fixed by repeating the identical guard
immediately after `available = await self._ollama_api_available()`, before touching
any button.

Added `test_a_screen_switch_mid_probe_leaves_buttons_untouched` in
`Tests/UI/test_llm_screen_ollama_ux_unchanged.py`: holds the probe open on an
`asyncio.Event`, pushes a new screen on top of Models (so `screen.is_active` goes
`False` while the window stays fully mounted), releases the probe with a result that
WOULD flip every gated button, and asserts every button's `(disabled, tooltip)` is
byte-identical to its pre-probe baseline. Mutation-tested (removed the new guard,
confirmed red; restored, confirmed green).

One trap hit while writing this test, worth recording since it nearly produced a
vacuous pass: the first version simulated the race with `await window.remove()`
(genuinely flips `is_attached` to `False`) instead of a screen push. That mutation-
tested as a FALSE PASS -- removing the window also tears down its descendants, so
`view.query(Button)` returns zero buttons post-removal and the loop body never runs
regardless of whether the guard exists, i.e. the test would have passed even with
the guard deleted. Switched to pushing a new screen on top (confirmed: `is_attached`
stays `True`, buttons remain real and query-able, only `screen.is_active` flips) --
that version killed the mutant. Lesson candidate for
`backlog/docs/lessons-testing-evidence.md` if this pattern recurs: a widget-removal
race test can silently test nothing if removal also empties the exact query the
code under test uses.

Also hit and fixed independently: this file's mount helper used a single
`pilot.pause()` (unlike `test_llm_screen_lab_adoption.py`'s established two-pause
convention for this same deferred mount), which surfaced as a genuine low-rate
flake (`NoMatches` on `#llm-view-ollama`) reproduced in a 30-run loop at roughly
1-in-15. Fixed by matching the two-`pilot.pause()` convention inside the shared
`_mount_models_with_probe_result` helper; reran 20/20 clean afterward.

Reviewer's second minor accepted as residual, not fixed: the refused-connection
path's promptness (`_probe_local_server` returning `False` quickly on ECONNREFUSED)
is logically OS-immediate but has no test asserting an upper time bound --
`test_probe_reports_down_on_a_refused_connection` in
`Tests/UI/test_llm_screen_ollama_probe_nonblocking.py` asserts the boolean outcome
only. Left as-is per reviewer guidance.
