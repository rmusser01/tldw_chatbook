---
id: TASK-42415
title: "The right-rail live-work geometry test asserts a quiet rail after a pause and fails admission-sensitive 4 when an outer reconcile is scheduled again"
status: Done
assignee:
- '@claude'
created_date: 2026-11-22 21:17
updated_date: 2026-11-23 02:24
labels:
- follow-up
- ci
- console
- tests
- flaky
priority: high
references:
- Tests/UI/test_console_right_rail.py
- tldw_chatbook/UI/Console_Modules/right_rail.py
- .github/workflows/derived-artifacts.yml
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22), while checking which tests the required "PR Fast Lane (admission-sensitive 4)" job fails on now. Checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports); paths are under `tldw_chatbook/` unless they start with `Tests/` or `.github/`.

**Why.** In a scan of the 140 Derived Artifacts PR runs created from 2026-11-20 00:00Z to about 21:50Z on 2026-11-22 (every attempt; times are job start times), this node failed 18 admission-4 jobs, from 2026-11-20 04:10Z to 2026-11-22 20:48Z, and after 2026-11-21 22:39Z it is the only node that fails that lane: `Tests/UI/test_console_right_rail.py::test_live_work_widget_swaps_cover_real_twenty_twenty_one_geometry[readiness-to-pending-250-10-not_configured-15-21]`, with `AssertionError: (15, 46, 15, False, True)`. Recent runs: 44957243776, 44974726809, 44974749607, 44979357368, 45031398878. It passes locally at the base. The admission-4 reds TASK-33620.15.61's notes list (`test_console_reference_custody.py` ×4 and `test_change_turn_tracking.py` ×2, in 69 jobs from 2026-11-20 03:23Z) last failed at 2026-11-21 22:39Z and pass locally.

**The failure.** The test waits with `_wait_for_right_rail_condition` until `initial_geometry_is_stable()` holds, then `await pilot.pause()` and asserts the same condition again (`test_console_right_rail.py:596-603`). The tuple's last field is `rail._outer_reconcile_scheduled`, so between the wait and the assert the rail scheduled another outer reconcile, while demand (15), width (46), viewport (15) and hint (False) were all as expected. On a loaded runner a late layout or refresh pass schedules a reconcile after the condition first held. The test also clicks `#console-inspector-rail-open` (line 546) without checking the click landed, which is the shape behind another flake, `test_console_edge_focus_restyle_scope.py`.

The logged tuple does not include `bounded._reconcile_scheduled`, which the predicate also checks, so a late local reconcile may be pending too. TASK-33620.7.2 (retry the Inspect live-work card swap dropped while its viewport is still mounting) is a candidate source of a second swap and reconcile.

Whether the late reconcile is expected (a second pass after the swap) or a wasted one is not known; the fix has to say which.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The source of the reconcile scheduled after the initial geometry first holds is identified and recorded in the notes, and the assertion message reports `bounded._reconcile_scheduled` as well
- [x] #2 If that reconcile is redundant, the product stops scheduling it (shown red first); if it is expected, the test waits for the rail to be quiet with nothing scheduled instead of asserting quiet after one pause
- [x] #3 The test confirms the Inspect rail opened before reading its geometry
- [x] #4 Both directions of the parametrized test pass in repeated runs of the file under parallel load (at least 20 runs, or a soak recorded in the notes)
<!-- AC:END -->

## Implementation Plan

1. Instrument (temporarily, in the test) every `_request_outer_reconcile` / `bounded.request_reconcile` call made after the initial geometry first holds, with the caller's stack, and reproduce CI's tuple locally with a forced delay (for example a slowed config save or environment landing) so the late reconcile lands between the wait and the assert. Record the red, then revert the probe.
2. Classify the late reconcile: a legitimate trailing pass (new data or geometry landing after the open) or a redundant/looping one (for example the TASK-33620.7.2 dropped-swap shape).
3. Fix where it lives: a redundant reconcile is stopped in product code with a product assertion shown red then green; a legitimate one makes the test wait for real quiescence (both scheduled flags false, geometry settled, unchanged across consecutive polls) with the existing deadline helper, keeping every geometry assertion.
4. Make the failure messages report every flag the predicates check, including `bounded._reconcile_scheduled`; confirm the Inspect rail opened before reading geometry.
5. Check the other parametrize case and nearby right-rail tests that assert quiet after a fixed pause; fix the same shape where found.
6. Soak the parametrized test 20 times at -n 4 with its file, run the admission-4 step once with its CI flags, then preflight, ruff F and the size ratchets.

## Implementation Notes

**Cause: the swap's own reconcile, which is legitimate.** `_apply_console_live_work_card_swap` queues `_request_console_live_work_reconcile` with `call_after_refresh`. That request asks for one local pass on the live-work section and one owner pass on the rail; it is the documented "live-work card swap" trigger in `right_rail.py`.

In `readiness-to-pending` the first swap replaces the readiness card with another readiness card. The demand stays 15, and nothing is scheduled yet when the swap returns, so `initial_geometry_is_stable()` held on the wait's first poll, before the swap's request ran. A temporary probe logged every `_request_outer_reconcile` and `request_reconcile` call with its callers. It showed:
- the wait returning 14 ms after the swap began;
- the swap's request arriving 38 ms later, from `chat_screen.py:_request_console_live_work_reconcile`.

The single `pilot.pause()` then ran that request. On a starved runner `wait_for_idle` returns before the owner pass's second hop (`_finish_scheduled_outer_reconcile`), which gives `(15, 46, 15, False, True)`.

`pending-to-readiness` never failed, in CI either, because its first swap moves the demand from 15 to 21, which cannot happen before that request.

This is not a product defect: each swap mounts a fresh card that has to be measured, and the request merges with any pass already in flight.

It is not TASK-33620.7.2 either. A dropped swap queues no request at all, so it cannot set either flag. The late request comes from a swap that completed. A dropped swap now fails this test with `swap_reconcile_requests=[]` in the message, where before it passed.

**Red/green.** The temporary edits are reverted.
- Natural load, base test under `taskpolicy -b` at load 23-42: 1 of 8 runs red with CI's exact tuple at line 597.
- Forced delay: a temporary `loop.call_later(0.3, ...)` in `_run_scheduled_outer_reconcile` made the swap-requested owner pass finish 0.3 s late. Base test: 3 of 3 red with `AssertionError: (15, 46, 15, False, True)`, readiness-to-pending only. Fixed test: 3 of 3 green.
- Control for the request clause: the first swap's request was delayed 0.3 s. Without `swap_reconcile_settled(1)`, the fixed test was red 1 of 3 (`Timed out ... owner_passes=3`: the first swap's late pass was counted against the second swap), and the other 2 passes were vacuous. With the clause: 3 of 3 green.

**Change (test only, `Tests/UI/test_console_right_rail.py`).**
- `_wait_for_right_rail_condition` gains `stable_polls`. The default of 1 keeps the old behaviour; above 1, the predicate must hold on that many consecutive polls.
- The live-work test waits for the Inspect rail to open (`_right_rail_open`) instead of `pause(0.2)` (AC #3).
- It wraps `screen._request_console_live_work_reconcile` to record each swap request with the owner-pass count it saw.
- Both waits now need, on two consecutive polls:
  - an owner pass counted after the swap's request (`swap_reconcile_settled(1)` and `swap_reconcile_settled(2)`);
  - both flags false;
  - the expected geometry.
- The one-pause re-asserts are gone. Every geometry, identity and order assertion stays. The final `_reconcile_scheduled is False` and `_outer_reconcile_scheduled is False` asserts now run straight after a wait that saw them false.
- Both wait messages report demand, width, viewport, hint, `local_scheduled` (`bounded._reconcile_scheduled`), `outer_scheduled`, the swap requests and the owner-pass count (AC #1).

**Soak and lane (AC #4).**
- Whole file at -n 4, 20 runs at load 27-42: 20 of 20 green, 37 passed each, so 40 of 40 for the parametrized cases.
- Admission-sensitive 4 step with its CI flags (`--timeout=300 --tb=short`, serial): 598 passed in 36:37 at load 29-43.
- Preflight passed, `ruff --select F` is clean, and `Tests/lint/ruff_gate.py` exits 0.
- `Tests/Architecture/test_screen_size_ratchet.py` and `test_module_size_ratchet.py`: 234 passed. No product module was touched.

**Siblings.**
- The other parametrize case had the same one-pause re-asserts and is covered by the same waits. No other test in this file asserts the reconcile flags.
- `Tests/UI/test_console_rail_reconciliation.py`, in the same lane step, asserts `_outer_reconcile_scheduled is False` after a fixed `_settle(pilot, passes=8|10)` in three tests. Those tests request the reconcile synchronously, so the vacuous-wait shape does not apply, and none failed in the 140-run scan. They are left as they are.
- `Tests/UI/test_console_inspector_navigation.py` imports the helper. It is in no PR lane and fails all 39 tests locally on the `RecoveryRequired` trap. Under a temporary `bootstrap_profile` mark it is red the same way at base and on the branch:
  - two timeouts every time: `test_staged_owner_sync_drives_ten_eleven_ten_cue_and_clamp` and `test_responsive_hide_and_reveal_hands_off_focus_without_losing_offset`;
  - one or two intermittent ones that rotate;
  - 4 and 3 failures in two whole-file runs at base, 4 and 3 on the branch.

**Files.** `Tests/UI/test_console_right_rail.py`; `backlog/docs/lessons-testing-evidence.md` (new entry on waits over a state the action leaves unchanged).
