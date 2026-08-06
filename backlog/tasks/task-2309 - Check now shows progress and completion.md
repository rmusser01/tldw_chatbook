---
id: TASK-2309
title: Check now shows progress and completion
status: In Progress
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: pressing "Check now" produces ~5 seconds of dead air — no progress
indicator, no completion signal, nothing preventing a confused second click
from queuing duplicate work.

UAT finding F19.

## Acceptance Criteria (the what)

- [x] Triggering a check gives immediate visible acknowledgment, a busy
      state while running, and a completion signal (including the failure
      case).
- [x] A second activation while a check runs is debounced or explicitly
      queued, never silently duplicated.

## Implementation Plan (the how)

1. The screen records which source ids have a check in flight
   (`_checks_in_flight`). `handle_check_now_requested` refuses a second
   activation for a source already being checked, with a toast that says so
   -- debounce, stated, never silent.
2. Immediate acknowledgment: a `Checking <name>...` toast posted before the
   worker starts, and a busy state that outlives the toast -- both Check now
   buttons (Sources pane and Inspector) go disabled and read `Checking...`
   for the source being checked.
3. Completion: the existing success/failure toasts gain the source's name and
   the run's own counters, and the busy state is cleared in a `finally` so a
   raising check cannot strand a permanently-disabled button.
4. Drop `exclusive=True` from the check worker in favour of a named group: it
   made a second press CANCEL the first mid-write, which is exactly the
   unsound cancellation-supersede TASK-1541 documents (a cancelled
   `execute_run` leaves its run row at `running` forever).
5. Tests: a new UI file covering acknowledgment, busy state, the debounce
   refusal, and the failure path's completion signal.

## Implementation Notes

Not started by the WIP commit (`html_text.py`/`humane_time.py` covered
tasks 2307/2308 only); this task was implemented in full this session, in
`tldw_chatbook/UI/Screens/watchlists_collections_screen.py`,
`sources_pane.py` and `inspector_pane.py`.

**Design, matching the plan:** `_checks_in_flight: set[str]` on the screen
(keyed by the normalized `id`, e.g. `local:subscription:5` -- the same key
`selected_source`/`selected_entity` already use) is the one source of truth.
`handle_check_now_requested` checks it first: a source already in the set
gets a stated `"Already checking {name}."` warning toast and nothing else
happens; otherwise the id is added, an immediate `"Checking {name}..."`
toast fires, `_set_check_now_busy()` paints the busy state onto both
Check-now buttons (Sources pane AND the Inspector -- both post the identical
`CheckNowRequested`, so both had to show the same state, or the Inspector's
copy would stay clickable while a duplicate was already refused elsewhere),
and the worker runs in a NAMED group (`"wc_check_now"`) rather than the old
`exclusive=True`. That drop matters on its own: `exclusive=True` with no
group name lands in the shared default group used by ~25 other call sites
on this screen, and CANCELS whatever else is running in it -- for a source
check that is the unsound cancellation-supersede shape TASK-1541 documents
(a cancelled `execute_run` leaves its run row at `running` forever). The
worker's whole body is wrapped in `try`/`finally`: the id is discarded from
`_checks_in_flight` and the busy state is repainted off unconditionally, on
every exit path including an exception the method does not itself expect --
so a raising check cannot strand a button permanently disabled.

Busy state is a plain (non-recompose) reactive on `SourcesPane`
(`busy_source_ids`), repainted by `_update_action_buttons` the same
surgical way selection already is -- a recompose here would rebuild the
live table under the user. On `InspectorPane` it IS `recompose=True`: that
pane already fully recomposes on every selection/scope change, a check
starting or ending is rare next to that, and there is no live table or
scroll position to disturb. Both panes are reconstructed from scratch on
every workbench rebuild (`_build_detail_pane`/`_build_inspector_pane`), so
both are re-seeded from `_checks_in_flight` on every rebuild -- the same
rebuild-survival pattern every other piece of screen state on this pane
already uses.

**Backward compatibility:** `_check_now_source` grew two new parameters
(`source_key`, `name`) but they default to `None` and are derived
internally when omitted, because `Tests/UI/test_watchlists_rail_counts_
and_scope.py` calls this worker directly in three places, bypassing the
message handler entirely -- an existing, established pattern in this test
suite for exercising the worker in isolation. Making the params optional
(rather than editing those three call sites) kept that test file's own
scope untouched.

**Live-verified**, real network check against `https://hnrss.org/frontpage`:
pressing Check now produced an immediate "Checking HN Frontpage..." toast
with BOTH the Sources pane's and the Inspector's Check-now buttons reading
"Checking..." and disabled; on completion both reverted to "Check now", the
Sources "Last scraped" cell updated, and 4 consecutive runs all appear in
the Runs tab with correct counts (20 found / 20 processed each). The
button-level debounce (a disabled button cannot post a second
`CheckNowRequested` at all) was directly observed; the message-level
debounce (the `c` keyboard action, which bypasses the button) could not be
independently reproduced live because the real feed check completes in
well under the ~150-300ms a tmux send-keys round trip costs here -- it is
covered instead by `test_a_second_press_while_checking_is_refused_not_
duplicated`, which uses an `asyncio.Event`-gated fake executor to hold a
check open deterministically, and by a mutation (disabling the debounce
`if` turned that test red).

Tests: new `Tests/UI/test_watchlists_check_now_progress.py` (5 cases --
immediate ack + busy button, second-press refusal, a DIFFERENT source is
unaffected by another's in-flight check, failure clears the busy state,
both activation sites agree). Mutation-verified: disabling the debounce
check, removing the `finally` cleanup, and inverting each pane's busy-state
predicate all turned the relevant test(s) red; all four reverted with an
md5-verified byte-identical restore.

Modified/added: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
(`_checks_in_flight`, `_set_check_now_busy`, `_check_now_entity_name`,
rewritten `handle_check_now_requested`/`_check_now_source`, seeding in
`_build_detail_pane`/`_build_inspector_pane`), `sources_pane.py`
(`busy_source_ids`, `_is_check_now_busy`, `watch_busy_source_ids`,
`_update_action_buttons`), `inspector_pane.py` (`busy_source_ids`, the
source-kind Check-now button's busy branch), `Tests/UI/
test_watchlists_check_now_progress.py` (new), and a one-line fix to
`Tests/UI/test_watchlists_check_now_failure.py`'s polling loop (it broke on
the new immediate-ack toast landing before the failure toast it was
actually waiting for).
