---
id: TASK-1010
title: >-
  Three worker components are dead or broken on first call -- decide whether to fix or delete
status: In Progress
assignee: []
created_date: '2026-07-27 12:30'
labels:
  - ui
  - dead-code
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while auditing `@work(thread=True)` async workers for TASK-981. Three components are not merely untidy — each is broken in a way that proves nothing exercises it. They were reachable only through code paths that never run, which is why the defects survived.

**1. `Widgets/Media_Creation/swarmui_widget.py::generate_image` could not have worked.** It called `loop.run_until_complete()` from inside the fresh event loop that `@work(thread=True)` on an `async def` already creates via `asyncio.run()`. Reproduced directly: `RuntimeError: Cannot run the event loop while another loop is running`. TASK-981 converted it to a plain `def`, matching the working pattern used twice elsewhere in the same file — but the widget appears never to be mounted, so the fix is untested in situ. Confirm whether the widget is live; if it is not, delete it rather than carrying a fixed-but-unused component.

**2. `Widgets/multi_item_review_window.py::_generate_analyses_worker` references `app.llm_api_client`, which does not exist.** Its worker is otherwise sound (self-contained awaits, no loop-bound sharing), so TASK-981 left it async. But an attribute that is not defined anywhere means this path cannot have run.

**3. `Subscriptions/textual_scheduler_worker.py::SubscriptionSchedulerWorker` fails Textual's own guard immediately.** `@work` asserts `isinstance(self, DOMNode)`, and this class is not a `DOMNode`, so calling `start_scheduler` raises `AssertionError` on the spot — reproduced. Left unfixed under TASK-981 because the component is deprecated and the ADR-019 migration is underway, but it should not sit in the tree pretending to work.

For each: establish whether anything constructs and uses it. Delete what is dead; fix and add a test for whatever is meant to be live. Do not leave a third state where the code looks maintained but cannot execute.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] Each of the three is confirmed live or dead by finding its real construction site
- [ ] Dead components are deleted, including their tests and any registration
- [ ] Live components are fixed and covered by a test that would fail against the broken version
- [ ] `SubscriptionSchedulerWorker`'s status is resolved against the ADR-019 migration rather than left ambiguous
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Deletions (commit `4735d8461`, already on branch):** `SwarmUIWidget` (`Widgets/Media_Creation/swarmui_widget.py`) and `MultiItemReviewWindow` (`Widgets/multi_item_review_window.py`) were removed along with their tests and the now-empty `Widgets/Media_Creation` package. Verified: `grep -rn "SwarmUIWidget\|MultiItemReviewWindow" --include="*.py" .` returns nothing repo-wide, and `python -c "import tldw_chatbook.app"` still succeeds. Both were confirmed dead by finding zero construction sites before deletion.

**Item 3 (`SubscriptionSchedulerWorker`) — investigated, not touched.** Traced the ADR-019 rollback path end to end:

- `scheduling.watchlist_checks_enabled` is read exactly once, in `app.py` (`TldwCli.__init__`, ~line 4860). It gates only whether a `WatchlistCheckHandler` (`Scheduling/scheduler/handlers/watchlist_check_handler.py`, imports `FeedMonitor`/`URLMonitor` directly — no dependency on `SubscriptionScheduler`) is registered as the `"watchlist_job"` handler on the new `SchedulerLoop`, and whether `watchlist_projection` is passed to it. There is no `else` branch. Verified by reading app.py and by `inspect.getsource(TldwCli)` containing zero occurrences of `SubscriptionScheduler`, `SubscriptionSchedulerWorker`, or `SubscriptionBackendController` (ran directly).
- `SubscriptionScheduler(...)` is constructed in exactly two places repo-wide (`grep -rn "SubscriptionScheduler(\|create_scheduler("`): inside `SubscriptionSchedulerWorker.__init__` (the dead-on-first-call worker), and inside `create_scheduler()`, a factory with zero callers anywhere in `tldw_chatbook/` or `Tests/`. `SubscriptionBackendController(` has zero construction sites anywhere. `git log -S"SubscriptionSchedulerWorker("` on `app.py` returns nothing — app.py has never constructed the worker directly since the controller was extracted (commit `20f015aa8`). The current live Watchlists screen (`UI/Watchlists_Modules/watchlists_backend_controller.py`) is a different, newer controller with zero scheduler references at all.
- **Verdict: rollback does not depend on the worker or the controller — because it never reaches `SubscriptionScheduler` by any route, not even a broken one.** Setting `watchlist_checks_enabled = false` and restarting does not "resume" the old scheduler as ADR-019's rollback plan states; it only stops the new handler from being registered on `SchedulerLoop`. A user following the documented rollback procedure would see no crash (the assertion only fires if the dead worker is ever invoked, and nothing invokes it) but silently **zero** watchlist execution via the scheduling module — neither old nor new — which is a worse outcome than the ADR describes, not a merely-cosmetic gap.
- Confirmed independently, by direct instantiation (ran a script constructing `SubscriptionScheduler` via `create_scheduler()` against a real `SubscriptionsDB`, then `await scheduler.start()` / `await scheduler.stop()`): the old scheduler class itself is fully functional standalone — it starts its worker tasks and stops cleanly. So the class is not broken; it is simply unreachable from the running app on either flag setting. The deferral in ADR-019 ("removal deferred... it remains functional for the dual-run validation period") only makes sense if something still wires it up, and nothing currently does.
- This is a live gap in an Accepted ADR, not a defect in dead code that's safe to delete outright: deleting `SubscriptionSchedulerWorker` now would make the ADR's stated rollback plan permanently false rather than accidentally false, and re-wiring it (fixing the `DOMNode` assertion, instantiating `SubscriptionBackendController` somewhere real, and restoring an `else` branch in app.py that starts the old scheduler when the flag is off) is a real feature-restoration change, not a cleanup. Minimum fix sketch: either (a) wire `app.py` to construct/start `SubscriptionScheduler` directly (bypassing the Textual-worker wrapper entirely, since the class works standalone) when `watchlist_checks_enabled` is false, or (b) update ADR-019 to reflect that rollback is no longer live and document the actual current fallback behavior (none). Which of these is correct is a product/architecture call outside the scope of this investigation task — hence left as an owner decision.

Per the task owner's instruction, AC1 is ticked (all three components' live/dead status is now established with construction-site evidence). AC2 and AC3 are left unticked: only two of the three dead components were deleted; the third is intentionally not deleted because doing so, or leaving it as-is, both have consequences for an Accepted ADR's documented guarantee that this investigation cannot resolve. AC4 is left unticked for the same reason — "resolved... rather than left ambiguous" needs an owner decision on which of the two remediation paths above (or updating the ADR) to take, not just a diagnosis. Status set to `In Progress` rather than `Done`: real work remains (an explicit decision on `SubscriptionSchedulerWorker`/ADR-019, plus whichever deletion or fix follows from it), and forcing this to `Done` would misrepresent that gap as closed.
<!-- SECTION:NOTES:END -->