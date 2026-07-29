---
id: TASK-1022
title: >-
  ADR-019's rollback path does not exist and scheduled watchlist checks are off by default
status: Done
assignee: []
created_date: '2026-07-27 14:00'
labels:
  - scheduling
  - adr
  - watchlists
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while resolving TASK-1010. ADR-019 (**Accepted**) migrates watchlist checks to the unified scheduler behind `scheduling.watchlist_checks_enabled`, and makes two explicit promises:

> "Removal of the old scheduler is deferred to a follow-up release after dual-run validation has completed."
> "A runtime toggle allows instant rollback to the old scheduler **without a code deploy**."

Neither holds today.

**The flag has no `else` branch.** `app.py:4860-4871` reads it and, when true, constructs a `WatchlistCheckHandler`. When false, `watchlist_handler` is simply left `None`. Nothing starts the old scheduler in its place:

```python
watchlist_handler = None
if watchlist_checks_enabled:
    watchlist_handler = WatchlistCheckHandler(...)
```

**The old scheduler has no construction path from the app.** `SubscriptionScheduler` is instantiated in exactly two places repo-wide: inside `SubscriptionSchedulerWorker` (whose only constructor, `SubscriptionBackendController`, is itself never instantiated anywhere), and inside `create_scheduler()`, which has **zero callers**. Confirmed by `inspect.getsource(TldwCli)`: none of `SubscriptionScheduler`, `SubscriptionSchedulerWorker` or `SubscriptionBackendController` appear anywhere in the app class.

**The flag defaults to false** — in both the `get_cli_setting` call and `config.py:2296`.

Taken together: on a default install, no *scheduled* watchlist execution runs at all — neither the new handler nor the old scheduler. Toggling the flag off does not roll back to the old scheduler; it silently disables watchlist scheduling entirely, with no error.

**What still works:** the manual path. `watchlists_collections_screen.py:2246` handles `CheckNowRequested` and calls `controller.check_now(...)` directly, touching no scheduler. So user-triggered "Check now" is unaffected, and the old `SubscriptionScheduler` class is itself still functional if constructed directly — it was verified to start and stop cleanly. It is unreachable, not broken.

This needs an owner decision rather than a mechanical fix, which is why it is filed rather than patched:
- If dual-run rollback is still wanted, wire the `else` branch to a working old-scheduler entry point — noting `SubscriptionSchedulerWorker` cannot serve as that entry point as written, because `@work` asserts `isinstance(self, DOMNode)` and the class is not one, so it raises on first call.
- If the migration is far enough along that rollback is no longer wanted, update ADR-019 to say so and retire the dead scheduler, worker and controller together.

Either way the ADR and the code should stop disagreeing.
**How it came to be unreachable.** The ADR was not wrong when written. `019-watchlist-scheduler-migration.md` landed at 2026-07-19 09:29 (`8a9ce5cf9`), while `SubscriptionWindow.py` — the live caller that constructed `SubscriptionBackendController` — still existed. Roughly twelve hours later the same day, `fc9e50da5` ("retire SubscriptionWindow and fold subscriptions route into Watchlists") removed that window, orphaning the controller and with it the only construction path to the old scheduler.

So this is not an ADR that was authored against code that never worked; it is a rollback path that was silently severed by a later refactor on the same day, with nothing to detect the loss. That is worth knowing for the decision below: restoring rollback means giving the old scheduler a new entry point, not merely repairing an old one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] A decision is recorded on whether ADR-019's rollback guarantee is still wanted
- [ ] If kept: toggling the flag off demonstrably runs the old scheduler, proven by a test
- [ ] If dropped: ADR-019 is amended, and the dead scheduler/worker/controller are removed together
- [ ] The default value of `scheduling.watchlist_checks_enabled` is a deliberate, documented choice
<!-- AC:END -->

## Ownership

Handed to the **watchlists workstream** (2026-07-27). This was found incidentally while deleting dead worker components under TASK-1010, and the investigation deliberately stopped at establishing the facts: the decision — restore the rollback path, or amend ADR-019 and retire the old scheduler chain — belongs with whoever owns the watchlist migration, not with a cleanup task.

Everything needed to make that call is in the description above: the missing `else` branch, the absent construction path, the default-false flag, what still works (manual "Check now"), and the twelve-hour window in which the rollback path was severed. No code was changed.

## Implementation Notes

Resolved by TASK-1210 (#1054), TASK-1211 (#1058) and the ADR-019 amendment they carried. Closing
against that work rather than doing it twice.

**This task was right, and it was here first.** Filed 2026-07-27 from a reading of the code, it had
already established every load-bearing fact: the flag has no `else` branch, `SubscriptionScheduler`
has no construction path from the app, the flag defaults false, and manual "Check now" is
unaffected because it bypasses the scheduler entirely.

A day later TASK-1210 re-derived the same conclusion from a runtime import trace, without checking
whether the board already held it. The duplicated effort was mine; this file was the better
starting point and I should have found it. The repo's own hygiene note — *audit the board, do not
trust your own summary* — exists for exactly this.

What shipped against it:

- `watchlist_checks_enabled` now defaults **true** and `watchlist_checks_shadow` **false**, in the
  shipped TOML and in `app.py`'s in-code fallbacks, so scheduled checks actually run. Verified live:
  a seeded overdue source fetched 5 real items with no user interaction.
- The unreachable `SubscriptionScheduler` / `SubscriptionSchedulerWorker` and the briefing island
  they anchored were removed — ~8,150 LOC.
- ADR-019 is amended to record that the dual-run and its rollback lever were never implemented, and
  that the promotion gate it defined was unsatisfiable because the path it compared against did not
  run.

