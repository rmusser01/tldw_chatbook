---
id: TASK-34924
title: Restoring a backup can undo a Dreams switch-off
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- backup-recovery
priority: medium
updated_date: 2026-10-15 15:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Switching Dreams off stops the sources, alert rules and reminders Dreams created. Those live in the subscriptions and scheduling stores, which a backup restores, while what Dreams recorded about them does not change. After restoring a backup taken while Dreams was on into a profile where Dreams is now off, the restored sources run again: the next start reads none of them, the dispatch gate allows them, and their checks and alerts run with Dreams off. Since the URL-provenance rework they are fetched under the public-only rule, so this is no longer a private-address exposure, but it is work the user switched off. The user guide does not mention it. TASK-34816's provenance criteria now look met by that rework; this is the part it does not cover. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a restore with Dreams off, no source, rule or reminder Dreams created runs until Dreams is switched on
- [x] #2 After a restore with Dreams on, every live watch's source runs
- [x] #3 A test restores each case with real stores and fails before the fix
- [x] #4 The user guide says what a restore does to Dreams' watches
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Owner decision 2026-10-14: option C, a boot-time check, no schema change, without importing the track service at boot.
2. Confirm the stores are already open at boot: app_service_wiring builds app.subscriptions_db (and queries it, capture_prior_process_boundary) and SchedulingService(db=ScheduledTasksDB) on every start.
3. At each start, in track_reconcile._boot_db, with the live watches' rows from the Dreams read it already makes: one read of the run state (is_active, is_paused) of their sources and the enabled flag of their Dreams rules (subscriptions store), and one of the reminders linked to them (scheduling store). Skipped when there is no live watch.
4. Compare with the saved switch: Dreams off and a live watch settled off whose Dreams-created source runs, whose Dreams rule is enabled or whose reminder is armed -> un-settle its record; Dreams on and a live watch on a Dreams-created source that is off while its record owes nothing for it -> give it a restart debt. Then the existing pass runs in the switch's direction (the boot already runs it when a record is unsettled). The record writes happen after the store gates; the dispatch guard keeps holding Dreams work until the check has run (a Dreams-off disagreement must precede dispatch).
5. Tests first, real stores, red first: restore (files replaced) with Dreams off brings back a running source, an armed reminder and an enabled rule; with Dreams on brings back an off source. Plus: a start that finds them in step writes nothing; the drift reads never import the track service.
6. Measure the reads; boot censuses before and after; guide; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decision 2026-10-14: option C, a check at every start; no schema change.

- Stores at boot (confirmed): app_service_wiring builds app.subscriptions_db on every start and queries it (capture_prior_process_boundary), and SchedulingService(db=ScheduledTasksDB(...)) (the scheduler loop's store). The check reads both by key and builds neither; nothing new is imported in track_reconcile (no track service at boot; boot censuses green before and after).
- Reads (track_reconcile._read_drift, on the boot worker inside _boot_db): DreamsDB.tracking_boot_rows (the live rows in the same hop as before, plus, with Dreams off, retired rows that carry a record; both plan-pinned on the status index); SubscriptionsDB.run_states (is_active/is_paused of those sources and, with Dreams off, enabled of the live watches' Dreams rules, by primary key; read-only); with Dreams off and an event-dated live watch or a retired record listing a reminder, ScheduledTasksDB.list_reminder_tasks_by_link. Nothing is read when there is no watch to look at. The subscriptions store's per-thread connection is closed after the read, as Subscriptions.db_offload does.
- Measured 2026-10-15 (20 live watches, real stores, test profile, load ~10): queries 0.4 ms on an open connection; the cost is opening one through the private-SQLite admission (plain sqlite3 open 1 ms): about 80 ms per start with Dreams on (one store), 150 ms with Dreams off (two), on the boot worker; it runs before the first-dispatch gate opens, so the scheduler's first dispatch waits for it too.
- Compare (_restore_drift, pure): Dreams off, a live watch settled off whose Dreams-created source runs (is_active; Resume clears only is_paused, so is_active 0 with is_paused 1 does not run), whose Dreams rule is enabled, or whose reminder is armed -> its record is written back unsettled and the existing switch-off pass stops it; Dreams off, a retired watch whose retirement record says Dreams switched its source off (no live watch riding it) while it runs, or lists a disarmed reminder that is armed -> stop_restored_leftovers (track_service, imported only when there is work) switches it off again; Dreams on, a live watch with created_by_dreams on a source that exists and is off, whose record owes nothing for it -> a restart debt, and the existing switch-on pass restarts it after its public-address check.
- Decision on the restart: everything else follows a record; the restart is the one inference, limited to a Dreams-created source under a live watch, because nothing in Watchlists or the agent tools switches an existing source off today. A future disable-source control must leave a mark this check can read (stated in the code). A rule or reminder left off with Dreams on stays off (it may be the user's choice). With Dreams on, retired watches are not looked at.
- Writes happen in _boot_pass only after the storage-admission and store gates; the first-dispatch gate and the per-task dispatch guard keep holding Dreams-owned work until the check has run (a Dreams-off disagreement or a retired leftover must precede dispatch; a restart does not, like any resume).
- Tests (Tests/Dreams/test_track_reconcile.py, a restore modelled as both store files replaced by an earlier SQLite backup while the app is down): test_a_restore_with_dreams_off_runs_nothing_dreams_created, test_a_restore_with_dreams_on_runs_every_live_watchs_source, test_a_restore_with_dreams_off_stops_what_a_retired_watch_left (all three red on the code before this change), test_a_start_that_finds_the_stores_in_step_writes_and_loads_nothing (Dreams on and off: no write, track service not imported), test_a_paused_source_the_switch_disabled_is_not_seen_as_running; Tests/Dreams/test_dreams_db_track.py pins both new statements' plans and their literals.
- Guide (dreams.md): after the restore paragraph, what the next start does with each direction, and what stays as the restore left it.
- Not covered (pre-existing, not a restore's doing): Dreams' change rule of a watch retired while Dreams was on stays enabled by design (rules survive an untrack), so with Dreams off it still alerts on the user's own attached source.
- TASK-34816: its ACs 1, 2 and 4 look met by the provenance rework (Tests/Backup_Recovery/test_restored_source_first_fetch.py passes here) and AC 3 by the Dreams store being a backup owner (Backup_Recovery.data_groups: db.dreams in conversations); with this task the cross-store drift is handled too. Reported to the coordinator; not closed here.
- No logger change. Verified 2026-10-15 on fix/task-34924-dreams-restore-boot-check off dev c12611946c: Tests/Dreams 907 passed (-n 6), Tests/Backup_Recovery test_dreams_activation + test_dreams_owner + test_restored_source_first_fetch 53 passed, Tests/Performance/test_ui_ready_module_census.py + test_boot_worker_census.py 7 passed before and after (serial), all through the scratch wrapper.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
