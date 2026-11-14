---
id: TASK-34925
title: Dreams code comments and guide state behaviour dev no longer has
status: Done
created_date: 2026-10-12 23:18
labels:
- dreams
- docs
priority: medium
updated_date: 2026-10-14 15:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After the URL-provenance rework merged with PR #2921, several statements contradict the code. A track-service docstring says a restored unmarked source is fetched as a trusted origin (it is not) and gives a boot-path cost that is not the real reason. The startup-reconcile docstring names two of the three stores it asks and gives no reason for checking once instead of holding a scope. The user guide says a user's own sources are fetched exactly as before, private address or not, which is false since every source a user made before the rework reads as unknown, and its restore paragraph does not say what a restore brings back. Found in the independent review of PR #2921 (round 5) and re-checked on dev 7b38b26c69 after the merge, with real stores.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every statement listed in the review report about egress provenance, restores and the startup reconcile matches dev's behaviour
- [x] #2 The guide says how a user's own private sources are treated after the provenance rework
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Scope (coordinator, 2026-10-14): AC 2 was covered by #3101 (TASK-34926); do AC 1 -- the backfill_egress_marks docstring, the startup-reconcile docstring with its store list and the reason it checks once, and the guide's restore paragraph.
2. Check each statement against dev: how an unmarked source is fetched (Utils.egress.provenance_of_stored_source), which stores the reconcile asks (track_reconcile._stores_may_run), which backup groups hold Dreams' parts (Backup_Recovery.data_groups), and what a restore replaces (ADR-126).
3. Rewrite the statements that are wrong; no behaviour change.
4. Run the Dreams tests that cover these modules; notes, ACs, Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- AC 2 was done by #3101 (TASK-34926: the guide's Tracking paragraph and the Watchlists section on feeds at a private address).
- backfill_egress_marks docstring: it said an unmarked source reads as user-typed and is fetched as a trusted origin, and that a restored unmarked source stays one until Dreams is switched; since ADR-207 a source with no record or a malformed one is fetched under the public-only rule (Utils.egress.provenance_of_stored_source), and the docstring now says so and why the mark is still written. It said only the subscriptions store is covered by backup; the Dreams store is backed up too, in another group (Backup_Recovery.data_groups: db.dreams in conversations, db.subscriptions in research). The reason a start with no other work does not verify is now the real one -- an unmarked source is not trusted any more, so the boot cost buys nothing -- and the cycle verifies on every cycle run (not at most once a day).
- track_reconcile._stores_may_run docstring: it named two of the three stores the reconcile asks (db.scheduled_tasks, db.subscriptions; db.dreams added), and gives the reason it checks once instead of retaining activation.execution_scope: the scope is meant to be held inside one synchronous worker, the pass is many awaits long, and a maintenance pause drains live storage leases up to its deadline (storage_admission._LocalPause.drain), so a held lease would make a backup capture wait on name lookups; the cost (a restore that begins after the check is not held off) is stated. The _boot_pass comment no longer calls an unmarked source a trusted egress origin.
- Guide (dreams.md): a new paragraph says what a restore brings back -- stories, watches and their records with Conversations, notes, and personas; Watchlists sources and change alerts with Research and subscriptions; event reminders with Schedules, notifications, and sync; the switch with Settings; each as it was in the backup, a group left out keeping what it has (ADR-126) -- and the reconcile's approval sentence names the Dreams store too.
- No behaviour change, no logger change.
- Verified 2026-10-14 on fix/dreams-tracking-followups-34919: Tests/Dreams 884 passed (-n 6, scratch wrapper).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
