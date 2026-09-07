---
id: TASK-15864
title: >-
  Restart-staged wake: sidebar badge absent and delivery waits for an unrelated
  retry trigger
status: Done
assignee: []
created_date: '2026-08-13 21:44'
updated_date: '2026-08-14 01:23'
labels:
  - fleet
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-2 Task 7 live restart (scenario 5): with a durable fleet_unseen mark surviving SIGKILL and an owed run in the ledger, the fresh app rendered NO ◈ badge on the marked conversation's sidebar browser row (Task 4's restart claim covers a fresh screen's first read from the DB — live restart contradicted it on the sidebar surface, possibly because no session tab existed for the conversation). Opening the marked conversation view-cleared the mark and created the session, but the seeded wake did NOT deliver on open — the retry-trigger list (composer poke, terminal transitions, drains, mount) omits session-open, so the wake sat pending until a composer keystroke. Also observed and worth a ruling: a wake deferred while its conversation is being VIEWED view-clears the mark while the ledger still owes the wake — a restart in that window leaves an owed, unmarked run the mount-claim will never seed (same shape as the unmarked mid-run orphan, which is by-design per the corrected spec §3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a restart with a staged wake, the marked conversation's sidebar row shows the ◈ badge before the conversation is opened
- [x] #2 Opening a marked conversation (creating its session) is a wake retry trigger — delivery does not wait for an unrelated keystroke
- [x] #3 A ruling is recorded (fix or documented limit) for the owed-but-unmarked window: wake deferred in a viewed conversation, then restart
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AC#1: unseen-mark badge derivation exists only for open-session (native) browser rows; thread the mark into persisted/membership rows so a restart (no session) still shows the diamond; RED first.\n2. AC#2: _resume_console_workspace_conversation creates the session but never pokes the wake; add fleet_wake.retry_soon() at resume; RED first.\n3. AC#3: verify the marks-indexed mount-claim does NOT close the owed-but-unmarked window; fix by making the view-clear yield while the coordinator holds pending for the conversation (mark survives the deferral window across restart); record the ruling in the task file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on fix/fleet-wake-ui-residue (commit cb50508f3).

AC#1: the unseen-badge derivation existed only on the open-session (native) browser-row
path; membership AND persisted rows (both listing variants) now thread the durable mark
via _console_browser_unseen_marker. A native row for the same conversation still wins
the merge's identity slot with its full precedence chain. Live: after SIGKILL+relaunch
the sidebar row rendered '◈ Spawn a researcher' BEFORE the conversation was opened
(frame RESTART_boot_181341 archived) -- the exact surface Task 7 saw empty.

AC#2: _resume_console_workspace_conversation (the one loader of persisted conversations
into sessions) now pokes fleet_wake.retry_soon() via an injected wiring callable; every
delivery gate applies unchanged. Live: one click on the marked row opened the
conversation and the owed wake delivered with ZERO keystrokes, stamped exactly once, and
the notice painted in the viewed transcript (frame RESTART_delivered_181442).

AC#3 RULING -- fixed at the window's entry point, with the limit verified first:
seed_from_marks is marks-INDEXED (the ledger defines WHAT is owed, but only for
conversations the mark names), so an owed-but-unmarked run is invisible to the
mount-claim -- pinned as a documented-limit test. A globally ledger-driven claim was
considered and REJECTED: undelivered_wake_runs' predicate also matches restart-orphan
children swept error in the same pass as their parent (updated_at equality passes >=),
and corrected spec §3 deliberately leaves unmarked mid-run orphans to next-turn handling
-- a global claim would silently promote those to wake deliveries. Instead the
view-clear now YIELDS while the coordinator holds pending for the conversation, so the
mark (the restart staging bit) survives every deferral window -- including autowake-OFF,
which records pending by design. Cost stated honestly: an owed-but-undelivered wake
keeps its ◈ on a viewed conversation until delivery commits (with OFF, until re-enable);
the badge in that window means 'background completion not yet delivered', not merely
'unseen'. Task 4's clear-on-view is preserved and re-asserted for the nothing-owed case.
Live: with OFF + the conversation viewed, mark stayed present alongside the owed ledger
through settle+20s, survived SIGKILL, and drove the restart claim.

Tests: Tests/UI/test_console_fleet_wake_restart_staging.py (5; 4 reproduced RED against
production, 1 documents the verified limit). Mutations: 5 run, 5 killed (MR5 also died
in Task 4's pre-existing badge-clear pin). Exactly-once/ledger semantics untouched.

New findings filed from this task's live pass: task-15970 (user-wins-ties probe blind to
a live-TYPED draft) and task-15971 (leaving Console no longer stages -- resident-screen
nav keeps the controller live and the wake fires off-screen, contradicting Task 7's
verified staging and the spec's honesty line).
<!-- SECTION:NOTES:END -->
