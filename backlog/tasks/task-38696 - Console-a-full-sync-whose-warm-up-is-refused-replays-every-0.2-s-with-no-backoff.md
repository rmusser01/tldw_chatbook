---
id: TASK-38696
title: "Console: a full sync whose warm-up is refused replays every 0.2 s with no backoff"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- performance
- console
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
dependencies:
- TASK-38649
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-38649's review, PR #3372, 2026-11-03). Checked by reading the code on dev 9bbd5c2515 and at #3372's head; not measured.

A full Console sync awaits `prepare_console_presentation_sync` (config_sync.py:100-140) before it renders. That function returns False at once while the screen is maintenance-paused or closing (:127-129). Otherwise, for the stock disk-loaded projection (it returns True early for a config not loaded from disk, :131-132, or a non-standard projection, :134-138), it returns `ConsoleReadinessConfigProjection.warm()` (:140, console_spend_projection.py:462). When it returns False, the sync sets `_console_control_bar_replay_whole_sync` and asks for a delayed control-bar sync (chat_screen.py:17847-17850). That request is a fixed `set_timer(0.2, ...)` (chat_screen.py:20847), so the whole full sync replays.

`warm()` itself refuses up front only for a closing screen or a disposed runtime (console_spend_projection.py:464-467). After its read settles, it returns False when the key has moved or the projection has no fresh value.

Each replay calls `warm()` again, which calls `self.run(...)` and so can start another config read. So while a refusal lasts (the cases TASK-38649 lists: `RecoveryRequired` from a storage pause, a non-stock reader, replaced pricing sources), the screen can pay a full sync and a read about 5 times a second.

During a maintenance pause the same 0.2 s timer is also the resume path. `_run_coalesced_control_bar_sync` re-arms itself while the screen is paused (chat_screen.py:20888-20896), and `_console_sync_maintenance_resume` leaves the replay to that timer (:17665-17672). A backoff must not delay or drop that resume.

TASK-38649 bounds the projection's own retry (backoff from 0.2 s to 5 s, one shared timer). This replay path is separate, and #3372 does not touch it.

Depends on TASK-38649 (in PR #3372).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While warm-up keeps refusing, full-sync replays back off instead of repeating every 0.2 s, and a pin counts replays over a refusal of several seconds and fails on the base build
- [ ] #2 A refusal that clears still lets the next full sync render promptly (pinned), a closing screen schedules no replay, and a full sync requested during a maintenance pause still runs exactly once after resume (pinned)
<!-- AC:END -->
