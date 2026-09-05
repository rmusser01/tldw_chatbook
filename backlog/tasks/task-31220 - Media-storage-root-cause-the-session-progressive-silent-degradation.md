---
id: TASK-31220
title: Media storage - root-cause the session-progressive silent degradation
status: To Do
assignee:
  - '@claude'
created_date: '2026-09-03 22:30'
updated_date: '2026-09-05 06:19'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-critique 2026-09-03 P0 (user: full root-cause hunt): in a ~25-min live session, a completed review set never durably persisted (UI 'All 6 reviewed' but DB done=0/active=1/completed_at=NULL), the Sets picker failed twice then went silently dead, Read later persisted nothing, and a plain row click stuck on 'Media item is unavailable.' with dead Retry - spanning BOTH DBs. The picker worker's except Exception is the one review-set wrapper without traceback logging, so no trace exists. Lead: stuck DB thread w/ open transaction exhausting the asyncio.to_thread pool; CancelledError (BaseException) explains the silent-dead phase.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The picker worker's exception path logs the traceback
- [ ] #2 The root cause of the wedge is identified with reproduction evidence
- [ ] #3 A durable fix lands with a regression pin, and displayed state can no longer silently diverge from durable state
- [ ] #4 One storage-health surface tells the user when local storage is unhealthy
- [ ] #5 The bulk-delete receipt is derived from the service result: a write that did not land renders as a failure with a reason, never as ✓
- [ ] #6 The bulk-mutation interlock is released on every path, and Retry, row opens and select mode are never gated behind it
- [ ] #7 Undo is enabled whenever the receipt says ✓, and the receipt never says ✓ when Undo cannot be
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Instrument: log the picker worker's swallowed exception (the one wrapper #2346 missed)\n2. Reproduce: scripted long session (create/walk/complete/exit/picker/dismiss/read-later/trash/type-chooser loop) with an external DB probe between phases to catch divergence onset\n3. On wedge: kill -USR2 the app pid -> faulthandler.log all-thread stacks -> identify the stuck thread/lock\n4. Root-cause fix + regression pin + storage-health surface
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Hunt round 1 (2026-09-03): 24-round scripted repro (create->complete->picker->dismiss + read-later + trash + chooser, external RO probes every round) reproduced NOTHING - all 24 completions durably persisted (done_marks=6, completed_at stamped), every Sets press produced the modal, zero probe errors. The critique session's divergence evidence is now suspect: Assessment A ran a scratch-profile harness mid-session and its external DB dump may have read the scratch profile's collections DB (which would contain exactly the observed fresh-set shape). The in-app dead controls (Sets silent, Read later inert, media detail unavailable) remain unexplained and observed-once. AC1 SHIPPED (picker except now logs the traceback - the one wrapper 30042 missed), so the next natural occurrence self-documents in tldw_cli_app.log. AC2-4 open pending a reproducible occurrence; repro script kept at scratchpad/hunt_31202.sh (SIGUSR2 all-thread dump wired).
<!-- SECTION:NOTES:END -->

## Renumbering

Renumbered from task-31202 on 2026-09-03: id collision with an older dev arrival (owner rule TASK-19601; older keeps the id).

## Reproduction evidence (critique #5, 2026-09-04 evening, dev c09717a7cb)

Reproduced twice, independently, by two assessment agents running the app under tmux at 235x52 against the real profile — with a second app instance alive on the same profile (the app's own "Another copy of tldw is already using this profile" guard fired and was bypassed) and the host out of POSIX semaphores. Sequence: select mode → select one item → Delete → confirm → receipt `✓ deleted · 1 item · in Trash` with `○ Undo` and `Media changed; retry to load a current page.`; every row, Export, Select, sort and the pager disabled; Retry inert (two clicks, 3 s apart, byte-identical captures); Dismiss clears the receipt but not the gate; `s` inert; a Media → Notes → Media round-trip does not clear it; the DB row is untouched (`is_trash=0, deleted=0`); only killing the process recovers.

Mechanism traced in source — CORRECTED the same evening: `_delete_library_media_selection` treats "no exception" as success, which is sound for the local backend (`local_media_reading_service.delete_media_item` raises `ValueError` when `mark_as_trash` returns `False`, `KeyError` when the row is missing; `mark_as_trash` commits inside its transaction before returning `True`). A `✓` over an untouched row is therefore NOT explained by the code read. Open candidates: the assessment's own DB read on a stale WAL snapshot (long-lived `MediaDatabase` connection opened before the app's commit); the app writing through a different connection/path; a later revert. The profile-lock guard is advisory only (`_maybe_warn_second_instance`: "Everything keeps working"), so both instances shared one DB. First step of the fix PR: an instrumented reproduction (real DB, fresh read connection after the receipt) before any code change. `_complete_library_media_mutation` raises the mutation gate through `reconcile_committed_mutation` and refreshes only while the screen holds authority. `handle_library_media_retry`, `handle_library_media_row` and `_toggle_library_media_select_mode` all early-return while `_library_media_bulk_delete_in_flight` is set — exactly the three controls observed inert — so the flag not being released on that path explains the wedge. Undo is stale-gated by design, so a `✓` with a dead Undo is what the code does whenever the gate is up.

Fix shape: derive the receipt from the result (`False` = failure with a reason); release the interlock in a `finally` on every path; never gate Retry behind the interlock it exists to escape; enable Undo iff the receipt says `✓`. Snapshot: `.impeccable/critique/2026-09-05T06-05-33Z__tldw-chatbook-ui-screens-library-screen-py.md`.
