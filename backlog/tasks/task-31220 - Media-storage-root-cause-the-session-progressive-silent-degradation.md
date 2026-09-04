---
id: TASK-31220
title: Media storage - root-cause the session-progressive silent degradation
status: To Do
assignee:
  - '@claude'
created_date: '2026-09-03 22:30'
updated_date: '2026-09-03 22:44'
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
