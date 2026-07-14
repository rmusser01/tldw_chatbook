---
id: TASK-228
title: Fix Console run/sync worker-group collision silently cancelling streams
status: To Do
assignee: []
created_date: '2026-07-14 10:30'
labels:
  - console
  - bug
  - streaming
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-222's vision QA exposed a latent defect (probe-confirmed on Textual 8.2.7, root-caused via systematic debugging): the Console send worker (`chat_screen.py:7279`) and the UI-sync re-kick (`chat_screen.py:7071`) both call `run_worker(..., exclusive=True)` with NO `group=`, so they share Textual's default worker group — and an exclusive worker starting cancels every other worker in its group. When sync requests overlap (which happens exactly when the transcript carries inline image rows, making syncs slower than the 0.2s transcript timer), the re-kick silently cancels the in-flight send worker. Cancel timing produces three observed faces: mid-think → no token/no DB row/no error (~half of vision sends); mid-content → assistant DB row frozen at last incremental write (observed: 7-char "The ima") + run_state stuck STREAMING ([streaming] suffix, Stop button, disabled Send persist forever); post-persist → complete DB row but stuck UI. Text-only runs never overlap syncs, so the bug hid until a vision-capable model (mmproj) made successful image streams reachable. Blast radius beyond vision: ALL ungrouped exclusive workers on ChatScreen collide — sync kicks fire from ~9 more call sites (672/898/1125/1857/5421/6876/8522/9889) and retry/regenerate/continue run ungrouped too (8060/8064/8105), so tab switches or store events mid-run can kill any response. Full evidence: Docs/superpowers/qa/console-config-caps-2026-07/README.md (defects V1–V3) and the TASK-222 session ledger.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Send/retry/regenerate/continue workers run in a dedicated worker group (e.g. "console-run"); UI-sync workers run in their own group (e.g. "console-sync"); no exclusive run_worker call on ChatScreen relies on the default group
- [ ] #2 A vision-model streaming run completes end-to-end live: assistant reply renders, [streaming] clears, Stop/Send re-enable, and the persisted assistant row contains the full reply text (V1/V2/V3 all gone under the TASK-222 QA rig)
- [ ] #3 A regression test reproduces the collision (overlapping sync requests during an active fake stream cancel the run) RED against the pre-fix code and GREEN after
- [ ] #4 A guard test (lint-style) fails if any exclusive run_worker call in chat_screen.py omits an explicit group
<!-- AC:END -->
