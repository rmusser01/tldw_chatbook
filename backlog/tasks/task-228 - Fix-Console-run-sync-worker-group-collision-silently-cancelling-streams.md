---
id: TASK-228
title: Fix Console run/sync worker-group collision silently cancelling streams
status: Done
assignee:
  - '@claude'
created_date: '2026-07-14 10:30'
updated_date: '2026-07-15 02:05'
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
- [x] #1 Send/retry/regenerate/continue workers run in a dedicated worker group (e.g. "console-run"); UI-sync workers run in their own group (e.g. "console-sync"); no exclusive run_worker call on ChatScreen relies on the default group
- [x] #2 A vision-model streaming run completes end-to-end live: assistant reply renders, [streaming] clears, Stop/Send re-enable, and the persisted assistant row contains the full reply text (V1/V2/V3 all gone under the TASK-222 QA rig)
- [x] #3 A regression test reproduces the collision (overlapping sync requests during an active fake stream cancel the run) RED against the pre-fix code and GREEN after
- [x] #4 A guard test (lint-style) fails if any exclusive run_worker call in chat_screen.py omits an explicit group
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Guard test (AST): ban ungrouped exclusive run_worker in chat_screen.py — RED at 14 sites\n2. Textual worker-group semantics regression test (ungrouped exclusive collide; distinct groups do not — pins the fix premise on the installed textual)\n3. Fix: group='console-sync' (9 sync kicks), group='console-run' (submit/retry/regenerate/continue), group='console-save-as' (save-as dispatch)\n4. Chat/console test sweep + legacy image gate\n5. Live QA on the TASK-222 vision rig: tiff+svg sends complete end-to-end ([streaming] clears, Stop/Send re-enable, full reply persisted — V1/V2/V3 gone)\n6. User screenshot gate, then PR to dev
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (probe-confirmed on textual 8.2.7): ungrouped exclusive=True workers share Textual's default group and cancel each other — the UI-sync re-kick (chat_screen finally-block) silently cancelled the in-flight send worker whenever sync requests overlapped, which image-bearing transcripts made routine. Fix: dedicated groups on all 15 formerly-ungrouped exclusive worker sites (console-run for submit/retry/regenerate/continue, console-sync for the 9 UI-sync kicks, console-save-as for the save-as dispatch, console-library-rag-search for the @work-decorated RAG search the pre-merge review caught). Regression protection: AST guard test bans ungrouped exclusive workers in chat_screen.py covering BOTH run_worker calls and @work decorators, failing closed on non-literal exclusive= values (RED at 14 sites pre-fix; extension proven RED against the pre-fix decorator source); a group-disjointness test pins that the sync kicks never share the run workers' group; semantics tests pin the collide/don't-collide Textual behavior. AC #3 honesty note (pre-merge review Important #3): the shipped reproduction is lexical (AST guard, RED pre-fix) plus Textual-semantics probes, not a full ChatScreen fake-stream harness — live QA covered the behavior end-to-end (8/8 sends finalized incl. 129s think, 0 no-token stalls vs prior ~50%, full DB persists, within-session multi-send) and the disjointness test closes the sync-kick-in-console-run regression hole the review identified; a screen-level behavioral harness was judged not worth its weight on top of that evidence. Review Important #2 (mid-run retry/regenerate/continue cancels the stream before the controller gate rejects it — pre-existing, recoverable via Stop) filed as task-232. Evidence: Docs/superpowers/qa/console-run-groups-2026-07/. Files: tldw_chatbook/UI/Screens/chat_screen.py, Tests/UI/test_chat_screen_worker_groups.py.
<!-- SECTION:NOTES:END -->
