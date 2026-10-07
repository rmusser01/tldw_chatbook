---
id: TASK-33802
title: Census typing-burst helper-spawn flake under concurrent I/O load
status: In Progress
assignee:
- '@claude'
created_date: 2026-10-02 18:25
labels:
- perf
- testing
- flaky
dependencies: []
priority: low
references:
- https://github.com/rmusser01/tldw_chatbook/pull/2968
modified_files:
- Tests/Performance/test_console_keystroke_work_census.py
updated_date: 2026-10-03 19:35
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console storage-unit census (Tests/Performance/test_console_keystroke_work_census.py::test_console_storage_units_stay_within_their_ratchets) sometimes fails with "typing (whole burst) helper_spawns: 1 > ceiling 0" (storage_admissions 3 in the same burst): some private-SQLite open lands inside the 40-keystroke window, which holds the credential-poll and draft-spend timers still.

Evidence gathered 2026-10-02 while fixing TASK-33801:
- 7 of 24 runs at dev 52620c3a08 (and TASK-33801's branch on it) failed this way, every one while a parallel pytest -n 6 sweep was running on the same machine.
- 0 of 46 runs at earlier commits (92a95170a5 through 30ca4552b3), measured without a concurrent sweep.
- Interleaved A/B under four CPU burners (yes > /dev/null): 0/8 at 30ca4552b3, 0/8 at 52620c3a08. 0/10 with per-spawn stack instrumentation.
So the cause is load-dependent and not shown to be any one commit; CPU load alone does not reproduce it, concurrent I/O-heavy test sweeps did.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The code path of the helper spawn billed to the typing burst is identified from a captured stack (record the caller and thread of every HelperLease.start while counting["phase"] is the burst, under a concurrent pytest -n 6 sweep)
- [x] #2 Either that work is moved off the keystroke path, or the census holds it still for the burst like the credential poll and draft-spend refresh, with the reason recorded; no ceiling is raised
- [ ] #3 A real-timer regression proves startup cleanup is held without suppressing other timers, and its separately billed phase proves one completed real candidate query and a counted cold SQLite helper.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the caller of every storage admission and helper spawn billed to the typing burst under load
2. Move that work off the keystroke path, or hold it still for the burst
PR2968 approved bounded follow-up: retain incoming caller/GC instrumentation; verify census-only media cleanup timer pause against actual Textual regression and both private-profile realSQLite variants; retain separate reported maintenance cost and all existing ceilings. Full historical-flake attribution/concurrent full-suite evidence is not claimed.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Not reproduced on 2026-10-03, so AC#1 stays open. Attempts, each with per-spawn stack capture: 12 census runs on dev `2612fc56b2` beside a concurrent `pytest Tests/Chat -n 6` sweep (load average 30-39); 2 runs with the burst stretched to ~10 s (0.25 s between keystrokes), which would catch any periodic work; 6 runs at `ecc0a531c8`, where it had failed 4 of 6 the night before. None billed a storage admission or helper spawn to the burst. Whatever fired depended on that evening's machine state, not on load alone or on a commit.

So the census now names the culprit itself the next time: every storage admission and helper spawn billed to the burst records its thread and innermost app frames (`_TYPING_BURST_CALLERS`), and a typing-ceiling failure prints them ("Typing-burst callers: ..."). Checked with a forced private-SQLite open mid-burst: the message lists `acquire_storage <- raw_participants._scope <- config_participants.operation <- ...`. The next failure (local or perf-guard) gives AC#1's stack; AC#2 then follows from it.
PR2968 follow-up (human approved): isolated d294 untested census failed naturally with typing helper1>0 and3storage admissions. Latest9b28 controlled real media cleanup callback scheduled0.01s into typing reproduced helper1>0 (2admissions138opens). Captured HelperLease.start caller on asyncio_0: app_lifecycle.run_cleanup_method -> Client_Media_DB_v2.get_deletion_candidates -> execute_query/get_connection -> private_sqlite.prepare_in_helper. This is controlled attribution, not a claim all historical failures have one cause. Temporary instrumentation removed. Adapter keeps actual Textual startup cleanup timer paused and awaits actual callback in separately reported real-DB cleanup phase, retaining all original budgets/canaries; deterministic real-timer regression red without pause, green with pause. Both actual mounted variants and approved tooltip marker pass (4cases131.52s), no real-profile refusals. Incoming PR2969 a56003663 independently added caller reporting and first GC-pass census: preserved in integration, including incoming owner's notes and acceptance criteria. Latest-base qualification pending.
Final integrated a560 qualification:227passed2platform-specificskips3headroomwarnings333.80s; bothprivate-profileSQLitecensusvariants includeincomingGCphase/callerreporting andcomplete exactly1realcandidatequery withcoldhelpercounted. Querycompletion assertionredas0beforecall-throughcounter,greenafter; noresponsesubstituted. Separatelyreportedmaintenancecost visiblewithactualguards. IndependentfinalstaticreviewReady/noactionablefindingsafterP2counterrepair. Alloriginalceilings/floors unchanged. OriginalAC1's historical/concurrent-load attribution remainsopen: controlledmedia-cleanup attributiondoesnotprove everyolderfailurehadthiscause. Incomingowner's caller instrumentation remainsinplaceforfuture failures.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved bounded startup-media-cleanup census correction complete and qualified on a56003663. Real startup timer held; callback independently billed with completed-query/helper proof. Historical general-flake AC1 remains open, without a full-sweep or universal root-cause claim.
<!-- SECTION:FINAL_SUMMARY:END -->
