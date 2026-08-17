---
id: TASK-16199
title: 'TTS shutdown sweep can still zero an in-flight export source'
status: Done
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - bug
  - concurrency
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-15471's fix round made TTS export and the 5s per-file cleanup mutually exclusive: the export claims the message id under `_audio_files_lock` and `_cleanup_audio_file` defers its secure-delete (which ZEROES the file in place before unlinking, `Utils/secure_temp_files.py:216-224`) while a claim is held. The re-review verified the claim logic on every path but found the one uncovered window (its N1): `cleanup_tts_resources`'s shutdown sweep (`Event_Handlers/TTS_Events/tts_events.py:2728-2742`) deletes every artifact WITHOUT consulting `_exporting_audio_refcounts`, and cancelling the export task does not stop its already-running pool thread — so quitting the app mid-copy can still produce a zeroed export file. Present before TASK-15471 too (the window was just narrower). Fix direction: make the shutdown sweep honor the claims (skip-and-log or bounded-wait), and add the one-sentence bound rationale to the cleanup poll loop the review asked for (its N2). Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The shutdown sweep cannot destroy a file an export currently claims (test or forced-interleave probe as evidence)
- [x] #2 Shutdown still completes promptly when an export hangs (bounded wait or skip, stated)
- [x] #3 The cleanup poll loop's termination bound is documented at the loop
<!-- AC:END -->

## Implementation Plan

1. Born-red forced-interleave test in `Tests/TTS/test_tts_improvements.py`: gate `shutil.copy2` (same pattern as the task-15471 test), start an export registered through `_add_active_task` exactly as production's `on_tts_export_event` does, fire `cleanup_tts_resources()` mid-copy, assert the source survives un-zeroed and the export output carries the real payload. Must fail against current code.
2. Fix in `cleanup_tts_resources` (`Event_Handlers/TTS_Events/tts_events.py`): snapshot `_exporting_audio_refcounts` BEFORE cancelling active tasks — the cancel makes each export's `finally` release its claim while the pool thread keeps copying, so sweep-time refcounts alone under-report in-flight copies. At the sweep, skip-and-log any file whose message id is in that snapshot OR in the live refcounts (the union also covers an export admitted after the cancel pass). Apply the same skip to the `_artifact_cleanup_retry` pass by path. SKIP, not bounded-wait: shutdown stays O(0) on export duration; the cost is leaking at most the in-flight exports' temp files at quit.
3. AC #3: one-sentence termination-bound comment at `_cleanup_audio_file`'s claim-poll loop (why a claim always clears: the export's `finally` runs even under shutdown cancellation because releasing only needs `_audio_files_lock` and `Lock.acquire()` on a free lock has no suspension point).
4. Run the new test (red then green), the TTS test file, ruff check + format on touched files.

## Implementation Notes

Extended the task-15471 claim invariant to the shutdown sweep in `cleanup_tts_resources` (`tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`) — no change to the claim mechanism itself.

- **The subtle half of N1**: a sweep that consulted `_exporting_audio_refcounts` only at sweep time would still miss the window, because the sweep's own cancel pass makes each cancelled export's `finally` release its claim while the copy keeps running on a pool thread (cancellation cannot stop a thread). Fix: snapshot the claims BEFORE cancelling active tasks, then at the sweep skip-and-log any id in that snapshot ∪ the live refcounts (the union covers an export admitted after the cancel pass, which was never cancelled and still holds its claim). The retry-set pass (`_artifact_cleanup_retry`) skips the same files by path — it is part of the same shutdown sweep and runs the same zero-in-place delete.
- **SKIP, not bounded-wait (AC #2)**: shutdown never waits on an export thread at all — the cancel is immediate and claimed files are skipped, so shutdown latency is independent of copy duration (a hung copy included). A bounded wait would add per-export shutdown latency and still need the skip as its fallback. **Trade-off**: at most the temp audio files of exports in flight at the moment of quit are left on disk un-shredded (ordinary OS temp-dir hygiene applies); that beats either blocking shutdown on a possibly-hung copy or shipping zeroed exports under a success toast.
- **AC #3**: added the termination-bound sentence at `_cleanup_audio_file`'s claim-poll loop (review N2): every claim is released by `handle_tts_export`'s `finally`, which completes even for a cancelled export because the release only needs `_audio_files_lock` and `Lock.acquire()` on a free lock has no suspension point for a further cancellation to land in; shutdown additionally cancels the cleanup task outright.
- **Born-red evidence**: new forced-interleave test `test_shutdown_sweep_skips_a_source_an_export_is_still_copying` (`Tests/TTS/test_tts_improvements.py`) gates `shutil.copy2`, registers the export via `_add_active_task` exactly as production's `on_tts_export_event` does (so the cancel-releases-the-claim interleaving is real), fires `cleanup_tts_resources()` mid-copy, and asserts the source survives un-zeroed, shutdown returned promptly, and the finished copy carries the real bytes. Failed on unfixed code at exactly `AssertionError: shutdown sweep destroyed the source mid-copy`; passes after.
- **Tests**: `Tests/TTS/test_tts_improvements.py` 26 passed; full `Tests/TTS/` 4075 passed / 6 failed — the identical 6 fail at the untouched base commit `573de5dd0` (verified in a throwaway baseline worktree; unrelated subsystems: audio_cpp guided-text readiness, OpenAI backend key seam, request-admission publication). `ruff check` clean on both touched files; `ruff format` diff on `tts_events.py` is pre-existing whole-file churn (base fails format-check too) and none of its hunks touch the added lines.
- Files: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`, `Tests/TTS/test_tts_improvements.py`.

**Review scope note (post-review, controller):** the independent review confirmed the fix and
proved the one uncovered window (export admitted between snapshot and cancel pass) unreachable
by construction — but found the hazard is LATENT, not user-facing today: `TTSExportEvent` is
never constructed or posted in production, and `TTSEventHandler` is not a MessagePump, so the
export path is currently test-only. The fix pays forward for when export is wired. Review
follow-ups to file: `_discard_tts_artifact` still ignores claims (the last unguarded
secure-delete path); the claim should key by PATH not message id (the id→path join breaks when
the cache entry is evicted); wire-or-retire the export path; a test pinning the union's live
half. The AC#3 comment's shutdown-cancellation clause was trimmed to the primary bound.
