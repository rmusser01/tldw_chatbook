---
id: TASK-2391
title: Empty realtime transcript row is never persisted and its status is never shown
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05'
updated_date: '2026-08-05 06:39'
labels:
  - realtime
  - console
dependencies: []
priority: medium
---

## Description (the why)

task-2364 added `MessageMetadata.transcript_status`, and the realtime wiring stamps
`"empty"` when a committed turn produces no transcript. Two gaps remain, found by that
task's review (findings F3):

1. The row it stamps has empty content, and the store defers persistence for content-less
   rows — so the status exists only in memory and is lost on restart.
2. Nothing reads `transcript_status` anywhere. The person looking at the screen still sees
   an unexplained blank row, which was the third consequence task-2364's description set
   out to close.

The data model can now express "why this row is empty"; the user still cannot see it.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A committed voice turn that produced no transcript is visible to the user as an explained state (not a silently blank row), in the transcript itself.
- [x] #2 That explanation survives a restart, or the row is not created at all — pick one and say which in the notes; a row that exists only until restart is not acceptable.
- [x] #3 transcript_status has at least one real consumer, or is removed as dead weight.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the DB layer's hard constraint (CharactersRAGDB.add_message refuses a row with neither content nor image_data), which rules out a metadata-only persisted row -- the explanation must live in real, non-blank message content to be durable.
2. Design choice: write a short human-readable placeholder ("(no speech detected)") into the row's content via the SAME update_message_content path the "final" transcript already uses (this flushes the store's deferred-persistence guard for content-less rows exactly like a real transcript does), then stamp transcript_status="empty" after the content write succeeds -- mirrors the existing interrupted-marker pattern (chrome baked into content, metadata as the machine-readable fact).
3. Give transcript_status a real consumer: the realtime reseed builder (_console_realtime_seed_items) must skip rows whose transcript_status == "empty" so the placeholder text is never replayed into a reconnect's model context as if the user said it.
4. RED: write/extend tests first -- wiring test asserting the placeholder content + persistence flush trigger, a console_chat_store test proving the deferred row flushes with the placeholder + "empty" status (RecordingPersistence fake), a resume test proving a durably-written row round-trips through a real CharactersRAGDB, and a reseed-skip test.
5. GREEN: implement _mark_console_realtime_transcript_empty in chat_screen.py, wire it into _on_console_realtime_input_transcript's empty branch, add the CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER constant, and the reseed skip check.
6. Run the contract trio unchanged + covering suites; update task file AC boxes and Implementation Notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design choice (AC#2): the explanation SURVIVES A RESTART, not "row not created at all".
Reasoning: CharactersRAGDB.add_message hard-refuses a row with neither text nor image_data
(InputError), so a metadata-only "empty" record could never durably exist regardless of the
Console store's own deferred-persistence guard -- the explanation has to live in real,
non-blank message CONTENT to persist at all. Given that constraint, and that AC#1 requires
the turn be visibly explained "in the transcript itself" (not just transiently), durable
placeholder content was the only design that satisfies both AC#1 and AC#2 together; silently
dropping the row would leave the user unsure whether the app heard them, which is the exact
problem task-2364 set out to fix.

Implementation: a committed voice turn whose transcript resolves empty now writes
CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER ("(no speech detected)") as the row's real
content via update_message_content -- the SAME call the "final" (real transcript) branch
already uses, so it flows through the store's existing deferred-persistence flush
(_persist_pending_message_if_ready) with no store-level changes needed. transcript_status is
set to "empty" AFTER the content write succeeds (mirrors the "final" branch's own ordering:
a status without its content landed would be a lie). The row renders through the existing
message-body path unchanged -- no new widget, matching the "Generating…" streaming-placeholder
precedent already in console_transcript.py.

transcript_status's real consumer (AC#3): the realtime reseed builder
(_console_realtime_seed_items) now skips any row with transcript_status == "empty" before
building a reconnect's seed items -- the placeholder is UI chrome, not user words, and must
never be replayed into the model's context as if it were. (The interrupted-marker precedent
already accepts this same chrome riding into the ordinary skip_failed text-provider context
builder unstripped, so no change was made there -- out of this task's scope.)

New/changed logic: tldw_chatbook/UI/Screens/chat_screen.py -- added
CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER; added _mark_console_realtime_transcript_empty
(idempotent/race-safe via the existing _console_realtime_row_has_text guard, same as the
late-final-transcript case); wired it into _on_console_realtime_input_transcript's empty
branch (replacing the old metadata-only, never-persisted call); added the transcript_status
skip check to _console_realtime_seed_items.

Tests: RED-first across three layers proving persistence end to end --
Tests/UI/test_console_realtime_wiring.py (updated the empty-transcript assertion to the
placeholder content; added a double-mark idempotency test and a reseed-exclusion test),
Tests/Chat/test_console_chat_store.py (store-level proof the deferred row flushes through
update_message_content with the placeholder + "empty" metadata_json, RecordingPersistence
fake), Tests/UI/test_console_resume_active_path.py (real CharactersRAGDB round trip proving
a durably-written empty-transcript row survives resume with content + status intact).
Contract trio (test_console_hands_free.py, test_console_hands_free_wiring.py,
test_console_dictation.py) left byte-identical -- confirmed via git status, and all three
pass unmodified. Full targeted run: 435 passed, 0 failed (contract trio + covering suites +
message_metadata + native_transcript render tests).

Related but out of scope: the "failed" transcript_status branch (a content write that raises)
has a similar-shaped unpersisted-in-memory edge once content is mutated before the exception;
not touched here since it is a distinct failure mode not named in this task's AC.
<!-- SECTION:NOTES:END -->
