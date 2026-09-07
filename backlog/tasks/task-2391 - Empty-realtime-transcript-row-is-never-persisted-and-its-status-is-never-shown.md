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

FIX-NOW addendum (review escalation, same day): the review confirmed the AC premises above
but caught a real regression the "matches the interrupted-marker precedent" reasoning in
paragraph 3 got wrong. Because the placeholder is now real, non-blank CONTENT, it flows
through `_provider_message_payloads` (`console_chat_controller.py`) exactly like a genuine
user turn -- before this task, an empty-content row was silently dropped by that function's
`if not text: continue` guard; after task-2391's fix alone, the model would receive
`{"role": "user", "content": "(no speech detected)"}` as if the user had typed it, on every
ordinary send/retry/edit/fork/regenerate for the life of the conversation. The reseed builder
exclusion (paragraph 3) only covers the REALTIME reconnect path, not this one -- a distinct
mainline consumer the interrupted-marker analogy doesn't cover (that marker is a suffix on
otherwise-real spoken words; this placeholder IS the entire, zero-real-words content of the
turn).

Fix: added `_is_empty_transcript_row(message)` (module-level helper, `console_chat_controller.
py`, reads `message.metadata` via `getattr` so it stays safe against test doubles that
duck-type only role/content/status) and wired it into every place that walks the transcript to
build a request FOR A MODEL: `_provider_message_payloads`'s two loops (image-budget reservation
+ the `_emit` loop -- skip entirely, mirroring the existing `skip_failed` idiom, rather than
emitting empty text), `summarize_up_to`'s `before`/`span` list comprehensions (this hand-builds
a prompt sent to `_collect_summary_completion`, a REAL provider call -- so an empty-transcript
row was also silently fabricating a turn in the SUMMARIZER's context, and the "nothing to
summarize" gate needed the same exclusion so it doesn't fall through to sending an empty span),
and `impersonate_user_reply`'s hand-rolled transcript builder (its own comment already said
"mirror `_provider_message_payloads`'s rules exactly" -- it simply predates `transcript_status`
existing; found by continuing the audit, not separately requested, but the same defect class
and a one-line fix reusing the same helper, and arguably the most dangerous of the three since
this prompt explicitly asks the model to draft the user's NEXT message "in their voice" from
this exact transcript).

Export path: deliberately LEFT UNCHANGED, not silently -- audited
`_save_console_message_as_note/_media/_prompt/_chatbook` (`chat_screen.py`) and
`build_context_snapshot`'s `current_messages` field. All are human-facing: the save-as actions
operate on exactly one message the user explicitly selected and save its literal visible
content (the placeholder read there is exactly what the user saw and chose to save -- hiding it
would defeat this task's own AC#1), and `current_messages` is a raw snapshot for a UI inspector
panel, not sent to a provider (`next_send_payload`, the part that IS sent, already routes
through the now-fixed `_provider_messages_for_session`). No model-facing consumer was found
outside the three fixed above (also checked: `retry_message`/`regenerate_message`/
`edit_and_resend_message`/the cost-ticker fingerprint baseline all route through the same fixed
`_provider_message_payloads`/`_provider_messages_for_session`, so no separate fix needed there).

Tests (RED-first, each mutation-verified by temporarily reverting the guard and re-running):
`Tests/Chat/test_console_chat_controller.py::test_provider_payloads_exclude_an_empty_
transcript_placeholder` and `::test_impersonate_excludes_an_empty_transcript_placeholder`;
`Tests/Chat/test_console_rewind_summarize.py::test_summarize_span_excludes_an_empty_
transcript_placeholder` and `::test_summarize_nothing_before_target_when_only_prior_is_empty_
transcript`. Foreground-only per repo convention: covering suites (chat_controller,
rewind_summarize, composer_menu, chat_store, realtime_wiring, resume_active_path,
message_metadata) all green; contract trio (test_console_hands_free.py,
test_console_hands_free_wiring.py, test_console_dictation.py) confirmed byte-identical via
`git status` and all pass unmodified.

PR #1366 Qodo round (same day, HEAD f134ea109 -- task-2390 landed on this branch in between,
verified before editing): 5 findings, 2 real bugs fixed, 1 trivial fix, 2 judgement calls.

Q4 (BUG, fixed): `_mark_console_realtime_transcript_empty`'s early return
(`_console_realtime_row_has_text`) treated the PLACEHOLDER itself as "already has text" and
bailed -- so once the content write landed but the FOLLOWING metadata write failed (reachable
because `_set_console_realtime_transcript_status` deliberately swallows its own exceptions),
every later retry hit the same early return and could never reach the status write again. A
row stuck with placeholder content but no `transcript_status="empty"` is invisible to
`_is_empty_transcript_row` and reaches providers as a fabricated user turn -- reopening the
exact leak the prior fix-now commit closed, by a different route. Fixed by reading the row's
actual content and branching three ways: non-blank and NOT the placeholder -> return (a real
transcript, never overwritten, unchanged behavior); blank -> write the placeholder then stamp
the status; already the placeholder -> skip the (redundant) content write but still (re-)stamp
the status, so a stranded row recovers on the next event instead of being permanently stuck.
`_console_realtime_row_has_text` is now dead (its only caller) and was removed rather than left
orphaned. RED test (`Tests/UI/test_console_realtime_wiring.py::test_an_empty_transcript_
retries_the_status_after_a_swallowed_metadata_failure`) monkeypatches `store.set_message_
metadata` to raise exactly once, drives the exact partial state, and proves a SECOND empty
event reaches `transcript_status == "empty"` and that `_is_empty_transcript_row` then
recognizes the row -- failed against the pre-fix code (confirmed), passes after.

Q5 (BUG, fixed): `_is_empty_transcript_row` read `.metadata` via `getattr` (duck-type-safe) but
then read `.transcript_status` off that metadata object via a plain attribute access -- so a
metadata object missing that one attribute raised `AttributeError` inside all three model-facing
builders this helper guards (send/summarize/impersonate), i.e. a crash reachable from the main
send path, not just a narrow test double. Fixed: `getattr(metadata, "transcript_status", None)
== "empty"` (also simplifies away the now-redundant `metadata is not None` check, since
`getattr(None, ..., None)` already resolves to `None`). RED test
(`Tests/Chat/test_console_chat_controller.py::test_is_empty_transcript_row_tolerates_a_metadata_
object_without_the_attribute`) passes a `SimpleNamespace` metadata with no `transcript_status` --
raised before, returns `False` after.

Q1 (trivial, fixed): `Tests/Chat/test_console_chat_store.py`'s
`test_an_empty_transcript_placeholder_persists_through_the_deferred_create` hardcoded
`"(no speech detected)"` instead of importing `CONSOLE_REALTIME_EMPTY_TRANSCRIPT_PLACEHOLDER` --
now imports the constant, so the test can't silently drift from the real chrome text.

Q2 (JUDGEMENT, evidence-based decision -- keep the existing test on `RecordingPersistence`,
ADD a new real-DB test rather than converting): checked this file's own convention first, same
as the earlier `transaction()` precedent. `Tests/Chat/test_console_chat_store.py` uses
`RecordingPersistence` in 9 tests and a real `CharactersRAGDB`/`ChatPersistenceService` pair in
5 -- and each of the 5 carries its own docstring explaining WHY a fake can't stand in
(`test_persist_session_if_needed_flushes_held_rag_scope_through_real_db`: "not a hand-rolled
fake... end to end"; `test_stop_path_usage_flush_uses_local_write_and_leaves_version_unchanged`:
"proving the usage-only flush actually lands... rather than only through a fake that can't
observe that distinction"). My test's direct precedent/sibling,
`test_set_message_metadata_on_an_unpersisted_row_rides_the_later_create` (same file, tests the
identical "unpersisted row rides the later create" claim for the `"final"` case), already uses
`RecordingPersistence` -- converting only the `"empty"`-case sibling would make it the odd one
out against its own precedent, worse than declining outright. BUT weighing the substance
honestly: `RecordingPersistence` records exactly which kwargs `create_message`/
`update_message_content` were called with, which is what BOTH tests actually claim ("the row is
durably created" == "create_message got called with this content") -- it is not, in itself, a
version/sync_log/local-only-column nuance a fake can't represent, unlike the 5 tests above. So
the sibling stays as-is (declined). The one place Qodo's concern has real teeth: neither this
test NOR the existing real-DB coverage
(`Tests/UI/test_console_resume_active_path.py::test_resume_restores_an_empty_transcript_row_
and_its_explanation`) actually drives `update_message_content`'s deferred-create flush against a
REAL DB and reads the row back -- the resume test HAND-SEEDS the row via `db.add_message(...)`
directly, so it proves resume reconstructs an already-durable row correctly, not that the
deferred-flush mechanism itself durably writes one. Closed that specific gap with a new test,
`test_empty_transcript_placeholder_reaches_a_real_db_through_the_deferred_create`, mirroring
this file's own two cited real-DB tests: drives the real flow (deferred row -> content write ->
status write) against a real `ChatPersistenceService`/`CharactersRAGDB` pair and reads the
persisted row straight off the DB (`db.get_message_by_id`). Net effect: both of Qodo's concerns
addressed -- the established per-file convention respected, and the actual durability claim now
has real-DB coverage that didn't exist before in either file.

Q3 (DECLINED, coordinator's adjudication, recorded verbatim): Qodo wants Google-style Args/
Returns docstrings on the new zero-arg test functions. Declined -- their narrative docstrings
carry the WHY (the bug, the mechanism, the review finding they close), which is more valuable
here than Args/Returns boilerplate on a function with neither; Qodo itself rated this Moderate
with "no clear precedent... may be tolerated narrative test docstrings," and every test suite in
this repo already uses narrative-only test docstrings uniformly. Not churned.

Second foreground verification round (never background, per repo convention): `Tests/Chat/
test_console_chat_controller.py` + `test_console_chat_store.py` + `test_console_rewind_
summarize.py` (319 passed); `Tests/UI/test_console_realtime_wiring.py` (75 passed, was 74 before
the new Q4 RED test); contract trio (103 passed, confirmed byte-identical via `git status` both
before and after this round).
<!-- SECTION:NOTES:END -->
