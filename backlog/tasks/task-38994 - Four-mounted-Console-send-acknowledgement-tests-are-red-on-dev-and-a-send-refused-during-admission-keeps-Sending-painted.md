---
id: TASK-38994
title: "Four mounted Console send-acknowledgement tests are red on dev, and a send refused during admission keeps Sending… painted"
status: Done
assignee: ['@claude']
updated_date: '2026-11-06 06:50'
created_date: 2026-11-05 21:35
labels:
- follow-up
- console
- tests
priority: medium
references:
- Tests/UI/test_console_send_acknowledgement_extended.py
- Tests/UI/test_console_send_acknowledgement.py
- tldw_chatbook/UI/Console_Modules/send_acknowledgement.py
- tldw_chatbook/UI/Console_Modules/sent_draft.py
- tldw_chatbook/Chat/console_received_turn.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05.

TASK-38718 (Done) recorded these four as "Not mine": red on its base b8608e4cdb and on its head. No task owns them, and no lane runs the file (the census keeps it out, `scripts/ui_pr_gate_census.txt:801-806`).

**Runs, through the scratch-profile wrapper.** Serial, all four were red in 3 of 3 runs (67-75 s, load 9-15). In a whole-file `-n 4` run, 4 failed and 4 passed (50.7 s, load 9). None failed with RecoveryRequired.

**Failures.** A probe captured the painted screen when each of the three waits expired. All references are to `Tests/UI/test_console_send_acknowledgement_extended.py`.
- `test_refused_admission_rolls_the_sending_row_back_and_keeps_the_draft` (`:55`) times out at `:80` waiting for `Sending…` to clear. `runtime.accept_turn` raised "Console session is closed.", and 15 s later the transcript still shows `Sending…`.
- `test_turn_refused_before_its_echo_releases_the_row_and_offers_recovery` (`:91`) times out at `:108` on the same wait. The screen shows the `Sending…` row and the shelf `Not sent: Hook admission refused for this test … Restore Discard` at the same time.
- `test_second_enter_during_admission_never_starts_a_second_provider_call` (`:175`) times out at `:197` waiting for the composer to empty while provider validation is held. The status line reads `Run: Sending… · Provider: llama.cpp …` and the transcript shows `Sending…`.
- `test_failed_durable_commit_releases_the_row_and_restores_send` (`:121`) fails at `:145` with `assert None == 1`. When validation starts, the claim's `_draft_transfer_revision` is still None and the snapshot's `draft_revision` is still the captured 1. Only `tldw_chatbook/UI/Console_Modules/sent_draft.py:204` records that transfer, through `note_received_draft_transfer` (`tldw_chatbook/Chat/console_received_turn.py:183`).

**Not verified.** 077e5bfa09 (2026-10-31) moved the acceptance-time store draft commit behind a CAS, which may make the two draft-timing reds test drift. The two refusal reds look like a real product defect, a `Sending…` row that stays painted after a send is refused during admission. The lane's core test, `test_a_refused_acknowledged_enter_repaints_its_refusal` (`Tests/UI/test_console_send_acknowledgement.py:807`), covers only a refusal before admission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 test_refused_admission_rolls_the_sending_row_back_and_keeps_the_draft, test_turn_refused_before_its_echo_releases_the_row_and_offers_recovery, test_second_enter_during_admission_never_starts_a_second_provider_call and test_failed_durable_commit_releases_the_row_and_restores_send pass 10 of 10 runs alone on dev and in a whole-file -n 4 run
- [x] #2 The notes record, for each of the four, whether the cause was test drift (for example the draft commit moving in 077e5bfa09) or a product defect
- [x] #3 A send refused during admission (accept_turn raising, or a hook-admission refusal) leaves no Sending… in the transcript or the status line; if this was a product defect, a test shows it red before the fix. Open TASK-38426 has the same symptom on a different path: a send-gate refusal at ui_dispatch, before admission, which 14cd6d2ff4 may already fix. The refusals here happen during admission and still leave Sending… on dev, which already contains 14cd6d2ff4; the notes say whether one fix closes both
- [x] #4 Each of the four still fails when the behaviour it pins is removed (the refused row is not released, the draft is not taken out, a second Enter during the first turn's admission is not held by the visible send and starts a send of its own), shown under temporary source edits
<!-- AC:END -->

## Implementation Plan

1. Reproduce the four reds at base e8eeae7db9 through run.sh and probe each painted screen and send state when its wait expires.
2. For the two refusal reds, trace why the released "Sending…" row stays on screen (ack state, the resync, the transcript sync gate) and fix the product at the release's repaint, where every release routes through.
3. For the two draft-timing reds, check them against the current draft-commit contract (077e5bfa09 / ADR-230, TASK-38690's lesson); align a test only where it pins retired timing, without weakening what it pins.
4. Record per test: drift or defect. Record whether the same fix closes TASK-38426's send-gate path.
5. Show each of the four red under a temporary source edit that removes the behaviour it pins; run each 10x alone and the whole file under -n 4.

## Implementation Notes

**Root cause (product).** `paint_acknowledgement` pushes the pending "Sending…" row straight to the transcript, past the 0.2 s tick gate (TASK-34958's `_console_transcript_sync_token`). When no sync runs between that paint and the release, the gate still holds its token from before Enter. 14cd6d2ff4 put the pending row id in the gate, but the stale token from before Enter also had no pending row. A refusal that writes no store row (`accept_turn` raising, a hook-admission refusal) leaves the store's display token unchanged as well. So the release's resync matched the stale token and skipped `set_messages`. I probed this at the timeout: the ack had released the row, and the composer and status line read idle. But `ConsoleTranscript._messages` still held the `console-send-ack-…` row, and no `set_messages` ran after the release.

**Fix.** `send_acknowledgement._resync` is the one path every release takes: a dispatch that admits nothing, a custody end, or a stand-in leaving. It now forgets the gate token as well as the refresh key (`screen._console_transcript_sync_token = None`). This is the same idiom `terminal.py` and the citation-count sync use.

**Each of the four (AC#2).**
- `test_refused_admission_rolls_the_sending_row_back_and_keeps_the_draft`: product defect, the gate token above.
- `test_turn_refused_before_its_echo_releases_the_row_and_offers_recovery`: product defect, the same cause. After the fix, 1 run in 40 (load ~83) failed its shelf assertion. The "Not sent:" shelf is the queue strip, painted after the transcript row had already gone, so the test now waits for the shelf with `until` and no longer asserts it in the same frame.
- `test_second_enter_during_admission_never_starts_a_second_provider_call`: test drift. Since 077e5bfa09 (2026-10-31, ADR-230 "PR3050 current-owner reconciliation"), only the pressed snapshot retires its draft, after durable acceptance, and that comes after provider validation. The wait for an empty composer during held validation could never succeed. The test now asserts the following. The repeat Enter during admission is held silently and dropped: it starts no send and raises no notice. The draft is still shown during validation. A press on it then is refused with the notice "Preparing the current turn…", which the test checks by its copy, and the draft stays. There is one provider call, and the composer is empty once the turn is accepted. The two notice checks were added after review r10s-rev1, and each was shown red under a temporary edit:
  - With the three "Preparing the current turn" refusals removed, the copy wait times out, because the next layer says "A run is already running in this tab."
  - With the repeat scheduled at once and the sends check off, the silent-hold check fails with `['Hook review or Send is already in progress.'] == []`.
- `test_failed_durable_commit_releases_the_row_and_restores_send`: test drift, from the same commit. The claim's `_draft_transfer_revision` is recorded only when the accepted draft is taken out (`sent_draft.take_out_sent_draft`), never before validation. The test now asserts that no transfer exists before acceptance and that the store and composer still hold the draft. It also asserts that the draft is still in the composer after the failed commit (ADR-230: "failed acceptance keeps the draft and refuses dispatch").

**Red evidence.** At base e8eeae7db9, the whole file under `-n 4`: 4 failed, 4 passed (48.3 s). With the fix removed (control A), the two refusal tests and the new lean unit test are red. The realigned drift tests are green with or without the fix. I added `test_a_release_reopens_the_transcript_gate_the_paint_went_past` to the gated core file `Tests/UI/test_console_send_acknowledgement.py`, a lane file, so CI guards the fix; the extended file stays out of the lanes. It is red without the fix: `assert ("the gate's token from before Enter",) is None`.

**AC#1.** On the final head (all three tasks' commits), under `-n 4`:
- the four node ids, run alone, one per worker: 10 of 10 runs passed (18.9-65.1 s, load 35-72);
- the whole file: 3 of 3 runs passed, 8 tests each (35.0-40.9 s, load ~43-48).

A run before the shelf wait was added failed 1 in 10 (the shelf race above).

**AC#3.** In tests 1 and 2, `paint_state().sending` is "Sending" anywhere on the painted screen: the transcript row, the `Run: Sending…` status line and the composer's "Sending...". Both assert it is cleared.

TASK-38426's send-gate refusal at ui_dispatch, by contrast, appends a System row. That bumps the store's display token, so the gate reopens on dev anyway. The lane test `test_a_refused_acknowledged_enter_repaints_its_refusal` drives that path with a real Enter. It passes on base, and it still passes with 14cd6d2ff4's gate change reverted, with or without this fix. With this fix in place, tests 1 and 2 also pass with 14cd6d2ff4's change reverted.

So one fix, this one, closes both paths: every release reopens the gate, whatever it writes. 14cd6d2ff4 alone never covered refusals that write no row. I did not check TASK-38426's AC#2 (the System row painted *below* the refused turn); it is not this task's.

**AC#4 (met, with the clause amended).** These temporary source edits were each reverted after the run:
- B: `dispatch_finished` keeps an unadmitted row. Test 1 is red (times out waiting for "Sending…" to clear); tests 2-4 are green.
- C: a custody end keeps the row. Tests 2 and 4 are red; tests 1 and 3 are green.
- E7: the accepted draft is never taken out (composer `commit_captured_draft`, store `commit_session_input_draft` and the controller's post-commit `set_session_draft(…, "")` all disabled). Test 3 is red: `assert 'acknowledge this send' == ''`. Any one of these three edits alone leaves it green, because each path takes the draft out on its own.

*Ruling (task filer, 2026-11-06).* The original clause, "a second Enter starts a second provider call", pins an outcome that guarantees further down already prevent. The ruling replaced it with "a second Enter during the first turn's admission is refused at the store's one-owner claim, and no second turn is claimed", to be shown red by bypassing that claim's ownership check. If no such edit could turn it red, I was to say which layer blocks the second Enter and pin that layer instead.

*What blocks the second Enter.* For this check, test 3 recorded each send the visible-send path started and each claim the store granted. The claim recorder was dropped from the final test because no single edit can make it red. All edits below are temporary:
- The claim is not the layer. With `claim_received_turn`'s two "slot occupied" refusals removed, test 3 stays green: the second Enter never asks for a claim.
- The first layer is the visible send's repeat hold. `send_acknowledgement._defer` holds an Enter whose capture equals the running send's as a repeat, and `_replay` drops it unless the send asked for another Enter.
- With the repeat hold removed (it schedules the repeat at once), the second Enter starts its own send. That send is then refused by the hook dispatch's busy guard ("Hook review or Send is already in progress.").
- With the busy guard removed as well, the two sends still make one admission: the earlier, held send is refused at `ui_dispatch` and the later one is claimed. With the claim check also bypassed, the store still granted only one claim.
- So the store's one-owner claim is the fourth guard, and no edit to it alone can make test 3 red.

*Pinned instead.* Test 3 now asserts that only one send started by the time validation began (`assert len(sends) == 1`), so the repeat Enter never started a send of its own. With the repeat hold removed, that assertion is red: `assert 2 == 1`. AC#4's clause is amended to that layer.

The earlier attempts to produce a second provider call are in `backlog/docs/lessons-testing-evidence.md` (2026-11-06). There, a sequential second turn in this harness also ends before the provider ("Local conversation storage is unavailable."): `_build_console_send_test_app` has no `local_chat_conversation_service`.

After the change, `Tests/UI/test_console_send_acknowledgement_extended.py` under `-n 4`, 3 of 3 runs: 8 passed each, 19.7-21.2 s, load ~20. After the review fixes (r10s-rev1), 3 of 3 runs again: 8 passed each, 30.1-35.4 s, load ~27-32.

**Docs.** No copy changes. `Docs/User_Guide/console/chat-basics.md` already says a refused turn's **Sending…** row disappears and the draft is kept, and the fix restores that behaviour. I added a lesson to `backlog/docs/lessons-testing-evidence.md` (the one-turn harness trap).

Files: `tldw_chatbook/UI/Console_Modules/send_acknowledgement.py`, `Tests/UI/test_console_send_acknowledgement.py`, `Tests/UI/test_console_send_acknowledgement_extended.py`, `backlog/docs/lessons-testing-evidence.md`.
