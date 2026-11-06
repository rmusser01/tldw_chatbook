---
id: TASK-38994
title: "Four mounted Console send-acknowledgement tests are red on dev, and a send refused during admission keeps Sending… painted"
status: To Do
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
- [ ] #1 test_refused_admission_rolls_the_sending_row_back_and_keeps_the_draft, test_turn_refused_before_its_echo_releases_the_row_and_offers_recovery, test_second_enter_during_admission_never_starts_a_second_provider_call and test_failed_durable_commit_releases_the_row_and_restores_send pass 10 of 10 runs alone on dev and in a whole-file -n 4 run
- [ ] #2 The notes record, for each of the four, whether the cause was test drift (for example the draft commit moving in 077e5bfa09) or a product defect
- [ ] #3 A send refused during admission (accept_turn raising, or a hook-admission refusal) leaves no Sending… in the transcript or the status line; if this was a product defect, a test shows it red before the fix. Open TASK-38426 has the same symptom on a different path: a send-gate refusal at ui_dispatch, before admission, which 14cd6d2ff4 may already fix. The refusals here happen during admission and still leave Sending… on dev, which already contains 14cd6d2ff4; the notes say whether one fix closes both
- [ ] #4 Each of the four still fails when the behaviour it pins is removed (the refused row is not released, the draft is not taken out, a second Enter starts a second provider call), shown under temporary source edits
<!-- AC:END -->
