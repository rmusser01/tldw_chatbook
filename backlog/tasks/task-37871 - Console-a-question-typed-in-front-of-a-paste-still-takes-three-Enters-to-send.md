---
id: TASK-37871
title: 'Console: a question typed in front of a paste still takes three Enters to send'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-30 01:36'
updated_date: '2026-10-31 03:35'
labels:
  - console
  - composer
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Under the TASK-37800 ruling (owner, 2026-10-29) a caret right before a collapsed token counts as on it. A user who pastes first, presses Home and types a question in front leaves the caret right before the token, so Enter asks 'Expand pasted text (N characters)?' instead of sending, and the draft takes three Enters (ask, expand, send). A review probe on the TASK-33622.1 branch head showed 0 sends and 'Q: Expand pasted text (199 characters)? Enter: yes · Esc: no'. This follows the ruling's words, but it is the original P1 symptom with the question typed in front; every flow the ruling names ends with the caret after the token. No test pins this flow either way. Needs an owner ruling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner has ruled whether Enter sends or asks to expand when the caret sits right before a token because text was just typed in front of it, and the ruling is recorded in this task
- [x] #2 A Pilot test on the real ChatScreen pins the ruled outcome of paste, Home, type a question, Enter (one send with the full text, or the named expand question)
- [x] #3 If the ruling changes behaviour, Docs/User_Guide/console/chat-basics.md 'Large pastes' states it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Owner decision 2026-10-30: Enter always sends, so paste, Home, type a question, Enter sends once with the full text.
2. Pilot test on the real ChatScreen (test_one_enter_sends_wherever_the_caret_is[question_in_front_of_paste]), red on the pre-change head with the expand question shown.
3. Implemented together with TASK-37870 (Enter never steps a token; Ctrl+E expands); User Guide Large pastes states it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner ruling (2026-10-30), recorded for AC#1.** ENTER ALWAYS SENDS: Enter sends when the caret sits right before a token because a question was typed in front of it, and wherever else the caret is; Enter never expands a token. A dedicated key, Ctrl+E (TASK-37870), expands one, and a click still does.

**Implementation.** Shared with TASK-37870: ChatScreen's Enter no longer steps a collapsed token (`activate_focused_paste_token` is gone), so Enter goes straight to the send path; the composer expands a token only on Ctrl+E or a click.

**Test (AC#2).** Tests/UI/test_console_composer_token_enter.py::test_one_enter_sends_wherever_the_caret_is[question_in_front_of_paste] drives the real ChatScreen with real keys: paste, Home, type 'Q: ', one Enter, one provider stream call carrying 'Q: ' plus the full paste. Red on the pre-change head 2f85b25daf ("one Enter did not send; the draft shows 'Q: Expand pasted text (199 characters)? Enter: yes · Esc: no'") and on dev 1eaa6a9429 ('Q: Expand?'); green after. The same file's caret_before_a_trailing_token case (Left onto a trailing token, Enter) and test_enter_sends_while_a_clicked_token_asks_to_expand were red on the pre-change head for the same reason. The node id is in the UI PR gate census.

**Live** (real app, claude-haiku-4-5 via the review proxy, isolated profile, null keyring, 160x45). Pre-change head: paste, Home, 'Q: ', Enter showed 'Q: ▌Expand pasted text (156 characters)? Enter: yes · Esc: no' with 0 proxy requests. Fix: paste, Home, 'In 5 words: ', one Enter sent at once (1 request) and the model replied 'Worker-1 crashed, restarted successfully.' Evidence: tools/evidence/bd5/U11/B0-B1 (before), B5-B7 (fix).

**Docs (AC#3).** User Guide console/chat-basics.md 'Large pastes' says Enter always sends, a question typed in front of a paste included.

**Merge evidence (2026-10-30, after the TASK-37870 review round, head on dev 7536ac3b87, 0 behind).** Plain command on head: test_console_composer_token_enter.py 14 passed (14 failed on a detached origin/dev worktree, 13 with 'one Enter did not send'); test_console_composer_token_expand_key.py 20 passed (19 failed on dev; the no-token control passes). Touched neighbours compared by failing id with bootstrap_profile forced on both trees: no new failure (details in TASK-37870). ./scripts/preflight.sh passes. Live on this head (claude-haiku-4-5, 160x45): a paste plus a typed question sent on one Enter and the model answered.
<!-- SECTION:NOTES:END -->
