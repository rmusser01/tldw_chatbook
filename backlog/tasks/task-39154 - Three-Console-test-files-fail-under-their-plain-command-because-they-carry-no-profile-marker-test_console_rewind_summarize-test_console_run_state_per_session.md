---
id: TASK-39154
title: "Three Console test files fail under their plain command because they carry no profile marker: test_console_rewind_summarize, test_console_run_state_per_session and test_console_mcp_approval"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-10 18:40'
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- test-infra
- console
priority: low
references:
- Tests/Chat/test_console_rewind_summarize.py
- Tests/Chat/test_console_run_state_per_session.py
- Tests/UI/test_console_mcp_approval.py
- Tests/conftest.py
- Tests/private_profile.py
dependencies:
- TASK-38111
- TASK-39141
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39138's covering run, round-12 branch fu/r12-ctlr). Checked on dev b590a15afd, 2026-11-08; the round-12 stack fails the same nodes. All runs: run.sh, `-n 4`, load 24-35.

**The runs.** The ctlr stream named six files.
- Plain, they gave 91 failed and 141 passed.
- A scratch plugin then added the `bootstrap_profile` marker to every node. This was a temporary run-time change and was not committed. Under it they gave 16 failed and 216 passed.

**Red only from the per-test profile redirect.** These 75 nodes fail with `RecoveryRequired: raw_source_selection_changed`, or with a symptom of it, and pass under the marker:
- `Tests/Chat/test_console_rewind_summarize.py`: 58. That is 57 `RecoveryRequired`, plus `test_native_message_id_key_stripped_before_provider` (`assert False is True`). Only three of its tests carry a marker.
- `Tests/Chat/test_console_run_state_per_session.py`: 7. That is 3 `RecoveryRequired`, plus 4 `assert False is True` or `TimeoutError`. The file has no marker.
- `Tests/UI/test_console_mcp_approval.py`: 10, all `RecoveryRequired`. One node in the file is `@private_profile_test`.

**Red on their own assertions.** These fail under the marker, and plain too:
- `test_console_run_state_per_session.py::test_submit_draft_closed_session_id_fails_closed_without_touching_active` (`KeyError: 'Unknown Console chat session'`). TASK-39138 (round 12) fixes the cause; its notes say the node passes on its head under a temporary marker.
- `test_console_mcp_approval.py`, 8 nodes:
  - three CSS pins: `_agentic_terminal.tcss` is missing `.approval-row {`, the `.approval-row-decision {` block's `width: 27;`, and the `.approval-row-full-command` block's `width: 1fr`;
  - `test_chat_screen_forwards_approval_decided_to_controller`: `'Mock' object is not iterable`;
  - `test_refusing_one_call_does_not_get_overwritten_by_approving_another` and `test_a_no_app_round_does_not_record_a_user_denial_on_the_mcp_hook`: a `ToolReviewDecision` where a string verdict is expected;
  - `test_revocation_cannot_split_a_batch_snapshot`: `'ReadGap' object has no attribute 'denial_reasons'`;
  - `test_approved_definitive_tool_stays_mounted_until_real_terminal[crash]`: "approved handler never started".

**The other three named files are not this class.**
- `test_console_edit_resend.py` and `test_console_regenerate_branching.py` already carry a module `bootstrap_profile` marker; their reds are their own and out of scope here.
- `Tests/integration/test_console_chat_fork_flow.py` is listed in TASK-33621.36, which now has a dated note with this round's evidence.

No PR lane runs any of the three files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The 75 redirect-only nodes pass under their files' plain command, through the suite's existing profile seams; no test binds config to the real profile, and RecoveryRequired admission is unchanged
- [x] #2 Each own-assertion red is classified as test drift or a product defect, and fixed or retired with the commit that moved the behaviour named; a product defect gets a test that is red before its fix
- [x] #3 Each of the three files runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

1. Show the plain reds at base ae9f9e8d21 (run.sh, `-n 4`): RecoveryRequired through the per-test profile redirect.
2. Put the three files on the bootstrap profile with a module `pytestmark = pytest.mark.bootstrap_profile` (TASK-39141 seam); no conftest, private_profile or admission change.
3. Re-run under the marker; classify each remaining own-assertion red by the commit that moved the behaviour (git log -S / blame): re-pin drift to the shipped behaviour, fix a product defect with a test red before the fix.
4. Time each file serially three times for the lane (the parent wires the lane).

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-11-09 evidence (review r12fr; dev 3a6498ae14, run.sh, `-n 4`, load ~20). `Tests/UI/test_console_mcp_approval.py` plain still has 11 `RecoveryRequired` nodes. Under the temporary `bootstrap_profile` marker it now gives 31 failed / 92 passed. Changes against the list above:
- `test_revocation_cannot_split_a_batch_snapshot` now passes.
- `test_raw_shell_command_view_has_bounded_scrollable_geometry` now fails with `_agentic_terminal.tcss is missing '.approval-row-full-command'`, because #3036 moved the approval rules into `css/features/_console_approvals.tcss`.
- `test_approved_definitive_tool_stays_mounted_until_real_terminal[success]` is red before #3036 too, like `[crash]` above: on the round-12 stack 9091820bf2, which has no #3036, it fails under the marker with "approved handler never started" (review r12fr2: 3 of 3; re-run here: 2 of 2, load ~31-41).
- #3036 (the approval UX merge) makes 22 more nodes red. They are filed as their own task, which leaves this file's lane to this task.
2026-11-10 (round 13, branch fu/r13-cred, base ae9f9e8d21). All runs through run.sh.

**Approach.** The three files take the TASK-39141 seam: a module `pytestmark = pytest.mark.bootstrap_profile` in `Tests/Chat/test_console_rewind_summarize.py`, `Tests/Chat/test_console_run_state_per_session.py` and `Tests/UI/test_console_mcp_approval.py`. No conftest, `Tests/private_profile.py`, `app_factory` or run.sh change; `RecoveryRequired` admission is unchanged. The file's one `@private_profile_test` node keeps its child process.

**Red first, same command** (`pytest <the three files> -n 4`):
- plain, base: 104 failed / 111 passed (rewind 58, run-state 8, approval 38; load ~14-23).
- marker only, base: the two Chat files 0 failed; the approval file 30 failed (load ~55). `test_native_message_id_key_stripped_before_provider` was a RecoveryRequired symptom. `test_submit_draft_closed_session_id_fails_closed_without_touching_active` passes: TASK-39138 (#3483) fixed it. `test_revocation_cannot_split_a_batch_snapshot` already passed.
- after (this branch, with the controller node): 215 passed, twice (load 70-88).

**This task's own reds, all test drift** (TASK-40097's notes classify the post-#3036 ones):
- The three CSS pins: bc1c716e43 (ADR-161) replaced the literals with `$ds-*` tokens, 838acbf037 merged `.approval-row` into `.approval-row, .approval-row-controls`, and a2fcbe61bc (#3036) moved the rules to `css/features/_console_approvals.tcss` (spliced into `_agentic_terminal.tcss`'s `/* @console-approvals */` slot) and gave the Select `$ds-approval-decision-width: 28`. The shared helper now reads that file and the bundle, resolves each file's own tokens, and still asserts the values (1fr, 6, 3, auto, 28) and the splice slot. Negative control: expecting `width: 27` or `height: 7` failed both nodes (reverted).
- `test_chat_screen_forwards_approval_decided_to_controller`: 3e618d7119 (#3023) makes `set_chat_controller` adopt the controller's `_preparation_reads`; the bare `Mock` now carries the empty set a real controller has.
- `test_refusing_one_call_does_not_get_overwritten_by_approving_another`, `test_a_no_app_round_does_not_record_a_user_denial_on_the_mcp_hook`: 55121153dd made the hook's per-call verdicts `ToolReviewDecision`s. Re-pinned to the exact decisions: refused `(USER_DENIED_REFUSAL, "denied")`, approved `("proceed", "approved")`, headless `(USER_DENIED_REFUSAL, None)`. The same drift had made two green pins vacuous (`ToolReviewDecision(...) != "proceed"` is always true); `test_a_refusal_never_stamps_the_name_even_when_it_is_decided_last` and `test_a_revoked_childs_card_can_no_longer_execute_its_tool` now compare the decision. The latter's fixed `time.sleep(0.2)` before its "card is on screen" precondition failed once at load 77; it now polls with a 5 s bound.
- `test_approved_definitive_tool_stays_mounted_until_real_terminal[success|crash]`: two post-#3036 causes, read from a temporary instrumented run (reverted). The click landed while the fast button was still disabled (`_information_ready` False): a41da77591 holds decisions until the round's facts lay out, so the wait now also requires an enabled button. The finishing card's Details navigation buttons are hidden, not disabled (a2fcbe61bc composes them once): the pin now asserts the Details panel is closed and every displayed control disabled. The pre-#3036 "never started" on the round-12 stack was not re-derived; that tree is not this base.

No product defect found. Also: `Tests/UI/test_chat_approval_card.py`'s comment pointing at the needs-decision tests now says what they pin since #3036.

**Lane** (AC#3, left to the coordinator). Serial timings, run.sh, `pytest <file> --timeout=180 --tb=short -q`, wall seconds (1-min load before each run), three runs:
- `Tests/UI/test_console_mcp_approval.py` (122 passed each run): 294 (98), 250 (65), 255 (72).
- `Tests/Chat/test_console_rewind_summarize.py` (73 passed): 214 (76), 145 (54), 150 (65).
- `Tests/Chat/test_console_run_state_per_session.py` (19 passed): 52 (71), 55 (76), 46 (49).

Lane choice: the approval file is a whole-file UI census entry; the two Chat files are bootstrap_profile and go in a derived-artifacts lane after its sandboxed suites.
<!-- SECTION:NOTES:END -->

**Lanes (AC#3).**
- `Tests/UI/test_console_mcp_approval.py` is a whole-file UI census entry at the tail. Serial timings with the lane's command: 294 s, 250 s and 255 s (122 passed each; load 98, 65, 72).
- `Tests/Chat/test_console_rewind_summarize.py` and `Tests/Chat/test_console_run_state_per_session.py` join `recovered-suites` after its sandboxed block. The file timings were 214/145/150 s (73 passed) and 52/55/46 s (19 passed).
- The lane runs under `--storage-lease-guard`. The rewind file's first step run failed 58 nodes on the open lease of `_durable_controller`'s CharactersRAGDB, and one more on the startup-rider test's AgentRunsDB plus the workspace store the agent turn opens. An autouse fixture now closes each durable store, and that test closes the kernel stores it records: 73 passed under the guard.
- `recovered-suites`' whole pytest command, serial through run.sh: 943 s, 1088 s and 1020 s wall, 871 passed and 1 skipped each (1-minute load 32-44). In CI the same step before this round ran 733 passed in 307 s (run 42841528703). This round's additions take about 330 s serial locally at most, so the job should stay near half its 20-minute cap.
- Census: 71 round-13 entries charged at their local serial seconds, then `--rebalance 13.3` swapped 6 pairs over the 22 cached PR runs' shard logs (max 13.1 pytest min); floor 1984.
