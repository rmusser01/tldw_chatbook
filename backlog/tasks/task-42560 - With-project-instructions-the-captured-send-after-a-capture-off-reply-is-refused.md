---
id: TASK-42560
title: "With project instructions, the captured send after a capture-off reply is refused"
status: Done
assignee:
- '@claude'
created_date: 2026-11-26 08:07
updated_date: 2026-11-26 18:32
labels:
- follow-up
- console
- trace
priority: medium
references:
- tldw_chatbook/Chat/console_trace_rewritten_tail.py
- tldw_chatbook/Chat/console_trace_continuation.py
- tldw_chatbook/Chat/console_trace_service.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42550 (round 27, 2026-11-26).

**Why.** In a captured chat with project instructions (AGENTS.md), several actions send their reply with capture off: Edit & resend, Continue, Regenerate, and Retry of a failed reply. After any of them, the next ordinary message stops with "Trace capture could not start", so every later captured send in the chat costs a **Send without capture** click. This happens on any provider, including chats that never switch. Without project instructions the same next message is sent.

**Evidence.** Probes through the real controller, bridge, gateway and trace factory, with only provider HTTP faked (llama.cpp; scratch, not committed). Each probe sent "Hello" and "And now?", ran the action, then sent one more message. The results were identical on dev 6e41919880 and on fu/r27-slotdrop.

| Action | Without AGENTS.md | With AGENTS.md |
|---|---|---|
| none (control) | sent | sent |
| Edit & resend | sent | refused: `unsupported_surface_change` (`count_or_span`) |
| Continue | sent | refused: `unsupported_surface_change` |
| Regenerate | sent | refused: `completed_tool_turn_revision` |
| Retry of a failed reply | sent | refused: `completed_tool_turn_unavailable` (also found by the retry stream) |
| Impersonate | sent | sent (it never appends to the transcript) |
| Citation repair | not probed | not probed |

Citation repair needs a RAG citation contract to drive through the controller. It replaces the saved reply after the captured call, so it is expected to behave like Regenerate.

**Cause.** The next send must replace the recorded project-context row with saved rows the trace never saw, plus the new user message and the turn's new project context.
- **Edit & resend and Continue:** the history changed. The rewritten-tail rule (ADR-337, 2026-10-27) would prove it, but `build_rewritten_tail_witness` requires the last incoming row to be the current user turn, and here it is the project-context artifact. The amendment says "tool or project artifacts in the replacement ... stay refused". Without AGENTS.md there is no trailing project row, so the rewritten tail admits the send.
- **Regenerate:** the counts match the completed-turn transition (2026-09-05/09-07), so the planner selects it. Its validation then refuses, because the saved assistant is the regenerated sibling and not the captured call's response revision. The ADR forbids retrying another proof after a validation error. Without AGENTS.md the send is a plain append.
- **Retry of a failed reply:** the uncaptured reply follows a failed captured call, and the completed-turn witness's chain and terminal checks refuse (`console_trace_continuation.py`, `validate_completed_tool_turn`). TASK-42555 (a captured Retry, blocked on PR #3717) changes this shape.

**Owner decision (2026-11-26).** The owner was asked: "TASK-42560: in a captured chat with AGENTS.md, after Edit & resend, Continue or Regenerate (and Retry, the same family), the next message is refused with 'Trace capture could not start'. It happens even when the provider never changes. Those replies go out with capture off (per the 2026-10-27 ruling), and the rule that admits the next send after them (the rewritten-tail rule) refuses project-instruction rows. Without AGENTS.md the next send works. What should happen?"

The owner chose **"Capture those actions"**: "Reverse the 2026-10-27 ruling: Edit & resend, Continue and Regenerate are traced themselves, as Retry will be (TASK-42555). This is larger work, and it overlaps #3717."

The investigation had recommended the other option, extending the rewritten-tail rule to project context for the send after a capture-off reply; the owner did not choose it. Retry stays TASK-42555's. Impersonate is unaffected (it never adds to the transcript), and citation repair is left for a follow-up (TASK-42585).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision is recorded verbatim in this task, and an ADR-337 amendment (2026-11-26, owner-approved) that defines how each action's call is admitted is written before any code
- [x] #2 With Capture On, Edit & resend, Continue and Regenerate each dispatch a traced call of their turn, with and without AGENTS.md and with prompt mix-ins, and the message after each is sent and traced
- [x] #3 One test per action drives it through the real controller, with only the provider handlers faked; each is red before the change and green after
- [x] #4 Retry of a failed reply is covered by TASK-42555 (which waits on PR #3717); this task leaves Retry unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the owner's decision and the new ACs; write the ADR-337 amendment (2026-11-26) and commit it before any code.
2. Controller: a new `Chat/console_action_capture.py` freezes each action's Capture-On request the way a normal send does (the chat's effective capture policy, a policy-carrier preparation, the saved rows read before any transform, the mix-ins composed into the request), hooked into the three action bodies with a few lines each. A temporary chat with Capture On refuses the action; a trace refusal on the direct path fails the reply with the agent path's copy.
3. Trace: the run's route descriptor names the action (from the request the action froze); the runtime finds a Continue's turn behind its continuation instruction; the planner re-records the action's turn from its user row, proven as an extended rewritten tail (Continue reply and instruction, project context) and rechecked at binding; the send after a captured Continue replaces its instruction; a direct-path action call counts as its turn's first call (slot rules, completed-turn origin). New logic in new modules; minimal diffs in the files PR #3717 touches.
4. Tests through `_project_console` (real controller, bridge, gateway, trace factory; provider HTTP or the hosted handler faked): each action on the agent and direct paths, llama.cpp and hosted, with and without AGENTS.md, with mix-ins, warm and after a trace-runtime restart, then the next message; hard negatives for each new rule with mutations. Lane the file where there is headroom.
5. User Guide, size rows, lane timing, the brief's checks; list the census/test/guide rows to flip after the restack onto reds3.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** Under Capture On, Edit & resend, Regenerate and Continue now each dispatch a traced call of their turn, on the agent path and the plain provider path, on llama.cpp and a hosted provider, with and without AGENTS.md and with prompt mix-ins. The message after each is sent and traced, also after the action's own reply failed. The ADR-337 amendment (2026-11-26, owner-approved) was committed before the code (05057ff7b9, corrections 3fd7956175).

**Approach.**
- *Controller* (`Chat/console_action_capture.py`, new; a few lines in each action body of `console_chat_controller.py`). Each action freezes the Capture-On request a manual send would build: the chat's effective capture policy (a one-time next-send choice applies and the action uses it up, as a send does), a policy-carrier `ConsoleTurnPreparation` (never registered), the saved rows read before skill, dictionary and world-info transforms, and the mix-ins composed into the request (the reply path's own composition is then a no-op, the prototype's trap). A temporary chat with Capture On refuses an action that would be traced before any row is created (decision 15). Some actions keep Capture Off, as before, each a follow-up (TASK-42583, TASK-42584, and since the review TASK-42586 for a Continue of an earlier message). One sends a pinned prefill: a captured send with a pinned prefill already stops at the chat's second message (`surface_prefix_mismatch`, shown on the base 7ebae7bd52 with a scratch probe). The other has a provider that replays its own continuation checkpoints (Kimi reasoning). Its agent requests carry the saved replies as provider artifacts the proofs cannot name, so traced actions there were refused (the covering run's `test_continue_and_regenerate_keep_one_row_per_reply[moonshot-*]`, green on base). On the plain path a trace refusal before dispatch fails the reply with "Trace capture could not be saved, so this request was not sent."; the agent path already settles it inside the run.
- *Naming the action* (`console_trace_provenance.py`, `console_agent_bridge.py`, `console_provider_gateway.py`). `RequestRouteTraceProvenance.action` names Edit & resend, Regenerate or Continue on an agent run's calls and on a llama.cpp resend; on the plain path the route is the action. The bridge copies it from the request the action froze, and the gateway's route check ignores it as it does `parent_revision_id`. It is never stored.
- *The action's call* (`console_trace_runtime.py`, `console_trace_service.py`, new `console_trace_action_calls.py`, `console_trace_rewritten_tail.py`, `console_trace_final_values.py`). A Continue's turn is found behind its instruction and saved replies, and a later call of its run takes the turn from its last saved user row. The planner re-records the turn from its user row: the prefix is capped there, and the suffix, completed-turn witness and both-domains append do not apply. The proof is the rewritten-tail witness, extended with a Continue's replies and instruction and the turn's project rows under the call's policy. It is required even for a one-row replacement and rechecked at binding. A plain-path action route counts as a turn's first call for the slot rules (`console_trace_slot_restore.py`) and for an unresolved earlier call of the turn.
- *The send after an action's run.* `after_action_call` recognises the run from the ledger. Its first call wrote its pinned user row as the first row of a replacement that started at a user row; every later call is the run's; the latest is settled. Such a send skips the completed-turn witness and is planned as a rewritten tail that removes only the run's rows after its user row. It carries the run's verified reply, or, when the run bore none, nothing or an earlier call's verified reply of the same turn. It may hang off the run's own failed empty reply, and ends with the new user and its project context. It is rechecked at binding from the call before the reserved one. A replacement that starts earlier (Compact now) stays an ordinary rewritten tail. No change in `console_trace_continuation.py`.
- *Retry* is untouched (AC #4, TASK-42555). A reply a Capture Off Retry wrote after a failed captured action is refused at the next send (`test_a_reply_the_failed_run_did_not_bear_is_refused`).

**Decisions (engineering).** A Continue of a failed reply is traced like any Continue: the failed reply is a saved row of the branch (the UI already refuses it with "Retry the failed response instead"). A Regenerate on another provider than the replaced reply's is admitted under the rules for a turn's first call (slot drop/restore), like any send. The replaced reply's earlier call stays in the trace (decision 10), so a regenerated turn lists both calls. Resend of a broken turn that re-runs from its user message uses Continue's body and route, so it is traced too; its in-place Retry stays Retry's.

**Evidence.** Probes (scratch, not committed) through `_project_console` (real controller, bridge, gateway, trace factory; provider HTTP or the hosted handler faked) reproduced the task's table on the base: the action had no call, and with AGENTS.md the next send was refused (`unsupported_surface_change`, `completed_tool_turn_revision`). With the change, a 160-case matrix passed with every request traced as sent: each action, edits and regenerations of the first message, agent and plain path, llama.cpp and hosted, with and without AGENTS.md, with mix-ins, warm and cold. So did failed-reply probes, tools, prefill, sequences and a llama.cpp resend. The first failed-reply probe found that a failed Regenerate (AGENTS.md) and every failed Continue left the next message refused. That gap led to the send-after-an-action rule, and to the lessons entry.

**Red/green.** `Tests/Chat/test_console_trace_action_calls.py` (56 tests at the first build) on the base sources (7ebae7bd52; the product files restored to base in this worktree, test unchanged, then restored and md5-verified) gave 49 failed and 7 passed. Every journey test failed on `assert [] == [('agent_first', 'complete')]` (or the plain path's route): the action had no trace call. The 7 passes are refusals the base already makes (four Continue shapes, the stray edited branch on the plain route, the Retry-written reply) and the Compact-now-after-Regenerate guard. With the change: 56 passed.

**Mutations** (each alone, its subset run, source restored and md5-verified):
- no prefix cap: 2 failed;
- no Continue step-back: 2 failed;
- no action on the bridge: 1 failed;
- the send-after rule accepts any reply: 1 failed;
- no failed-reply anchor: 2 failed;
- no earlier-reply proof: 1 failed;
- no witness requirement for a one-row action replacement: 1 failed, after adding the agent-run variant (the plain-route case was refused for another reason);
- the completed-turn witness not skipped after an action: 1 failed;
- no temporary-chat refusal: 1 failed.

A reply-role check in the chain survived its mutation. It repeated the planner's assistant-row check and binding's saved-value comparison, so it was deleted.

**Post-commit read-back** (coordinator note, slotdrop e90af1535d). A Regenerate's and a Continue's committed rewrite reads back durable after a post-commit failure on this base (`test_an_actions_committed_rewrite_is_read_back_after_a_post_commit_failure`). After the restack it imports slot-drop's `_fail_after_commit` instead of a trimmed copy, and it passed together with slot-drop's own read-back test.

**Covering run.** The 109 test files that drive the actions or touch the changed trace code (`Tests/Architecture` excluded; it ran serially) were run at `-n 4` against the final code: 76 failed and 4708 passed. The same 76 node ids fail on the base sources (7ebae7bd52, rerun by id). On the first covering run, four ids failed only on this branch, and both causes are fixed:
- `test_local_queue_retry_current_regenerate_reopen[False|True]`: the Capture Off helper now covers the actions too;
- `test_continue_and_regenerate_keep_one_row_per_reply[moonshot-*]`: Kimi continuation chats keep the actions Capture Off.

Other checks:
- `Tests/CI` (`-n 4`): 2009 passed, 2 skipped;
- `Tests/Architecture` (serial): 1507 passed, 5 skipped;
- `Tests/Docs`: 44 passed;
- `./scripts/preflight.sh`: rc 0;
- `ruff check tldw_chatbook --select F`: clean;
- `Tests/lint/ruff_gate.py`: rc 0;
- the Dreams riders step (serial, `--timeout=180`): 1651 passed, 6 skipped in 1184.5 s, run while the covering run loaded the machine (load about 10).

Soak (first build): the new file passed 5 of 5 times at `-n 4` (56 passed each, 53-62 s, load 7-11). `/verify` (a live TUI launch) was not run: the round's binding rules forbid launching the live TUI.

**Lane.** The file runs in the Dreams tracking riders step, after Tool_Packs (bootstrap_profile). After the review round: 96 passed in 274.6 s at load 6-7 (2026-11-26) serially with the step's flags (`--timeout=180 --tb=short`). At the first build, 56 passed in 296.0 s at load 31-54; before trimming, 68 tests took 393.0 s. On ubuntu:24.04 the riders step without this file takes 580 s, about 14.1 min of its 20-min cap (coordinator's Linux check, 2026-11-26), so this file's Linux time decides whether it fits there. That is over this round's stated riders budget (about 3 min for all streams), so the coordinator may move part of it.

**Files.** New: `tldw_chatbook/Chat/console_action_capture.py`, `tldw_chatbook/Chat/console_trace_action_calls.py`, `Tests/Chat/test_console_trace_action_calls.py`. Changed: `console_chat_controller.py` (+27), `console_agent_bridge.py` (+21), `console_provider_gateway.py` (+3), `console_trace_service.py` (+36), `console_trace_final_values.py`, `console_trace_provenance.py`, `console_trace_rewritten_tail.py`, `console_trace_runtime.py`, `console_trace_slot_restore.py`. Also changed: `Tests/Chat/test_console_prompt_mixin_retry.py` (its Capture Off helper now turns the actions' capture off too; `test_local_queue_retry_current_regenerate_reopen` counted their calls otherwise), ADR-337, `Docs/User_Guide/console/semantic-trace-capture.md`, `Tests/Architecture/test_module_size_ratchet.py` (four rows, exact growth, dated), `.github/workflows/derived-artifacts.yml` and `Tests/CI/test_ci_queue_pressure_contract.py`, `backlog/docs/lessons-testing-evidence.md`.

**After the restack onto reds3 (0c012e7a84).**
- Flipped: the census rows CONTINUE, REGENERATE and EDIT are `CONVERSATION_TRACE` again, with the comment above the census rewritten; `test_console_request_provenance_routes.py`'s Capture-Off set drops them; the census doc's paragraph says they are traced (TASK-42555's Retry wording and the summary decision are unchanged); slot-drop's `test_edit_and_resend_after_a_drop_keeps_the_next_send` docstring no longer says the message after an Edit & resend is refused. `test_console_route_census_capture.py` needed no edit: it reads the census, so its three action cases now expect one traced call, and every follow-up-send assertion stays.
- One-time choice: the first build left a one-time **Next send** choice armed after an action that honoured it, so it decided two sends. The action now uses it up when it follows the chat's capture mode (Capture Off, or a traced action); a temporary, prefill or Kimi chat's action is not a captured send, so it leaves the choice for the next send. `test_an_action_uses_up_the_one_time_capture_choice` (Edit & resend, Regenerate, Continue): 3 failed before, 3 passed after. ADR-337 amendment point 1 and the User Guide say so.
- Follow-ups filed: TASK-42583 (prefill), TASK-42584 (Kimi), TASK-42585 (citation repair).
**Review round (2026-11-26, controller and trace-admission reviews of 581287e4d2..531b61acb2).** ADR-337 amendment first (12c515f996, point 8 narrowed in 693c834e7e), then the code (94da87a134).
- *Continue of an earlier message* (Important 1): Continue is offered on any message; its request runs through the selected message, but its reply is appended at the end of the chat, so the traced call re-recorded the selected turn and dropped the later ones. The next message was refused even without AGENTS.md (a regression), and with a prompt mix-in a Continue of an earlier user message raised `mixin_anchor_unavailable` and left the run in `validating` ("A run is already running in this tab."). Such a Continue (any non-system row after its target) now keeps Capture Off and leaves a one-time Capture On armed; with AGENTS.md its next message is refused as before (TASK-42586, pinned by `test_a_continue_of_an_earlier_message_keeps_capture_off[*-agents_md]`). `turn_user_id` returns the last user before the new reply, which is the request's own user for every Continue that is still traced.
- *Request freezing* (Important 2): any failure composing the mix-ins or freezing the trace request now ends the action visibly: a mix-in composition failure is blocked as the reply path blocks it ("Mix-in attempt could not be saved; response was not sent."), anything else fails the reply with the trace copy, and a failed Kimi continuation lookup keeps the action Capture Off so the stream refuses the conflict itself.
- *Wording and exclusions* (Minors 3-5): the prefill exclusion is "an action that sends the pinned prefill" (Edit & resend, Regenerate, Resend); a plain Continue is traced. Only an action that follows the policy uses up the one-time choice. A temporary chat refuses only an action that would be traced.
- *Logging* (Minor 6): a failed capture policy read logs `capture_policy_preparation_failed`, as a normal send does; the request-freezing failure logs `action_trace_request_failed`. Both bind only the exception's class name; the inventory pin was regenerated after reading the statements.
- *Trace side* (Important A): a send after an action's run whose replacement starts at the run's user row but carries another row there (a one-time Capture Off Edit & resend of that message, a switch back to the branch a traced Edit replaced) falls back to the ordinary rewritten tail; before, it and every later send were refused (`unsupported_surface_change`).
- *Trace side* (Important B): after a failed Edit & resend with prompt mix-ins, the next request restores the run user's saved source, and the new user hangs off the failed, empty reply. The witness now names that reply (`skipped_reply_id`), only after an action's run that bore nothing, with the one restored history row; binding rechecks it. A first version allowed this in any rewritten tail; the covering run showed it then replaced a legacy wrong-owner ledger (`test_legacy_artifact_history_remains_frozen_and_has_explicit_recovery`), so it was narrowed.
- *Trace side* (Minor C): the Continue step-back needs a call naming Continue or Regenerate and a trailing row whose text ends with Console's instruction (`CONTINUE_INSTRUCTION`, pinned equal to the controller's constant).
- *Known limit, found in this round:* when the trace refuses an action's own reservation, the reply fails and stays as a failed, empty reply; each later captured send in the chat then pauses at "Trace capture could not start" (the rewritten tail cannot skip that reply outside an action's run), and Send without capture sends it. `test_a_refused_plain_path_action_fails_its_reply` now reaches the copy through a real reservation refusal and pins this; the earlier version patched the stream, which never persisted the reply, so it hid the pause. The ADR's "Still refused" list and the User Guide state it.
- *Red/green:* the 49 new or changed cases on the product code of 531b61acb2: 22 failed, 27 passed (the passes pin behaviour that was already right, such as the one-time choices, Edit of an earlier message and the step-back's admitted controls); all pass here. Covering run (the same 109 files, `-n 4`): the only failures outside the earlier base list were `test_an_unproven_edited_branch_stays_refused[gap_in_chain]` (the wide skipped-reply rule read columns its fixture lacks) and the legacy ledger test above, both green after the narrowing. On the final code the covering run gave 66 failed and 4769 passed, every failure on the earlier base list. The touched files (15 entries, `-n 4`) passed twice, 495 each; `Tests/CI` 2015 passed; `Tests/Architecture` (serial) 1507 passed; preflight, the diagnostic inventory check, Ruff F and `ruff_gate.py` are clean.
- Also: Minor 8 (the ADR commits are 05057ff7b9 and 3fd7956175 after the restack), Minor 9 (the guide's Retry sentence now says that with AGENTS.md the message after any Retry of a failed reply stops at "Trace capture could not start"), Minor 10 (tests for Edit & resend of an earlier message, a real Capture Off chat via the chat's own Capture choice, since `_project_console(capture_enabled=False)` only arms a one-time choice, and the real reservation refusal).
- Follow-up filed: TASK-42586 (trace a Continue of an earlier message).
**Owner decision (2026-11-26, refused action).** The owner was asked: "When the trace refuses one of these actions' own call (rare), the empty reply it already saved stays as a failed reply. Every later captured send in that chat then stops at 'Trace capture could not start' until the user picks Send without capture. Which fix should go in?" The owner chose **"Drop the empty reply (Recommended)"**: "When the trace refuses before anything is sent, remove the action's empty reply and say why. The chat is left as it was before the action, so the user can try again or turn Capture off. Small, and stays clear of PR #3717." Recorded as ADR-337 amendment point 7 before the code.

**Correction.** The message of 94da87a134 says the new or changed cases gave "23 failed, 26 passed" on 531b61acb2's product code. That count was taken before the refusal test changed; with the final tests it is 22 failed and 27 passed, as the review-round notes above say.
<!-- SECTION:NOTES:END -->
