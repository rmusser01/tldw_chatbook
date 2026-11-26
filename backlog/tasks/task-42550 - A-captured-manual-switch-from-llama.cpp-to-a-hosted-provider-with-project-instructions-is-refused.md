---
id: TASK-42550
title: "A captured manual switch from llama.cpp to a hosted provider with project instructions is refused"
status: Done
assignee:
- '@claude'
created_date: 2026-11-25 17:35
updated_date: 2026-11-26 16:40
labels:
- follow-up
- console
- trace
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_fallback_rewrite.py
- tldw_chatbook/Chat/console_trace_slot_restore.py
- Tests/Chat/test_console_failed_call_retry_ownership.py
- Tests/Chat/test_console_trace_slot_drop.py
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while doing TASK-42525 (round 26, 2026-11-25); the owner's third decision for that task left it out of scope.

**Why.** A user whose captured chat has project instructions (AGENTS.md) and who switches the session from a llama.cpp server to a hosted provider by hand gets "Trace capture could not start" on the next message (`unsupported_surface_change`); **Send without capture** sends it. llama.cpp sends the system prompt as the first message, so the trace surface has a leading system slot. A hosted provider takes the system prompt outside the message list, so its request has no row for that slot. ADR-337 lets only a provider fallback's first call remove the slot (amendment 2026-11-24); a manual switch is a new turn's first call, which may only put a slot back (amendment 2026-11-25), not take one away.

**Evidence.** A probe on fu/r26-trace2 (deleted after use) with the real controller, bridge, gateway and trace factory, only the provider handlers faked: first message to llama.cpp, then the session resolves to a hosted provider. With AGENTS.md the next message was refused (`['unsupported_surface_change']`, run `blocked`, the hosted handler never called); without project instructions it was sent.

**Decision needed.** Whether a new turn's first call may also remove the leading slot (the mirror of amendment 2026-11-25 for the other direction), and under which checks. That is an owner decision for ADR-337.

**Owner decision (2026-11-26).** The question asked was: "Should a new turn's first call be allowed to drop that leading system slot (the mirror of the 2026-11-25 slot-restore amendment, with the same checks and the existing removal marker)?" The owner answered: "Allow it (Recommended)". It is written into ADR-337 as the owner-approved amendment of 2026-11-26, after the 2026-11-25 slot-restore amendment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision on removing the leading system slot at a manual switch is recorded in the task and, if granted, in ADR-337
- [x] #2 If granted: with Capture On and project instructions, a manual switch from llama.cpp to a hosted provider is sent and traced, and the next message after it is sent too
- [x] #3 A test drives that switch through the real controller; it is red before the change and green after
<!-- AC:END -->

## Implementation Plan

1. Record the owner's decision (above) and write it into ADR-337 as the 2026-11-26 amendment, before any code.
2. Red first: a new test file `Tests/Chat/test_console_trace_slot_drop.py` (real controller, bridge, gateway and trace factory; only the provider handlers faked) with the probe shape (llama.cpp first message with AGENTS.md, then a hosted provider: that message and the next are sent and traced), the no-AGENTS.md control, switching back to llama.cpp, Compact now and Edit & resend after a drop, and a restart.
3. Planner: beside `restored_slot_projection`, a `dropped_slot_projection` in `Chat/console_trace_slot_restore.py` returns the surface without its leading rendered-system slot when the call is a new turn's first (AGENT_FIRST/FRESH, no fallback parent, no live-native custody), the slot is the surface's leading active entry, and the request has no leading system row. Every other row is then planned by the existing rules, on a cold parent whose rows follow the removal marker's sequence.
4. Binding: recheck (still the turn's first call; the slot still leads the surface; the request still has no leading system row) and write the existing content-free `surface_removal` marker (`TraceOmissionReason.SURFACE_SLOT_REMOVED`) before the call's other rows; no schema change. Keep the service diff small (PR #3717 also edits it).
5. Hard negatives at gateway level, each pinned by its refusal code, with a mutation of each guard: a non-first call, a request that keeps its system row, a leading row that is not a slot, a fallback or retry route, and a binding recheck that finds the surface changed.
6. Lane the new file (recovered-suites preferred), time it with the step's flags, update the lane contract tests; update the User Guide; raise any size ratchet by the exact growth; run the brief's checks.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Approach.** ADR-337 amendment 2026-11-26 (written before the code, commit e98e2993b7). A new turn's first call (AGENT_FIRST or FRESH, no fallback parent, no live-native custody) whose request has no leading system row, on a surface that leads with the rendered-system slot, is planned as if the slot were gone. Every other row goes through the existing rules unchanged: append, completed-turn replacement (project context, a run's tool rows), rewritten tail. The cold parent builds on the sequence after the tail, as for a restore. At dispatch binding the shape is rechecked: the call is still its turn's first (route and `call_sequence == 0`), the slot still leads the surface, and the request still has no leading system row. Then the slot's removal is written first, with the fallback rewrite's content-free `surface_removal` marker (`remove_surface_slot`, `TraceOmissionReason.SURFACE_SLOT_REMOVED`), and the call's rows follow it. No schema change. A refusal at binding raises `rendered_system_slot_unavailable` (already a send-diagnostic category), writes nothing and leaves the call reserved.
- **Why first, not last.** The fallback rewrite writes its marker last. Written first, this marker sits before the call's rows. So the next send's completed-turn range (project context or the run's tool rows, which starts after the origin call's head) never contains it, and `rewritten_sequences` (fallback markers only) needs no change. A later Compact now or Edit & resend range that starts before it spans it as a dead sequence, like other dead rows.
- **Binding signal.** No new admission field. A drop plan, like a restore plan, leaves the sequence after the tail to a node written first (the planned parent's `next_sequence` is the tail's + 2). The binding takes the restore branch when the plan carries the slot it writes, and otherwise rechecks and writes the removal (`drop_planned`).
- **Behaviour change without project instructions.** That switch was already sent: the rewritten-tail rule replaced the slot and the first message with the history. It now drops the slot and appends, which is the same rule as the project case.
- **Files.** `Chat/console_trace_slot_restore.py` gets `dropped_slot_projection`, `drop_planned` and `write_dropped_slot`. `Chat/console_trace_service.py` gets the planner hook (`restored` is renamed `slot_change`, since it now covers both) and the binding's removal-first write; its size row goes 7037 -> 7056 (+19, exact, dated). Also changed: ADR-337, `Docs/User_Guide/console/semantic-trace-capture.md` (the bullet that said such a switch is refused; `console.md` does not mention it), and the new test file plus its lane. The `console_trace_continuation.py` diff is empty, and the service hunks do not touch PR #3717's (`operational_replacement_attempt`).

**Tests** (`Tests/Chat/test_console_trace_slot_drop.py`, 24). They use the real controller, bridge, gateway and trace factory; only the provider handlers are faked.
- The probe shape (llama.cpp first, then the session resolves to groq), with and without AGENTS.md, and with groq answering with a tool call. The switched message and the next one are sent and traced as sent, with one removal marker written by the switched turn's first call.
- llama.cpp -> groq -> groq -> llama.cpp: the slot comes back under the restore rule. Each case also runs with a cold factory before every send (a restart).
- Edit & resend of the first message, and of the message that dropped the slot: the next message and one back on llama.cpp are sent. The control is a chat always on llama.cpp.
- Compact now after a drop, warm and after a restart: the next message on the hosted wire and one back on llama.cpp's are sent.
- Gateway-level shapes, each pinned by its outcome or refusal code:
  - `valid` and `valid_fresh`: the marker sits at the tail + 1, the call's rows follow it, and the checkpoint names each row at its written sequence.
  - `keeps_system_row`: the existing header-owned rule; no marker, the slot stays.
  - `not_a_slot`: no marker, the leading row stays.
  - `tool_loop` and `retry`: `unsupported_surface_change`.
  - `fallback`: `trace_fallback_rewrite_unavailable`, the fallback's own rule.
  - Forged at binding only: `binding_not_first`, `binding_no_slot` and `binding_system_row`, each `rendered_system_slot_unavailable`, with no marker and the call left reserved.
- **Red before.** The final file was run on the base source (`6e41919880`; the two source files checked out from it, then restored and checked). 15 of the 24 failed:
  - the 4 project controller cases, with `['unsupported_surface_change'] == []` and the run `blocked`;
  - the 2 no-project probe cases, both Edit & resend cases after a drop, and both Compact now cases, on the removal-marker assertion (base sent those shapes through the rewritten-tail rule, with no marker);
  - `valid` and `valid_fresh`, on the marker;
  - the 3 binding forges, because nothing refused.

  The 9 that pass on base are the controls: the no-project switch back to llama.cpp, Edit & resend on a chat always on llama.cpp, `keeps_system_row`, `not_a_slot`, `tool_loop`, `retry` and `fallback`. All 24 pass after.
- **Mutations** (each guard removed, then the file run; restore verified by `cmp` after every run):
  - planner route: killed by `tool_loop` and `retry`;
  - planner slot: 13 killed, `not_a_slot` among them;
  - planner system row: 5 killed, `keeps_system_row` among them;
  - binding first call: killed by `binding_not_first`;
  - binding slot: killed by `binding_no_slot`;
  - binding system row: killed by `binding_system_row`;
  - the cold parent: 13 killed; the planned tail: 14 killed;
  - the root's slot removal: 12 killed; a second marker: 14 killed;
  - the replacement's predecessor after the marker: killed by the 4 project cases.
  - The binding route recheck alone kills nothing, because the planner refuses first (as for TASK-42525's binding first-call check). With the planner route and both binding route checks removed, `tool_loop`, `retry` and `binding_not_first` fail. Removing both slot checks fails 10 tests; removing both system-row checks fails 6.
- **Lane.** recovered-suites ("Run the recovered suites"), appended at the tail after the other `bootstrap_profile` suites; `RECOVERED_SUITES_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py` mirrors it. The step runs under `--storage-lease-guard`, and the file's first serial run left the chat database's executor-thread handle (and the Compact test's agent-runs handle) live, so 14 tests errored. An autouse finalizer now closes idle handles, as `test_console_rewind_summarize.py` does for TASK-39154. Timings with the step's flags (`-p no:xdist --storage-lease-guard --timeout=180`): the file alone, 24 passed in 155.2 s at load 23-42; the whole step via `r19-step.sh`, 87 targets, 2046 passed, 1 skipped, 1 deselected in 51.7 min at load 26-40 (other sessions' load plus my own checks), of which this file is about 148 s (4.8%), so about 1 min of CI's 16-20.

**Runs** (final tree, 2026-11-26; load 23-63 from other sessions):
- The new file: 24 passed at `-n 4`, and 24 passed serially under the step's flags.
- Soak: the new file plus the CI contract file, 20 rounds at `-n 4`: 181 passed in every round (60-150 s a round), at load 15.8-48.8.
- Wide: every `Tests/Chat/test_console_trace_*.py` plus the retry, rate-limit, gateway, fallback-chain, preset-fallback, model-retry, redirect, bridge, adapter-thinking, routes, project-instruction, stream-cut, named-agent, provider-grammar, self-hosted-accounting, group-merge, luna and kimi files (73 files) at `-n 4`: 2889 passed, 21 failed. The same 21 nodes fail on base (a diff of the two node lists is empty): TASK-32032's `[stopped]`, the 3 gateway reds, and group-merge (10), named-agent (2), redirect stream-cut (3) and routes (2). These are TASK-42525's known list.
- Neighbours, before the final guard cleanup (retry, chained-replacement, project-instruction and rewritten-history files): 331 passed.
- `Tests/CI` at `-n 4`: 2009 passed, 2 skipped.
- `Tests/Architecture` serially: 1506 passed, 5 skipped, 1 failed. The failure, `test_persistent_diagnostic_inventory.py::test_task_15103_review_ledger_canonical_schema_and_arithmetic`, is a load timeout: it takes 128 s alone and 221.7 s in its file's serial rerun, against the suite's 300 s default timeout, and that rerun passed whole (80 passed, 1 skipped). The full run was at load 26-63. This change adds no diagnostic site.
- `./scripts/preflight.sh` passed; `ruff check tldw_chatbook --select F` is clean; `Tests/lint/ruff_gate.py` exits 0 (RATCHET OK).

**Review fixes (2026-11-26, review of 6e41919880..06848c48e9).**
- **Important 1.** A dispatch bind that committed and then raised on exit was read back as unknown, and the run blocked (`settlement_failed`) though its trace was whole. This hit a drop with a replacement (project context), and the 2026-11-25 restore with one too. `_reconcile_dispatch_write` checked the rows in a fallback's write order (replacement first, removal last) and required the first to follow the planned head, but a drop writes its removal first and a restore its slot first.
  - The fix: each replacement row must follow the node written just before its own, and the first node written must follow the planned head. That holds for every write order. Service size row 7056 -> 7059 (+3).
  - `test_a_committed_slot_change_is_read_back_after_a_post_commit_failure`, built from the reviewer's template, injects the failure after the commit. On the pre-fix service, `drop-project`, `restore-project` and `forged_row` failed (3 of 6); all 6 pass after.
  - The two forged cases must read back unknown: `forged_row`, a replacement row written against an earlier ancestor, which the ledger's lineage trigger allows; and `forged_head`, another planned head. Removing the per-row check fails only `forged_row`; removing the first-node check fails only `forged_head`.
- **Minor 1.** The `drops_the_slot` case of the restored-slot Edit & resend test now says the send drops the slot under the 2026-11-26 amendment, and pins one `surface_removal` marker written by that turn's first call (one assertion; 5 passed).
- **Minor 2.** `remove_surface_slot`'s docstring names both callers and says only a fallback chain's marker is counted by `rewritten_sequences`.
- **Minor 3.** A `bundled` gateway scenario (the slot dropped and the first row changed) is refused with `unsupported_surface_change`, with no marker and nothing reserved.
- **Re-review addition.** `test_a_committed_fallback_rewrite_is_read_back_after_a_post_commit_failure` (4 cases, built from the re-reviewer's probe) pins the read-back of TASK-42525's fallback rewrite, which writes its removal last: slot only, and tool rows plus slot, each with and without AGENTS.md. All read back durable and the run completes. Both read-back tests now share one arming helper (`_fail_after_commit`).
  - Under the mutation `first = persisted.removal_node or min(written, ...)`, which makes the read-back assume every removal is written first (a drop's order), both tool-rows cases fail. The slot-only cases pass under it, since their write is a single removal row with nothing written before it. All 4 pass on head.
- **Lane.** The file is now 35 tests: 35 passed serially in 216.9 s under the step's flags at load 34-50, with no lease errors, and 35 passed at `-n 4`. The workflow comment says so.
- **Runs.**
  - The slot-drop, retry, chained-replacement, dispatch-recovery and project-instruction files at `-n 4`: 411 passed (load 47-57).
  - Soak, 20 rounds at `-n 4` (the slot-drop file, the edited retry test and the chain read-back test): 37 passed every round, 63-106 s a round, at load 25.4-61.6.
  - `Tests/CI` at `-n 4`: 2009 passed, 2 skipped. The size ratchet: 213 passed.
  - `./scripts/preflight.sh` passed; `ruff check tldw_chatbook --select F` is clean; `Tests/lint/ruff_gate.py` exits 0 (RATCHET OK).

**Found, not fixed (pre-existing).** With AGENTS.md, the message after an Edit & resend is refused under Capture On (`unsupported_surface_change`, `count_or_span`) in a chat that never switches providers: always on llama.cpp or always hosted. A probe was identical on base and branch (2 of 2 refused on each), so the Edit & resend test here covers the no-project case only. Filed as TASK-42560, with Continue, Regenerate and Retry, which fail the same way; it needs an owner decision.
<!-- SECTION:NOTES:END -->
