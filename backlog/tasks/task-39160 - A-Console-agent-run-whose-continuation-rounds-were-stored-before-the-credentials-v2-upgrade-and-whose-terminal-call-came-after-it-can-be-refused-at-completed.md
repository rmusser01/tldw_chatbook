---
id: TASK-39160
title: "A Console agent run whose continuation rounds were stored before the credentials-v2 upgrade and whose terminal call came after it can be refused at completed-replay attachment"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 17:46
labels:
- follow-up
- trace
- console
priority: low
references:
- tldw_chatbook/Chat/console_trace_service.py
- tldw_chatbook/Chat/console_trace_continuation.py
- tldw_chatbook/Chat/console_trace_redaction.py
- Tests/Chat/test_console_dispatch_continuation_handoff.py
- Tests/Chat/test_console_completed_replay.py
dependencies:
- TASK-39123
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (review r12b-rev3 of TASK-39123, round-12 branch fu/r12-backup). Read on the round-12 stack 9091820bf2, 2026-11-08; not built or reproduced. This code exists only in round 12, so it is named by symbol.

**What round 12 changed.** TASK-39123 stores new traces under `credentials-v2`. `trace_policies_agree` lets a `credentials-v1` round and a `credentials-v2` round agree.

**How the completed replay compares.**
- `ConsoleTraceService._completed_replay_trace_checkpoint(value, policy, …)` sanitizes the whole checkpoint under one version. That is the version of the policy it is given (`previous_policy`, the terminal call's), or the default when that version is not readable.
- Its caller in `console_trace_continuation.py` then compares each prior continuation node, as stored, with that checkpoint. If `prior.rounds != complete.rounds[: len(prior.rounds)]`, it raises `completed_replay_prior_attachment`.

**The case.** Take an agent run interrupted before the upgrade and resumed after it. Its early continuation rounds were stored under v1, and its terminal call is under v2. If an early round holds a bare `sk_live_`-style key, the replay compares v1-stored bytes with a v2-masked checkpoint, and the run would be refused. TASK-39123's own tests cover a v1 terminal followed by a v2 follow-up, not mixed versions inside one run's rounds.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test builds a run whose early continuation rounds are stored under `credentials-v1` with a bare key and whose terminal call is `credentials-v2`, and shows whether the completed replay is refused; if it is, the test is red before the change
- [ ] #2 Such a run's replay is admitted, with every captured row equal to what the current version masks; if it must stay refused, the notes say why and what the user sees
- [ ] #3 A replay whose rounds really changed is still refused
- [ ] #4 The test runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace where a run's trace policy is frozen and whether a run's calls can carry different credential versions.
2. If such a run can be recorded, build it in `Tests/Chat/test_console_dispatch_continuation_handoff.py` (the upgraded-from-v1 harness) and show whether the completed replay refuses it; otherwise stop and report NEEDS_CONTEXT with the evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Stopped: NEEDS_CONTEXT (2026-11-10, base ae9f9e8d21).** AC #1 asks for a test that builds a run whose early continuation rounds are stored under `credentials-v1` and whose terminal call is `credentials-v2`. Read on the base, no live path can record that run, so a test could only build it by bypassing two invariants:

- **One policy per run, enforced at capture.** `console_trace_runtime.py`'s tool-loop capture refuses a continuation call unless the run origin's `policy_id` and the previous call's `policy_id` equal the new call's (`origin.policy_id ... != policy.policy_id` and `previous.policy_id != policy.policy_id` -> `trace_tool_chain_unavailable`, around lines 527-546 and 575-592). A resumed or retried chain reuses its reserved policy (`policy.policy_id != reserved.policy_id`, around line 189).
- **A call's policy is immutable.** The `console_trace_calls_immutable_guard` trigger (migration `chachanotes_v55_to_v56_console_semantic_trace.sql`, recovery schema) aborts any `UPDATE` that changes `policy_id`. So the terminal call of a v1 run stays v1, and `_completed_replay_trace_checkpoint` projects the checkpoint under v1, the same projection the stored rounds used. Nothing is refused.
- **Not traced:** whether a dispatch-recovery Retry after the upgrade (a new origin under a v2 policy) can reuse the interrupted run's v1 continuation nodes inside the completed tool turn's replay range, through `_durable_reference_matches`' every-version surface-prefix match. If such reuse lands in `plan.start_sequence..plan.end_sequence`, the case is real.

**Options for the owner or coordinator:**
1. Close as not reachable through the live chain, on the evidence above.
2. Re-scope AC #1 to the recovery Retry path above (build: interrupt a v1 agent turn after round 1, upgrade, Retry, complete, then send a follow-up turn), and fix only if that is refused.
3. Harden regardless: compare each prior node and the checkpoint under the current projection (v2 re-masks v1 idempotently, per `console_trace_redaction.py`'s version rule), with a unit test on mixed-version inputs. This is cheap, but it changes refusal semantics for a state no live run is shown to produce.

No code was changed for this task.
<!-- SECTION:NOTES:END -->
