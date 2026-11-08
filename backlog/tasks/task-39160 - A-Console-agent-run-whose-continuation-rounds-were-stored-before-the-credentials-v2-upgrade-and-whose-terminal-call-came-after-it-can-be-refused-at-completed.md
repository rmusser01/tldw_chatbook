---
id: TASK-39160
title: "A Console agent run whose continuation rounds were stored before the credentials-v2 upgrade and whose terminal call came after it can be refused at completed-replay attachment"
status: To Do
created_date: 2026-11-08 06:54
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
