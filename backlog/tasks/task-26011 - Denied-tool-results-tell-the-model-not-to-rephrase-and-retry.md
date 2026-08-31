---
id: TASK-26011
title: 'Denied tool results: tell the model not to rephrase and retry'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:44'
updated_date: '2026-08-31 17:35'
labels:
  - agents
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A denied tool call invites an immediate near-identical retry. Verified on origin/dev: the refusal copy is fixed text - Agents/mcp_tool_provider.py:87 USER_DENY_REFUSAL and Agents/builtin_tool_gate.py:359 - and states the denial without instructing against retrying by another route, so the model commonly rephrases and re-asks, burning turns and approval prompts. Hermes states explicitly that the model must not retry, must not rephrase, and must not pursue the same outcome by a different path, and that silence is not consent. Complements task-18920 (deny with a reason) and task-18929 (denial circuit breaker) without depending on either.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Denied-call results instruct the model not to retry the same call, not to rephrase it, and not to pursue the same outcome by another route
- [x] #2 The instruction is distinct from the user's own reason text where one is supplied, so user words are never confused with system policy
- [x] #3 Timeout, unresolved, kill-switch and Off refusals keep their existing distinct copy - only the user-denial path changes
- [x] #4 Copy is asserted by a test so it cannot silently drift
- [x] #5 No permission-model change: this is result text only
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Result text only; the permission model is untouched.

1. One shared constant, not three copies. The existing comments say the wording is deliberately kept in sync across the MCP provider, the builtin gate and the console review hook, so triplicating a new sentence would guarantee drift.
2. Home it in builtin_tool_gate, the lowest of the three (nothing imports back into it).
3. Keep the policy separate from the refusal text so a user-authored denial reason (TASK-18920) can sit beside it without the model reading the user's words as system policy.
4. Leave timeout/unresolved/kill-switch/Off copy exactly as-is and pin all four, since TASK-294 established that provenance deliberately.
5. Update pinning tests to derive from the constant rather than restating the literal.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Denied tool calls now tell the model what NOT to do next. The refusal stated the denial and stopped there, so a model would commonly rephrase the same call and re-ask -- costing turns and putting a second approval card in front of a user who had already decided.

**One constant, three call sites.** `builtin_tool_gate.DENIAL_POLICY` reads: "Do not retry this call, do not rephrase it, and do not pursue the same outcome by another route. Ask the user what to do instead." The existing comments in all three modules state that this wording is deliberately kept in sync, so it is imported rather than repeated -- `mcp_tool_provider.USER_DENY_REFUSAL`, `console_chat_controller.USER_DENIED_REFUSAL` and the gate's own `user_denial_refusal()` helper all compose it. A test asserts the three are the *same object*, so a future copy-paste fails loudly.

`builtin_tool_gate` is the home because nothing imports back into it; the other two already depend on it, so no cycle.

**Kept separate from the refusal text (AC#2)** so TASK-18920's user-authored denial reason can be shown alongside without the model reading the user's words as system policy, or the reverse.

**AC#3 pinned exactly.** Timeout, unresolved, kill-switch and Off keep their own copy, asserted verbatim, plus an assertion that the policy sentence appears in none of them. TASK-294 established that provenance deliberately -- those four describe genuinely different situations (nobody decided; permissions are Off; the switch is thrown) and collapsing them would destroy it. The TASK-294 invariants are re-asserted here too: the user-denial text still says "denied by the user" and still never says "permissions".

**Two stale test pins found and fixed.** `test_console_raw_shell_revocation` asserted the literal old string and now derives it from `user_denial_refusal()`. `test_console_activity_presentation` fed a hand-written literal into the activity classifier; it passed by prefix match while no longer being a string production emits, so the classifier was not actually being exercised against real builtin-gate output. Both now derive from the shared constant.

**Verification.** 211 pass across the six files that touch this copy. The `Tests/Agents/` suite shows 15 failures both before and after -- verified as the *identical set* by diffing sorted failure names, not just equal counts, since equal counts can hide a swap.

Method note: the full `Tests/Chat/` suite exceeds a 10-minute timeout, and the stash-and-compare baseline technique is unsafe there -- a stall between `git stash` and `git stash pop` would strand the work. Coverage for that suite was established by grepping for every site pinning the copy and running those files directly.

**Files:** `tldw_chatbook/Agents/builtin_tool_gate.py`, `tldw_chatbook/Agents/mcp_tool_provider.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `Tests/Agents/test_denial_anti_retry_copy.py` (new), `Tests/Chat/test_console_raw_shell_revocation.py`, `Tests/Chat/test_console_activity_presentation.py`.
<!-- SECTION:NOTES:END -->
