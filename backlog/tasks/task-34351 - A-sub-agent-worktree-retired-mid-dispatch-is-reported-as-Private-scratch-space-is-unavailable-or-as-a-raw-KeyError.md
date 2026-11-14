---
id: TASK-34351
title: >-
  A sub-agent worktree retired mid-dispatch is reported as 'Private scratch
  space is unavailable', or as a raw KeyError
status: Done
assignee:
  - '@claude'
created_date: '2026-10-03 18:37'
updated_date: '2026-10-04 08:33'
labels:
  - workspace
  - agents
  - follow-up-33940
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33940.2 stopped worker failures on an admitted workspace folder from being reported as "Private scratch space is unavailable; the tool was not run." The same misleading copy, or worse, survives on the sub-agent worktree path.

A sub-agent run that asks for isolation gets its own git worktree, admitted to `LocalToolProvider` under the alias `agent-<run id>` (`admit_run_workspace_root`) and retired when that run ends (`retire_run_workspace_root`, via `AgentService._retire_agent_worktree`). Console folder bindings are never retired; only these agent worktrees are. A path tool call from that run can lose the race with its own retirement in two places:

- **Before the approval gate.** The dispatch-spec lookup finds the alias gone and refuses with the scratch copy, which sends the model, and the user reading the activity row, to the wrong place. That confusion is what TASK-33940 started from.
- **After the approval gate.** The handler lookup indexes the same cache again. A run that ended while its approval card was waiting raises a bare `KeyError`, and the model sees the alias name as the error text. This is the realistic window, because an approval card can wait on the user indefinitely.

This belongs with the Console copy audit in TASK-33621, which tracks failure copy that states false facts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When a sub-agent's path tool call loses the race with that run's worktree retirement, before or after the approval gate, it is refused with copy saying the run has ended and its isolated worktree was released; the copy does not mention scratch space, and no raw exception text reaches the model
- [x] #2 The scratch-unavailable copy is still returned when the private scratch space itself is unavailable
- [x] #3 The new copy is registered with the Console activity presentation so the activity row shows it as blocked
- [x] #4 Tests reproduce both windows against the real provider with a real retire (no patched internals); each fails on dev `01a2020981` and passes with the fix, and nothing is written to the worktree
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: real-provider tests for both retire windows (guard-triggered retire before the spec lookup; approval callback that retires, then approves) plus a real closed-scratch lease that must keep the scratch copy.
2. Add LOCAL_RUN_WORKTREE_RELEASED_REFUSAL; use it at the pre-gate spec lookup.
3. After the gate, resolve the handler with .get() and refuse with the same copy (AUTHORITY_UNAVAILABLE, not dispatched) instead of raising KeyError.
4. Register the copy as blocked in the Console activity presentation, and extend its parametrized test.
5. Compare the provider/presentation suites against dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A sub-agent run's isolated worktree is un-routed by retire_run_workspace_root when the run ends. A path tool call from that run could lose the race in two places, and both now refuse with LOCAL_RUN_WORKTREE_RELEASED_REFUSAL: "This run has ended and its isolated worktree was released; the tool was not run."

- Before the gate (Agents/local_tool_provider.py, spec lookup): this used to return the private-scratch refusal. Only a retired agent worktree reaches this branch, because static Console roots are never retired.
- After the gate (the realistic window, since an approval card can wait indefinitely): _invoke_allowed re-indexed _path_specs_by_alias[alias][name] and raised a bare KeyError, whose text (the alias) reached the model as the error. The handler spec is now resolved once, before any side effect, with .get(); a missing spec refuses as AUTHORITY_UNAVAILABLE, not dispatched.
- The new copy is registered as blocked in the Console activity presentation (console_agent_bridge).
- The scratch copy is unchanged for real scratch failures. The other three sites that use it never see an admitted workspace root, since only scratch carries an authority_scope.

Evidence: Tests/Agents/test_agent_worktree_retire_race_copy.py (3, private profiles, real provider, real subprocess WorkspaceToolExecutor, a real retire triggered from the provider's own guard / approval callback, no patched internals). Both race tests were RED on dev (scratch copy; KeyError text 'agent-run-child'). The closed-scratch control passes on both sides. Activity presentation row added. Regression: the provider, virtual CLI, worker-copy, review-hook, activity-presentation and fleet-runtime suites have identical failure sets on branch and dev (393 shared, all the local RecoveryRequired trap).
<!-- SECTION:NOTES:END -->
