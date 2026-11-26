---
id: TASK-42566
title: "A sub-agent's native tool round after a preset fallback fails with \"Active provider continuation requires an owner\""
status: To Do
assignee: []
created_date: 2026-11-26 15:59
labels:
- follow-up
- console
- agents
priority: medium
references:
- tldw_chatbook/Agents/agent_service.py
- backlog/decisions/200-preset-pre-tool-fallback-targets.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Raised by the round-27 re-review of TASK-42564 (2026-11-26). Pre-existing, and not caused by capture.

**What happens.** A spawned child's preset target fails, the child falls back (ADR-200), and the fallback model answers with a native tool call. The child's next model call, which carries that tool round as a provider continuation, ends the child with "unexpected provider error (Active provider continuation requires an owner)". It happens under Capture Off too. The reviewer probed it with `_spawn_turn(..., preset_fallback=True, child_native=True)` in `Tests/Chat/test_console_spawn_subagent_trace_capture.py`.

**Where.** `AgentService._build_for_candidate` (`Agents/agent_service.py`) builds an explicit preset candidate's call with `() if explicit else continuation_groups` and `None if explicit else continuation_owner_key`. That is right for the switch itself, where the child has no continuation yet, but the built call keeps no owner key for the continuation its own native round then creates.

**Why it matters.** A preset fallback to a model with native tools works only until the child's first tool call, so the fallback rescues the run only for text-only answers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a preset fallback, a child whose fallback model makes a native tool call completes its tool round and answers, under Capture Off and Capture On
- [ ] #2 A test drives it through the real controller (`_spawn_turn` with `preset_fallback` and `child_native`), red before and green after, on a PR lane
<!-- AC:END -->
