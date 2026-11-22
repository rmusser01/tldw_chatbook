---
id: TASK-42420
title: "The Console live report save test is red on dev because its spawned sub-agent never starts"
status: To Do
assignee: []
created_date: 2026-11-22 22:10
updated_date: 2026-11-22 22:10
labels:
- follow-up
- console
- agents
- tests
- dev-red
priority: medium
references:
- Tests/UI/test_console_agent_progress.py
- tldw_chatbook/Agents/run_hooks.py
- tldw_chatbook/Agents/fleet_message_tools.py
- .github/workflows/derived-artifacts.yml
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during round 21 of the follow-up burn-down (2026-11-22): one of the 12 local reds TASK-42342's covering run reported on dev ddd3c02052 (its notes, line 84). Re-checked on the round-21 stack 77f124a740 (dev ddd3c02052 plus seven ports), whose ports touch none of these files; paths are under `tldw_chatbook/` unless they start with `Tests/`, `scripts/` or `.github/`. Split from TASK-42417, whose chat-start reds have a different signature.

**Why.** `Tests/UI/test_console_agent_progress.py::test_real_live_report_save_preserves_open_selection_and_navigation` (a `@private_profile_test`) fails every time: in its file, in an isolated `-n 4` run, and alone serially. The file is in no PR step and not in the UI census, so nothing reports it. The test pins that saving a live sub-agent report keeps the open selection and navigation (the "queued progress from sub-agents" feature, TASK-42004).

**The failure.** In the child process the primary's `spawn_subagent` call returns `outcome=failed`, so the scripted sub-agent never reaches the held gateway, and `assert await asyncio.to_thread(entered.wait, 10)` at line 453 fails with the primary's `ConsoleSubmitResult(accepted=True, visible_copy='done', ...)`. The child also logs `run-hooks: event=UserPromptSubmit ... hook=unavailable engine error (exception_type=AgentActivationRequired)` from `Agents/run_hooks.py:1190`. Whether the activation refusal is what fails the spawn is not known.

The other three reds in this file are the known `RecoveryRequired: raw_source_selection_changed` local trap and are not part of this task.

**Coordination.** `Agents/run_hooks.py` and `.github/workflows/derived-artifacts.yml` are under open work in another session: coordinate with tldw-chatbook-30 (#3683 / TASK-42281) before editing either.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Why `spawn_subagent` fails in this test is recorded: test drift after a named product change, or a product defect
- [ ] #2 Drift is repaired in the test; a defect is fixed in product code with a test shown red first
- [ ] #3 The test passes alone and in a `-n 4` run of its file (the three RecoveryRequired nodes aside)
- [ ] #4 The test runs in a PR lane, with its serial time under the lane's flags recorded in the notes
<!-- AC:END -->
