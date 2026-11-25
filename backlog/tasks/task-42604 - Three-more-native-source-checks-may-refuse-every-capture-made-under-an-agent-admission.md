---
id: TASK-42604
title: "Three more native source checks may refuse every capture made under an agent admission"
status: To Do
assignee: []
created_date: '2026-11-25 15:43'
labels:
- follow-up
- agents
- hooks
- bug-suspected
priority: medium
references:
- tldw_chatbook/Agents/hooks_v2/native_commands.py
- tldw_chatbook/Chat/console_plugin_hook_preparation.py
- tldw_chatbook/Chat/console_reference_preparation.py
- tldw_chatbook/Agents/legacy_hook_ownership.py
dependencies:
- TASK-41943
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in review of TASK-41943 (PR #3778, 2026-11-25). Paths are under `tldw_chatbook/`, line numbers are dev aecdc35670's.

**Why.** TASK-41943 found that legacy hooks captured under an agent admission were always refused when the AgentRuns database is a file, as in the shipped app. The capture recorded the caller's admitted sources. Its nested `execution(sources=...)` also observes the service-less family sources and adds `("db.agent_runs", <config selector>)`. The next check compared the grown set with the capture and refused. A notify worker with no admission refused the same way. Three sibling sites keep that capture/check shape and were not driven under an agent admission:
- `Agents/hooks_v2/native_commands.py`: `self.sources = _guard.captured_sources(None)` (130); `sources_changed(self.sources)` (365, 376); `execution(sources=self.sources)` (428); admission on a `worker_guard()` worker (`hooks_v2/engine.py` 575, 634).
- `Chat/console_plugin_hook_preparation.py`: `captured_sources(None)` (64); `sources_changed(self.sources)` (177); `execution(sources=self.sources)` (183, 221, 288).
- `Chat/console_reference_preparation.py`: `captured_sources(None)` (258); `sources_changed(self.sources)` (311); `execution(sources=self.sources)` (320).

Likely failure, not reproduced: a capture made while the caller holds an agent admission (an agent bridge or service over a file AgentRuns DB) records that admitted set, and the first check inside `execution(sources=...)`, or on a worker with no admission, sees a different set and refuses. The feature then fails closed or is silently dropped.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each of the three sites, a test captures under an agent admission over a file-backed AgentRuns database and shows whether the next check refuses it
- [ ] #2 Each site shown refusing is fixed so capture and check observe the same sources, and its test is red on the unfixed code
- [ ] #3 At each fixed site, a source changed between capture and check is still refused (test)
<!-- AC:END -->
