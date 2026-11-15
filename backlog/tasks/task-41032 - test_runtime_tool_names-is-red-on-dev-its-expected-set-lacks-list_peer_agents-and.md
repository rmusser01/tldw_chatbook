---
id: TASK-41032
title: "test_runtime_tool_names is red on dev: its expected set lacks list_peer_agents and send_to_peer"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- agents
priority: medium
references:
- Tests/Agents/test_agent_models.py
- tldw_chatbook/Agents/agent_models.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Agents/test_agent_models.py::test_runtime_tool_names` pins the exact `RUNTIME_TOOL_NAMES` set. On 2026-09-29, 7212692c8b (TASK-33430) added `list_peer_agents` and `send_to_peer` to the runtime set but not to the test, and the test has failed ever since with "Extra items in the left set". Re-checked alone on dev ecad404bd4 (2026-11-15).

The file runs on no PR lane, so every agent-runtime change has to subtract this red by hand. TASK-36950.1 did so in Phase 4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev and still pins the exact set (not a subset check)
- [ ] #2 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
