---
id: TASK-42556
title: "A spawned child agent's preset fallback under Capture On is untested"
status: To Do
assignee: []
created_date: 2026-11-26 04:50
labels:
- follow-up
- console
- agents
- trace
priority: low
references:
- tldw_chatbook/Agents/agent_runtime.py
- backlog/decisions/200-preset-pre-tool-fallback-targets.md
- backlog/decisions/337-console-reference-backed-semantic-trace-ledger.md
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in TASK-42525 (round 26, 2026-11-25). That task made a primary Console agent's provider fallback work under Capture On: the fallback starts a new trace chain. A spawned child's preset fallback was left as it was: children keep one chain and one segment. No test drives a child whose preset target fails into its fallback with Capture On, so it is not known whether that fallback is sent and traced, refused with a clear reason, or fails in a way the user cannot read.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test through the real controller, bridge, gateway and trace factory (only the provider handlers faked) drives a spawned child's preset fallback with Capture On, and the notes say what happens
- [ ] #2 If the fallback is sent, it is traced and the parent's next message is sent too; if it is refused, the refusal happens before any provider call or reservation and the user sees why
- [ ] #3 Any product change is shown red before and green after; the test runs on a PR lane
<!-- AC:END -->
