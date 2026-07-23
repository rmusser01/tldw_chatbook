---
id: TASK-498
title: >-
  Console branching: agent inline TOOL markers vanish each turn on active-path
  recompute
status: To Do
assignee: []
created_date: '2026-07-23'
labels:
  - console
  - chat
  - agents
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console conversation branching (Phase A, PR #799) made `_messages_by_session` a derived active-path view rebuilt by `_recompute_active_path`. Live agent TOOL markers (the inline `⚙ tool → …` / `⤷ spawned …` scrollback rows) are appended display-only (`append_message(role=TOOL, persist=False)`), so they are correctly kept out of the message tree — but any subsequent recompute (the next send, a swipe, a delete) rebuilds the view from real tree nodes and drops them. Net effect: after an agent turn, the inline TOOL markers disappear from the transcript on the very next turn (the rail's agent summary is unaffected). This is broader than the Phase-A spec's "markers drop on swipe" wording. Phase C (agent-marker anchoring, `agent_runs` v1→v2 + `assistant_message_id`) is the designated real fix; this task tracks it and the honest-limitation documentation in the interim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Inline agent TOOL markers persist in the Console transcript across subsequent turns/swipes for a conversation that used the agent runtime (or the limitation is documented honestly where users see it)
- [ ] #2 Marker persistence does not reintroduce TOOL rows into the conversation tree (they must stay display-only, never parents)
- [ ] #3 Behavior verified in the live TUI with an agent-runtime conversation
<!-- AC:END -->
