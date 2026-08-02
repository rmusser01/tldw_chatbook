---
id: TASK-570
title: >-
  Console branching: agent inline TOOL markers vanish each turn on active-path
  recompute
status: Done
assignee: []
created_date: '2026-07-25'
labels:
  - console
  - chat
  - agents
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
(Re-filed from task-498 in the PR #803 rework: that ID was taken over by the image-gen program on dev; re-triaged post-Phase-C.)

Console conversation branching (Phase A, PR #799) made `_messages_by_session` a derived active-path view rebuilt by `_recompute_active_path`. Live agent TOOL markers (the inline `⚙ tool → …` / `⤷ spawned …` scrollback rows) are appended display-only (`append_message(role=TOOL, persist=False)`), so they are correctly kept out of the message tree — but any subsequent recompute (the next send, a swipe, a delete) rebuilds the view from real tree nodes and drops them. Net effect: after an agent turn, the inline TOOL markers disappear from the transcript on the very next turn (the rail's agent summary is unaffected).

Post-Phase-C re-triage (2026-07-25): Phase C (PR #827) landed durable id-anchoring for RESUME marker placement (`agent_runs.assistant_message_id`, id-match → anchor-after, off-path → hidden), but markers remain transient by design — `apply_resume_marker_overlay`'s own docstring states the next `_recompute_active_path` drops them, exactly as live markers are ephemeral. So the drop-each-turn behavior still reproduces on current dev. The remaining fix is retention: re-derive/re-overlay the markers after each recompute (Phase C's id-anchored placement makes this well-defined for the first time), or an equivalent durable display-row mechanism.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Inline agent TOOL markers persist in the Console transcript across subsequent turns/swipes for a conversation that used the agent runtime (or the limitation is documented honestly where users see it)
- [x] #2 Marker persistence does not reintroduce TOOL rows into the conversation tree (they must stay display-only, never parents)
- [x] #3 Behavior verified in the live TUI with an agent-runtime conversation
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Closed as already fixed, by TASK-1842 (PR #1192).** This task's own prescription -- "re-derive/re-overlay the markers after each recompute" -- is what shipped: markers are anchored to the node they followed and spliced back by `_with_tool_markers` on every `_recompute_active_path`, while still never becoming tree nodes. The resume door (`restore_state`, which writes the view directly) was closed in the same change, and `_purge_tool_markers` was added afterwards so markers do not outlive their anchor or their session.

Kept rather than deleted because the diagnosis here is the better-written one, and it records the Phase A/Phase C history that explains WHY the markers were transient by design.

**Filed twice.** TASK-1842 was written from a user report without first searching the backlog for an existing task describing the same defect. Search before filing.
<!-- SECTION:NOTES:END -->
