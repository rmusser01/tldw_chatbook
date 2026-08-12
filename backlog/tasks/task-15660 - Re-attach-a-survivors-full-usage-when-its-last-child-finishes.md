---
id: TASK-15660
title: 'Re-attach a survivor''s full usage when its last child finishes (fleet F3)'
status: To Do
assignee: []
created_date: '2026-08-11 21:30'
labels:
  - console
  - agents
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-1 made a surviving sub-agent's post-turn token spend OBSERVABLE (the Console cost chip's `Sub-agents: N tok (not priced)` line) rather than attributed. The real fix needs a "last child done" signal the bridge does not emit today; PR 3a-2 builds exactly that signal for auto-wake, so this task consumes it rather than building a second one. Re-attach is already known to be idempotent (`_attach_stream_usage` recomputes from all payloads and `set_message_usage` REPLACES), and that idempotence is pinned by a test, so the path is safe to reuse.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A sub-agent that finishes after its turn has its usage folded into the originating assistant message's own usage row, not only into the cost chip
- [ ] #2 Re-attaching twice produces the same stored total (the existing idempotence guard still passes)
- [ ] #3 A conversation export includes a survivor's spend once the re-attach has run
- [ ] #4 The chip's unattributed line falls to zero for a run whose children have all been re-attached
<!-- AC:END -->
