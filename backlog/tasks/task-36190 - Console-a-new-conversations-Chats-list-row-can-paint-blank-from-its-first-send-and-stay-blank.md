---
id: TASK-36190
title: >-
  Console: a new conversation's Chats list row can paint blank from its first
  send and stay blank
status: To Do
assignee: []
created_date: '2026-10-22 18:20'
labels:
  - console
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the visual check on TASK-33620.15.7's rebased head (2026-10-22, 235x52, `ready` mode, Anthropic claude-haiku-4-5 through the review proxy in `drip` mode, one send of "In three short sentences, describe the moon." in a fresh profile). The send creates a conversation, and the left rail's Chats list should show it under "Today", with a running marker while the reply streams and "In three short sentences, d...  💬" after. Often that row is painted blank instead: the line under "Today" keeps its height but shows no title and no marker. It can go blank while the reply streams or only after the reply ends; once blank, it stayed blank in every launch, through the run and in captures taken 2 s, 6 s and 12 s after the reply ends.

It is dev's own defect, not caused by TASK-33620.15.7. Over two alternating loops (detached origin/dev 7700c63a68 then 28beeb9ed5, against the branch rebased on each), the row was blank mid-run in 3 of 11 dev launches and 4 of 14 branch launches, and blank at the end in 7 of 11 dev launches and 7 of 14 branch launches.

The row is drawn by `ConsoleWorkspaceTree` (a Textual `Tree`). `Tree._render_line` paints a blank strip for a line past the end of its cached tree lines, and `TreeNode.set_label` schedules `_refresh_node`, which repaints only the lines whose path contains the node at that moment; either could leave the row blank if the node is added, moved or relabelled while the line cache is stale. Neither is confirmed.

Evidence (captures with .txt/.ansi/.png): `scratchpad/uxrev/evidence/bd5/LRF/visual/` (`235x52-{dev,fix}-*`, `r2/`, `loop-1..10/235x52-{dev,fix}-*`, `final/235x52-fix-*`) and `scratchpad/uxrev/evidence/bd5/LRF/loop.log`, `loop2.log`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the blank row is established with evidence (which update leaves the row unpainted, and why)
- [ ] #2 A new conversation's row shows its title and marker during and after its first run without the user hovering or scrolling the rail
- [ ] #3 A deterministic test reproduces the blank row on the base build and passes with the fix
- [ ] #4 Checked live at 235x52 over at least ten fresh-profile first sends, with no blank row mid-run or after
<!-- AC:END -->
