---
id: TASK-36190
title: >-
  Console: the Chats list row of a just-finished run paints blank until
  something else repaints the rail
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
Found by the visual check on TASK-33620.15.7's rebased head (2026-10-22, 235x52, `ready` mode, Anthropic claude-haiku-4-5 through the review proxy in `drip` mode, one send of "In three short sentences, describe the moon." in a fresh profile). While the run streams, the left rail's Chats list shows the new conversation under "Today" with its running marker. After the reply ends, that row is often painted blank: the line under "Today" keeps its height but shows no title and no chat marker, and it stays blank in captures taken 2 s, 6 s and 12 s later. In other launches the same row reads "In three short sentences, d...  💬" as expected.

It happens on dev without TASK-33620.15.7: a detached origin/dev 7700c63a68 worktree painted the row blank in 3 of 5 launches, the rebased branch in 4 of 7, alternating arms in one loop. So it is dev's own defect, not caused by the run-reflow change, though fewer whole-screen layouts could make a missing repaint easier to see. A later probe with shorter prompts did not reproduce it in 3 sends, so it may depend on the first send's retitle landing during or just after the run.

The row is drawn by `ConsoleWorkspaceTree` (a Textual `Tree`); `TreeNode.set_label` schedules `_refresh_node`, which repaints only the lines whose path contains the node at that moment. A label or data change that lands while the tree's line cache is being rebuilt may therefore leave the row unpainted.

Evidence (captures with .txt/.ansi/.png): `scratchpad/uxrev/evidence/bd5/LRF/visual/` (`235x52-fix-end*`, `loop-1..4/235x52-{dev,fix}-end*`, `loop.log`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the blank row is established with evidence (which update leaves the row unpainted, and why)
- [ ] #2 After a run ends, the Chats list shows the conversation's title and marker without the user hovering or scrolling the rail
- [ ] #3 A deterministic test reproduces the blank row on the base build and passes with the fix
- [ ] #4 Checked live at 235x52 over at least five fresh-profile first sends, with no blank row
<!-- AC:END -->
