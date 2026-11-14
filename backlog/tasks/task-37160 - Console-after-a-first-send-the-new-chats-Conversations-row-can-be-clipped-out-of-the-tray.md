---
id: TASK-37160
title: >-
  Console: after a first send the new chat's Conversations row can be clipped
  out of the tray
status: To Do
assignee: []
created_date: '2026-10-27 22:28'
labels:
  - console
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while verifying TASK-33620.15.20 live (2026-10-27, 160x45, Anthropic claude-haiku-4-5, fresh profile, origin/dev 0923ec0ca2). After the first send in a new chat, the Conversations section's Chats list shows its "Today" divider and then a blank line where the chat's row should be: the renamed row ("In one short sent…  💬") is not painted. The selected-chat summary above the filter has just wrapped to two lines, which moves the list down one row, and the tray keeps its old height, so the row falls outside the tray's box. Typing in the filter (a rebuild) brings it back.

Live, the row was missing after the first send in 7 of 10 origin/dev launches, and in 3 of 7 launches of the TASK-33620.15.20 branch (a pre-existing race; that branch does not change the fit). Headless, through the Console send harness, the tray ended one row short in 1 of 4 origin/dev runs.

What a trace of `_fit_height_to_content` showed (headless, both arms): during startup a fit pass ran while only some children had a layout (child bottoms 5, 0, 0). The TASK-2154.3 guard only waits when every bottom is 0 (content bottom 1), so that pass pinned an explicit height of 5, and the next pass pinned 13. With an explicit height the tray no longer grows with its content. After the first send's rebuild, the last fit pass measured the summary before it wrapped (13 rows), and nothing fits again afterwards: the run's end is now patched in place (TASK-36012) rather than rebuilt and refitted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After the first send in a new chat, the chat's row is visible in the Conversations list in every live launch of a paired round against origin/dev (at least 10 launches per arm, 160x45)
- [ ] #2 A fit pass that runs while the tray's children are only partly laid out does not pin a height smaller than the settled content
- [ ] #3 A deterministic headless test reproduces the clipped row on origin/dev and passes on the fix
<!-- AC:END -->
