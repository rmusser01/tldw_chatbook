---
id: TASK-36012
title: >-
  Console: send start and end still request layouts for widgets whose size does
  not change
status: To Do
assignee: []
created_date: '2026-10-21 07:45'
labels:
  - console
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.7 stopped a Console run's 0.2 s poll ticks from requesting whole-screen layouts for unchanged widgets. In a headless count at 160x45 (one held send through the real Console, 2026-10-21), layout requests over six unchanged poll ticks fell from 128 on dev 906497d9b4 to 8, all eight from the Character section (TASK-35970). Its AC#2 ("no full layout when the widget's size is unchanged") is not yet shown for the send's own transitions, though. From Enter to the first streamed chunk, requests fell from 417 to 292, and from release to the end of the reply from 359 to 214. Most of the remaining requests leave the requesting widget's display, outer size, region size and virtual size unchanged at the next frame.

The largest same-size requesters at send start were the status line (6), the Inspector "More" control (6), the workspace context tray (6), the workspace details tray (4), the workspace status pair (4), the Inspector rail handle (4), the run inspector (4), the Send disabled-reason strip (3), and each of the Inspector rail's bounded sections and their viewports (3 each). At reply end, after the Character section (TASK-35970), they were the workspace context tray (6), the rail handle (4), the workspace details tray (4), the status pair (4) and the status line (3-4).

Each one needs checking: either its size can change in that transition (and the request is right), or it should repaint without a layout.

The same remainder keeps TASK-33620.15.7's AC#3 ("full reflows per send window drop by at least half") from holding reliably. Live and paired at 160x45 (36-turn chat, Anthropic claude-haiku-4-5, 16 sends per arm), the branch measured 13 reflows per send window in both rounds, while dev measured 27.5 in the first round and 20.5 in the round on that task's final head: -53%, then -37%, and -46% pooled.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each requester listed above is classified, with evidence, as one whose size can change in that transition or one that does not need a layout
- [ ] #2 The ones that do not need a layout no longer request one during a send
- [ ] #3 A deterministic test counts layout requests across a scripted send's start and end and fails on the base build
- [ ] #4 Full reflows per send window do not rise (live, real provider, paired)
<!-- AC:END -->
