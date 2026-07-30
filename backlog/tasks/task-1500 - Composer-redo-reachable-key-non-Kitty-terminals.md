---
id: TASK-1500
title: 'Composer redo needs a reachable key on non-Kitty terminals (ctrl+y alias)'
status: To Do
assignee: []
created_date: '2026-07-30 12:00'
labels: [console, composer, ux, accessibility]
dependencies: [TASK-1281]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-1281 shipped composer undo/redo on ctrl+z / ctrl+shift+z per its AC. During review it was
established (probed via XTermParser._parse_extended_key) that terminals without the Kitty keyboard
protocol cannot transmit ctrl+shift+z distinctly -- it collapses to plain ctrl+z at the wire level,
so redo is unreachable for most terminal users (Terminal.app, stock iTerm2). The TASK-1281 reviewer
ruled the shipped behavior correct-per-AC and recommended the alias land as its own task with its
own AC rather than being slipped into 1281.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 ctrl+y triggers composer redo through the same ChatScreen.on_key whitelist routing as ctrl+shift+z, under the same composer-owns-the-keystroke conditions.
- [ ] #2 ctrl+shift+z continues to work where the terminal can transmit it.
- [ ] #3 Any Console key-help surface that documents undo/redo lists both redo keys.
<!-- AC:END -->
