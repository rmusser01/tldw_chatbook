---
id: TASK-1500
title: 'Composer redo needs a reachable key on non-Kitty terminals (ctrl+y alias)'
status: Done
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
- [x] #1 ctrl+y triggers composer redo through the same ChatScreen.on_key whitelist routing as ctrl+shift+z, under the same composer-owns-the-keystroke conditions.
- [x] #2 ctrl+shift+z continues to work where the terminal can transmit it.
- [x] #3 Any Console key-help surface that documents undo/redo lists both redo keys.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Route ctrl+y in ChatScreen.on_key immediately after the ctrl+shift+z handling, same
   composer-owns-keystroke conditions, consume only on successful redo
2. Verify the key token terminals deliver for ctrl+y (C0 control EM - reliable everywhere)
3. Check Console F1/help shortcut surfaces for undo/redo documentation; list both redo keys
   wherever undo is documented
4. RED-first routed test + help-surface test; mutation-check consume-only-on-success
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped on feat/console-voice-control-v2 (6dfe91e9c + 36be7284e). ctrl+y routed in
ChatScreen.on_key immediately beside ctrl+shift+z with the identical guard set and the same
always-consume/silent-no-op shape (consistency between the two redo keys pinned by test).
AC3 finding: NO surface documented undo/redo at all (TASK-1281 never added it) — added
undo/redo with both redo keys to the F1 help panel's Composer group (the compact footer and
User Guide defer per-key bindings to F1 by existing design). Collision check: no other
production ctrl+y usage app-wide. RED-first + 2 mutation checks; reviewer independently
reproduced RED, one mutation, and both sweeps (189/0). Review: PASS/PASS, one stylistic note
(duplicated branch body kept deliberately for per-key provenance comments).
<!-- SECTION:NOTES:END -->
