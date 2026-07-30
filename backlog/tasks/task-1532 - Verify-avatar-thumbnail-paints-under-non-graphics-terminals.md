---
id: TASK-1532
title: 'Verify avatar thumbnail paints under non-graphics terminals (blank in tmux harness)'
status: In Progress
assignee: []
created_date: '2026-07-30 15:30'
labels: [roleplay, ux, verification, P3]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Dev (c329def0d) implements character avatar thumbnails (Roleplay Inspector
`#personas-inspector-avatar-thumb`, character editor thumb, generation card)
with a graphics mode (`textual_image`) and a pixels fallback
(`_build_avatar_pixels` → `rich_pixels.Pixels`, personas_screen.py:5343).

In a live tmux run (TERM without graphics support) with a character whose
366 KB portrait was confirmed present in the DB, the Inspector reserved the
thumb box but painted nothing: `capture-pane -e` over the region shows zero
half-block glyphs (▀▄█) and only 4 unique background colors (panel chrome) —
a rendered Pixels globe would emit many colored half-block cells. No decode
error appeared in the app log.

Needs: (1) verification in a real graphics terminal (iTerm2/Kitty) that the
graphics path paints; (2) a check of why the pixels fallback produced no
visible cells under tmux — mode resolution may be picking "graphics" in a
terminal that swallows the escape sequences, or the Pixels renderable is
lost between build and mount. Assert with `render_line()`/`render_strips()`,
not widget presence (renderable ≠ painted).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Root cause (proven by env probe): host-terminal env (`TERM_PROGRAM=iTerm.app`, `ITERM_SESSION_ID`) leaks into tmux panes, so detection recommended graphics whose escapes tmux swallows -> blank thumb.
2. Fix `detect_terminal_capabilities`: when `TMUX` is set or TERM starts with screen/tmux, classify as `tmux`, force pixels, no tgp/sixel.
3. TDD: new Tests/Utils/test_terminal_utils.py (tmux leak cases + iTerm-outside-tmux graphics guard).
4. Live tmux re-run: avatar thumb must paint half-block cells.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Avatar thumbnail visibly paints in a graphics-capable terminal for a card with an embedded portrait.
- [x] #2 The pixels fallback visibly paints (half-block cells) in a non-graphics terminal (e.g. under tmux), or mode resolution is fixed to pick pixels there.
- [x] #3 A test asserts painted output at the render seam for the fallback path.
<!-- AC:END -->

## Implementation Notes

Root cause proven live: the host terminal's env (`TERM_PROGRAM=iTerm.app`,
`ITERM_SESSION_ID`) leaks into tmux panes, so `detect_terminal_capabilities`
classified the pane as iTerm2 and recommended graphics mode -- whose escape
sequences tmux swallows, painting nothing. Fix: when `TMUX` is set or TERM
starts with screen/tmux, classify as `tmux` and force pixels (no tgp/sixel).

Live re-verification under tmux: the Inspector avatar region went from 0
half-block glyphs / 4 chrome-only colors to 200 half-block glyphs / 94 unique
colors (the card portrait painting via rich_pixels).

Tests: Tests/Utils/test_terminal_utils.py (tmux leak cases + an
iTerm-outside-tmux graphics guard, watched RED first) and
Tests/UI/test_personas_avatar_render.py (characterizes visible half-block
output at the `_build_avatar_pixels` render seam).

AC #1 (graphics-capable terminal paint) is intentionally left unchecked and
the task In Progress: it can only be confirmed in a real iTerm2/Kitty session
outside tmux -- open Roleplay, select a character with an embedded portrait,
and confirm the Inspector thumbnail shows the image.
