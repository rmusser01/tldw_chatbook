---
id: TASK-22218
title: >-
  Composer caret blink: no per-tick draft wrap, history scan, or under-modal ticking
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - console
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22218).

Pre-existing (the TASK-21692 layout fix holds — verified live, idle layouts 0 at tip vs 8
at pin — but the per-tick COMPUTE remains). `Widgets/Console/console_composer_bar.py:
2952-3003`: at 1.89 Hz while the composer has focus (the Console steady state), each blink
fires 2 `query_one` + placeholder/draft render; with a non-empty draft it additionally
runs `_ghost_suffix()` — a linear `startswith` scan over up to 1000 history entries
(`Chat/prompt_history.py:261-264`) — and a grapheme-aware `cell_len` wrap of the ENTIRE
draft (window sliced after the full wrap, `:2282-2296`): a pasted 20 KB draft is re-wrapped
1.89x/s forever. The resume gate is `has_focus_within` (`:2993-2996`), which survives
`push_screen` — every modal leaves the blink ticking and repainting underneath.

## Acceptance Criteria

- [ ] A blink tick with unchanged draft, width, and history performs no wrap and no history scan (memoized by those inputs; only the caret cell repaints)
- [ ] The wrap, when it does run, is bounded to the visible window rather than the whole draft, or the whole-draft cost is measured and accepted
- [ ] The blink pauses while the composer's screen is not the active screen (modal on top)
- [ ] Tick cost with a 20 KB draft measured before/after
