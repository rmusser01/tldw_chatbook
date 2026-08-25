---
id: TASK-22228
title: >-
  Console and reader small-residue batch (press/scroll/recompose leftovers)
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - cleanup
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22228).

Verified small items that do not warrant tasks alone (each with cites in the evidence
doc):
1. `chat_screen.py:18953-18956` — `on_mouse_up` runs an id `query_one` on every Console
   mouse-up before any cheap guard (same physical press 21119 cleaned; cheaper class).
2. `left_rail.py:145-148`, `:1250-1267` — the left rail lacks the TASK-21117 pure-scroll
   split the Inspector rail has (guarded, but 2 query_one + max_scroll_y per frame).
3. `left_rail.py:561-585` — `_focusable_body_controls` does an uncached subtree
   `query("*")` per focus change in the rail.
4. `Workspaces/display_state.py:631-688` — 3+ stat/realpath syscalls per bound folder per
   state build (deliberate ADR-028 posture; frequency drops with TASK-22201 — re-evaluate
   a short-TTL cache for network mounts after it lands).
5. `Widgets/Prompts/prompt_block_editor.py:818-888` — `_sync_footer` fires 3 unguarded
   `Static.update()` (layout=True) + an unguarded tooltip write per keystroke while the
   prompt workbench is open (partly PR #2053).
6. `library_screen.py:32300`, `:32311`, `:32506-32521`, `:33599`, `:33611`, `:14316` — six
   reader button presses still whole-screen recompose; the two delete-confirm presses
   re-parse the document in Read mode.
7. `library_screen.py:5741-5755` + `library_media_reader_shell.py:117-127` — two layout
   resolves per Resize event (screen-level one fires even when Media is not active).

## Acceptance Criteria

- [ ] Each numbered item is fixed as described or explicitly declined with a reason in the notes
- [ ] No behavior change beyond the stated mechanics; touched areas keep their tests green
- [ ] Fixes verified by the cheap probe named in the evidence doc where one exists
