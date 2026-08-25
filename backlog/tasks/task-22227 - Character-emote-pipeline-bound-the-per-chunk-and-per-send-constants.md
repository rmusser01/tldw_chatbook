---
id: TASK-22227
title: >-
  Character emote pipeline: bound the per-chunk and per-send constants
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - chat
  - personas
priority: low
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22227).

New with PR #2020, character sessions only (armed only when the assistant is a character).
(a) `Character_Chat/emote_directives.py:251-263`, `:333-337`, `:434-438`: the streaming
parser consumes per CHARACTER with a per-character `str.encode('utf-16-le')`
(`:92-95`) and list append — O(len(chunk)) with a high constant (~16k encodes per reply)
plus a clone + `safe_copy` per chunk. (b) `Chat/console_chat_store.py:7905-7947`:
`detect_character_mood` runs 14 compiled regex passes + 2 more over the full reply at the
terminal seam, on the loop. (c) `Chat/console_chat_controller.py:16091-16150`:
`_build_character_emote_snapshot` is O(assets^2) in regex evaluations per send (~1600 for
a 40-asset pack).

## Acceptance Criteria

- [ ] The parser publishes visible text in runs, not per character (or its per-chunk cost is measured and shown acceptable at 16k-char replies)
- [ ] The snapshot projection is O(assets) (normalize each asset once)
- [ ] Mood detection cost per turn is measured and bounded (or moved off-loop)
- [ ] Emote semantics unchanged: existing directive/mood tests green
