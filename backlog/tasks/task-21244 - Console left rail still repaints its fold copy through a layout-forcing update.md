---
id: TASK-21244
title: >-
  Console left rail still repaints its fold copy through a layout-forcing
  update
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - console
  - performance
dependencies: []
priority: low
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; informational
finding from the TASK-21117 review (PR #2016, `7489a0ec8`) promoted to a follow-up.

TASK-21117 split the Inspector right rail's pure-scroll path from its layout reconcile and
documented the underlying Textual trap in a new `backlog/docs/lessons-textual.md` entry:
`Static.update(str)` schedules `refresh(layout=True)`, so a copy-only repaint costs roughly
**two extra whole-screen layout passes**. `UI/Console_Modules/left_rail.py:1086` is now the
**only remaining instance** of that trap in the Console rails — it repaints the fold hint with
`hint.update(text)` on rail transitions.

The same file also keeps a shadow `_outer_hint_text` field (`:323`, `:1083`, `:1085`) — the
shape TASK-21117's AC3 forbade on the right rail. Here it is safe: the skip requires the
shadow **and** the live renderable to match (`text == self._outer_hint_text and
str(hint.renderable) == text`), so the worst case is one redundant write rather than a missed
repaint. The left rail has no stand-down path and the two rails' copy predicates genuinely
differ, so this is not a copy-paste of the right-rail fix. The TASK-21117 reviewer confirmed
none of this is a latent left-rail bug — the cost is real but bounded to transitions, which is
why it is low priority rather than dropped.

## Acceptance Criteria

- [ ] A left-rail fold-copy change does not force a whole-screen layout pass when only the text changed
- [ ] Left-rail fold and unfold copy is unchanged in every state the existing rail tests cover
- [ ] The shadow-field shape is brought in line with the right rail's, or the reason the left rail needs a different one is recorded in the code
- [ ] A test fails if a copy-only left-rail update reintroduces a layout pass
