---
id: task-1583
title: 'Scope Inspector: bottom clipping and mid-token wraps'
status: To Do
assignee: []
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - rescore-p2
dependencies: []
priority: medium
---

## Description (the why)

Critique rescore P2: in 8 of 20 evidence captures the Scope Inspector's
last visible line is cut mid-sentence ("Saves apply to your local",
"…Nothing is sent to", "Recovery: use each prompt's") — reassurance copy
that reads worse truncated than absent. The 34-char column also breaks
tokens mid-word ("crede/ntial_source", "config.tom/l"). A scrollbar exists
but the default viewport reliably clips the standing local-scope note.

## Acceptance Criteria (the what)

- [ ] The inspector's default viewport does not cut the standing
      local-scope note mid-sentence on common category/terminal sizes
      (shorten the copy, reflow, or reserve a fold indicator row)
- [ ] Config paths and TOML key names wrap at token boundaries or are
      ellipsized, not split mid-word
- [ ] Overflow remains reachable (scrollbar/fold indicator preserved)
