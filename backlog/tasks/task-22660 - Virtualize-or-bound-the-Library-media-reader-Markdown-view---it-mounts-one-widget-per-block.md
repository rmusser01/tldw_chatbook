---
id: TASK-22660
title: >-
  Virtualize or bound the Library media reader Markdown view - it mounts
  one widget per block
status: To Do
assignee: []
created_date: '2026-08-26'
labels:
  - performance
  - library
priority: medium
dependencies: []
---

## Description

Source: TASK-22500's close-out measurement (2026-08-26). TASK-22500 virtualized the
Library media reader's Raw text view (`VirtualizedRawContent`, `Widgets/Library/
library_media_raw_view.py`), fixing the O(document) `Widget._render_content` cost on
every repaint. Its Markdown view (`mode="rendered"`, backed by Textual's own `Markdown`
widget) was an explicit scope decision -- out of scope for that task, measured instead of
fixed, with the numbers carried forward here per the task's own instructions.

Measured directly (`time.perf_counter()` bracketing the actual awaited
`Markdown.update()` call -- no harness wall time) against the same three probe document
sizes used to verify TASK-22500's Raw-view win, reshaped into paragraphs (blank-line
separated 5-line blocks, a realistic Markdown shape rather than one unbroken block):

| size    | `MarkdownIt.parse()` alone | `Markdown.update()` (parse + mount) | blocks mounted |
|---------|----------------------------|--------------------------------------|-----------------|
| 100 KB  | 24.2 ms                    | 129.3 ms                             | 291             |
| 1 MB    | 245.4 ms                   | 2307.2 ms                            | 2,865           |
| 2.5 MB  | 830.4 ms                   | **10,022.5 ms**                      | 7,139           |

At 2.5 MB the Rendered view takes over ten seconds to first paint -- a hang, not a
slowdown -- against the Raw view's post-TASK-22500 first paint of well under 2 ms of
`render_line` self-time (plus a one-time ~141 ms index build) at the same size.

This is a structurally different problem from the one TASK-22500 solved. Textual's
`Markdown` widget parses the whole document with `markdown-it` and mounts ONE
`MarkdownBlock` widget (itself a `Static` subclass, subject to the exact same
`Widget._render_content` O(document)-per-widget cost TASK-22500 fixed for the Raw view)
per block. Cost scales with BOTH document size (parse time, ~linear) and block count
(per-widget mount overhead, which dominates -- `update()` costs roughly 8-12x the raw
parse time at every size measured). TASK-22500's row-windowing approach does not apply
here without a redesign of how Markdown blocks are mounted.

## Acceptance Criteria

- [ ] A design decision is recorded for how the Markdown reader view should behave on
      large documents (options include: capping auto-mounted block count with an
      explicit "switch to Raw view for documents this large" fallback, paginating
      blocks, lazily mounting blocks as they scroll into view, or a genuinely
      virtualized Markdown renderer), with trade-offs stated
- [ ] Large Markdown-eligible documents (>=1 MB) no longer take multiple seconds to
      first paint in Rendered mode
- [ ] The Raw <-> Rendered toggle and Markdown-specific features (table of contents,
      link clicks) keep working for documents within whatever new bound is chosen
- [ ] A guard pins the property (Rendered-mode first paint does not scale unboundedly
      with document size) so it cannot regress silently
