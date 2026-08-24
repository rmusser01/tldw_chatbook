---
id: TASK-21595
title: >-
  Sweep timer and tick paths for Static update calls that lay out by default
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - performance
  - ui
  - textual
priority: medium
---
## Description

TASK-21501 found that the Console composer's cursor-blink tick called `Static.update(renderable)`
without `layout=False`, and since Textual's `Static.update` signature is
`(content, *, layout: bool = True)`, every blink armed a **full screen reflow** — 396
`Widget.arrange` calls per 6 ticks, about 1.9 reflows per second on an idle focused composer. The
method's own docstring said it must not do that.

That is unlikely to be the only place. Any `.update(` on a timer, tick, watcher or animation path
carries the same default.

## Acceptance Criteria

- [ ] Every `.update(` call reachable from a `set_interval`, `set_timer`, animation or high-frequency watcher is enumerated
- [ ] Each is either given `layout=False` with a stated reason the rendered size cannot change, or documented as legitimately needing layout
- [ ] Each `layout=False` is justified by a geometry-equivalence check across the states the path can produce — not by inspection; TASK-21501's mutation showed `outer_size` alone is insufficient, since painted rows changed 1 → 2 while `outer_size` stayed constant
- [ ] Layout-pass counts are measured before and after against a measured idle floor, not asserted as zero
- [ ] A lint or guard test prevents a new timer-path `.update(` from defaulting to `layout=True`

## Evidence

From TASK-21501, measured with counters around `Screen._refresh_layout`, `Compositor.reflow`,
`Widget.arrange` and `Static._layout_updates`, over six draft shapes:

| per 6 blink ticks | before | after |
|---|---|---|
| `Screen._refresh_layout` | 6 | 0 |
| `Widget.arrange` calls | 396 | 0 |
| time in `_refresh_layout` | 3.1-6.5 ms/tick | 0 |
