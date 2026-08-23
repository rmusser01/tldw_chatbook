---
id: TASK-21117
title: >-
  Inspector right-rail scroll forces refresh(layout=True) per scroll frame - split the pure-scroll path
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - console
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21117).

`UI/Console_Modules/right_rail.py` (249 -> 1,092 lines since the pin): `_InspectorOuterBody.
watch_scroll_y` (:116-120) routes every scroll frame into the outer geometry reconcile, which
unconditionally runs two DOM queries plus `self.refresh(layout=True)` on the whole rail (:313)
and a second call_after_refresh hop with focus-recovery + fold measurement - even for a pure
scroll where no geometry changed. Coalescing trims a wheel gesture to ~2-3 full layout passes,
on the app's hottest screen.

## Acceptance Criteria

- [ ] A pure scroll updates only what scroll can change (hint text / clamp) without refresh(layout=True) or the refold chain
- [ ] The full reconcile still runs for resize / section-demand / virtual-size changes (the `_size_updated` override already distinguishes them)
- [ ] A counter probe while wheel-scrolling the rail shows zero whole-rail layout refreshes on pure scrolls; rail behavior (fold, focus recovery) unchanged
