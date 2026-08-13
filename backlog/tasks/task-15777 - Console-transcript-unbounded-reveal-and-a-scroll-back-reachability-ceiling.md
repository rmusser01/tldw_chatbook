---
id: TASK-15777
title: 'Console transcript: unbounded reveal on a far jump, and a scroll-back reachability ceiling'
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - perf
  - console
  - design
priority: medium
---

## Description

Two untracked residuals recorded explicitly "for the controller to file" in
task-15455's Implementation Notes (input-latency burn-down's Console
transcript windowing work). Both are inherited from the merged windowing
design (task-15455), not introduced by its later reconciliation PR, and both
need a design decision rather than a small tweak:

1. **Unbounded reveal on a far jump.** Selecting or jumping to a message near
   the start of a long session reveals everything from it to the tail in one
   pass — measured ~490 rows mounted for message 10 of 500. Latent for any
   future "jump to search hit" feature, which would hit the same cost.
2. **Scroll-back reachability ceiling.** Once the mounted view reaches the
   LOW prune watermark (~12,000 rendered rows by default), scroll-back stops
   loading older history entirely; that content is reachable only via export
   or a jump, not by continuing to scroll. Removing the ceiling needs
   two-sided windowing (trimming the tail as the head grows), which neither
   the merged implementation nor its reconciliation has, and which would
   touch tail-follow, streaming, and the jump-pill mechanics task-15455
   built.

Related context (already fixed, not part of this task): task-15455's
reconciliation PR already fixed a read-order defect in `restore_reading_state`
(clamping against the pre-reveal `max_scroll_y` used to silently drop the
reader elsewhere); that fix is why a design for #1/#2 needs to account for
reveal-then-restore ordering rather than re-deriving it from scratch.

## Acceptance Criteria

- [ ] A design decision is made and documented for bounding the reveal cost
      of a far jump (e.g. windowed reveal + progressive hydration toward the
      target, rather than mounting everything between the jump target and
      the tail in one pass)
- [ ] A design decision is made and documented for two-sided windowing (or an
      explicit, justified decision not to pursue it), covering how it
      interacts with tail-follow, in-flight streaming, and the jump pill
- [ ] Whichever lever(s) ship are covered by tests pinning: bounded mount
      count on a far jump, and (if two-sided windowing ships) that
      scroll-back remains reachable past the current low-watermark ceiling
- [ ] `Tests/UI/test_console_transcript_window_reconcile.py` and
      `Tests/UI/test_console_transcript_windowing.py` stay green
