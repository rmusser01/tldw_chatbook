---
id: TASK-15777
title: 'Console transcript: unbounded reveal on a far jump, and a scroll-back reachability ceiling'
status: In Progress
assignee:
  - '@claude'
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

## Implementation Plan

Design decision (both symptoms share one mechanism): add a second, contiguous
hidden **suffix** boundary next to the existing hidden prefix — two-sided
windowing — but gate it so every existing regime is byte-identical:

1. **State.** `_hidden_tail_ids` (set) + `_hidden_tail_start` (index), always
   derived together from one index over `_messages`, so mounted rows stay ONE
   contiguous slice by construction (no islands, no gap seam — the failure
   mode 15455 rejected came from two independently mutated sets).
2. **Gate.** Two-sided behavior activates only when windowing is enabled
   (kill switch respected), pruning is enabled (`high_mark > 0`), and the
   watermarks are sane (`low >= chunk + 2*viewport` and `high - low >= chunk`).
   Degenerate/tiny marks (the 45/70 fixed-point fixtures) keep today's
   one-sided behavior INCLUDING the low-watermark hydration refusal — that
   refusal is the fixed-point loop-breaker there and must survive.
3. **Ceiling fix (symptom 2).** With the gate active, boundary hydration is no
   longer refused at the low watermark; after each hydrated chunk the tail is
   trimmed back into the hidden suffix (estimated-lines walk, keeping at least
   `max(low_mark, scroll_y + 2*viewport)` mounted, never trimming a group
   holding focus). Post-trim height < high, so the prune never fires and the
   15455 fixed point is preserved by construction.
4. **Downward reachability.** Symmetric hydration at the bottom boundary
   (wheel-down / PageDown / watch_scroll_y) reveals the next suffix chunk;
   the existing prefix prune bounds the other end. Jump pill re-windows the
   tail; a new-user-send yank clears the suffix and re-windows the tail;
   `set_messages` keeps the suffix sticky across streaming ticks (appended
   ids join it) and remaps it by surviving ids.
5. **Far-jump bound (symptom 1).** `reveal_message` re-centers the window on
   the target (initial-window budget below the target's turn) when the
   full reveal would exceed the low watermark in estimated lines; smaller
   reveals keep today's extend-the-boundary behavior, so the pinned
   contiguous-reveal tests stay green.
6. **Evidence.** Mounted probe reproducing both symptoms on the unmodified
   widget; born-red pins for both; DOM bound asserted after sustained
   scroll-back; both protected suites run in full, unmodified.
7. Docs: `Docs/User_Guide/console.md` "Long conversations" + `config.py`
   comment lose the ceiling caveat.
