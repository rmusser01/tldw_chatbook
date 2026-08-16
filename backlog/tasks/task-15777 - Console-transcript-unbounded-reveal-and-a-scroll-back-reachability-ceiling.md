---
id: TASK-15777
title: 'Console transcript: unbounded reveal on a far jump, and a scroll-back reachability ceiling'
status: Done
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

- [x] A design decision is made and documented for bounding the reveal cost
      of a far jump (e.g. windowed reveal + progressive hydration toward the
      target, rather than mounting everything between the jump target and
      the tail in one pass)
- [x] A design decision is made and documented for two-sided windowing (or an
      explicit, justified decision not to pursue it), covering how it
      interacts with tail-follow, in-flight streaming, and the jump pill
- [x] Whichever lever(s) ship are covered by tests pinning: bounded mount
      count on a far jump, and (if two-sided windowing ships) that
      scroll-back remains reachable past the current low-watermark ceiling
- [x] `Tests/UI/test_console_transcript_window_reconcile.py` and
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

## Implementation Notes

Both symptoms shipped fixed by ONE mechanism: a second contiguous hidden
boundary (a hidden TAIL) alongside TASK-15455's hidden prefix — two-sided
windowing — gated so every existing regime is behavior-identical.

**Design decisions (the two ACs).**

- *Two-sided windowing: SHIPPED*, scoped by a gate (`_two_sided_active`):
  windowing enabled AND pruning enabled (`high_mark > 0`) AND sane marks
  (`low >= chunk + 2*viewport`, `high - low >= chunk`). The two boundaries
  are always indices over one message list, so mounted rows are one
  contiguous slice by construction — islands are impossible and no gap-seam
  row is needed (the failure mode 15455's reconciliation rejected came from
  two independently mutated SETS, not two indices). Interactions, as the
  filing demanded:
  - *tail-follow*: an `anchor()` override funnels every "follow the newest"
    path through `_reveal_tail_window()` — clear the suffix and re-window
    onto a fresh bounded tail (clearing alone would remount everything
    between the reader and the tail: the unbounded reveal again).
  - *in-flight streaming*: the suffix is sticky across `set_messages`
    ingests (remapped by surviving ids); ids appended while the reader is
    deep in scroll-back join the suffix automatically, so the 0.2s sync tick
    neither remounts the tail nor moves the reader (pinned).
  - *jump pill / send*: both land on a fresh bounded tail window; the send
    branch recomputes `window_start` explicitly because `_set_hidden_prefix`
    at the end of `set_messages` would overwrite what `anchor()` set.
- *Far-jump bound: SHIPPED as windowed reveal (re-center), not progressive
  hydration toward the target.* When the newly revealed stretch between the
  target and the current window exceeds the LOW watermark in estimated
  lines, `reveal_message` mounts an initial-window-sized slice from the
  target's turn (the same shape a session load produces) and hides the rest
  in the suffix; the target row is scrolled to the top once mounted. Near
  reveals (j/k over the boundary, nearby restores) keep the plain boundary
  extension. *Alternatives considered*: progressive hydration toward the
  target (bounds the latency spike but NOT the DOM — the selected target is
  prune-protected at the top, so nothing ever trims); a hard row cap on the
  reveal (arbitrary knob, still unbounded below it); islands + gap seam
  (rejected by 15455 for reconciler complexity). The low watermark is the
  principled budget: a jump may mount at most what the watermarks allow to
  STAY mounted.

**Fixed-point preservation** (the 15455 reconciliation headline): hydration
is no longer refused at the low mark — instead each hydrated chunk is
followed by an estimated-lines tail-trim back to
`max(low, scroll_y + 2*viewports)` (never trimming a row group holding
focus), which lands the height under `high` BEFORE the prune check runs, so
the prune can never chase hydration. Where that argument cannot hold
(degenerate marks like the 45/70 fixtures, where one chunk overshoots
`high`), the gate keeps the 15455 refusal — the fixed-point tests pass
unmodified. Downward: symmetric boundary hydration (wheel-down / PageDown /
watch_scroll_y + a self-chaining check for the anchored-at-bottom case,
where the anchor's auto-scroll happens while the in-flight latch swallows
the boundary signal), with the existing prefix prune bounding that end.

**Evidence.**
- Probe on unmodified base `4ea8bf1c9` (400 msgs, 600/900 marks): scroll-back
  grew 24→152 mounted rows then froze at `first=m248` forever — 248 messages
  unreachable; far jump `m10` of 500 mounted 490 rows.
- Born-red pins (both failed on base, verbatim):
  `test_sustained_scrollback_reaches_the_oldest_message_and_stays_bounded`
  and `test_far_jump_mounts_a_bounded_recentered_window` in the new
  `Tests/UI/test_console_transcript_two_sided_window.py` (8 tests: the two
  symptoms, tail-trim + downward recovery, post-jump tail walk, jump pill,
  streaming-tick stickiness, send re-window, kill-switch inertness).
- Probe after fix: mounted DOM constant at 101 rows / height 407 while the
  window walks m360→m0 (reachability + bound); far jump at DEFAULT 12k/20k
  marks with realistic 30-line messages mounts 5 rows (m10–m14) vs 490.
- Suites: new 8/8; protected `window_reconcile` + `windowing` 17/17 GREEN
  AND UNMODIFIED; adjacent transcript suites (native, pruning, tail-follow,
  selection contract, jump pill, region, fence throttle, diff row, markdown,
  composer collapse, citation sources, native chat flow) ~670 passed; the 9
  failures observed are all PRE-EXISTING on base `4ea8bf1c9` (identical
  lists re-run against the unmodified widget: 4 markdown-widget API-drift
  reds, 3 speak-action reds, 2 session-switcher reds). ruff clean.

**Files.** `tldw_chatbook/Widgets/Console/console_transcript.py`,
`tldw_chatbook/config.py` (comment), `Docs/User_Guide/console.md`
("Long conversations" + stamp),
`Tests/UI/test_console_transcript_two_sided_window.py` (new),
`backlog/docs/lessons-backlog-hygiene.md` (checkout-baseline trap), this
task file.
