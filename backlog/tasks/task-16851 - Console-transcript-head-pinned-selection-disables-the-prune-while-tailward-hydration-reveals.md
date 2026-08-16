---
id: TASK-16851
title: 'Console transcript: head-pinned selection disables the prune while tailward hydration reveals'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
labels:
  - bug
  - console
  - design
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the TASK-15777 round-3 review (PR #1733, merged `ee741cf10`), verified against the
exact merged commit (`5e1e9e9ac`) and explicitly deferred out of that merge gate as
pre-existing (it reproduces identically at the round-2 commit; round 3's End-drain fix
only unmasked it):

**A far jump both selects its target and lands it at the top of the window.** The prune
protects `selected_message_id` and stops at the first protected group
(`console_transcript.py`, `_compute_prunable_prefix`), so with the target selected at
the *head*, the prune can never trim anything — while the tailward hydration chain keeps
revealing. Probe on the headline flow (far jump, then scroll down): mounted rows
24 → **490**, virtual height **1966 against a high watermark of 900** (2.18x), stable
after 120 idle frames; clearing the selection with Esc collapses it to 150/603. Bounded
only by session length — a 10k-message session would mount all of it. This is the mirror
image of the disclosed tail-pinned trade-off (which stays bounded because the prune still
trims the head); the head-pinned case is genuinely unbounded. Diagnosable (the prune logs
its blocked walk) but invisible to the user; the only recovery is Esc.

Suggested fixes from the review: refuse `_hydrate_tailward` while
`virtual_size.height >= high_mark` and the prune is blocked (the 15455 loop-breaker
applied to the other boundary), or stop protecting the selection when it is the first
mounted group. Include the review's second residual while in the file: a **one-frame
End-during-prune race** — an End pressed between prune entry and `_restore_scroll` has
its anchor cancelled by `_release_anchor_quietly` (entry state wins), so the drain may
stop after one chunk; pill is up and a second End resumes, but a settle-varied pin would
document the bound.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 With a head-pinned selection, a downward walk keeps total mounted height bounded near the high watermark (probe or test on the far-jump-then-scroll-down flow as evidence)
- [x] #2 The selection and its action row survive whatever bounding mechanism ships (no teleport, no selection loss)
- [x] #3 The 15777 two-sided suite (13), protected pruning suite, and End-drain pins stay green
- [x] #4 The one-frame End-during-prune residual is either closed or pinned-and-documented with its recovery
<!-- AC:END -->

## Implementation Plan

1. Reproduce the head-pinned unbounded reveal as a born-red test: far jump
   (`select_message` on a windowed-out target, which re-centers and pins the
   selection at the window head), then a paced walk down; assert
   `virtual_size.height` stays bounded near the high watermark and the
   selection row + action row stay mounted. Run it RED at HEAD first.
2. Fix shape (the review's suggestion — reading the code confirmed it is the
   only shape compatible with slice-contiguity + the no-eviction rule):
   `_hydrate_tailward` refuses to reveal another chunk while the measured
   height is at/over the high mark AND the prune walk is blocked (empty
   `_compute_prunable_prefix`) — hydration must not outrun a prune that
   cannot make room. Selection stays mounted and highlighted; recovery for
   full downward reachability is Esc (clear selection → prune unblocks →
   hydration resumes) or the jump pill, both pinned.
3. Consequence, documented: with a head-pinned selection HELD, the downward
   walk now stalls bounded instead of mounting to the tail — contiguity
   (one slice), a mounted selection, and a mounted far tail are mutually
   exclusive, so full reachability under a held head-pinned selection is
   mathematically impossible.
   `test_far_jump_then_scrolling_down_walks_back_to_the_tail` is updated to
   the new contract (clear the selection, then the walk reaches the tail);
   the head-pinned stall + bound + Esc-resume is pinned by the new test.
   Finding D's accepted twin (tail-pinned selection pauses the slide,
   prune still bounds) is upward-hydration territory — untouched.
4. End-during-prune race: stamp `scroll_end` with a monotonic intent time;
   the prune's restore, on seeing an entry-detached reader whose raw anchor
   is engaged at restore time, honors a stamp NEWER than prune entry — skip
   the quiet release and the offset compensation, re-arm the drain — instead
   of cancelling the user's End. Entry-state restore for every other case is
   byte-identical (round-3 semantics preserved). Born-red pin injects End in
   the entry→restore window via a wrapped `_run_prune_check`.
5. Verification: new pins born red at HEAD; two-sided suite green; protected
   `window_reconcile` + `windowing` suites green AND byte-unmodified;
   pruning suite green; ruff on touched files.

## Implementation Notes

Both findings closed, on base `ecbcd5cd8`, commit `aefd3b60e`.

**Finding 1 — head-pinned selection (the review's suggested shape shipped).**
`_hydrate_tailward` now refuses to reveal another chunk while the measured
`virtual_size.height` is at/over the high mark AND the prune walk is blocked
(`_compute_prunable_prefix` returns empty) — hydration must not outrun a
prune that cannot make room. Reading the code confirmed no better boring
option exists: contiguity (one mounted slice by construction) + a mounted
selection + a mounted far tail are mutually exclusive, so the only
alternatives were evicting the selection (review D's teleport, forbidden) or
islands (rejected by 15455). The refusal logs a debug line mirroring the
prune's and trim's blocked-walk logs. Two consequences, both deliberate:

- With the selection HELD, the downward walk stalls BOUNDED at ~high + one
  chunk; Esc (clear selection → the prune trims → real boundary gestures
  resume the slide) or the jump pill restore full reachability. This is the
  exact mirror of finding D's ACCEPTED twin (tail-pinned selection pauses
  the slide upward while the prune bounds height) — that twin is upward
  territory and is untouched (its pin stays green).
- `test_far_jump_then_scrolling_down_walks_back_to_the_tail` reached the
  tail only BY the unbounded mount (selection held the whole walk), so it
  was updated to the new contract: clear the selection before the walk (and
  walk with `action_page_down`, a real gesture). The held-selection stall +
  bound + Esc recovery are pinned in the new suite.

Trap found while landing it: the refusal's `_compute_prunable_prefix` walks
`self.children`, and the first version ran it OUTSIDE `_refresh_lock` — read
mid-reconcile (a prune's), the transient child order faked "blocked" and
stalled a selection-free drain. The whole hydrate decision now runs under
the lock, with the guards re-checked after acquiring it.

**Finding 2 — End-during-prune race: CLOSED (clean shape existed).**
`scroll_end` stamps `_scroll_end_intent_time = monotonic()`; the prune
captures `entry_time` beside its anchor-state capture, and the restore's
entry-detached-but-now-engaged branch treats a stamp newer than entry as
the user's End: keep the anchor, skip the stale entry-offset compensation,
re-arm the drain (`_schedule_tailward_hydration`). Every other restore path
is byte-identical, so the round-2/3 faithful-restore semantics survive (the
detached-reader pin and the End-drain pin both stay green). The racing
interleaving, pinned deterministically: `Widget.scroll_end` engages the raw
anchor synchronously but DEFERS its scroll via `call_after_refresh`, so an
End injected during the prune's reconcile enqueues its scroll AHEAD of
`_restore_scroll` — the restore had the last word and quietly cancelled the
user's drain (218 of 400 messages stranded, second End resumed).

**Evidence.**
- Born-red on the unmodified base widget (all three, re-run against
  `git show ecbcd5cd8:` of the widget after the fix was committed):
  walk-down height escaped to 1966 vs the 900 high mark (the review's exact
  number) at 490 mounted rows; the stall pin reached m499 on an unbounded
  slice; the End-race pin stranded 218 messages at last=m181.
- Post-fix: walk-down peaks at 934 virtual rows (900 high + one ~34-row
  chunk overshoot), 232 mounted rows, stable across 20 idle frames; the
  selection row stays first + mounted + selected with its action row; Esc
  resumes the slide to m499 bounded (≤260 rows).
- Suites: new pins 3/3; two-sided 13/13; protected `window_reconcile` +
  `windowing` 17/17 green AND byte-unmodified (`git diff` empty); pruning
  8/8 (41 total green). Adjacent transcript set 153 passed + the 3
  speak-action reds reproduced on the UNMODIFIED base widget in the same
  run (pre-existing, per the 15777 notes). ruff clean.

**Files.** `tldw_chatbook/Widgets/Console/console_transcript.py`,
`Tests/UI/test_console_transcript_selection_prune_bound.py` (new, 3 pins),
`Tests/UI/test_console_transcript_two_sided_window.py` (walk-back test
updated to the new contract), `Docs/User_Guide/console.md` ("Long
conversations" prose + stamp), `backlog/docs/lessons-testing-evidence.md`
(bare-scroll_to walk trap + children-walk-under-lock twin), this file.
