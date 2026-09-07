---
id: TASK-15455
title: 'Console transcript: windowed mount for long-conversation load'
status: Done
assignee:
  - codex
created_date: '2026-08-11 12:05'
labels:
  - perf
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: session resume loads the entire persisted tree (`depth_cap=10_000`) and the first `refresh_messages` mounts every row via individual awaited `mount()` calls — no batching, no windowing (`Widgets/Console/console_transcript.py:2283-2301`; old rows likewise removed one awaited `remove()` at a time). Height-watermark pruning runs only after first layout, so a long conversation pays full mount plus full-history Markdown parse (one Textual Markdown widget per assistant row, one child widget per markdown block) before anything is trimmed — and up to the 12k-20k-line watermarks stay mounted permanently, which also inflates every reconcile pass (task-15453) and layout.

Fix direction: mount a tail-first window (bottom N lines) and hydrate scrollback lazily on scroll; batch mounts. This is structural — stability first: anchor()/tail-follow semantics (`:1295/:1344/:1399-1408`), selection, pruning, and branch navigation must be pinned by tests before the windowing lands. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Loading a 500+-message conversation mounts only the visible tail window initially (evidence + session-switch latency before/after)
- [x] #2 Scrollback hydrates on demand without breaking anchor/tail-follow, selection, or branch navigation (tests)
- [x] #3 Prune watermarks still bound total mounted height
<!-- AC:END -->

## Implementation Plan

1. Characterize the existing transcript refresh, pruning, selection, branch-navigation, and viewport-follow contracts with focused tests and latency instrumentation.
2. Add a bounded tail-first projection and batched transcript reconciliation in `UI/Console_Modules/`, preserving the transcript widget's public API and current region ownership.
3. Hydrate earlier messages when the viewport reaches the scrollback boundary while restoring the reading anchor and retaining selection and branch behavior.
4. Verify the 500+ message resume path, hydration behavior, and mounted-height watermarks with focused automated tests and before/after timing evidence.
5. Run affected Console suites, static analysis, and a self-review; document implementation notes and complete the acceptance criteria only when the evidence supports them.

ADR required: no

ADR path: N/A

Reason: This is an internal performance refinement of the existing Console transcript-region and message-store contracts; it does not change persistence, ownership, security, or a cross-module service interface.

## Implementation Notes

> **Two independent implementations.** This task was implemented twice in
> parallel on 2026-08-11 by two sessions that never saw each other's work
> (neither ran the `gh pr list --search` check the backlog-hygiene lesson
> prescribes — see the incident recorded there). PR #1538 (branch
> `codex/console-transcript-windowing`, commit `84baf5689`) merged first and,
> per the repo's merged-first rule, owns the design. The second implementation
> (branch `task/15455-windowed-mount`, PR #1556) was closed unmerged; its
> review-hardened findings were ported onto the merged design in this PR. Both
> sessions' work is credited below.

### Part 1 — merged implementation (PR #1538), design of record

- Viewport-scaled, turn-aligned tail projection built before row signatures or
  Markdown widgets. The complete history stays in the transcript model for
  export, persistence, selection, and branch operations.
- Coalesced scrollback hydration on upward scroll, wheel-at-boundary, and Page
  Up; prepended rows compensate the scroll offset by measured virtual height.
- Batched reconciliation (`remove_children` + multi-widget `mount`), reorder
  contract intact.
- ONE contiguous hidden prefix: pruning and windowing share
  `_pruned_message_ids`, so hydration is simply the boundary moving back.
- Probe on that session's host: 19.85 s -> 0.24 s for a 500-message swap.
- Tests: `Tests/UI/test_console_transcript_windowing.py` (4), green and
  unmodified here.

### Part 2 — reconciliation delta (this PR)

Ported from the closed implementation after its review round, adapted to the
merged design. What was **dropped**: that implementation's 40-message window
(the viewport-derived line budget is better), its separate unhydrated/pruned
sets, its `_scrollback_protected` latch, and its gap-seam row — see below.

- **Prune/hydration fixed point (the headline).** Measured on the merged build
  before this change: with the reader at the scroll boundary and a 45/70
  watermark configuration, the hidden prefix oscillated between 169 and 152
  messages (height 47 <-> 115) on every idle frame, indefinitely, with no user
  input — the prune's own scroll restoration lands back at the boundary and
  re-triggers the hydration that produced it. Automatic hydration is now
  refused at/above the LOW watermark; since the walk fires only above the HIGH
  mark and always leaves the remainder above the low one, a prune can never
  restore a hydratable state. Explicit `_hydrate_scrollback()` stays ungated
  (the merged watermark test calls it directly and still passes).
- **`restore_reading_state`** now reveals the selection it assigns directly,
  and applies the restored offset only AFTER those rows mount — clamping
  against the pre-reveal `max_scroll_y` silently drops the reader elsewhere.
  (This is the mirror of the read-order defect the second implementation's
  re-review found in its own latch.)
- **Config surface + kill switch**: `[chat_defaults] transcript_window_lines` /
  `transcript_scrollback_lines` are floors over the viewport-derived budget;
  `transcript_window_lines = 0` mounts the whole history as before this task.
  Documented in `config.py` and the Console user guide ("Long conversations",
  including the caveat that scroll-back stops once the mounted view reaches the
  watermark).
- **`reveal_message()`** factors the boundary move out of `select_message` so
  selection, the task-501 swipe handoff, and reading-state restore share one
  implementation.
- **No gap-seam row was ported.** It was necessary in the closed design, whose
  pruned and unhydrated sets were independent and could mount two islands with
  no seam between them. Here the hidden set is always a PREFIX and every reveal
  path extends the same boundary, so mounted rows are one contiguous suffix —
  now pinned by a test rather than asserted in a comment.
- **Correction to the porting brief**: `select_message` on a windowed-out id
  was reported missing from the merged implementation; it is present and works
  (probe: selecting `m10` of 500 reveals and mounts through it, action row
  included). Nothing was ported; a regression test was added, since the merged
  suite had none.

**Verification (this PR).** 12 new tests in
`Tests/UI/test_console_transcript_window_reconcile.py`, of which 6 were born
red on the merged build (fixed point, restore reveal, restore offset, config
resolution, kill switch, configured floors) and 6 are pins that were already
green (load bound, session switch, watermark bound, reveal-on-select,
contiguity). Merged suite 4/4 green unmodified; transcript suites 52 passed;
broader console suites (native transcript, markdown, fence throttle, sibling
nav, composer collapse) 195 passed; ruff clean. Load probe (500 messages,
120x40, medians of 3): 10.9 s with everything mounted vs 1.08 s windowed, 1002
-> 58 rows and 5502 -> 310 widgets. Delta overhead against a merged-equivalent
arm (window budget patched back to the constants, same build, same window
size): 357.3 ms vs 360.0 ms medians over 5 runs each — inside run-to-run noise,
no regression.

**Review round (reconciliation).** One Important: the kill switch
(`transcript_window_lines = 0`) forced `window_start = 0` on EVERY ingest,
which cleared the watermark-pruned prefix — an over-watermark session
re-mounted its whole history on each 0.2s sync tick and pruned it back down
(reproduced: 180 rows remounted, settled to 11, every tick). Pre-task code kept
pruning sticky across ingests; the disabled branch now carries the preserved
boundary forward (`0 if preserved_start is None else preserved_start`), so a
fresh/disjoint ingest still mounts everything while the watermarks keep their
prefix. Regression test runs at churn-triggering marks (45/70, 180 messages,
repeated ingests, instrumented mounts) — the default-watermark kill-switch test
cannot see this, because nothing is ever pruned there. Minors: the
"still hydrates" test now runs with watermarks ENABLED so the new gate is
actually evaluated, and the guide states that the two line settings are FLOORS
under `viewport x 6` (inert at the shipped 144 on any terminal >= 24 rows) and
names the LOW watermark as the scroll-back ceiling.

**Residuals, untracked (for the controller to file).**
1. *Unbounded reveal.* Selecting/jumping to a message near the start of a long
   session reveals everything from it to the tail in one pass — measured ~490
   rows mounted for `m10` of 500. Inherited from the merged design, not
   introduced here, and latent for any future "jump to search hit" feature.
2. *Scroll-back reachability ceiling.* Once the mounted view reaches the low
   watermark (~12,000 rendered rows by default) scroll-back stops loading
   older history; the content is reachable only via export or a jump. Removing
   the ceiling needs two-sided windowing (trim the tail as the head grows),
   which neither implementation has and which would touch tail-follow,
   streaming, and the jump pill — a design task, not a tweak.

**Files (this PR).** `tldw_chatbook/Widgets/Console/console_transcript.py`,
`tldw_chatbook/UI/Console_Modules/transcript.py`, `tldw_chatbook/config.py`,
`Docs/User_Guide/console.md`,
`Tests/UI/test_console_transcript_window_reconcile.py`, this task file, and
`backlog/docs/lessons-backlog-hygiene.md`.

**ACs re-verified against the combined result**: #1 (500-message load mounts a
bounded tail — probe + `test_pin_long_load_is_bounded_...`), #2 (scrollback
hydrates without breaking anchor/tail-follow, selection, or branch navigation —
merged suite + the reveal/restore tests here), #3 (watermarks still bound the
mounted height — `test_pin_watermarks_still_bound_the_mounted_height`, and the
fixed-point test proves they do so without churning).

### Appendix — merged session's original notes


- Added a viewport-scaled, turn-aligned tail projection before row signatures or Markdown widgets are built. The complete history remains in the transcript model for export, persistence, selection, and branch operations.
- Added coalesced scrollback hydration for upward scrolling, wheel-at-boundary, and Page Up. Prepended rows compensate the scroll offset by their measured virtual height, preserving the reader's visible message and detached follow state.
- Reconciliation now removes stale direct children and mounts contiguous missing runs through Textual's batch DOM APIs. The existing reorder contract remains intact.
- Kept the implementation in the existing `ConsoleTranscript` leaf rather than adding a new `UI/Console_Modules/` controller: the window is internal view state of the actual scroll container, and moving it outward would split the established pruning, selection, row-signature, and reconciliation invariants. No screen-size ratchet growth was introduced.
- Identical 500-message Markdown session-swap probe at 100x30: merged baseline `19.8457s`, 500 mounted messages; windowed implementation `0.2411s`, 22 mounted messages and 478 lazily retained messages. This is an 82x speedup and 98.8% lower elapsed time on the test host.
- Verification: 4 focused windowing tests, 17 combined windowing/pruning/tail-follow contracts, and 127 broader native-transcript/Markdown/selection/diff/jump/region test bodies passed; two mutation checks killed removal of initial windowing and hydration; Ruff, `py_compile`, and `git diff --check` passed. The repository pytest wrapper cannot bootstrap asyncio on this Windows host because its process-wide network guard intercepts Python's loopback `socketpair`; the same test bodies were therefore executed directly in isolated Textual harnesses.
- Added the reorder/window lesson to `backlog/docs/lessons-testing-evidence.md` after the broader regression pass caught a boundary-preservation defect that focused tests missed.
- Modified: `tldw_chatbook/Widgets/Console/console_transcript.py`, `Tests/UI/test_console_transcript_windowing.py`, this task, and the testing-evidence lessons document.
- ADR required: no. The existing Console transcript-region and leaf-widget ownership remain unchanged.
