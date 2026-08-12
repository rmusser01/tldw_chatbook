---
id: TASK-15455
title: Console transcript: windowed mount for long-conversation load
status: Done
assignee:
  - '@claude'
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

<!-- SECTION:PLAN:BEGIN -->
1. **Pin first (no production change).** New pin suite covering the invariants
   windowing could break on a LARGE history (60+ messages, which the existing
   suites never exercise — their biggest fixture is 30): anchor sits at the
   bottom after load, tail message is mounted, selection + action row on an old
   row, prune watermarks bound the height, branch/variant swap and session
   switch reconcile, jump pill + `jump_to_latest`. Run green BEFORE touching
   production code; commit as its own slice.
2. **Batch the reconciler's DOM churn** (no visible behavior change): contiguous
   newly-built rows mount in ONE `mount(*widgets, after=...)` call; stale rows
   are removed with one `remove_children([...])`. Keeps the task-15453
   already-in-position induction intact.
3. **Tail-first window.** `set_messages` detects a fresh load (first ingest or
   an id-set disjoint from the last one = session switch) and marks all but the
   newest N messages "unhydrated"; `_transcript_rows` filters them exactly like
   the task-1365 prune window. N derives from a message cap AND an estimated
   line budget (`[chat_defaults] transcript_window_messages` /
   `transcript_window_lines`, `<= 0` disables = kill switch).
4. **Hydration on scroll-up.** `watch_scroll_y` schedules a coalesced check;
   near the top (or when the window is too short to scroll) the next chunk of
   older messages hydrates, restoring the reader's offset by the measured added
   height (mirrors the prune path's `_restore_scroll`).
   **Anti-oscillation, structural:** hydration is refused unless the virtual
   height is BELOW the low watermark, and pruning only ever fires ABOVE the high
   watermark leaving the remainder strictly above the low watermark — so a prune
   can never re-enable hydration, and no hydrate→prune→hydrate cycle exists for
   ANY watermark configuration. Second, independent guard: ids hydrated on
   demand are protected from pruning while the reader is detached from the tail,
   so scrollback never vanishes under the reader; protection lifts the moment
   they follow the tail again, which is what keeps AC#3 true.
5. **Jump/selection into unhydrated history**: `ensure_message_hydrated()`
   hydrates the target's window before selection/focus so programmatic jumps
   land on a mounted row.
6. **Evidence**: isolated 500-message session-load probe (windowed vs kill
   switch, same build), mounted-row-count bound test, hydration order tests, the
   oscillation-impossibility test, all pinned suites green unmodified.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A conversation LOAD now mounts a tail-first window instead of the whole
persisted tree, and older messages hydrate as the reader scrolls back. Four
commits, pins first.

**Pins first (6517267c).** The pre-existing transcript suites top out at 30
messages, so none of them could have noticed a 40-message window. Eight pins
were written and run green against unmodified dev code on a 60-message load
(anchor lands at the tail, selection mounts the action row, watermarks bound
the height, branch swap keeps the shared prefix rows, variant navigation,
session switch, jump pill + `jump_to_latest`, exports render full history).
All eight still pass unmodified.

**Batching (dce47985).** The reconciler now mounts contiguous new rows with
one `mount(*widgets, after=…)` and removes stale rows with one
`remove_children([...])`. Textual inserts a multi-widget mount in argument
order, so child order is unchanged; the batch is flushed before any decision
that reads the real child list, which keeps the task-15453 already-in-position
induction exactly as it was.

**Windowing + hydration (9d2b3147).** `set_messages` (re)establishes the
window when the ingest is a load — the first one, or one whose ids are
disjoint from the last (a session switch); a tick/send/branch swap shares ids
and keeps whatever the reader hydrated. Window size = message cap AND an
estimated line budget (`[chat_defaults] transcript_window_messages` /
`transcript_window_lines`, `<= 0` disables). Unhydrated ids are filtered in
`_transcript_rows` exactly like the task-1365 prune window, but reversibly:
`watch_scroll_y` schedules a coalesced check that hydrates the next step when
the reader is within a viewport of the top (or the window is too short to
scroll), restoring their offset by the measured height added above them.

**Anti-oscillation is structural, and three things had to be true, not one.**
(1) Hydration is refused at/above the LOW watermark while pruning only fires
above the HIGH one and always leaves the remainder above the low one — so a
prune can never restore a hydratable state. (2) Each step is sized against the
headroom under the low mark: *measured during development*, without this a
3-message window under a 20/40 config mounted its full 10-message chunk,
crossed the high mark, and had 9 of those 10 pruned away one tick later —
wasted mounts, and permanently unreachable history. (3) Rows the reader
hydrated are protected from the walk by an explicit `_scrollback_protected`
latch that drops when they return to the tail. The latch replaced a sampled
`_is_following_tail()` check after that version silently failed: `anchor()`
does not clear Textual's `_anchor_released`, so the prune check scheduled by
`jump_to_latest` ran while the widget still read as detached and skipped the
reclaim entirely.

Jump targets outside the window hydrate from the target forward
(`ensure_message_hydrated`, wired into `select_message` and the task-501
swipe handoff), keeping mounted rows one contiguous suffix. Keyboard
selection walks the mounted window only.

**Trade-off, deliberate:** while the latch holds (reader parked in hydrated
scrollback), the watermark walk cannot trim, so a marathon run streaming into
a transcript the reader has left behind can grow past the high mark until they
return to the tail. Pruning content out from under a reader mid-read is worse.

**Evidence.** Isolated 500-message load probe (bare transcript, 120x40,
production-like row CSS, medians of 3): per-row mounts of every row (dev
behavior, emulated) 47.2 s / 1002 rows / 5502 widgets → batched only 14.1 s →
windowed + batched 2.1 s / 82 rows / 442 widgets. The ratio is the finding;
absolute numbers are harness-specific. Tests: 27 in the new suite, each of ten
mutations of the new mechanisms killed by a targeted test; transcript surface
(13 suites) 269 passed with one pre-existing failure
(`test_console_citation_sources` stub lacking `set_presentation_context`);
Chat-side console suites 119 passed; config/decomposition 204 passed with the
known pre-existing `test_console_left_rail_sections_use_available_space`.

**Suites (read counts).** New window suite 27 passed; transcript surface (13 suites)
269 passed + 1 pre-existing failure; Chat-side console suites 119 passed;
native_transcript + native_chat_flow 404 passed / 1 xfailed; parallel_runs +
composer_collapse + realtime_wiring 141 passed (one non-reproducible collapse flake on
the first run — that fixture seeds 24 messages, below the window, so this branch is
inert in it); config/decomposition 204 passed + 1 pre-existing failure; ruff clean.

**Files:** `tldw_chatbook/Widgets/Console/console_transcript.py`,
`tldw_chatbook/config.py`, `Docs/User_Guide/console.md`,
`Tests/UI/test_console_transcript_windowed_mount.py`.
<!-- SECTION:NOTES:END -->
