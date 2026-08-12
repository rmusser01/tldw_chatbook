---
id: TASK-15455
title: Console transcript: windowed mount for long-conversation load
status: In Progress
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
- [ ] #1 Loading a 500+-message conversation mounts only the visible tail window initially (evidence + session-switch latency before/after)
- [ ] #2 Scrollback hydrates on demand without breaking anchor/tail-follow, selection, or branch navigation (tests)
- [ ] #3 Prune watermarks still bound total mounted height
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
