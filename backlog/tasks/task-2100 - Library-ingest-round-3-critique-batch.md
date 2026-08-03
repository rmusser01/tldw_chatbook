---
id: TASK-2100
title: >-
  Library ingest round-3 critique batch (scroll-yank, vanishing focused title, P2 sweep)
status: In Progress
assignee: []
created_date: '2026-08-03 21:00'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description (the why)

The round-3 dual-agent critique (snapshot `2026-08-03T20-28-31Z…`, 29/40 —
trend 21 → 24 → 29) found the remaining defects are interaction mechanics:

1. Retry / Dismiss / Clear-finished handlers keep trailing belt-and-braces
   `refresh(recompose=True)` calls that bypass the task-2042 in-place path
   — pressing them yanks the viewport off the queue, stranding the armed
   "Press again to clear N finished" confirm off-screen.
2. The focused options/Recent collapsible title's label VANISHES: the
   app-wide `Collapsible > CollapsibleTitle:focus { border-bottom: solid }`
   (bundle :8114) collides with the ingest-scoped `height: 1` — the border
   consumes the only row.
3. P2 sweep: solo-unsupported copy contradiction ("will be recorded as a
   failure" beside a blocked submit); server-mode sentence greets
   local-only installs (availability gated on the service seam, not a
   configured server); unsupported files counted but never named
   pre-submit; "Recent ingests" renders as an empty unlabeled shell after
   a clear; Encoding is a free-text Input where a Select prevents error.

Diagnosis record (revised during implementation): the round-3 "Clear
button dead to mouse" WAS a real defect, but not the suspected stale
hit-region — `handle_library_ingest_clear_path` was the one handler the
task-2042 sweep missed: it still ran a whole-screen
`refresh(recompose=True)`, so a press replaced every canvas widget
mid-interaction (a re-queried value LOOKED cleared in the first harness
diag, which mis-reclassified the finding as driving noise; a second diag
pinned object identity and caught it). Live driving carried a compounding
trap: computing click columns from `grep -bo`/`wc -c` returns BYTE
offsets, and the canvas rows are dense with 3-byte box-drawing glyphs —
clicks aimed ~12 columns right of the button. The `bold reverse`
Button:focus style IS applied (`rich_style` reverse when focused; the
live "no change" reports were attribute-blind glyph diffs) — pinned by a
computed-style regression test.

## Acceptance Criteria (the what)

- [x] Pressing Retry, Dismiss, or the Clear-finished confirm updates the
      queue in place: canvas scroll holds and the form widgets keep
      identity (the armed clear label stays where the user is looking).
- [x] A focused collapsible title on the ingest canvas keeps its label
      visible (glyph-level, monochrome capture).
- [x] Clear-button mouse click and Button:focus reverse styling are pinned
      by regression tests (click-based; computed-style-based).
- [x] A solo unsupported file no longer sees "will be recorded as a
      failure" beside a blocked submit — the copy matches the gate.
- [x] The server-mode sentence renders only when a server is actually
      configured, not whenever the seam exists. Runtime `server_configured`
      alone cannot carry this (live-verified: the shipped config template
      pre-fills `[tldw_api]` with a placeholder URL+token, so it is True on
      a virgin profile, and reachability normalizes to "unknown" whenever
      the runtime is local) — the gate additionally treats an untouched
      template binding as not-configured.
- [x] The unsupported-files forecast names the files (first few basenames).
- [x] "Recent ingests" does not render when there is nothing recent.
- [x] Encoding is a Select of known encodings.
- [x] Investigated-with-notes (fix if root cause surfaces cheaply): stray
      `│` fragments on gap rows after mouse activity (both round-3 agents,
      survives resize repaint; no matching focus-border rule found in the
      bundle on first sweep).

## Implementation Plan (the how)

1. Swap the trailing `refresh(recompose=True)` in the retry / dismiss /
   retry-faster-whisper / clear-finished handlers for
   `_update_library_ingest_dynamic_regions()`.
2. Scope a `border-bottom: none; text-style: bold reverse` title-focus
   rule to `LibraryIngestCanvas` in `_agentic_terminal.tcss`; regenerate
   the bundle.
3. State-model additions: `unsupported_line` (named files, gate-matched
   copy); canvas renders it and hides empty Recent; screen gates
   `server_ingest_available` on a real (non-placeholder) server binding;
   capabilities flip encoding to a Select.
4. Regression tests: in-place retry (widget identity), focused-title
   geometry, Clear-path click (identity-pinned), Button:focus
   `rich_style.reverse`, server-line gating incl. the placeholder
   binding.
5. Live-verify every AC on a fresh isolated profile; targeted suites +
   `--collect-only` sweep.

## Implementation Notes

**Approach.** Three handler swaps plus one newly-found missed handler;
one CSS collision fixed at its true specificity; the P2 sweep landed in
the state model so copy and gating stay testable without a UI.

**Core changes.**
- `library_screen.py`: retry/dismiss/retry-faster-whisper/clear-finished
  handlers now end in `_update_library_ingest_dynamic_regions()` (no
  trailing full recompose). `handle_library_ingest_clear_path` — the
  2042 sweep's missed handler and the real "mouse-dead Clear" — now
  clears the Input value directly, refocuses the path field, and updates
  in place. `server_ingest_available` additionally requires runtime
  `server_configured` AND a non-placeholder `[tldw_api]` binding
  (`_server_binding_is_shipped_placeholder`; exact template values, fails
  open on drift).
- `library_ingest_state.py`: new `unsupported_line` field — solo-blocked
  copy `Unsupported: a.xyz, b.xyz.` matches the gate; mixed-batch copy
  names the first 3 basenames and keeps the recorded-as-failures phrasing.
- `library_ingest_canvas.py`: renders `state.unsupported_line`; Recent
  collapsible only composed when `state.recent_jobs` (deliberate flip of
  the earlier always-visible contract, per round-3 empty-shell evidence).
- `ingest_capabilities.py`: encoding → `type="select"`, options
  `auto/utf-8/utf-16/latin-1/cp1252`.
- `_agentic_terminal.tcss` (+ regenerated bundle): scoped
  `border-bottom: none; text-style: bold reverse` title-focus rules —
  BOTH the plain and `.-collapsed` variants, because the app-wide
  `Collapsible.-collapsed > CollapsibleTitle:focus` border rule outranks
  a purely type-scoped fix at (0,2,2) vs (0,1,3) and every type-group
  panel starts collapsed.

**Verification.** Harness: 236 canvas/state/runner + 104 shell-subset
tests green; 29,105 collect cleanly. New tests are mutation-honest: the
title test failed until the `.-collapsed` variant landed; the Clear test
pins widget identity so the recompose version fails it. Live (fresh
profile, unique socket, identity-verified pane): focused title label
visible in bold reverse (ANSI-verified); Retry holds the viewport (Queue
row identical before/after, retry count increments); Clear-finished arms
in place and clears on second press; server line absent on a virgin
profile; `Unsupported: data.json.` named forecast; Recent absent when
empty; Encoding renders as `auto ▼`; Clear click clears path and drops
the stale summary.

**Stray `│` fragments (investigation note).** No vertical-border rule
scoped to the canvas area was found in the bundle; the two focus-border
rules removed by this task drew horizontal rules, not `│`. Not reproduced
under the harness. Mechanism still unknown — but the round-3 live
sessions drove clicks with byte-offset column math (see diagnosis
record), so misplaced presses on gap rows are a plausible contaminant.
Re-check at the next critique before treating it as an app defect.

**Trap recorded for live driving:** compute tmux click columns with
character indexing (python `str.find`), never `grep -bo`/`wc -c` — the
TUI's box-drawing glyphs are 3 UTF-8 bytes each, so byte math lands
clicks ~12 columns off in the canvas region.
