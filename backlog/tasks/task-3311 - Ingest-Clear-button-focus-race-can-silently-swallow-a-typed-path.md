---
id: TASK-3311
title: Ingest Clear-button focus race can silently swallow a typed path
status: Done
assignee: []
created_date: '2026-08-08 00:30'
updated_date: '2026-08-09 03:58'
labels:
  - library
  - ingest
  - ux
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the live verification of the 3300-3305 arc (2026-08-08, worktree branch feat/media-ingest-ux-parity). Intermittent — 2 of 4 Clear clicks did NOT return focus to the path field; subsequent typing was hijacked: once the path's tail landed in the rail search box, once the typed path vanished entirely (a leading `/` from an unfocused state likely triggered the global "/ focus search" binding). Two controlled retests refocused correctly, so it is a race, not deterministic — plausibly the ⚠ tooling-warning block's relayout racing the post-Clear refocus. Consequence is silent loss of a typed path plus keystrokes running a Library search.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After Clear, focus deterministically lands on the path field even when the preflight/warning region relayouts concurrently (looped live or harness reproduction, not a single pass)
- [x] #2 A typed leading "/" immediately after Clear edits the path, never triggers the focus-search binding
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Root-cause: Clear press with staged preflight takes the STRUCTURAL branch (type_groups shrink) -> _refresh_library_ingest_canvas_preserving_context captures app.focused = Clear button (path_input.focus() is deferred via app.call_later) -> restore targets the NEW hidden Clear button -> Widget.focus() silently NOPs on a non-focusable widget -> focus adrift (rail search / None).
2. Fix: focus the path field synchronously (screen.set_focus) BEFORE mutating state, so the structural capture/restore round-trips #library-ingest-path deterministically.
3. Looped RED test: stage crafted preflight (non-generic group + warning), click Clear, type '/', assert path focused + value=='/' , N iterations; verify RED pre-fix.
4. Mutation check: revert refocus mechanism -> looped test RED.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (found via Textual 8.2.8 source + looped harness repro, NOT the warning-relayout hypothesis): with a pre-flight staged, Clear shrinks the type-group set, so _update_library_ingest_dynamic_regions takes the STRUCTURAL branch -> _refresh_library_ingest_canvas_preserving_context. That helper captures app.focused for its post-recompose restore, but Widget.focus() (the old refocus) defers via app.call_later, so at capture time app.focused is still the just-clicked Clear button. The restore then targets the NEW Clear button, hidden for an empty path, and Screen.set_focus silently no-ops on a non-focusable widget -- focus stays wherever the recompose prune's _reset_focus dropped it (measured headless: the rail search box, matching the live symptom; a '/' from a no-focus state runs the global focus-search binding, matching the other).

Fix: handle_library_ingest_clear_path now focuses the path field SYNCHRONOUSLY via Screen.set_focus BEFORE _update_library_ingest_dynamic_regions, so the capture/restore round-trips #library-ingest-path deterministically; every interleaving (Hide/Blur on the button, prune reset, restore) converges on the path field.

Evidence: new Tests/UI/test_library_ingest_clear_focus.py -- an 8-iteration clear-and-type loop staging a warning-bearing preflight (AC#1) plus a leading-slash test (AC#2). RED pre-fix on iteration 0 with focused=LibraryRailSearchInput (the live symptom); GREEN post-fix; mutation check (revert to deferred path_input.focus()) sends both tests RED.

Files: tldw_chatbook/UI/Screens/library_screen.py (Clear handler), Tests/UI/test_library_ingest_clear_focus.py (new). Battery: 433 passed across the ingest keep-green set.
xhigh review + live-verify round (2026-08-09): the task fixed MISROUTING but not LOSS. Measured live:
5/5 characters typed within ~150ms of Clear vanished; 3/3 landed at >=400ms. Mechanism: the Clear
handler focuses synchronously and then takes the STRUCTURAL branch, whose `refresh(recompose=True)`
is DEFERRED -- the path Input stays mounted and typeable until the rebuild runs, and the rebuild
sources its value from `_library_ingest_form.path` (""), so anything typed in that window dies with
the widget. Fix: `_refresh_library_ingest_canvas_preserving_context` now carries the pre-recompose
path Input OBJECT (plus its value at capture time) into `_restore_library_ingest_canvas_context`;
if that widget's value CHANGED during the window the text is written into the live Input, which
re-enters the ordinary `Input.Changed` seam so the gate/Clear button/intros/pre-flight debounce all
run exactly once. Gating on "changed since capture" is what keeps a deliberate form rewrite under an
untouched field (the retry re-stage) from being undone by a stale echo. Test: a 6-iteration loop in
Tests/UI/test_library_ingest_clear_focus.py that runs the Clear handler to completion and types
BEFORE the pump gets a turn -- `pilot.click`/`pilot.press` cannot express this window (both settle
the pump), which is why the two original tests pass either way. RED pre-fix on iteration 0 with an
empty field; mutation check (disable the rescue) sends it RED again.
<!-- SECTION:NOTES:END -->
