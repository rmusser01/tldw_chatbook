---
id: TASK-22219
title: Gate the file-notes filesystem reconcile on screen visibility
status: Done
assignee:
  - '@claude'
created_date: '2026-08-24'
updated_date: '2026-08-25 15:57'
labels:
  - performance
  - library
  - notes
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22219).

Pre-existing. `Widgets/Library/library_file_notes_workspace.py:1453-1456` arms a 1.5 s
`set_interval` (`pause=False`) whose fire runs `to_thread(service.reconcile)` — a
walk/stat of the notes root — 40x/minute for the Library screen's lifetime once the File
Notes surface has been opened, including while other screens or modals are on top (the
only gates are `_active`/transitioning/in-flight; no `screen.is_active`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No reconcile fires while the Library screen is not active or is covered (probe)
- [x] #2 Polling resumes on return to the screen; a change-driven or backoff cadence is considered and the choice stated
- [x] #3 Filesystem scan count per minute measured before/after in the covered state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify Textual 8.2.8 Screen.is_active semantics against the installed source (top-of-stack only; False under a pushed modal and when off the stack)
2. Red-first probes in Tests/UI/test_library_file_notes_workspace.py: (a) covered-by-modal -> zero reconcile fires over a multi-interval window; (b) covered-by-pushed-plain-screen -> zero fires; (c) resume -> reconcile fires within one interval of the cover being popped
3. Implement: add 'not self.is_attached or not self.screen.is_active' to the _start_poll gate (precedent: UI/Navigation/main_navigation.py:469); timer keeps ticking, fire early-returns while covered, so every return path resumes on the next tick with no pause/resume bookkeeping
4. Measure covered-state scans/min before/after at the production 1.5s cadence (scratch probe)
5. Targeted file-notes suites + --collect-only sweep (tee'd), ./scripts/preflight.sh
6. Mutation-test: remove the visibility gate -> covered probes red; force the gate to always skip -> resume probe red
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-condition visibility gate added to `_start_poll`
(`Widgets/Library/library_file_notes_workspace.py`): the 1.5 s timer fire now
also early-returns on `not self.is_attached or not self.screen.is_active`,
before any worker is spawned — so `to_thread(service.reconcile)` (the notes-root
walk/stat) never runs while the workspace's screen is covered by a pushed
modal/plain screen or is otherwise not top-of-stack. Precedent gate:
`UI/Navigation/main_navigation.py` overflow tick.

**Cadence decision (AC #2):** visibility gate + unchanged fixed 1.5 s cadence.
Change-driven watching (FSEvents/watchdog) or a backoff schedule was considered
and rejected: the finding's entire cost was the covered state, which the gate
removes completely (40/min -> 0/min); a watcher adds an optional dependency and
platform-specific failure modes, and backoff adds visible-state staleness — the
owner's stability-over-cleverness ruling applies.

**Resume semantics (AC #2):** the timer is never paused — each covered fire is a
no-op costing a property check — so the resume path IS the still-ticking timer:
the first tick after the screen is active again runs the catch-up reconcile,
within one poll interval (measured 0.96 s after popping the cover at the
production 1.5 s cadence). No pause/resume bookkeeping exists to miss a return
path (screen resume and modal dismiss are both covered by the same re-admission).
The tab-switch-away case needs no gate: `switch_screen` unmounts the Library
screen, `on_unmount` stops the timer, and on return the remounted workspace's
`_initialize` resuming path already performs an immediate `service.reconcile`
catch-up before polling restarts.

**Textual 8.2.8 semantics (verified in installed source):** `Screen.is_active`
is `self.app.screen is self` (screen.py:557) where `App.screen` is
`_screen_stack[-1]` (app.py:1627) — False both under a pushed modal AND for a
plain pushed screen; `Screen.is_current` would wrongly stay True for background
screens and was not used. `is_attached` (message_pump.py:270) guards the
`self.screen` walk; `not self._active` stays first in the gate so a straggler
fire during unmount never touches `self.screen`.

**Measurements (AC #3, production 1.5 s cadence, covered by a modal):**
before 10 reconcile fires / 15 s = 40.0/min; after 0 / 15 s = 0.0/min.

**Probes/tests:** new parametrized
`test_poll_reconcile_pauses_while_covered_and_resumes_on_return`
(ModalScreen + plain Screen covers) in
`Tests/UI/test_library_file_notes_workspace.py` — red-first against the old
code (12x and 13x covered fires in 12 intervals), green after. Mutation
results: gate removed -> both covered probes red (12x fires); visibility leg
forced to never re-admit -> probe red at "poll did not reconcile while the
screen was active". Targeted suites: workspace+git+git_push 336 passed /
2 failed and journey+modal-dismissal+session-owner-lifecycle 211 passed /
1 failed — all 3 failures proven pre-existing dev reds unrelated to this
change (theme-contrast `high_contrast_yellow_black` 'Save failed' 3.33:1,
reproduced with the gate Edit-reverted; and an AST pin on
`LibraryScreen.on_unmount` broken by a dev-side `_library_conversation_
reader_mounted_authority` assignment — file untouched here). Whole-tree
`--collect-only`: 59,368 collected; 28 errors, all `No module named 'numpy'`
optional-deps suites (numpy absent from this venv). `scripts/preflight.sh`
fully green.
<!-- SECTION:NOTES:END -->
