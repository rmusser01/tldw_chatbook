---
id: TASK-1076
title: >-
  Evals: the primary action is a silent no-op and the empty state has no
  ordering
status: Done
assignee: []
created_date: '2026-07-27 16:00'
updated_date: '2026-07-28 00:37'
labels:
  - evals
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during UAT of the Evals screen, walking it as a first-time user.

**The primary action does nothing, silently.** With no bench selected, the Inspector still presents a **Run Bench** button. Clicking it produces no toast, no inline error, no state change — nothing. The user cannot distinguish "I did something wrong", "the app is busy", and "the app is broken". Either disable it with a reason, or have it explain what is missing.

**The empty state offers three competing actions with no primacy.** On first open the rail shows `Benches (0)` / `Datasets (0)` / `Runs (0)` simultaneously, each with its own affordance — `Create sample bench`, `+ New dataset`, `Import…`. Nothing signals that the sample bench is the intended first step, nor that a bench depends on a dataset. A newcomer has to reverse-engineer the model from three equal-looking options.

**The Detail pane's empty text is unactionable at exactly the moment it is shown.** It reads "Select a bench, dataset, or run in the library rail to see its detail here" — while the rail is empty and there is nothing to select. It should acknowledge the zero-data case and point at the one action that helps.

**Unexplained jargon on first contact.** The run header reads `loaded-nouns (sample) 4465779b · raw · K 20 · 4 cells · 0 failed`. For a first-time reader `raw`, `K 20` and `cells` are all undefined, and the screen's own subtitle — "Run and review evaluation jobs" — never says what an eval or a bench is or why one would want either.

**Layout.** At 200 columns the Inspector is roughly 25 characters wide, so its content wraps mid-phrase ("K requested 20 · K returned" / "canary degenerate"). Three panes at that width leaves the rightmost one too narrow for the text it carries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Run Bench` is disabled with a stated reason, or explains what is missing when pressed
- [x] #2 The empty state establishes one obvious first step
- [x] #3 The Detail pane's zero-data text is actionable
- [x] #4 First-contact jargon is defined somewhere reachable from the screen
- [x] #5 The Inspector has enough width for its content at common terminal sizes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read the Evals screen/inspector/library-rail source to find the existing readiness (Ready/Unavailable/Blocked) convention and reuse it rather than inventing a new one.
2. Primary action: keep it truly disabled (Textual disabled Buttons never emit Pressed) and add an always-visible inline status badge + reason line mirroring EvalsInspector's target-readiness rows, so the reason is reachable without hovering.
3. Empty state: mark the sample-bench affordance as the recommended first step only when the WHOLE rail is empty (no benches, classic tasks, datasets, or runs), leaving the other affordances untouched.
4. Detail pane: distinguish "nothing selected, but rows exist" from "the library is genuinely empty" and point the latter at the sample bench.
5. Jargon: add a tooltip to the run header's meta line (raw/K/cells) rather than growing the header text.
6. Layout: measure #lab-inspector's actual rendered width via a throwaway script, confirm it was hard-fixed at 30 cells at every terminal size (the same anti-pattern already fixed for #lab-rail), and make it fractional with bounds, checking the shared Lab-mode test files (Models/Speech) are not broken.
7. Add a test per behaviour change and revert-check each one (temporarily disable the change, rerun, observe the failure, restore).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all five UAT findings, following the existing readiness (Ready/Unavailable/Blocked) vocabulary already in EvalsInspector rather than inventing a new one.

1. Primary action (evals_screen.py): kept the button genuinely disabled (Textual disabled Buttons never emit Pressed -- "disable + explain on click" is not a valid combination) and added two always-visible Statics (#evals-primary-action-status, #evals-primary-action-reason) mirroring EvalsInspector's own target-readiness row format ("{label}: Blocked" + a callout), so the reason is reachable without a mouse hover, not just carried in the Button's tooltip.

2. Library rail first step (library_rail.py): _benches_section/_benches_section_body now take is_first_run (benches, classic tasks, datasets, AND runs all empty). Only in that fully-empty condition does the Benches section swap "No benches yet." for a bold "Start here" callout (#evals-rail-first-run-hint, new .evals-rail-first-run-hint class in _evals.tcss); Datasets/Runs keep their own affordances unchanged. A user who already has a dataset or run is past "first open" and gets the plain wording.

3. Detail pane empty text (evals_screen.py): new _empty_detail_text() distinguishes "nothing selected but real rows exist" (unchanged generic copy) from "the whole library is empty" (new copy pointing at the sample bench), keeping the #evals-detail-empty id stable for existing tests.

4. Jargon (results_grid.py): added a tooltip to #evals-grid-meta defining raw/K/cells without growing the header. Declined touching the shared WorkbenchHeaderState subtitle -- DestinationHeader's subtitle Static is height:1 with no wrap/overflow handling, so adding text risks the same silent-truncation bug _lab.tcss already documents for rail labels; that's cross-cutting shared chrome (11 call sites) for a "matters less" item, not worth the blast radius.

5. Layout (_lab.tcss, lab_workbench.py): #lab-inspector was hard-fixed at 30 cells at every terminal width (measured 80/120/160/200 with a throwaway script before touching anything) -- the exact anti-pattern #lab-rail was already fixed for. Changed to width:2fr; min-width:30; max-width:50 (measured 30->50 scaling 80->200 cols); LAB_INSPECTOR_WIDTH constant split into LAB_INSPECTOR_MIN_WIDTH/MAX_WIDTH (was already unused/dead, so the rename is safe). Verified test_lab_workbench.py/test_lab_frame.py/test_llm_screen_lab_adoption.py/test_lab_server_status.py (Models/Speech-shared) still pass -- none pin an open #lab-inspector width.

Tests: 8 new tests across test_evals_screen.py, test_evals_empty_states.py, test_evals_results_grid.py. Every one was revert-checked (temporarily disabled the corresponding source change, reran, observed a real failure, restored) -- see PR/session notes for the exact error text. Full required suite: 110 passed.

CSS bundle rebuilt via build_css.py after every _evals.tcss/_lab.tcss edit; rebuilding again produces no further diff (idempotent).
<!-- SECTION:NOTES:END -->
