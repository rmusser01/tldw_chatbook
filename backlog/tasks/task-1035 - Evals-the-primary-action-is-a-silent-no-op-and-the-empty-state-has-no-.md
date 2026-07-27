---
id: TASK-1035
title: >-
  Evals: the primary action is a silent no-op and the empty state has no ordering
status: To Do
assignee: []
created_date: '2026-07-27 16:00'
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
- [ ] `Run Bench` is disabled with a stated reason, or explains what is missing when pressed
- [ ] The empty state establishes one obvious first step
- [ ] The Detail pane's zero-data text is actionable
- [ ] First-contact jargon is defined somewhere reachable from the screen
- [ ] The Inspector has enough width for its content at common terminal sizes
<!-- AC:END -->
