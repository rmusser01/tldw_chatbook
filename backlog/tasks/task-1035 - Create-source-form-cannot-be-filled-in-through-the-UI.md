---
id: TASK-1035
title: >-
  The create-source form cannot be filled in — no field accepts typed text
status: To Do
assignee: []
created_date: '2026-07-28 01:30'
labels:
  - watchlists
  - bug
  - ui
  - uat
priority: critical
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Opening the create-source form and typing puts text nowhere. Observed in the experienced-user UAT on `origin/dev` `13551ae8a`, clean profile, 235x52.

Steps, all of which failed to enter a single character into `Name` or `URL`:

1. Sources → `New Source` → type immediately. Nothing lands. **The form does not focus its first field**, which alone is a defect — the New-watchlist dialog *does* auto-focus, so the two creation flows behave differently.
2. Click directly on the `Name` input's own row (row 26, char column 40 — computed by character, not byte). Type. Nothing lands.
3. Press `Tab` five times, then type. Nothing lands.

**Control, run immediately afterwards in the same session:** clicking `Notifications` in the tab strip switched the section correctly. The app was alive and processing clicks throughout, so this is not a hung UI or a dead harness.

If this reproduces at the code level, **source creation through the form is unusable** and the only remaining path into the product is OPML import — which would mean the new-user path found in the previous UAT (task-995) is still blocked, just one step further along.

Recorded as observed-through-the-harness rather than proven in code. The first thing to establish is whether the form's `Input` widgets can take focus at all — check what `Screen.focused` is after the form mounts, and whether anything upstream is holding focus or the inputs are mounted outside the focus chain.

## Also seen on this form

`Create` and `Cancel` render stacked vertically with blank rows between them, unlike the New-watchlist dialog where they sit side by side. Cosmetic, but it makes the form look unfinished next to the dialog it is paired with.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Whether the form can be filled at all is established in code, and the answer recorded here
- [ ] #2 If broken: a source can be created end to end through the form, verified live from a clean profile
- [ ] #3 The form focuses its first field on open, matching the New-watchlist dialog
- [ ] #4 `Tab` moves through the form's fields in visual order
- [ ] #5 A test drives the form the way a user does — focus, type, submit — and fails against current code
- [ ] #6 `Create` and `Cancel` are laid out consistently with the New-watchlist dialog
<!-- AC:END -->
