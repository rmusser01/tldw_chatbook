---
id: TASK-31813
title: First-run wizard cancel lands on the Console
status: Done
assignee: []
created_date: '2026-09-04 01:00'
updated_date: '2026-09-06 00:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Esc-exiting the first-run setup wizard leaves the user on Home (the screen it was pushed over). Cancel should drop the user onto the Console workbench, while cancelling a Settings/palette re-run must stay put. Also pins that the workspace tree keeps empty registered workspaces (UAT artifact triage).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Cancelling the boot-offered first-run wizard (Esc exit / finish-later) switches to the Console tab
- [x] Cancelling a Settings/command-palette wizard re-run leaves the current screen unchanged
- [x] Completed wizard results keep routing per their exit_route (regression pin)
- [x] A registered workspace with zero conversations keeps its tree node (mounted pin)
<!-- AC:END -->

## Implementation Plan

1. Tests first: cancel routes to TAB_CHAT; rerun cancel stays; completed-result routing regression; empty-workspace tree node pin.
2. app.py: cancel branch of the wizard result handler posts NavigateToScreen(TAB_CHAT) and consumes a deferred focus request; Settings/palette re-run sites pass cancel_to_console=False through the interview wrapper.
3. Suites, lint, live UAT (fresh profile, Esc-exit lands on Console), PR.

## Implementation Notes

Renumbering provenance: Canvas claimed TASK-31226 in811fe0188 on2026-09-03
at17:26:11-07:00; this task was added later in df645d94c on2026-09-05
at17:17:29-07:00. The all-remote/worktree sweep found maximum31811, with31812
subsequently assigned to the other collision in this integration. This task and
all shipped references move to31813; implementation and acceptance are unchanged.

UAT triage (2026-08-31) of three findings; one product fix, two pins.

- Wizard cancel routes to Console: the boot flow parks first-run launches on
  Home and pushes the wizard over it; a cancelled result (no dict) made the
  handler return early, stranding the user on Home. The cancel branch now
  posts NavigateToScreen(TAB_CHAT) and consumes a deferred focus request
  under the same Chat-route rule the completed paths use. The Settings/
  command-palette re-run sites opt out via cancel_to_console=False (cancel
  returns to Settings); the interview wrapper forwards the flag.
- Tree drop = UAT artifact, pinned not fixed: tree nodes derive from the
  registry's list_workspaces, independent of conversation rows; a new
  mounted pin holds that an empty registered workspace keeps its node.
- Home cards = synthetic-input artifact: mounted tests click the same cards
  successfully; no product change.
- Verification: 6-test cancel-routing suite green; 424 across the wizard/
  focus/interview suites after updating the palette-reentry pin to the new
  callback contract; lint debt unchanged from baseline. Live UAT on a fresh
  scratch profile: Esc-exit lands on the Console workbench, clean exit.
