---
id: TASK-576
title: 'Console /rewind: discoverability beyond the bare slash command'
status: To Do
assignee: []
created_date: '2026-07-25'
labels:
  - console
  - chat
  - rewind
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Documented v1 limitation of the `/rewind` menu (SP2, PR #844): the menu opens only by typing `/rewind` in the composer. Nothing in the UI advertises it — no command-palette entry, no transcript-selection action, no help/guide mention — so the feature is invisible to anyone who has not read the docs. Add at least one discoverable surface: a command-palette entry, and/or a message-action affordance on a selected transcript row (which would also pre-seed the restore target), and/or a line in the transcript action guide. Keep the composer command as the primary path; `/rewind` parses BEFORE readiness gating and any new surface must preserve that (the menu opens while sends are blocked).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 /rewind is reachable from at least one discoverable UI surface in addition to typing the command
- [ ] #2 Any new surface opens the same menu with identical behavior, including while sends are blocked (readiness-gated states)
- [ ] #3 Discoverability copy (guide/help/palette label) is present and accurate
<!-- AC:END -->
