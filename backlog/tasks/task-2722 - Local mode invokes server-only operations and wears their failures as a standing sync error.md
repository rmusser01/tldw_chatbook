---
id: TASK-2722
title: >-
  Local mode invokes server-only operations and wears their failures as a
  standing sync error
status: To Do
assignee: []
created_date: '2026-08-06 17:00'
labels:
  - schedules
  - home
  - bug
  - uat
  - local-server-split
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Full-app UAT on `origin/dev` `b0185749c`, local-only profile (no server configured, backend "Local"):

- The Schedules screen header wears a persistent **"1 sync error"** badge, and its sync strip shows `notifications.reminders.list.server requires server mode.` — i.e. the screen itself called a `*.server` operation while in local mode and then reports the predictable refusal as a sync error to the user.
- Home's active-work adapter does the same: `Home.active_work_adapter WARNING Failed to fetch server event feed for Home: notifications.feed.list.server requires server mode.` (twice per visit in the session log buffer).

A local-mode user who has never configured a server sees a standing error badge they cannot clear and did nothing to cause. Server-only feeds should be gated on the active runtime source rather than called-and-caught, and a "requires server mode" refusal in local mode should not be classified as a sync error.

Evidence: Schedules pane captures + Logs-screen warning entries, 2026-08-06 UAT session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [ ] With runtime source = local and no server configured, the Schedules screen shows no sync-error badge from server-only operations.
- [ ] Server-only feed/list calls are not issued while in local mode (or their local-mode refusal is classified as "not applicable", never as an error surfaced to the user).
- [ ] Home renders without logging server-feed failure warnings in local mode.
- [ ] Switching to server mode restores the current behavior (real failures still surface).
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->
