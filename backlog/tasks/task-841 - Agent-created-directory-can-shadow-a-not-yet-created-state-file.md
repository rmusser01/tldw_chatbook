---
id: TASK-841
title: Agent-created directory can shadow a not-yet-created state file
status: To Do
assignee: []
created_date: '2026-07-27 02:36'
labels:
  - tools
  - security
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The denylist's direct-child rule is gated on whether a path is an existing directory, so an agent can create a directory named after a state file the app has not created yet (verified: search_history.db/ is permitted). The app's later attempt to open that path as a database would fail. Denial of service only -- no disclosure and no gate bypass. Filed from the PR #953 review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An agent cannot create a directory whose name collides with a known app state file,Existing container subdirectories under the user data dir stay reachable,A regression test covers the collision
<!-- AC:END -->
