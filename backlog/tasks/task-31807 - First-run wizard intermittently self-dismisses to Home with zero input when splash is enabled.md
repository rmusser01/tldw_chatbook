---
id: TASK-31807
title: First-run wizard intermittently self-dismisses to Home with zero input when splash is enabled
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ui
  - wizard
  - flaky
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Observed twice on origin/dev during the 2026-09-05 release-gate work (by the TASK-31741 fix agent): with the splash screen enabled, the first-run wizard occasionally mounts and then self-dismisses to Home with no input, persisting setup_started. Intermittent; likely a race between splash teardown and the wizard's screen push. Needs a reproducer and fix; related surface: TASK-31226 (cancel routing) and TASK-31741 (exit-dialog settle guard).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Root cause identified with a deterministic reproducer or instrumented evidence.
- [ ] #2 Wizard never dismisses without user input.
<!-- AC:END -->
