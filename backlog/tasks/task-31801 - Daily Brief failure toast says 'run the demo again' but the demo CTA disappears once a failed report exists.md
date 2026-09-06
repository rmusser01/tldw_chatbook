---
id: TASK-31801
title: Daily Brief failure toast says 'run the demo again' but the demo CTA disappears once a failed report exists
status: To Do
assignee: []
created_date: '2026-09-05 19:15'
labels:
  - bug
  - ux
  - artifacts
  - watchlists
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the 2026-09-05 pre-release live UAT sweep (fresh scratch profile, dev tip 8e9d1128d4, real tmux-driven app). With no API key, Artifacts > 'Create Your First Daily Report' seeds the watchlist, fetches RSS, fails at the LLM stage, and the toast instructs fixing the key then 'run the demo again' - but once the failed report row exists, the CTA button is no longer rendered anywhere on the Artifacts screen, so the advertised retry path is gone (user must discover Watchlists to regenerate).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 After a failed demo brief, a retry affordance for the demo remains reachable from the Artifacts screen (or the toast copy points at a path that exists).
- [ ] #2 Test covering the failed-demo retry affordance.
<!-- AC:END -->
