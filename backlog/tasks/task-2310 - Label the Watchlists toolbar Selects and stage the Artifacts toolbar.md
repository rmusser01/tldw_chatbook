---
id: TASK-2310
title: Label the Watchlists toolbar Selects and stage the Artifacts toolbar
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: filter Selects across the screen display bare values with no hint of
what they filter — Sources shows "All ▼ / All statuses ▼ / All ▼" (two
unlabeled), Artifacts shows "Auto + featured ▼ / App default ▼ / Off ▼"
(off for WHAT?), and the New Rule form's severity Select and Threshold field
are unlabeled/unexplained. The Artifacts toolbar also shows 12 controls
including "Stop Serving" before any briefing exists.

UAT findings F9, F37, F38.

## Acceptance Criteria (the what)

- [ ] Every Select on the Watchlists screen carries a visible label naming
      what it controls (border-title style is fine).
- [ ] The Rule form explains Threshold (unit/meaning) and labels severity.
- [ ] The Artifacts empty state foregrounds Generate; serve/export/keep
      controls appear only once they can act on something (or are visibly
      disabled with a reason).
