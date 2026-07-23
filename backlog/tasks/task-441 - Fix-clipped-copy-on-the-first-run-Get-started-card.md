---
id: TASK-441
title: Fix clipped copy on the first-run Get started card
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - home
  - ux
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live on a fresh profile: step 3 of the Get started card renders "Send your first message  Composer unlocks after" - the sentence is cut off mid-thought at the card width.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The step-3 explainer renders as a complete sentence at default terminal sizes
- [ ] #2 Card copy wraps instead of truncating when space is short
<!-- AC:END -->
