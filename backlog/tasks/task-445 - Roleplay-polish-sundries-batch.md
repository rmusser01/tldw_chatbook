---
id: TASK-445
title: Roleplay polish sundries batch
status: To Do
assignee: []
created_date: '2026-07-21 09:38'
labels:
  - roleplay
  - ux
  - polish
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Small items from the review, batched: (1) footer hint noise - persistent "ctrl+s save unavailable | esc back unavailable" text; (2) "1 characters" count grammar; (3) model discovery exists in Settings but its results do not offer themselves into the Model field (users hand-type 50-char gguf names); (4) transient rendering artifact - a tall empty selection frame appeared under the selected library row after first selection; (5) import success toast is easy to miss - consider inline confirmation near the list.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Footer shows only currently available actions (or renders unavailable ones dimmed without the word 'unavailable')
- [ ] #2 Count line uses correct singular/plural
- [ ] #3 Model discovery results can be applied to the Model field with one action
- [ ] #4 Library rail no longer renders the empty selection-frame artifact
- [ ] #5 Import success is confirmed visibly at normal reading pace
<!-- AC:END -->
