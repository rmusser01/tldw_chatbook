---
id: TASK-35023
title: Speech & TTS selects render one row tall like other Settings selects
status: To Do
created_date: 2026-10-14 18:30
labels:
- settings
- ux
- design-system
- ux-review-2026-10-14
dependencies:
- TASK-33007
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On Speech & TTS every select and input is drawn three rows tall, although each one asks for the compact one-row shape that other Settings pages show. Only about seven of its fields fit on a 44-row screen, so a keyboard or low-vision user scrolls far more on this page than on any other, and the page looks unlike the rest of Settings. The cause is not known yet. Changing styles without finding it could break Speech's stacked layout at narrower widths, so the cause has to be found before the fix. This was split out of TASK-35015, which converts on/off controls and does not need this investigation.

Plan: `backlog/docs/plan-2026-10-14-settings-save-honesty.md`. Findings closed: D6, Speech & TTS select rows (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 211x44 and 235x52, every Speech & TTS select and input is one row tall with its label on the left, like the selects on other Settings pages, and more Speech fields fit on one screen than the seven the review measured
- [ ] #2 Speech & TTS's stacked layout still shows every field with its label at the widths where it stacks
- [ ] #3 The cause of the three-row height is identified and fixed where it arises; every Speech field keeps its id, saved key, value and save model, and no bare-type CSS subject or raw dimension literal is added
- [ ] #4 A test that reads the painted rows with the production stylesheet fails if a Speech & TTS select or input is taller than one row at 211x44 or 235x52, and it runs on every pull request
<!-- AC:END -->
