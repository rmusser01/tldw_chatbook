---
id: TASK-35015
title: Every Settings form uses one on/off control and one row shape
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- ux
- design-system
- settings-ux-review-2026-10-14
dependencies:
- TASK-35014
- TASK-33007
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An on/off setting looks and behaves differently depending on the page:
- a checkbox labelled with the setting's name on Console Behavior;
- a checkbox with no label, whose mark clips to '▐X…', on Appearance's Animations and on Splash;
- Buttons reading Enabled/Disabled, Open/Collapsed or Automatic/Custom widths on Appearance;
- sliding Switches on Speech & TTS and Agents;
- a live On/Off word beside a name column in the Theme editor;
- 'Name: on' labels on Image Gen and Video Gen.

TASK-2827 had already settled on the checkbox, and later work drifted away from it.

Ordinary fields are inconsistent too. Goal runs puts its labels above full-width inputs three rows tall, and Speech & TTS's selects render three rows tall, so only about seven of its fields fit on a 44-row screen.

Nothing a user learns on one page carries over to the next, and the clipped mark looks broken. This task gives every Settings category form one on/off control and one label-left, one-row field shape, and writes both into DESIGN.md's dense-form convention. Every control keeps its id, saved key and save model.

Plan: `backlog/docs/plan-2026-10-14-settings-save-honesty.md`. Findings closed: D6, D-shell-design-M5, B10 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every on/off setting on a Settings category form is one checkbox whose label names the setting and whose mark is ✓ when on and blank when off. No category form uses a Switch, a Button whose label is the current state, a checkbox without a label, or a separate state word beside a checkbox. Exceptions: grids whose rows already name the item, such as Catalog refresh's per-provider grid and the Image/Video Gen backend rows, may keep an On/Off label. Action buttons named by a verb, and modal dialogs, are not on/off settings.
- [ ] #2 Every such checkbox paints its whole mark and label at 211x44 and 235x52, with no ellipsis in place of the mark.
- [ ] #3 Each converted control keeps its id, saved key, value and save model, and Space, Enter and a click each toggle it.
- [ ] #4 Goal runs' limits and every Speech & TTS select and input are one row tall with the label on the left, at 211x44 and 235x52, and Speech's stacked layout still works.
- [ ] #5 DESIGN.md's dense-form convention names the one on/off control and the one row shape, and Docs/User_Guide/settings.md describes the controls as they now render.
- [ ] #6 A test fails if a Settings module builds a Switch or a checkbox without a label. No new bare-type CSS subject or raw dimension literal is added.
- [ ] #7 settings_screen.py is no longer than it was before this change, and every new test runs and passes in the UI PR gate lane.
<!-- AC:END -->
