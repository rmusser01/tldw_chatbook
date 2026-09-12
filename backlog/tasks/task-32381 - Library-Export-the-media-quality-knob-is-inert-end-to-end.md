---
id: TASK-32381
title: 'Library Export: the media-quality knob is inert end to end'
status: To Do
assignee: []
created_date: '2026-09-11 10:30'
labels:
  - library
  - export
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Export canvas offers a `quality:` chooser (original / thumbnail) with helper copy and a bundle phrase describing what the choice does, but `_collect_media` never reads the selected value -- every bundle ships the same bytes whichever option is active. Raised during the task-32353 bot round and rated High there. A control that visibly settles on a choice and changes nothing is worse than no control: it makes the user believe a smaller bundle was written. Either the knob drives the collection or it is removed together with its helper copy and bundle phrase, in one change so the surface never half-claims the feature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting a non-default export quality changes what the written bundle contains, or the control and its copy are gone
- [ ] #2 The bundle phrase and the quality helper copy agree with what the export actually wrote
- [ ] #3 A test exercises the chosen behaviour end to end rather than asserting only the form state
<!-- AC:END -->
