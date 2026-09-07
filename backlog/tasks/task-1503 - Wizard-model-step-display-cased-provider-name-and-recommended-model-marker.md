---
id: TASK-1503
title: 'Wizard model step: display-cased provider name and recommended-model marker'
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:29'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: subtitle reads 'Models for anthropic.' (raw key); no model is marked recommended; no skip reassurance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Subtitle uses the provider display name
- [ ] #2 First/curated-default model is marked as recommended
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Subtitle uses provider_display_name(); first model row labeled '(recommended)' with the clean id stored as button._model_id — selection, pressed-radio fallback, and custom-input-clear fallback all read _model_id so decoration never reaches config. Four pre-existing label-list tests updated to the clean-id contract.
<!-- SECTION:NOTES:END -->
