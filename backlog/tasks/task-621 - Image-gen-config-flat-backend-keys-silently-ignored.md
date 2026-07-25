---
id: TASK-621
title: >-
  Image-gen config silently ignores flat backend keys in [image_generation]
status: To Do
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 10:15'
labels:
  - image-generation
  - config
  - uat
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT 2026-07-25: writing `openrouter_image_default_model = "..."` directly under `[image_generation]` — the exact FLAT field name the config dataclass uses — is silently ignored; only the nested `[image_generation.openrouter] default_model = "..."` shape parses. No warning is logged, so a user who guesses the flat spelling (which matches the dataclass and reads naturally) gets the shipped default with zero feedback. Cost during UAT: a full restart-and-retest cycle to discover the override never applied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Either the flat spellings are accepted as aliases of the nested keys, or an unknown/unmapped key found directly under `[image_generation]` logs a clear warning naming the key and the expected nested section (choose one; document the choice in the shipped config example).
- [ ] The nested shape keeps working unchanged; secrets/env precedence unaffected.
- [ ] Tests pin the chosen behavior for at least one backend field (accepted-alias or warn-on-unknown).
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
