---
id: TASK-496
title: >-
  Rename residual 'Personas'-as-destination copy in Settings domain-ownership
  audit
status: To Do
assignee: []
created_date: '2026-07-23 17:21'
labels:
  - ux
  - roleplay
  - copy
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-435 RP&CD rename retired 'Personas' as the destination's public name (it now means only the in-screen user-identity mode). Two user-visible strings in the Settings domain-ownership audit still name the destination 'Personas': the SettingsDomainCategoryContract for the PERSONAS category (title, owner_destination, and its prose source-of-truth/rows/follow-up text) and the SettingsCategorySummary entry. These need renaming to the destination's new public name, but the wording is ambiguous with the Settings-screen category taxonomy, so it was deferred from TASK-435 for a deliberate copy pass.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings domain-ownership audit no longer names the destination 'Personas' (uses RP&CD / Roleplay & Chat Dictionaries consistently)
- [ ] #2 Settings category summary for the persona/character domain reflects the destination's new public name
- [ ] #3 'Personas' remains only where it means the in-screen user-identity mode; no test regressions in test_settings_configuration_hub
<!-- AC:END -->
