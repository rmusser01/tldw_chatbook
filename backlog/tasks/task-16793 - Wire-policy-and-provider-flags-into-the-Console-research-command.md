---
id: TASK-16793
title: Wire policy and provider flags into the Console research command
status: Done
assignee:
  - '@robert'
created_date: '2026-08-16 14:31'
updated_date: '2026-08-16 14:37'
labels:
  - research
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console /research command always launches balanced-policy runs with the default provider set; the engine now honors source_policy and provider_overrides but the command cannot express them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 /research accepts a policy token (web_only, academic_only, web_first, academic_first, balanced) and a providers or category list,Both ride the launched run as source_policy and provider_overrides,Usage errors surface as native console messages,Tests cover flag parsing and the launch payload
<!-- AC:END -->
