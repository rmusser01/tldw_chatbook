---
id: TASK-2233
title: 'Roleplay: hide the voice-profile checkbox when no profile is assigned (R2)'
status: To Do
assignee: []
created_date: '2026-08-04 16:18'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The disabled 'Include assigned voice profile' checkbox renders as an unreadable dark smear exactly where the eye lands after the primary CTA. Disabled-with-reason is right for applicable actions; this one is not applicable when nothing is assigned. Post-fix re-review P2. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Checkbox is hidden (display False) when no voice profile is assigned,It reappears (enabled or disabled-with-reason) when a profile is assigned,Tests updated
<!-- AC:END -->
