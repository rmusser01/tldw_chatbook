---
id: TASK-31746
title: 'Meetings: pre-fill the You channel from user_display_name'
status: To Do
assignee: []
created_date: '2026-09-06 07:40'
labels:
  - audio
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The mic channel label uses a Meetings-local 'You' default instead of chat_defaults.user_display_name (spec 2026-09-05 §5.1 wanted the user's name). It was kept as 'You' because that config key's factory default is literally 'User', so pre-filling would render 'User:' on every install and it broke two tests. Add a config unset-sentinel (or equivalent) so a user who sets a real display name sees it on their mic channel, while an unset value falls back to 'You'. Deferred from the phase-2 diarization SDD run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A configured user_display_name appears on the mic channel; an unset value falls back to You
<!-- AC:END -->
