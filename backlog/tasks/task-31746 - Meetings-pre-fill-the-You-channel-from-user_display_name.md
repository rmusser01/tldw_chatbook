---
id: TASK-31746
title: 'Meetings: pre-fill the You channel from user_display_name'
status: Done
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
- [x] #1 A configured user_display_name appears on the mic channel; an unset value falls back to You
<!-- AC:END -->

## Implementation Notes

One shared helper (`meeting_user_display_name`, reusing `config.get_chat_defaults_user_display_name`) renders the mic channel everywhere: the configured name when set and different from the factory default, else "You"; `MeetingMeta.user_display_name` is stamped at Start (and the live screen uses the stamped value) so live rows, the partial preview, the Markdown transcript and the Library render agree. Overlap segments now render "<name> + Others" everywhere. Landed in PR #2471 (commits 05f2d353d, 48e8541df, 7fd57fe5c, b7c8fc29b).
