---
id: TASK-40545
title: 'TTS _hand_off_retained_until entries are never popped on the runner-retention path or at shutdown'
status: To Do
assignee: []
created_date: '2026-11-12 09:00'
labels:
  - follow-up
  - tts
dependencies: []
references:
  - tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py
  - Tests/Event_Handlers/TTS_Events
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_hand_off_retained_until` (a `dict[Path, float]`) gets one entry per clip handed to the default app on Windows, and its entries are popped only in `_discard_tts_artifact`'s delete paths. On the runner-retention path (`_cleanup_audio_file` deleting after the retention delay) and at shutdown's sweep, the clip is deleted but its entry is left behind, as is the entry of a clip deleted by `observe_late_delete`. The cost is one stale Path to float pair per handed-off clip for the life of the process, so it is a small leak. It exists on dev today (`4b90eb860b`); no open PR fixes it. TASK-39040 introduced the dict (merged in #3426); the owner-visible effect is nil. Found in #3426's round-3 re-review, recorded in the burn-down ledger.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every path that deletes a handed-off clip also removes its `_hand_off_retained_until` entry
- [ ] #2 After a clip is cleaned up on the runner-retention path and after shutdown, the dict holds no entry for it, shown by a test that is red on dev
- [ ] #3 The retention bound for a clip still pending deletion is unchanged
<!-- AC:END -->
