---
id: TASK-1051
title: TaskResumeState.from_dict drops pending_skill_script
status: To Do
assignee: []
created_date: '2026-07-27 14:31'
labels:
  - console
  - skills
  - state-restore
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`TaskResumeState.from_dict` (`tldw_chatbook/UI/Screens/chat_screen_state.py:86`) restores `pending_skill_install` from a persisted Console snapshot but hardcodes `pending_skill_script=None`, discarding whatever was saved for that field. This is a pre-existing asymmetry between the two skill-confirm payload types the dataclass otherwise treats identically (`has_pending_skill_install`/`has_pending_skill_script`, `to_dict` serializes both fields the same way).

The failure mode is silent and fails closed: a session that was snapshotted mid skill-script-confirm restores with no visible pending confirm at all rather than an error, so a user resuming that session simply never sees the card they were about to decide on. Whether this was ever a deliberate decision (e.g. a script-confirm round cannot legitimately survive a restart/restore for some reason) is not documented anywhere near the field.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `pending_skill_install` and `pending_skill_script` are restored symmetrically by `from_dict`, OR the asymmetry is documented as deliberate at the field with the dead `pending_skill_script` field removed from the dataclass/serialization entirely.
- [ ] #2 A regression test covers `from_dict` round-tripping a snapshot that carries a `pending_skill_script` payload.
<!-- AC:END -->
