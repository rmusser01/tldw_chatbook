---
id: TASK-32959
title: 'OmniVoice wizard follow-ups: configured-root install loop and Summary Voice line'
status: To Do
assignee: []
created_date: '2026-09-26 01:30'
labels:
  - wizard
  - tts
  - omnivoice
dependencies:
  - TASK-32958
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the final review and live UAT of TASK-32958 and left out of that PR. A user with an explicit but broken OmniVoice model path (the `[OmniVoiceSettings] model_root` setting or the `OMNIVOICE_MODEL_ROOT` env var) is shown "model missing" with an Install button. The managed install then succeeds, but the explicit path still wins, so the state stays "model missing" and the user can download 1.1 GB repeatedly. Separately, the setup Summary lists no Voice line for any service, so the user gets no read-back of what the Voice step saved (pre-existing on dev).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With an explicit OmniVoice model path that fails its layout check, the Voice step says the configured path is invalid and does not offer a download
- [ ] #2 The setup Summary shows a Voice line (service, and whether it was saved as the default) read back from saved config
- [ ] #3 After a sample finishes playing, a stale Test-and-Hear result cannot overwrite the status of a newer test
<!-- AC:END -->
