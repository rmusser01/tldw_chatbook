---
id: TASK-32155
title: Qualify real ElevenLabs cloud TTS delivery
status: To Do
assignee: []
created_date: '2026-09-09 05:51'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No usable ElevenLabs TTS credential was present in the scoped environment or application config audit. Real synthesis still requires an account and quota.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Usable provider credentials and quota are supplied through normal secret configuration without secret values in logs or artifacts.
- [ ] #2 An exact supported model, voice and output format produce complete Lab and repeated Console speech through ElevenLabs, with playback and full-content evidence.
- [ ] #3 Real cancellation, provider failure or invalid request, and a successful successor or retry settle ownership and expose actionable redacted errors.
<!-- AC:END -->
