---
id: TASK-32154
title: Qualify real OpenAI cloud TTS delivery
status: To Do
assignee: []
created_date: '2026-09-09 05:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No usable OpenAI TTS credential was present in the scoped environment or application config audit. Local compatible transport evidence cannot qualify the cloud provider.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Usable provider credentials and quota are supplied through normal secret configuration without secret values in logs or artifacts.
- [ ] #2 An exact supported model and voice produce complete Lab and repeated Console speech through the real OpenAI service, with real playback and full-content evidence.
- [ ] #3 Real cancellation, provider failure or invalid request, and a successful successor or retry settle ownership and expose actionable redacted errors.
<!-- AC:END -->
