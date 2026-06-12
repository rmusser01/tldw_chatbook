---
id: TASK-98
title: Extract PersonasPreviewController from personas_screen
status: To Do
assignee: []
created_date: '2026-06-11 13:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The preview block (~180-210 lines: seeding rule, system-prompt builder, worker, gateway lifecycle) is self-contained; extract to UI/Persona_Modules mirroring personas_conversations_controller before the screen grows past legacy size.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Screen delegates preview logic to a controller,Tests repointed,Behavior preserved
<!-- AC:END -->
