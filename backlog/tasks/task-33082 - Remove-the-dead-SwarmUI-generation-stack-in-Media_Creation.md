---
id: TASK-33082
title: Remove the dead SwarmUI generation stack in Media_Creation
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [cleanup, media]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Media_Creation still carries a pre-adapter-architecture SwarmUI pipeline: swarmui_client.py, the async generation paths of image_generation_service.py, and the Event_Handlers/Media_Creation_Events handler package all have zero external callers on current dev. Only generation_templates.py and one static helper (extract_context_from_messages) are live. This is a third parallel image-generation stack that predates the Image_Generation adapter architecture and was never retired, so every reader must triage which of the three stacks is real.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 swarmui_client.py and the Media_Creation_Events handler package are deleted with no external import breakage.
- [ ] #2 extract_context_from_messages is relocated and its caller in Chat/console_generate_image.py passes targeted tests.
- [ ] #3 generation_templates.py remains importable and functional for all existing consumers.
- [ ] #4 No packaging manifest or user-facing doc still advertises the SwarmUI generation stack.
<!-- AC:END -->
