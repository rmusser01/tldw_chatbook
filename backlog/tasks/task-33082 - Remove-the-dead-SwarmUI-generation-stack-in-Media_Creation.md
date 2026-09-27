---
id: TASK-33082
title: Remove the dead SwarmUI generation stack in Media_Creation
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [cleanup, media]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Superseded in part by upstream cleanup: dev already deleted Event_Handlers/Media_Creation_Events and landed the Media_Generation shared core (ADR-176). What remains in Media_Creation is generation_templates.py (live, many importers), swarmui_client.py (426 LOC, no external importers beyond its own package), and image_generation_service.py (442 LOC, whose only production use is the static extract_context_from_messages helper for Chat/console_generate_image.py). The stack is still a third parallel image-generation surface, but it is now woven into fresh TASK-32628-family backup/recovery tests — three test files construct ImageGenerationService and GenerationResult as generated-media capture subjects — and the aiofiles optional-import deferral test imports SwarmUIClient directly. Removal therefore requires reworking that fresh test coverage onto self-contained subjects first; it is not a pure deletion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 swarmui_client.py is deleted after its import chain (Media_Creation/__init__.py, image_generation_service.py) and the aiofiles deferral test in Tests/Utils/test_optional_import_deferral.py are reworked, with the deferral coverage either retargeted to another aiofiles consumer or dropped with recorded justification.
- [ ] #2 The async generation machinery of image_generation_service.py is removed and extract_context_from_messages is relocated to its sole caller or a small shared module, keeping Chat/console_generate_image.py behavior identical.
- [ ] #3 The three Backup_Recovery subjects (test_generated_media_capture.py, test_temporary_media_capture.py, test_saved_generation_roundtrip.py) are reworked onto self-contained fakes with identical coverage, or that rework is filed as a blocking follow-up task and referenced here.
- [ ] #4 generation_templates.py remains importable and functional for all existing consumers.
- [ ] #5 Targeted tests pass: optional-import deferral, the Backup_Recovery media-capture trio, and console image generation.
<!-- AC:END -->

## Revision provenance

Filed 2026-09-27 against a stale branch audit that called this a zero-risk ~1k LOC deletion (including the events package). Re-verified against origin/dev the same day: the events package is already gone upstream, and the remaining modules are entangled with TASK-32628 backup/recovery test coverage. Description and ACs rewritten to the dev-verified scope; priority lowered from high to medium accordingly.
