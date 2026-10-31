---
id: TASK-38420
title: >-
  Console: a tool name ending in _image makes a tool-schema 400 read as an image
  refusal
status: To Do
assignee: []
created_date: '2026-10-31 03:29'
labels:
  - console
  - errors-recovery
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The image hint on a failed reply follows a typed verdict: a 400, 413, 415 or 422 whose provider body names image or vision input. The body check matches 'image' after an underscore in any of the error's message, type, code or param fields, so it also matches a tool name like generate_image. A 400 that rejects such a tool's schema, in a conversation that holds an image, then shows the image hint beside the tool-definition copy, advising a vision model or /rewind for a failure that has nothing to do with the image. Found in review of TASK-33621.11.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A 400 that rejects a tool definition whose name contains 'image' does not add the image hint, even when the conversation holds an image
- [ ] #2 A real image refusal (for example OpenAI's 'invalid_image_format' or Anthropic's content image error) still adds the hint
- [ ] #3 Tests drive both cases through the real gateway and failure-copy code
<!-- AC:END -->
