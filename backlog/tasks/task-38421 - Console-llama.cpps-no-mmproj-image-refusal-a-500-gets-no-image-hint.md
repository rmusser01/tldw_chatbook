---
id: TASK-38421
title: 'Console: llama.cpp''s no-mmproj image refusal (a 500) gets no image hint'
status: To Do
assignee: []
created_date: '2026-10-31 03:30'
labels:
  - console
  - errors-recovery
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33621.11 limits the failure row's image hint to a 400, 413, 415 or 422 whose provider body is about image or vision input, as its AC#3 states. llama.cpp started without a multimodal projector answers an image with a 500 whose body says image input is not supported (mmproj). That is the most common real image refusal for local users, and it now gets the provider's body but no hint pointing at a vision-capable model or /rewind. Decide whether that body should be classified as an image refusal (for example a typed exception raised by the llama.cpp path) and, if so, restore the hint for it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner decides whether llama.cpp's no-mmproj 500 counts as an image refusal
- [ ] #2 If it does, that failure in a conversation holding an image shows the image hint, and other 500s still do not
- [ ] #3 A test drives the real llama.cpp error body through the gateway and failure-copy code
<!-- AC:END -->
