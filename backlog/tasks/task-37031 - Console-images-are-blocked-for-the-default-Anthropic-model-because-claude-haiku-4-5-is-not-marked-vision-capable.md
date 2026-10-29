---
id: TASK-37031
title: >-
  Console: images are blocked for the default Anthropic model because
  claude-haiku-4-5 is not marked vision-capable
status: To Do
assignee: []
created_date: '2026-10-27 16:12'
labels:
  - console
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live while verifying TASK-33620.15.22 (2026-10-27, branch fix/task-33620.15.22-send-ready-attachment). With the shipped default model (Anthropic claude-haiku-4-5) and no [model_capabilities] override, staging an image disables Send with 'Send blocked — choose a model to continue'. Neither the code defaults in model_capabilities.py (Anthropic patterns cover claude-3, opus-4 and sonnet-4; direct rows only claude-3.x) nor the shipped config template (which adds claude-sonnet-5) lists haiku-4 as vision-capable, so the default model refuses every image although the provider accepts images for it. Adding [model_capabilities.models."claude-haiku-4-5"] vision = true to the profile made the same image send and get a description from the model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With the shipped defaults and no capability override, a staged image with claude-haiku-4-5 selected leaves Send ready and the image reaches the model
- [ ] #2 Every Anthropic model in the shipped provider list that accepts image input is vision-capable by default, pinned by a test
<!-- AC:END -->
