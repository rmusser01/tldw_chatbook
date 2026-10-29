---
id: TASK-37031
title: >-
  Console: images are blocked for the default Anthropic model because
  claude-haiku-4-5 is not marked vision-capable
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-27 16:12'
updated_date: '2026-10-28 02:14'
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
- [x] #1 With the shipped defaults and no capability override, a staged image with claude-haiku-4-5 selected leaves Send ready and the image reaches the model
- [x] #2 Every Anthropic model in the shipped provider list that accepts image input is vision-capable by default, pinned by a test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Fixed with TASK-33620.2 (same root cause): see that task's plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Closed by TASK-33620.2 (branch fix/task-33620.2-image-attach-haiku, 2026-10-27). Root cause: no vision row in DEFAULT_MODEL_PATTERNS or the shipped [model_capabilities] template matched claude-haiku-4-5 (nor claude-opus-5 / claude-fable-5), and an older config.toml's Anthropic pattern list replaces the shipped one wholesale. Fix: an Anthropic family rule in ModelCapabilities.get_model_capabilities (claude-<tier>-<major>, major >= 4, provider anthropic) beneath the user's own rows. AC#2 is pinned by Tests/Chat/test_anthropic_vision_capability.py, which iterates the shipped Anthropic list under a fresh and an old config (red on dev f9baac8706 for haiku-4-5, opus-5, fable-5). AC#1 verified live: shipped defaults + no override, a 64x64 PNG staged on claude-haiku-4-5 left Send ready and the real model answered 'Green, purple.' (evidence tools/evidence/bd5/U14/20-21).
<!-- SECTION:NOTES:END -->
