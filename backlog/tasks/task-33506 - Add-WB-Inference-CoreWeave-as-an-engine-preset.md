---
id: TASK-33506
title: Add W&B Inference (CoreWeave) as an engine preset
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:25'
updated_date: '2026-09-29 20:41'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CoreWeave's serverless inference is W&B Inference, an OpenAI Chat Completions API that oh-my-pi supports. Its optional `OpenAI-Project` header selects the team/project to bill, which no preset can send today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A user with a W&B API key can chat through the engine and discover models
- [x] #2 An optional team/project setting is sent as the documented OpenAI-Project header, and nothing is sent when it is unset
- [x] #3 The Settings guide documents setup
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Engine: record config_headers (header name -> api_settings key); resolved value validated (non-empty, no CR/LF) and sent only when set.
2. WANDB record: api.inference.wandb.ai/v1, Bearer WANDB_API_KEY, discovery, reasoning field, OpenAI-Project <- project.
3. Hand lists, config table, tests, Settings guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
WANDB record (key wandb, config WandB): api.inference.wandb.ai/v1, Bearer WANDB_API_KEY (the W&B API key the service documents), discovery on (/v1/models), reasoning field allowlisted, streamed usage requested and optional. New engine capability config_headers (record: header -> api_settings key): resolve_hosted_request reads the setting, sends nothing when unset/blank, fails closed on a non-string, a CR/LF/NUL or >256 chars, and passes the headers to the transport (which already rejects reserved names). W&B maps OpenAI-Project <- [api_settings.wandb] project. Tests pin resolution, transport delivery and that only wandb/cloudflare declare headers. Discovery sends no project header (optional; listings use the default entity).

Qodo round: config-header values are validated by a Pydantic TypeAdapter (strict str, stripped, <=256 chars, no CR/LF/NUL) per the repo's Pydantic-at-boundaries rule; same behavior, pinned by the existing header tests.
<!-- SECTION:NOTES:END -->
