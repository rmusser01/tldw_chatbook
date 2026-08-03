---
id: TASK-1336
title: Add QwenCloud dual API provider support
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-03 03:24'
updated_date: '2026-08-03 03:24'
labels:
  - provider
  - qwencloud
  - settings
  - tools
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md
  - backlog/decisions/006-provider-aware-generation-settings.md
  - backlog/decisions/020-automatic-model-catalog-refresh.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add QwenCloud as a first-class API provider so users can select Responses or Chat Completions mode while retaining Chatbook function-tool execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 QwenCloud is selectable and passes provider readiness with supported endpoint and credential configuration.
- [ ] #2 api_mode persists under api_settings.qwencloud, defaults to responses, and accepts only responses or chat_completions.
- [ ] #3 Text and existing Chatbook function tools work in streaming and non-streaming requests for both API modes.
- [ ] #4 QwenCloud request translation is mode-aware and unsupported parameters or tool shapes fail clearly rather than being silently forwarded.
- [ ] #5 QwenCloud model discovery participates in the existing cached model catalog pipeline.
- [ ] #6 Settings exposes QwenCloud API mode without affecting other providers.
- [ ] #7 Automated tests cover registration, configuration, payload translation, tool-call normalization, streaming, discovery, retries, and errors without paid API calls.
- [ ] #8 Provider documentation records setup, API-mode behavior, limitations, and optional live-test instructions.
<!-- AC:END -->
