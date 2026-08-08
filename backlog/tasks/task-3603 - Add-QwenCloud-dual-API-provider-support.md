---
id: TASK-3603
title: Add QwenCloud dual API provider support
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-03 03:24'
updated_date: '2026-08-07'
labels:
  - provider
  - qwencloud
  - settings
  - tools
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md
  - backlog/decisions/045-qwencloud-dual-api-provider-boundary.md
  - backlog/decisions/006-provider-aware-generation-settings.md
  - backlog/decisions/012-provider-credential-settings-boundary.md
  - backlog/decisions/020-automatic-model-catalog-refresh.md
  - backlog/decisions/026-retire-chat-tab-conversation-entry-chain.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add QwenCloud as a normal first-class API provider, equivalent to existing
providers such as OpenAI and DeepSeek, while allowing users to select Responses
or Chat Completions as its external API mode.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 QwenCloud is selectable anywhere an ordinary hosted Console provider is selectable and passes shared provider readiness with its own endpoint and credential configuration.
- [ ] #2 `api_mode` persists under `api_settings.qwencloud`, defaults to `responses`, and accepts only `responses` or `chat_completions`.
- [ ] #3 Streaming and non-streaming text work through the standard provider dispatcher and Console gateway in both API modes.
- [ ] #4 Existing Chatbook function tools work in both modes through the existing native Console agent runtime, including multiple calls, paired tool results, errors, cancellation, and structured continuation without synthetic user messages.
- [ ] #5 QwenCloud model discovery participates in the existing cached model-catalog pipeline and does not create a parallel model source.
- [ ] #6 Canonical F9 Settings exposes QwenCloud API mode without changing another provider's configuration or behavior.
- [ ] #7 Request translation uses mode-specific parameter allowlists; invalid tools, messages, endpoints, modes, and unsupported content fail before network I/O.
- [ ] #8 Automated tests cover provider parity, registration, configuration, translation, streaming, native-tool continuation, discovery, retries, cancellation, and safe errors without paid API calls.
- [ ] #9 Provider documentation records setup, API-mode behavior, parameter limitations, existing function-tool support, built-in-tool exclusion, and optional isolated live verification.
<!-- AC:END -->

## Task Identity Note

This task was renumbered from `TASK-1336` after a full remote-ref and worktree
sweep found an older SoundDevice/VAD task using the same identifier. `TASK-3603`
was unclaimed in fetched refs, checked worktrees, and QwenCloud pull-request
searches on 2026-08-07.
