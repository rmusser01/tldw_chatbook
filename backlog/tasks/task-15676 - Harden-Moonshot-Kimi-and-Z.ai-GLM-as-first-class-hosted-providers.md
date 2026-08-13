---
id: TASK-15676
title: Harden Moonshot Kimi and Z.ai GLM as first-class hosted providers
status: To Do
assignee: []
created_date: '2026-08-12 20:45'
labels: []
dependencies:
  - TASK-15675
references:
  - Docs/superpowers/specs/2026-08-12-kimi-zai-hosted-chat-completions-design.md
  - >-
    Docs/superpowers/plans/2026-08-12-kimi-zai-hosted-chat-completions-implementation.md
  - backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bring the existing Moonshot AI and Z.ai integrations up to the same first-class reliability, security, streaming, tooling, Settings, discovery, and resumability standard as the newest hosted providers, while establishing a reusable hosted Chat-Completions wire boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Moonshot and Z.ai preserve their stable provider identities and public handler compatibility while using the neutral hosted Chat-Completions boundary; unrelated providers do not change behavior.
- [ ] #2 Fresh configuration defaults use `kimi-k3` for Moonshot and `glm-5.2` for Z.ai while explicit historical selections remain usable.
- [ ] #3 Explicit arguments, canonical configuration, environment credentials, defaults, and structural endpoint validation follow one documented fail-closed contract without mutating source configuration or disclosing secrets.
- [ ] #4 Streaming and non-streaming calls strictly validate payloads, errors, finish states, usage, retries, cancellation, ownership, and size/depth bounds.
- [ ] #5 Existing Chatbook function tools complete joined Console continuation for both providers; private reasoning uses TASK-15675 checkpoints and vendor built-in tools remain excluded.
- [ ] #6 Kimi K3 preserves and budgets every retained assistant reasoning turn required by its always-on Preserved Thinking contract; other curated Kimi/GLM models follow their exact policies, with GLM using `clear_thinking=false` only for active/restored tool runs.
- [ ] #7 Moonshot and best-effort Z.ai model discovery use the same normalized endpoint and credential resolution as chat, preserve prior cache on failure, and never log sensitive payloads.
- [ ] #8 Canonical Settings exposes actionable readiness, save, search/focus, endpoint, credential, model, and reasoning guidance without an API-mode selector.
- [ ] #9 QwenCloud Chat behavior remains unchanged after any shared parser extraction, proven by its complete contract suite and mutation checks.
- [ ] #10 Documentation and optional doubly-gated isolated live tests cover the current defaults, endpoints, controls, tools, recovery, and no-paid-default contract.
<!-- AC:END -->
