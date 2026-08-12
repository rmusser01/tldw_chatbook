---
id: TASK-15675
title: >-
  Harden Moonshot Kimi and Z.ai GLM as first-class hosted Chat-Completions
  providers
status: In Progress
assignee: []
created_date: '2026-08-12 18:06'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bring the existing Moonshot AI and Z.ai integrations up to the same first-class reliability, security, streaming, tooling, Settings, and discovery standard as the newest provider integrations, while establishing a reusable hosted Chat-Completions boundary for compatible providers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Moonshot and Z.ai preserve their stable provider identities and public handler compatibility while using the neutral hosted Chat-Completions boundary; unrelated providers do not change behavior.
- [ ] #2 Fresh configuration defaults use kimi-k3 for Moonshot and glm-5.1 for Z.ai while existing explicit model selections and historical model IDs remain usable.
- [ ] #3 Explicit arguments, canonical configuration, environment credentials, narrow legacy Moonshot fallback, and structural endpoint validation follow one documented fail-closed precedence contract without mutating source configuration or disclosing secrets.
- [ ] #4 Streaming and non-streaming calls validate provider payloads, errors, finish states, usage, retry budgets, cancellation, resource ownership, and size/depth bounds consistently.
- [ ] #5 Existing Chatbook function tools complete joined Console continuation for both providers; Kimi reasoning metadata is bounded, invisible, ephemeral, and call-scoped; Z.ai eligibility is enabled only after joined proof.
- [ ] #6 Moonshot and Z.ai model discovery uses the same normalized endpoint and credential resolution as chat, preserves prior cache on failure, and never logs sensitive payloads.
- [ ] #7 Canonical Settings exposes actionable readiness, save, search/focus, endpoint, credential, model, and reasoning guidance without adding an API-mode selector or vendor built-in tools.
- [ ] #8 QwenCloud Chat behavior remains unchanged after any shared parser extraction, proven by its complete contract suite and mutation checks.
- [ ] #9 README and provider guides document current defaults, endpoints, exact supported controls, tool limitations, usage behavior, recovery, and optional isolated live verification.
- [ ] #10 Default tests make no paid calls; optional Moonshot and Z.ai live tests require explicit gates plus credentials and run in isolated subprocess profiles.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and approve the provider design and ADR. 2. Produce a TDD implementation plan with explicit review gates. 3. Extract neutral hosted Chat-Completions wire primitives while preserving QwenCloud Chat behavior. 4. Migrate and harden Moonshot/Kimi. 5. Migrate and harden Z.ai/GLM and native tools. 6. Integrate call-scoped metadata and usage through Console. 7. Update defaults, Settings, catalog, docs, and live gates. 8. Run scoped and repository-wide verification, review, and task closure. ADR required: yes. ADR path: backlog/decisions/062-hosted-chat-completions-provider-boundary.md. Reason: this establishes a reusable provider transport contract and ephemeral metadata interface across adapter, gateway, and agent layers.
<!-- SECTION:PLAN:END -->
