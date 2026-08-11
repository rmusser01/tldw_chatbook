---
id: TASK-3771
title: Add QwenCloud dual API provider support
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-03 03:24'
updated_date: '2026-08-11'
labels:
  - provider
  - qwencloud
  - settings
  - tools
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md
  - Docs/superpowers/plans/2026-08-11-qwencloud-dual-api-provider-implementation.md
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
- [ ] #2 `api_mode` persists under `api_settings.qwencloud`, defaults to `responses`, accepts only `responses` or `chat_completions`, and is pinned with the effective base URL for every turn of a Console run.
- [ ] #3 Streaming and non-streaming text, finish state, and token usage work through the standard provider dispatcher and Console gateway in both API modes.
- [ ] #4 Existing Chatbook function tools work in both modes through the existing native Console agent runtime, including provider-emitted multiple-call batches, exact call/result pairing, errors, cancellation, and structured continuation without synthetic user messages.
- [ ] #5 QwenCloud model discovery participates in the existing cached model-catalog pipeline and does not create a parallel model source.
- [ ] #6 Canonical F9 Settings exposes QwenCloud API mode without changing another provider's configuration or behavior.
- [ ] #7 Request translation uses mode-specific parameter allowlists and enforces Responses call/output adjacency and stateless-request invariants plus Chat Completions reasoning-replay safety; invalid tools, messages, endpoints, modes, and unsupported content fail before network I/O.
- [ ] #8 Automated tests cover provider parity, registration, pinned resolution, configuration, translation, SSE framing/de-duplication, usage, native-tool continuation, discovery, retries, cancellation, and safe errors without paid API calls.
- [ ] #9 Provider documentation records setup, API-mode behavior, regional endpoint guidance, state/reasoning behavior, parameter limitations, existing function-tool support, built-in-tool exclusion, and optional isolated live verification.
<!-- AC:END -->

## Task Identity Note

This task was renumbered from `TASK-1336` after a full remote-ref and worktree
sweep found an older SoundDevice/VAD task using the same identifier. It moved
again from `TASK-3603` to `TASK-3771` after the completed Watchlists phase 3
task merged to `dev` with the same identifier. `TASK-3771` was unclaimed across
all remote refs and checked worktrees on 2026-08-08.

## Implementation Plan

The executable TDD plan is
`Docs/superpowers/plans/2026-08-11-qwencloud-dual-api-provider-implementation.md`.
Its work is ordered as follows:

1. Register QwenCloud configuration, identity, readiness, and endpoint rules.
2. Implement fail-closed dual-mode request, history, and tool translation.
3. Add non-streaming transport, normalization, retries, dispatch, and safe errors.
4. Add record-aware SSE translation and deterministic stream closure.
5. Pin mode/base in Console and carry usage/cancellation through the gateway.
6. Prove the real native-tool continuation path in both modes before enabling it.
7. Add the provider-isolated API mode selector to canonical F9 Settings.
8. Join the existing cached model-catalog pipeline.
9. Document, verify, self-review, and close out the task only when all evidence is complete.

ADR required: yes

ADR path: `backlog/decisions/045-qwencloud-dual-api-provider-boundary.md`

Reason: QwenCloud changes a provider/runtime boundary; ADR-045 already records
the approved dual-mode adapter ownership and pinned Console handoff, so a new
ADR would duplicate the existing decision.
