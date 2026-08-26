---
id: TASK-3771
title: Add QwenCloud dual API provider support
status: Done
assignee:
  - '@codex'
created_date: '2026-08-03 03:24'
updated_date: '2026-08-12 13:44'
labels:
  - provider
  - qwencloud
  - settings
  - tools
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-02-qwencloud-dual-api-provider-design.md
  - >-
    Docs/superpowers/plans/2026-08-11-qwencloud-dual-api-provider-implementation.md
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
- [x] #1 QwenCloud is selectable anywhere an ordinary hosted Console provider is selectable and passes shared provider readiness with its own endpoint and credential configuration.
- [x] #2 `api_mode` persists under `api_settings.qwencloud`, defaults to `responses`, accepts only `responses` or `chat_completions`, and is pinned with the effective base URL for every turn of a Console run.
- [x] #3 Streaming and non-streaming text, finish state, and token usage work through the standard provider dispatcher and Console gateway in both API modes.
- [x] #4 Existing Chatbook function tools work in both modes through the existing native Console agent runtime, including provider-emitted multiple-call batches, exact call/result pairing, errors, cancellation, and structured continuation without synthetic user messages.
- [x] #5 QwenCloud model discovery participates in the existing cached model-catalog pipeline and does not create a parallel model source.
- [x] #6 Canonical F9 Settings exposes QwenCloud API mode without changing another provider's configuration or behavior.
- [x] #7 Request translation uses mode-specific parameter allowlists and enforces Responses call/output adjacency and stateless-request invariants plus Chat Completions reasoning-replay safety; invalid tools, messages, endpoints, modes, and unsupported content fail before network I/O.
- [x] #8 Automated tests cover provider parity, registration, pinned resolution, configuration, translation, SSE framing/de-duplication, usage, native-tool continuation, discovery, retries, cancellation, and safe errors without paid API calls.
- [x] #9 Provider documentation records setup, API-mode behavior, regional endpoint guidance, state/reasoning behavior, parameter limitations, existing function-tool support, built-in-tool exclusion, and optional isolated live verification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added one first-class `qwencloud` provider across configuration, readiness,
  dispatch, Console, canonical F9 Settings, native function tools, and the
  shared cached model catalog. The dedicated adapter normalizes Responses and
  Chat Completions into the existing internal contracts; no Qwen-specific
  agent loop, cache, Settings surface, or built-in-tool execution path was
  introduced.
- Implemented Chatbook-side stateless Responses history without sending
  provider continuation IDs, requesting `store=false` where honored, and
  without relying on provider-managed session state. This makes no claim
  about provider operational retention or caching. Also implemented exact
  call/output pairing, Chat preserved-thinking replay disabled, fail-closed
  mode-specific parameters, record-aware streaming, terminal usage, bounded
  retries, and deterministic response closure/cancellation.
- Documented setup, endpoint and mode selection, parameter limits, tool scope,
  discovery, pricing uncertainty, recovery, and optional paid verification in
  `README.md` and the Settings/Console user guides. Added a two-mode live test
  isolated by `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`,
  and `[paths].data_dir` before Chatbook imports.
- Rebased the complete feature stack onto current `origin/dev` (`1b30717cb`)
  before final verification. The focused adapter, streaming, provider
  contract, native-tool, Settings, readiness, catalog, gateway, bridge, usage,
  and live-test collection gate passed with `517 passed, 2 skipped, 423
  deselected`. The two skips are the paid live modes; `TLDW_LIVE_QWENCLOUD`
  remained unset and no paid request was made.
- Every changed test file then produced `1498 passed, 2 skipped, 2 xfailed, 4
  failed`. The four failures were the existing canonical Settings nodes
  `test_settings_ownership_records_cover_categories_and_runtime_boundaries`,
  `test_settings_console_behavior_saves_display_name_exactly`,
  `test_settings_provider_category_saves_provider_defaults_without_sampling`,
  and `test_settings_provider_switch_does_not_save_stale_endpoint`; the exact
  four-node command failed identically on a clean current `origin/dev`
  worktree. A repository-wide run reached 89% after 8h35m before becoming stale
  when `origin/dev` advanced; it was interrupted and is not used as passing
  feature evidence.
- Ruff lint, MyPy for the QwenCloud URL/request/stream and discovery modules,
  compileall, and `git diff --check` pass on the rebased tree. Ruff format finds
  the same four legacy shared files on both the branch and clean
  `origin/dev`; all feature-owned/new files are formatted and there is no new
  formatter debt. Documentation term/link/anchor and live-harness isolation
  checks pass; no credential, prompt, or response content is emitted.
- Added the TASK-3771 cancellation incident to
  `backlog/docs/lessons-testing-evidence.md`: the original test cancelled at
  request receipt and survived a text-only mutation, while the corrected test
  triggers from a real downstream partial-tool checkpoint and proves exact
  stream closure without execution.
- Architecture: follows
  [ADR-045](../decisions/045-qwencloud-dual-api-provider-boundary.md), with no
  implementation deviation from its provider/runtime ownership. No schema
  migration was required.
<!-- SECTION:NOTES:END -->

## Task Identity Note

This task was renumbered from `TASK-1336` after a full remote-ref and worktree
sweep found an older SoundDevice/VAD task using the same identifier. It moved
again from `TASK-3603` to `TASK-3771` after the completed Watchlists phase 3
task merged to `dev` with the same identifier. `TASK-3771` was unclaimed across
all remote refs and checked worktrees on 2026-08-08.
