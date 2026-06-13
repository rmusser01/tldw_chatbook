# ADR 006: Provider-Aware Generation Settings

Status: Accepted
Date: 2026-06-08
Related Task: [backlog/tasks/task-83 - Add-provider-aware-generation-settings-and-thinking-controls.md](../tasks/task-83%20-%20Add-provider-aware-generation-settings-and-thinking-controls.md)
Supersedes: N/A

## Decision

Settings owns persisted global and per-model generation defaults, Console owns effective session resolution, and provider adapters own provider-specific request-shape translation for sampler and thinking controls.

## Context

Settings is the application configuration hub and Console is the primary agentic control surface. Existing Settings provider/model defaults expose only a partial sampling surface, while `chat_api_call()` and several provider adapters already support more generation controls such as seed and penalties. Newer OpenAI and Anthropic models also expose reasoning or thinking controls, but those controls are not interchangeable sampler fields: OpenAI uses a reasoning request object, while Anthropic uses thinking-specific request fields and token-budget constraints.

Users need all Console-supported providers to share a consistent configuration path without Settings pretending that unsupported controls are active. The implementation must preserve current provider identity normalization, per-model profile precedence, Console session override behavior, and explicit unsupported/WIP copy for unavailable paths.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add all controls directly to `chat_defaults` as untyped generic keys | This would make unsupported providers appear configurable and would allow provider-specific controls to leak into incompatible request payloads. |
| Put OpenAI and Anthropic thinking controls only in provider adapters | Runtime support alone would leave Settings unable to persist or explain the controls, and Console would still lack a consistent effective-settings contract. |
| Create a separate provider capability registry for this slice | A new registry would duplicate existing provider normalization and model-discovery ownership. Capability detection can be added later, but this slice should use explicit provider-key support lists and clear unavailable copy. |
| Expose only UI fields and defer runtime forwarding | That would repeat the current broken Settings pattern where controls render but do not reliably affect Console behavior. |

## Consequences

Global defaults remain under `chat_defaults`; selected provider/model overrides remain under `api_settings.<provider>.model_defaults.<model>`. Console effective settings must resolve per-model overrides first, then global chat defaults, then provider-level defaults, then hard fallbacks.

Common sampler controls are stored and forwarded as provider-neutral fields only when populated and valid. Provider-specific reasoning or thinking controls are stored in the same effective settings contract but are mapped only by providers that support them. Unsupported providers must show unavailable copy or omit disabled fields instead of silently sending invalid request payloads.

Provider adapters remain responsible for translating internal fields into external API payloads and for omitting blank unsupported values. Future provider capability metadata can refine which controls are visible per model without changing the storage or Console resolution boundary.

## Links

- [Backlog task TASK-83](../tasks/task-83%20-%20Add-provider-aware-generation-settings-and-thinking-controls.md)
- [OpenAI Responses API overview](https://developers.openai.com/api/reference/responses/overview)
- [Anthropic extended thinking documentation](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
