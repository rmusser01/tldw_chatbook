# Provider Default Model Refresh — Design

**Date:** 2026-07-26
**Backlog:** [TASK-519](../../../backlog/tasks/task-519%20-%20Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md)
**ADR required:** No
**ADR path:** [ADR-020](../../../backlog/decisions/020-automatic-model-catalog-refresh.md)
**Reason:** This refresh changes bundled provider defaults and model-aware request
shaping inside existing provider boundaries. ADR-020 already governs model catalog
discovery and persistence; no storage, ownership, service-contract, or provider
boundary changes are introduced.

## Summary

Refresh the bundled balanced general-purpose defaults to:

| Provider | New default | Role |
| --- | --- | --- |
| DeepSeek | `deepseek-v4-flash` | Fast, economical general-purpose model |
| Anthropic | `claude-sonnet-5` | Balanced speed and intelligence |
| OpenAI | `gpt-5.6-terra` | Balanced intelligence and cost |

The change is not a blind string replacement. Claude Sonnet 5 rejects non-default
sampling controls, while GPT-5.6 needs a model-aware Chat Completions token and
reasoning contract to preserve the current non-reasoning, tool-compatible default
behavior. The implementation therefore updates both configuration defaults and
the smallest required provider request-shaping logic.

## Goals

- Give fresh installations current, vendor-supported balanced defaults.
- Keep provider selectors useful by retaining supported alternative models.
- Preserve the existing provider endpoints and normalized response contracts.
- Preserve explicit user configuration; bundled defaults must not overwrite a
  user's persisted model choices.
- Keep ordinary GPT-5.6 chat and function-tool requests on the existing Chat
  Completions path with non-reasoning behavior unless the user explicitly selects
  Responses-only reasoning controls.
- Ensure the new OpenAI and Anthropic defaults remain recognized as vision-capable.

## Non-goals

- Automatically rewriting existing user `config.toml` files.
- Replacing historical examples, eval baselines, fixtures, or tests whose model
  IDs are part of the scenario under test.
- Changing specialized TTS, embeddings, transcription, RAG, character-chat, or
  media-analysis defaults.
- Removing older models that remain supported.
- Adding GPT-5.6 optional features such as Pro mode, persisted reasoning,
  programmatic tool calling, explicit prompt caching, or multi-agent execution.
- Reworking the existing Responses API tool-call normalization contract.
- Live provider benchmarking or choosing defaults dynamically by price.

## Vendor Basis

### DeepSeek

DeepSeek describes V4 Flash as the fast, efficient, economical V4 option whose
reasoning approaches V4 Pro and whose simple-agent performance is comparable to
V4 Pro. The legacy `deepseek-chat` and `deepseek-reasoner` names retired on
2026-07-24, so they must no longer be bundled as usable current defaults.

Source: <https://api-docs.deepseek.com/news/news260424/>

### Anthropic

Anthropic describes Claude Sonnet 5 as the best combination of speed and
intelligence. It is available through the Claude API as `claude-sonnet-5`.
Sonnet 5 enables adaptive thinking by default and rejects non-default
`temperature`, `top_p`, and `top_k` values.

Sources:

- <https://platform.claude.com/docs/en/about-claude/models/overview>
- <https://platform.claude.com/docs/en/about-claude/models/whats-new-sonnet-5>

### OpenAI

OpenAI describes GPT-5.6 Terra as the family tier that balances intelligence and
cost. It supports both Responses and Chat Completions, including function calling
and streaming. The explicit `gpt-5.6-terra` ID is preferred over the `gpt-5.6`
alias because the alias routes to the higher-capability Sol tier.

Sources:

- <https://developers.openai.com/api/docs/guides/latest-model>
- <https://developers.openai.com/api/docs/models/gpt-5.6-terra>
- <https://developers.openai.com/api/docs/guides/upgrading-to-gpt-5p6-sol>

## Configuration Design

### Bundled provider catalogs

Update both bundled catalog representations in `tldw_chatbook/config.py` so they
stay consistent:

- Python `API_MODELS_BY_PROVIDER`
- Embedded TOML `[providers]`

The relevant leading entries become:

- DeepSeek: `deepseek-v4-flash`, `deepseek-v4-pro`
- Anthropic: `claude-sonnet-5`, followed by current supported capability and
  efficiency alternatives
- OpenAI: `gpt-5.6-terra`, `gpt-5.6-sol`, `gpt-5.6-luna`, followed by supported
  existing alternatives

Remove `deepseek-chat` and `deepseek-reasoner` from the bundled DeepSeek list
because the official endpoint no longer serves them. Do not delete historical
references outside active bundled defaults and catalogs.

### Provider defaults and fallbacks

Update every active provider-default layer for these three providers:

- Embedded TOML `[api_settings.openai].model`,
  `[api_settings.anthropic].model`, and `[api_settings.deepseek].model`
- The `load_settings()` legacy fallback values
- The provider-handler fallback values used when the corresponding configuration
  entry is absent
- The global `[chat_defaults].model`, because its provider is OpenAI
- Minimal hardcoded catalog fallbacks used only when the embedded provider table
  is absent or malformed

Do not update `[character_defaults]`, `[analysis_defaults]`, or unrelated
feature-specific model choices in this task.

### Existing user configuration

The embedded defaults continue to seed fresh configuration. Existing persisted
values keep their current precedence and are not silently migrated. In
particular, an existing user-selected `deepseek-chat` value is not rewritten at
load time; the provider will return its normal unsupported-model error until the
user selects a current model.

## Provider Request Compatibility

### OpenAI GPT-5.6

Add a small model-family predicate for GPT-5.6 IDs and keep endpoint selection
behavior explicit:

1. For ordinary GPT-5.6 requests with no Responses-only controls, use Chat
   Completions.
2. Preserve the pre-migration non-reasoning behavior by sending
   `reasoning_effort: "none"` on that Chat Completions request.
3. Use `max_completion_tokens` instead of deprecated `max_tokens` for GPT-5.6
   Chat Completions requests.
4. Preserve `max_output_tokens` on Responses requests.
5. Preserve the existing Chat Completions function-tool schema and response
   normalization.
6. Continue using Responses when `reasoning_summary` or `verbosity` is set, or
   when an explicit reasoning effort other than `none` is selected.
7. Treat an explicit `reasoning_effort: "none"` without other Responses-only
   controls as a Chat Completions request, matching the ordinary default path.
8. Leave older OpenAI model request shaping unchanged.

This is a compatibility migration, not adoption of new GPT-5.6 capabilities.

### Anthropic Claude Sonnet 5

Add model-aware Anthropic predicates so that:

1. `claude-sonnet-5` is recognized as an adaptive-thinking model.
2. Explicit supported thinking effort maps to Anthropic adaptive thinking.
3. Fixed thinking budgets are ignored for adaptive-thinking models, matching the
   existing modern-Opus behavior.
4. `temperature`, `top_p`, and `top_k` are omitted for Claude Sonnet 5 even when
   generic or persisted defaults contain them.
5. Existing sampling behavior remains unchanged for older models that support
   these fields.

The default Sonnet 5 request may omit a `thinking` object because Anthropic
enables adaptive thinking by default.

### DeepSeek V4 Flash

The existing DeepSeek Chat Completions endpoint and payload shape are retained.
Only the model default, bundled catalog, and fallback IDs change. No new
thinking-mode behavior is introduced.

## Capability Metadata

Update `model_capabilities.py` and the embedded `[model_capabilities]` defaults
only as needed for the selected defaults:

- Recognize GPT-5.6 text-and-image models as vision-capable.
- Recognize Claude Sonnet 5 as vision-capable.
- Do not infer unverified DeepSeek V4 image support.

Existing capability precedence and user overrides remain unchanged.

## Error Handling

- Provider HTTP and configuration errors continue through the existing typed
  error paths.
- No silent fallback to a different model occurs after a provider rejects a
  configured model.
- Existing user configurations referencing retired models remain visible and
  actionable rather than being rewritten behind the user's back.
- Model-aware request shaping must be limited to exact, documented family
  predicates so unrelated custom or provider-compatible model IDs are unchanged.

## Testing

Add or extend focused tests for:

1. Embedded TOML parsing returns the three new provider defaults.
2. Bundled provider catalogs lead with the recommended models and exclude the
   retired DeepSeek aliases.
3. Legacy and handler fallbacks match the new provider defaults.
4. GPT-5.6 Terra ordinary Chat Completions requests send
   `reasoning_effort: "none"` and `max_completion_tokens`.
5. GPT-5.6 explicit higher reasoning still selects Responses and uses
   `max_output_tokens`.
6. GPT-5.6 Chat Completions tool payloads remain in the existing function-tool
   shape.
7. Claude Sonnet 5 omits `temperature`, `top_p`, and `top_k`.
8. Claude Sonnet 5 maps explicit thinking effort to adaptive thinking.
9. DeepSeek V4 Flash uses the existing `/chat/completions` request contract.
10. GPT-5.6 Terra and Claude Sonnet 5 are recognized as vision-capable.

Run the smallest relevant configuration, chat-provider, catalog, and capability
test modules, followed by lint or formatting checks for changed files. Live API
smoke tests are optional and must run only when the relevant environment keys are
already available.

## Documentation and Task Closeout

- Link this design and ADR-020 from TASK-519's implementation plan.
- Record exact changed files, compatibility decisions, and verification evidence
  in TASK-519 implementation notes.
- Mark acceptance criteria complete and move TASK-519 to Done only after all
  required tests and static checks pass.
