# ADR-066: Local provider thinking control wire formats

Status: Proposed
Date: 2026-08-15
Related Task: [TASK-16812](../tasks/task-16812%20-%20Console-thinking-levels-and-budget-for-local-providers.md)
Related Spec: [Local Provider Thinking Controls — Design](../../Docs/superpowers/specs/2026-08-15-local-thinking-controls-design.md)
Supersedes: N/A

## Decision

Chatbook reuses the Console's existing generic reasoning settings —
`reasoning_effort` for thinking level and `thinking_budget_tokens` for the
max-thinking-tokens cap — as the thinking controls for local providers,
rather than adding provider-specific fields. The Anthropic-style
`thinking_effort` remains Anthropic-only.

Values are sent verbatim. `reasoning_effort: none` additionally sends
`chat_template_kwargs: {"enable_thinking": false}`. Values a template does
not consume are unused Jinja variables and degrade to the model default;
Chatbook warns (model-family heuristic, non-blocking) instead of rewriting
them. The `thinking_budget_tokens` floor stays a global ≥ 1024.

Each local execution key composes its payload through one wire-format table
(see the spec) rather than per-call conditionals:

- llama.cpp family (`llama_cpp`, `local_llamacpp`, `local-llm`,
  `local_llamafile`): level inside `chat_template_kwargs.reasoning_effort`
  (llama-server does not parse top-level `reasoning_effort`); budget as
  top-level `reasoning_budget_tokens`, supported per-request since llama.cpp
  b9982
  and ignored by older servers.
- vLLM family (`vllm`, `local_vllm`): the level goes both top-level and
  inside `chat_template_kwargs.reasoning_effort` with the same value —
  whichever path that stack consumes wins and the other is ignored. This
  guards against the top-level-only assumption being wrong for a given vLLM
  release. Per-request budget unsupported, dropped and logged.
- Custom OpenAI endpoints (`custom-openai-api`, `custom-openai-api-2`):
  top-level OpenAI-style `reasoning_effort` only (no llama.cpp-specific
  fields — strict OpenAI proxies may reject them); budget dropped.
- `local_mlx_lm`: `chat_template_kwargs` pending live verification of
  `mlx_lm.server` support; drop-and-log if unsupported.

Prefill precedence stays `prefill > none > effort`: an assistant prefill
still forces `enable_thinking: false` because llama.cpp rejects prefilled
requests when the template's thinking mode is enabled.

Thinking output is kept out of the visible Console reply in both shapes:
the direct llama.cpp SSE path consumes `reasoning_content` separately
(server launched with `--reasoning-format`) and, when the server does not
split, stream-strips `<think>…</think>` blocks anchored to the start of the
response — mid-reply literal `<think>` text is legitimate content and
survives, and start-anchoring also removes the empty think prefix some Qwen
generations emit in no-think mode.

## Alternatives considered

- **Free-form `chat_template_kwargs` JSON field in Console settings.** Most
  flexible (any template kwarg), but duplicates the structured fields,
  needs JSON validation in a TUI input, and provides no levels semantics by
  itself. Rejected as the primary mechanism; a future escape hatch may layer
  on top.
- **Server-side only.** Launch `llama-server` with `--reasoning-budget` /
  `--chat-template-kwargs` defaults (the managed launcher passes user args
  through today). No request-path code, but not per-session/per-request and
  invisible to the Console. Rejected as the primary mechanism; launch-flag
  guidance is documented alongside.
- **Clamp unsupported values to the nearest supported one.** Rejected: the
  request would not match what the user typed, which is visible in the
  request preview and trust-eroding. Verbatim + warn chosen instead.

## Consequences

- Qwen3.8-27B's `low`/`medium`/`xhigh` levels and a hard thinking cap work
  from the Console against llama.cpp (≥ b9982 for the cap) and vLLM without
  new UI.
- Servers older than b9982 silently ignore `reasoning_budget_tokens`; the Console
  surfaces the build requirement as a hint rather than probing.
- The model-family warning table is a heuristic on model names and can be
  stale for new releases; it never blocks sends.
- Auxiliary requests (title generation etc.) inherit session thinking
  settings on mapped providers, matching existing cloud behavior.


## Errata (live verification)

Live verification (2026-08-15, `llama-server` b10430 `--jinja` +
Qwen3.8-27B) amended three points:

1. The per-request budget field is `reasoning_budget_tokens`, not
   `reasoning_budget` — the latter is silently ignored by llama.cpp
   (live: budget 8 → 35 reasoning chars, 32 → 101, vs 391 natural).
2. Qwen3.8's chat template validates `reasoning_effort` and raises on
   unknown values (effort `minimal` → HTTP 500). `high` is aliased to
   `xhigh` by the template and is safe; `none` is safe because it pairs
   with `enable_thinking: false`, which short-circuits the validation
   block. Non-safe efforts are dropped from `chat_template_kwargs`
   (debug-logged) rather than sent; vLLM and Custom OpenAI top-level
   `reasoning_effort` stays verbatim for all values.
3. The budget is not consumed by chat templates — it is a server-side
   mechanism only.
