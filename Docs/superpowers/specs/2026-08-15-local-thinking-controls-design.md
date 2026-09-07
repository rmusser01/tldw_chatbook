# Local Provider Thinking Controls — Design

Date: 2026-08-15
Status: Draft (pending user review)
Related ADR: [ADR-066: Local provider thinking control wire formats](../../backlog/decisions/066-local-provider-thinking-controls.md)
Related Task: [TASK-17170](../../backlog/tasks/task-17170%20-%20Console-thinking-levels-and-budget-for-local-providers.md)

## Context

Qwen3.8-27B ships adjustable thinking (`reasoning_effort`: `low` / `medium` /
`xhigh`, default `xhigh`) plus separate caps for reasoning vs. final output.
The Console settings modal already has generic **Reasoning** (`reasoning_effort`),
**Thinking** (`thinking_effort`), and **Budget** (`thinking_budget_tokens`)
fields, and these flow all the way into provider resolution for local
providers (`LlamaCppProviderConfig` carries them) — but they are dropped at
the last hop:

- The direct llama.cpp path (`build_llamacpp_chat_payload`,
  `stream_llamacpp_chat`, `complete_llamacpp_chat` in
  `Chat/console_provider_gateway.py`) only forwards sampling params.
- The adapter path drops them in `PROVIDER_PARAM_MAP` (`Chat/Chat_Functions.py`)
  — no local execution key maps `reasoning_effort` or `thinking_budget_tokens`.

Meanwhile the serving stacks all accept per-request thinking controls on the
OpenAI-compatible `/v1/chat/completions` endpoint every local handler already
targets:

| Stack | Level | Hard budget |
|---|---|---|
| llama.cpp (`--jinja`) | `chat_template_kwargs.reasoning_effort` (top-level `reasoning_effort` is not parsed) | top-level `reasoning_budget_tokens` (≥ b9982 / PR #23116) |
| vLLM | top-level `reasoning_effort` | none per-request |
| mlx-lm server | to verify live (`chat_template_kwargs` suspected) | to verify live |

Unknown template kwargs are unused Jinja variables — verbatim values a
template does not consume degrade to the model default instead of erroring.
That property is what makes verbatim sending safe.

## Goals

1. The Console's Reasoning and Budget fields control thinking depth and max
   thinking tokens for all Console-reachable local providers.
2. Enabling thinking never pollutes the visible Console reply with raw
   `<think>` text **on the llama.cpp direct path** (both server-split and
   unsplit shapes). The adapter path (vLLM/Custom without a server-side
   reasoning parser) keeps server-default behavior; see Follow-ups.
3. Values the selected model's template does not consume produce a warning,
   not a block and not a silent rewrite.

## Non-goals

- No new Console settings fields (reuse `reasoning_effort` /
  `thinking_budget_tokens`). A free-form `chat_template_kwargs` JSON escape
   hatch may be a future task.
- No changes to cloud providers (OpenAI/Anthropic/QwenCloud/Moonshot/Z.ai
  mappings stay as-is).
- No change to managed-server launch defaults (`--jinja` etc. remain the
   user's launch args; surfaced via hint + docs instead).
- Legacy Chat window is untouched; it benefits automatically where it calls
  `chat_api_call` with these params, which today it does not.

## Semantics

- `reasoning_effort` is the local thinking-level knob. `thinking_effort`
  (Anthropic-style) stays Anthropic-only; sending both to one provider would
  be a double signal. The modal marks Anthropic-only fields as
  "no effect on this provider" for local selections.
- `thinking_budget_tokens` is the max-thinking-tokens knob.
- Both are sent **verbatim**. `reasoning_effort: none` additionally sends
  `chat_template_kwargs: {"enable_thinking": false}`.
- Validation floor for `thinking_budget_tokens` stays ≥ 1024 globally. The
  floor exists as a sane thinking-budget minimum (below it, reasoning is cut
  uselessly early) and the validator has no provider context; introducing
  provider-aware floors is not worth the plumbing.
- Prefill precedence: an assistant prefill still forces
  `enable_thinking: false` (llama.cpp rejects prefill + thinking). Effective
  precedence is `prefill > none > effort`.

## Wire-format composition

One table (data, not scattered conditionals) keyed by execution key, placed
with the other provider-support constants in `Chat/console_provider_support.py`:

| Execution key | Level goes to | Budget goes to |
|---|---|---|
| `llama_cpp`, `local_llamacpp`, `local-llm`, `local_llamafile` | `chat_template_kwargs.reasoning_effort`; `enable_thinking: false` when effort is `none` — live-verify the model's template consumes `reasoning_effort` as a kwarg | top-level `reasoning_budget_tokens` |
| `vllm`, `local_vllm` | top-level **and** `chat_template_kwargs.reasoning_effort` (same value in both; whichever path that stack consumes wins and the other is ignored — guards against the top-level-only assumption being wrong) | dropped, debug-logged |
| `local_mlx_lm` | `chat_template_kwargs.reasoning_effort` if live verification confirms support; otherwise dropped, debug-logged | same verification outcome |
| `custom-openai-api`, `custom-openai-api-2` | top-level `reasoning_effort` only (no llama.cpp-specific fields — strict OpenAI proxies may reject them) | dropped |

Docs note: Custom OpenAI endpoints get levels only when the target server
parses top-level `reasoning_effort` (vLLM/SGLang-style). For llama.cpp
targets, use the first-class llama.cpp provider entries.

## Thinking-output handling

*Decision defaulted to "in scope: split + strip" after an unanswered scope
question; veto at spec review if undesired.*

- **Server-split case:** the direct llama.cpp SSE parser
  (`_content_from_sse_line` path) consumes `reasoning_content` separately from
  `content` so servers launched with `--reasoning-format deepseek/auto` keep
  thinking out of the visible reply. The non-streaming direct path reads
  `choices[0].message.content` only and never concatenates
  `message.reasoning_content`.
- **Unsplit case:** stream-aware `<think>…</think>` stripping in the Console's
  direct path, **anchored to the start of the response**: only a think block
  that opens at the very beginning of the streamed text (leading whitespace
  tolerated) is stripped. Qwen templates emit thinking first — some
  generations even emit an empty `<think>\n\n</think>` prefix in no-think
  mode, which start-anchoring also handles — while a literal `<think>` in the
  middle of a reply (e.g. the user asked for an XML example) is legitimate
  content and must survive. Stateful parsing across chunk boundaries; never
  emit a partial opening tag; if the stream ends inside an unterminated
  start-anchored think block, drop the tail. The legacy non-streaming
  `strip_thinking_tags` post-processor in `Chat/Chat_Functions.py` is
  reference, not reuse — it is non-streaming, config-gated, and not
  start-anchored.
- Adapter-path local handlers rely on server-side splitting; their payloads
  change only as listed under wire formats. Stream stripping for adapter-path
  providers without a server-side reasoning parser is a follow-up, not part
  of this task.

## UI changes

- **Model-family hints:** a small table mapping model-name patterns to the
  effort values that family's templates consume, generation-aware because
  Qwen3 generations differ: `qwen3.5`/`qwen3.8` (and dotted successors) →
  `low`/`medium`/`xhigh`; original `qwen3-`/`qwen3 ` generations →
  toggle-only (`none` does something, any other value warns); `gpt-oss` →
  `low`/`medium`/`high`. Drives (a) the settings-modal placeholder for the
  Reasoning field and (b) a non-blocking warning when the typed value falls
  outside the hint set. Heuristic only; never blocks a send; unknown models
  produce no warning.
- **Provider-aware field hints:** Thinking / Summary / Verbosity fields show
  "no effect on this provider" styling/hint when a local provider is selected.
- **Request preview:** the modal's request preview renders the composed
  thinking fields (`chat_template_kwargs`, `reasoning_budget_tokens`) so users see
  exactly what is sent — important because values pass verbatim.
- **Readiness hint:** when a thinking control is set on a llama.cpp-family
  provider, a one-line hint notes the server requirements (`--jinja`; budget
  requires llama.cpp ≥ b9982). No network probing to detect the build.

## Code touch-points

1. `Chat/console_provider_gateway.py`
   - `build_llamacpp_chat_payload`, `stream_llamacpp_chat`,
     `complete_llamacpp_chat`: accept `reasoning_effort` and
     `thinking_budget_tokens`; compose the payload per the table; merge with
     the existing prefill rule.
   - The direct-path call sites (~lines 2002 and 2216) pass the two fields
     from the resolution.
   - SSE parsing: `reasoning_content` handling + stream-aware `<think>`
     stripping.
2. `Chat/Chat_Functions.py` — `PROVIDER_PARAM_MAP`: add
   `reasoning_effort` / `thinking_budget_tokens` entries for the covered
   local keys.
3. `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` —
   `chat_with_vllm`, `chat_with_llama`, `chat_with_local_llm`,
   `chat_with_mlx_lm`, `chat_with_custom_openai`, `chat_with_custom_openai_2`
   accept the params and add them to the payload per the table.
4. `Chat/console_session_settings.py` — model-family hint table + warning
   generation (non-blocking).
5. `Widgets/Console/console_settings_modal.py` — placeholder/hints/warnings
   rendering; request-preview extension.

Known behavior carried forward: auxiliary requests (title generation etc.)
already forward `reasoning_effort` through `chat_api_call` for mapped
providers — cloud does this today; local reaches parity. Documented, not
changed.

## Testing

- Unit: payload builder composition per provider (level, `none` →
  `enable_thinking`, budget, prefill precedence, verbatim passthrough,
  vLLM dual placement); param-map entries; each handler's payload;
  hint-table warnings (match, non-match, generation mismatch, unknown
  model); SSE `reasoning_content` split; stream-aware tag stripping
  including chunk-boundary, unterminated-block, empty-prefix-block, and
  mid-reply-literal-tag-must-survive cases.
- Live (per `backlog/docs/lessons-live-verification.md`), against a real
  `llama-server` (current build, `--jinja`) serving a Qwen3.8-27B GGUF:
  - `low`/`medium`/`xhigh` observably change thinking depth (this also
    confirms the template consumes `chat_template_kwargs.reasoning_effort`);
  - `reasoning_budget_tokens` truncates thinking, checked both with and without
    `--reasoning-format` (the truncation mechanism may depend on it);
  - an older-build server (pre-b9982 if available) ignores the unknown
    `reasoning_budget_tokens` field rather than 400ing;
  - with and without `--reasoning-format`, the visible reply contains no
    `<think>` text.
- Live-check `mlx_lm.server` and llamafile `chat_template_kwargs` support
  before committing to their table rows; fall back to drop-and-log if
  unsupported.
- vLLM live check if available; otherwise the vLLM row ships on documented
  API behavior with unit tests only.

## Follow-ups (explicitly out of scope)

- Free-form `chat_template_kwargs` JSON escape hatch in Console settings.
- vLLM per-request thinking budget, if vLLM gains one.
- Thinking output surfaced in a collapsible Console "thinking" view rather
  than dropped.
- Stream-aware `<think>` stripping for adapter-path local providers
  (vLLM/Custom without a server-side reasoning parser); the direct-path
  stripper ships as a reusable helper so this is a thin follow-up.
- `--jinja` / `--reasoning-format` defaults for managed llama.cpp launches.

## ADR check

ADR required: yes — this fixes per-provider wire-format contracts for
thinking controls, which future contributors will otherwise re-litigate
("why is effort inside `chat_template_kwargs` for llama.cpp but top-level
for vLLM?"). Recorded as
[ADR-066](../../backlog/decisions/066-local-provider-thinking-controls.md).

## Errata (post live verification, 2026-08-15)

Verified against a real `llama-server` (b10430, `--jinja`) serving
Qwen3.8-27B:

1. **Budget field name:** the per-request field is
   `reasoning_budget_tokens`, not `reasoning_budget` — the latter is
   silently ignored by llama.cpp. Live: budget 8 → 35 reasoning chars,
   32 → 101, vs 391 natural. The wire-format table rows above are updated.
2. **Validated templates:** Qwen3.8's chat template *validates*
   `reasoning_effort` and raises on unknown values (effort `minimal` →
   HTTP 500: `raise_exception` outside `('xhigh','medium','low')`).
   `high` is aliased to `xhigh` by the template and is safe; `none` is
   safe because it is paired with `enable_thinking: false`, which
   short-circuits the validation block. The design assumption that
   "unused template kwargs degrade gracefully" was wrong for validated
   templates: non-safe efforts are now dropped from
   `chat_template_kwargs` (debug-logged) rather than sent. vLLM and
   Custom OpenAI top-level `reasoning_effort` stays verbatim for all
   values.
3. **Budget is not template-consumed:** `thinking_budget_tokens` /
   `reasoning_budget_tokens` is a server-side mechanism only — no chat
   template reads it.
