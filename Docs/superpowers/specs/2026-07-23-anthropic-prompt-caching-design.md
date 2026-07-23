# Anthropic prompt caching + cache-hit usage metrics (TASK-323 + 324)

**Date**: 2026-07-23
**Status**: Approved design, pending implementation plan
**Base**: origin/dev @ `e293b3313`
**Backlog**: TASK-323 (high), TASK-324 (medium, dep task-323)

## Why

Anthropic prompt caching is never activated: `chat_with_anthropic` sends the
system prompt as a plain string and tools as a plain list, with no
`cache_control` breakpoints. The request prefix is already byte-stable (audit +
the task-245 note `Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-
note.md` confirmed no cache-busting: no dynamic date, RAG/dictionary
substitution only touch the final user message, deterministic tool order), so
this is a pure cost/latency win for ordinary chats today (323). And no code
reads cache-hit usage fields — a grep for `cached_tokens`/`cache_read_input_
tokens`/`cache_creation_input_tokens` returns nothing — so there is no way to
confirm the win or measure the baseline (324). 324 is built alongside 323 and
verifies it (323 AC#4).

## Decisions (design)

1. **Always-on for cache-capable Claude models** (no config toggle — it is a
   pure win; YAGNI). Gated by `_anthropic_supports_caching(model)`.
2. **Metrics on the non-streaming path** (where `usage` is on the response,
   exactly like OpenAI's existing usage block). Streaming cache-usage (in the
   final SSE event) is a documented follow-up.
3. **AC#4 is verified structurally** (payload carries `cache_control`; metrics
   read `cache_read_input_tokens`) plus a documented **live** 2-request check;
   a literal cross-call cache read can't be a CI unit test.

## Architecture

Three focused changes, all in `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`, each
gated and graceful.

### Unit A — Anthropic `cache_control` breakpoints (323)

Add `_anthropic_supports_caching(model: str) -> bool` (module-level): `m =
(model or "").lower(); return m.startswith("claude-") and not
m.startswith("claude-2") and "instant" not in m`. This is True for all modern
Claude (3/3.5/3.7/4/future) and False for legacy `claude-2*`/`claude-instant*`,
which do not support caching (AC#3).

In `chat_with_anthropic`'s payload build (after `data = {...}`):
- **System (AC#1):** replace `if system_prompt is not None: data["system"] =
  system_prompt` with — when `_anthropic_supports_caching(current_model)` and
  `system_prompt` is a non-empty string — a structured content-block list with
  one breakpoint on the system prompt (the largest stable prefix):
  ```python
  if system_prompt is not None:
      if _anthropic_supports_caching(current_model) and system_prompt:
          data["system"] = [{
              "type": "text",
              "text": system_prompt,
              "cache_control": {"type": "ephemeral"},
          }]
      else:
          data["system"] = system_prompt  # unchanged for non-caching models
  ```
- **Tools (AC#2, optional breakpoint):** at the existing `data["tools"] =
  _anthropic_tools_payload(tools)` site, when the model supports caching and the
  converted list is non-empty, put a breakpoint on the **last** converted tool
  (Anthropic caches tools→system→messages, so this is a second, ≤4-total
  breakpoint; a fresh dict so the caller's input tools are never mutated):
  ```python
  if tools is not None:
      tools_payload = _anthropic_tools_payload(tools)
      if _anthropic_supports_caching(current_model) and tools_payload:
          tools_payload[-1] = {
              **tools_payload[-1],
              "cache_control": {"type": "ephemeral"},
          }
      data["tools"] = tools_payload
  ```

This applies to both streaming and non-streaming (the `data` dict is built
before the streaming branch), so caching works for streamed chats too.

**Caveats (documented, not blockers):**
- Anthropic **ignores** `cache_control` below the minimum cacheable length
  (~1024 tokens; 2048 for haiku) — no error, just no caching. Applying it is
  always safe; AC#4's live check must use a system prompt above the threshold.
- **Pre-existing baseline bug, out of scope:** `_anthropic_tools_payload`
  currently *drops* already-anthropic-shaped tools (returns `[]`), so
  `Tests/Chat/test_anthropic_native_tools.py::test_anthropic_shaped_tools_pass_
  through_untouched` is red on dev independent of this work (a separate
  tool-conversion bug, task-263 territory). This change does **not** fix it and
  is not affected by it — the tools breakpoint is gated on the *converted* list
  being non-empty, and is tested via the OpenAI-function-shaped → converted
  path (which works).

### Unit B — Cache-hit usage metrics (324)

- **OpenAI:** extend the existing non-streaming usage `log_histogram` block in
  `chat_with_openai` with a cached-tokens metric:
  ```python
  log_histogram(
      "openai_api_cached_tokens",
      (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
      labels={"model": final_model},
  )
  ```
- **Anthropic:** `chat_with_anthropic` has *no* usage logging today. Add a block
  right after the non-streaming `response_data = response.json()` (the existing
  Anthropic non-streaming return path):
  ```python
  anth_usage = response_data.get("usage") or {}
  if anth_usage:
      log_histogram("anthropic_api_input_tokens",
                    anth_usage.get("input_tokens", 0),
                    labels={"model": current_model})
      log_histogram("anthropic_api_output_tokens",
                    anth_usage.get("output_tokens", 0),
                    labels={"model": current_model})
      log_histogram("anthropic_api_cache_read_input_tokens",
                    anth_usage.get("cache_read_input_tokens", 0),
                    labels={"model": current_model})
      log_histogram("anthropic_api_cache_creation_input_tokens",
                    anth_usage.get("cache_creation_input_tokens", 0),
                    labels={"model": current_model})
  ```
- All reads use `.get(..., 0)` / `or {}`, so absent/malformed `usage` degrades
  gracefully (AC#3, 324-AC#3).

## Testing

Tests mock the provider call and inspect the sent payload via
`mock_post.call_args[1]["json"]` (existing `test_anthropic_native_tools.py`
harness), and monkeypatch/spy `log_histogram` to capture metrics.

- **Anthropic payload (323 AC#1/#2/#3):**
  - a caching model (e.g. `claude-3-opus-20240229`) → `data["system"]` is a
    list with one `cache_control: {"type":"ephemeral"}` block on the system
    text; with OpenAI-shaped tools, the last converted tool carries
    `cache_control`; total breakpoints ≤ 4.
  - a non-caching model (e.g. `claude-2.1`) → `data["system"]` stays the plain
    string and no `cache_control` appears anywhere (AC#3).
  - a caching model with no tools → system breakpoint only, no crash.
- **`_anthropic_supports_caching`:** True for `claude-3-haiku-20240307`,
  `claude-3-5-sonnet-20241022`, `claude-sonnet-4-…`; False for `claude-2.1`,
  `claude-instant-1.2`, `gpt-4o`, `""`.
- **Metrics (324 AC#1/#2/#3):**
  - OpenAI: a response with `usage.prompt_tokens_details.cached_tokens` →
    `openai_api_cached_tokens` logged with that value; absent details → 0, no
    crash.
  - Anthropic: a response with `usage.cache_read_input_tokens` /
    `cache_creation_input_tokens` → the four `anthropic_api_*` histograms logged;
    absent `usage` → no crash, no logging.
- **Non-Anthropic unchanged:** an OpenAI call's payload/behavior is unaffected
  by Unit A (it only touches `chat_with_anthropic`).

**AC#4 (cache read on 2nd request):** verified structurally by the payload +
metrics tests above; a documented live check (two identical real Anthropic
requests with a >2048-token system prompt, observing
`anthropic_api_cache_creation_input_tokens` on the first and
`anthropic_api_cache_read_input_tokens` on the second) confirms it end-to-end.
Fallback if no cache read appears: add the `anthropic-beta: prompt-caching-…`
header (current API version `2023-06-01` supports `cache_control` GA without
it).

## Out of scope

- Fixing the `_anthropic_tools_payload` anthropic-shape drop bug (task-263).
- Streaming-path cache-usage metrics (final SSE event) — follow-up.
- OpenAI/other-provider explicit cache markers (OpenAI prefix caching is
  automatic; 324's OpenAI metric measures whether byte-stability already earns
  it).
- Any config toggle for enabling/disabling caching.
