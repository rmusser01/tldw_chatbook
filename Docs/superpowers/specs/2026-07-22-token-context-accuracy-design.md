# Token & context accuracy: per-model window, real estimator, config cleanup (TASK-320 / 321 / 325)

**Date**: 2026-07-22
**Status**: Approved design, pending implementation plan
**Base**: origin/dev @ `dc21e3f04` (contains task-322, PR #778)
**Backlog**: TASK-320, TASK-321, TASK-325 (all "To Do", no listed deps)

## Why

The token/window numbers behind every budget and gauge are wrong or stale, and
task-322's Console history trimmer consumes them through the `token_counter`
seam, so its correctness is capped by theirs.

- **No authoritative per-model window (320).** `MODEL_TOKEN_LIMITS`
  (`token_counter.py`) predates current models (no gpt-4o, gpt-4.1, o1/o3,
  claude-3.5/3.7/4, gemini-1.5/2), so `get_model_token_limit` falls back to
  wrong conservative defaults (Anthropic 100k vs real 200k). No `context_window`
  exists in `model_capabilities.py` (only `vision`/`max_images`). The token
  gauge (`chat_token_events.py`) divides usage by the ~2048-token *output*
  budget instead of the model *input* window.
- **Word-count token estimation (321).** When tiktoken is absent,
  `count_tokens_messages` counts whitespace words (`content.split()`);
  `count_tokens_chat_history` uses a chars estimate for non-OpenAI even with
  tiktoken; and `approximate_token_count` (`Chat_Functions.py`) is a dead
  whitespace word count. Word counts under-count real tokens (~1.3–1.5× for
  English, far more for code/CJK), so any budget built on them still overflows.
- **Dead config key (325).** `chat_context_limit = 10` sits under `[rag_search]`,
  is never read anywhere, and silently no-ops for a user who sets it.

This makes the window authoritative and config-overridable, makes token
estimates a real conservative floor everywhere, and removes the dead key —
without changing task-322's code (it sharpens through the shared seam).

## Decisions (user-approved)

1. **Remove `chat_context_limit`** (not wire it). It is superseded by 322's
   model-aware token bound; adding an overlapping message-count cap under the
   wrong config section is confusing. Satisfies 325 AC#1's "or is removed".
2. **Fix the token gauge in place.** The gauge lives in the deprecated-but-still-
   scheduled enhanced-chat path (`chat_token_events.py`); the Console has none.
   Swap its denominator to the model input window (a ~1-line correctness fix on
   still-shipped code) rather than descoping 320 AC#3.

## Architecture

Three well-bounded units plus one deletion. All flow through the existing
`token_counter` seam; `console_history_budget` (322) is untouched and inherits
the improvements.

### Unit A — Per-model context window (320 AC#1/#2)

**`model_capabilities.py`:** add a `context_window` capability, exactly parallel
to `vision`/`max_images`.

- Extend `DEFAULT_MODEL_PATTERNS` entries and `DEFAULT_MODEL_CAPABILITIES` with
  a `context_window` (int) field for current families/models. `context_window`
  is the model **input** window and is distinct from the existing `max_tokens`
  (output cap) key already present on some entries — do not conflate them.
- Add `ModelCapabilities.get_context_window(provider, model) -> int | None`
  (reads `get_model_capabilities(provider, model).get("context_window")`) and a
  module convenience `get_context_window(provider, model)`.
- **Case-insensitive provider lookup (review finding — real latent bug).**
  `DEFAULT_MODEL_PATTERNS` keys providers title-case (`"OpenAI"`,`"Anthropic"`)
  and `get_model_capabilities` matches `provider in self._compiled_patterns`
  case-sensitively, but callers pass mixed/lowercase (`get_model_token_limit`'s
  own defaults use `"openai"`; 322 passes `resolution.provider`). A lowercase
  caller silently misses the pattern block. Fix by resolving the provider
  key case-insensitively (build `self._compiled_patterns` lookups against a
  lower-cased provider, or normalize both sides at match time). This also
  hardens the existing `is_vision_capable` path; a regression test asserts
  vision resolves for both `"openai"` and `"OpenAI"`.

**`token_counter.py`:** rewrite `get_model_token_limit(model, provider)` to
resolve in priority order:
1. `model_capabilities.get_context_window(provider, model)` if not `None`
   (lazy import inside the function to avoid the config circular import, the
   same pattern `ModelCapabilities.__init__` uses).
2. Exact match in a **refreshed** `MODEL_TOKEN_LIMITS`.
3. **Longest-prefix** match in `MODEL_TOKEN_LIMITS` (review finding: the current
   arbitrary-order `startswith` lets `"gpt-4"` (8192) shadow `"gpt-4o"`; iterate
   candidates and pick the longest matching prefix so specific entries win).
4. Provider default, else `MODEL_TOKEN_LIMITS["default"]`.

Refreshed `MODEL_TOKEN_LIMITS` (current published **input** windows; concrete
values, verify at implementation): gpt-4o/gpt-4o-mini 128000, gpt-4-turbo
128000, gpt-4.1* 1047576, gpt-4 8192, gpt-3.5-turbo 16385, o1 200000, o1-mini
128000, o3/o3-mini/o4-mini 200000; claude-3-opus/haiku/sonnet 200000,
claude-3-5-* 200000, claude-3-7-* 200000, claude-opus-4-*/sonnet-4-* 200000,
claude-2.1 200000, claude-2 100000; gemini-1.5-pro 2097152, gemini-1.5-flash
1048576, gemini-2.* 1048576, gemini-pro 30720; mistral-large 128000,
mistral-small/medium 32000, mixtral-8x7b 32000; `"default"` 8192. Provider
defaults lean to the provider's common current window but stay a conservative
floor (anthropic 200000, google 1048576, openai 128000, mistral 32000).
**Safety note:** for the 322 budget, under-estimating the window is safe (it
trims more, never 400s); over-estimating an unknown model is the only risk, so
truly-unknown models keep the conservative `"default"`.

### Unit B — One consistent estimator, no `.split()` (321 AC#1/#2/#3)

**`token_counter.py`:** add `estimate_tokens(text, model, provider) -> int`, the
single text-token estimator, tiered:

1. **Custom tokenizer** — `count_tokens_with_custom(text, model, provider)`,
   only behind a cheap availability gate. The `CustomTokenizerManager` singleton
   (`get_tokenizer_manager()`) logs metrics on every `count_tokens` call even
   when it will return `None`, so calling it per trim iteration adds overhead to
   322's hot path (review finding). Add `CustomTokenizerManager.has_tokenizers()
   -> bool` returning `bool(self._model_mappings)` (no metrics, no I/O) and a
   module `custom_tokenizers_available() -> bool` (`CUSTOM_TOKENIZERS_AVAILABLE
   and get_tokenizer_manager().has_tokenizers()`). Enter the custom tier only
   when that is true — the default install (no `mappings.json`) pays nothing.
2. **tiktoken** — `count_tokens_tiktoken(text, model)` when `TIKTOKEN_AVAILABLE`.
3. **Chars-based conservative estimate** (always available), replacing every
   `.split()`:
   ```
   estimate = int((non_cjk_chars * base_ratio + cjk_chars * CJK_TOKENS_PER_CHAR)
                   * ESTIMATE_HEADROOM)
   ```
   where `base_ratio = TOKENS_PER_CHAR_ESTIMATES.get(provider, 0.25)`,
   `CJK_TOKENS_PER_CHAR = 1.0` (each CJK code point is ≥ ~1 token — a floor),
   `ESTIMATE_HEADROOM = 1.2` (documented headroom so English/code lean high, the
   safe direction). CJK detection covers Hiragana/Katakana (U+3040–30FF), CJK
   Unified + Ext-A (U+3400–4DBF, U+4E00–9FFF), Hangul (U+AC00–D7AF), CJK compat
   (U+F900–FAFF), and fullwidth/CJK punctuation (U+FF00–FFEF).

**Delegation (single chain, no double-counting).**
- `estimate_tokens(text, model, provider: str = "")` — the one estimator.
  `provider` is optional and only selects the chars-path `base_ratio`
  (`""`/unknown → 0.25); the CJK weighting, tiktoken tier, and custom-gate are
  provider-independent (the custom tier resolves by `model`+`provider` when
  available).
- `count_tokens_messages(messages, model, provider: str = "")` — gains an
  optional trailing `provider`; sums `estimate_tokens` over role/content/name
  and keeps its per-message framing overhead (`tokens_per_message`,
  `base_tokens`) unchanged. The new param is trailing and optional, so 322's
  existing `count_tokens_messages(flattened, model)` call is unchanged (it
  resolves `provider=""` → generic ratio, which is fine — CJK weighting, the
  dominant factor, is provider-independent).
- `count_tokens_chat_history(history, model, provider)` — converts its
  tuple/dict history to messages and returns
  `count_tokens_messages(messages, model, provider)`. Its bespoke branches (the
  message-level `count_messages_with_custom` tier, the OpenAI-only tiktoken
  gate, and the divergent chars branch) are **removed** — all custom/tiktoken/
  chars logic now lives once, inside `estimate_tokens`, so there is exactly one
  estimator and no double custom-counting.
- Delete the dead `approximate_token_count` (`Chat_Functions.py`; zero callers
  dev-wide).

`console_history_budget` (322) is untouched: its `count_tokens_messages(
flattened, model)` call keeps working and now routes through `estimate_tokens`.

**Documented behavior shift:** non-OpenAI users *with* tiktoken installed now
get tiktoken-based counts (higher, more accurate) instead of the old chars
estimate; without tiktoken they get the improved chars floor instead of word
counts. Both directions lean higher (safer for budgets).

### Unit C — Gauge ceiling fix + config cleanup (320 AC#3, 325)

- **`chat_token_events.py`:** set the gauge denominator to the model input
  window (`total_limit`, already returned by `estimate_remaining_tokens`)
  instead of `max_tokens_response`, preserving the existing
  `#chat-custom-token-limit` manual override that follows. Remove the stale
  "measures against configured limit" comment.
- **`config.py`:** remove `chat_context_limit` from `DEFAULT_RAG_SEARCH_CONFIG`
  and from the sample TOML under `[rag_search]`. Verified zero references
  dev-wide outside `config.py`, so no consumer breaks; the loader merges over
  defaults, so a leftover key in an existing user file is a harmless no-op
  (no migration needed).

## Testing

- **Window lookup (320 AC#4):** `get_model_token_limit` returns the correct
  current window for ≥1 current model per major provider (gpt-4o→128000,
  claude-3-5-sonnet→200000, gemini-1.5-pro→2097152, mistral-large→128000);
  a `[model_capabilities]`-configured `context_window` pattern overrides the
  table; longest-prefix beats a shorter shadowing prefix (`gpt-4o` ≠ `gpt-4`);
  truly-unknown model → conservative default.
- **Case-insensitivity (Unit A):** `is_vision_capable` and `get_context_window`
  resolve identically for `"openai"` and `"OpenAI"`.
- **Estimator floors (321 AC#3), property-based (tiktoken absent, so no
  reference tokenizer):** a pure-CJK string estimates `>= len(text)` (CJK floor);
  a code sample estimates strictly greater than its whitespace word count
  (`> len(sample.split())`, proving it is not the old `.split()` and leans
  high); an empty string → 0; `count_tokens_messages` with the chars path
  returns a value strictly greater than the summed word counts of its contents.
- **Consistency:** for a single `{role, content}` message,
  `count_tokens_chat_history([...], model, provider)` equals
  `count_tokens_messages([...], model, provider)` (chat_history now delegates),
  and both are `> 0` for non-empty content with tiktoken absent (chars path).
- **Gauge (320 AC#3):** the display limit passed to `format_token_display`
  equals `get_model_token_limit(model, provider)` (not `max_tokens_response`)
  when no custom-limit widget value is set; a custom value still overrides.
- **Removal (325 AC#2):** a test asserts `chat_context_limit` is absent from the
  loaded default config and the generated sample TOML.

## Out of scope

- task-322 trimmer logic (unchanged; it only benefits).
- Per-provider `per_image_tokens` calibration (322's flat estimate stays).
- A Console token gauge (Console has none today; not added here).
- Downloading/bundling real provider tokenizer files (the custom tier stays
  opt-in via user-installed `~/.config/tldw_cli/tokenizers/mappings.json`).
- Migrating existing user configs that still contain `chat_context_limit`
  (harmless unknown key).
