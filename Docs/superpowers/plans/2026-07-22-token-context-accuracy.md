# Token & Context Accuracy Implementation Plan (TASK-320 / 321 / 325)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the model context window authoritative/config-overridable, replace all `.split()` token estimates with one consistent estimator, fix the token gauge's denominator, and remove the dead `chat_context_limit` config key — all through the existing `token_counter` seam so task-322's trimmer sharpens with no change to its code.

**Architecture:** `model_capabilities.py` gains a `context_window` capability (pattern + direct, case-insensitive provider lookup); `token_counter.get_model_token_limit` consults it first then a refreshed table; a single `token_counter.estimate_tokens` (custom-gated → tiktoken → CJK-weighted chars floor) replaces every word-count path and both `count_tokens_*` functions delegate to it; the enhanced-chat gauge divides by the input window; `chat_context_limit` is deleted.

**Tech Stack:** Python ≥3.11, pytest. tiktoken and the HF `tokenizers` lib are **absent** in the venv (estimator tests target the chars path).

## Global Constraints

- **Worktree:** all work happens in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-token-accuracy` (branch `feat/token-context-accuracy`, off `origin/dev @ dc21e3f04`, which contains task-322). Never touch the main checkout.
- **Test command (venv lives in the main checkout):**
  `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-token-accuracy && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <path> -v`
- **Do NOT modify** `tldw_chatbook/Chat/console_history_budget.py` (task-322); it inherits improvements through `count_tokens_messages`/`get_model_token_limit`.
- **Preserve these exact window values** (pinned by `Tests/Chat/test_token_counter.py`): `gpt-4`→8192, `gpt-4-32k`→32768, `gpt-4-turbo`→128000, `claude-3-opus-20240229`→200000, `claude-2`→100000, `"default"`→4096, unknown-openai-model→4096. The **only** intended value change: `gpt-3.5-turbo` 4096→16385.
- **Fallback windows stay conservative.** Over-estimating a window is the only way to 400 the 322 budget; only bump the `anthropic` provider default (100000→200000). Keep `openai` 4096, `google` 30720, `mistral` 32000, `"default"` 4096.
- **Capability patterns must be anchored and agree with the table** (a `context_window` a pattern returns must equal any table entry the same model would hit).
- **Estimator constants:** `base_ratio` = `TOKENS_PER_CHAR_ESTIMATES.get(provider, default 0.25)`, `CJK_TOKENS_PER_CHAR = 1.0`, `ESTIMATE_HEADROOM = 1.2`. A 100-ASCII-char string must estimate in `(25, 50)` (keeps `test_character_estimation_fallback` green). Never use `.split()` for a token estimate.
- **Signatures:** `estimate_tokens(text, model="gpt-3.5-turbo", provider="") -> int`; `count_tokens_messages(messages, model="gpt-3.5-turbo", provider="") -> int` (new trailing optional `provider`, so 322's `count_tokens_messages(flattened, model)` call is unchanged); `get_context_window(provider, model) -> int | None`.

---

## Task 1: `context_window` capability + case-insensitive provider lookup

**Files:**
- Modify: `tldw_chatbook/model_capabilities.py`
- Test: `Tests/test_model_capabilities.py`

**Interfaces:**
- Produces: `ModelCapabilities.get_context_window(self, provider: str, model: str) -> int | None`; module `get_context_window(provider: str, model: str) -> int | None`. Direct mappings and the OpenAI/Anthropic family patterns carry a `context_window` int. Provider matching is case-insensitive.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/test_model_capabilities.py`:

```python
from tldw_chatbook.model_capabilities import ModelCapabilities, get_context_window


def test_get_context_window_direct_mapping():
    caps = ModelCapabilities({})  # uses DEFAULT_MODEL_* (no config file)
    assert caps.get_context_window("OpenAI", "gpt-4o") == 128000
    assert caps.get_context_window("Anthropic", "claude-3-opus-20240229") == 200000
    assert caps.get_context_window("Google", "gemini-1.5-pro") == 2097152


def test_get_context_window_via_family_pattern():
    caps = ModelCapabilities({})
    # A novel dated variant not in direct mappings resolves via the anchored pattern.
    assert caps.get_context_window("OpenAI", "gpt-4o-2099-12-31") == 128000
    assert caps.get_context_window("Anthropic", "claude-3-5-haiku-20991231") == 200000


def test_get_context_window_unknown_is_none():
    caps = ModelCapabilities({})
    assert caps.get_context_window("OpenAI", "totally-unknown-model") is None
    # Generic gpt-4 variant must NOT match a pattern (so the table's gpt-4 wins later).
    assert caps.get_context_window("OpenAI", "gpt-4-some-variant") is None


def test_provider_lookup_is_case_insensitive():
    caps = ModelCapabilities({})
    assert caps.is_vision_capable("openai", "gpt-4o") is True
    assert caps.is_vision_capable("OpenAI", "gpt-4o") is True
    assert caps.get_context_window("anthropic", "claude-3-opus-20240229") == \
        caps.get_context_window("Anthropic", "claude-3-opus-20240229") == 200000
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-token-accuracy && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_model_capabilities.py -v -k "context_window or case_insensitive"`
Expected: FAIL (`AttributeError: 'ModelCapabilities' object has no attribute 'get_context_window'`).

- [ ] **Step 3: Add `context_window` to `DEFAULT_MODEL_CAPABILITIES`**

In `tldw_chatbook/model_capabilities.py`, add a `"context_window"` key to these existing entries (leave all other keys/entries as-is):

```python
    # OpenAI
    "gpt-4-vision-preview": {"vision": True, "max_images": 1, "context_window": 128000},
    "gpt-4-turbo": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-4-turbo-2024-04-09": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-4o": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-4o-mini": {"vision": True, "max_images": 10, "context_window": 128000},
    "gpt-4.1-2025-04-14": {"vision": True, "max_images": 10, "context_window": 1047576},
    "o4-mini-2025-04-16": {"vision": True, "max_images": 10, "context_window": 200000},
    "o3-2025-04-16": {"vision": True, "max_images": 10, "context_window": 200000},
    "o3-mini-2025-01-31": {"vision": True, "max_images": 10, "context_window": 200000},
    "gpt-4.1-mini-2025-04-14": {"vision": True, "max_images": 10, "context_window": 1047576},
    "gpt-4.1-nano-2025-04-14": {"vision": True, "max_images": 10, "context_window": 1047576},
    # Anthropic
    "claude-3-opus-20240229": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-sonnet-20240229": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-haiku-20240307": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-5-sonnet-20240620": {"vision": True, "max_images": 5, "context_window": 200000},
    "claude-3-5-sonnet-20241022": {"vision": True, "max_images": 5, "context_window": 200000},
    # Google
    "gemini-pro-vision": {"vision": True, "max_images": 1, "context_window": 12288},
    "gemini-1.5-pro": {"vision": True, "max_images": 10, "context_window": 2097152},
    "gemini-1.5-flash": {"vision": True, "max_images": 10, "context_window": 1048576},
    "gemini-2.0-flash": {"vision": True, "max_images": 10, "context_window": 1048576},
```

- [ ] **Step 4: Add `context_window` to OpenAI + Anthropic family patterns**

In `DEFAULT_MODEL_PATTERNS`, add `"context_window"` to these entries (keep `vision` and every other pattern unchanged; do NOT add `context_window` to Google patterns — their `(pro|flash)` pattern spans non-uniform windows, so Google windows resolve via direct mappings + table):

```python
    "OpenAI": [
        {"pattern": r"^gpt-4.*vision", "vision": True, "context_window": 128000},
        {"pattern": r"^gpt-4[o0](?:-mini)?", "vision": True, "context_window": 128000},
        {"pattern": r"^gpt-4.*turbo", "vision": True, "context_window": 128000},
        {"pattern": r"^gpt-4\.1", "vision": True, "context_window": 1047576},
        {"pattern": r"^o[34](?:-mini)?", "vision": True, "context_window": 200000},
        {"pattern": r"^dall-e", "vision": True, "image_generation": True},
    ],
    "Anthropic": [
        {"pattern": r"^claude-3", "vision": True, "context_window": 200000},
        {"pattern": r"^claude.*opus-4", "vision": True, "context_window": 200000},
        {"pattern": r"^claude.*sonnet-4", "vision": True, "context_window": 200000},
    ],
```

- [ ] **Step 5: Make the provider pattern lookup case-insensitive**

In `ModelCapabilities.__init__`, after `self._compiled_patterns = self._compile_patterns()` (currently line ~166), add a lower-cased index:

```python
        self._compiled_patterns = self._compile_patterns()
        # Case-insensitive provider index: callers pass mixed/lowercase provider
        # names ("openai") while pattern keys are title-case ("OpenAI").
        self._provider_key_by_lower = {
            provider.lower(): provider for provider in self._compiled_patterns
        }
```

In `get_model_capabilities`, replace the pattern-block branch (currently `elif provider in self._compiled_patterns:` ... loop) with a case-insensitive resolve:

```python
        # 2. Check provider-specific patterns (case-insensitive provider match)
        else:
            provider_key = self._provider_key_by_lower.get((provider or "").lower())
            if provider_key is not None:
                for pattern, pattern_capabilities in self._compiled_patterns[provider_key]:
                    if pattern.match(model):
                        capabilities = pattern_capabilities.copy()
                        logger.debug(
                            f"Pattern matched for {provider}/{model}: {capabilities}"
                        )
                        break
```

- [ ] **Step 6: Add `get_context_window` (method + module function)**

Add the method to `ModelCapabilities` (e.g. after `is_vision_capable`):

```python
    def get_context_window(self, provider: str, model: str) -> Optional[int]:
        """Return the model's input context window, or None if unknown."""
        return self.get_model_capabilities(provider, model).get("context_window")
```

Add the module convenience function near `is_vision_capable` (module level):

```python
def get_context_window(provider: str, model: str) -> Optional[int]:
    """Convenience function to resolve a model's input context window."""
    return get_model_capabilities().get_context_window(provider, model)
```

- [ ] **Step 7: Run the tests to verify pass**

Run: `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-token-accuracy && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/test_model_capabilities.py -v`
Expected: PASS (new tests + all pre-existing capability tests still green).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/model_capabilities.py Tests/test_model_capabilities.py
git commit -m "feat(capabilities): per-model context_window + case-insensitive provider lookup (TASK-320)"
```

---

## Task 2: refreshed table + capabilities-first `get_model_token_limit`

**Files:**
- Modify: `tldw_chatbook/Utils/token_counter.py` (`MODEL_TOKEN_LIMITS`, `get_model_token_limit`)
- Test: `Tests/Chat/test_token_counter.py`

**Interfaces:**
- Consumes: `model_capabilities.get_context_window(provider, model)` (Task 1).
- Produces: `get_model_token_limit(model, provider="openai") -> int` resolving capabilities → exact table → longest-prefix table → provider default.

- [ ] **Step 1: Update/extend the failing tests**

In `Tests/Chat/test_token_counter.py`, change the one pinned value and add current-model + capabilities-override + longest-prefix tests:

```python
    def test_estimate_remaining_tokens(self):
        """Test estimating remaining tokens"""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        used, limit, remaining = estimate_remaining_tokens(
            history, model="gpt-3.5-turbo", max_tokens_response=1000
        )
        assert used > 0
        assert limit == 16385  # gpt-3.5-turbo refreshed input window
        assert remaining < limit - 1000
```

```python
    def test_get_model_token_limit_current_models(self):
        assert get_model_token_limit("gpt-4o", "openai") == 128000
        assert get_model_token_limit("claude-3-5-sonnet-20241022", "anthropic") == 200000
        assert get_model_token_limit("gemini-1.5-pro", "google") == 2097152
        assert get_model_token_limit("mistral-large", "mistral") == 128000

    def test_get_model_token_limit_prefers_capabilities_over_table(self):
        # A novel dated gpt-4o variant is not a table key; capabilities resolves it.
        assert get_model_token_limit("gpt-4o-2099-12-31", "openai") == 128000

    def test_get_model_token_limit_longest_prefix_wins(self):
        # "gpt-4" (8192) must not shadow a more specific match; a bare gpt-4 variant
        # with no capability pattern falls to the gpt-4 table prefix.
        assert get_model_token_limit("gpt-4-some-variant", "openai") == 8192

    def test_get_model_token_limit_anthropic_default_bumped(self):
        # Unknown modern Claude falls back to the 200k floor, not the stale 100k.
        assert get_model_token_limit("claude-99-future", "anthropic") == 200000
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Chat/test_token_counter.py -v -k "current_models or capabilities or longest_prefix or anthropic_default or estimate_remaining"`
Expected: FAIL (`test_estimate_remaining_tokens` asserts 16385 vs current 4096; new tests fail on stale values).

- [ ] **Step 3: Refresh `MODEL_TOKEN_LIMITS`**

Replace the `MODEL_TOKEN_LIMITS` dict body in `token_counter.py` with (keys stay full-length; preserved values unchanged except gpt-3.5-turbo):

```python
MODEL_TOKEN_LIMITS = {
    # OpenAI
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4.1": 1047576,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16384,
    "o1": 200000,
    "o1-mini": 128000,
    "o3": 200000,
    "o3-mini": 200000,
    "o4-mini": 200000,
    # Anthropic
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-sonnet-20241022": 200000,
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-instant-1.2": 100000,
    # Google
    "gemini-1.5-pro": 2097152,
    "gemini-1.5-flash": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-pro": 30720,
    "gemini-pro-vision": 12288,
    # Others
    "mistral-large": 128000,
    "mistral-medium": 32000,
    "mistral-small": 32000,
    "mixtral-8x7b": 32000,
    # Default for unknown models
    "default": 4096,
}
```

- [ ] **Step 4: Rewrite `get_model_token_limit`**

Replace the whole function body:

```python
def get_model_token_limit(model: str, provider: str = "openai") -> int:
    """
    Get the input context-window token limit for a specific model.

    Resolves in priority order: the per-model capability `context_window`
    (config-overridable), an exact table entry, the longest matching table
    prefix, then a conservative provider default. Fallbacks lean conservative
    on purpose: under-estimating the window degrades gracefully (more trimming),
    while over-estimating is the only way to overflow the model on dispatch.
    """
    # 1. Per-model capability context window (authoritative, config-overridable).
    try:
        from tldw_chatbook.model_capabilities import get_context_window

        window = get_context_window(provider, model)
        if window is not None:
            return window
    except Exception as e:  # never let capability resolution break token limits
        logger.debug(f"context_window lookup failed for {provider}/{model}: {e}")

    # 2. Exact table match.
    if model in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model]

    # 3. Longest matching table prefix (so "gpt-4" can't shadow "gpt-4-turbo").
    best_limit = None
    best_len = -1
    for model_prefix, limit in MODEL_TOKEN_LIMITS.items():
        if model_prefix == "default":
            continue
        if model.startswith(model_prefix) and len(model_prefix) > best_len:
            best_limit = limit
            best_len = len(model_prefix)
    if best_limit is not None:
        return best_limit

    # 4. Conservative provider default.
    provider_defaults = {
        "anthropic": 200000,  # every modern Claude is >= 200k; safe floor
        "google": 30720,
        "openai": 4096,
        "mistral": 32000,
    }
    return provider_defaults.get(provider, MODEL_TOKEN_LIMITS["default"])
```

- [ ] **Step 5: Run the pinned + new tests to verify pass**

Run: `... -m pytest Tests/Chat/test_token_counter.py -v`
Expected: PASS — including the pinned `test_get_model_token_limit_known_model`, `..._unknown_model` (4096), `..._by_prefix` (gpt-4-some-variant 8192, claude-3-opus-custom 4096), and the new tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Utils/token_counter.py Tests/Chat/test_token_counter.py
git commit -m "feat(tokens): capabilities-first get_model_token_limit + refreshed window table (TASK-320)"
```

---

## Task 3: one `estimate_tokens`, delegate both counters, delete dead word-count

**Files:**
- Modify: `tldw_chatbook/Utils/custom_tokenizers.py` (add `has_tokenizers` + `custom_tokenizers_available`)
- Modify: `tldw_chatbook/Utils/token_counter.py` (`estimate_tokens`, CJK helpers, `count_tokens_messages`, `count_tokens_chat_history`, import)
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (delete `approximate_token_count`)
- Test: `Tests/Chat/test_token_counter.py`

**Interfaces:**
- Produces: `estimate_tokens(text, model="gpt-3.5-turbo", provider="") -> int`; `count_tokens_messages(messages, model="gpt-3.5-turbo", provider="") -> int`; `CustomTokenizerManager.has_tokenizers() -> bool`; `custom_tokenizers_available() -> bool`.

- [ ] **Step 1: Write the failing estimator tests**

Add to `Tests/Chat/test_token_counter.py`:

```python
from tldw_chatbook.Utils.token_counter import estimate_tokens, count_tokens_messages


class TestEstimator:
    def test_empty_text_is_zero(self):
        assert estimate_tokens("", "gpt-4o", "openai") == 0

    def test_cjk_floor_at_least_one_token_per_char(self):
        # CJK code points are >= ~1 token each; a conservative floor never under-counts.
        cjk = "你好世界" * 10  # 40 CJK chars
        assert estimate_tokens(cjk, "gemini-1.5-pro", "google") >= len(cjk)

    def test_code_sample_exceeds_word_count(self):
        code = "def f(x):\n    return [i*i for i in range(x) if i % 2 == 0]\n" * 3
        assert estimate_tokens(code, "claude-3-5-sonnet-20241022", "anthropic") > len(code.split())

    def test_ascii_100_chars_in_band(self):
        # Keeps the pinned test_character_estimation_fallback assumptions valid.
        assert 25 < estimate_tokens("A" * 100, "unknown", "unknown") < 50

    def test_messages_and_chat_history_agree_for_one_message(self):
        msg = [{"role": "user", "content": "hello world foo bar"}]
        assert count_tokens_chat_history(msg, model="claude-3-5-sonnet-20241022",
                                         provider="anthropic") == \
            count_tokens_messages(msg, "claude-3-5-sonnet-20241022", "anthropic")
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Chat/test_token_counter.py::TestEstimator -v`
Expected: FAIL (`cannot import name 'estimate_tokens'`).

- [ ] **Step 3: Add the availability gate to `custom_tokenizers.py`**

Add a method to `CustomTokenizerManager` (near `count_tokens`):

```python
    def has_tokenizers(self) -> bool:
        """Cheap availability check (no metrics, no I/O) — are any mappings loaded?"""
        return bool(self._model_mappings)
```

Add a module function next to `count_tokens_with_custom`:

```python
def custom_tokenizers_available() -> bool:
    """True only when a custom tokenizer could actually resolve (mappings loaded)."""
    return get_tokenizer_manager().has_tokenizers()
```

- [ ] **Step 4: Extend the token_counter import to include the gate**

In `token_counter.py`, replace the custom-tokenizers import block:

```python
try:
    from .custom_tokenizers import (
        count_tokens_with_custom,
        count_messages_with_custom,
        custom_tokenizers_available,
    )

    CUSTOM_TOKENIZERS_AVAILABLE = True
except ImportError:
    CUSTOM_TOKENIZERS_AVAILABLE = False
    count_tokens_with_custom = None
    count_messages_with_custom = None
    custom_tokenizers_available = None
```

- [ ] **Step 5: Add the CJK constants + helpers + `estimate_tokens`**

In `token_counter.py`, after `TOKENS_PER_CHAR_ESTIMATES` add:

```python
# Conservative chars-based estimate constants (used when no tokenizer is available).
CJK_TOKENS_PER_CHAR = 1.0   # each CJK code point is >= ~1 token
ESTIMATE_HEADROOM = 1.2     # documented headroom so estimates lean high (safe)

_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ext-A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Fullwidth / halfwidth (CJK punctuation)
)


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _chars_estimate(text: str, provider: str) -> int:
    """Conservative chars-based token floor; weights CJK higher, applies headroom."""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    base_ratio = TOKENS_PER_CHAR_ESTIMATES.get(
        provider or "default", TOKENS_PER_CHAR_ESTIMATES["default"]
    )
    return int((other * base_ratio + cjk * CJK_TOKENS_PER_CHAR) * ESTIMATE_HEADROOM)


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo", provider: str = "") -> int:
    """Single text token estimator: custom tokenizer (gated) -> tiktoken -> chars floor.

    Never uses whitespace word counts. `provider` only selects the chars-path
    ratio; CJK weighting and the tiktoken/custom tiers are provider-independent.
    """
    if not text:
        return 0
    if CUSTOM_TOKENIZERS_AVAILABLE and custom_tokenizers_available():
        custom = count_tokens_with_custom(text, model, provider)
        if custom is not None:
            return custom
    if TIKTOKEN_AVAILABLE:
        return count_tokens_tiktoken(text, model)
    return _chars_estimate(text, provider)
```

- [ ] **Step 6: Delegate `count_tokens_messages` and `count_tokens_chat_history`**

Replace `count_tokens_messages` (keep the per-message framing overhead; swap text counting to `estimate_tokens`; add the optional trailing `provider`):

```python
def count_tokens_messages(
    messages: List[Dict[str, Any]], model: str = "gpt-3.5-turbo", provider: str = ""
) -> int:
    """Count tokens for OpenAI-format messages (framing overhead + estimate_tokens)."""
    if not messages:
        return 0

    if model.startswith("gpt-3.5") or model.startswith("gpt-4"):
        tokens_per_message = 3
        tokens_per_name = 1
        base_tokens = 3
    else:
        tokens_per_message = 2
        tokens_per_name = 1
        base_tokens = 2

    total_tokens = base_tokens
    for message in messages:
        total_tokens += tokens_per_message
        role = message.get("role", "")
        if role:
            total_tokens += estimate_tokens(role, model, provider)
        content = message.get("content", "")
        if content:
            total_tokens += estimate_tokens(content, model, provider)
        name = message.get("name", "")
        if name:
            total_tokens += tokens_per_name
            total_tokens += estimate_tokens(name, model, provider)
    return total_tokens
```

Also unify the `system_prompt` branch of `estimate_remaining_tokens` (so every count feeding that budget uses the one estimator). Replace its `if system_prompt:` block:

```python
    # Add system prompt if present
    if system_prompt:
        current_tokens += estimate_tokens(system_prompt, model, provider)
```

Replace `count_tokens_chat_history` so it converts formats and delegates (drop its custom/tiktoken/chars branches):

```python
def count_tokens_chat_history(
    history: List[Union[Tuple[Optional[str], Optional[str]], Dict[str, Any]]],
    model: str = "gpt-3.5-turbo",
    provider: str = "openai",
) -> int:
    """Count tokens in chat-history format (tuples or message dicts) via the one estimator."""
    if not history:
        return 0

    messages: List[Dict[str, Any]] = []
    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        elif isinstance(item, dict) and "role" in item and "content" in item:
            messages.append(item)
        else:
            logger.warning(f"Unknown history format: {type(item)}")

    return count_tokens_messages(messages, model, provider)
```

- [ ] **Step 7: Delete the dead `approximate_token_count`**

In `tldw_chatbook/Chat/Chat_Functions.py`, delete the entire `approximate_token_count` function (the `def approximate_token_count(history): ...` block, ~lines 100-113). It has zero callers.

- [ ] **Step 8: Run estimator + existing token tests + the 322 suite**

Run: `... -m pytest Tests/Chat/test_token_counter.py Tests/Chat/test_console_history_budget.py -v`
Expected: PASS — `TestEstimator`, the pre-existing `TestTokenCounter` (incl. `test_character_estimation_fallback` in (25,50) and `test_provider_specific_ratios`), and all task-322 tests (322's real-counter test asserts a *relative* image cost, so it stays green).

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/Utils/custom_tokenizers.py tldw_chatbook/Utils/token_counter.py tldw_chatbook/Chat/Chat_Functions.py Tests/Chat/test_token_counter.py
git commit -m "feat(tokens): one estimate_tokens (custom/tiktoken/CJK-chars), delegate counters, drop dead word-count (TASK-321)"
```

---

## Task 4: route the Console draft estimate through `estimate_tokens`

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_estimate_tokens`, import at line ~234)
- Test: `Tests/Chat/test_chat_screen_token_estimate.py` (new)

**Interfaces:**
- Consumes: `estimate_tokens` (Task 3).

- [ ] **Step 1: Write the failing test**

Create `Tests/Chat/test_chat_screen_token_estimate.py`:

```python
from types import SimpleNamespace

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Utils.token_counter import estimate_tokens


def test_estimate_tokens_uses_shared_estimator_no_split():
    # Call the unbound method with a minimal self; it must delegate to estimate_tokens,
    # never a .split() word count.
    self_stub = SimpleNamespace()
    text = "def f(x): return x*x  # a code-ish draft with symbols"
    result = ChatScreen._estimate_tokens(self_stub, {"draft": text})
    assert result == estimate_tokens(text, "", "")
    # A word-count would be far lower than the chars-based estimate.
    assert result != len(text.split())


def test_estimate_tokens_none_for_empty_draft():
    self_stub = SimpleNamespace()
    assert ChatScreen._estimate_tokens(self_stub, {"draft": ""}) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Chat/test_chat_screen_token_estimate.py -v`
Expected: FAIL (`result` uses `count_tokens_tiktoken`/`.split()`, not `estimate_tokens`).

- [ ] **Step 3: Update the import and `_estimate_tokens`**

In `tldw_chatbook/UI/Screens/chat_screen.py`, change the import at line ~234 from:

```python
from ...Utils.token_counter import count_tokens_tiktoken
```

to:

```python
from ...Utils.token_counter import estimate_tokens
```

Replace the `_estimate_tokens` method body (currently uses `count_tokens_tiktoken(text)` with a `.split()*1.3` except-fallback):

```python
    def _estimate_tokens(self, payload: dict[str, Any]) -> int | None:
        """Return a token estimate for the current draft text."""
        text = payload.get("draft", "")
        if not text:
            return None
        return estimate_tokens(text, "", "")
```

`count_tokens_tiktoken` has exactly one use in this file (inside `_estimate_tokens`), so replacing the import is safe; if a grep shows another use, keep both imports.

- [ ] **Step 4: Run to verify pass**

Run: `... -m pytest Tests/Chat/test_chat_screen_token_estimate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_chat_screen_token_estimate.py
git commit -m "feat(console): route draft token estimate through estimate_tokens (TASK-321)"
```

---

## Task 5: gauge input-window denominator + remove `chat_context_limit`

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py` (add helper; both fix sites, lines ~155 & ~281)
- Modify: `tldw_chatbook/config.py` (remove `chat_context_limit` at lines 84 & 3034)
- Test: `Tests/Chat/test_token_display_limit.py` (new), `Tests/Chat/test_config_no_chat_context_limit.py` (new)

**Interfaces:**
- Produces: `chat_token_events._resolve_token_display_limit(total_limit: int, custom_limit: int) -> int`.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_token_display_limit.py`:

```python
from tldw_chatbook.Event_Handlers.Chat_Events.chat_token_events import (
    _resolve_token_display_limit,
)


def test_display_limit_is_input_window_by_default():
    # AC#3: the gauge denominator is the model input window, not the output budget.
    assert _resolve_token_display_limit(total_limit=128000, custom_limit=0) == 128000


def test_display_limit_honors_custom_override():
    assert _resolve_token_display_limit(total_limit=128000, custom_limit=5000) == 5000


def test_display_limit_ignores_nonpositive_custom():
    assert _resolve_token_display_limit(total_limit=200000, custom_limit=0) == 200000
    assert _resolve_token_display_limit(total_limit=200000, custom_limit=-3) == 200000
```

Create `Tests/Chat/test_config_no_chat_context_limit.py`:

```python
from pathlib import Path

import tldw_chatbook.config as config_mod
from tldw_chatbook.config import DEFAULT_RAG_SEARCH_CONFIG


def test_chat_context_limit_removed_from_defaults():
    assert "chat_context_limit" not in DEFAULT_RAG_SEARCH_CONFIG


def test_chat_context_limit_absent_from_config_source():
    # 325 AC#2: no references remain (dict default + sample TOML both gone).
    source = Path(config_mod.__file__).read_text(encoding="utf-8")
    assert "chat_context_limit" not in source
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Chat/test_token_display_limit.py Tests/Chat/test_config_no_chat_context_limit.py -v`
Expected: FAIL (`cannot import name '_resolve_token_display_limit'`; `chat_context_limit` still present).

- [ ] **Step 3: Add the display-limit helper**

In `tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py`, add a module-level helper (near the top, after imports):

```python
def _resolve_token_display_limit(total_limit: int, custom_limit: int) -> int:
    """Gauge denominator: the model input window, unless a positive custom limit is set.

    task-320 AC#3: usage is measured against the model *input* context window,
    not the ~2048-token output-response budget.
    """
    return custom_limit if custom_limit > 0 else total_limit
```

- [ ] **Step 4: Use the helper at both gauge sites**

At **both** occurrences (lines ~153-162 and ~279-288), replace this block:

```python
        # Use max_tokens_response as the display limit instead of model's total limit
        # This allows users to see how their conversation measures against their configured limit
        display_limit = max_tokens_response

        # Check if there's a custom token limit setting (we'll add this later)
        try:
            custom_limit_widget = app.query_one("#chat-custom-token-limit", Input)
            custom_limit = int(custom_limit_widget.value or "0")
            if custom_limit > 0:
                display_limit = custom_limit
        except (QueryError, ValueError):
            # No custom limit widget or invalid value, use max_tokens_response
            pass
```

with:

```python
        # task-320 AC#3: measure usage against the model input window (total_limit),
        # not the output-response budget. A positive custom-limit widget still wins.
        try:
            custom_limit_widget = app.query_one("#chat-custom-token-limit", Input)
            custom_limit = int(custom_limit_widget.value or "0")
        except (QueryError, ValueError):
            custom_limit = 0
        display_limit = _resolve_token_display_limit(total_limit, custom_limit)
```

(`total_limit` is already unpacked from `_estimate_tokens_cached(...)` above each site.)

- [ ] **Step 5: Remove `chat_context_limit` from config**

In `tldw_chatbook/config.py`, delete the line `"chat_context_limit": 10,` inside `DEFAULT_RAG_SEARCH_CONFIG` (line 84), and delete the line `chat_context_limit = 10` in the sample TOML under `[rag_search]` (line 3034). After this, `grep -c chat_context_limit tldw_chatbook/config.py` returns 0.

- [ ] **Step 6: Run to verify pass**

Run: `... -m pytest Tests/Chat/test_token_display_limit.py Tests/Chat/test_config_no_chat_context_limit.py Tests/Chat/test_footer_token_dirty_gate.py -v`
Expected: PASS (new tests green; the footer dirty-gate suite — caching behavior — still green).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py tldw_chatbook/config.py Tests/Chat/test_token_display_limit.py Tests/Chat/test_config_no_chat_context_limit.py
git commit -m "fix(tokens): gauge divides by input window; remove dead chat_context_limit (TASK-320/325)"
```

---

## Final verification (after all tasks)

- [ ] Run the full affected suite:
  `... -m pytest Tests/test_model_capabilities.py Tests/Chat/test_token_counter.py Tests/Chat/test_console_history_budget.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_footer_token_dirty_gate.py Tests/Chat/test_chat_screen_token_estimate.py Tests/Chat/test_token_display_limit.py Tests/Chat/test_config_no_chat_context_limit.py -v`
  Expected: all pass (baseline non-related failures — the pre-existing `test_anthropic_native_tools` failure and the `mocker`-fixture errors in `test_chat_functions.py` — are unaffected by this branch).
- [ ] Confirm `git grep -n "\.split()" tldw_chatbook/Utils/token_counter.py tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/UI/Screens/chat_screen.py` shows no `.split()` token estimate remains.
- [ ] Update the three backlog task files (`task-320`, `task-321`, `task-325`) — check ACs, add Implementation Notes — as the final step before finishing the branch.
