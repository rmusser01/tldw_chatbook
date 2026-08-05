# Anthropic Prompt Caching + Cache Metrics Implementation Plan (TASK-323/324)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate Anthropic prompt caching by emitting `cache_control` breakpoints on the stable system/tools prefix (gated to cache-capable Claude), and log the provider-reported cache-hit usage fields for OpenAI and Anthropic.

**Architecture:** Two changes in `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`. Task 1 adds a `_anthropic_supports_caching` gate and puts `cache_control` on the system content block + last converted tool in `chat_with_anthropic`'s payload. Task 2 appends cache-token `log_histogram` metrics to the existing OpenAI and Anthropic non-streaming usage blocks.

**Tech Stack:** Python, pytest. Tests mock `requests.Session.post` and inspect the sent JSON / spy `log_histogram`.

## Global Constraints

- **Worktree:** all work in `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-anthropic-caching` (branch `feat/anthropic-prompt-caching`, off `origin/dev @ e293b3313`). Never touch the main checkout `/Users/macbook-dev/Documents/GitHub/tldw_chatbook`.
- **Test command (venv in main checkout):** `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-anthropic-caching && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <path> -v`
- **Gate:** `_anthropic_supports_caching(model)` = `m.startswith("claude-") and not m.startswith("claude-2") and "instant" not in m` (on the lower-cased model). Only caching-capable Claude gets `cache_control`; non-caching models and non-Anthropic providers are unchanged (AC#3).
- **System (AC#1):** for caching models with a non-empty system prompt, `data["system"]` becomes `[{"type":"text","text": system_prompt, "cache_control": {"type":"ephemeral"}}]` — one breakpoint on the largest stable prefix; else the plain string, unchanged.
- **Tools (AC#2):** for caching models, put `cache_control: {"type":"ephemeral"}` on the **last converted** tool (a fresh dict — never mutate the caller's input), gated on the converted list being non-empty. ≤ 4 breakpoints total.
- **Metrics (324):** OpenAI — append `openai_api_cached_tokens` from `usage["prompt_tokens_details"]["cached_tokens"]` to the existing usage block. Anthropic — **append** `anthropic_api_cache_read_input_tokens` + `anthropic_api_cache_creation_input_tokens` to the **existing** `if usage:` block (do NOT duplicate the existing `input/output/total` histograms). All reads use `.get(..., 0)` / `or {}` — graceful on absent fields (AC#3).
- **Do NOT fix** `Tests/Chat/test_anthropic_native_tools.py::test_anthropic_shaped_tools_pass_through_untouched` — it is a pre-existing baseline failure (a `_anthropic_tools_payload` anthropic-shape drop bug, task-263 territory), out of scope. Test the tools breakpoint via the OpenAI-function-shaped → converted path (`OPENAI_TOOLS`), which works.
- **Commit only the files each task names**, with explicit `git add <paths>` — never `git add -A`/`-am` (tracked `.superpowers/` scratch + stray files must not be swept in).

---

## Task 1: Anthropic `cache_control` breakpoints (323)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (add `_anthropic_supports_caching`; the `data["system"]` and `data["tools"]` sites in `chat_with_anthropic`)
- Test: `Tests/Chat/test_anthropic_native_tools.py`

**Interfaces:**
- Produces: `_anthropic_supports_caching(model: str) -> bool`; `chat_with_anthropic` payload with `cache_control` on the system block (and last tool) for caching models.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chat/test_anthropic_native_tools.py` (it already imports `chat_api_call`, `Mock`, `patch`, and defines `_anthropic_text_response`, `OPENAI_TOOLS`):

```python
from tldw_chatbook.LLM_Calls.LLM_API_Calls import _anthropic_supports_caching


def _sent_anthropic(mock_post, model, **extra):
    """Drive chat_with_anthropic with an explicit model; return the sent JSON."""
    mock_response = Mock()
    mock_response.json.return_value = _anthropic_text_response()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response
    chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model=model,
        streaming=False,
        **extra,
    )
    return mock_post.call_args[1]["json"]


def test_anthropic_supports_caching_gate():
    assert _anthropic_supports_caching("claude-3-haiku-20240307") is True
    assert _anthropic_supports_caching("claude-3-5-sonnet-20241022") is True
    assert _anthropic_supports_caching("claude-sonnet-4-20250514") is True
    assert _anthropic_supports_caching("claude-2.1") is False
    assert _anthropic_supports_caching("claude-instant-1.2") is False
    assert _anthropic_supports_caching("gpt-4o") is False
    assert _anthropic_supports_caching("") is False


@patch("requests.Session.post")
def test_caching_model_system_gets_cache_control(mock_post):
    sent = _sent_anthropic(
        mock_post, "claude-3-opus-20240229", system_message="You are helpful."
    )
    assert isinstance(sent["system"], list)
    assert sent["system"][0]["type"] == "text"
    assert sent["system"][0]["text"] == "You are helpful."
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


@patch("requests.Session.post")
def test_caching_model_last_tool_gets_cache_control(mock_post):
    sent = _sent_anthropic(
        mock_post, "claude-3-opus-20240229", tools=OPENAI_TOOLS
    )
    assert sent["tools"], "tools should convert and survive"
    assert sent["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    # <=4 breakpoints total (system optional + one tool here)
    n = sum(
        1 for t in sent["tools"] if "cache_control" in t
    ) + (
        1 if isinstance(sent.get("system"), list) else 0
    )
    assert n <= 4


@patch("requests.Session.post")
def test_non_caching_model_unchanged(mock_post):
    sent = _sent_anthropic(
        mock_post, "claude-2.1", system_message="You are helpful.", tools=OPENAI_TOOLS
    )
    assert sent["system"] == "You are helpful."  # plain string, no blocks
    assert all("cache_control" not in t for t in sent["tools"])


@patch("requests.Session.post")
def test_caching_model_no_tools_system_only(mock_post):
    sent = _sent_anthropic(
        mock_post, "claude-3-opus-20240229", system_message="Hi."
    )
    assert isinstance(sent["system"], list)
    assert "tools" not in sent  # no tools passed -> no tools key
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Chat/test_anthropic_native_tools.py -v -k "caching or supports_caching"`
Expected: FAIL (`cannot import name '_anthropic_supports_caching'`).

- [ ] **Step 3: Add the `_anthropic_supports_caching` helper**

In `LLM_API_Calls.py`, add near `_anthropic_tools_payload` (module level):

```python
def _anthropic_supports_caching(model: str) -> bool:
    """True for Claude models that support prompt caching (``cache_control``).

    All modern Claude (3 / 3.5 / 3.7 / 4+) support it; legacy ``claude-2*`` and
    ``claude-instant*`` do not.

    Args:
        model: The model identifier.

    Returns:
        True when the model accepts ``cache_control`` breakpoints.
    """
    m = (model or "").lower()
    return (
        m.startswith("claude-")
        and not m.startswith("claude-2")
        and "instant" not in m
    )
```

- [ ] **Step 4: Put `cache_control` on the system block**

In `chat_with_anthropic`, replace:

```python
    if system_prompt is not None:
        data["system"] = system_prompt  # Anthropic uses 'system' at the top level
```

with:

```python
    if system_prompt is not None:
        if _anthropic_supports_caching(current_model) and system_prompt:
            # cache_control on the system prompt (the largest stable prefix)
            # activates Anthropic prompt caching; per the tools->system->messages
            # hierarchy this caches tools+system. Applied for both streaming and
            # non-streaming (the payload is built before the streaming branch).
            data["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            data["system"] = system_prompt  # unchanged for non-caching models
```

- [ ] **Step 5: Put a breakpoint on the last converted tool**

Replace:

```python
    if tools is not None:
        data["tools"] = _anthropic_tools_payload(tools)
```

with:

```python
    if tools is not None:
        tools_payload = _anthropic_tools_payload(tools)
        if _anthropic_supports_caching(current_model) and tools_payload:
            # Optional second breakpoint on the last converted tool. A fresh dict
            # so the caller's input `tools` are never mutated.
            tools_payload[-1] = {
                **tools_payload[-1],
                "cache_control": {"type": "ephemeral"},
            }
        data["tools"] = tools_payload
```

- [ ] **Step 6: Run to verify pass**

Run: `... -m pytest Tests/Chat/test_anthropic_native_tools.py -v`
Expected: PASS for the new tests; the pre-existing `test_anthropic_shaped_tools_pass_through_untouched` stays **FAILED** (known baseline bug, not ours — do not touch it).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_anthropic_native_tools.py
git commit -m "feat(llm): Anthropic cache_control breakpoints on system + last tool (TASK-323)"
```

---

## Task 2: Cache-hit usage metrics (324)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (OpenAI usage block in `chat_with_openai`; existing Anthropic usage block in `chat_with_anthropic`)
- Test: `Tests/Chat/test_cache_usage_metrics.py` (new)

**Interfaces:**
- Consumes: nothing from Task 1 (independent; both edit `LLM_API_Calls.py`).
- Produces: `openai_api_cached_tokens`, `anthropic_api_cache_read_input_tokens`, `anthropic_api_cache_creation_input_tokens` `log_histogram` metrics.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_cache_usage_metrics.py`:

```python
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _spy_histograms(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.log_histogram",
        lambda name, value, **kw: calls.__setitem__(name, value),
    )
    return calls


def _mock_post(resp):
    m = Mock()
    m.json.return_value = resp
    m.status_code = 200
    m.raise_for_status = Mock()
    return m


def test_anthropic_logs_cache_metrics(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10, "output_tokens": 5,
            "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20,
        },
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "anthropic", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="claude-3-opus-20240229", streaming=False,
        )
    assert calls["anthropic_api_cache_read_input_tokens"] == 100
    assert calls["anthropic_api_cache_creation_input_tokens"] == 20
    # existing metrics still fire
    assert calls["anthropic_api_input_tokens"] == 10


def test_anthropic_cache_metrics_absent_fields_zero(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "m", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 2},  # no cache fields
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "anthropic", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="claude-3-opus-20240229", streaming=False,
        )
    assert calls["anthropic_api_cache_read_input_tokens"] == 0
    assert calls["anthropic_api_cache_creation_input_tokens"] == 0


def test_openai_logs_cached_tokens(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "cmpl", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 8},
        },
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "openai", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="gpt-4o", streaming=False,
        )
    assert calls["openai_api_cached_tokens"] == 8


def test_openai_cached_tokens_absent_zero(monkeypatch):
    calls = _spy_histograms(monkeypatch)
    resp = {
        "id": "cmpl", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    with patch("requests.Session.post", return_value=_mock_post(resp)):
        chat_api_call(
            "openai", messages_payload=[{"role": "user", "content": "hi"}],
            api_key="k", model="gpt-4o", streaming=False,
        )
    assert calls["openai_api_cached_tokens"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `... -m pytest Tests/Chat/test_cache_usage_metrics.py -v`
Expected: FAIL (`KeyError: 'anthropic_api_cache_read_input_tokens'` / `'openai_api_cached_tokens'` — metrics not emitted yet).

- [ ] **Step 3: Add the OpenAI `cached_tokens` metric**

In `chat_with_openai`'s usage block, after the `openai_api_total_tokens` histogram (still inside `if usage:`), append:

```python
                log_histogram(
                    "openai_api_cached_tokens",
                    (usage.get("prompt_tokens_details") or {}).get(
                        "cached_tokens", 0
                    ),
                    labels={"model": final_model},
                )
```

- [ ] **Step 4: Add the Anthropic cache metrics**

In `chat_with_anthropic`'s **existing** usage block, after the `anthropic_api_total_tokens` histogram (still inside `if usage:`), append:

```python
                log_histogram(
                    "anthropic_api_cache_read_input_tokens",
                    usage.get("cache_read_input_tokens", 0),
                    labels={"model": current_model},
                )
                log_histogram(
                    "anthropic_api_cache_creation_input_tokens",
                    usage.get("cache_creation_input_tokens", 0),
                    labels={"model": current_model},
                )
```

- [ ] **Step 5: Run to verify pass**

Run: `... -m pytest Tests/Chat/test_cache_usage_metrics.py -v`
Expected: PASS (all four).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_cache_usage_metrics.py
git commit -m "feat(llm): log OpenAI cached_tokens + Anthropic cache-read/creation usage (TASK-324)"
```

---

## Final verification (after all tasks)

- [ ] Run both suites:
  `... -m pytest Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_cache_usage_metrics.py -v`
  Expected: all pass EXCEPT the pre-existing `test_anthropic_shaped_tools_pass_through_untouched` (known baseline failure, unchanged).
- [ ] Confirm `chat_with_openai` behavior is otherwise unchanged (Unit A only touches `chat_with_anthropic`): `... -m pytest Tests/Chat/test_chat_unit_mocked_APIs.py -q`.
- [ ] Update the `task-323` and `task-324` backlog files (check ACs, add Implementation Notes) as the final step before finishing the branch. Note AC#4's live-verification step in the notes.
