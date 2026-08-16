# Local Provider Thinking Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Console's existing Reasoning (`reasoning_effort`) and Budget (`thinking_budget_tokens`) settings through to local providers (llama.cpp family, vLLM family, MLX-LM, Custom OpenAI) so Qwen3.8-27B's adjustable thinking levels and max thinking tokens work per request, with thinking text kept out of the visible reply on the llama.cpp direct path.

**Architecture:** One pure wire-format table (`build_local_thinking_payload_fields`) maps each local execution key to payload fragments (`chat_template_kwargs` vs top-level fields vs `reasoning_budget`). The direct llama.cpp gateway path and the shared adapter-path builder both compose through it. A new start-anchored, stream-aware `<think>` filter strips unsplit thinking from the direct path only.

**Tech Stack:** Python 3.11+, httpx (direct path), requests (adapter path), pytest, Textual (modal).

**Spec:** `Docs/superpowers/specs/2026-08-15-local-thinking-controls-design.md` (argues from ADR-066: `backlog/decisions/066-local-provider-thinking-controls.md`; task: `backlog/tasks/task-16812 - Console-thinking-levels-and-budget-for-local-providers.md`)

## Global Constraints

- Values are sent **verbatim** — never clamped or rewritten. `reasoning_effort: none` additionally sends `enable_thinking: false`.
- `thinking_budget_tokens` validation floor stays a global ≥ 1024 (do not make it provider-aware).
- Prefill precedence is `prefill > none > effort`: a trailing assistant message always forces `chat_template_kwargs.enable_thinking = false`.
- llama.cpp family level goes inside `chat_template_kwargs.reasoning_effort` (llama-server does not parse top-level `reasoning_effort`); budget as top-level `reasoning_budget`.
- vLLM family level goes BOTH top-level and inside `chat_template_kwargs.reasoning_effort` (same value).
- Custom OpenAI endpoints (`custom-openai-api`, `custom-openai-api-2`) get top-level `reasoning_effort` ONLY — no llama.cpp-specific fields.
- `<think>` stripping is start-anchored: mid-reply literal `<think>` text is legitimate content and must survive.
- No new Console settings fields. No changes to managed-server launch defaults (`--jinja` etc. remain user-supplied).
- Warnings are non-blocking; unknown models produce no warning.
- All SQL/param conventions N/A (no DB changes). Tests must pass `pytest Tests/Chat/` before each commit.

---

### Task 1: Wire-format table (pure function)

**Files:**
- Create: `Tests/Chat/test_local_thinking_wire_formats.py`
- Modify: `tldw_chatbook/Chat/console_provider_support.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_local_thinking_payload_fields(execution_key: str, reasoning_effort: str | None, thinking_budget_tokens: int | None) -> dict[str, Any]` — payload fragments to `dict.update()` into an OpenAI-compatible chat payload. Returns `{}` for keys not in the table.

- [ ] **Step 0: Create the working branch**

The spec/ADR/task/plan files are currently untracked on `feat/model-catalog-consent-gate`, which holds unrelated WIP. Do NOT commit them there. Create the implementation branch off `main` and carry the doc files over:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git stash push --include-untracked -m "wip: unrelated model-catalog work" || true
git checkout main && git pull || git checkout main
git checkout -b feat/local-thinking-controls
git stash pop || true
# verify the four doc files are present and staged-able:
ls Docs/superpowers/specs/2026-08-15-local-thinking-controls-design.md \
   backlog/decisions/066-local-provider-thinking-controls.md \
   "backlog/tasks/task-16812 - Console-thinking-levels-and-budget-for-local-providers.md" \
   Docs/superpowers/plans/2026-08-15-local-thinking-controls.md
```

If the stash pop conflicts, resolve by keeping the untracked doc files on the new branch. Commit the docs:

```bash
git add Docs/superpowers/specs/2026-08-15-local-thinking-controls-design.md \
        backlog/decisions/066-local-provider-thinking-controls.md \
        "backlog/tasks/task-16812 - Console-thinking-levels-and-budget-for-local-providers.md" \
        Docs/superpowers/plans/2026-08-15-local-thinking-controls.md
git commit -m "docs: spec, ADR-066, and plan for local thinking controls"
```

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_local_thinking_wire_formats.py`:

```python
"""Wire-format composition for local thinking controls (ADR-066)."""

from tldw_chatbook.Chat.console_provider_support import (
    build_local_thinking_payload_fields,
)


class TestLlamaCppFamily:
    def test_level_goes_into_chat_template_kwargs(self):
        fields = build_local_thinking_payload_fields(
            "llama_cpp", "low", None
        )
        assert fields == {"chat_template_kwargs": {"reasoning_effort": "low"}}

    def test_budget_goes_top_level(self):
        fields = build_local_thinking_payload_fields(
            "local_llamacpp", None, 2048
        )
        assert fields == {"reasoning_budget": 2048}

    def test_level_and_budget_together(self):
        fields = build_local_thinking_payload_fields(
            "local_llamafile", "xhigh", 4096
        )
        assert fields == {
            "chat_template_kwargs": {"reasoning_effort": "xhigh"},
            "reasoning_budget": 4096,
        }

    def test_none_effort_sends_verbatim_and_disables_thinking(self):
        fields = build_local_thinking_payload_fields(
            "local-llm", "none", None
        )
        assert fields == {
            "chat_template_kwargs": {
                "reasoning_effort": "none",
                "enable_thinking": False,
            }
        }

    def test_all_family_keys_share_the_shape(self):
        for key in ("llama_cpp", "local_llamacpp", "local_llamafile", "local-llm"):
            assert build_local_thinking_payload_fields(key, "low", 1024) == {
                "chat_template_kwargs": {"reasoning_effort": "low"},
                "reasoning_budget": 1024,
            }


class TestVllmFamily:
    def test_level_is_dual_placed(self):
        for key in ("vllm", "local_vllm"):
            assert build_local_thinking_payload_fields(key, "medium", None) == {
                "reasoning_effort": "medium",
                "chat_template_kwargs": {"reasoning_effort": "medium"},
            }

    def test_budget_is_dropped(self):
        assert build_local_thinking_payload_fields("vllm", None, 2048) == {}


class TestCustomOpenAI:
    def test_level_is_top_level_only(self):
        for key in ("custom-openai-api", "custom-openai-api-2"):
            assert build_local_thinking_payload_fields(key, "low", 2048) == {
                "reasoning_effort": "low"
            }


class TestMlx:
    def test_level_via_template_kwargs_budget_dropped(self):
        fields = build_local_thinking_payload_fields(
            "local_mlx_lm", "low", 2048
        )
        assert fields == {"chat_template_kwargs": {"reasoning_effort": "low"}}


class TestUnknownKeysAndHygiene:
    def test_unknown_key_returns_empty(self):
        assert build_local_thinking_payload_fields("ollama", "low", 1024) == {}
        assert build_local_thinking_payload_fields("openai", "low", 1024) == {}

    def test_blank_effort_and_missing_budget_return_empty(self):
        assert build_local_thinking_payload_fields("llama_cpp", "", None) == {}
        assert build_local_thinking_payload_fields("llama_cpp", None, None) == {}

    def test_whitespace_effort_is_normalized(self):
        assert build_local_thinking_payload_fields("llama_cpp", "  low ", None) == {
            "chat_template_kwargs": {"reasoning_effort": "low"}
        }

    def test_non_int_budget_is_ignored(self):
        assert build_local_thinking_payload_fields("llama_cpp", None, "2048") == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_local_thinking_wire_formats.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_local_thinking_payload_fields'`

- [ ] **Step 3: Implement the table**

Append to `tldw_chatbook/Chat/console_provider_support.py` (after `_provider_display_name`; add `from typing import Any` to imports if absent, plus `from loguru import logger` if absent):

```python
# ADR-066: per-execution-key wire formats for Console thinking controls.
# Level = reasoning_effort; budget = thinking_budget_tokens.
_LLAMA_CPP_THINKING_KEYS = frozenset(
    {"llama_cpp", "local_llamacpp", "local_llamafile", "local-llm"}
)
_VLLM_THINKING_KEYS = frozenset({"vllm", "local_vllm"})
_CUSTOM_OPENAI_THINKING_KEYS = frozenset(
    {"custom-openai-api", "custom-openai-api-2"}
)
# MLX-LM: template-kwargs shape pending live verification of mlx_lm.server
# support; if unsupported this row degrades to drop-and-log.
_TEMPLATE_KWARGS_THINKING_KEYS = frozenset({"local_mlx_lm"})


def build_local_thinking_payload_fields(
    execution_key: str | None,
    reasoning_effort: str | None,
    thinking_budget_tokens: int | None,
) -> dict[str, Any]:
    """Compose thinking-control payload fragments for a local provider.

    Args:
        execution_key: ``chat_api_call`` provider key (e.g. ``llama_cpp``).
        reasoning_effort: Verbatim user-selected effort level, if any.
        thinking_budget_tokens: Max thinking tokens, if any.

    Returns:
        Fragments to merge into an OpenAI-compatible chat payload. Empty
        dict when the key has no thinking support or no values are set.
    """
    key = str(execution_key or "").strip().lower()
    effort = str(reasoning_effort or "").strip().lower() or None
    budget: int | None = (
        thinking_budget_tokens
        if isinstance(thinking_budget_tokens, int)
        and not isinstance(thinking_budget_tokens, bool)
        else None
    )
    fields: dict[str, Any] = {}
    if key in _LLAMA_CPP_THINKING_KEYS or key in _TEMPLATE_KWARGS_THINKING_KEYS:
        if effort is not None:
            template_kwargs: dict[str, Any] = {"reasoning_effort": effort}
            if effort == "none":
                template_kwargs["enable_thinking"] = False
            fields["chat_template_kwargs"] = template_kwargs
        if budget is not None and key in _LLAMA_CPP_THINKING_KEYS:
            fields["reasoning_budget"] = budget
        if budget is not None and key in _TEMPLATE_KWARGS_THINKING_KEYS:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    elif key in _VLLM_THINKING_KEYS:
        if effort is not None:
            fields["reasoning_effort"] = effort
            fields["chat_template_kwargs"] = {"reasoning_effort": effort}
        if budget is not None:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    elif key in _CUSTOM_OPENAI_THINKING_KEYS:
        if effort is not None:
            fields["reasoning_effort"] = effort
        if budget is not None:
            logger.debug(
                "thinking budget not supported for provider {}; dropped",
                key,
            )
    return fields
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Chat/test_local_thinking_wire_formats.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_support.py Tests/Chat/test_local_thinking_wire_formats.py
git commit -m "feat(chat): ADR-066 wire-format table for local thinking controls"
```

---

### Task 2: Direct llama.cpp payload wiring (gateway)

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (`build_llamacpp_chat_payload` ~line 845, `stream_llamacpp_chat` ~1795, `complete_llamacpp_chat` ~1877, call sites ~2008 and ~2219/2233)
- Test: `Tests/Chat/test_console_provider_gateway.py` (extend)

**Interfaces:**
- Consumes: `build_local_thinking_payload_fields` from Task 1.
- Produces: `build_llamacpp_chat_payload(..., reasoning_effort: str | None = None, thinking_budget_tokens: int | None = None)`; `stream_llamacpp_chat` and `complete_llamacpp_chat` gain the same two keyword params. Thinking fields are NOT applied in this task's stream output — filtering lands in Task 4.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chat/test_console_provider_gateway.py` (follow the file's existing import style for `build_llamacpp_chat_payload`):

```python
class TestLlamacppThinkingPayload:
    def test_effort_composes_chat_template_kwargs(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=True, reasoning_effort="low",
        )
        assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}

    def test_budget_composes_reasoning_budget(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=False, thinking_budget_tokens=2048,
        )
        assert payload["reasoning_budget"] == 2048

    def test_none_effort_disables_thinking(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=True, reasoning_effort="none",
        )
        assert payload["chat_template_kwargs"]["enable_thinking"] is False

    def test_prefill_overrides_effort(self):
        payload = build_llamacpp_chat_payload(
            model="qwen",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "Sure"},
            ],
            stream=True, reasoning_effort="xhigh",
        )
        # prefill > none > effort (llama.cpp rejects prefill + thinking)
        assert payload["chat_template_kwargs"] == {
            "reasoning_effort": "xhigh",
            "enable_thinking": False,
        }

    def test_no_thinking_fields_by_default(self):
        payload = build_llamacpp_chat_payload(
            model="qwen", messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        assert "chat_template_kwargs" not in payload
        assert "reasoning_budget" not in payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_console_provider_gateway.py -k TestLlamacppThinkingPayload -v`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'reasoning_effort'`

- [ ] **Step 3: Implement**

In `tldw_chatbook/Chat/console_provider_gateway.py`:

3a. Add to imports: `from tldw_chatbook.Chat.console_provider_support import build_local_thinking_payload_fields`

3b. In `build_llamacpp_chat_payload`, add keyword params `reasoning_effort: str | None = None,` and `thinking_budget_tokens: int | None = None,` (after `frequency_penalty`), extend the docstring Args with both, and REPLACE the current prefill tail:

```python
    # OLD (replace):
    # if messages and messages[-1].get("role") == "assistant":
    #     payload["chat_template_kwargs"] = {"enable_thinking": False}
    # NEW:
    payload.update(
        build_local_thinking_payload_fields(
            "llama_cpp", reasoning_effort, thinking_budget_tokens
        )
    )
    if messages and messages[-1].get("role") == "assistant":
        template_kwargs = dict(payload.get("chat_template_kwargs") or {})
        template_kwargs["enable_thinking"] = False
        payload["chat_template_kwargs"] = template_kwargs
    return payload
```

Also update the docstring paragraph about prefills to state the precedence: `prefill > none > effort`.

3c. In `stream_llamacpp_chat` and `complete_llamacpp_chat`: add the same two keyword params to the signatures, forward them into their `build_llamacpp_chat_payload(...)` calls, and document them in the docstrings.

3d. At the two direct-path call sites: the auxiliary site (~line 2008, inside `complete_auxiliary`) currently has a comment claiming reasoning controls are "deliberately omitted" — that comment is now wrong. Replace the call with:

```python
                    text = await self.complete_llamacpp_chat(
                        base_url=resolution.base_url,
                        model=model,
                        messages=messages,
                        temperature=resolution.temperature,
                        top_p=resolution.top_p,
                        min_p=resolution.min_p,
                        top_k=resolution.top_k,
                        max_tokens=request.max_output_tokens,
                        seed=resolution.seed,
                        presence_penalty=resolution.presence_penalty,
                        frequency_penalty=resolution.frequency_penalty,
                        reasoning_effort=resolution.reasoning_effort,
                        thinking_budget_tokens=resolution.thinking_budget_tokens,
                        strict_response=True,
```
and rewrite the comment above it to: `# Thinking controls follow ADR-066: level via chat_template_kwargs, budget via top-level reasoning_budget. Auxiliary requests inherit session thinking settings (documented parity with cloud providers).`

The send-path site (~lines 2219 and 2233, inside `stream_chat`): add the same two kwargs to BOTH the `complete_llamacpp_chat(...)` and `stream_llamacpp_chat(...)` calls:

```python
                        reasoning_effort=resolution.reasoning_effort,
                        thinking_budget_tokens=resolution.thinking_budget_tokens,
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/Chat/test_console_provider_gateway.py -v`
Expected: all PASS (existing tests included — no regressions).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat(console): send thinking level+budget on direct llama.cpp path"
```

---

### Task 3: Start-anchored think filter

**Files:**
- Create: `tldw_chatbook/Chat/llamacpp_think_filter.py`
- Test: `Tests/Chat/test_llamacpp_think_filter.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `class StartAnchoredThinkFilter` with `feed(self, chunk: str) -> str` and `flush(self) -> str`. Pure string state machine; no I/O. Consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_llamacpp_think_filter.py`:

```python
from tldw_chatbook.Chat.llamacpp_think_filter import StartAnchoredThinkFilter


def run_filter(text: str) -> str:
    f = StartAnchoredThinkFilter()
    return f.feed(text) + f.flush()


class TestSplitAcrossChunks:
    def test_open_tag_split_across_chunks(self):
        f = StartAnchoredThinkFilter()
        assert f.feed("<thi") == ""
        assert f.feed("nk>reasoning") == ""
        assert f.feed(" here</think>answer") == "answer"
        assert f.flush() == ""

    def test_close_tag_split_across_chunks(self):
        f = StartAnchoredThinkFilter()
        f.feed("<think>abc")
        assert f.feed("</thi") == ""
        assert f.feed("nk>done") == "done"

    def test_one_char_at_a_time(self):
        f = StartAnchoredThinkFilter()
        out = []
        for ch in "<think>hidden</think>visible":
            out.append(f.feed(ch))
        assert "".join(out) + f.flush() == "visible"


class TestAnchoring:
    def test_mid_reply_literal_tag_survives(self):
        assert run_filter("Here is XML: <think>stuff</think> done") == (
            "Here is XML: <think>stuff</think> done"
        )

    def test_leading_whitespace_before_tag_is_tolerated(self):
        assert run_filter("\n\n<think>x</think>hi") == "hi"

    def test_empty_prefix_block_is_removed(self):
        # Some Qwen generations emit an empty think prefix in no-think mode.
        assert run_filter("<think>\n\n</think>\n\nAnswer") == "Answer"

    def test_plain_text_passes_through(self):
        assert run_filter("just an answer") == "just an answer"

    def test_text_starting_like_tag_but_not_tag_passes(self):
        assert run_filter("<thumbs up>") == "<thumbs up>"

    def test_thinking_tag_variant_supported(self):
        assert run_filter("<thinking>deep</thinking>ok") == "ok"


class TestUnterminated:
    def test_unterminated_start_anchored_block_dropped_on_flush(self):
        f = StartAnchoredThinkFilter()
        assert f.feed("<think>forever") == ""
        assert f.flush() == ""

    def test_ambiguous_prefix_at_stream_end_dropped(self):
        f = StartAnchoredThinkFilter()
        assert f.feed("<thin") == ""
        assert f.flush() == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_llamacpp_think_filter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Chat.llamacpp_think_filter'`

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Chat/llamacpp_think_filter.py`:

```python
"""Start-anchored, stream-aware ``<think>`` filtering for llama.cpp output.

Qwen-family chat templates emit thinking at the very start of the response
(an empty ``<think>\\n\\n</think>`` prefix appears in no-think mode on some
generations). Only a think block that opens at the beginning of the stream
is stripped; a literal ``<think>`` mid-reply (e.g. the user asked for an XML
example) is legitimate content and passes through. See ADR-066.
"""

from __future__ import annotations

_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")


class StartAnchoredThinkFilter:
    """Stateful filter: feed() chunks in, get visible text out; flush() at end."""

    def __init__(self) -> None:
        self._inside_think = False
        self._decided_visible = False
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        if self._decided_visible:
            return chunk
        self._buffer += chunk
        while True:
            if self._inside_think:
                for tag in _CLOSE_TAGS:
                    idx = self._buffer.find(tag)
                    if idx != -1:
                        self._buffer = self._buffer[idx + len(tag):]
                        self._inside_think = False
                        self._decided_visible = True
                        return self._buffer.lstrip("\n")
                return ""
            stripped = self._buffer.lstrip()
            if not stripped:
                return ""  # whitespace-only so far; keep probing
            for tag in _OPEN_TAGS:
                if stripped.startswith(tag):
                    self._inside_think = True
                    self._buffer = stripped[len(tag):]
                    break
            if self._inside_think:
                continue  # re-run close-tag scan on the remainder
            if any(tag.startswith(stripped) for tag in _OPEN_TAGS):
                return ""  # still ambiguous: could be a split tag opener
            self._decided_visible = True
            return self._buffer

    def flush(self) -> str:
        # Stream ended while still probing or still inside an unterminated
        # think block: drop the tail (spec'd behavior).
        return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Chat/test_llamacpp_think_filter.py -v`
Expected: all PASS. If `test_text_starting_like_tag_but_not_tag_passes` fails, check the ambiguity branch order: divergence must be detected only when NO open tag starts with `stripped`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/llamacpp_think_filter.py Tests/Chat/test_llamacpp_think_filter.py
git commit -m "feat(chat): start-anchored stream-aware think-tag filter"
```

---

### Task 4: Wire the filter into the direct path

**Files:**
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py` (`stream_llamacpp_chat`, `complete_llamacpp_chat`)
- Test: `Tests/Chat/test_console_provider_gateway.py` (extend)

**Interfaces:**
- Consumes: `StartAnchoredThinkFilter` from Task 3.
- Produces: no new public interface — behavior change only. Note: the SSE parser `_content_from_sse_line` and the completion parser `_content_from_completion_response` ALREADY ignore `reasoning_content` (they read `content` only), so the server-split case needs no change; do not touch them.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chat/test_console_provider_gateway.py`. Follow the existing file's fake-SSE harness pattern for `stream_llamacpp_chat` (search the file for `stream_llamacpp_chat` tests and reuse their transport mock); the shapes below assume a helper `fake_sse_gateway(lines)` like the existing tests use — copy the closest existing test's setup verbatim and adapt lines:

```python
class TestDirectPathThinkFiltering:
    def test_stream_strips_start_anchored_think_block(self):
        # SSE lines whose content deltas spell:
        #   "<think>ponder</think>Hello"
        lines = [
            'data: {"choices":[{"delta":{"content":"<think>pon"}}]}',
            'data: {"choices":[{"delta":{"content":"der</think>Hello"}}]}',
            "data: [DONE]",
        ]
        gateway = make_gateway_with_sse(lines)  # reuse existing test harness
        chunks = [
            c
            async for c in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert "".join(chunks) == "Hello"

    def test_stream_passes_mid_reply_literal_tag(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"XML: <think>x</think>"}}]}',
            "data: [DONE]",
        ]
        gateway = make_gateway_with_sse(lines)
        chunks = [
            c
            async for c in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert "".join(chunks) == "XML: <think>x</think>"

    def test_stream_ignores_reasoning_content_deltas(self):
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"secret"}}]}',
            'data: {"choices":[{"delta":{"content":"Answer"}}]}',
            "data: [DONE]",
        ]
        gateway = make_gateway_with_sse(lines)
        chunks = [
            c
            async for c in gateway.stream_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        assert "".join(chunks) == "Answer"

    def test_complete_strips_start_anchored_think_block(self):
        gateway = make_gateway_with_completion(
            {"choices": [{"message": {"content": "<think>x</think>Done"}}]}
        )
        text = asyncio.run(
            gateway.complete_llamacpp_chat(
                base_url="http://127.0.0.1:8080",
                model="qwen",
                messages=[{"role": "user", "content": "hi"}],
            )
        )
        assert text == "Done"
```

If the existing file has no reusable SSE harness, build one with `httpx.MockTransport` mounted on an `httpx.AsyncClient` passed as the gateway's `http_client` constructor arg (the gateway accepts an injected client — see its `__init__`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_console_provider_gateway.py -k TestDirectPathThinkFiltering -v`
Expected: FAIL — streams yield `<think>ponder</think>Hello` unfiltered.

- [ ] **Step 3: Implement**

In `stream_llamacpp_chat` (`tldw_chatbook/Chat/console_provider_gateway.py`), instantiate a filter at the top and apply it to every yield. The `emitted_content` flag must track VISIBLE text only (it decides whether the non-streaming fallback runs):

```python
        from tldw_chatbook.Chat.llamacpp_think_filter import StartAnchoredThinkFilter

        think_filter = StartAnchoredThinkFilter()
        emitted_content = False
        stream_error: httpx.HTTPError | None = None
        try:
            async with self._active_http_client().stream(
                ...
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    chunk = self._content_from_sse_line(line)
                    if chunk:
                        visible = think_filter.feed(chunk)
                        if visible:
                            emitted_content = True
                            yield visible
        except httpx.HTTPError as exc:
            if emitted_content:
                raise
            stream_error = exc

        if emitted_content:
            tail = think_filter.flush()
            if tail:
                yield tail
            return
```

(Move the `StartAnchoredThinkFilter` import to module top with the other local imports rather than inline if style demands.)

In `complete_llamacpp_chat`, filter the final string before returning — replace `return content or ""` with:

```python
        think_filter = StartAnchoredThinkFilter()
        return think_filter.feed(content or "") + think_filter.flush()
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_llamacpp_think_filter.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_provider_gateway.py Tests/Chat/test_console_provider_gateway.py
git commit -m "feat(console): strip start-anchored think blocks on direct llama.cpp path"
```

---

### Task 5: Adapter path — param maps, shared builder, handlers

**Files:**
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (`PROVIDER_PARAM_MAP` entries at lines ~391 `llama_cpp`, ~484 `local-llm` (inside the vLLM block), ~535 `custom-openai-api`, ~560 `custom-openai-api-2`, ~611 `local_llamacpp`, ~630 `local_llamafile`, ~665 `local_vllm`, ~686 `local_mlx_lm`, and `vllm` at ~463)
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` (`_chat_with_openai_compatible_local_server` at ~103, `chat_with_local_llm` ~500, `chat_with_llama` ~632, `chat_with_vllm` ~1327, `chat_with_custom_openai` ~1790, `chat_with_custom_openai_2` ~1949, `chat_with_mlx_lm` ~2125)
- Test: Create `Tests/Chat/test_local_adapter_thinking_dispatch.py`

**Interfaces:**
- Consumes: `build_local_thinking_payload_fields` from Task 1.
- Produces: `chat_api_call(api_endpoint=..., reasoning_effort=..., thinking_budget_tokens=...)` now forwards these to the covered local handlers; the shared builder merges the wire fragments into its POST payload.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Chat/test_local_adapter_thinking_dispatch.py`:

```python
"""Adapter-path thinking dispatch: param maps + shared local builder."""

from unittest.mock import patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call

COVERED_KEYS = [
    "llama_cpp",
    "local_llamacpp",
    "local_llamafile",
    "local-llm",
    "vllm",
    "local_vllm",
    "local_mlx_lm",
    "custom-openai-api",
    "custom-openai-api-2",
]


def test_param_maps_forward_thinking_fields(monkeypatch):
    captured = {}

    def fake_handler(**kwargs):
        captured.update(kwargs)
        return "ok"

    import tldw_chatbook.Chat.Chat_Functions as cf

    for key in COVERED_KEYS:
        captured.clear()
        monkeypatch.setitem(cf.API_CALL_HANDLERS, key, fake_handler)
        chat_api_call(
            api_endpoint=key,
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key=None,
            reasoning_effort="low",
            thinking_budget_tokens=2048,
        )
        assert captured.get("reasoning_effort") == "low", key
        assert captured.get("thinking_budget_tokens") == 2048, key


def test_shared_builder_composes_llama_wire_format(monkeypatch):
    from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
        _chat_with_openai_compatible_local_server,
    )

    posted = {}

    class FakeResponse:
        status_code = 200
        text = '{"choices":[{"message":{"content":"ok"}}]}'

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            return None

    class FakeSession:
        def __init__(self):
            self.adapters = None

        def mount(self, *a, **k):
            pass

        def post(self, url, json=None, headers=None, timeout=None):
            posted["url"] = url
            posted["payload"] = json
            return FakeResponse()

        def close(self):
            pass

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.requests.Session",
        return_value=FakeSession(),
    ):
        _chat_with_openai_compatible_local_server(
            api_base_url="http://127.0.0.1:8080",
            model_name="qwen",
            input_data=[{"role": "user", "content": "hi"}],
            streaming=False,
            reasoning_effort="low",
            thinking_budget_tokens=2048,
            thinking_wire_key="llama_cpp",
        )

    payload = posted["payload"]
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "low"}
    assert payload["reasoning_budget"] == 2048


def test_shared_builder_composes_vllm_dual_placement(monkeypatch):
    # Same harness as above; only assertions differ.
    posted = {}

    class FakeResponse:
        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

        def raise_for_status(self):
            return None

    class FakeSession:
        def mount(self, *a, **k):
            pass

        def post(self, url, json=None, headers=None, timeout=None):
            posted["payload"] = json
            return FakeResponse()

        def close(self):
            pass

    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls_Local.requests.Session",
        return_value=FakeSession(),
    ):
        from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
            _chat_with_openai_compatible_local_server,
        )

        _chat_with_openai_compatible_local_server(
            api_base_url="http://127.0.0.1:8000",
            model_name="qwen",
            input_data=[{"role": "user", "content": "hi"}],
            streaming=False,
            reasoning_effort="medium",
            thinking_budget_tokens=2048,
            thinking_wire_key="vllm",
        )

    payload = posted["payload"]
    assert payload["reasoning_effort"] == "medium"
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "medium"}
    assert "reasoning_budget" not in payload
```

Note: if the shared builder's retry/session plumbing differs (e.g. it configures `Retry` via `requests.adapters`), keep the FakeSession's `mount`/attribute surface compatible with what the code touches — read the function body around `requests.Session()` before finalizing the fake.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_local_adapter_thinking_dispatch.py -v`
Expected: FAIL — `captured.get("reasoning_effort") is None` (maps don't forward) and `TypeError` on `thinking_wire_key`.

- [ ] **Step 3: Implement**

3a. `Chat_Functions.py` — add these two entries to each of the nine `PROVIDER_PARAM_MAP` keys listed in Files (place near the other generic entries):

```python
        "reasoning_effort": "reasoning_effort",
        "thinking_budget_tokens": "thinking_budget_tokens",
```

3b. `LLM_API_Calls_Local.py` — shared builder: add keyword params to `_chat_with_openai_compatible_local_server` (after `user_identifier`):

```python
    reasoning_effort: Optional[str] = None,
    thinking_budget_tokens: Optional[int] = None,
    thinking_wire_key: Optional[str] = None,
```

Add to module imports: `from tldw_chatbook.Chat.console_provider_support import build_local_thinking_payload_fields`

In the payload-construction block (after the `user_identifier` mapping, before the URL construction), add:

```python
    if thinking_wire_key and (
        reasoning_effort is not None or thinking_budget_tokens is not None
    ):
        payload.update(
            build_local_thinking_payload_fields(
                thinking_wire_key, reasoning_effort, thinking_budget_tokens
            )
        )
```

3c. Each handler gains the two params and forwards them plus its wire key to the shared builder call:

| Handler | Signature additions | Forward as `thinking_wire_key=` |
|---|---|---|
| `chat_with_llama` | `reasoning_effort: Optional[str] = None, thinking_budget_tokens: Optional[int] = None` | `provider_name or "llama_cpp"` |
| `chat_with_local_llm` | same | `provider_name or "local-llm"` |
| `chat_with_vllm` | same | `provider_name or "vllm"` |
| `chat_with_mlx_lm` | same | `provider_name or "local_mlx_lm"` |
| `chat_with_custom_openai` | same | `"custom-openai-api"` |
| `chat_with_custom_openai_2` | same | `"custom-openai-api-2"` |

Example for `chat_with_vllm`'s delegation call — add alongside the other forwards:

```python
        reasoning_effort=reasoning_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        thinking_wire_key=provider_name or "vllm",
```

(For `chat_with_llama`/`chat_with_custom_openai*` the delegation shape is identical — they all `return _chat_with_openai_compatible_local_server(...)`.)

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/Chat/test_local_adapter_thinking_dispatch.py Tests/Chat/ -v`
Expected: all PASS, no regressions across `Tests/Chat/`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/Chat_Functions.py tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py Tests/Chat/test_local_adapter_thinking_dispatch.py
git commit -m "feat(chat): forward thinking controls through adapter-path local providers"
```

---

### Task 6: Hints, warnings, placeholder, preview

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py` (hint table + `console_settings_warnings` + summary row, ~lines 103-110 for constants, ~768 for `sampling_parts`)
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py` (`_choice_placeholder` ~1758, save path ~1408/1426 and settings construction ~2042, `PROVIDER_CHOICE_INPUTS` ~109-125)
- Test: `Tests/Chat/test_console_session_settings.py` (extend)

**Interfaces:**
- Consumes: `build_local_thinking_payload_fields` (Task 1); `resolve_console_provider_identity` (existing in `console_provider_support.py`).
- Produces: `reasoning_effort_hint_for_model(model: str | None) -> frozenset[str] | None` and `console_settings_warnings(settings: ConsoleSessionSettings) -> list[str]`, both importable from `tldw_chatbook.Chat.console_session_settings`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Chat/test_console_session_settings.py`:

```python
from tldw_chatbook.Chat.console_session_settings import (
    console_settings_warnings,
    reasoning_effort_hint_for_model,
)


class TestReasoningEffortHints:
    def test_dotted_qwen_generations_are_effort_capable(self):
        for model in ("Qwen3.8-27B", "qwen3.5-397b-gguf:q4"):
            assert reasoning_effort_hint_for_model(model) == frozenset(
                {"low", "medium", "xhigh"}
            )

    def test_original_qwen3_is_toggle_only(self):
        assert reasoning_effort_hint_for_model("Qwen3-32B") == frozenset({"none"})

    def test_gpt_oss(self):
        assert reasoning_effort_hint_for_model("gpt-oss-120b") == frozenset(
            {"low", "medium", "high"}
        )

    def test_unknown_model_has_no_hint(self):
        assert reasoning_effort_hint_for_model("llama-3-8b") is None
        assert reasoning_effort_hint_for_model(None) is None
        assert reasoning_effort_hint_for_model("") is None


class TestConsoleSettingsWarnings:
    def _settings(self, **overrides):
        base = dict(provider="llama_cpp", model="Qwen3.8-27B")
        base.update(overrides)
        # Use the file's existing settings-construction helper if one exists
        # (search for ConsoleSessionSettings( in this test file); otherwise:
        return ConsoleSessionSettings(**base)

    def test_value_outside_hint_warns(self):
        settings = self._settings(reasoning_effort="high")
        warnings = console_settings_warnings(settings)
        assert len(warnings) == 1
        assert "high" in warnings[0]
        assert "xhigh" in warnings[0]

    def test_value_inside_hint_does_not_warn(self):
        settings = self._settings(reasoning_effort="xhigh")
        assert console_settings_warnings(settings) == []

    def test_unknown_model_does_not_warn(self):
        settings = self._settings(model="llama-3-8b", reasoning_effort="high")
        assert console_settings_warnings(settings) == []

    def test_llama_family_thinking_note_included(self):
        settings = self._settings(reasoning_effort="low")
        warnings = console_settings_warnings(settings)
        assert any("--jinja" in w for w in warnings)

    def test_llama_family_note_requires_a_thinking_value(self):
        settings = self._settings()
        assert console_settings_warnings(settings) == []
```

Adjust `_settings` to match the real `ConsoleSessionSettings` required fields (read the dataclass first; reuse an existing fixture from the test file if there is one).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Chat/test_console_session_settings.py -k "TestReasoningEffortHints or TestConsoleSettingsWarnings" -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement**

3a. `console_session_settings.py` — add near the other value sets (~line 103), plus `import re` at top if absent:

```python
# Generation-aware: dotted Qwen3.x generations consume effort levels;
# original Qwen3 is a thinking toggle only. Ordered most-specific first.
_REASONING_EFFORT_MODEL_HINTS: tuple[tuple[str, frozenset[str]], ...] = (
    ("gpt-oss", frozenset({"low", "medium", "high"})),
    ("qwen3", frozenset({"none"})),
)
_QWEN_DOTTED_EFFORT_VALUES = frozenset({"low", "medium", "xhigh"})
_LLAMA_CPP_FAMILY_PROVIDERS = frozenset(
    {"llama_cpp", "local_llamacpp", "local_llamafile"}
)
_LLAMACPP_THINKING_REQUIREMENTS_NOTE = (
    "Thinking controls on llama.cpp need llama-server started with --jinja; "
    "per-request reasoning_budget needs llama.cpp b9982 or newer."
)


def reasoning_effort_hint_for_model(model: str | None) -> frozenset[str] | None:
    """Return the effort values this model family's template consumes."""
    lowered = str(model or "").strip().lower()
    if not lowered:
        return None
    if re.search(r"qwen3\.\d", lowered):
        return _QWEN_DOTTED_EFFORT_VALUES
    for needle, values in _REASONING_EFFORT_MODEL_HINTS:
        if needle in lowered:
            return values
    return None


def console_settings_warnings(settings: ConsoleSessionSettings) -> list[str]:
    """Non-blocking warnings for the Console settings modal (ADR-066)."""
    warnings: list[str] = []
    effort = str(settings.reasoning_effort or "").strip().lower()
    has_thinking_value = bool(effort) or settings.thinking_budget_tokens is not None
    if effort:
        hint = reasoning_effort_hint_for_model(settings.model)
        if hint is not None and effort not in hint:
            warnings.append(
                f"Reasoning effort '{effort}' is not consumed by this model "
                f"family; expected one of: {', '.join(sorted(hint))}."
            )
    if has_thinking_value and settings.provider in _LLAMA_CPP_FAMILY_PROVIDERS:
        warnings.append(_LLAMACPP_THINKING_REQUIREMENTS_NOTE)
    return warnings
```

3b. Summary row (same file, in the summary builder containing `sampling_parts`, ~line 768) — extend after the existing reasoning/thinking lines:

```python
    if settings.thinking_budget_tokens is not None:
        sampling_parts.append(
            f"think budget {settings.thinking_budget_tokens}"
        )
    if settings.reasoning_effort or settings.thinking_budget_tokens is not None:
        identity = resolve_console_provider_identity(settings.provider)
        wire_fields = build_local_thinking_payload_fields(
            identity.execution_key,
            settings.reasoning_effort,
            settings.thinking_budget_tokens,
        )
        if wire_fields:
            wire_parts = []
            template_kwargs = wire_fields.get("chat_template_kwargs")
            if template_kwargs:
                rendered = ", ".join(
                    f"{k}={v}" for k, v in sorted(template_kwargs.items())
                )
                wire_parts.append(f"chat_template_kwargs[{rendered}]")
            if "reasoning_budget" in wire_fields:
                wire_parts.append(
                    f"reasoning_budget={wire_fields['reasoning_budget']}"
                )
            if "reasoning_effort" in wire_fields:
                wire_parts.append(
                    f"reasoning_effort={wire_fields['reasoning_effort']}"
                )
            sampling_parts.append("wire: " + "; ".join(wire_parts))
```

Add the imports `from tldw_chatbook.Chat.console_provider_support import build_local_thinking_payload_fields, resolve_console_provider_identity` if not already present (check — `resolve_console_provider_identity` may already be imported in this module).

3c. `console_settings_modal.py` — dynamic placeholder in `_choice_placeholder` (~1758):

```python
    def _choice_placeholder(self, input_id: str) -> str:
        """Return the accepted-values placeholder for an enumerated choice input."""
        if input_id == "console-settings-reasoning-effort":
            hint = reasoning_effort_hint_for_model(self._settings.model)
            if hint is not None:
                return " / ".join(sorted(hint)) + " (consumed by this model)"
        for _label, choice_input_id, placeholder in PROVIDER_CHOICE_INPUTS:
            if choice_input_id == input_id:
                return placeholder
        return ""
```

3d. Warning surfacing — in the settings-construction function (~line 2042, where the parsed settings object is built) compute `console_settings_warnings(settings)`; in the save handlers (~1408/1426, where `_validated_result_or_show_errors()` succeeds) notify each warning without blocking:

```python
        for warning in console_settings_warnings(settings):
            self.notify(warning, severity="warning", timeout=8000)
```

Import `console_settings_warnings` and `reasoning_effort_hint_for_model` from `tldw_chatbook.Chat.console_session_settings` alongside the existing `validate_console_session_settings` import (~line 60).

- [ ] **Step 4: Run tests**

Run: `python -m pytest Tests/Chat/test_console_session_settings.py Tests/UI/ -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_session_settings.py tldw_chatbook/Widgets/Console/console_settings_modal.py Tests/Chat/test_console_session_settings.py
git commit -m "feat(console): model-family thinking hints, warnings, and wire preview"
```

---

### Task 7: Full verification and task wrap-up

**Files:**
- Modify: `backlog/tasks/task-16812 - Console-thinking-levels-and-budget-for-local-providers.md`
- No production code changes expected.

**Interfaces:**
- Consumes: everything above.
- Produces: task marked Done with Implementation Notes; ACs checked.

- [ ] **Step 1: Run the full test suite and lint**

```bash
python -m pytest Tests/ -x -q
```
Expected: PASS. Fix any fallout before continuing (report honestly if pre-existing failures are unrelated — verify on `main` before claiming).

- [ ] **Step 2: Live verification against a real llama-server**

Per `backlog/docs/lessons-live-verification.md`, run the app against a real server. Requires: a Qwen3.8-27B GGUF and a current llama.cpp build.

```bash
llama-server -m /path/to/Qwen3.8-27B-*.gguf --jinja --host 127.0.0.1 --port 8080
```

Then `python3 -m tldw_chatbook.app`, open the Console, set provider to llama.cpp / local llama.cpp pointing at `http://127.0.0.1:8080`, and verify each item, capturing evidence (request logs or observed output) for the task notes:

1. Reasoning `low` vs `xhigh` on the same prompt observably changes thinking depth.
2. Budget 1024 truncates thinking well before `xhigh`'s natural depth.
3. With `--reasoning-format deepseek` added to the server flags: no `reasoning_content` leaks into the visible reply.
4. Without `--reasoning-format`: no `<think>` text in the visible reply (the filter's job).
5. Reasoning `none`: the model answers without thinking.
6. A response-prefill send still works (llama.cpp must not 400 on prefill + thinking controls).
7. If an older llama.cpp build (pre-b9982) is available: budget field present in the request does not cause a 400.

If the GGUF or a current llama-server is unavailable in this environment, do NOT mark the task Done — record exactly which items were verified live and which remain, and surface that in the final report. Optionally live-check `mlx_lm.server` and a llamafile server for `chat_template_kwargs` support; if unsupported, update the `local_mlx_lm` row to drop-and-log in `console_provider_support.py` and note it.

- [ ] **Step 3: Update the backlog task**

```bash
backlog task edit 16319 -s "In Progress" -a @Robert --plan "1. Wire-format table (ADR-066)\n2. Direct llama.cpp payload wiring\n3. Start-anchored think filter\n4. Wire filter into direct path\n5. Adapter-path param maps + shared builder + handlers\n6. Hints, warnings, placeholder, preview\n7. Full verification and wrap-up"
```

Then edit the task file: check every satisfied `- [ ]` AC to `- [x]`, and add an `## Implementation Notes` section (approach, deviations from plan, files touched, live-verification evidence). Only set status Done when ALL ACs including live verification are satisfied:

```bash
backlog task edit 16319 -s Done --notes "Implemented per ADR-066 spec/plan; live-verified against llama-server --jinja with Qwen3.8-27B"
```

- [ ] **Step 4: Final commit**

```bash
git add "backlog/tasks/task-16812 - Console-thinking-levels-and-budget-for-local-providers.md"
git commit -m "docs(backlog): complete task-16812 local thinking controls"
```

---

## Self-Review Notes (completed during planning)

- Spec coverage: wire table (T1), direct path + prefill precedence + stale-comment fix + aux inheritance (T2), start-anchored filter + split-case confirmation (T3/T4), adapter path incl. dual placement and custom-only-top-level (T5), hints/warnings/placeholder/preview/requirements note (T6), live verification + budget-±`--reasoning-format` checks + older-build tolerance (T7). Budget floor untouched (global ≥1024) — constraint honored by omission.
- Known deliberate deviations: none.
- Type consistency: `build_local_thinking_payload_fields(execution_key, reasoning_effort, thinking_budget_tokens)` used identically in T2/T5/T6; `StartAnchoredThinkFilter.feed/flush` in T3/T4; `console_settings_warnings(settings)` and `reasoning_effort_hint_for_model(model)` in T6.
