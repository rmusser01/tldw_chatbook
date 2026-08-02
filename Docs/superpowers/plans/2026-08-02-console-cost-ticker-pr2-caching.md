# Console Cost Ticker PR2 — Prompt-Caching Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Anthropic prompt caching by adding the per-turn message breakpoint (making the whole conversation prefix reusable turn over turn), a `[caching]` kill-switch, and a cache_control 400-degrade — and amend the spec whose PR2 premise went stale.

**Architecture:** Task-323 (already merged to dev, before this program's spec was written against an older snapshot) ships `cache_control` on the system block and last tool for **every** `chat_with_anthropic` caller, gated on `_anthropic_supports_caching(model)`. PR1 already maps Anthropic cache fields and OpenAI `cached_tokens` into the disjoint usage buckets. PR2 therefore: (1) amends the spec's stale premise; (2) adds the one missing breakpoint — last content block of the final message; (3) adds `[caching] anthropic_enabled` (default true) gating **all three** breakpoints at the provider level (matching shipped reality — all callers benefit; the spec's "console-gateway-only injection" note is part of the stale premise being amended); (4) adds a 400-naming-cache_control retry-once-without fallback; (5) pins prefix stability with tests.

**Tech Stack:** Python ≥3.11, requests (mocked in tests), pytest.

**Spec:** `Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md` — PR2 section, as amended by Task 1 of this plan.

## Global Constraints

- Worktree `/private/tmp/tldw-cost-ticker`, branch `feat/console-prompt-caching` (off origin/dev @ `814521ed4`). Venv exists: run pytest ONLY via `/private/tmp/tldw-cost-ticker/.venv/bin/pytest`, FOREGROUND. NEVER `git stash`.
- Caching must **never break sends**: a 4xx naming `cache_control` triggers exactly one retry without breakpoints + a diagnostic log; all other errors behave exactly as before.
- TTL is the 5-minute default only — never emit a `ttl` key (1-hour doubles the write premium; out of scope per spec).
- Breakpoint budget: system(1) + last-tool(1) + per-turn(1) = 3 of Anthropic's 4 allowed — assert the total in tests.
- `[caching] anthropic_enabled` defaults **true** when the section/key is absent — the 5 existing caching tests in `Tests/Chat/test_anthropic_native_tools.py:813-866` stub no config and MUST stay green unmodified.
- Non-caching models (`claude-2*`, `*instant*`) and non-Anthropic providers: zero behavior change.
- Config reads use the 3-arg form `get_cli_setting("caching", "anthropic_enabled", True)` (the 2-arg dotted form has the TASK-1771 default-dropping hazard).
- Line anchors below verified at `814521ed4` — re-locate by symbol if drifted.

---

### Task 1: Amend the spec's stale PR2 premise

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md` (Problem bullet 3 at ~line 25-27; the PR2 section)

**Interfaces:** none (docs).

- [ ] **Step 1: Amend the Problem section**

Replace the third Problem bullet ("No request ever sets `cache_control` — for Anthropic there is currently no cache to break...") with:

```markdown
- Anthropic caching is PARTIAL (amended 2026-08-02: task-323 landed system-block
  + last-tool `cache_control` for all callers after this spec's exploration
  snapshot): conversation HISTORY is never cached — no per-turn message
  breakpoint exists, so every send re-pays the full message history. There is
  no config kill-switch and no degrade path if an endpoint rejects
  `cache_control`. OpenAI-style providers cache automatically; PR1's adapters
  already account for it.
```

- [ ] **Step 2: Amend the PR2 section**

At the top of the "PR2 — Prompt-caching enablement" section, insert:

```markdown
> **Amended 2026-08-02 (PR2 planning):** task-323 shipped the system-block and
> last-tool breakpoints for ALL `chat_with_anthropic` callers (not console-only)
> before PR2 started, and PR1 shipped the OpenAI implicit-cache accounting.
> PR2's remaining scope: the per-turn message breakpoint, the
> `[caching] anthropic_enabled` toggle (provider-level, gating all three
> breakpoints, default on — matching shipped reality; the original
> "injected by the console gateway only" note described a world that no longer
> exists), the 4xx degrade, and prefix-stability tests. The
> "system string → block array" implementation note is already implemented.
```

Also strike (with `~~strikethrough~~` + "(amended: already shipped via task-323/PR1)") the OpenAI implicit-caching bullet and the system-string conversion implementation note.

- [ ] **Step 3: Commit**

```bash
git add Docs/superpowers/specs/2026-08-01-console-cost-ticker-design.md
git commit -m "docs(console): amend cost-ticker spec for task-323 caching reality"
```

---

### Task 2: `[caching] anthropic_enabled` kill-switch

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (new helper near `_anthropic_supports_caching` at :946; gate sites :1214 and :1255)
- Modify: `tldw_chatbook/config.py` (default TOML template — add a `[caching]` section near `[splash_screen]` at ~:2264)
- Test: `Tests/Chat/test_anthropic_native_tools.py` (append)

**Interfaces:**
- Consumes: `get_cli_setting(section, key, default)` from `tldw_chatbook.config` (config.py:4447).
- Produces (Tasks 3-4 rely on): `_anthropic_caching_enabled() -> bool` module function in LLM_API_Calls.py; a local `caching_active: bool` computed once inside `chat_with_anthropic` and used by every breakpoint site.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_anthropic_native_tools.py` (reuse its `_sent_anthropic` helper at :795-810; caching model string used by the existing tests):

```python
@patch("requests.Session.post")
def test_caching_disabled_via_config_strips_all_breakpoints(mock_post):
    """[caching] anthropic_enabled = false removes system AND tool AND
    message breakpoints; payload shape otherwise unchanged."""
    with patch(
        "tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting",
        side_effect=lambda section, key=None, default=None: (
            False if (section, key) == ("caching", "anthropic_enabled") else default
        ),
    ):
        sent = _sent_anthropic(
            mock_post, "claude-sonnet-4-6", system_message="be terse", tools=OPENAI_TOOLS
        )
    assert isinstance(sent["system"], str)  # string form, no block array
    assert "cache_control" not in json.dumps(sent)


@patch("requests.Session.post")
def test_caching_default_on_when_section_absent(mock_post):
    """No [caching] section -> enabled (the existing task-323 behavior)."""
    sent = _sent_anthropic(mock_post, "claude-sonnet-4-6", system_message="be terse")
    assert isinstance(sent["system"], list)
    assert sent["system"][-1]["cache_control"] == {"type": "ephemeral"}
```

Note: the patch target is the name as imported INTO LLM_API_Calls — Step 3 imports it at module top so `tldw_chatbook.LLM_Calls.LLM_API_Calls.get_cli_setting` is patchable. If `json` is not already imported in the test file, add it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /private/tmp/tldw-cost-ticker && .venv/bin/pytest Tests/Chat/test_anthropic_native_tools.py -k caching -v`
Expected: the two new tests FAIL (no gate exists; patch target missing); the 5 pre-existing caching tests PASS.

- [ ] **Step 3: Implement**

1. In `LLM_API_Calls.py`, add a module-level import `from tldw_chatbook.config import get_cli_setting` (check it isn't already imported; config.py imports must not cycle — `config.py` does not import `LLM_API_Calls`, verified by the existing `load_settings` usage).
2. Add next to `_anthropic_supports_caching` (:946):

```python
def _anthropic_caching_enabled() -> bool:
    """[caching].anthropic_enabled kill-switch for ALL Anthropic cache_control.

    Defaults to True when the section or key is absent (prompt caching is the
    shipped task-323 behavior; this gate only adds an opt-out). Any config
    read failure also defaults to True so a broken config file cannot
    silently change request shapes.

    Returns:
        True when cache_control breakpoints should be emitted.
    """
    try:
        return bool(get_cli_setting("caching", "anthropic_enabled", True))
    except Exception:
        return True
```

3. In `chat_with_anthropic`, compute once before the payload build (just above the `data = {` literal at :1207):

```python
    caching_active = _anthropic_supports_caching(current_model) and _anthropic_caching_enabled()
```

and change the two existing gate sites from `if _anthropic_supports_caching(current_model) and system_prompt:` (:1214) to `if caching_active and system_prompt:`, and `if _anthropic_supports_caching(current_model) and tools_payload:` (:1255) to `if caching_active and tools_payload:`.

4. In `config.py`'s default TOML template (near `[splash_screen]` at ~:2264), add:

```toml
[caching]
# Anthropic prompt caching (cache_control breakpoints on system prompt, tool
# list, and the latest message). Cache writes bill at 1.25x input and reads at
# ~0.1x, so multi-turn chat wins after two sends inside the 5-minute TTL.
# Set false to disable all Anthropic cache_control emission.
anthropic_enabled = true
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/Chat/test_anthropic_native_tools.py -v`
Expected: all PASS including the 5 pre-existing caching tests unmodified.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py tldw_chatbook/config.py Tests/Chat/test_anthropic_native_tools.py
git commit -m "feat(anthropic): [caching] anthropic_enabled kill-switch for cache_control"
```

---

### Task 3: Per-turn message breakpoint

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (insertion at :1266, after `data["output_config"] = output_config`, before `api_url = (`)
- Test: `Tests/Chat/test_anthropic_native_tools.py` (append)

**Interfaces:**
- Consumes: `caching_active` local from Task 2.
- Produces: the outbound body's final message carries `cache_control` on its last content block; total breakpoints ≤ 4 in every combination.

- [ ] **Step 1: Write the failing tests**

```python
def _count_cache_controls(obj):
    """Count every cache_control key anywhere in the payload."""
    if isinstance(obj, dict):
        return ("cache_control" in obj) + sum(
            _count_cache_controls(v) for v in obj.values()
        )
    if isinstance(obj, list):
        return sum(_count_cache_controls(item) for item in obj)
    return 0


@patch("requests.Session.post")
def test_caching_model_marks_last_message_block(mock_post):
    """The final message's LAST content block carries the per-turn breakpoint;
    earlier messages and earlier blocks of the final message carry none."""
    sent = _sent_anthropic(
        mock_post,
        "claude-sonnet-4-6",
        messages_payload=[
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
    )
    messages = sent["messages"]
    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert _count_cache_controls(messages[:-1]) == 0
    assert _count_cache_controls(messages[-1]["content"][:-1]) == 0


@patch("requests.Session.post")
def test_breakpoint_budget_never_exceeds_four(mock_post):
    """system + last-tool + per-turn = 3 total, within Anthropic's 4-cap."""
    sent = _sent_anthropic(
        mock_post,
        "claude-sonnet-4-6",
        system_message="be terse",
        tools=OPENAI_TOOLS,
        messages_payload=[{"role": "user", "content": "hi"}],
    )
    assert _count_cache_controls(sent) == 3


@patch("requests.Session.post")
def test_non_caching_model_gets_no_message_breakpoint(mock_post):
    sent = _sent_anthropic(
        mock_post, "claude-2.1", messages_payload=[{"role": "user", "content": "hi"}]
    )
    assert _count_cache_controls(sent) == 0


@patch("requests.Session.post")
def test_message_breakpoint_never_emits_ttl_key(mock_post):
    """5-minute default only — a ttl key would double the write premium."""
    sent = _sent_anthropic(
        mock_post,
        "claude-sonnet-4-6",
        messages_payload=[{"role": "user", "content": "hi"}],
    )
    assert sent["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "ttl" not in json.dumps(sent)
```

Note: `_sent_anthropic` hardcodes `messages_payload=[{"role": "user", "content": "hi"}]` — extend the helper to accept an optional `messages_payload=None` parameter defaulting to that list (backward-compatible; existing callers unchanged).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/Chat/test_anthropic_native_tools.py -k "message_breakpoint or budget or marks_last" -v`
Expected: FAIL — no cache_control on messages today.

- [ ] **Step 3: Implement**

At `LLM_API_Calls.py:1266` (immediately after the `output_config` assignment, before `api_url = (` — payload complete, ahead of the streaming branch so both modes get it):

```python
    if (
        caching_active
        and anthropic_messages
        and isinstance(anthropic_messages[-1].get("content"), list)
        and anthropic_messages[-1]["content"]
    ):
        # Per-turn breakpoint (cost-ticker PR2): mark the last content block
        # of the final message so the WHOLE conversation prefix becomes a
        # reusable cache entry next turn -- the task-323 system/tools
        # breakpoints alone never cache message history. Budget:
        # system(1) + last-tool(1) + this(1) = 3 of the 4 allowed.
        # Fresh dict so no caller-held block is mutated (same rule as the
        # tools breakpoint above).
        last_content = anthropic_messages[-1]["content"]
        last_content[-1] = {
            **last_content[-1],
            "cache_control": {"type": "ephemeral"},
        }
```

(The task-263 conversion always emits list-form `content`, so the isinstance guard is belt-and-suspenders for exotic future branches, not a live path.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_anthropic_streaming_usage.py -v`
Expected: all PASS (streaming-usage file is the adjacency regression check).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_anthropic_native_tools.py
git commit -m "feat(anthropic): per-turn cache_control breakpoint on the final message"
```

---

### Task 4: 400-naming-cache_control degrade (retry once without breakpoints)

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py` (the session/post block at :1282-1301; helper next to `_anthropic_caching_enabled`)
- Test: `Tests/Chat/test_anthropic_caching_degrade.py` (new)

**Interfaces:**
- Consumes: payloads that may carry cache_control (Tasks 2-3).
- Produces: `_without_cache_control(obj)` and `_contains_cache_control(obj)` module helpers (recursive, pure); the post block checks status INSIDE the `with` (the current code calls `raise_for_status()` after the session closes — the retry must happen while the session is open).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_anthropic_caching_degrade.py
"""A 400 naming cache_control retries ONCE without breakpoints; any other
error behaves exactly as before (caching must never break sends)."""

import json
from unittest.mock import Mock, patch

import pytest
import requests

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import (
    _contains_cache_control,
    _without_cache_control,
)


def _ok_response(text="ok"):
    response = Mock()
    response.status_code = 200
    response.raise_for_status = Mock()
    response.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    return response


def _bad_response(body):
    response = Mock()
    response.status_code = 400
    response.text = body
    response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=response
    )
    return response


@patch("requests.Session.post")
def test_400_naming_cache_control_retries_stripped(mock_post):
    bad = _bad_response('{"error": {"message": "cache_control is not supported"}}')
    mock_post.side_effect = [bad, _ok_response()]

    chat_api_call(
        "anthropic",
        messages_payload=[{"role": "user", "content": "hi"}],
        api_key="test-key",
        model="claude-sonnet-4-6",
        system_message="be terse",
        streaming=False,
    )

    assert mock_post.call_count == 2
    retry_body = mock_post.call_args_list[1][1]["json"]
    assert "cache_control" not in json.dumps(retry_body)
    # system degrades from block-array back to plain blocks sans cache keys
    first_body = mock_post.call_args_list[0][1]["json"]
    assert "cache_control" in json.dumps(first_body)


@patch("requests.Session.post")
def test_400_not_naming_cache_control_raises_unretried(mock_post):
    bad = _bad_response('{"error": {"message": "max_tokens too large"}}')
    mock_post.return_value = bad

    with pytest.raises(Exception):
        chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            model="claude-sonnet-4-6",
            streaming=False,
        )
    assert mock_post.call_count == 1


@patch("requests.Session.post")
def test_no_retry_when_payload_has_no_cache_control(mock_post):
    """Non-caching model: even a cache_control-naming 400 must not retry
    (nothing to strip -- the guard requires the param in OUR payload)."""
    bad = _bad_response('{"error": {"message": "cache_control invalid"}}')
    mock_post.return_value = bad

    with pytest.raises(Exception):
        chat_api_call(
            "anthropic",
            messages_payload=[{"role": "user", "content": "hi"}],
            api_key="test-key",
            model="claude-2.1",
            streaming=False,
        )
    assert mock_post.call_count == 1


def test_without_cache_control_strips_recursively():
    data = {
        "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ],
            }
        ],
        "tools": [{"name": "t", "cache_control": {"type": "ephemeral"}}],
        "max_tokens": 5,
    }
    stripped = _without_cache_control(data)
    assert "cache_control" not in json.dumps(stripped)
    assert stripped["messages"][0]["content"][0]["text"] == "hi"
    assert stripped["max_tokens"] == 5
    assert _contains_cache_control(data) is True
    assert _contains_cache_control(stripped) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest Tests/Chat/test_anthropic_caching_degrade.py -v`
Expected: ImportError on the helpers, then retry-count failures.

- [ ] **Step 3: Implement**

1. Module helpers next to `_anthropic_caching_enabled`:

```python
def _without_cache_control(obj: Any) -> Any:
    """Deep-copy ``obj`` with every ``cache_control`` key removed.

    Args:
        obj: Any JSON-shaped structure (dicts/lists/scalars).

    Returns:
        The same structure minus all ``cache_control`` entries.
    """
    if isinstance(obj, dict):
        return {
            key: _without_cache_control(value)
            for key, value in obj.items()
            if key != "cache_control"
        }
    if isinstance(obj, list):
        return [_without_cache_control(item) for item in obj]
    return obj


def _contains_cache_control(obj: Any) -> bool:
    """True when any nested dict carries a ``cache_control`` key."""
    if isinstance(obj, dict):
        return "cache_control" in obj or any(
            _contains_cache_control(value) for value in obj.values()
        )
    if isinstance(obj, list):
        return any(_contains_cache_control(item) for item in obj)
    return False
```

2. Restructure the post block (:1282-1301). The current shape closes the session BEFORE `raise_for_status()` — move the degrade check inside the `with` (mirroring `chat_with_openai`'s stream_options degrade at :715-742):

```python
        with requests.Session() as session:
            session.mount("https://", adapter)
            response = session.post(
                api_url,
                headers=headers,
                json=data,
                stream=current_streaming,
                timeout=180,
            )
            if (
                response.status_code == 400
                and _contains_cache_control(data)
                and "cache_control" in (response.text or "")
            ):
                # Caching must never break sends (cost-ticker PR2): odd
                # proxies/gateways can reject cache_control. Retry exactly
                # once without any breakpoints; every other error path is
                # untouched. Reading .text here is safe -- it is the error
                # body, not a stream.
                logger.warning(
                    "Anthropic: endpoint rejected cache_control; retrying without prompt caching."
                )
                response = session.post(
                    api_url,
                    headers=headers,
                    json=_without_cache_control(data),
                    stream=current_streaming,
                    timeout=180,
                )
        response.raise_for_status()
```

(`urllib3 Retry`'s status_forcelist covers only 429/5xx, so the 400 reaches this check untouched. `response.iter_lines()` in the streaming branch still works on the retried response object — it was created with `stream=current_streaming`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest Tests/Chat/test_anthropic_caching_degrade.py Tests/Chat/test_anthropic_native_tools.py Tests/Chat/test_anthropic_streaming_usage.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/LLM_Calls/LLM_API_Calls.py Tests/Chat/test_anthropic_caching_degrade.py
git commit -m "feat(anthropic): degrade gracefully when an endpoint rejects cache_control"
```

---

### Task 5: Prefix-stability pins

**Files:**
- Test: `Tests/Chat/test_anthropic_prefix_stability.py` (new; test-only task)

**Interfaces:**
- Consumes: the full payload build from Tasks 2-3 via the `_call_anthropic`-style mock idiom.

- [ ] **Step 1: Write the tests (they should PASS immediately — they pin invariants)**

```python
# Tests/Chat/test_anthropic_prefix_stability.py
"""Cache-prefix stability pins (cost-ticker PR2).

Anthropic caching is a byte-exact prefix match over tools -> system ->
messages. These tests pin that consecutive turn builds keep the shared
prefix identical: same system bytes, same tool bytes, and message history
content-identical except (a) the appended turn and (b) the per-turn
cache_control marker, which MOVES to the newest message each build
(metadata designating the cache boundary -- earlier content bytes stay
identical, which is what the server matches on).
"""

import json
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.LLM_Calls.LLM_API_Calls import _without_cache_control


def _sent_body(mock_post, messages):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.json.return_value = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-x",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    mock_post.return_value = mock_response
    chat_api_call(
        "anthropic",
        messages_payload=messages,
        api_key="test-key",
        model="claude-sonnet-4-6",
        system_message="You are terse.\n\nAlways answer in one line.",
        streaming=False,
    )
    return mock_post.call_args[1]["json"]


TURN_1 = [{"role": "user", "content": "first question"}]
TURN_2 = TURN_1 + [
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
]


@patch("requests.Session.post")
def test_system_bytes_identical_across_consecutive_builds(mock_post):
    body_1 = _sent_body(mock_post, TURN_1)
    body_2 = _sent_body(mock_post, TURN_2)
    assert json.dumps(body_1["system"], sort_keys=True) == json.dumps(
        body_2["system"], sort_keys=True
    )


@patch("requests.Session.post")
def test_history_prefix_content_identical_across_builds(mock_post):
    """Build 2's earlier messages == build 1's messages, modulo the moved
    per-turn marker (strip cache_control from both sides before comparing)."""
    body_1 = _sent_body(mock_post, TURN_1)
    body_2 = _sent_body(mock_post, TURN_2)
    prefix_2 = _without_cache_control(body_2["messages"][: len(body_1["messages"])])
    stripped_1 = _without_cache_control(body_1["messages"])
    assert prefix_2 == stripped_1


@patch("requests.Session.post")
def test_marker_sits_only_on_newest_message_each_build(mock_post):
    body_2 = _sent_body(mock_post, TURN_2)
    dumped_history = json.dumps(body_2["messages"][:-1])
    assert "cache_control" not in dumped_history
    assert body_2["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }


@patch("requests.Session.post")
def test_no_volatile_keys_reach_the_wire(mock_post):
    """No timestamps/uuids/internal annotations in the request body."""
    body = _sent_body(mock_post, TURN_2)
    dumped = json.dumps(body)
    for forbidden in ("_native_message_id", "timestamp", "uuid"):
        assert forbidden not in dumped
```

- [ ] **Step 2: Run and confirm all pass**

Run: `.venv/bin/pytest Tests/Chat/test_anthropic_prefix_stability.py -v`
Expected: all PASS. If any fails, that is a REAL prefix-stability bug found by this task — stop and report rather than adjusting the assertion.

- [ ] **Step 3: Commit**

```bash
git add Tests/Chat/test_anthropic_prefix_stability.py
git commit -m "test(anthropic): pin cache-prefix stability across consecutive turns"
```

---

### Task 6: Gates + push + PR

**Files:** none (verification only).

- [ ] **Step 1: Touched-area suites**

Run: `.venv/bin/pytest Tests/Chat/ Tests/LLM_Calls/ -q`
Expected: green (Tests/Chat was 3029 passed at branch time).

- [ ] **Step 2: Config regression**

Run: `.venv/bin/pytest Tests/Config/ -q 2>/dev/null || .venv/bin/pytest Tests/ -k "config and caching" -q`
Expected: green (locate the config-template tests if any assert the default TOML's section list — update in lockstep if one pins sections).

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feat/console-prompt-caching
gh pr create --base dev --title "feat(anthropic): complete prompt caching (cost ticker PR2)" --body "$(cat <<'EOF'
PR2 of the Console cost-ticker program (spec amended in-PR: the original PR2 premise predated task-323, which already shipped system+tool breakpoints).

- Per-turn cache_control breakpoint on the final message — conversation history becomes a reusable cache prefix (task-323's system/tools breakpoints never cached history). Budget: 3 of 4 allowed breakpoints.
- `[caching] anthropic_enabled` kill-switch (default on; absent section = shipped behavior; the 5 pre-existing task-323 tests pass unmodified).
- Degrade path: a 400 naming cache_control retries exactly once with all breakpoints stripped — caching can never break sends.
- Prefix-stability pins: system/tool bytes identical across consecutive turn builds; history content-identical modulo the moving per-turn marker; no volatile keys on the wire.
- Accounting rides PR1 unchanged (cache_read/cache_write buckets, usage_json, pricing catalog cache rates).

Live acceptance (needs a real key, per spec): `cache_read_input_tokens > 0` on the second consecutive Console send — visible in the PR1 usage records.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Spec-coverage checklist (PR2 section, as amended by Task 1)

| Requirement | Task |
|---|---|
| Spec premise corrected (task-323 reality) | 1 |
| Per-turn breakpoint, incremental prefix growth | 3 |
| 5-min TTL only, never a ttl key | 3 (test) |
| `[caching] anthropic_enabled` default on | 2 |
| Never-break-sends 4xx degrade + diagnostic | 4 |
| Sub-minimum prefixes silently uncached | nothing to do (server behavior; PR3 shows ground truth) |
| Prefix-stability audit + byte-identical consecutive builds | 5 |
| Leading-system extraction stability | 5 (system-bytes test rides the real extraction path via chat_api_call) |
| OpenAI implicit accounting | already shipped in PR1 (amended out of scope) |
| Cache accounting into PR1 buckets | already shipped (provider_usage.py:177-185; test_cache_usage_metrics.py) |
| Legacy-path unaffected | 2 (default-on test) + existing task-323 suite unmodified |
| Live acceptance: cache_read > 0 on 2nd send | 6 (PR body; user-verifiable with a real key) |
