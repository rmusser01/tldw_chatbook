# Doc-pass + Broken chat_with_provider Call Repair — Implementation Plan (TASK-333 + TASK-334)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the four runtime-broken `chat_with_provider` call sites by repointing them to the real `chat_api_call` dispatcher (with response-content extraction), and comprehensively correct the stale/wrong claims in CLAUDE.md.

**Architecture:** A small shared `extract_response_content(resp)` helper (in `Chat/Chat_Functions.py`, next to `chat_api_call`) normalizes the dispatcher's OpenAI-shaped dict to the assistant text string each broken site expects. Each site drops its dead `from ..LLM_Calls.LLM_API_Calls import chat_with_provider` import and calls `chat_api_call(..., streaming=False)` + the helper. Then a documentation pass rewrites the verified-wrong CLAUDE.md sections, anchored to stable symbol/file names.

**Tech Stack:** Python 3.11, pytest, unittest.mock. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-24-docs-and-dead-import-cleanup-design.md` (committed `bfb6e0c86`). Read it for rationale; THIS plan carries the exact code.

## Global Constraints

1. **`chat_api_call` is the dispatcher** at `tldw_chatbook/Chat/Chat_Functions.py` (function `chat_api_call`, ~line 671). It returns the provider handler's **full OpenAI-shaped dict** for non-streaming. Signature essentials: `chat_api_call(api_endpoint, messages_payload, api_key=None, temp=None, system_message=None, streaming=None, model=None, max_tokens=None, ...)`.
2. **Kwarg mapping at every repoint:** `provider=`→`api_endpoint=` (first positional), `messages=`→`messages_payload=`, `temperature=`→`temp=`; `model=`/`max_tokens=`/`api_key=` unchanged; **drop** `timeout=`; **add `streaming=False`** explicitly (leaving it None makes the return shape handler-dependent).
3. **Preserve each site's existing message assembly** (e.g. MCP's system-prompt-in-`messages`-list). Do NOT move the system prompt to `system_message=`. The repoint fixes dispatch only, not message construction.
4. **`MCP/tools.py` is left untouched** — its local `chat_with_provider`/`save_conversation_from_messages` `NotImplementedError` stubs are intentional (loudly-failing, caught). `MCPTools.chat_with_character` also depends on the dead `save_conversation_from_messages`, so it is NOT repointed here — it goes to a follow-up task (Task 6).
5. **Line numbers in this plan are as-of `origin/dev` @ `f939cf151`** — re-verify with `grep -n` before editing; the target text (not the line number) is authoritative.
6. **Docs anchor to stable references** (symbol names, file paths), never line numbers or bare version integers.
7. Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-docs-cleanup` (branch `feat/docs-and-dead-import-cleanup`); tests via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` FROM the worktree. Never touch the main checkout. `git add` only the files each task lists — never `-A`.
8. Edit **origin/dev's** CLAUDE.md (the worktree copy), which already differs from the main checkout's older copy.

**Baseline check:** before Task 1, run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/ Tests/Tools/ -q` (if those dirs exist) and note any pre-existing failures — report them rather than fixing.

---

### Task 1: Shared response-content extraction helper

**Files:**
- Modify: `tldw_chatbook/Chat/Chat_Functions.py` (add `extract_response_content`, placed just after `chat_api_call`'s definition or near it)
- Test: `Tests/Chat/test_extract_response_content.py` (create)

**Interfaces:**
- Produces: `extract_response_content(resp: Any) -> str` — returns the assistant text from a `chat_api_call` non-streaming result. Handles the OpenAI shape (`resp["choices"][0]["message"]["content"]`), a flat `{"content": ...}` shape (what `code_audit_tool` historically expected), an empty/malformed `choices`, and a non-dict input. Never raises.

- [ ] **Step 1: Write the failing test**

Create `Tests/Chat/test_extract_response_content.py`:

```python
from tldw_chatbook.Chat.Chat_Functions import extract_response_content


def test_openai_shape():
    resp = {"choices": [{"message": {"role": "assistant", "content": "Hello there"}}]}
    assert extract_response_content(resp) == "Hello there"


def test_flat_content_shape():
    # Shape code_audit_tool historically read via resp.get("content")
    assert extract_response_content({"content": "flat text"}) == "flat text"


def test_openai_shape_wins_over_flat():
    resp = {"choices": [{"message": {"content": "nested"}}], "content": "flat"}
    assert extract_response_content(resp) == "nested"


def test_empty_choices_list_no_indexerror():
    assert extract_response_content({"choices": []}) == ""


def test_choices_missing_message():
    assert extract_response_content({"choices": [{}]}) == ""


def test_null_content_coerced_to_empty_string():
    resp = {"choices": [{"message": {"content": None}}]}
    assert extract_response_content(resp) == ""


def test_non_dict_input():
    assert extract_response_content("already a string") == "already a string"
    assert extract_response_content(None) == ""


def test_missing_everything():
    assert extract_response_content({}) == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_extract_response_content.py -q`
Expected: ImportError (`cannot import name 'extract_response_content'`).

- [ ] **Step 3: Add the helper**

In `tldw_chatbook/Chat/Chat_Functions.py`, add near `chat_api_call` (e.g. immediately after its `def`):

```python
def extract_response_content(resp: Any) -> str:
    """Extract the assistant text from a non-streaming ``chat_api_call`` result.

    ``chat_api_call`` returns the provider handler's full response. For the
    common OpenAI-shaped dict that is ``resp["choices"][0]["message"]["content"]``;
    some paths return a flat ``{"content": ...}``. Returns "" for any missing/
    malformed/None content. Never raises.

    Args:
        resp: A ``chat_api_call`` non-streaming return value (dict), or any value.

    Returns:
        The assistant text, or "" when it cannot be found.
    """
    if not isinstance(resp, dict):
        return resp if isinstance(resp, str) else ""
    choices = resp.get("choices") or []
    if choices:
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")
        if content:
            return content
    return resp.get("content") or ""
```

(`Any` is already imported in this module via `from typing import ... Any`; if not, add it to the existing typing import.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chat/test_extract_response_content.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/Chat_Functions.py Tests/Chat/test_extract_response_content.py
git commit -m "feat(chat): add extract_response_content helper for non-streaming responses"
```

---

### Task 2: Repoint MCP/server.py chat_with_llm

**Files:**
- Modify: `tldw_chatbook/MCP/server.py` (`_register_tools` import ~line 169; `chat_with_llm` call ~line 194)
- Test: `Tests/MCP/test_server_chat_repoint.py` (create; make `Tests/MCP/__init__.py` if the dir/package doesn't exist)

**Interfaces:**
- Consumes: `chat_api_call`, `extract_response_content` from `tldw_chatbook.Chat.Chat_Functions`.

- [ ] **Step 1: Write the failing test**

First check the package: `ls Tests/MCP/ 2>/dev/null` — if absent, create `Tests/MCP/__init__.py` (empty). Then create `Tests/MCP/test_server_chat_repoint.py`:

```python
"""TASK-334: MCP server chat_with_llm repointed to chat_api_call (was ImportError)."""

import importlib

import pytest


def test_server_module_imports_without_error():
    # The dead `from ..LLM_Calls.LLM_API_Calls import chat_with_provider` used to
    # crash TldwMCPServer.__init__ -> _register_tools at construction time.
    mod = importlib.import_module("tldw_chatbook.MCP.server")
    assert mod is not None


def test_no_dead_chat_with_provider_import_in_server():
    import tldw_chatbook.MCP.server as srv
    src = open(srv.__file__, encoding="utf-8").read()
    assert "import chat_with_provider" not in src
    assert "chat_api_call" in src
```

Note: fully exercising the `chat_with_llm` MCP tool requires constructing `TldwMCPServer` (needs DBs / the `fastmcp` dependency) — that is heavier than this task needs. The import-resolution + source guard above prove the ImportError is gone and the repoint landed; the call-shape correctness is covered structurally by Task 1's helper test plus the Task 4 repo-wide grep. If `fastmcp` and DB fixtures are readily available in the env, optionally add a construction test, but do not build heavy fixtures for it.

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_server_chat_repoint.py -q`
Expected: `test_no_dead_chat_with_provider_import_in_server` FAILS (the dead import string is still present). `test_server_module_imports_without_error` may pass already (the dead import is *inside* `_register_tools`, executed at construction, not at module import) — that's fine; it guards against regressions.

- [ ] **Step 3: Repoint the code**

In `tldw_chatbook/MCP/server.py`, in `_register_tools`, replace the import line:

```python
        from ..LLM_Calls.LLM_API_Calls import chat_with_provider
```
with:
```python
        from ..Chat.Chat_Functions import chat_api_call, extract_response_content
```

Then replace the `chat_with_llm` call block. Current:
```python
                response = await asyncio.to_thread(
                    chat_with_provider,
                    api_key=api_key,
                    provider=provider,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                return {
                    "response": response,
                    "conversation_id": conversation_id or "new_conversation",
                }
```
New:
```python
                raw = await asyncio.to_thread(
                    chat_api_call,
                    api_endpoint=provider,
                    messages_payload=messages,
                    api_key=api_key,
                    model=model,
                    temp=temperature,
                    max_tokens=max_tokens,
                    streaming=False,
                )
                response = extract_response_content(raw)

                return {
                    "response": response,
                    "conversation_id": conversation_id or "new_conversation",
                }
```
(Message assembly above — the `messages` list with optional system entry — is unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_server_chat_repoint.py -q`
Expected: both pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/server.py Tests/MCP/test_server_chat_repoint.py
git commit -m "fix(mcp): repoint chat_with_llm to chat_api_call (was ImportError crashing server init) [TASK-334]"
```

---

### Task 3: Repoint Tools/code_audit_tool.py

**Files:**
- Modify: `tldw_chatbook/Tools/code_audit_tool.py` (`_request_llm_analysis`, import ~line 134, call ~line 137)
- Test: `Tests/Tools/test_code_audit_repoint.py` (create; add `Tests/Tools/__init__.py` if needed)

**Interfaces:**
- Consumes: `chat_api_call`, `extract_response_content`.

- [ ] **Step 1: Write the failing test**

Check `ls Tests/Tools/ 2>/dev/null`; create `Tests/Tools/__init__.py` if absent. Create `Tests/Tools/test_code_audit_repoint.py`:

```python
"""TASK-334: code_audit_tool._request_llm_analysis repointed to chat_api_call."""

import asyncio
from unittest.mock import patch

from tldw_chatbook.Tools.code_audit_tool import CodeAuditTool


def test_request_llm_analysis_calls_chat_api_call_and_extracts_content():
    tool = CodeAuditTool()
    captured = {}

    def fake_chat_api_call(**kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "RISK: HIGH — hardcoded return"}}]}

    with patch(
        "tldw_chatbook.Chat.Chat_Functions.chat_api_call", side_effect=fake_chat_api_call
    ):
        result = asyncio.run(tool._request_llm_analysis("analyze this code"))

    assert result == "RISK: HIGH — hardcoded return"
    assert captured["api_endpoint"] == "anthropic"
    assert captured["messages_payload"] == [
        {"role": "user", "content": "analyze this code"}
    ]
    assert captured["streaming"] is False
    assert "timeout" not in captured  # dropped — chat_api_call has no timeout param


def test_no_dead_import_in_code_audit():
    import tldw_chatbook.Tools.code_audit_tool as m
    src = open(m.__file__, encoding="utf-8").read()
    assert "import chat_with_provider" not in src
```

(If `CodeAuditTool()` requires constructor args, check its `__init__` and pass what's needed — it takes none per current code. The patch targets `tldw_chatbook.Chat.Chat_Functions.chat_api_call`; the implementation must import it from there for the patch to bind — see Step 3.)

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_code_audit_repoint.py -q`
Expected: FAIL (import ImportError currently caught → returns "LLM analysis unavailable: ..."; and the dead-import string is present).

- [ ] **Step 3: Repoint the code**

In `tldw_chatbook/Tools/code_audit_tool.py`, `_request_llm_analysis`, replace:
```python
            from ..LLM_Calls.LLM_API_Calls import chat_with_provider

            # Use a fast model for analysis
            response = await asyncio.to_thread(
                chat_with_provider,
                prompt=prompt,
                model="claude-3-haiku",  # Fast model for quick analysis
                provider="anthropic",
                max_tokens=500,
                temperature=0.1,  # Low temperature for consistent analysis
                timeout=30,
            )

            return response.get("content", "Analysis failed")
```
with:
```python
            from ..Chat.Chat_Functions import chat_api_call, extract_response_content

            # Use a fast model for analysis
            raw = await asyncio.to_thread(
                chat_api_call,
                api_endpoint="anthropic",
                messages_payload=[{"role": "user", "content": prompt}],
                model="claude-3-haiku-20240307",  # fast model for quick analysis
                temp=0.1,  # low temperature for consistent analysis
                max_tokens=500,
                streaming=False,
            )

            return extract_response_content(raw) or "Analysis failed"
```
Model id: `claude-3-haiku` was a stale short id. `claude-3-haiku-20240307` is the canonical Anthropic haiku id. Before finalizing, grep the repo for an existing anthropic haiku id (`grep -rn "claude-3.*haiku" tldw_chatbook/`) and use whatever form the codebase already uses if it differs; the exact id only affects a live call (tests mock `chat_api_call`), but pick a real one since we're restoring the feature.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_code_audit_repoint.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/code_audit_tool.py Tests/Tools/test_code_audit_repoint.py
git commit -m "fix(tools): repoint code_audit deception analysis to chat_api_call [TASK-334]"
```

---

### Task 4: Repoint the two Tools_Settings_Window sites + repo-wide guard

**Files:**
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py` (`_test_chat_connection` ~4166/4168; `_test_all_api_keys` ~4303/4305)
- Test: `Tests/Tools/test_no_dead_chat_with_provider_imports.py` (create)

**Interfaces:**
- Consumes: `chat_api_call`, `extract_response_content`.

- [ ] **Step 1: Write the failing test (repo-wide guard + import resolution)**

Create `Tests/Tools/test_no_dead_chat_with_provider_imports.py`:

```python
"""TASK-334: no dead `chat_with_provider` imports from LLM_API_Calls remain."""

import importlib
import pathlib
import re

import pytest

REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "tldw_chatbook"
DEAD_IMPORT = re.compile(
    r"import\s+chat_with_provider|from\s+[.\w]*LLM_API_Calls\s+import\s+chat_with_provider"
)


def test_no_dead_chat_with_provider_import_anywhere():
    offenders = []
    for py in REPO_SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if DEAD_IMPORT.search(text):
            offenders.append(str(py.relative_to(REPO_SRC)))
    # MCP/tools.py defines a LOCAL stub (def chat_with_provider) — that is NOT an
    # import and must not match; if it does, tighten the regex.
    assert offenders == [], f"dead chat_with_provider imports remain: {offenders}"


@pytest.mark.parametrize(
    "module",
    [
        "tldw_chatbook.MCP.server",
        "tldw_chatbook.Tools.code_audit_tool",
        "tldw_chatbook.UI.Tools_Settings_Window",
    ],
)
def test_touched_modules_import_clean(module):
    assert importlib.import_module(module) is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_no_dead_chat_with_provider_imports.py -q`
Expected: `test_no_dead_chat_with_provider_import_anywhere` FAILS listing `UI/Tools_Settings_Window.py` (Tasks 2-3 already fixed server + code_audit; this file still has 2 dead imports). If `UI.Tools_Settings_Window` fails to import for an unrelated reason (heavy Textual imports), note it — but it should import.

- [ ] **Step 3: Repoint `_test_chat_connection`**

Replace:
```python
            from ..LLM_Calls.LLM_API_Calls import chat_with_provider

            test_response = await self.run_worker(
                lambda: chat_with_provider(
                    provider=provider,
                    model=model,
                    messages=[
                        {"role": "user", "content": "Test connection. Reply with 'OK'."}
                    ],
                    temperature=0.1,
                    max_tokens=10,
                ),
                thread=True,
                exclusive=True,
            )

            if test_response and "OK" in str(test_response).upper():
```
with:
```python
            from ..Chat.Chat_Functions import chat_api_call, extract_response_content

            raw = await self.run_worker(
                lambda: chat_api_call(
                    api_endpoint=provider,
                    messages_payload=[
                        {"role": "user", "content": "Test connection. Reply with 'OK'."}
                    ],
                    model=model,
                    temp=0.1,
                    max_tokens=10,
                    streaming=False,
                ),
                thread=True,
                exclusive=True,
            )
            test_response = extract_response_content(raw)

            if test_response and "OK" in test_response.upper():
```

- [ ] **Step 4: Repoint `_test_all_api_keys`**

Replace:
```python
                    from ..LLM_Calls.LLM_API_Calls import chat_with_provider

                    test_response = await self.run_worker(
                        lambda: chat_with_provider(
                            provider=provider,
                            model=model,
                            messages=[{"role": "user", "content": "Test. Reply OK."}],
                            temperature=0.1,
                            max_tokens=10,
                        ),
                        thread=True,
                        exclusive=True,
                    )

                    if test_response:
                        results.append(f"✅ {provider}: Working")
```
with:
```python
                    from ..Chat.Chat_Functions import chat_api_call, extract_response_content

                    raw = await self.run_worker(
                        lambda: chat_api_call(
                            api_endpoint=provider,
                            messages_payload=[{"role": "user", "content": "Test. Reply OK."}],
                            model=model,
                            temp=0.1,
                            max_tokens=10,
                            streaming=False,
                        ),
                        thread=True,
                        exclusive=True,
                    )

                    if extract_response_content(raw):
                        results.append(f"✅ {provider}: Working")
```
(The `provider`-loop-variable-in-lambda closure behavior is unchanged from the original — not introduced here.)

- [ ] **Step 5: Run test to verify it passes**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_no_dead_chat_with_provider_imports.py -q`
Expected: all pass (zero offenders; all three modules import).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Tools_Settings_Window.py Tests/Tools/test_no_dead_chat_with_provider_imports.py
git commit -m "fix(ui): repoint Settings connection-test buttons to chat_api_call [TASK-334]"
```

---

### Task 5: Comprehensive CLAUDE.md doc-pass

**Files:**
- Modify: `CLAUDE.md` (worktree copy = origin/dev's)

No automated tests (docs). Deliverable = each correction applied + verified against code + a final path-existence spot-check. **Re-verify every claim with `grep -n`/`git show origin/dev:` before writing** — line numbers below are as-of origin/dev @ `f939cf151` and may drift. Anchor corrected text to stable symbol/file names, never line numbers or bare version integers.

Apply these corrections (each is one AC of TASK-333). README.md was checked — it carries **no** stale `chat_with_provider`/schema/tool claims, so **leave README untouched** (verified 2026-07-24).

- [ ] **Step 1: Schema version** — CLAUDE.md line ~68 (`schema v7`) and line ~161 (`Schema v7 adds tool messages`).
  - Line ~68 `- **`ChaChaNotes_DB.py`** - Main DB (...), schema v7` → `- **`ChaChaNotes_DB.py`** - Main DB (conversations, messages, characters, notes); current schema version is `_CURRENT_SCHEMA_VERSION` in this file (24 at time of writing) — bump it and add a migration when changing schema`
  - Line ~161 `- Schema v7 adds tool messages` → `- Tool messages are persisted in the ChaChaNotes DB (see `_CURRENT_SCHEMA_VERSION` for the current schema version)`
  - Verify: `git show origin/dev:tldw_chatbook/DB/ChaChaNotes_DB.py | grep -n "_CURRENT_SCHEMA_VERSION ="` — confirm the number, and phrase so it won't re-rot.

- [ ] **Step 2: Tool Calling status** — line ~164 (`- Status: Detection works, execution pending`).
  - → `- Status: detection AND execution implemented — `ToolExecutor.execute_tool_call()` runs tools; wired into `Event_Handlers/worker_events.py` and `Event_Handlers/Chat_Events/chat_streaming_events.py``
  - Also line ~163 (`- Provider parsing implemented`) — leave if accurate; the "execution pending" line is the wrong one.

- [ ] **Step 3: New-Tool recipe** — lines ~120-124 (`Implement: get_name(), get_description(), get_parameters(), execute()` / `Register in AVAILABLE_TOOLS`).
  Replace the "New Tool" recipe body with:
  ```
  **New Tool**:
  1. Subclass `Tool` (ABC) in `Tools/tool_executor.py`
  2. Implement the `name`, `description`, `parameters` properties and the async `execute(**kwargs)` method
  3. Register the instance via `ToolExecutor.register_tool()` (the singleton executor is obtained through `get_tool_executor()` in `tool_executor.py`); built-in registration is gated by the `[tools]` config section
  ```
  Verify: `git show origin/dev:tldw_chatbook/Tools/tool_executor.py | grep -nE "class Tool\(|def register_tool|def get_tool_executor|def name|def description|def parameters|async def execute"`. Confirm there is no `AVAILABLE_TOOLS` (grep returns nothing).

- [ ] **Step 4: chat_with_provider → chat_api_call** — line ~64 (`unified `chat_with_provider()` interface`) and lines ~109-112 (New LLM Provider recipe).
  - Line ~64 `- **`LLM_Calls/`** - ..., unified `chat_with_provider()` interface` → `- **`LLM_Calls/`** - Provider integrations (`chat_with_<provider>()` functions); the unified dispatcher is `chat_api_call()` in `Chat/Chat_Functions.py``
  - New LLM Provider recipe (line ~110 `1. Add to LLM_Calls/ with chat_with_provider() method`):
    ```
    **New LLM Provider**:
    1. Add a `chat_with_<provider>()` function in `LLM_Calls/LLM_API_Calls.py`
    2. Register it in `API_CALL_HANDLERS` / `PROVIDER_PARAM_MAP` (dispatched by `chat_api_call()` in `Chat/Chat_Functions.py`)
    3. Add the provider's config section
    ```
  - Grep the whole file for any other `chat_with_provider` mention and replace with `chat_api_call`.

- [ ] **Step 5: UI "Main Windows" list** — lines ~40-52 (`**Main Windows** (all extend Screen):` + the stale file list).
  Replace the "Main Windows" block with a description of the real architecture:
  ```
  **Screens** (tab content, registered in `UI/Navigation/screen_registry.py`):
  - `UI/Screens/chat_screen.py` - Chat (embeds `Chat_Window_Enhanced.py`, a Container)
  - `UI/Screens/search_screen.py` - RAG search (embeds `SearchRAGWindow.py`, a Container)
  - `UI/Screens/media_screen.py` / `media_ingest_screen.py` - Media hub + ingestion
  - `UI/Screens/personas_screen.py`, `evals_screen.py`, `notes_screen.py`, etc.

  Each tab is a `Screen` registered in `UI/Navigation/screen_registry.py`. Note: `Chat_Window_Enhanced.py` and `SearchRAGWindow.py` are embedded `Container` widgets, not Screens.
  ```
  Verify: `git show origin/dev:tldw_chatbook/UI/Navigation/screen_registry.py | grep -nE "Screen|register|:"` for the real screen list, and confirm the retired files are gone: `git ls-tree origin/dev tldw_chatbook/UI/ | grep -E "Conv_Char|Notes_Window|MediaWindow|Coding_Window|Evals_Window_v3|IngestTldwApiWindow"` should return nothing. Use only Screen files that actually exist. Also update the **New Tab** recipe (in Development Guidelines) to reference `UI/Screens/` + `screen_registry.py` — verify the real registration mechanism there before writing it.

- [ ] **Step 6: Add an Agents Runtime subsection** — under `## Architecture`, after the Business Logic or Event System block. Add:
  ```
  ### Agents Runtime (`Agents/`)

  A from-scratch agent framework, distinct from the `Tools/tool_executor.py` tool layer:
  - `agent_models.py`, `agent_runtime.py`, `agent_stream.py` - pure logic (control loop, streaming fence-gate, dataclasses; no I/O)
  - `agent_service.py` - the one impure seam: wires the loop to `chat_api_call`, the permission gate, sub-agent spawning, and run persistence (`AgentRuns_DB.py`); runs on a worker thread
  - `tool_catalog.py` (`ToolProvider` interface), `native_tools.py` (provider-native tool-calls), `mcp_tool_provider.py` (MCP bridge)
  ```
  Verify each module exists: `git ls-tree origin/dev tldw_chatbook/Agents/`.

- [ ] **Step 7: File Reference / Data Layer path fixes.**
  - `model_capabilities.py`: confirm its real location `git ls-tree origin/dev tldw_chatbook/ | grep model_capabilities` (top-level `tldw_chatbook/model_capabilities.py`, NOT under `LLM_Calls/`). Fix any CLAUDE.md reference that implies `LLM_Calls/model_capabilities.py` (line ~205 lists it bare — make the path explicit `model_capabilities.py` (top-level) if it's ambiguous).
  - Pre-commit hook (line ~182 `auto_review.py`): → `Helper_Scripts/fixed_auto_review.py`. Verify: `git ls-tree -r origin/dev | grep -E "auto_review"`.
  - Data Layer DB list (the `- Other DBs: Evals, Prompts, Subscriptions` line): expand to include the DBs that exist: `git ls-tree origin/dev tldw_chatbook/DB/ | grep _DB` — add `AgentRuns_DB`, `Workspace_DB`, `Library_Collections_DB`, `Library_Ingest_Jobs_DB`, `Research_DB`, `Writing_DB`, `Mindmap_DB`, `search_history_db`, `Sync_Client` (list only those that actually exist).

- [ ] **Step 8: Remove React-boilerplate gotchas** — lines ~188-189 (`1. **No localStorage** ...` / `2. **Tailwind limitations** ...`). Delete both (they describe React/web artifacts, irrelevant to this Python/Textual TUI). Renumber the remaining gotchas so the list stays sequential.

- [ ] **Step 9: Final verification pass.** Re-read the edited CLAUDE.md end to end. For EVERY file path it now names, confirm it exists on origin/dev (`git ls-tree`/`git show origin/dev:<path>` succeeds). For every symbol it names (`chat_api_call`, `ToolExecutor`, `register_tool`, `get_tool_executor`, `_CURRENT_SCHEMA_VERSION`, `screen_registry`, the Agents modules), confirm it exists. Fix any that don't. The Backlog.md CLI section at the end is untouched.

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: correct stale/wrong CLAUDE.md claims — schema, tool exec, tool/provider recipes, Screens, Agents, paths [TASK-333]"
```

---

### Task 6: Backlog AC updates + file the MCP-character follow-up

**Files:**
- Modify: `backlog/tasks/task-333 - Fix-incorrect-and-stale-developer-documentation.md`
- Modify: `backlog/tasks/task-334 - Remove-dead-chat-with-provider-imports.md` (filename may differ — find it)
- Create: one follow-up task file (id via the collision-safe scan)

- [ ] **Step 1: Update TASK-334** — its title/ACs assume "remove dead imports"; reality was "repair broken calls." Check all its AC boxes that are satisfied, set `status: Done`, and add an `## Implementation Notes` section: the imports were live (not dead) — all four sites (`MCP/server.py` chat_with_llm, `Tools/code_audit_tool.py` `_request_llm_analysis`, `UI/Tools_Settings_Window.py` `_test_chat_connection` + `_test_all_api_keys`) repointed to `chat_api_call(..., streaming=False)` + new `extract_response_content` helper; `MCP/server.py` was an unguarded ImportError crashing `TldwMCPServer.__init__`. `MCP/tools.py`'s stubs left intentionally; `MCPTools.chat_with_character` deferred to a follow-up (it also depends on the dead `save_conversation_from_messages`). Files: `Chat/Chat_Functions.py`, `MCP/server.py`, `Tools/code_audit_tool.py`, `UI/Tools_Settings_Window.py` + tests. If the existing ACs literally say "remove"/"delete", update their text to "repair/repoint" before checking them (per CLAUDE.md's "update the AC first" rule).

- [ ] **Step 2: Update TASK-333** — check all ACs, set `status: Done`, add `## Implementation Notes` listing the comprehensive corrections (schema→`_CURRENT_SCHEMA_VERSION`, tool exec wired, `Tool(ABC)` recipe, `chat_api_call` provider recipe, Main-Windows→Screens/`screen_registry`, Agents Runtime section, `model_capabilities` top-level path + `Helper_Scripts/fixed_auto_review.py` + DB list, dropped React gotchas; README verified clean). Note the extra items beyond the original ACs were folded in (user-approved comprehensive scope); if the original ACs don't enumerate them, add AC lines for the extras so the notes match the ACs.

- [ ] **Step 3: File the MCP-character follow-up.** Assign an id via the collision-safe scan:
  ```bash
  cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook-docs-cleanup && git fetch -q origin
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'EOF'
  import os, re, subprocess
  ids=set()
  for name in subprocess.run(["git","ls-tree","-r","--name-only","origin/dev","backlog/"],capture_output=True,text=True).stdout.splitlines():
      m=re.search(r"task-(\d+)",name)
      if m: ids.add(int(m.group(1)))
  for root in ("backlog/tasks","backlog/drafts"):
      if os.path.isdir(root):
          for name in os.listdir(root):
              m=re.search(r"task-(\d+)",name)
              if m: ids.add(int(m.group(1)))
  print("next id:", max(ids)+1)
  EOF
  ```
  Create `backlog/tasks/task-<id> - Restore-MCP-character-chat-dead-dispatch-and-persistence.md` (copy an existing task file's frontmatter format; labels `mcp,bug`; priority medium). Description: `MCPTools.chat_with_character` (`MCP/tools.py`) still calls two dead local stubs — `chat_with_provider` (repointable to `chat_api_call`, as TASK-334 did elsewhere) AND `save_conversation_from_messages` (a removed persistence helper with no obvious successor). Restoring MCP character chat needs both: repoint the dispatch AND find/rebuild the conversation-save path. Deferred from TASK-334 because the persistence half needs separate investigation. AC: character chat via MCP performs a real LLM call and persists the conversation, with a test; the `NotImplementedError` stubs in `MCP/tools.py` are removed once unused.

- [ ] **Step 4: Commit**

```bash
git add "backlog/tasks/task-333 - Fix-incorrect-and-stale-developer-documentation.md" backlog/tasks/task-334*.md backlog/tasks/task-<id>*.md
git commit -m "docs(backlog): close TASK-333/334; file MCP-character-chat restore follow-up"
```

---

## Post-plan notes for the controller (not for task implementers)

- SDD models: Task 1 (isolated helper, complete code) → cheapest tier; Tasks 2-4 (integration edits on real files) → mid tier; Task 5 (doc-pass, judgment + verification) → mid tier; Task 6 (backlog) → cheapest tier. Final whole-branch review → most capable.
- The two UI sites aren't unit-tested through `run_worker`; the repo-grep + import-resolution guards (Task 4) are the regression net, plus the shared helper is tested in Task 1.
- Live check before PR (optional, not CI): construct `TldwMCPServer()` in a REPL to confirm no ImportError, if `fastmcp` + DBs are available.

