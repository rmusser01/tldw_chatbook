# Doc-pass + broken-call repair (TASK-333 + TASK-334)

**Date:** 2026-07-24
**Backlog:** TASK-334 (repair the broken `chat_with_provider` call sites — MEDIUM; retitled from "remove dead imports") + TASK-333 (fix incorrect/stale developer documentation — HIGH). One cohesive sub-project, one PR.
**Branch:** `feat/docs-and-dead-import-cleanup` (worktree off `origin/dev` @ `f939cf151`).

All claims below were verified against `origin/dev` (not the working tree, which carries an older CLAUDE.md on a different branch). Coordination check done: **no other branch edits CLAUDE.md ahead of dev**, so the doc-pass won't collide.

## Scope reframe (both tasks mis-scoped as written)

- **TASK-334** is titled "remove dead imports (latent ImportError)" but the imports are **not dead** — all four sites are live calls that currently break. The fix is to **repoint them to the real dispatcher `chat_api_call`**, not delete them. The task AC is updated to "repair" (user-approved).
- **TASK-333** enumerates a subset of the wrong claims; verification found more. The doc-pass is **comprehensive** — every verified inaccuracy is fixed (user-approved). The task AC gains the extra items.

## Part A — TASK-334: repair the broken `chat_with_provider` call sites

### Ground truth

`chat_with_provider` no longer exists in `LLM_Calls/LLM_API_Calls.py` (only per-provider `chat_with_<name>` functions exist). Four sites still `from ..LLM_Calls.LLM_API_Calls import chat_with_provider` → **ImportError at runtime**. The real dispatcher is `chat_api_call` (`Chat/Chat_Functions.py:671`).

`chat_api_call(streaming=False)` returns the provider handler's **full OpenAI-shaped response dict** (`response = handler(**call_kwargs); return response`), NOT a bare string. The canonical content-extraction (used by `Agents/agent_service.py:114`) is:

```python
content = (resp or {}).get("choices", [{}])[0].get("message", {}).get("content") or ""
```

**This is the load-bearing correction:** every repointed site consumed `chat_with_provider`'s result as a *string*, so each must add this extraction — a kwarg-only repoint would surface a dict where a string is expected and break the feature differently.

### Kwarg mapping (verified against `chat_api_call`'s signature)

`provider=` → `api_endpoint=` (first positional) · `messages=` → `messages_payload=` · `temperature=` → `temp=` · `model=`/`max_tokens=`/`api_key=` unchanged · `timeout=` **dropped** (no such param). `chat_api_call` is synchronous — compatible with all four sites (they call it via `asyncio.to_thread` or a threaded `run_worker`).

### The four sites (line numbers re-verified at implementation time)

| Site | Current breakage | Repoint |
|---|---|---|
| `MCP/server.py` (`chat_with_llm` tool, import ~169 / call ~195) | **Unguarded → `TldwMCPServer.__init__` hard-crashes on ImportError; MCP server won't start** | `chat_api_call(api_endpoint=provider, messages_payload=messages, api_key=api_key, model=model, temp=temperature, max_tokens=max_tokens)`, then extract content; the tool's `{"response": <content-string>, ...}` shape is preserved |
| `Tools/code_audit_tool.py` (`_request_llm_analysis`, ~134/138) | Soft-fails (caught) → deception-risk analysis silently dead; passes `prompt=` (singular) | `chat_api_call(api_endpoint="anthropic", messages_payload=[{"role":"user","content":prompt}], model=<valid haiku id>, temp=0.1, max_tokens=500)`, extract content |
| `UI/Tools_Settings_Window.py` (`_test_chat_connection`, ~4166/4169) | Soft-fails → "Test Connection" button always reports failure | `chat_api_call(api_endpoint=provider, messages_payload=[{"role":"user","content":"Test connection. Reply with 'OK'."}], model=model, temp=0.1, max_tokens=10)`, extract content, keep the `"OK"` check |
| `UI/Tools_Settings_Window.py` (`_test_all_api_keys`, ~4303/4306) | Soft-fails → "Test All API Keys" always reports failure | same repoint per provider |

Notes:
- `MCP/tools.py`'s `NotImplementedError` stub `chat_with_provider()` is a **separate intentional local stub** (not imported by these sites) — see the fold-in decision below.
- `code_audit_tool`'s `model="claude-3-haiku"` is likely a stale short id; verify it resolves through `chat_api_call`'s anthropic handler and update to a valid id (e.g. `claude-3-5-haiku-latest` or the id used elsewhere in the repo) since we're restoring the feature. Verify at implementation, don't guess in the spec.

### Fifth usage — fold-in decision (surfaced, not silently expanded)

`MCP/tools.py` also has `MCPTools.chat_with_character()` calling the local `NotImplementedError` stub (character chat via MCP always fails, wrapped in try/except). It's outside TASK-334's four sites. **Decision: repoint it too, in the same PR** — it's the same bug class (a `chat_with_provider` usage that should be `chat_api_call`), the fix is identical, and leaving one MCP chat tool broken while fixing the other is inconsistent. If the character-message assembly is non-trivial, it may be split to a follow-up; the plan will make that call after reading the function. Either way the `NotImplementedError` stub is removed once nothing uses it.

### Testing

- **Repo-wide guard (strongest, covers the whole ImportError class):** a test asserting **zero** `chat_with_provider` imports remain anywhere under `tldw_chatbook/` (a `from ... import chat_with_provider` / `import chat_with_provider` grep returning empty). If the stub is removed, also assert no `def chat_with_provider` remains; if the fold-in is deferred, exclude the `MCP/tools.py` stub definition explicitly.
- **Import-resolution:** importing each touched module (`MCP.server`, `Tools.code_audit_tool`, `UI.Tools_Settings_Window`) raises no ImportError.
- **Functional (plain-function sites):** with `chat_api_call` mocked to return a canned OpenAI-shaped dict — (a) constructing `TldwMCPServer()` no longer raises, and the `chat_with_llm` tool calls `chat_api_call` with the mapped kwargs and returns the extracted content string; (b) `code_audit_tool._request_llm_analysis` builds the `messages_payload` from the prompt, calls `chat_api_call`, returns the extracted analysis, and no longer swallows an ImportError.
- **UI sites:** covered by the repo-grep + import-resolution guards (Textual `run_worker(thread=True)` methods aren't cleanly unit-testable through the worker; the content-extraction logic is identical to the tested sites). If a site's call is extractable into a plain helper without disrupting the file, prefer that + a direct functional test; do not restructure the UI purely for testability.

## Part B — TASK-333: comprehensive CLAUDE.md doc-pass

Edit **origin/dev's** CLAUDE.md. **Durability principle:** anchor to stable references (symbol names, file paths), not rot-prone specifics — avoid line numbers and bare version integers in the doc (that hardcoded "v7" is exactly what rotted). Every corrected claim is verified against code; a final pass confirms every file path the revised doc names exists on origin/dev.

Corrections (each → its acceptance criterion):

1. **Schema version** (Overview + Special Systems/Tool Calling): "v7" → "current schema version is defined by `_CURRENT_SCHEMA_VERSION` in `DB/ChaChaNotes_DB.py` (currently 24); increment it and add a migration when changing schema" — no bare integer that re-rots.
2. **Tool Calling status** (Special Systems): "Status: Detection works, execution pending" → detection AND execution are implemented; `ToolExecutor.execute_tool_call()` runs tools, wired into `Event_Handlers/worker_events.py` and `Event_Handlers/Chat_Events/chat_streaming_events.py`.
3. **New-Tool recipe** (Adding Features): `get_name()/get_description()/get_parameters()/execute()` + `AVAILABLE_TOOLS` → real: subclass `Tool` (ABC, `Tools/tool_executor.py`), implement properties `name`/`description`/`parameters` + async `execute(**kwargs)`; register via `ToolExecutor.register_tool()` obtained from `get_tool_executor()`, gated by `[tools]` config. (There is no `AVAILABLE_TOOLS`.)
4. **New-Provider recipe** + `chat_with_provider` references (Overview/Business Logic/Adding Features): add `chat_with_<provider>()` in `LLM_Calls/LLM_API_Calls.py`, register in `API_CALL_HANDLERS`/`PROVIDER_PARAM_MAP`, dispatched via `chat_api_call()` (`Chat/Chat_Functions.py`). Replace every `chat_with_provider()` mention with `chat_api_call()`.
5. **UI "Main Windows" list** (Core Structure/UI Layer): 6 of 8 listed files are retired (`Conv_Char_Window`, `Notes_Window`, `Evals_Window_v3`, `MediaWindow`, `Coding_Window`, `IngestTldwApiWindow` — `Tests/UI/test_legacy_entrypoints_retired.py` asserts some unimportable). Rewrite to the `Screen`-per-tab architecture: `UI/Screens/*.py` registered in `UI/Navigation/screen_registry.py`. Note `Chat_Window_Enhanced.py`/`SearchRAGWindow.py` are embedded `Container` widgets, not Screens. Update the **New-Tab recipe** accordingly (verify the real registration path against `screen_registry.py`).
6. **Agents Runtime** (new subsection under Architecture): document the `Agents/` package — `agent_models`/`agent_runtime`/`agent_stream` (pure), `agent_service` (impure seam → `chat_api_call`, permission gate, sub-agent spawn, `AgentRuns_DB` persistence), `tool_catalog` (`ToolProvider`), `native_tools`, `mcp_tool_provider`. Two-to-four sentences; note it's distinct from the `Tools/tool_executor.py` framework.
7. **File Reference / Data Layer path fixes:** `model_capabilities.py` is top-level `tldw_chatbook/model_capabilities.py` (not `LLM_Calls/model_capabilities.py`); pre-commit hook is `Helper_Scripts/fixed_auto_review.py` (not `auto_review.py`); complete the Data Layer DB list (add `AgentRuns_DB`, `Workspace_DB`, `Library_Collections_DB`, `Library_Ingest_Jobs_DB`, `Research_DB`, `Writing_DB`, `Mindmap_DB`, `search_history_db`, `Sync_Client`).
8. **Project-Specific Gotchas #1/#2** (localStorage / Tailwind): remove — React/web-artifact boilerplate irrelevant to this Python/Textual TUI.
9. **README:** verify first — only edit if it carries the same stale claims (`chat_with_provider`, schema version, tool-calling status). No speculative edits.

**Scope guard:** the auto-appended Backlog.md CLI section is untouched; no unrelated restructuring; only wrong claims are fixed.

### Testing

Docs have no automated tests. Correctness = the per-claim verification (done during design + re-verified at implementation with fresh line numbers) plus a final read-through and a path-existence spot-check of every file the revised doc names. The Part A repo-grep test doubles as a guard that no `chat_with_provider` reference the doc might mention is reintroduced.

## Commit structure

One PR, two commits: (1) TASK-334 code repair + tests + AC update; (2) TASK-333 doc-pass + AC update. Code first so the doc accurately describes the fixed dispatch path.

## Out of scope / residual

- Deeper MCP or code-audit feature work beyond restoring the dispatch call.
- Broader README rewrite beyond the stale-claim fixes.
- Line-number-precise citations in CLAUDE.md (deliberately avoided for durability).
