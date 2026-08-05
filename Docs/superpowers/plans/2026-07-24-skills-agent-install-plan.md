# Agent-Callable Skill Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5th agent runtime tool `install_skill(url)` so the top-level Console agent can install a skill from a GitHub/zip URL — pausing for an in-chat human confirm before any fetch, and always landing the skill trust-pending.

**Architecture:** `install_skill` follows the `skill_file` runtime-tool pattern (name in `RUNTIME_TOOL_NAMES`, pinned `ToolSchema`, a `LoopDeps.install_skill` dispatch closure). The closure is built in `ConsoleAgentBridge.run_reply` (enforce policy → classify URL → in-chat confirm → `install_skill_from_url` → wrap), gated to the top-level agent via the `agent_kind == AGENT_KIND_PRIMARY` parameter. The confirm is a new controller-owned human-in-the-loop block modeled on the MCP approval flow (`threading.Event` + `call_from_thread` a card, poll with cooperative cancel/timeout/context-change → deny), rendered by a new `SkillInstallConfirmCard` with its own `TaskResumeState` field.

**Tech Stack:** Python 3.12 (repo `.venv`), Textual widgets, httpx (test `MockTransport`), pytest. Reuses the merged `install_skill_from_url` seam unchanged.

## Global Constraints

- Test runner is the repo-root venv only: `.venv/bin/python -m pytest <path> -v` (system `python3` is 3.9 and breaks collection). Prefix env for keyring: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`.
- Actual file paths (NOT the assumed ones): `console_agent_bridge.py` and `console_chat_controller.py` live under `tldw_chatbook/Chat/`, not `Skills_Interop/`/`UI/Screens/`.
- Runtime-tool name is exactly `install_skill`; schema id `runtime:install_skill`.
- Closure order is load-bearing: `enforce_install_remote()` (no prompt on denial) → `classify_skill_source_url()` (no prompt on bad URL) → confirm (plain `threading.Event`, **outside** any `asyncio.run`) → `asyncio.run(install_skill_from_url(...))` → **broad `except Exception`** wrap. `import_skill_file` raises a bare `ValueError("local_skill_exists:…")` on collision, so the catch MUST be broad.
- Scope: wired **only when `self._skills_service is not None`**, and pinned/dispatched **only when `agent_kind == AGENT_KIND_PRIMARY`** — never for spawned subagents.
- Confirm timeout: 120 s (matching `_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`); poll granularity reuses `_MCP_APPROVAL_POLL_SECONDS` (1.0 s). Cancel/timeout/headless/context-change → deny (fail-closed).
- The confirm card must NOT reuse `pending_approval`; it needs its own `TaskResumeState.pending_skill_install` field, its own `ChatTaskCards` child, its own `sync_state` branch, and the container `display` gate extended with `has_pending_skill_install()` — else it renders invisibly.
- `install_skill` must NOT be added to `_QUIET_STEP_TOOLS` (so its step marker renders).
- E2E must construct services with a real `ServicePolicyEnforcer` + `RuntimeSourceState`, so the policy gate is genuinely enforced (not a silent no-op).
- Trust-pending is mandatory: `install_skill_from_url` always passes `trust_approved=False`; do not add or expose `overwrite` to the agent.
- Never `git add -A`. Do not touch `.superpowers/sdd/progress.md`.

---

## File Structure

- `tldw_chatbook/Agents/agent_models.py` — add `INSTALL_SKILL_TOOL_NAME`, extend `RUNTIME_TOOL_NAMES` (Task 1).
- `tldw_chatbook/Agents/tool_catalog.py` — add `INSTALL_SKILL_TOOL_SCHEMA` (Task 1).
- `tldw_chatbook/Agents/agent_runtime.py` — add `LoopDeps.install_skill` field + dispatch branch (Task 2).
- `tldw_chatbook/Agents/agent_service.py` — `install_skill_tool` kwarg + `agent_kind`-gated schema pin & wiring (Task 3).
- `tldw_chatbook/Chat/console_agent_bridge.py` — build the install closure + `run_reply` confirm kwarg + pass to `AgentService` (Task 4).
- `tldw_chatbook/Chat/console_chat_controller.py` — confirm HITL machinery + pass into `run_reply` (Task 5).
- `tldw_chatbook/UI/Screens/chat_screen_state.py` — `TaskResumeState.pending_skill_install` (Task 6).
- `tldw_chatbook/Widgets/Chat_Widgets/skill_install_confirm_card.py` — new card widget (Task 6).
- `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py` — third child + sync + display gate (Task 6).
- `tldw_chatbook/UI/Screens/chat_screen.py` — setter + `@on` handler + controller wiring (Task 6).
- Tests: `Tests/Agents/test_install_skill_runtime_tool.py` (new, Tasks 1-3), `Tests/Chat/test_console_agent_bridge.py` (extend, Task 4), `Tests/UI/test_console_skill_install_confirm.py` (new, Tasks 5-6), `Tests/Skills/test_skill_remote_fetch.py` (extend, Task 7).

---

### Task 1: Runtime tool name, schema, and collision guards

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py:31-40` (tool-name block)
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (add schema near `SKILL_FILE_TOOL_SCHEMA`)
- Modify: `Tests/Agents/test_agent_models.py` (the `RUNTIME_TOOL_NAMES == {...}` assertion)
- Modify: `Tests/Library/test_library_skills_state.py` (the `RUNTIME_TOOL_NAMES <= _SHADOWED_BUILTIN_NAMES` guard) and, if that guard fails, the `_SHADOWED_BUILTIN_NAMES` source
- Test: `Tests/Agents/test_install_skill_runtime_tool.py` (new)

**Interfaces:**
- Produces: `INSTALL_SKILL_TOOL_NAME = "install_skill"` (in `agent_models.py`), added to `RUNTIME_TOOL_NAMES`; `INSTALL_SKILL_TOOL_SCHEMA: ToolSchema` (in `tool_catalog.py`, id `"runtime:install_skill"`, one required string param `url`).

- [ ] **Step 1: Write the failing test** — create `Tests/Agents/test_install_skill_runtime_tool.py`:

```python
"""install_skill: the fifth runtime tool — name, schema, dispatch, gating.

Model: Tests/Agents/test_skill_file_runtime_tool.py (the 4th runtime tool).
install_skill is NOT a ToolProvider — its schema is pinned into
runtime_schemas (never disclosure-gated) only for the top-level agent
(agent_kind == primary), and its closure lives on LoopDeps.install_skill.
"""

import json

from tldw_chatbook.Agents.agent_models import (
    INSTALL_SKILL_TOOL_NAME,
    RUNTIME_TOOL_NAMES,
    SPAWN_TOOL_NAME,
    FIND_TOOLS_NAME,
    LOAD_TOOLS_NAME,
    SKILL_FILE_TOOL_NAME,
)
from tldw_chatbook.Agents.tool_catalog import INSTALL_SKILL_TOOL_SCHEMA


def test_install_skill_name_in_runtime_tool_names():
    assert INSTALL_SKILL_TOOL_NAME == "install_skill"
    assert RUNTIME_TOOL_NAMES == {
        SPAWN_TOOL_NAME,
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SKILL_FILE_TOOL_NAME,
        INSTALL_SKILL_TOOL_NAME,
    }


def test_install_skill_schema_shape():
    s = INSTALL_SKILL_TOOL_SCHEMA
    assert s.id == "runtime:install_skill"
    assert s.name == INSTALL_SKILL_TOOL_NAME
    assert s.parameters["required"] == ["url"]
    assert s.parameters["properties"]["url"]["type"] == "string"
    # Description must tell the model the key facts.
    assert "pending" in s.description.lower()
    assert "confirm" in s.description.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Agents/test_install_skill_runtime_tool.py -v`
Expected: FAIL — `ImportError: cannot import name 'INSTALL_SKILL_TOOL_NAME'`.

- [ ] **Step 3: Add the name + schema.** In `agent_models.py`, extend the tool-name block (currently lines 31-40):

```python
SPAWN_TOOL_NAME = "spawn_subagent"
FIND_TOOLS_NAME = "find_tools"
LOAD_TOOLS_NAME = "load_tools"
SKILL_FILE_TOOL_NAME = "skill_file"
INSTALL_SKILL_TOOL_NAME = "install_skill"
RUNTIME_TOOL_NAMES = frozenset(
    {
        SPAWN_TOOL_NAME,
        FIND_TOOLS_NAME,
        LOAD_TOOLS_NAME,
        SKILL_FILE_TOOL_NAME,
        INSTALL_SKILL_TOOL_NAME,
    }
)
```

In `tool_catalog.py`, add after `SKILL_FILE_TOOL_SCHEMA` (import `INSTALL_SKILL_TOOL_NAME` alongside the other names it already imports from `agent_models`):

```python
INSTALL_SKILL_TOOL_SCHEMA = ToolSchema(
    id="runtime:install_skill",
    name=INSTALL_SKILL_TOOL_NAME,
    description=(
        "Install a skill from a GitHub repository/tree URL or a direct "
        "https .zip URL. The user is asked to confirm before anything is "
        "downloaded. On success the skill is installed but left pending the "
        "user's review — it cannot run until the user approves it in "
        "Library > Skills. If the repository contains multiple skills, the "
        "tool returns the list of candidates; re-call with a URL that points "
        "at one skill's subdirectory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": (
                    "A GitHub repo/tree URL or a direct https .zip URL for "
                    "the skill to install."
                ),
            }
        },
        "required": ["url"],
    },
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Agents/test_install_skill_runtime_tool.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Fix the collision-guard tests.** Run them and update:

Run: `.venv/bin/python -m pytest Tests/Agents/test_agent_models.py Tests/Library/test_library_skills_state.py -v`
Expected: FAIL on the `RUNTIME_TOOL_NAMES == {...}` assertion in `test_agent_models.py` and possibly the `RUNTIME_TOOL_NAMES <= _SHADOWED_BUILTIN_NAMES` guard in `test_library_skills_state.py`.

Fixes:
- In `Tests/Agents/test_agent_models.py`, add `INSTALL_SKILL_TOOL_NAME` to the expected `RUNTIME_TOOL_NAMES` set literal (import it too).
- If the Library guard fails: `grep -rn "_SHADOWED_BUILTIN_NAMES" tldw_chatbook/` — it must remain a superset of `RUNTIME_TOOL_NAMES`. If it enumerates names literally, add `"install_skill"`; if it is derived from `RUNTIME_TOOL_NAMES`, no source change is needed and the test now passes. (Precedent: adding `skill_file` fired this same guard and was fixed with a one-line addition.)

Re-run until green.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_install_skill_runtime_tool.py Tests/Agents/test_agent_models.py Tests/Library/test_library_skills_state.py
# add tldw_chatbook/<file with _SHADOWED_BUILTIN_NAMES> only if you changed it
git commit -m "feat(agents): install_skill runtime tool name + schema + collision guards"
```

---

### Task 2: LoopDeps.install_skill field + dispatch branch

**Files:**
- Modify: `tldw_chatbook/Agents/agent_runtime.py:190` (add field after `read_skill_file`), dispatch chain `:528-539`
- Test: `Tests/Agents/test_install_skill_runtime_tool.py` (extend)

**Interfaces:**
- Consumes: `INSTALL_SKILL_TOOL_NAME` (Task 1).
- Produces: `LoopDeps.install_skill: Callable[[str], ToolResult] | None = None`; a dispatch branch routing an `install_skill` call to `deps.install_skill(url)` when wired, else falling through to `deps.invoke_tool`.

- [ ] **Step 1: Write the failing test** — append to `Tests/Agents/test_install_skill_runtime_tool.py`:

```python
from tldw_chatbook.Agents.agent_models import (
    AgentConfig,
    ModelTurn,
    RUN_DONE,
    RunBudget,
    ToolCatalogEntry,
    ToolResult,
    ToolSchema,
)
from tldw_chatbook.Agents.agent_runtime import LoopDeps, run_agent_loop

_CALC = ToolSchema(
    id="builtin:calculator", name="calculator", description="math",
    parameters={"type": "object"},
)


def _fence(name, args):
    return f"```tool_call\n{json.dumps({'name': name, 'arguments': args})}\n```"


def _deps(turns, *, install_skill=None, invoke=None):
    script = list(turns)

    def call_model(messages, active_schemas):
        return script.pop(0)

    return LoopDeps(
        call_model=call_model,
        invoke_tool=invoke or (lambda c: ToolResult(ok=False, error=f"Tool not permitted: {c.name}")),
        spawn=lambda task: ToolResult(ok=True, content="sub"),
        find_tools=lambda q: [],
        load_schemas=lambda ids: [],
        should_cancel=lambda: False,
        clock=lambda: 0.0,
        install_skill=install_skill,
    )


_CFG = AgentConfig(model="m", system_prompt="s", allowed_tools=("calculator",))


def test_install_skill_dispatches_to_deps_when_wired():
    seen = []

    def installer(url):
        seen.append(url)
        return ToolResult(ok=True, content=f"installed {url}")

    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(text=_fence("install_skill", {"url": "https://github.com/o/r"})),
                ModelTurn(text="done"),
            ],
            install_skill=installer,
        ),
    )
    assert out.status == RUN_DONE
    assert seen == ["https://github.com/o/r"]
    assert any(s.kind == "tool_result" and "installed" in (s.result or "") for s in out.steps)


def test_install_skill_falls_through_when_not_wired():
    out = run_agent_loop(
        _CFG,
        [{"role": "user", "content": "hi"}],
        [_CALC],
        _deps(
            [
                ModelTurn(text=_fence("install_skill", {"url": "https://github.com/o/r"})),
                ModelTurn(text="done"),
            ],
            install_skill=None,  # not wired -> generic invoke_tool path
        ),
    )
    assert out.status == RUN_DONE
    results = [s.result for s in out.steps if s.kind == "tool_result"]
    assert any("Tool not permitted: install_skill" in (r or "") for r in results)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Agents/test_install_skill_runtime_tool.py -k install_skill_dispatches -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'install_skill'`.

- [ ] **Step 3: Add the field.** In `agent_runtime.py`, add after the `read_skill_file` field (currently ends at line 190):

```python
    read_skill_file: Callable[[str, str], ToolResult] | None = None
    # install_skill: the fifth runtime tool (agent-callable skill install).
    # Wired ONLY for the top-level agent (agent_kind == primary) by the
    # service; a spawned subagent never receives it. `None` (the default)
    # means the run is not wired for install_skill and a call by that name
    # falls through to the generic deps.invoke_tool path.
    install_skill: Callable[[str], ToolResult] | None = None
```

- [ ] **Step 4: Add the dispatch branch.** In `agent_runtime.py`, insert a new `elif` after the `SKILL_FILE_TOOL_NAME` branch (currently lines 528-536) and before the generic `else` (line 537):

```python
                elif (
                    call.name == INSTALL_SKILL_TOOL_NAME
                    and deps.install_skill is not None
                ):
                    add(STEP_TOOL_CALL, tool_name=call.name, args=dict(call.args))
                    result = deps.install_skill(str(call.args.get("url", "")))
```

Import `INSTALL_SKILL_TOOL_NAME` at the top of `agent_runtime.py` alongside `SKILL_FILE_TOOL_NAME` (in the existing `from .agent_models import (...)` block).

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Agents/test_install_skill_runtime_tool.py -v`
Expected: PASS (all 4 tests).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_install_skill_runtime_tool.py
git commit -m "feat(agents): install_skill dispatch branch + LoopDeps.install_skill"
```

---

### Task 3: AgentService wiring gated to the top-level agent

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py:165-214` (`__init__`), `:353-362` (runtime_schemas), `:599-625` (LoopDeps build)
- Test: `Tests/Agents/test_install_skill_runtime_tool.py` (extend)

**Interfaces:**
- Consumes: `INSTALL_SKILL_TOOL_NAME`, `INSTALL_SKILL_TOOL_SCHEMA`, `LoopDeps.install_skill`, `AGENT_KIND_PRIMARY`.
- Produces: `AgentService(..., install_skill_tool: Callable[[str], ToolResult] | None = None)`. The schema is pinned and the dep wired **only when `agent_kind == AGENT_KIND_PRIMARY and self._install_skill_tool is not None`**.

- [ ] **Step 1: Write the failing test** — append to `Tests/Agents/test_install_skill_runtime_tool.py`. This mirrors `Tests/Agents/test_skill_tool_spawn.py::test_native_spawn_child_cannot_call_a_skill_tool`:

```python
from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolCatalogRegistry
from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB


def _svc_fence(name, args):
    return {"choices": [{"message": {"content": _fence(name, args)}}]}


def test_top_level_agent_dispatches_install_skill(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())
    seen = []

    def installer(url):
        seen.append(url)
        return ToolResult(ok=True, content=f"installed {url}")

    script = [
        _svc_fence("install_skill", {"url": "https://github.com/o/r"}),
        {"choices": [{"message": {"content": "Done."}}]},
    ]
    service = AgentService(
        db, reg, chat_call=lambda **k: script.pop(0), install_skill_tool=installer
    )
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "install it"}],
        config=AgentConfig(
            model="m", system_prompt="s",
            allowed_tools=("calculator",), budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    assert seen == ["https://github.com/o/r"]


def test_subagent_cannot_call_install_skill(tmp_path):
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    reg = ToolCatalogRegistry()
    reg.register_provider(BuiltinToolProvider())

    def installer(url):
        raise AssertionError("subagent must never reach the installer")

    script = [
        _svc_fence(SPAWN_TOOL_NAME, {"task": "native task"}),   # parent spawns
        _svc_fence("install_skill", {"url": "https://github.com/o/r"}),  # child tries
        {"choices": [{"message": {"content": "child gave up"}}]},
        {"choices": [{"message": {"content": "final"}}]},
    ]
    service = AgentService(
        db, reg, chat_call=lambda **k: script.pop(0), install_skill_tool=installer
    )
    _rid, outcome = service.run_turn(
        conversation_id="c1",
        messages=[{"role": "user", "content": "go"}],
        config=AgentConfig(
            model="m", system_prompt="s",
            allowed_tools=("calculator", SPAWN_TOOL_NAME), budget=RunBudget(),
        ),
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    child_runs = [r for r in db.list_runs("c1") if r["agent_kind"] == "subagent"]
    assert len(child_runs) == 1
    tool_results = [
        s["result"] for s in child_runs[0]["steps"] if s["kind"] == "tool_result"
    ]
    assert any("Tool not permitted: install_skill" in r for r in tool_results)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Agents/test_install_skill_runtime_tool.py -k "top_level or subagent" -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'install_skill_tool'`.

- [ ] **Step 3: Add the kwarg + storage.** In `agent_service.py` `__init__` (after the `skill_file_bindings` param at line 169 and its storage at line 186):

Add to the signature (keyword, after `review_state_scope`):
```python
        install_skill_tool: Callable[[str], ToolResult] | None = None,
```
Add to the body (after `self.skill_file_bindings = skill_file_bindings`):
```python
        # Agent-callable skill install (5th runtime tool). A ready-built
        # closure (enforce -> classify -> confirm -> install -> wrap) supplied
        # by the bridge. Pinned/wired ONLY for the top-level agent
        # (agent_kind == primary) in _run_one; a spawned subagent never gets
        # it. `None` (the default) means the run is not wired for install.
        self._install_skill_tool = install_skill_tool
```

- [ ] **Step 4: Gate the schema pin.** In `agent_service.py` runtime_schemas block (lines 353-362), append after the `skill_file_bindings` pin:

```python
        if agent_kind == AGENT_KIND_PRIMARY and self._install_skill_tool is not None:
            runtime_schemas.append(INSTALL_SKILL_TOOL_SCHEMA)
```

(Confirm `AGENT_KIND_PRIMARY` and `INSTALL_SKILL_TOOL_SCHEMA` are imported at the top of `agent_service.py`; `AGENT_KIND_PRIMARY` already is — it is used at line 706. Add `INSTALL_SKILL_TOOL_SCHEMA` to the `from .tool_catalog import (...)` block and `INSTALL_SKILL_TOOL_NAME` is not needed here.)

- [ ] **Step 5: Wire the dep.** In the `LoopDeps(...)` construction (lines 599-625), add after the `read_skill_file=` entry:

```python
            install_skill=(
                self._install_skill_tool
                if agent_kind == AGENT_KIND_PRIMARY
                and self._install_skill_tool is not None
                else None
            ),
```

- [ ] **Step 6: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Agents/test_install_skill_runtime_tool.py -v`
Expected: PASS (all 6 tests).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/agent_service.py Tests/Agents/test_install_skill_runtime_tool.py
git commit -m "feat(agents): wire install_skill_tool, gated to the top-level agent"
```

---

### Task 4: Bridge — build the install closure + thread the confirm through run_reply

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:886-902` (`run_reply` signature), `:938-985` (closure build), `:1096-1106` (AgentService call)
- Test: `Tests/Chat/test_console_agent_bridge.py` (extend)

**Interfaces:**
- Consumes: `install_skill_from_url`, `classify_skill_source_url`, `RemoteSkillError` (`Skills_Interop/skill_remote_fetch.py`); `PolicyDeniedError` (`runtime_policy/types.py`); `SkillsScopeService.enforce_install_remote()`; `AgentService(install_skill_tool=...)` (Task 3).
- Produces: `run_reply(..., request_skill_install_confirm: Callable[[str], bool] | None = None)`. When `self._skills_service is not None`, builds `install_skill_tool` and passes it to `AgentService`.

- [ ] **Step 1: Write the failing test** — append to `Tests/Chat/test_console_agent_bridge.py`. Extend the existing `_FakeSkillsService` (add `enforce_install_remote` + `import_skill_file`) or define a subclass inline:

```python
def _install_skills_service():
    svc = _FakeSkillsService()

    def enforce_install_remote():
        return None

    async def import_skill_file(*a, **k):  # not used (install_skill_from_url is patched)
        return {"name": "unused"}

    svc.enforce_install_remote = enforce_install_remote
    svc.import_skill_file = import_skill_file
    return svc


def test_install_skill_confirm_allow_installs(tmp_path, monkeypatch):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    installed = []

    async def fake_install(url, *, scope_service, **kw):
        installed.append(url)
        return {"name": "demo", "trust_status": "quarantined_added", "trust_blocked": True}

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)

    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["Installed it."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    confirmed = []

    outcome = _run(
        bridge, store, session, assistant.id,
        conversation_id="conv-install",
        request_skill_install_confirm=lambda url: confirmed.append(url) or True,
    )
    assert outcome.status == "done"
    assert confirmed == ["https://github.com/o/r"]
    assert installed == ["https://github.com/o/r"]
    tool_msgs = [m.content for m in store.messages_for_session(session.id)
                 if m.role == ConsoleMessageRole.TOOL]
    assert any("demo" in c and "pending" in c.lower() for c in tool_msgs)


def test_install_skill_confirm_deny_does_not_install(tmp_path, monkeypatch):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    async def fake_install(url, *, scope_service, **kw):
        raise AssertionError("install must not run when the user denies")

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)

    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["Okay, cancelled."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge, store, session, assistant.id,
        conversation_id="conv-deny",
        request_skill_install_confirm=lambda url: False,
    )
    assert outcome.status == "done"
    tool_msgs = [m.content for m in store.messages_for_session(session.id)
                 if m.role == ConsoleMessageRole.TOOL]
    assert any("declined" in c.lower() for c in tool_msgs)


def test_install_skill_malformed_url_never_prompts(tmp_path):
    """A URL that fails classification returns an error WITHOUT prompting."""
    prompted = []
    scripts = [
        [_fence("install_skill", {"url": "not-a-url"})],
        ["That URL is not valid."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-bad",
        request_skill_install_confirm=lambda url: prompted.append(url) or True,
    )
    assert outcome.status == "done"
    assert prompted == []  # classification failed before any prompt


def test_install_skill_collision_error_survives_turn(tmp_path, monkeypatch):
    """A bare ValueError('local_skill_exists:...') is wrapped, not fatal."""
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf

    async def fake_install(url, *, scope_service, **kw):
        raise ValueError("local_skill_exists:demo")

    monkeypatch.setattr(srf, "install_skill_from_url", fake_install)
    scripts = [
        [_fence("install_skill", {"url": "https://github.com/o/r"})],
        ["It already exists."],
    ]
    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=_install_skills_service(),
    )
    outcome = _run(
        bridge, store, session, assistant.id, conversation_id="conv-exists",
        request_skill_install_confirm=lambda url: True,
    )
    assert outcome.status == "done"  # turn survives the bare ValueError
    tool_msgs = [m.content for m in store.messages_for_session(session.id)
                 if m.role == ConsoleMessageRole.TOOL]
    assert any("local_skill_exists" in c for c in tool_msgs)
```

Note: `ToolResult` must be importable in `console_agent_bridge.py` — it already imports it from `agent_models` (used by `_BridgeSkillRunner`). Confirm before relying on it; add it to the existing `from ..Agents.agent_models import (...)` block if absent.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest "Tests/Chat/test_console_agent_bridge.py::test_install_skill_confirm_allow_installs" -v`
Expected: FAIL — `run_reply() got an unexpected keyword argument 'request_skill_install_confirm'`.

- [ ] **Step 3: Add the run_reply kwarg.** In `console_agent_bridge.py` `run_reply` signature (after `turn_bundle_block`):

```python
        request_skill_install_confirm: Callable[[str], bool] | None = None,
```

- [ ] **Step 4: Build the install closure.** In `run_reply`, after the `skill_file_bindings` block (line 985), add:

```python
        # Agent-callable skill install (5th runtime tool). Built only when a
        # skills service exists; wired to AgentService, which pins/dispatches
        # it for the top-level agent only. Order (load-bearing): enforce
        # policy (no prompt on denial) -> classify URL (no prompt on a bad
        # URL) -> in-chat confirm (plain blocking call, OUTSIDE asyncio.run)
        # -> asyncio.run(install) -> broad-catch wrap. import_skill_file
        # raises a bare ValueError("local_skill_exists:...") on collision, so
        # the install catch is broad.
        install_skill_tool = None
        if self._skills_service is not None:
            scope = self._skills_service

            def install_skill_tool(url: str) -> ToolResult:
                from tldw_chatbook.Skills_Interop.skill_remote_fetch import (
                    classify_skill_source_url,
                    install_skill_from_url,
                )
                from tldw_chatbook.runtime_policy.types import PolicyDeniedError

                try:
                    scope.enforce_install_remote()
                except PolicyDeniedError as exc:
                    return ToolResult(ok=False, error=exc.user_message)
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=str(exc))
                try:
                    classify_skill_source_url(url)
                except Exception as exc:  # noqa: BLE001 (RemoteSkillError etc.)
                    return ToolResult(ok=False, error=str(exc))
                try:
                    allowed = bool(
                        request_skill_install_confirm(url)
                        if request_skill_install_confirm is not None
                        else False
                    )
                except Exception:  # noqa: BLE001 — a UI error fails closed
                    allowed = False
                if not allowed:
                    return ToolResult(
                        ok=False, error="The user declined to install this skill."
                    )
                try:
                    result = asyncio.run(
                        install_skill_from_url(url, scope_service=scope)
                    )
                except Exception as exc:  # noqa: BLE001
                    return ToolResult(ok=False, error=str(exc))
                name = result.get("name", "") if isinstance(result, dict) else ""
                return ToolResult(
                    ok=True,
                    content=(
                        f'Installed "{name}" — it is pending your review and '
                        "cannot run until you approve it in Library > Skills."
                    ),
                )
```

- [ ] **Step 5: Pass it to AgentService.** In the `AgentService(...)` call (lines 1096-1106), add:

```python
            install_skill_tool=install_skill_tool,
```

- [ ] **Step 6: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -k install_skill -v`
Expected: PASS (2 tests). Then the whole file: `.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py -v` — no regressions.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat(chat): build the install_skill closure and thread the confirm through run_reply"
```

---

### Task 5: Controller — in-chat confirm HITL

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — add confirm constants/fields/methods, wire `switch_session`, pass into the `run_reply` call site (~2930)
- Test: `Tests/UI/test_console_skill_install_confirm.py` (new)

**Interfaces:**
- Produces: `ConsoleChatController.request_skill_install_confirm(url: str) -> bool` (worker thread, blocking), `resolve_pending_skill_install(allow: bool)` (UI thread), `_deny_pending_skill_install_on_context_change()`, `set_pending_skill_install: Callable[[dict | None], None] | None`. Payload shape: `{"url": <str>, "timeout_seconds": <float>}`.

- [ ] **Step 1: Write the failing test** — create `Tests/UI/test_console_skill_install_confirm.py`. Mirror `Tests/UI/test_console_mcp_approval.py`'s controller round-trip:

```python
from __future__ import annotations

import asyncio
import time

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore


class _FakeApp:
    def call_from_thread(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)


def _controller():
    store = ConsoleChatStore()
    return ConsoleChatController(store=store, provider_gateway=object()), store


@pytest.mark.asyncio
async def test_confirm_round_trip_allow():
    controller, _ = _controller()
    received: list[dict | None] = []
    controller.app = _FakeApp()
    controller.set_pending_skill_install = received.append

    async def resolve_soon():
        await asyncio.sleep(0.05)
        assert received and received[0] is not None
        assert received[0]["url"] == "https://github.com/o/r"
        controller.resolve_pending_skill_install(True)

    task = asyncio.create_task(
        asyncio.to_thread(controller.request_skill_install_confirm, "https://github.com/o/r")
    )
    await resolve_soon()
    allowed = await task
    assert allowed is True
    assert received[-1] is None  # card cleared afterwards


@pytest.mark.asyncio
async def test_confirm_round_trip_deny():
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None

    async def resolve_soon():
        await asyncio.sleep(0.05)
        controller.resolve_pending_skill_install(False)

    task = asyncio.create_task(
        asyncio.to_thread(controller.request_skill_install_confirm, "https://x/y")
    )
    await resolve_soon()
    assert (await task) is False


def test_confirm_timeout_denies():
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    controller.skill_install_confirm_timeout_seconds = lambda: 0.05
    started = time.monotonic()
    allowed = controller.request_skill_install_confirm("https://x/y")
    assert allowed is False
    assert time.monotonic() - started < 2.5


def test_context_change_denies_pending_confirm():
    controller, _ = _controller()
    controller.app = _FakeApp()
    controller.set_pending_skill_install = lambda payload: None
    import threading

    result = {}

    def worker():
        result["allowed"] = controller.request_skill_install_confirm("https://x/y")

    t = threading.Thread(target=worker)
    t.start()
    time.sleep(0.1)
    controller._deny_pending_skill_install_on_context_change()
    t.join(timeout=3.0)
    assert result["allowed"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_skill_install_confirm.py -v`
Expected: FAIL — `AttributeError: 'ConsoleChatController' object has no attribute 'request_skill_install_confirm'`.

- [ ] **Step 3: Add constants + fields.** Near the MCP approval constants (`_DEFAULT_MCP_APPROVAL_TIMEOUT_SECONDS`, line 69), add:

```python
_DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS = 120.0
```

In `__init__`, near the MCP approval fields (lines 396-403), add:

```python
        #: UI-thread callback that pushes/clears the pending skill-install
        #: confirm payload into the owning screen's task-resume state
        #: (ChatScreen._set_console_pending_skill_install). Invoked through
        #: self.app.call_from_thread from request_skill_install_confirm.
        self.set_pending_skill_install: Callable[[dict | None], None] | None = None
        #: Optional test override for the confirm timeout.
        self.skill_install_confirm_timeout_seconds: Callable[[], float] | None = None
        #: The active confirm round's release Event + shared decision box.
        self._pending_skill_install_event: threading.Event | None = None
        self._pending_skill_install_decision: dict[str, bool] | None = None
```

- [ ] **Step 4: Add the confirm method (worker thread).** Mirror `request_mcp_approvals`:

```python
    def request_skill_install_confirm(self, url: str) -> bool:
        """WORKER THREAD: ask the user to confirm a skill install before any fetch.

        Blocks on a fresh threading.Event, surfacing an Allow/Deny card via
        set_pending_skill_install (marshaled onto the UI thread), then polls
        re-checking this run's cancel signals and a deadline. Cancel/stop,
        timeout, context-change, or no wired UI all resolve to DENY
        (fail-closed). Returns True only on an explicit Allow.
        """
        event = threading.Event()
        decision: dict[str, bool] = {}
        self._pending_skill_install_event = event
        self._pending_skill_install_decision = decision

        timeout_seconds = (
            self.skill_install_confirm_timeout_seconds()
            if self.skill_install_confirm_timeout_seconds is not None
            else _DEFAULT_SKILL_INSTALL_CONFIRM_TIMEOUT_SECONDS
        )
        deadline = time.monotonic() + timeout_seconds
        payload = {"url": url, "timeout_seconds": timeout_seconds}
        try:
            self._marshal_pending_skill_install(payload)
            while not event.wait(_MCP_APPROVAL_POLL_SECONDS):
                if self._stop_requested or (
                    self._active_cancel_event is not None
                    and self._active_cancel_event.is_set()
                ):
                    break
                if time.monotonic() >= deadline:
                    break
            return bool(decision.get("allow", False))
        finally:
            self._pending_skill_install_event = None
            self._pending_skill_install_decision = None
            try:
                self._marshal_pending_skill_install(None)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).debug(
                    "Failed to clear skill-install confirm during teardown"
                )

    def _marshal_pending_skill_install(self, payload: dict[str, Any] | None) -> None:
        if self.app is not None and self.set_pending_skill_install is not None:
            self.app.call_from_thread(self.set_pending_skill_install, payload)

    def resolve_pending_skill_install(self, allow: bool) -> None:
        """UI THREAD: apply the user's Allow/Deny, releasing the worker thread."""
        decision = self._pending_skill_install_decision
        event = self._pending_skill_install_event
        if decision is None or event is None:
            return
        decision["allow"] = bool(allow)
        event.set()

    def _deny_pending_skill_install_on_context_change(self) -> None:
        """Force-deny a pending confirm (Event set, decision left False)."""
        event = self._pending_skill_install_event
        if event is not None:
            event.set()
```

- [ ] **Step 5: Wire context-change.** In `switch_session` (line 696), add alongside the existing MCP deny:

```python
        self._deny_pending_approval_on_context_change()
        self._deny_pending_skill_install_on_context_change()
```

- [ ] **Step 6: Pass the confirm into run_reply.** At the `run_reply` call site (lines 2930-2945), add:

```python
                request_skill_install_confirm=self.request_skill_install_confirm,
```

- [ ] **Step 7: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_skill_install_confirm.py -v`
Expected: PASS (4 tests).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/UI/test_console_skill_install_confirm.py
git commit -m "feat(chat): controller in-chat confirm HITL for agent skill install"
```

---

### Task 6: UI — TaskResumeState field, confirm card, task-cards rendering

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py:204-251` (`TaskResumeState`)
- Create: `tldw_chatbook/Widgets/Chat_Widgets/skill_install_confirm_card.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py` (whole file)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (setter, `@on` handler, controller wiring ~3307-3316)
- Test: `Tests/UI/test_console_skill_install_confirm.py` (extend)

**Interfaces:**
- Consumes: `ConsoleChatController.set_pending_skill_install`, `resolve_pending_skill_install` (Task 5).
- Produces: `TaskResumeState.pending_skill_install` + `has_pending_skill_install()`; `SkillInstallConfirmCard` with `set_install(payload)` and `InstallDecided(allow: bool)` message; `ChatTaskCards` renders it; `ChatScreen._set_console_pending_skill_install`.

- [ ] **Step 1: Write the failing tests** — append to `Tests/UI/test_console_skill_install_confirm.py`:

```python
from dataclasses import replace

from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState


def test_task_resume_state_pending_skill_install_roundtrip():
    s = TaskResumeState()
    assert s.has_pending_skill_install() is False
    s2 = replace(s, pending_skill_install={"url": "https://x/y", "timeout_seconds": 120.0})
    assert s2.has_pending_skill_install() is True
    assert TaskResumeState.from_dict(s2.to_dict()).pending_skill_install == {
        "url": "https://x/y", "timeout_seconds": 120.0,
    }


@pytest.mark.asyncio
async def test_skill_install_card_allow_and_deny():
    from textual import on
    from textual.app import App, ComposeResult
    from textual.widgets import Button
    from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
        SkillInstallConfirmCard,
    )

    class _Host(App[None]):
        def __init__(self):
            super().__init__()
            self.decided = []

        def compose(self) -> ComposeResult:
            yield SkillInstallConfirmCard()

        @on(SkillInstallConfirmCard.InstallDecided)
        def _cap(self, event: SkillInstallConfirmCard.InstallDecided) -> None:
            self.decided.append(event.allow)

    app = _Host()
    async with app.run_test() as pilot:
        card = app.query_one(SkillInstallConfirmCard)
        # A URL containing Rich-markup-like text must render literally.
        card.set_install(
            {"url": "https://github.com/o/[bold]r[/]", "timeout_seconds": 120.0}
        )
        await pilot.pause()
        assert card.display is True
        from textual.widgets import Static
        url_text = str(app.query_one("#skill-install-url", Static).render())
        assert "[bold]" in url_text  # not interpreted as markup
        app.query_one("#skill-install-allow", Button).press()
        await pilot.pause()
        assert app.decided == [True]
        card.set_install({"url": "https://github.com/o/r", "timeout_seconds": 120.0})
        await pilot.pause()
        app.query_one("#skill-install-deny", Button).press()
        await pilot.pause()
        assert app.decided == [True, False]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_skill_install_confirm.py -k "roundtrip or card_allow" -v`
Expected: FAIL — `AttributeError: ... 'has_pending_skill_install'` / `ModuleNotFoundError: skill_install_confirm_card`.

- [ ] **Step 3: Extend TaskResumeState.** In `chat_screen_state.py`, add the field + method + dict entries:

Add field (after `pending_approval`):
```python
    pending_skill_install: Optional[Dict[str, Any]] = None
```
Add method (after `has_pending_approval`):
```python
    def has_pending_skill_install(self) -> bool:
        """Return True when a skill-install confirm should be shown."""
        return bool(self.pending_skill_install)
```
Add to `to_dict` return dict: `"pending_skill_install": self.pending_skill_install,`
Add to `from_dict`: `pending_skill_install=data.get("pending_skill_install"),`

- [ ] **Step 4: Create the card.** Write `tldw_chatbook/Widgets/Chat_Widgets/skill_install_confirm_card.py`:

```python
"""Single-item Allow/Deny card for an agent-initiated skill install.

Distinct from ChatApprovalCard (the MCP batch card): one URL, one boolean
decision, no per-row Selects and no 5-way vocabulary. The URL is
agent/attacker-influenced, so it is rendered with markup=False.
"""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.message import Message
from textual.widgets import Button, Static


class SkillInstallConfirmCard(Container):
    """Prompts the user to allow/deny installing a skill from a URL."""

    class InstallDecided(Message):
        """Posted when the user allows or denies the install."""

        def __init__(self, allow: bool) -> None:
            self.allow = allow
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static(
            "An agent wants to install a skill:",
            id="skill-install-prompt",
            markup=False,
        )
        yield Static("", id="skill-install-url", markup=False)
        yield Static(
            "It will be installed pending your review and cannot run until "
            "you approve it in Library > Skills.",
            id="skill-install-note",
            markup=False,
        )
        yield Horizontal(
            Button("Allow", id="skill-install-allow", variant="primary"),
            Button("Deny", id="skill-install-deny", variant="error"),
            id="skill-install-buttons",
        )

    def on_mount(self) -> None:
        self.display = False

    def set_install(self, payload: dict[str, Any] | None) -> None:
        """Show the card for ``payload`` (``{"url": ...}``), or hide it if None."""
        if not payload:
            self.display = False
            return
        self.query_one("#skill-install-url", Static).update(str(payload.get("url", "")))
        self.display = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-install-allow":
            event.stop()
            self.display = False
            self.post_message(self.InstallDecided(True))
        elif event.button.id == "skill-install-deny":
            event.stop()
            self.display = False
            self.post_message(self.InstallDecided(False))
```

- [ ] **Step 5: Wire ChatTaskCards.** Replace `chat_task_cards.py` with:

```python
from textual.app import ComposeResult
from textual.containers import Container
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard
from tldw_chatbook.Widgets.Chat_Widgets.chat_resume_panel import ChatResumePanel
from tldw_chatbook.Widgets.Chat_Widgets.skill_install_confirm_card import (
    SkillInstallConfirmCard,
)


class ChatTaskCards(Container):
    """Inline task-surface wrapper for approvals, skill-install, and resume."""

    def compose(self) -> ComposeResult:
        yield ChatApprovalCard(id="chat-approval-card")
        yield SkillInstallConfirmCard(id="chat-skill-install-card")
        yield ChatResumePanel(id="chat-resume-panel")

    def on_mount(self) -> None:
        self.display = False

    def sync_state(self, task_state) -> None:
        """Sync the approval, skill-install, and resume cards from task state."""
        approval_card = self.query_one(ChatApprovalCard)
        install_card = self.query_one(SkillInstallConfirmCard)
        resume_panel = self.query_one(ChatResumePanel)

        approval = task_state.pending_approval
        calls = approval.get("calls") if isinstance(approval, dict) else None
        if calls:
            approval_card.set_batch(
                calls, timeout_seconds=approval.get("timeout_seconds", 0.0)
            )
        else:
            approval_card.set_approval(approval)
        install_card.set_install(task_state.pending_skill_install)
        resume_panel.set_resume_state(task_state)
        self.display = (
            task_state.has_pending_approval()
            or task_state.has_pending_skill_install()
            or task_state.has_resume_content()
        )
```

- [ ] **Step 6: Wire ChatScreen.** In `chat_screen.py`:

Add a setter near `_set_console_pending_approval` (line 15227):
```python
    def _set_console_pending_skill_install(
        self, payload: Dict[str, Any] | None
    ) -> None:
        """Set/clear the pending skill-install confirm, then sync the task cards.

        UI-thread bridge target for ConsoleChatController.
        request_skill_install_confirm, invoked via call_from_thread. Mutates
        only pending_skill_install so an in-flight approval/resume state is
        never clobbered.
        """
        current = self.chat_state.task_resume_state
        self.set_task_resume_state(replace(current, pending_skill_install=payload))
```

Add an `@on` handler near `handle_console_approval_decided` (line 15241) — import `SkillInstallConfirmCard` at the top of the file:
```python
    @on(SkillInstallConfirmCard.InstallDecided)
    def handle_console_skill_install_decided(
        self, event: SkillInstallConfirmCard.InstallDecided
    ) -> None:
        """Forward the user's install decision to the controller's worker thread."""
        event.stop()
        controller = self._console_chat_controller
        if controller is not None:
            controller.resolve_pending_skill_install(event.allow)
```

Wire the setter onto the controller in `_ensure_console_chat_controller` (after `set_pending_approval` at line 3313-3316):
```python
        self._console_chat_controller.set_pending_skill_install = (
            self._set_console_pending_skill_install
        )
```

- [ ] **Step 7: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_console_skill_install_confirm.py -v`
Expected: PASS (all tests). Confirm the chat screen still imports/loads: `.venv/bin/python -c "import tldw_chatbook.UI.Screens.chat_screen"`.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/Widgets/Chat_Widgets/skill_install_confirm_card.py tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_skill_install_confirm.py
git commit -m "feat(ui): SkillInstallConfirmCard + task-cards rendering + resume-state field"
```

---

### Task 7: End-to-end + full regression

**Files:**
- Test: `Tests/Skills/test_skill_remote_fetch.py` (extend)

**Interfaces:**
- Consumes: everything above, plus real `LocalSkillsService`/`SkillsScopeService`/`SkillTrustService` + `ServicePolicyEnforcer` (mirror `test_e2e_install_skill_from_github_tree_url_real_services`).

- [ ] **Step 1: Write the e2e test** — append to `Tests/Skills/test_skill_remote_fetch.py`. Drives `install_skill` through the real bridge + real services, with only the HTTP fetch + branch listing faked (keeping classify/enforce/re-root/import real):

```python
@pytest.mark.asyncio
async def test_e2e_agent_install_skill_confirm_allow(tmp_path, monkeypatch):
    """A scripted model calls install_skill(url); the confirm auto-allows;
    real services install the skill trust-pending on disk. Only the network
    (fetch_zip_bytes) and branch listing are faked; policy is REAL."""
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    from tldw_chatbook.Skills_Interop.skill_trust_store import (
        FileSkillTrustGenerationMarkerStore, SkillTrustStore,
    )
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState
    from tldw_chatbook.Utils.github_api_client import GitHubAPIClient

    trust_service = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    trust_service.unlock_with_passphrase("e2e-passphrase", salt=b"7" * 32)
    trust_service.bootstrap_trust()
    policy_enforcer = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
    )
    local_service = LocalSkillsService(
        store_dir=tmp_path, trust_service=trust_service, policy_enforcer=policy_enforcer,
    )
    scope_service = SkillsScopeService(
        local_service=local_service, server_service=None, policy_enforcer=policy_enforcer,
    )

    async def _fake_get_branches(self, owner, repo):
        return ["main", "master"]
    monkeypatch.setattr(GitHubAPIClient, "get_branches", _fake_get_branches)

    zip_bytes = _zipball(
        [("skills/demo/SKILL.md", "---\nname: demo\n---\nBody.\n"),
         ("skills/demo/references/api.md", "# API\n")],
        wrapper="superpowers-abc/",
    )

    async def _fake_fetch(url, *, token=None, transport=None, resolver=None):
        return zip_bytes
    monkeypatch.setattr(srf, "fetch_zip_bytes", _fake_fetch)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="install it")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")

    scripts = [
        [_fence("install_skill",
                {"url": "https://github.com/obra/superpowers/tree/main/skills/demo"})],
        ["Installed."],
    ]
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store, provider_gateway=_ChunkGateway(scripts),
        skills_service=scope_service,
    )
    _rid, outcome = bridge.run_reply(
        conversation_id="conv-e2e", session_id=session.id, resolution=object(),
        assistant_message_id=assistant.id, model="m", session_system_prompt="",
        agent_messages=[{"role": "user", "content": "install it"}],
        should_cancel=lambda: False,
        request_skill_install_confirm=lambda url: True,
    )
    assert outcome.status == "done"
    skill_dir = tmp_path / "skills" / "demo"
    assert (skill_dir / "SKILL.md").is_file()
    assert (skill_dir / "references" / "api.md").is_file()
    fetched = await scope_service.get_skill("demo", mode="local")
    assert fetched["trust_blocked"] is True


@pytest.mark.asyncio
async def test_e2e_agent_install_skill_confirm_deny(tmp_path, monkeypatch):
    """Denying the confirm installs nothing and never fetches."""
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as srf
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    from tldw_chatbook.Skills_Interop.skill_trust_store import (
        FileSkillTrustGenerationMarkerStore, SkillTrustStore,
    )
    from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
    from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
    from tldw_chatbook.runtime_policy.types import RuntimeSourceState

    trust_service = SkillTrustService(
        skills_dir=tmp_path / "skills",
        trust_store=SkillTrustStore(
            store_dir=tmp_path / "trust",
            marker_store=FileSkillTrustGenerationMarkerStore(tmp_path / "marker.json"),
        ),
    )
    trust_service.unlock_with_passphrase("e2e-passphrase", salt=b"7" * 32)
    trust_service.bootstrap_trust()
    policy_enforcer = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
    )
    local_service = LocalSkillsService(
        store_dir=tmp_path, trust_service=trust_service, policy_enforcer=policy_enforcer,
    )
    scope_service = SkillsScopeService(
        local_service=local_service, server_service=None, policy_enforcer=policy_enforcer,
    )

    async def _boom(*a, **k):
        raise AssertionError("fetch must not run on deny")
    monkeypatch.setattr(srf, "fetch_zip_bytes", _boom)

    db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
    store = ConsoleChatStore()
    session = store.ensure_session()
    store.append_message(session.id, role=ConsoleMessageRole.USER, content="install it")
    assistant = store.append_message(session.id, role=ConsoleMessageRole.ASSISTANT, content="")
    bridge = ConsoleAgentBridge(
        agent_runs_db=db, store=store,
        provider_gateway=_ChunkGateway(
            [[_fence("install_skill", {"url": "https://github.com/o/r"})], ["Cancelled."]]
        ),
        skills_service=scope_service,
    )
    _rid, outcome = bridge.run_reply(
        conversation_id="conv-e2e-deny", session_id=session.id, resolution=object(),
        assistant_message_id=assistant.id, model="m", session_system_prompt="",
        agent_messages=[{"role": "user", "content": "install it"}],
        should_cancel=lambda: False,
        request_skill_install_confirm=lambda url: False,
    )
    assert outcome.status == "done"
    assert not (tmp_path / "skills" / "demo").exists()
```

Note: this file already defines `_zipball`. Add these two local helpers at the top of the appended block (a local copy avoids a cross-file test import):

```python
from tldw_chatbook.Agents.agent_runtime import FENCE_OPEN


def _fence(name, args):
    return f"{FENCE_OPEN}\n{json.dumps({'name': name, 'arguments': args})}\n```"


class _ChunkGateway:
    """A provider gateway whose stream_chat replays a script per call index."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = 0

    async def stream_chat(self, resolution, messages, tools=None):
        chunks = self._scripts[self.calls]
        self.calls += 1
        for chunk in chunks:
            yield chunk
```

(`json` is already imported at the top of `test_skill_remote_fetch.py`.)

- [ ] **Step 2: Run the e2e**

Run: `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring .venv/bin/python -m pytest Tests/Skills/test_skill_remote_fetch.py -k e2e_agent_install -v`
Expected: PASS (2 tests).

- [ ] **Step 3: Full regression gate**

Run:
```
PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring .venv/bin/python -m pytest \
  Tests/Agents Tests/Skills Tests/Chat Tests/UI/test_console_skill_install_confirm.py \
  Tests/UI/test_console_mcp_approval.py Tests/Library/test_library_skills_state.py -q
```
Expected: 0 failures beyond known baselines (`Tests/RuntimePolicy` 2 pre-existing — not in this set; `test_skill_editor_canvas_scrolls_trust_panel_into_view` flaky; `Tests/Chat/test_anthropic_native_tools` pre-existing). Investigate anything else.

- [ ] **Step 4: Commit**

```bash
git add Tests/Skills/test_skill_remote_fetch.py
git commit -m "test(skills): e2e — agent installs a skill via install_skill with in-chat confirm"
```

---
