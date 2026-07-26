# Built-in Tool Packs — Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the agent runtime's built-in tool surface safe to grow, then land the pack architecture with a read-only `files` pack as its first customer.

**Architecture:** Phase 0 fixes six runtime properties that make a larger catalog unsafe or unreachable (tool ceiling, unbounded results, per-run-only timeouts, an untagged network risk class, silent MCP shadowing, and a turn budget sized for 20 rounds). Phase 1 adds `Agents/builtin_packs/` — a registry of tool packs with an injected service seam — and migrates TASK-584's ad-hoc `read_file`/`list_directory` wiring into it, adding `glob_files` and `grep_files` beside them. It closes by deleting the `ToolExecutor`, which has no callers left.

**Tech Stack:** Python ≥3.11, pytest, SQLite, Textual. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-25-builtin-tool-packs-design.md`

## Global Constraints

- Every tool returns `ToolResult`; tools never raise. `BuiltinToolProvider.invoke` already wraps.
- `BuiltinToolProvider()` constructed with no arguments must keep working — TASK-656's permissions enumerator relies on it.
- Every pack tool's `__init__` accepts `services: BuiltinToolServices | None = None`, and its `name`/`description`/`parameters`/`risk_tags` properties MUST NOT touch `services`. Enumeration constructs with `services=None`.
- **Default posture stays disabled.** TASK-584 shipped `read_file`/`list_directory` off by default. No task in this plan turns a tool on by default.
- **Do not widen the filesystem root.** The sandbox root stays `<user data dir>/tool_sandbox` (overridable via config). Spec §4.6 rule 1's workspace-rooting is explicitly NOT in this plan — it is a security-posture change requiring separate sign-off.
- Risk tag vocabulary is `HIGH_RISK_TAGS` in `MCP/permission_store.py`. Never write built-in permissions under `builtin:tldw_chatbook`; the namespace is `agent:builtin`.
- Run tests with the project venv: `source .venv/bin/activate` first. The `timeout` command is unavailable in this environment.
- Commit after every task. Never mark a task complete with failing tests.

## File Structure

**Phase 0 — modified only:**
- `tldw_chatbook/Agents/agent_models.py` — budget constants, `RunBudget` fields
- `tldw_chatbook/Agents/agent_runtime.py` — result truncation at the history-append seam
- `tldw_chatbook/Agents/agent_service.py` — per-tool timeout resolution
- `tldw_chatbook/Agents/tool_catalog.py` — registry timeout lookup
- `tldw_chatbook/MCP/permission_store.py` — `network` tag
- `tldw_chatbook/Chat/console_agent_bridge.py` — shadowing visibility, Console budget

**Phase 1 — created:**
- `tldw_chatbook/Agents/builtin_services.py` — `BuiltinToolServices` frozen dataclass. Separate module so `tool_catalog` stays dependency-light.
- `tldw_chatbook/Agents/builtin_packs/__init__.py` — `PACKS` registry + resolution
- `tldw_chatbook/Agents/builtin_packs/files.py` — the `files` pack
- `tldw_chatbook/Utils/sensitive_paths.py` — shared path denylist (Phase 4's `run_command` reuses it verbatim)
- `tldw_chatbook/Tools/base.py` — relocated `Tool` ABC

**Phase 1 — deleted:**
- `tldw_chatbook/Tools/code_audit_tool.py`, `ToolExecutor`/`ToolResultCache` in `tool_executor.py`, `Tools_Settings_Window`'s tool switches

---

## Task 1: Raise the tool ceiling and disclosure threshold

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py:48` (`DIRECT_DISCLOSE_THRESHOLD`), `:143` (`max_active_tools`)
- Test: `Tests/Agents/test_agent_models.py`

**Interfaces:**
- Consumes: nothing
- Produces: `DIRECT_DISCLOSE_THRESHOLD == 16`, `RunBudget.max_active_tools == 24`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_agent_models.py`:

```python
def test_direct_disclose_threshold_admits_a_three_pack_catalog():
    """A files+corpus+authoring set is 14 tools; it must disclose directly.

    Below the threshold `initial_disclosure` skips find_tools/load_tools
    entirely, which is the point: those two round trips are pure overhead
    repeated on every user message.
    """
    from tldw_chatbook.Agents.agent_models import DIRECT_DISCLOSE_THRESHOLD

    assert DIRECT_DISCLOSE_THRESHOLD >= 14


def test_max_active_tools_clears_the_disclosure_threshold():
    """Everything directly disclosed must fit in the active set.

    `initial_disclosure` truncates to `max_active_tools`, so a ceiling below
    the threshold would silently drop tools it just decided to disclose.
    """
    from tldw_chatbook.Agents.agent_models import (
        DIRECT_DISCLOSE_THRESHOLD,
        RunBudget,
    )

    assert RunBudget().max_active_tools >= DIRECT_DISCLOSE_THRESHOLD
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_agent_models.py -k "threshold or ceiling" -v`
Expected: FAIL — `assert 8 >= 14`

- [ ] **Step 3: Apply the constant changes**

In `agent_models.py`, replace line 48:

```python
DIRECT_DISCLOSE_THRESHOLD = 16
```

And in `RunBudget`, replace the `max_active_tools` line:

```python
    max_active_tools: int = 24
```

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ -v`
Expected: PASS, including the whole existing Agents suite. If any existing test hardcodes `8`, update it to derive from the constant rather than restating the number.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py Tests/Agents/test_agent_models.py
git commit -m "feat(agents): raise tool ceiling to 24 and disclosure threshold to 16"
```

---

## Task 2: Cap tool results at the history-append seam

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (add `max_tool_result_chars`), `tldw_chatbook/Agents/agent_runtime.py:265` and `:613-616`
- Test: `Tests/Agents/test_agent_runtime.py`

**Interfaces:**
- Consumes: `RunBudget` from Task 1
- Produces: `RunBudget.max_tool_result_chars: int = 16000`; `agent_runtime._truncate_tool_result(content: str, max_chars: int, tool_name: str) -> str`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_agent_runtime.py`:

```python
def test_truncate_tool_result_bounds_content_and_names_a_continuation():
    from tldw_chatbook.Agents.agent_runtime import _truncate_tool_result

    out = _truncate_tool_result("x" * 5000, 100, "grep_files")

    assert len(out) < 5000
    assert out.startswith("x" * 100)
    assert "grep_files" in out
    assert "5000" in out


def test_truncate_tool_result_is_a_noop_under_the_cap():
    from tldw_chatbook.Agents.agent_runtime import _truncate_tool_result

    assert _truncate_tool_result("small", 100, "t") == "small"


def test_truncate_tool_result_zero_means_unlimited():
    """0 restores today's behaviour exactly, for an operator who wants it."""
    from tldw_chatbook.Agents.agent_runtime import _truncate_tool_result

    assert _truncate_tool_result("x" * 5000, 0, "t") == "x" * 5000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_agent_runtime.py -k truncate_tool_result -v`
Expected: FAIL with `ImportError: cannot import name '_truncate_tool_result'`

- [ ] **Step 3: Add the budget field**

In `agent_models.py`, add to `RunBudget` immediately after `max_subagent_result_chars`:

```python
    # Ceiling on how much of ONE tool result enters conversation history.
    # Enforced at the history-append seam (agent_runtime), NOT per tool, so
    # built-in, MCP, and skill results are all bounded by the same rule.
    # Derived from max_subagent_result_chars (4000): four times a whole
    # sub-agent result is generous for a single call while keeping a
    # 30-turn run tractable. 0 = unlimited, restoring pre-cap behaviour.
    max_tool_result_chars: int = 16000
```

- [ ] **Step 4: Add the truncation helper**

In `agent_runtime.py`, immediately above `_append_tool_result` (line 265):

```python
def _truncate_tool_result(content: str, max_chars: int, tool_name: str) -> str:
    """Bound one tool result before it enters history.

    Applied at the append seam rather than inside each tool so a tool that
    forgets to paginate cannot blow the context, and so MCP and skill
    results are covered by the same rule as built-ins.

    Args:
        content: The tool's full result text.
        max_chars: Ceiling from ``RunBudget.max_tool_result_chars``; 0 or
            negative means unlimited.
        tool_name: Named in the trailer so the model knows which call was
            cut and can re-issue it more narrowly.

    Returns:
        ``content`` unchanged when under the cap or when unlimited;
        otherwise the first ``max_chars`` characters plus a trailer stating
        the original length and how to retrieve the remainder.
    """
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    return (
        content[:max_chars]
        + f"\n\n[truncated: {tool_name} returned {len(content)} characters; "
        f"showing the first {max_chars}. Re-issue the call with a narrower "
        f"query, or use the tool's offset/limit arguments to read the rest.]"
    )
```

- [ ] **Step 5: Apply the cap at the append seam**

In `agent_runtime.py`, replace lines 613-616 (the `content = ...` assignment through `_append_tool_result`):

```python
                content = result.content if result.ok else f"ERROR: {result.error}"
                content = _truncate_tool_result(
                    content, budget.max_tool_result_chars, call.name
                )

            add(STEP_TOOL_RESULT, tool_name=call.name, result=content[:2000])
            _append_tool_result(messages, call, content)
```

Note the truncation happens inside the same `else` block that assigns `content`, so the existing indentation is preserved.

- [ ] **Step 6: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/test_agent_runtime.py
git commit -m "feat(agents): cap tool results at the history-append seam"
```

---

## Task 3: Per-tool timeout override

**Files:**
- Modify: `tldw_chatbook/Tools/tool_executor.py` (`Tool` ABC), `tldw_chatbook/Agents/tool_catalog.py`, `tldw_chatbook/Agents/agent_service.py:399-406`
- Test: `Tests/Agents/test_agent_service.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Tool.timeout_seconds -> float` (property, default `0.0`); `BuiltinToolProvider.timeout_for(tool_id) -> float | None`; `ToolCatalogRegistry.timeout_for(name) -> float | None`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_agent_service.py`:

```python
def test_registry_timeout_for_reports_a_tools_own_ceiling():
    from tldw_chatbook.Agents.tool_catalog import (
        BuiltinToolProvider,
        ToolCatalogRegistry,
    )
    from tldw_chatbook.Tools.tool_executor import Tool

    class _Slow(Tool):
        @property
        def name(self) -> str:
            return "slow_thing"

        @property
        def description(self) -> str:
            return "d"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {}}

        @property
        def timeout_seconds(self) -> float:
            return 42.0

        async def execute(self, **kwargs):
            return {}

    provider = BuiltinToolProvider()
    provider._tools["slow_thing"] = _Slow()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)

    assert registry.timeout_for("slow_thing") == 42.0
    assert registry.timeout_for("calculator") is None
    assert registry.timeout_for("no_such_tool") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_agent_service.py -k timeout_for -v`
Expected: FAIL with `AttributeError: 'ToolCatalogRegistry' object has no attribute 'timeout_for'`

- [ ] **Step 3: Add the ABC property**

In `tool_executor.py`, add to the `Tool` class immediately after the `risk_tags` property:

```python
    @property
    def timeout_seconds(self) -> float:
        """Per-call wall-clock ceiling, or 0 to use the run's default.

        Concrete with a 0 default so every existing subclass is unchanged.
        A tool whose real work legitimately outlasts
        ``RunBudget.max_tool_call_seconds`` (ingestion, transcription)
        raises this; a tool that must be cut short sooner (``run_command``)
        lowers it. Note the timeout ABANDONS the worker thread rather than
        killing it, so a tool raising this must be idempotent or must say
        so in its timeout message.

        Returns:
            Seconds, or 0.0 to defer to the run budget.
        """
        return 0.0
```

- [ ] **Step 4: Add the provider and registry lookups**

In `tool_catalog.py`, add to `BuiltinToolProvider` after `tool_for`:

```python
    def timeout_for(self, tool_id: str) -> float | None:
        """Return this tool's own timeout ceiling, if it declares one."""
        tool = self._tools.get(tool_id.split(":", 1)[-1])
        seconds = float(getattr(tool, "timeout_seconds", 0.0) or 0.0)
        return seconds if seconds > 0 else None
```

And add to `ToolCatalogRegistry`:

```python
    def timeout_for(self, name: str) -> float | None:
        """Resolve a tool's per-call timeout override by LLM-facing name.

        Duck-typed like the rest of the provider interface: a provider that
        does not implement ``timeout_for`` simply has no overrides, so MCP
        and skill tools keep using the run budget unchanged.

        Args:
            name: The tool name the model called.

        Returns:
            A positive seconds value, or None to use the run default.
        """
        tool_id = self.resolve_name(name)
        if tool_id is None:
            return None
        provider = self._owner_and_id(tool_id)
        getter = getattr(provider, "timeout_for", None)
        return getter(tool_id) if getter is not None else None
```

- [ ] **Step 5: Consume it in the invoke seam**

In `agent_service.py`, replace the timeout resolution inside `invoke_tool` (lines 399-400):

```python
            timeout = self.registry.timeout_for(call.name) or (
                config.budget.max_tool_call_seconds
            )
```

- [ ] **Step 6: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Tools/tool_executor.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Agents/agent_service.py Tests/Agents/test_agent_service.py
git commit -m "feat(agents): per-tool timeout override resolved through the registry"
```

---

## Task 4: Add the `network` risk tag

**Files:**
- Modify: `tldw_chatbook/MCP/permission_store.py:69`
- Test: `Tests/Agents/test_builtin_tool_gate.py`

**Interfaces:**
- Consumes: nothing
- Produces: `HIGH_RISK_TAGS == frozenset({"mutates", "process", "network"})`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_builtin_tool_gate.py`:

```python
class _Networked(Tool):
    @property
    def name(self) -> str:
        return "fetch_thing"

    @property
    def description(self) -> str:
        return "d"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def risk_tags(self) -> tuple[str, ...]:
        return ("network",)

    async def execute(self, **kwargs):
        return {}


def test_network_tag_floors_inherited_allow_to_ask():
    """Egress is the exfiltration leg of a prompt-injection chain.

    An untagged read-only fetch would resolve to the built-in allow floor
    and execute silently, so `network` joins HIGH_RISK_TAGS.
    """
    from tldw_chatbook.Agents.builtin_tool_gate import tool_ref
    from tldw_chatbook.MCP.permission_store import resolve_builtin_state

    state = resolve_builtin_state({}, tool_ref(_Networked()))

    assert state.state == "ask"
    assert state.risk_floored is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_builtin_tool_gate.py -k network_tag -v`
Expected: FAIL — state is `allow`, `risk_floored` is False

- [ ] **Step 3: Extend the vocabulary**

In `permission_store.py`, replace line 69:

```python
# task-545 P2: `network` joins the vocabulary so an egress-capable tool
# cannot execute silently. This is SHARED with MCP resolution (see the two
# HIGH_RISK_TAGS uses below), so an MCP tool already declaring `network`
# begins flooring to `ask` — a deliberate, documented behaviour change.
HIGH_RISK_TAGS = frozenset({"mutates", "process", "network"})
```

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ Tests/MCP/ -v`
Expected: PASS. The MCP resolver's existing tests must pass untouched — if one fails, it is asserting on a tool tagged `network`, which is the intended behaviour change; update that test's expectation and note it in the commit body.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/MCP/permission_store.py Tests/Agents/test_builtin_tool_gate.py
git commit -m "feat(mcp): add network to HIGH_RISK_TAGS so egress tools floor to ask"
```

---

## Task 5: Make MCP name shadowing visible

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`_non_colliding_mcp_names`)
- Test: `Tests/Chat/test_console_agent_bridge.py` (create if absent)

**Interfaces:**
- Consumes: nothing
- Produces: `console_agent_bridge.shadowed_mcp_names(mcp_provider, collision_names) -> tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create or append to `Tests/Chat/test_console_agent_bridge.py`:

```python
from tldw_chatbook.Agents.agent_models import ToolCatalogEntry


class _StubProvider:
    def __init__(self, names):
        self._names = names

    def list_catalog(self):
        return [
            ToolCatalogEntry(
                id=f"mcp:{n}", name=n, one_line_description="d", source="mcp"
            )
            for n in self._names
        ]


def test_shadowed_mcp_names_reports_what_the_filter_drops():
    """A user's configured MCP tool must never vanish silently.

    Built-ins keep winning the collision — inverting that would let a
    compromised server name-squat an audited built-in — so the shadowing
    is surfaced instead.
    """
    from tldw_chatbook.Chat.console_agent_bridge import (
        _non_colliding_mcp_names,
        shadowed_mcp_names,
    )

    provider = _StubProvider(["read_file", "weather"])
    collisions = frozenset({"read_file"})

    assert shadowed_mcp_names(provider, collisions) == ("read_file",)
    assert _non_colliding_mcp_names(provider, collisions) == ("weather",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Chat/test_console_agent_bridge.py -k shadowed -v`
Expected: FAIL with `ImportError: cannot import name 'shadowed_mcp_names'`

- [ ] **Step 3: Add the reporter and log the drop**

In `console_agent_bridge.py`, add immediately after `_non_colliding_mcp_names`:

```python
def shadowed_mcp_names(
    mcp_provider: Any,
    collision_names: frozenset[str] | set[str],
) -> tuple[str, ...]:
    """MCP tool names this run drops because a built-in owns the name.

    The exact complement of ``_non_colliding_mcp_names``. Built-ins win
    collisions deliberately -- letting the MCP side win would let a
    compromised server name-squat an audited built-in like ``write_file``
    and intercept calls the user believes are gated -- but a user whose
    configured tool silently stops working has no way to discover why.
    TASK-656's permissions view renders these rows.

    Args:
        mcp_provider: A composed ``MCPToolProvider`` (or test double).
        collision_names: Names owned by builtins, runtime tools, or skills.

    Returns:
        The dropped names, in catalog order.
    """
    return tuple(
        entry.name
        for entry in mcp_provider.list_catalog()
        if entry.name in collision_names
    )
```

Then, inside `_compose_run_registry_and_allowed` where `_non_colliding_mcp_names` is called, add immediately after that call:

```python
        for shadowed in shadowed_mcp_names(mcp_provider, collision_names):
            logger.warning(
                "MCP tool {name} is shadowed by a built-in of the same name "
                "and is not offered this run",
                name=shadowed,
            )
```

Match the local variable names already in scope at that call site.

- [ ] **Step 4: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Chat/test_console_agent_bridge.py Tests/Agents/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/test_console_agent_bridge.py
git commit -m "feat(agents): surface MCP tools shadowed by built-in name collisions"
```

---

## Task 6: Resize the Console budget for 30 turns

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:119-137`, `tldw_chatbook/Agents/agent_models.py:54` (`DEFAULT_MAX_MODEL_TURNS`)
- Test: `Tests/Agents/test_agent_runtime.py` (existing `test_console_budget_step_cap_admits_a_full_model_turn_run` self-validates)

**Interfaces:**
- Consumes: `RunBudget.max_total_tokens` (existing field)
- Produces: `CONSOLE_MAX_MODEL_TURNS == 30`, `CONSOLE_MAX_STEPS == 96`, `CONSOLE_MAX_WALL_SECONDS == 1800.0`, `CONSOLE_MAX_TOTAL_TOKENS == 1_000_000`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_agent_runtime.py`:

```python
def test_console_budget_bounds_spend_not_only_time():
    """Sub-agents inherit the turn budget by an explicit operator decision.

    Worst case is max_model_turns * (1 + max_subagents) provider turns for
    one message -- 90 at 30/2. The wall clock bounds that in TIME but not
    in SPEND, so the Console budget carries a token ceiling.
    """
    from tldw_chatbook.Chat.console_agent_bridge import CONSOLE_RUN_BUDGET

    assert CONSOLE_RUN_BUDGET.max_model_turns == 30
    assert CONSOLE_RUN_BUDGET.max_total_tokens > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_agent_runtime.py -k console_budget -v`
Expected: FAIL — `assert 20 == 30`

- [ ] **Step 3: Update the Console constants**

In `console_agent_bridge.py`, replace the constant block (lines 119-137). Update the preceding comment's `max_model_turns=20` / `max_steps=64` / `max_wall_seconds=1200` bullets to the new values and derivations, then:

```python
#: Tool-calling rounds the Console agent gets per user message. THE primary
#: limiter -- the three constants below exist to keep it reachable and to
#: bound what it costs.
CONSOLE_MAX_MODEL_TURNS = 30

#: Step backstop. A fence round costs 3 steps (STEP_MODEL + STEP_TOOL_CALL +
#: STEP_TOOL_RESULT) and the wrap-up reply costs 1, so N turns need
#: 3*(N-1)+1 steps -- 88 at N=30. 96 clears that while staying a real
#: backstop for native multi-call batches (1 + 2N steps per turn).
#: `test_console_budget_step_cap_admits_a_full_model_turn_run` fails if this
#: ever drops below the derived minimum.
CONSOLE_MAX_STEPS = 96

#: Wall-clock backstop, at the slow local-model pace this gate exercises
#: (25-50s per turn x CONSOLE_MAX_MODEL_TURNS = 750-1500s at N=30).
CONSOLE_MAX_WALL_SECONDS = 1800.0

#: Cumulative prompt+completion spend ceiling. Sub-agents INHERIT the turn
#: and step budget (agent_models.clamp_child_budget, operator decision
#: 2026-07-25), so one message can reach 30 * (1 + max_subagents) = 90
#: provider turns. The wall clock bounds that in time but not in spend; at a
#: ~20k-token working prompt a runaway approaches ~1.8M tokens. This stops
#: that while sitting far above any normal 30-turn run.
CONSOLE_MAX_TOTAL_TOKENS = 1_000_000

CONSOLE_RUN_BUDGET = RunBudget(
    max_steps=CONSOLE_MAX_STEPS,
    max_wall_seconds=CONSOLE_MAX_WALL_SECONDS,
    max_model_turns=CONSOLE_MAX_MODEL_TURNS,
    max_total_tokens=CONSOLE_MAX_TOTAL_TOKENS,
)
```

- [ ] **Step 4: Update the engine default**

In `agent_models.py`, replace `DEFAULT_MAX_MODEL_TURNS = 20` with:

```python
DEFAULT_MAX_MODEL_TURNS = 30
```

Leave the surrounding comment's "provably unreachable at engine defaults" note intact — it still holds, because engine `max_steps` remains 8 and every model turn appends at least one step.

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ Tests/Chat/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py tldw_chatbook/Agents/agent_models.py Tests/Agents/test_agent_runtime.py
git commit -m "feat(console): raise agent turn cap to 30 and bound run spend"
```

---

## Task 7: `BuiltinToolServices` injection seam

**Files:**
- Create: `tldw_chatbook/Agents/builtin_services.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (`BuiltinToolProvider.__init__`)
- Test: `Tests/Agents/test_tool_catalog.py`

**Interfaces:**
- Consumes: nothing
- Produces: `BuiltinToolServices` frozen dataclass with fields `notes_library`, `media_reading`, `prompt_service`, `chunk_service`, `rag_search` (all `Any | None = None`); `BuiltinToolProvider(gate=None, services=None)`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_tool_catalog.py`:

```python
def test_provider_accepts_services_and_still_works_without_them():
    """TASK-656's permissions enumerator builds a bare provider.

    Services must therefore stay optional: metadata is readable with
    services=None, and only execute() needs them.
    """
    from tldw_chatbook.Agents.builtin_services import BuiltinToolServices
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    bare = BuiltinToolProvider()
    assert {e.name for e in bare.list_catalog()} >= {"calculator", "datetime"}

    services = BuiltinToolServices(notes_library=object())
    injected = BuiltinToolProvider(services=services)
    assert injected.services is services
    assert bare.services is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_tool_catalog.py -k accepts_services -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.builtin_services'`

- [ ] **Step 3: Create the services module**

Create `tldw_chatbook/Agents/builtin_services.py`:

```python
# tldw_chatbook/Agents/builtin_services.py
"""Local service seams injected into built-in pack tools.

A separate module so ``tool_catalog`` stays dependency-light: importing the
provider must not drag in the notes, media, or RAG subsystems.

**Contract.** Every service assigned here MUST be:

1. thread-safe -- tools execute on a fresh per-call daemon thread
   (``agent_service._call_with_timeout``), so a service holding
   non-thread-safe state will corrupt under concurrent tool calls;
2. free of event-loop-bound state -- no ``httpx.AsyncClient`` or other
   object bound to the app's loop, because ``BuiltinToolProvider.invoke``
   drives async tools through ``asyncio.run`` on that fresh thread;
3. free of Textual/UI handles -- a worker thread must never touch widgets.

Violations surface as failures that are miserable to diagnose from a
worker thread, which is why the contract is stated rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BuiltinToolServices:
    """Per-run handles the built-in packs operate against.

    All fields default to ``None`` so a bare instance is valid: metadata
    enumeration (TASK-656) constructs tools with ``services=None`` and only
    reads their name/description/parameters/risk_tags.

    Attributes:
        notes_library: ``Notes/Notes_Library.py`` handle.
        media_reading: ``Media/local_media_reading_service.py`` handle.
        prompt_service: ``Prompt_Management/local_prompt_service.py`` handle.
        chunk_service: ``Chunking`` entry point.
        rag_search: RAG search entry point resolved for the active profile.
    """

    notes_library: Any | None = None
    media_reading: Any | None = None
    prompt_service: Any | None = None
    chunk_service: Any | None = None
    rag_search: Any | None = None
```

- [ ] **Step 4: Accept services on the provider**

In `tool_catalog.py`, change `BuiltinToolProvider.__init__`'s signature and record the value:

```python
    def __init__(self, gate: Any | None = None, services: Any | None = None) -> None:
        self.services = services
        self._tools = {t.name: t for t in (CalculatorTool(), DateTimeTool())}
```

Leave the rest of `__init__` (the TASK-584 gate loop and the `self._gate = gate` line) unchanged — **Task 9** replaces the gate loop.

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/builtin_services.py tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_tool_catalog.py
git commit -m "feat(agents): add BuiltinToolServices injection seam"
```

---

## Task 8: Pack registry and the `files` pack

**Files:**
- Create: `tldw_chatbook/Agents/builtin_packs/__init__.py`, `tldw_chatbook/Agents/builtin_packs/files.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (replace the TASK-584 gate loop)
- Test: `Tests/Agents/test_builtin_packs.py` (create)

**Interfaces:**
- Consumes: `BuiltinToolServices` from Task 7
- Produces: `builtin_packs.PACKS: dict[str, ModuleType]`; `builtin_packs.pack_tool_classes(enabled: frozenset[str]) -> tuple[type, ...]`; `builtin_packs.files.TOOLS`, `.REQUIRES`; classes `files.ReadFile`, `files.ListDirectory`

- [ ] **Step 1: Write the failing test**

Create `Tests/Agents/test_builtin_packs.py`:

```python
import pytest

from tldw_chatbook.Agents.builtin_services import BuiltinToolServices


def test_files_pack_declares_its_tools_and_no_optional_deps():
    from tldw_chatbook.Agents.builtin_packs import files

    assert files.REQUIRES == ()
    assert {c.__name__ for c in files.TOOLS} == {"ReadFile", "ListDirectory"}


def test_every_pack_tool_constructs_with_services_none():
    """The metadata contract: enumeration never has live services."""
    from tldw_chatbook.Agents.builtin_packs import PACKS

    for pack in PACKS.values():
        for cls in pack.TOOLS:
            tool = cls(services=None)
            assert isinstance(tool.name, str) and tool.name
            assert isinstance(tool.description, str) and tool.description
            assert isinstance(tool.parameters, dict)
            assert isinstance(tool.risk_tags, tuple)


def test_pack_tool_classes_returns_only_enabled_packs():
    from tldw_chatbook.Agents.builtin_packs import pack_tool_classes

    assert pack_tool_classes(frozenset()) == ()
    assert len(pack_tool_classes(frozenset({"files"}))) == 2
    assert pack_tool_classes(frozenset({"nope"})) == ()


def test_services_are_accepted_but_unused_by_file_tools():
    from tldw_chatbook.Agents.builtin_packs import files

    tool = files.ReadFile(services=BuiltinToolServices())
    assert tool.name == "read_file"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_builtin_packs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.builtin_packs'`

- [ ] **Step 3: Create the `files` pack**

Create `tldw_chatbook/Agents/builtin_packs/files.py`:

```python
# tldw_chatbook/Agents/builtin_packs/files.py
"""The `files` pack: sandbox-rooted filesystem reads.

Wraps the existing implementations in ``Tools/file_operation_tools.py``
rather than reimplementing them. Those tools already confine every path to
``_tool_sandbox_root()``; this pack does not widen that (see the plan's
Global Constraints -- workspace-rooting is a separate, signed-off change).

The thin subclasses exist to satisfy the pack contract: every pack tool
constructs as ``cls(services=...)`` and its metadata properties never touch
services, so TASK-656's enumerator can describe them with ``services=None``.
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Tools.file_operation_tools import ListDirectoryTool, ReadFileTool


class ReadFile(ReadFileTool):
    """`read_file`, constructed under the pack contract."""

    def __init__(self, services: Any | None = None) -> None:
        super().__init__()
        self.services = services


class ListDirectory(ListDirectoryTool):
    """`list_directory`, constructed under the pack contract."""

    def __init__(self, services: Any | None = None) -> None:
        super().__init__()
        self.services = services


#: Tool classes this pack contributes, in catalog order.
TOOLS: tuple[type, ...] = (ReadFile, ListDirectory)

#: Optional-dependency feature names required for this pack to appear.
#: Empty: the file tools use only the standard library.
REQUIRES: tuple[str, ...] = ()
```

- [ ] **Step 4: Create the registry**

Create `tldw_chatbook/Agents/builtin_packs/__init__.py`:

```python
# tldw_chatbook/Agents/builtin_packs/__init__.py
"""Registry of built-in tool packs.

A pack groups tools the user enables together. Packs bound the catalog so
it stays near ``DIRECT_DISCLOSE_THRESHOLD``, and they give the permission
gate a coarse consent surface above the per-tool one.

Each pack module exports ``TOOLS`` (tool classes, catalog order) and
``REQUIRES`` (optional-dependency feature names). A pack whose
dependencies are unmet is ABSENT from the catalog rather than present and
failing at invoke -- the model must not spend turns discovering a tool is
broken.
"""

from __future__ import annotations

from types import ModuleType

from loguru import logger

from . import files

#: pack name -> module. Add new packs here.
PACKS: dict[str, ModuleType] = {"files": files}


def pack_available(pack: ModuleType) -> bool:
    """Whether every optional dependency this pack declares is installed.

    Args:
        pack: A pack module exporting ``REQUIRES``.

    Returns:
        True when ``REQUIRES`` is empty or every named feature resolves.
    """
    requires = getattr(pack, "REQUIRES", ())
    if not requires:
        return True
    from tldw_chatbook.Utils.optional_deps import check_dependency

    return all(check_dependency(name) for name in requires)


def pack_tool_classes(enabled: frozenset[str]) -> tuple[type, ...]:
    """Tool classes contributed by the enabled, available packs.

    Unknown names are ignored rather than raising: a config naming a pack
    from a newer release must not break the run.

    Args:
        enabled: Pack names the user has switched on.

    Returns:
        Tool classes in ``PACKS`` iteration order, then pack order.
    """
    classes: list[type] = []
    for name, pack in PACKS.items():
        if name not in enabled:
            continue
        if not pack_available(pack):
            logger.info("Built-in pack {name} hidden: missing dependencies", name=name)
            continue
        classes.extend(pack.TOOLS)
    return tuple(classes)
```

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_builtin_packs.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/builtin_packs/ Tests/Agents/test_builtin_packs.py
git commit -m "feat(agents): add built-in tool pack registry and files pack"
```

---

## Task 9: Wire packs into the provider with a deprecated `[tools]` fallback

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py:194-214` (replace the TASK-584 loop)
- Create: `tldw_chatbook/Agents/builtin_pack_config.py`
- Test: `Tests/Agents/test_builtin_pack_config.py` (create), `Tests/Agents/test_builtin_file_tools.py` (update)

**Interfaces:**
- Consumes: `pack_tool_classes` from Task 8
- Produces: `builtin_pack_config.enabled_packs() -> frozenset[str]`

- [ ] **Step 1: Write the failing test**

Create `Tests/Agents/test_builtin_pack_config.py`:

```python
import pytest


@pytest.fixture
def cli_setting(monkeypatch):
    """Drive get_cli_setting the way the app really reads it."""
    values = {}
    import tldw_chatbook.Agents.builtin_pack_config as mod

    def fake(section, key=None, default=None):
        return values.get((section, key), default)

    monkeypatch.setattr(mod, "get_cli_setting", fake)
    return values


def test_defaults_to_no_packs_enabled(cli_setting):
    """TASK-584 shipped these tools OFF. Restructuring must not turn them on."""
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    assert enabled_packs() == frozenset()


def test_reads_the_pack_list(cli_setting):
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("agent_tools", "enabled_packs")] = ["files"]
    assert enabled_packs() == frozenset({"files"})


def test_legacy_tools_flags_enable_the_files_pack(cli_setting):
    """A user who already set read_file_enabled must not be switched off."""
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("tools", "read_file_enabled")] = True
    assert enabled_packs() == frozenset({"files"})


def test_explicit_pack_list_wins_over_legacy_flags(cli_setting):
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("agent_tools", "enabled_packs")] = []
    cli_setting[("tools", "read_file_enabled")] = True
    assert enabled_packs() == frozenset()


def test_non_list_pack_setting_is_ignored(cli_setting):
    """Hand-edited config must never crash a run."""
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    cli_setting[("agent_tools", "enabled_packs")] = "files"
    assert enabled_packs() == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_builtin_pack_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Agents.builtin_pack_config'`

- [ ] **Step 3: Create the config reader**

Create `tldw_chatbook/Agents/builtin_pack_config.py`:

```python
# tldw_chatbook/Agents/builtin_pack_config.py
"""Which built-in tool packs the user has enabled.

Reads ``[agent_tools] enabled_packs``. Uses ``get_cli_setting``'s 3-arg
form deliberately: the 2-arg form's second positional slot carries the
DEFAULT, not a key, and mis-slotting it there is exactly the defect
TASK-547 records in ``get_tool_executor()``.

Back-compat: TASK-584 shipped per-tool ``[tools] read_file_enabled`` /
``list_directory_enabled`` flags. Those keep working as a deprecated
fallback so an existing user is never silently switched off, but an
explicit ``enabled_packs`` list always wins -- including an explicitly
empty one, which means "no packs" rather than "fall back".
"""

from __future__ import annotations

from loguru import logger

from tldw_chatbook.config import get_cli_setting

#: `[tools]` flags that used to enable individual file tools, and the pack
#: that now owns them.
_LEGACY_FILE_FLAGS = ("read_file_enabled", "list_directory_enabled")

_MISSING = object()


def enabled_packs() -> frozenset[str]:
    """Pack names the user has switched on.

    Returns:
        The configured pack names; an empty set when nothing is enabled.
        Defaults to empty -- built-in file tools ship disabled (TASK-584)
        and this function must not change that posture.
    """
    configured = get_cli_setting("agent_tools", "enabled_packs", _MISSING)
    if configured is not _MISSING:
        if not isinstance(configured, list):
            logger.warning(
                "[agent_tools] enabled_packs must be a list; ignoring {value!r}",
                value=configured,
            )
            return frozenset()
        return frozenset(str(name) for name in configured)

    if any(get_cli_setting("tools", flag, False) for flag in _LEGACY_FILE_FLAGS):
        logger.warning(
            "[tools] {flags} are deprecated; set [agent_tools] enabled_packs = "
            '["files"] instead. Enabling the files pack for now.',
            flags=", ".join(_LEGACY_FILE_FLAGS),
        )
        return frozenset({"files"})

    return frozenset()
```

- [ ] **Step 4: Replace the TASK-584 loop in the provider**

In `tool_catalog.py`, replace the whole `for gate_key, factory_name in (...)` block (lines 194-214) with:

```python
        # task-545 P2: pack-resolved built-ins replace TASK-584's per-tool
        # [tools] gating. Same tools, same sandbox root, same
        # disabled-by-default posture -- the enablement moved up a level to
        # the pack. Import is local so `tool_catalog` stays importable
        # without the packs' own dependencies.
        try:
            from .builtin_pack_config import enabled_packs
            from .builtin_packs import pack_tool_classes

            packs = enabled_packs()
            # Pack availability varies per machine (optional deps), so the
            # resolved set is logged: without it, whether progressive
            # disclosure even engages differs between users and bug reports
            # become unreproducible (spec 4.1).
            logger.info("Built-in packs resolved for run: {packs}", packs=sorted(packs))
            for cls in pack_tool_classes(packs):
                tool = cls(services=services)
                self._tools[tool.name] = tool
        except Exception:  # noqa: BLE001 — an unavailable pack is just absent
            logger.warning("Built-in pack resolution failed; packs unavailable")
```

Add `from loguru import logger` to `tool_catalog.py`'s imports if it is not already present.

- [ ] **Step 5: Add the UNMOCKED config integration test**

Spec §4.3 requires this specifically: every test above patches `get_cli_setting`, and a
patched reader cannot catch a section the real reader cannot reach. That is precisely
the TASK-547 class of defect. Append to `Tests/Agents/test_builtin_pack_config.py`:

```python
def test_enabled_packs_reads_a_real_config_file(tmp_path, monkeypatch):
    """No mocks: prove the key is reachable the way the app reads it.

    TASK-547 shipped a config section no reader could reach, and every
    mocked test of it would have passed. This test would have failed.
    """
    import tldw_chatbook.config as config_module
    from tldw_chatbook.Agents.builtin_pack_config import enabled_packs

    config_path = tmp_path / "config.toml"
    config_path.write_text('[agent_tools]\nenabled_packs = ["files"]\n')
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert enabled_packs() == frozenset({"files"})
```

If `load_cli_config_and_ensure_existence` does not accept `force_reload`, clear whatever
module-level cache it uses instead — the requirement is that this test reads the real
TOML through the real accessor, with nothing patched between them.

- [ ] **Step 6: Update the TASK-584 test to the new config surface**

In `Tests/Agents/test_builtin_file_tools.py`, change the `tools_config` fixture to drive `[agent_tools] enabled_packs` instead of the `[tools]` flags, patching `tldw_chatbook.Agents.builtin_pack_config.get_cli_setting`. Keep every existing assertion about the tools appearing in the catalog — the behaviour under test is unchanged, only the switch that turns it on has moved.

- [ ] **Step 7: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Agents/builtin_pack_config.py tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_builtin_pack_config.py Tests/Agents/test_builtin_file_tools.py
git commit -m "feat(agents): resolve built-in tools from packs with a legacy [tools] fallback"
```

---

## Task 10: Shared sensitive-path denylist

**Files:**
- Create: `tldw_chatbook/Utils/sensitive_paths.py`
- Modify: `tldw_chatbook/Tools/file_operation_tools.py` (apply in the shared validator)
- Test: `Tests/Utils/test_sensitive_paths.py` (create)

**Interfaces:**
- Consumes: nothing
- Produces: `sensitive_paths.is_sensitive_path(candidate: Path) -> bool`

- [ ] **Step 1: Write the failing test**

Create `Tests/Utils/test_sensitive_paths.py`:

```python
from pathlib import Path

import pytest

from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path


@pytest.mark.parametrize(
    "path",
    [
        "~/.ssh/id_rsa",
        "~/.aws/credentials",
        "~/.gnupg/secring.gpg",
        "~/.config/tldw_cli/config.toml",
        "~/.config/tldw_cli/mcp_permissions.json",
    ],
)
def test_credential_and_app_state_paths_are_refused(path):
    """read_file is untagged and therefore silent.

    An unconfined read is a zero-prompt path from a credential file into a
    persisted transcript that may be sent to any provider. run_command
    reuses this same list (spec 8.4).
    """
    assert is_sensitive_path(Path(path).expanduser())


def test_ordinary_paths_are_allowed(tmp_path):
    assert not is_sensitive_path(tmp_path / "notes.md")


def test_matching_is_by_resolved_ancestry_not_substring(tmp_path):
    """`~/.sshfoo` is not `~/.ssh`."""
    assert not is_sensitive_path(Path("~/.sshfoo/file").expanduser())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Utils/test_sensitive_paths.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Utils.sensitive_paths'`

- [ ] **Step 3: Create the denylist**

Create `tldw_chatbook/Utils/sensitive_paths.py`:

```python
# tldw_chatbook/Utils/sensitive_paths.py
"""Paths no agent tool may read or write, regardless of configured root.

Shared by the `files` pack and (from Phase 4) ``run_command``, so the two
cannot drift. Two distinct reasons a path lands here:

1. **Credentials.** ``read_file`` carries no risk tag, so it resolves to
   the built-in ``allow`` floor and executes with no prompt. An unconfined
   read is therefore a zero-prompt path from a private key into a
   persisted transcript that may be sent to any provider.
2. **This application's own gate state.** A tool able to rewrite
   ``mcp_permissions.json`` or ``config.toml`` can turn every ``ask`` into
   ``allow`` -- a one-step bypass of the permission system.

This is a guardrail, not a security boundary: it stops accidents and naive
injected payloads, not a determined ``python -c``. The sandbox track is
the real answer for shell execution.
"""

from __future__ import annotations

from pathlib import Path

#: Directory prefixes that are refused along with everything beneath them.
_SENSITIVE_DIRS = (
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    "~/.config/gcloud",
    "~/.docker",
    "~/.kube",
    "~/.local/share/keyrings",
)

#: Individual files that are refused.
_SENSITIVE_FILES = (
    "~/.config/tldw_cli/config.toml",
    "~/.config/tldw_cli/mcp_permissions.json",
)


def _resolved(path_str: str) -> Path | None:
    try:
        return Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def is_sensitive_path(candidate: Path) -> bool:
    """Whether ``candidate`` is a credential or gate-state path.

    Comparison is by RESOLVED ancestry, never by string prefix, so
    ``~/.sshfoo`` is not mistaken for ``~/.ssh`` and a symlink cannot
    smuggle a path past the check.

    Args:
        candidate: The path a tool intends to touch.

    Returns:
        True when the path is refused. Fails CLOSED: a path that cannot be
        resolved is treated as sensitive.
    """
    resolved = _resolved(str(candidate))
    if resolved is None:
        return True

    for entry in _SENSITIVE_FILES:
        target = _resolved(entry)
        if target is not None and resolved == target:
            return True

    for entry in _SENSITIVE_DIRS:
        root = _resolved(entry)
        if root is not None and (resolved == root or root in resolved.parents):
            return True

    return False
```

- [ ] **Step 4: Apply it in the file tools' containment check**

In `file_operation_tools.py`, rename `_is_within` to a public `is_within` (the `files`
pack imports it in Task 11, and reaching across modules for a private name is a defect)
and make containment also require the path not be sensitive. Update the two existing
call sites in this module, and keep `_is_within = is_within` as a module-local alias only
if any other caller exists — check with
`grep -rn "_is_within" tldw_chatbook/ Tests/`.

```python
def is_within(candidate: Path, root: Path) -> bool:
    """Return whether ``candidate`` resolves inside ``root``.

    Also refuses credential and gate-state paths outright (see
    ``Utils.sensitive_paths``), so widening the configured root can never
    expose them.

    Args:
        candidate: Path to test.
        root: The sandbox root it must stay under.

    Returns:
        True only when the fully-resolved candidate is the root or below it
        AND is not a sensitive path.
    """
    from ..Utils.sensitive_paths import is_sensitive_path

    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    if is_sensitive_path(resolved):
        return False
    return resolved == root_resolved or root_resolved in resolved.parents
```

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Utils/test_sensitive_paths.py Tests/Agents/ -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Utils/sensitive_paths.py tldw_chatbook/Tools/file_operation_tools.py Tests/Utils/test_sensitive_paths.py
git commit -m "feat(tools): refuse credential and gate-state paths in file tools"
```

---

## Task 11: Add `glob_files` and `grep_files`

**Files:**
- Modify: `tldw_chatbook/Agents/builtin_packs/files.py`
- Test: `Tests/Agents/test_builtin_packs.py`

**Interfaces:**
- Consumes: `_tool_sandbox_root`, `_is_within` from `file_operation_tools`; `Tool` ABC
- Produces: `files.GlobFiles` (`glob_files`), `files.GrepFiles` (`grep_files`); both appended to `files.TOOLS`

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_builtin_packs.py`:

```python
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point the file-tool sandbox root at a temp dir."""
    import tldw_chatbook.Tools.file_operation_tools as fot

    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path))
    (tmp_path / "a.py").write_text("import os\nDEBUG = True\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.py").write_text("DEBUG = False\n")
    (tmp_path / "notes.md").write_text("nothing here\n")
    return tmp_path


@pytest.mark.asyncio
async def test_glob_files_matches_recursively_within_the_sandbox(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    result = await GlobFiles().execute(pattern="**/*.py")

    assert sorted(Path(p).name for p in result["matches"]) == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_grep_files_reports_matching_lines(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    result = await GrepFiles().execute(pattern="DEBUG", glob="**/*.py")

    assert len(result["matches"]) == 2
    assert all("DEBUG" in m["line"] for m in result["matches"])
    assert all(m["line_number"] >= 1 for m in result["matches"])


@pytest.mark.asyncio
async def test_grep_files_rejects_a_bad_regex_without_raising(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    result = await GrepFiles().execute(pattern="([", glob="**/*.py")

    assert "error" in result


@pytest.mark.asyncio
async def test_glob_files_refuses_parent_traversal(sandbox):
    """`Path.glob('../**/*')` does not raise -- it yields ~1.4M paths.

    Filtering by containment afterwards still walks all of them, so the
    pattern is refused up front.
    """
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    result = await GlobFiles().execute(pattern="../**/*")

    assert "error" in result
    assert "matches" not in result


@pytest.mark.asyncio
async def test_glob_files_refuses_absolute_patterns(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GlobFiles

    result = await GlobFiles().execute(pattern="/etc/*")

    assert "error" in result


@pytest.mark.asyncio
async def test_grep_files_refuses_parent_traversal(sandbox):
    from tldw_chatbook.Agents.builtin_packs.files import GrepFiles

    result = await GrepFiles().execute(pattern="DEBUG", glob="../**/*.py")

    assert "error" in result
```

Add `from pathlib import Path` to the test module's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_builtin_packs.py -k "glob or grep" -v`
Expected: FAIL with `ImportError: cannot import name 'GlobFiles'`

- [ ] **Step 3: Implement the two tools**

Append to `builtin_packs/files.py` (before the `TOOLS` assignment), and add `import re` plus `from pathlib import Path` and `from tldw_chatbook.Tools.file_operation_tools import is_within, _tool_sandbox_root` to its imports:

```python
#: Most matches either tool returns. Results also pass through the runtime's
#: own `max_tool_result_chars` cap, but bounding here keeps the JSON small
#: enough that the cap rarely has to cut mid-structure.
_MAX_MATCHES = 200

#: Most filesystem entries either tool will EXAMINE, independent of how many
#: match. Verified necessary: `Path.glob("../**/*")` does not raise -- it
#: happily yields ~1.4M paths from a temp dir. Since none of them pass the
#: containment check, a match-only bound never trips and the tool walks the
#: entire filesystem. This bound is what actually stops that.
_MAX_CANDIDATES = 20_000


def _rejects_traversal(pattern: str) -> bool:
    """Whether a glob pattern tries to leave the sandbox root.

    Checked before globbing rather than filtering afterwards: containment
    filtering alone still pays the cost of walking everything the pattern
    matched (see ``_MAX_CANDIDATES``).

    Args:
        pattern: A user- or model-supplied glob pattern.

    Returns:
        True when the pattern is absolute or contains a `..` component.
    """
    return pattern.startswith("/") or ".." in Path(pattern).parts


class GlobFiles(Tool):
    """`glob_files` -- path-pattern search inside the sandbox root."""

    def __init__(self, services: Any | None = None) -> None:
        self.services = services

    @property
    def name(self) -> str:
        return "glob_files"

    @property
    def description(self) -> str:
        return (
            "Find files by path pattern inside the tool sandbox. Supports "
            "glob syntax including ** for recursive matches, e.g. '**/*.py'. "
            f"Returns at most {_MAX_MATCHES} paths."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '**/*.py'.",
                }
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs) -> dict:
        pattern = str(kwargs.get("pattern") or "").strip()
        if not pattern:
            return {"error": "pattern is required"}
        if _rejects_traversal(pattern):
            return {"error": "pattern must stay inside the sandbox root"}
        root = _tool_sandbox_root()
        try:
            candidates = root.glob(pattern)
        except (ValueError, NotImplementedError) as exc:
            return {"error": f"invalid pattern: {exc}"}
        matches = []
        for examined, path in enumerate(candidates, start=1):
            if len(matches) >= _MAX_MATCHES or examined > _MAX_CANDIDATES:
                break
            if path.is_file() and is_within(path, root):
                matches.append(str(path))
        return {"matches": sorted(matches)}


class GrepFiles(Tool):
    """`grep_files` -- content search inside the sandbox root."""

    def __init__(self, services: Any | None = None) -> None:
        self.services = services

    @property
    def name(self) -> str:
        return "grep_files"

    @property
    def description(self) -> str:
        return (
            "Search file contents by regular expression inside the tool "
            "sandbox, optionally narrowed by a path glob. Returns matching "
            f"lines with their file and line number, at most {_MAX_MATCHES}."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression to search for.",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional path glob to narrow the search.",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs) -> dict:
        raw_pattern = str(kwargs.get("pattern") or "")
        if not raw_pattern:
            return {"error": "pattern is required"}
        try:
            regex = re.compile(raw_pattern)
        except re.error as exc:
            return {"error": f"invalid regular expression: {exc}"}

        root = _tool_sandbox_root()
        glob_pattern = str(kwargs.get("glob") or "**/*")
        if _rejects_traversal(glob_pattern):
            return {"error": "glob must stay inside the sandbox root"}
        try:
            candidates = root.glob(glob_pattern)
        except (ValueError, NotImplementedError) as exc:
            return {"error": f"invalid glob: {exc}"}

        matches: list[dict] = []
        # Deliberately NOT sorted(candidates): materialising and sorting the
        # generator defeats _MAX_CANDIDATES on a broad pattern.
        for examined, path in enumerate(candidates, start=1):
            if len(matches) >= _MAX_MATCHES or examined > _MAX_CANDIDATES:
                break
            if not path.is_file() or not is_within(path, root):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if len(matches) >= _MAX_MATCHES:
                    break
                if regex.search(line):
                    matches.append(
                        {
                            "path": str(path),
                            "line_number": number,
                            "line": line[:500],
                        }
                    )
        return {"matches": matches}
```

Add `from tldw_chatbook.Tools.tool_executor import Tool` to the module's imports.

- [ ] **Step 4: Register them in the pack**

Replace the `TOOLS` assignment in `builtin_packs/files.py`:

```python
TOOLS: tuple[type, ...] = (ReadFile, ListDirectory, GlobFiles, GrepFiles)
```

- [ ] **Step 5: Update the pack-contents test**

In `Tests/Agents/test_builtin_packs.py`, update `test_files_pack_declares_its_tools_and_no_optional_deps` and `test_pack_tool_classes_returns_only_enabled_packs`:

```python
    assert {c.__name__ for c in files.TOOLS} == {
        "ReadFile",
        "ListDirectory",
        "GlobFiles",
        "GrepFiles",
    }
```

```python
    assert len(pack_tool_classes(frozenset({"files"}))) == 4
```

- [ ] **Step 6: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/Agents/ -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/builtin_packs/files.py Tests/Agents/test_builtin_packs.py
git commit -m "feat(agents): add glob_files and grep_files to the files pack"
```

---

## Task 12: Relocate the `Tool` ABC out of `tool_executor`

**Files:**
- Create: `tldw_chatbook/Tools/base.py`
- Modify: `tldw_chatbook/Tools/tool_executor.py`, `tldw_chatbook/Tools/__init__.py`, `tldw_chatbook/Agents/builtin_tool_gate.py`, `tldw_chatbook/Agents/tool_catalog.py`, `tldw_chatbook/Agents/builtin_packs/files.py`, `tldw_chatbook/Tools/file_operation_tools.py`, `tldw_chatbook/Tools/note_management_tools.py`, `tldw_chatbook/Tools/rag_search_tool.py`, `tldw_chatbook/Tools/web_search_tool.py`
- Test: `Tests/Agents/test_tool_protocol.py`

**Interfaces:**
- Consumes: nothing
- Produces: `tldw_chatbook.Tools.base.Tool` — the ABC, unchanged in behaviour. `tool_executor.Tool` remains importable as a re-export until Task 13.

- [ ] **Step 1: Write the failing test**

Append to `Tests/Agents/test_tool_protocol.py`:

```python
def test_tool_abc_lives_in_a_module_free_of_the_executor():
    """The ABC must outlive ToolExecutor's removal.

    builtin_tool_gate imports Tool for risk_tags, so deleting the executor
    module wholesale would break the permission gate.
    """
    import inspect

    import tldw_chatbook.Tools.base as base

    assert inspect.isclass(base.Tool)
    assert not hasattr(base, "ToolExecutor")
    assert not hasattr(base, "get_tool_executor")


def test_tool_executor_reexports_the_same_abc():
    from tldw_chatbook.Tools.base import Tool as BaseTool
    from tldw_chatbook.Tools.tool_executor import Tool as LegacyTool

    assert BaseTool is LegacyTool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_tool_protocol.py -k abc -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.Tools.base'`

- [ ] **Step 3: Create the base module**

Create `tldw_chatbook/Tools/base.py` containing the `Tool` ABC moved verbatim from `tool_executor.py` (lines 21-66 plus the `timeout_seconds` property added in Task 3), with this module docstring:

```python
# tldw_chatbook/Tools/base.py
"""The `Tool` ABC.

Lives apart from ``tool_executor`` because the executor is being removed
(it has no callers left) while the ABC is load-bearing: ``risk_tags`` is
what the permission gate resolves against, and every built-in pack tool
subclasses this.
"""
```

Keep the required imports (`abc`, `typing`).

- [ ] **Step 4: Re-export from the old location and repoint importers**

In `tool_executor.py`, delete the `Tool` class body and replace it with:

```python
from .base import Tool  # re-exported for callers not yet repointed
```

Then update every module that imports `Tool` from `tool_executor` to import from `tldw_chatbook.Tools.base` instead: `Agents/builtin_tool_gate.py`, `Agents/tool_catalog.py`, `Agents/builtin_packs/files.py`, and the four `Tools/*_tool*.py` implementation modules (which use `from . import Tool` — repoint to `from .base import Tool`).

- [ ] **Step 5: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/ -x -q`
Expected: PASS across the whole suite — this task touches imports app-wide, so run everything, not just `Tests/Agents/`.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Tools/ tldw_chatbook/Agents/ Tests/Agents/test_tool_protocol.py
git commit -m "refactor(tools): relocate the Tool ABC to Tools/base.py"
```

---

## Task 13: Delete the ToolExecutor and its settings surface

**Files:**
- Delete: `tldw_chatbook/Tools/code_audit_tool.py`
- Modify: `tldw_chatbook/Tools/tool_executor.py` (remove `ToolExecutor`, `ToolResultCache`, `get_tool_executor`, `reload_tool_executor`, the registration block), `tldw_chatbook/Tools/__init__.py`, `tldw_chatbook/UI/Tools_Settings_Window.py`
- Test: `Tests/Agents/test_tool_protocol.py`; delete any test file covering only the removed executor

**Interfaces:**
- Consumes: `Tools.base.Tool` from Task 12
- Produces: nothing importable named `ToolExecutor` or `get_tool_executor`

- [ ] **Step 1: Confirm there are no callers**

Run:

```bash
grep -rn "get_tool_executor\|execute_tool_call\|execute_tool_calls\|ToolExecutor" tldw_chatbook/ --include="*.py" | grep -v "Tools/tool_executor.py"
```

Expected: only `UI/Tools_Settings_Window.py` hits. If anything else appears, STOP and report it — the spec's §4.7 premise no longer holds and this task must be re-planned.

- [ ] **Step 2: Write the failing test**

Append to `Tests/Agents/test_tool_protocol.py`:

```python
def test_tool_executor_is_gone():
    """TASK-545 P3: nothing may dispatch tools outside the gated runtime.

    The executor had no callers left, and its own [tools] config read was
    the TASK-547 defect. Removing it closes both.
    """
    import tldw_chatbook.Tools as tools_pkg
    import tldw_chatbook.Tools.tool_executor as executor_mod

    assert not hasattr(executor_mod, "ToolExecutor")
    assert not hasattr(tools_pkg, "get_tool_executor")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest Tests/Agents/test_tool_protocol.py -k executor_is_gone -v`
Expected: FAIL — the attributes still exist

- [ ] **Step 4: Remove the executor**

In `tool_executor.py`, delete `ToolResultCache`, `ToolExecutor`, `get_tool_executor`, `reload_tool_executor`, the module-level `_global_executor`, and the whole tool-registration block. What remains is the `Tool` re-export from Task 12 plus `DateTimeTool` and `CalculatorTool` (still imported by `tool_catalog`). Update the module docstring to say the dispatcher is gone and the module now only holds the two dependency-free built-ins.

Update `Tools/__init__.py` to stop re-exporting `get_tool_executor` / `reload_tool_executor` and anything else now absent.

Delete `tldw_chatbook/Tools/code_audit_tool.py` — it audits the executor's own file operations, and `MCP/execution_log.py` already records the gated runtime's decisions.

- [ ] **Step 5: Remove the settings switches**

In `UI/Tools_Settings_Window.py`, remove the three `get_tool_executor()` blocks and the per-tool enable/disable switches they render, along with any now-unused `tools_config` reads. Leave the rest of the window intact. Tool enablement now lives in `[agent_tools] enabled_packs` (Task 9) and per-tool permissions in TASK-656's matrix.

- [ ] **Step 6: Run the tests**

Run: `source .venv/bin/activate && pytest Tests/ -x -q`
Expected: PASS. Delete any test file that exists solely to exercise the removed executor, and note the deletions in the commit body.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(tools): remove the ToolExecutor, code audit tool, and settings switches

The executor had no callers left after the TASK-577 deletion campaign, and
its 2-arg get_cli_setting call was the TASK-547 defect. Removing it closes
TASK-547 and satisfies TASK-545 P3's 'no tool executes ungated' criterion.
Tool enablement now lives in [agent_tools] enabled_packs; per-tool
permissions in the agent:builtin matrix."
```

---

## Task 14: Update the backlog

**Files:**
- Modify: `backlog/tasks/task-545 - Wire-built-in-tool-executor-into-MCP-permission-gate.md`, `backlog/tasks/task-547 - Fix-tools-config-unreachable-via-get_cli_setting.md`

- [ ] **Step 1: Correct TASK-545's stale premise**

Its description names `worker_events.py` and `chat_streaming_events.py` as System A's callers. Both are gone. Add an Implementation Notes paragraph recording that, that P3's "no tool executes ungated" criterion is satisfied by the executor's removal, and that `read_file`/`list_directory` were already reachable via TASK-584 before this work. Check P3's boxes.

- [ ] **Step 2: Close TASK-547**

Add Implementation Notes: the defect was confined to `get_tool_executor()`'s 2-arg `get_cli_setting("tools", {})`; the `[tools]` section itself was always reachable via the 3-arg form, which TASK-584's gates and `_resolve_sandbox_config` both used. Removing the executor removed the only broken caller. The live keys migrated to `[agent_tools]` with a deprecated fallback.

- [ ] **Step 3: Verify no ID collisions**

Run:

```bash
python3 -c "
import os, re, collections
ids = collections.Counter()
for f in os.listdir('backlog/tasks'):
    m = re.match(r'task-(\d+)', f)
    if m: ids[int(m.group(1))] += 1
print([i for i, n in ids.items() if n > 1] or 'no duplicates')
"
```

Expected: `no duplicates`

- [ ] **Step 4: Commit**

```bash
git add backlog/
git commit -m "docs(backlog): close task-547, correct task-545's stale System A premise"
```

---

## Deferred, with reasons

Recorded so a later reader does not think they were forgotten:

- **Workspace-rooted filesystem access** (spec §4.6 rule 1) — changes TASK-584's shipped security posture and needs explicit sign-off. The sandbox root stays as shipped.
- **`find()` upgrade** — at threshold 16 a one-to-three-pack user never reaches `find_tools`. Returns when a phase pushes a realistic enabled set past 16.
- **DB connection churn** (spec §9) — each tool call opens and orphans a thread-local connection. Needs its own task before Phase 2.
- **`write_file` / `edit_file`** — Phase 2, gated on TASK-656 landing so persistent permissions are reversible.
