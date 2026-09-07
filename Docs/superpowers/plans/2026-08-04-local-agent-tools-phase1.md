# Local Agent Tools — Phase 1 (Plumbing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the plumbing that lets workspace-local tools (`fs_*`, later `web_*`/`todo_*`) register on the Console agent's tool catalog with the same approval/permission machinery MCP tools already use — proven end-to-end with one pilot read-only tool, `fs_list`.

**Architecture:** Sync core implementations in `Tools/local_tool_impls.py`; a `LocalToolProvider` in `Agents/` implementing the frozen `ToolProvider` protocol with per-turn approval stamps mirroring `MCPToolProvider`; a `build_local_review_hook` in `Chat/console_chat_controller.py` reusing the existing `request_mcp_approvals` card round-trip; registration in `Chat/console_agent_bridge.py`'s per-run registry composition. Permissions resolve through `MCP/permission_store.py` under the synthetic server key `local:__local__`.

**Tech Stack:** Python ≥3.11, pytest, Textual (existing app), existing MCP permission store.

**Spec:** `Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md`
**ADR required:** yes — Task 0 creates `backlog/decisions/032-local-agent-tool-permission-boundary.md`.

**Spec deviation (approved-by-plan):** the spec assigns all real tools to phase 2; this plan includes `fs_list` as the phase-1 pilot because a provider with zero tools cannot be integration-tested. Phase 2's backlog task text should drop `fs_list` from its scope when created.

---

## Key facts from codebase verification (do not re-derive)

- `ToolProvider` protocol (`tldw_chatbook/Agents/tool_catalog.py:70-77`): sync `list_catalog() -> list[ToolCatalogEntry]`, `load_schema(tool_id) -> ToolSchema`, `invoke(tool_id, args) -> ToolResult`. Context-free.
- Models (`tldw_chatbook/Agents/agent_models.py`): `ToolCatalogEntry{id,name,one_line_description,source}`, `ToolSchema{id,name,description,parameters}`, `ToolResult{ok,content,error}`, `ToolCall{name,args,call_id}`.
- Registry is first-registrant-wins on name (`tool_catalog.py:311-323`). Builtin ids are `builtin:<name>`; ours are `local:<name>`.
- `build_mcp_review_hook` (`Chat/console_chat_controller.py:70-169`) is the exact pattern to mirror: `provider.apply_batch_decisions({})` at entry (clear-first), collect `pending_gate_for(name,args)`, one `request_mcp_approvals(pending)` round trip, `apply_batch_decisions(decisions)`, return `{llm_name: "proceed"}` only.
- `MCPPendingCall` (`Agents/mcp_tool_provider.py:78-87`): frozen dataclass `{llm_name, server_key, tool_name, server_label, arguments, reason}` — reason is `ask|config_changed|risk_floored`. The approval card consumes exactly this shape; reuse it unchanged.
- Refusal constants to mirror (`mcp_tool_provider.py:60-62`), with the pinned local wording from spec §3.3.
- `run_agent_loop` fails OPEN on hook exception (`agent_runtime.py:366-376`) — fail-closed lives in the provider: clear-first stamps + no-stamp refusal in `invoke()`.
- `MCPToolProvider.stamp_scope` (`mcp_tool_provider.py:347-375`) snapshots/restores per-turn stamps around nested sub-agent runs; the bridge threads it as `AgentService(review_state_scope=...)` (`console_agent_bridge.py:927-941`). `LocalToolProvider` needs an identical `stamp_scope()`, and the bridge must compose BOTH scopes.
- `HubTool` (`MCP/hub_tool_catalog.py:20-47`): dataclass `{server_key, server_label, source, name, description, input_schema, tags, stale, executable}`. Risk floor keys off `tags` containing `"mutates"`.
- `permission_store.definition_hash(description, input_schema)` (`permission_store.py:454`) — sha256 over canonical JSON; required when persisting `allow`.
- `resolve_effective_state(payload, hub_tool)` (`permission_store.py:516`) — precedence tool → server → global; rug-pull downgrade; risk floor on inherited allows.
- `validate_path(user_path, base_directory)` (`Utils/path_validation.py:16`) confines to base but hard-bans hidden components (line 60-66) with no opt-out — Task 1 adds `allow_hidden`.
- `_compose_run_registry_and_allowed(context, *, mcp_provider=None)` (`console_agent_bridge.py:613-653+`): builds fresh per-run registry, registers Builtin → Skill → MCP; returns `(registry, allowed_tools, builtin_names)`. We add `local_provider=None`.
- `run_reply(..., mcp_provider=None, review_tool_calls=None)` (`console_agent_bridge.py:772-786`): takes an `Any`-typed composed provider; threads `getattr(mcp_provider, "stamp_scope", None)` into `AgentService`.
- Config: `[console]` loaded/coerced at `config.py:770-791` (pattern: `coerce_bool_setting(final_console_settings_cli.get("collapse_large_pastes", True), True)`); template near `config.py:2134`.
- `request_mcp_approvals` (`console_chat_controller.py:626+`) runs on the agent worker thread, surfaces the card via `app.call_from_thread`, blocks on `threading.Event` with 1 s poll, 120 s default timeout. Payload-agnostic — it consumes `MCPPendingCall`s regardless of which provider built them.
- Test layout: `Tests/Agents/`, `Tests/Tools/` exist; project uses real tmp dirs and Hypothesis.

---

## Task 0: ADR + backlog task

**Files:**
- Create: `backlog/decisions/032-local-agent-tool-permission-boundary.md`
- Create: backlog task (via CLI if available, else markdown file)

- [ ] **Step 1: Write ADR-032**

Content (fill the standard sections — Context / Decision / Consequences / Alternatives considered):

- **Context:** The Console agent runtime (`Agents/tool_catalog.py` seam) currently offers only calculator/datetime plus MCP and skill tools. The spec `2026-08-04-local-agent-tools-design.md` adds workspace-local file/web/todo tools, which gives model-initiated calls local filesystem read/write and network access for the first time outside the MCP boundary.
- **Decision:** (1) Local tools register as a first-class `ToolProvider` (`local:<name>` catalog ids, `fs_`/`web_`/`todo_` naming per the ADR-030 `library_*` precedent). (2) They reuse the MCP permission store under the synthetic server key `local:__local__` — no schema change — with `mutates` risk tags on write tools. (3) Fail-closed approval uses the three-mechanism discipline (clear-first hook, no-stamp refusal in `invoke()`, `stamp_scope` around sub-agent runs). (4) All path tools confine to a configurable `[console] workspace_root` (default app cwd); hidden path components are allowed under the root via a new `allow_hidden` parameter on `validate_path`. (5) Pinned refusal strings: `LOCAL_DENY_REFUSAL`, `LOCAL_TIMEOUT_REFUSAL`, `LOCAL_KILL_SWITCH_REFUSAL` (spec §3.3).
- **Alternatives considered:** (a) self-hosted MCP server consumed via the in-process delegate — rejected: JSON-RPC plumbing for local file reads, runtime depends on MCP lifecycle for basic capabilities; (b) separate local permission store — rejected: duplicates audit trail and approval UX; (c) config-flag-only gating — rejected: no interactive approval, weakest safety story.
- **Consequences:** adding a local tool never touches the runtime loop; "Always allow" persists with a `definition_hash` like MCP tools; the MCP workbench's permission UI can display local tools later without migration.

- [ ] **Step 2: Create the backlog task**

Run:

```bash
backlog task create "Local agent tools phase 1: plumbing + fs_list pilot" \
  -d "Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md. Plan: Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md. ADR: backlog/decisions/032. Build LocalToolProvider + approval-hook generalization + workspace-root config, proven end-to-end with fs_list." \
  --ac "LocalToolProvider lists/schemas/invokes fs_list through the agent runtime loop,Approval card gates fs_list with allow/session/always/deny wired to the permission store under local:__local__,Kill switch and fail-closed no-callback paths return the pinned refusal strings,workspace_root and local_tools_enabled config keys coerce and default correctly,All new tests pass" \
  --plan "See Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md" || echo "backlog CLI unavailable — create backlog/tasks/ markdown manually"
```

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/032-local-agent-tool-permission-boundary.md backlog/
git commit -m "docs: ADR-032 local agent tool permission boundary + phase-1 task"
```

---

## Task 1: `validate_path` gains `allow_hidden`

**Files:**
- Modify: `tldw_chatbook/Utils/path_validation.py:16-95`
- Test: `Tests/Utils/test_path_validation.py` (append; create if absent)

- [ ] **Step 1: Write the failing test**

```python
def test_validate_path_allow_hidden_permits_dotdirs_under_base(tmp_path):
    from tldw_chatbook.Utils.path_validation import validate_path
    hidden = tmp_path / ".github" / "workflows"
    hidden.mkdir(parents=True)
    result = validate_path(".github/workflows", tmp_path, allow_hidden=True)
    assert result == hidden.resolve()


def test_validate_path_default_still_rejects_hidden(tmp_path):
    import pytest
    from tldw_chatbook.Utils.path_validation import validate_path
    (tmp_path / ".hidden").mkdir()
    with pytest.raises(ValueError, match="hidden"):
        validate_path(".hidden", tmp_path)


def test_validate_path_allow_hidden_still_blocks_traversal(tmp_path):
    import pytest
    from tldw_chatbook.Utils.path_validation import validate_path
    with pytest.raises(ValueError, match="outside"):
        validate_path("../escape", tmp_path, allow_hidden=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Tests/Utils/test_path_validation.py -k allow_hidden -x -q`
Expected: FAIL (`TypeError: validate_path() got an unexpected keyword argument 'allow_hidden'`)

- [ ] **Step 3: Implement**

In `validate_path`, change the signature and the hidden-component check:

```python
def validate_path(
    user_path: Union[str, Path],
    base_directory: Union[str, Path],
    allow_hidden: bool = False,
) -> Path:
```

and gate the hidden check:

```python
        if not allow_hidden and any(
            part.startswith(".") for part in full_path.parts if part != "."
        ):
```

Keep the docstring updated (new `allow_hidden` arg, default preserves prior behavior).

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/Utils/ -q`
Expected: PASS, including all pre-existing path-validation tests (default behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Utils/path_validation.py Tests/Utils/test_path_validation.py
git commit -m "feat: validate_path allow_hidden opt-out for workspace-confined tools"
```

---

## Task 2: `Tools/local_tool_impls.py` — workspace resolution + `fs_list` core

**Files:**
- Create: `tldw_chatbook/Tools/local_tool_impls.py`
- Test: `Tests/Tools/test_local_tool_impls.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest

from tldw_chatbook.Tools.local_tool_impls import (
    LocalToolError,
    list_directory,
    resolve_workspace_path,
)


def test_resolve_workspace_path_confines(tmp_path):
    assert resolve_workspace_path("a/b", tmp_path) == (tmp_path / "a/b").resolve()
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        resolve_workspace_path("../x", tmp_path)


def test_list_directory_shows_dirs_first_then_files(tmp_path):
    (tmp_path / "zeta.txt").write_text("z")
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "inner.txt").write_text("i")
    out = list_directory(".", workspace_root=tmp_path)
    lines = out.splitlines()
    assert lines[0] == "alpha/"
    assert lines[1] == "zeta.txt"


def test_list_directory_caps_entries(tmp_path):
    for i in range(10):
        (tmp_path / f"f{i}.txt").write_text("x")
    out = list_directory(".", workspace_root=tmp_path, max_entries=3)
    assert out.count("\n") + 1 == 4  # 3 entries + truncation notice
    assert "7 more entries" in out


def test_list_directory_rejects_file_and_missing(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(LocalToolError, match="not a directory"):
        list_directory("f.txt", workspace_root=tmp_path)
    with pytest.raises(LocalToolError, match="not a directory"):
        list_directory("nope", workspace_root=tmp_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Tests/Tools/test_local_tool_impls.py -x -q`
Expected: FAIL (`ModuleNotFoundError: tldw_chatbook.Tools.local_tool_impls`)

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Tools/local_tool_impls.py
"""Sync core implementations for workspace-local agent tools.

Plain functions, no async, no Textual, no event loop — callable from the
agent runtime's worker thread via Agents/local_tool_provider.py. Every
failure raises LocalToolError; the provider converts those (and any other
exception) into ToolResult error strings — nothing raises across the
provider boundary.
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.Utils.path_validation import validate_path

MAX_LIST_ENTRIES = 200


class LocalToolError(ValueError):
    """Model-actionable failure from a local tool (path, not-found, …)."""


def resolve_workspace_path(path: str, workspace_root: Path) -> Path:
    """Resolve ``path`` against ``workspace_root``, confined to it.

    Hidden components (``.github/``) are allowed under the root; anything
    resolving outside it is refused. Raises LocalToolError.
    """
    try:
        return validate_path(path, workspace_root, allow_hidden=True)
    except ValueError as exc:
        raise LocalToolError(
            f"Path '{path}' is outside the workspace root ({workspace_root})"
        ) from exc


def list_directory(
    path: str, *, workspace_root: Path, max_entries: int = MAX_LIST_ENTRIES
) -> str:
    """One-level listing of ``path``: ``name/`` for dirs, ``name`` for files.

    Directories sort before files, each group case-insensitively by name.
    Output is capped at ``max_entries`` with a trailing truncation notice.
    Raises LocalToolError when ``path`` is not an existing directory.
    """
    root = resolve_workspace_path(path, workspace_root)
    if not root.is_dir():
        raise LocalToolError(f"not a directory: {path}")
    entries = sorted(
        root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
    )
    lines = [
        f"{p.name}/" if p.is_dir() else p.name for p in entries[:max_entries]
    ]
    remaining = len(entries) - max_entries
    if remaining > 0:
        lines.append(f"… ({remaining} more entries, truncated)")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/Tools/test_local_tool_impls.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Tools/local_tool_impls.py Tests/Tools/test_local_tool_impls.py
git commit -m "feat: local tool impls scaffold with fs_list core"
```

---

## Task 3: `Agents/local_tool_provider.py`

**Files:**
- Create: `tldw_chatbook/Agents/local_tool_provider.py`
- Test: `Tests/Agents/test_local_tool_provider.py`

Design notes for the implementer:

- Mirror `MCPToolProvider`'s stamp mechanics (`mcp_tool_provider.py`): per-turn `_stamps: dict[str, str]`; `apply_batch_decisions(decisions)` REPLACES the dict; `pending_gate_for(name, args) -> MCPPendingCall | None`; `stamp_scope()` is a contextmanager snapshotting/restoring `_stamps` (copy the shape of `mcp_tool_provider.py:347-375`).
- State resolution and persistence are injected as callables by the controller (which owns the permission-store access) — the provider stays store-agnostic and unit-testable with fakes.
- Decision-string handling in `invoke()` must mirror `MCPToolProvider._apply_verdict` (`mcp_tool_provider.py:551+`) return shapes exactly; substitute the pinned LOCAL_* constants.
- Reuse `MCPPendingCall` from `.mcp_tool_provider` for pending payloads (the approval card consumes exactly that shape).

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from tldw_chatbook.Agents.local_tool_provider import (
    LOCAL_DENY_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
    LOCAL_TIMEOUT_REFUSAL,
    LocalToolProvider,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState

ALLOW = EffectiveToolState(state="allow", origin="tool_override")
ASK = EffectiveToolState(state="ask", origin="global_default")
DENY = EffectiveToolState(state="deny", origin="tool_override")


def make_provider(state=ALLOW, kill=False, **kwargs):
    return LocalToolProvider(
        workspace_root=Path(kwargs.pop("root", ".")).resolve() if "root" in kwargs else Path("."),
        resolve_state=lambda hub: state,
        kill_switch=lambda: kill,
        **kwargs,
    )


def test_catalog_lists_fs_list_with_local_ids(tmp_path):
    p = make_provider(root=tmp_path)
    entries = p.list_catalog()
    assert [e.id for e in entries] == ["local:fs_list"]
    assert entries[0].name == "fs_list" and entries[0].source == "local"
    schema = p.load_schema("local:fs_list")
    assert schema.parameters["required"] == ["path"]


def test_invoke_happy_path(tmp_path):
    (tmp_path / "hello.txt").write_text("hi")
    p = make_provider(root=tmp_path)
    r = p.invoke("local:fs_list", {"path": "."})
    assert r.ok and "hello.txt" in r.content


def test_invoke_unknown_tool(tmp_path):
    r = make_provider(root=tmp_path).invoke("local:nope", {})
    assert not r.ok and "Unknown local tool" in r.error


def test_kill_switch_refuses(tmp_path):
    r = make_provider(root=tmp_path, kill=True).invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_KILL_SWITCH_REFUSAL


def test_deny_state_refuses(tmp_path):
    r = make_provider(state=DENY, root=tmp_path).invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_DENY_REFUSAL


def test_ask_without_stamp_or_callback_fails_closed(tmp_path):
    r = make_provider(state=ASK, root=tmp_path).invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL


def test_ask_with_approve_once_stamp_executes(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    p = make_provider(state=ASK, root=tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    assert p.invoke("local:fs_list", {"path": "."}).ok


def test_stamps_replace_not_merge(tmp_path):
    p = make_provider(state=ASK, root=tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    p.apply_batch_decisions({})  # next turn cleared first
    r = p.invoke("local:fs_list", {"path": "."})
    assert not r.ok and r.error == LOCAL_TIMEOUT_REFUSAL


def test_pending_gate_for_ask_returns_pending_call(tmp_path):
    p = make_provider(state=ASK, root=tmp_path)
    gate = p.pending_gate_for("fs_list", {"path": "."})
    assert gate is not None
    assert gate.server_key == "local:__local__" and gate.tool_name == "fs_list"
    assert gate.reason == "ask"
    assert p.pending_gate_for("unknown", {}) is None


def test_stamp_scope_isolates_nested_run(tmp_path):
    p = make_provider(state=ASK, root=tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    with p.stamp_scope():
        assert not p.invoke("local:fs_list", {"path": "."}).ok  # child: no stamps
    assert p.invoke("local:fs_list", {"path": "."}).ok  # parent stamps restored


def test_execution_error_becomes_result_string(tmp_path):
    r = make_provider(root=tmp_path).invoke("local:fs_list", {"path": "../escape"})
    assert not r.ok and "outside the workspace root" in r.error
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Tests/Agents/test_local_tool_provider.py -x -q`
Expected: FAIL (`ModuleNotFoundError`)

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Agents/local_tool_provider.py
"""ToolProvider for workspace-local fs_/web_/todo_ tools.

Spec: Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md.
ADR: backlog/decisions/032. Mirrors MCPToolProvider's approval discipline:
clear-first per-turn stamps, fail-closed invoke with pinned refusal
strings, stamp_scope() isolation around nested sub-agent runs. All Protocol
methods are sync and worker-thread safe; no Textual/event-loop imports.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState

from .agent_models import ToolCatalogEntry, ToolResult, ToolSchema
from .mcp_tool_provider import MCPPendingCall

SOURCE = "local"
LOCAL_SERVER_KEY = "local:__local__"
LOCAL_SERVER_LABEL = "Local workspace"

# Pinned refusal strings (spec §3.3) — tests assert on these verbatim.
LOCAL_DENY_REFUSAL = "blocked by local tool permissions (set to Off)"
LOCAL_TIMEOUT_REFUSAL = "user did not approve within the time limit; do not retry"
LOCAL_KILL_SWITCH_REFUSAL = "blocked — local tools are switched off"

_MAX_RESULT_BYTES = 32 * 1024
_MAX_ERROR_CHARS = 300


@dataclass(frozen=True)
class LocalToolSpec:
    """One local tool: schema plus its sync handler (args dict -> text)."""

    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], str]
    tags: tuple[str, ...] = ()


def _fit_result(text: str) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= _MAX_RESULT_BYTES:
        return text
    return raw[:_MAX_RESULT_BYTES].decode("utf-8", errors="ignore") + "\n… [truncated]"


class LocalToolProvider:
    """Exposes LocalToolSpecs behind the ToolProvider protocol, gated per call.

    Args:
        workspace_root: Confinement root for all path-taking tools.
        specs: Tool specs; defaults to the built-in set (fs_list pilot).
        resolve_state: (HubTool) -> EffectiveToolState, injected by the
            controller (owns permission-store access).
        kill_switch: () -> bool master off-switch.
        approval_callback: invoke()'s single-call fallback gate for an
            "ask"-state tool with no batch stamp; None fails closed.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        specs: list[LocalToolSpec] | None = None,
        resolve_state: Callable[[HubTool], EffectiveToolState] | None = None,
        kill_switch: Callable[[], bool] = lambda: False,
        approval_callback: Callable[[list[MCPPendingCall]], dict[str, str]] | None = None,
    ) -> None:
        self._root = workspace_root
        self._specs = {s.name: s for s in (specs if specs is not None else _default_specs(workspace_root))}
        self._resolve_state = resolve_state or (lambda hub: EffectiveToolState(state="ask", origin="global_default"))
        self._kill_switch = kill_switch
        self._approval_callback = approval_callback
        self._stamps: dict[str, str] = {}

    # -- catalog ------------------------------------------------------

    def _tool_id(self, name: str) -> str:
        return f"{SOURCE}:{name}"

    def list_catalog(self) -> list[ToolCatalogEntry]:
        return [
            ToolCatalogEntry(
                id=self._tool_id(s.name),
                name=s.name,
                one_line_description=s.description.splitlines()[0],
                source=SOURCE,
            )
            for s in self._specs.values()
        ]

    def load_schema(self, tool_id: str) -> ToolSchema:
        spec = self._specs[tool_id.split(":", 1)[1]]
        return ToolSchema(
            id=tool_id, name=spec.name,
            description=spec.description, parameters=spec.parameters,
        )

    def hub_tool_for(self, name: str) -> HubTool:
        """The HubTool view used for permission resolution (carries risk tags)."""
        spec = self._specs[name]
        return HubTool(
            server_key=LOCAL_SERVER_KEY,
            server_label=LOCAL_SERVER_LABEL,
            source="local",
            name=spec.name,
            description=spec.description,
            input_schema=spec.parameters,
            tags=spec.tags,
            stale=False,
            executable=True,
        )

    # -- approval stamps (mirror MCPToolProvider) ----------------------

    def apply_batch_decisions(self, decisions: dict[str, str]) -> None:
        """REPLACE this turn's stamps (never merge) — clear-first discipline."""
        self._stamps = dict(decisions)

    @contextmanager
    def stamp_scope(self) -> Iterator[None]:
        """Snapshot/restore stamps around a nested sub-agent run."""
        saved = self._stamps
        self._stamps = {}
        try:
            yield
        finally:
            self._stamps = saved

    def pending_gate_for(self, name: str, args: dict) -> MCPPendingCall | None:
        """The approval payload when this call needs human gating, else None."""
        spec = self._specs.get(name)
        if spec is None:
            return None
        state = self._resolve_state(self.hub_tool_for(name))
        if state.state != "ask":
            return None
        reason = (
            "config_changed" if state.config_changed
            else "risk_floored" if state.risk_floored
            else "ask"
        )
        return MCPPendingCall(
            llm_name=name,
            server_key=LOCAL_SERVER_KEY,
            tool_name=name,
            server_label=LOCAL_SERVER_LABEL,
            arguments=args,
            reason=reason,
        )

    # -- invocation -----------------------------------------------------

    def invoke(self, tool_id: str, args: dict) -> ToolResult:
        name = tool_id.split(":", 1)[1] if ":" in tool_id else tool_id
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ok=False, error=f"Unknown local tool: {name}")
        if self._kill_switch():
            return ToolResult(ok=False, error=LOCAL_KILL_SWITCH_REFUSAL)
        verdict = self._verdict_for(name)
        if verdict in ("deny",):
            return ToolResult(ok=False, error=LOCAL_DENY_REFUSAL)
        if verdict in ("timeout", "no_callback"):
            return ToolResult(ok=False, error=LOCAL_TIMEOUT_REFUSAL)
        try:
            return ToolResult(ok=True, content=_fit_result(spec.handler(args)))
        except Exception as exc:  # noqa: BLE001 — never raises across the boundary
            return ToolResult(ok=False, error=str(exc)[:_MAX_ERROR_CHARS])

    def _verdict_for(self, name: str) -> str:
        """Resolve this call's gate decision: allow executes; anything else refuses."""
        state = self._resolve_state(self.hub_tool_for(name))
        if state.state == "allow":
            return "allow"
        if state.state == "deny":
            return "deny"
        # ask: per-turn stamp wins; then single-call fallback; then fail closed.
        stamp = self._stamps.get(name)
        if stamp in ("approve_once", "approve_session", "always_allow"):
            return "allow"
        if stamp == "deny":
            return "deny"
        if stamp == "timeout":
            return "timeout"
        if self._approval_callback is not None:
            decision = self._approval_callback([self.pending_gate_for(name, {})]).get(name, "timeout")
            return "allow" if decision in ("approve_once", "approve_session", "always_allow") else decision
        return "no_callback"


def _default_specs(workspace_root: Path) -> list[LocalToolSpec]:
    from tldw_chatbook.Tools.local_tool_impls import list_directory

    return [
        LocalToolSpec(
            name="fs_list",
            description="List a directory's entries (dirs first, then files), relative to the workspace root.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, relative to the workspace root (use \".\" for the root)."},
                },
                "required": ["path"],
            },
            handler=lambda args: list_directory(args["path"], workspace_root=workspace_root),
            tags=(),
        ),
    ]
```

Note: `EffectiveToolState` field names (`state`, `origin`, `config_changed`, `risk_floored`) — verify against `MCP/permission_store.py`'s dataclass and adjust the test fixtures to match its real constructor.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/Agents/test_local_tool_provider.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/local_tool_provider.py Tests/Agents/test_local_tool_provider.py
git commit -m "feat: LocalToolProvider with MCP-parity approval discipline"
```

---

## Task 4: `[console]` config — `local_tools_enabled` + `workspace_root`

**Files:**
- Modify: `tldw_chatbook/config.py:770-791` (console coercion block) and the template near `config.py:2134`
- Test: `Tests/Config/test_console_config.py` (append or create in the existing config test layout — find the right file with `ls Tests/Config*` / `grep -rn collapse_large_pastes Tests/`)

- [ ] **Step 1: Write the failing test**

Follow the existing `collapse_large_pastes` test's fixture pattern (however that test loads config) and assert:

```python
def test_console_local_tools_defaults(config_dict):
    console = config_dict["console"]
    assert console["local_tools_enabled"] is False
    assert console["workspace_root"] == ""


def test_console_local_tools_coerced(toml_with):
    # with [console] local_tools_enabled = "yes", workspace_root = 123
    console = load(toml_with(local_tools_enabled="yes", workspace_root=123))
    assert console["local_tools_enabled"] is True
    assert console["workspace_root"] == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest <the config test file> -k local_tools -x -q`
Expected: FAIL (KeyError / assertion)

- [ ] **Step 3: Implement**

In the console coercion block (`config.py:770-791`), after `collapse_large_pastes`:

```python
    final_console_settings_cli["local_tools_enabled"] = coerce_bool_setting(
        final_console_settings_cli.get("local_tools_enabled", False),
        False,
    )
    workspace_root = final_console_settings_cli.get("workspace_root", "")
    if not isinstance(workspace_root, str):
        workspace_root = ""
    final_console_settings_cli["workspace_root"] = workspace_root.strip()
```

In the config template (near `config.py:2134`), add under `[console]`:

```toml
# local_tools_enabled = false   # workspace-local agent tools (fs_*); approvals via MCP permission store
# workspace_root = ""           # confinement root for fs_* tools; empty = app cwd at startup
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest <the config test file> -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/config.py Tests/
git commit -m "feat: [console] local_tools_enabled + workspace_root config"
```

---

## Task 5: Controller — `build_local_review_hook` + `_compose_local_provider`

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (near `build_mcp_review_hook`, line 70; and near `_compose_mcp_provider`, line 857)
- Test: `Tests/Chat/test_console_local_review_hook.py`

- [ ] **Step 1: Write the failing test**

```python
from tldw_chatbook.Agents.agent_models import ToolCall
from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Chat.console_chat_controller import (
    build_combined_review_hook,
    build_local_review_hook,
)
from tldw_chatbook.MCP.permission_store import EffectiveToolState

ASK = EffectiveToolState(state="ask", origin="global_default")
ALLOW = EffectiveToolState(state="allow", origin="tool_override")


def provider(state, tmp_path):
    return LocalToolProvider(workspace_root=tmp_path, resolve_state=lambda hub: state)


def test_hook_clears_stamps_before_gating(tmp_path):
    p = provider(ASK, tmp_path)
    p.apply_batch_decisions({"fs_list": "approve_once"})
    hook = build_local_review_hook(p, lambda pending: {})
    hook([])  # a turn with no calls still clears
    assert p._stamps == {}


def test_hook_gates_ask_calls_in_one_batch(tmp_path):
    p = provider(ASK, tmp_path)
    seen = []
    hook = build_local_review_hook(p, lambda pending: seen.append(pending) or {"fs_list": "approve_once"})
    verdicts = hook([ToolCall(name="fs_list", args={"path": "."}),
                     ToolCall(name="fs_list", args={"path": "sub"})])
    assert len(seen) == 1 and len(seen[0]) == 2  # ONE round trip for the batch
    assert verdicts == {"fs_list": "proceed"}
    assert p._stamps == {"fs_list": "approve_once"}


def test_hook_skips_non_ask_calls(tmp_path):
    p = provider(ALLOW, tmp_path)
    hook = build_local_review_hook(p, lambda pending: (_ for _ in ()).throw(AssertionError("must not ask")))
    assert hook([ToolCall(name="fs_list", args={"path": "."})]) == {}


def test_combined_hook_merges_verdicts(tmp_path):
    p1, p2 = provider(ASK, tmp_path), provider(ASK, tmp_path)
    hook = build_combined_review_hook([
        build_local_review_hook(p1, lambda pending: {"fs_list": "approve_once"}),
        build_local_review_hook(p2, lambda pending: {"fs_list": "deny"}),
    ])
    # each provider only gates what it owns; both see the batch
    out = hook([ToolCall(name="fs_list", args={"path": "."})])
    assert out == {"fs_list": "proceed"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Tests/Chat/test_console_local_review_hook.py -x -q`
Expected: FAIL (`ImportError: cannot import name 'build_local_review_hook'`)

- [ ] **Step 3: Implement**

Immediately after `build_mcp_review_hook` (`console_chat_controller.py:169`), add, mirroring its docstring discipline (copy the clear-first/I3 rationale comment block and adapt):

```python
def build_local_review_hook(
    provider: "LocalToolProvider",
    request_approvals: Callable[[list["MCPPendingCall"]], dict[str, str]],
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Build this run's review_tool_calls hook for the local provider.

    Identical discipline to build_mcp_review_hook (see its docstring):
    clear-first stamps at entry, ONE approval round trip per batch,
    verdicts only ever "proceed" — invoke() single-sources refusals.
    """

    def review_tool_calls(calls: list["ToolCall"]) -> dict[str, str]:
        provider.apply_batch_decisions({})
        pending: list["MCPPendingCall"] = []
        for call in calls:
            gate = provider.pending_gate_for(call.name, call.args)
            if gate is not None:
                pending.append(gate)
        if not pending:
            return {}
        decisions = request_approvals(pending)
        provider.apply_batch_decisions(decisions)
        return {call.llm_name: "proceed" for call in pending}

    return review_tool_calls


def build_combined_review_hook(
    hooks: list[Callable[[list["ToolCall"]], dict[str, str]]],
) -> Callable[[list["ToolCall"]], dict[str, str]]:
    """Fan one batch through every provider's hook; merge verdict maps.

    Each hook gates only the calls its provider owns (pending_gate_for
    returns None for foreign tools), so merging is collision-free.
    """

    def review_tool_calls(calls: list["ToolCall"]) -> dict[str, str]:
        verdicts: dict[str, str] = {}
        for hook in hooks:
            verdicts.update(hook(calls))
        return verdicts

    return review_tool_calls
```

Then add `_compose_local_provider(self)` next to `_compose_mcp_provider` (`console_chat_controller.py:857`), which:

1. Returns `(None, None)` when `[console] local_tools_enabled` is false. Read the section via `get_cli_setting` (imported at `console_chat_controller.py:43`, used at line 812 for the `[mcp]` section) — there is no `_console_config()` helper in this file.
2. Resolves the workspace root: `Path(cfg.get("workspace_root") or os.getcwd()).resolve()`.
3. Builds `LocalToolProvider(workspace_root=root, resolve_state=..., kill_switch=..., approval_callback=self.request_mcp_approvals)`.
   - `resolve_state`: load the permission payload the same way the MCP composition path does (read `_compose_mcp_provider` and reuse its payload source), then `resolve_effective_state(payload, hub_tool)`.
   - `kill_switch`: the same kill-switch accessor the MCP provider uses.
   - **Persistence of `always_allow`:** when `_verdict_for` returns allow via an `always_allow` stamp, someone must persist `set_tool_state(..., definition_hash=...)` under `local:__local__`. Read how the MCP flow persists always-allow decisions (search `always_allow` in `unified_control_plane_service.py`) and wire the equivalent through the injected callables — if the cleanest seam is a `persist_allow: Callable[[HubTool], None]` param on `LocalToolProvider`, add it (with a provider test). **Anticipate a follow-up edit to Task 3's provider here — do not treat the Task 3 commit as final until this seam lands.**
4. Returns `(provider, build_local_review_hook(provider, self.request_mcp_approvals))`.

Where the run is dispatched (the caller of `_compose_mcp_provider`), compose the local pair too and combine hooks:

```python
hooks = [h for h in (mcp_hook, local_hook) if h is not None]
combined = build_combined_review_hook(hooks) if hooks else None
```

and pass `combined` + `local_provider` into `bridge.run_reply(...)`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/Chat/test_console_local_review_hook.py -q`
Expected: PASS. Also run the existing MCP hook tests to confirm no regression: `python3 -m pytest Tests/Chat/ -k "mcp_review or approval" -q`

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_controller.py Tests/Chat/test_console_local_review_hook.py
git commit -m "feat: local tool review hook + provider composition in console controller"
```

---

## Task 6: Bridge — register local provider, compose stamp scopes

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py:613-660` (`_compose_run_registry_and_allowed`) and `772-941` (`run_reply` signature + scope threading)
- Test: `Tests/Chat/test_console_agent_bridge_local.py` (or extend the existing bridge test file — find with `ls Tests/Chat/ | grep bridge`)

- [ ] **Step 1: Write the failing test**

```python
def test_run_registry_includes_local_tools(tmp_path):
    from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
    from tldw_chatbook.Chat.console_agent_bridge import _compose_run_registry_and_allowed

    local = LocalToolProvider(workspace_root=tmp_path)
    registry, allowed, builtin_names = _compose_run_registry_and_allowed(
        {}, local_provider=local
    )
    names = [e.name for e in registry.list_catalog()]
    assert "fs_list" in names and "calculator" in names
    assert "fs_list" in allowed
    assert "fs_list" not in builtin_names  # skills never narrow/grant local tools


def test_combined_stamp_scope_isolates_both(tmp_path):
    from contextlib import contextmanager

    from tldw_chatbook.Chat.console_agent_bridge import _combined_review_state_scope

    class FakeProvider:
        def __init__(self):
            self.stamps = {"x": "approve_once"}
            self.log = []

        @contextmanager
        def stamp_scope(self):
            saved = self.stamps
            self.stamps = {}
            self.log.append("enter")
            try:
                yield
            finally:
                self.stamps = saved
                self.log.append("exit")

    p1, p2 = FakeProvider(), FakeProvider()
    scope = _combined_review_state_scope(p1, p2)
    assert scope is not None
    with scope():
        assert p1.stamps == {} and p2.stamps == {}
    assert p1.stamps == {"x": "approve_once"} and p2.stamps == {"x": "approve_once"}
    assert p1.log == ["enter", "exit"] and p2.log == ["enter", "exit"]
    assert _combined_review_state_scope(None, None) is None
    assert _combined_review_state_scope(p1, None) is not None  # Nones skipped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest Tests/Chat/test_console_agent_bridge_local.py -x -q`
Expected: FAIL (`TypeError: _compose_run_registry_and_allowed() got an unexpected keyword argument 'local_provider'`)

- [ ] **Step 3: Implement**

In `_compose_run_registry_and_allowed` (`console_agent_bridge.py:613`): add `local_provider: Any | None = None` keyword param; after the builtin registration, add:

```python
    local_names: tuple[str, ...] = ()
    if local_provider is not None:
        registry.register_provider(local_provider)
        local_names = tuple(e.name for e in local_provider.list_catalog())
```

Register order: Builtin → **Local** → Skill → MCP (first-registrant-wins; local names are `fs_`-prefixed so shadowing is theoretical, but register local before skills/MCP so a malicious MCP server can never shadow `fs_*`). Include `local_names` in the returned allow-list union; keep `builtin_names` unchanged. Update the docstring.

In `run_reply` (`console_agent_bridge.py:772`): add `local_provider: Any | None = None` keyword param. Change the registry-composition condition from `if self._skills_service is not None or mcp_provider is not None:` to also trigger when `local_provider is not None`, and pass it through. Replace the `review_state_scope` block (lines 927-931) with:

```python
        review_state_scope = _combined_review_state_scope(mcp_provider, local_provider)
```

and add at module level:

```python
def _combined_review_state_scope(*providers: Any | None):
    """Compose every provider's stamp_scope into one review_state_scope.

    None providers and providers without stamp_scope (test doubles) are
    skipped; returns None when nothing contributes, preserving the
    pre-existing AgentService default.
    """
    import contextlib

    scopes = [
        p.stamp_scope for p in providers
        if p is not None and getattr(p, "stamp_scope", None) is not None
    ]
    if not scopes:
        return None

    @contextlib.contextmanager
    def combined():
        with contextlib.ExitStack() as stack:
            for scope in scopes:
                stack.enter_context(scope())
            yield

    return combined
```

(Check how `AgentService` consumes `review_state_scope` — factory-called per sub-agent spawn per `agent_service.py:133-165` — and match that call signature exactly.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest Tests/Chat/ -k "bridge or local" -q`
Expected: PASS, existing bridge tests green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_agent_bridge.py Tests/Chat/
git commit -m "feat: register LocalToolProvider in per-run registry with composed stamp scopes"
```

---

## Task 7: End-to-end integration test (fence protocol)

**Files:**
- Test: `Tests/Agents/test_local_tools_integration.py`

Find the existing fence-protocol integration test to copy its fake-model harness: `grep -rln "tool_call" Tests/Agents/ | head` and look for the test that drives `AgentService.run_turn` with a fake `chat_call`.

- [ ] **Step 1: Write the test**

Using that harness: fake model whose first turn emits a ```` ```tool_call ```` fence for `{"name": "fs_list", "args": {"path": "."}}` and whose second turn (after the tool result) emits final text. Wire a `LocalToolProvider` on a `tmp_path` containing one file, `resolve_state` returning ask, and `build_local_review_hook` with a fake approval function returning `{"fs_list": "approve_once"}`.

Assert:
1. The run completes; the tool step's result contains the filename.
2. The approval function was called exactly once with an `MCPPendingCall` whose `server_key == "local:__local__"`.
3. Variant: approval returns `{"fs_list": "deny"}` → the tool result content/error equals `LOCAL_DENY_REFUSAL` and the model's second turn still runs.

- [ ] **Step 2: Run**

Run: `python3 -m pytest Tests/Agents/test_local_tools_integration.py -q`
Expected: PASS. Fix integration mismatches (marker formatting, step assertions) — do NOT weaken the approval assertions.

- [ ] **Step 3: Full regression sweep**

Run: `python3 -m pytest Tests/Agents/ Tests/Tools/test_local_tool_impls.py Tests/Chat/ -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add Tests/Agents/test_local_tools_integration.py
git commit -m "test: end-to-end fs_list through agent loop with approval gate"
```

---

## Task 8: Close out

- [ ] **Step 1: Mark the backlog task Done**

Check off every AC (`- [ ]` → `- [x]`), then:

```bash
backlog task edit <id> -s Done --notes "LocalToolProvider + build_local_review_hook + combined hook/scope composition; fs_list pilot proven end-to-end through the fence-protocol loop with the approval card; config keys [console] local_tools_enabled/workspace_root; ADR-032. Spec deviation: fs_list landed here instead of phase 2." || echo "update the task markdown manually"
```

- [ ] **Step 2: AGENTS.md note**

Defer the AGENTS.md "Tool Calling" rewrite to the phase-2/3 task (it should describe the full tool set, not the pilot). No AGENTS.md change in phase 1 beyond confirming nothing it says is now false — the `Tools/` section remains accurate since the legacy executor is untouched.
