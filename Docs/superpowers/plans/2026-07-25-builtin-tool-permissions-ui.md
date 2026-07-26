# Built-in Tool Permissions UI Implementation Plan (TASK-627)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface `agent:builtin` in the MCP workbench's Permissions mode with its effective state, and make persistent allow/deny writable and reversible — unblocking TASK-545/P2 from offering anything beyond session-scoped decisions.

**Architecture:** Four layers, bottom-up. (1) `permission_store` gains a hash-free-namespace exemption so `allow` is writable for keys that carry no meaningful definition hash. (2) A new pure helper enumerates built-in tools and resolves them via `resolve_builtin_state` — never the MCP resolver. (3) `MCPWorkbench` merges that result into the Permissions rows without touching the MCP path. (4) The approval card flags rows a bulk action could not apply to.

**Tech Stack:** Python 3.11, Textual, pytest. No new dependencies.

**Spec:** `Docs/superpowers/specs/2026-07-25-builtin-tool-permissions-ui-design.md` (commits `397260145`, `42668ff5e`). Read it — especially the spike findings, which justify choices that otherwise look like extra work.

## Global Constraints

1. **Never route built-ins through `effective_tool_states()` or `resolve_effective_state`.** They apply MCP semantics (ask-floor + hash check) and call `store.mark_config_changed()` — a rug-pull marker `resolve_builtin_state` ignores. This is the single most important rule here: a comparative spike found CheetahClaws' `plan` mode silently auto-approves mutating MCP tools precisely because external tools fall through a decider written for built-ins.
2. **MCP behavior stays byte-identical.** `resolve_effective_state`, `effective_tool_states`, and the hash guard for every non-exempt `server_key` are unmodified. Existing MCP permission tests must pass untouched.
3. **Namespace:** `agent:builtin` (`BUILTIN_TOOL_SERVER_KEY`). Never `builtin:tldw_chatbook` — that is the built-in MCP *server*. The UI must present them as visibly distinct.
4. **Fail closed on an unrecognized `server_key`** — render `deny`/unknown, never inherit a branch.
5. **Do not reuse `HubTool`** for built-in rows: its `source` enum (`local|builtin|server`) already uses `builtin` for the MCP server.
6. **Session-vs-persistent:** this task makes persistent decisions *possible*. It does not change `BuiltinToolGate`'s runtime behavior, and P1's per-turn payload cache means a mid-run change lands on the agent's **next turn**.
7. Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-p2-tools` (branch `feat/builtin-permissions-ui`); tests via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` FROM the worktree. Never touch the main checkout. `git add` only each task's listed files, never `-A`.
8. **Line numbers below are as-of this branch — re-verify with `grep -n` before editing. The target TEXT is authoritative.** `mcp_workbench.py` is large; **run `python -c "import ast;ast.parse(open(F).read())"` after any structural edit** (an `Edit` on a bare `def foo(` will also match an indented method and silently break the class).

**Baseline:** before Task 1, run `pytest Tests/MCP/ Tests/UI/test_mcp_permissions_mode.py Tests/UI/test_mcp_workbench.py Tests/Agents/ -q` and record pre-existing failures — report, don't fix.

---

### Task 1: Hash-free namespaces in the permission store

**Files:** Modify `tldw_chatbook/MCP/permission_store.py`, `tldw_chatbook/MCP/unified_control_plane_service.py`; Test `Tests/MCP/test_permission_store.py`

**Interfaces:** Produces `HASH_FREE_SERVER_KEYS: frozenset[str]`. `MCPPermissionStore.set_tool_state` and `UnifiedMCPControlPlaneService.set_tool_state` accept `state="allow"` without a hash for keys in that set.

- [ ] **Step 1: Write the failing tests** — append to `Tests/MCP/test_permission_store.py` (reuse its existing store fixtures; check how it builds a temp-path store):

```python
from tldw_chatbook.MCP.permission_store import (
    BUILTIN_TOOL_SERVER_KEY,
    HASH_FREE_SERVER_KEYS,
)


def test_hash_free_keys_contains_exactly_the_builtin_namespace():
    """Pin the CONTENTS. Adding a remote namespace here would silently
    disable the rug-pull guard for it -- the one way this change could
    become a real weakening."""
    assert HASH_FREE_SERVER_KEYS == frozenset({BUILTIN_TOOL_SERVER_KEY})


def test_allow_without_hash_is_permitted_for_the_builtin_namespace(tmp_path):
    store = _store(tmp_path)          # adapt to the file's real helper
    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "allow")
    entry = store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "write_thing")
    assert entry["state"] == "allow"
    assert not entry.get("definition_hash")


def test_allow_without_hash_still_raises_for_an_mcp_server(tmp_path):
    """MCP's guard is unchanged."""
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="definition_hash"):
        store.set_tool_state("local:docs", "some_tool", "allow")


def test_deny_and_clear_need_no_hash_for_either_namespace(tmp_path):
    store = _store(tmp_path)
    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "deny")
    store.set_tool_state("local:docs", "some_tool", "deny")
    store.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", None)
    assert store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "write_thing") in (None, {}) \
        or store.get_tool_entry(BUILTIN_TOOL_SERVER_KEY, "write_thing").get("state") is None
```

Also add a service-level test (in whichever file covers `UnifiedMCPControlPlaneService.set_tool_state`; find it with `grep -rln "set_tool_state" Tests/`): calling `service.set_tool_state(BUILTIN_TOOL_SERVER_KEY, "write_thing", "allow")` with **no** `tool=` argument must succeed, while the same call for an MCP `server_key` must still raise `ValueError`.

- [ ] **Step 2: Run — verify fail.** `pytest Tests/MCP/test_permission_store.py -q -k "hash_free or without_hash"` → ImportError, then ValueError on the built-in case.

- [ ] **Step 3a: Add the constant** in `permission_store.py`, immediately after `BUILTIN_TOOL_SERVER_KEY`:

```python
#: Server keys whose tools carry no meaningful ``definition_hash``, so
#: ``set_tool_state(..., "allow")`` does not require one.
#:
#: The hash is a RUG-PULL guard: it detects a *remote* server changing a
#: tool's description/schema after the user trusted it. ``agent:builtin``
#: tools are in-process code shipped with the app -- an attacker who can
#: change them already has code execution, so the check protects nothing,
#: while a stored hash would force a re-prompt on every release that edits
#: a docstring. ``resolve_builtin_state`` correspondingly never reads one.
#:
#: Adding a REMOTE namespace here would silently disable the rug-pull guard
#: for it; the contents are pinned by test.
HASH_FREE_SERVER_KEYS = frozenset({BUILTIN_TOOL_SERVER_KEY})
```

- [ ] **Step 3b: Relax the store guard.** In `MCPPermissionStore.set_tool_state`, the current guard reads:

```python
        if state is not None:
            _validate_state(state)
            if state == "allow" and not definition_hash:
                raise ValueError("definition_hash is required when state is 'allow'")
```

Change the inner condition to exempt hash-free keys:

```python
        if state is not None:
            _validate_state(state)
            if (
                state == "allow"
                and not definition_hash
                and server_key not in HASH_FREE_SERVER_KEYS
            ):
                raise ValueError("definition_hash is required when state is 'allow'")
```

Extend the method docstring's `Raises:` to say the requirement is waived for `HASH_FREE_SERVER_KEYS`.

- [ ] **Step 3c: Relax the service guard.** In `UnifiedMCPControlPlaneService.set_tool_state`, the body currently computes:

```python
        hash_value: str | None = None
        if ui_state == "allow":
            if tool is None:
                raise ValueError(
                    "tool is required to set state 'allow' (need its description/input_schema)"
                )
            hash_value = definition_hash(tool.description, tool.input_schema)
```

Change to skip both the requirement and the hashing for exempt keys:

```python
        hash_value: str | None = None
        if ui_state == "allow" and server_key not in HASH_FREE_SERVER_KEYS:
            if tool is None:
                raise ValueError(
                    "tool is required to set state 'allow' (need its description/input_schema)"
                )
            hash_value = definition_hash(tool.description, tool.input_schema)
```

Import `HASH_FREE_SERVER_KEYS` alongside the other `permission_store` imports at the top of the module, and update the docstring's `Args:`/`Raises:` to note the exemption.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/MCP/ -q` (the WHOLE MCP suite — Constraint 2 requires MCP behavior provably unchanged).

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/MCP/permission_store.py tldw_chatbook/MCP/unified_control_plane_service.py Tests/MCP/
git commit -m "feat(mcp): exempt hash-free namespaces from the allow definition_hash guard [TASK-627]"
```

---

### Task 2: Enumerate and resolve built-in tool states

**Files:** Modify `tldw_chatbook/Agents/builtin_tool_gate.py` (or create `tldw_chatbook/Agents/builtin_tool_permissions.py` — implementer's call, see note); Test `Tests/Agents/test_builtin_tool_gate.py`

**Interfaces:** Produces `builtin_permission_rows(payload: dict) -> list[BuiltinPermRow]`, where `BuiltinPermRow` carries `name`, `description`, `effective: EffectiveToolState`, and `orphaned: bool`.

**Placement (resolved):** put it in `builtin_tool_gate.py`. It already imports `BUILTIN_TOOL_SERVER_KEY`, `resolve_builtin_state`, `GatedToolRef` and defines `tool_ref()`, so this needs no new imports, and the file is **281 lines** — comfortably under the threshold that would justify a new module. Note in a comment that this is a settings-time enumerator living beside the runtime gate deliberately, to keep one definition of how a built-in tool maps to a `GatedToolRef`.

- [ ] **Step 1: Write the failing tests**

```python
def test_builtin_permission_rows_lists_live_tools_with_resolved_state():
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows

    rows = builtin_permission_rows({})          # empty payload -> the allow floor
    by_name = {r.name: r for r in rows}
    assert "calculator" in by_name and "get_current_datetime" in by_name
    # Untagged tools resolve to the built-in floor, not the MCP "ask" default.
    assert by_name["calculator"].effective.state == "allow"
    assert by_name["calculator"].effective.origin == "builtin_default"
    assert by_name["calculator"].orphaned is False
    assert by_name["calculator"].description        # carried for display


def test_builtin_permission_rows_reflects_a_stored_override():
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows
    from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY

    payload = {"profiles": {"default": {"servers": {
        BUILTIN_TOOL_SERVER_KEY: {"tools": {"calculator": {"state": "deny"}}}
    }}}}
    row = {r.name: r for r in builtin_permission_rows(payload)}["calculator"]
    assert row.effective.state == "deny"
    assert row.effective.origin == "tool_override"


def test_builtin_permission_rows_surfaces_orphaned_stored_entries():
    """A decision stored for a tool a later release removed must still be
    listed, or the user cannot clear it."""
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows
    from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY

    payload = {"profiles": {"default": {"servers": {
        BUILTIN_TOOL_SERVER_KEY: {"tools": {"tool_that_no_longer_exists": {"state": "allow"}}}
    }}}}
    rows = {r.name: r for r in builtin_permission_rows(payload)}
    assert rows["tool_that_no_longer_exists"].orphaned is True
    assert rows["calculator"].orphaned is False


def test_builtin_permission_rows_needs_no_agent_run():
    """Enumeration must not start a run or build a gate."""
    import tldw_chatbook.Agents.tool_catalog as tc
    from tldw_chatbook.Agents.builtin_tool_gate import builtin_permission_rows

    calls = []
    original = tc.build_builtin_gate
    tc.build_builtin_gate = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        builtin_permission_rows({})
    finally:
        tc.build_builtin_gate = original
    assert calls == []          # the lazy gate was never built
```

- [ ] **Step 2: Run — verify fail.** ImportError on `builtin_permission_rows`.

- [ ] **Step 3: Implement.**

```python
@dataclass(frozen=True)
class BuiltinPermRow:
    """One built-in tool's row for the permissions UI.

    Attributes:
        name: The tool's LLM-facing name.
        description: One-line description, empty for an orphaned entry.
        effective: State resolved by ``resolve_builtin_state`` -- NEVER by
            the MCP resolver (see the design doc's spike findings).
        orphaned: True when a stored decision exists for a name no live
            built-in tool provides. Such rows must stay listed so the user
            can clear a decision for a tool a later release removed.
    """

    name: str
    description: str
    effective: EffectiveToolState
    orphaned: bool = False


def builtin_permission_rows(payload: dict) -> list[BuiltinPermRow]:
    """Enumerate built-in tools with their effective permission state.

    Settings-time enumeration: constructs a throwaway ``BuiltinToolProvider``
    (cheap -- it builds two Tool objects and its gate is lazy, built only on
    ``invoke()``), so no agent run is started and no gate is created.

    Args:
        payload: A loaded permission-store payload; ``{}`` is valid and
            resolves everything to the built-in allow floor.

    Returns:
        One row per live built-in tool, plus one per stored ``agent:builtin``
        tool entry with no matching live tool (``orphaned=True``), sorted by
        name.
    """
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider

    provider = BuiltinToolProvider()
    rows: list[BuiltinPermRow] = []
    live: set[str] = set()
    for entry in provider.list_catalog():
        tool = provider.tool_for(entry.name)
        if tool is None:            # defensive: catalog/registry disagree
            continue
        live.add(entry.name)
        rows.append(
            BuiltinPermRow(
                name=entry.name,
                description=entry.one_line_description,
                effective=resolve_builtin_state(payload, tool_ref(tool)),
            )
        )

    for name in _stored_builtin_tool_names(payload) - live:
        rows.append(
            BuiltinPermRow(
                name=name,
                description="",
                effective=resolve_builtin_state(
                    payload,
                    GatedToolRef(
                        server_key=BUILTIN_TOOL_SERVER_KEY,
                        name=name,
                        description="",
                        input_schema=None,
                        tags=(),
                    ),
                ),
                orphaned=True,
            )
        )
    return sorted(rows, key=lambda row: row.name)
```

Plus a small `_stored_builtin_tool_names(payload) -> set[str]` walking `payload["profiles"]["default"]["servers"][BUILTIN_TOOL_SERVER_KEY]["tools"]`. **Import `_as_mapping` from `permission_store`** — it is module-level (`permission_store.py:102`) and is exactly the isinstance-guard chain used by both resolvers, so reusing it keeps malformed-payload behavior identical rather than re-deriving it. It must not raise on a malformed payload.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/Agents/ -q`

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Agents/builtin_tool_gate.py Tests/Agents/test_builtin_tool_gate.py
git commit -m "feat(agents): enumerate built-in tools with resolved permission state [TASK-627]"
```

---

### Task 3: Render the built-in section in Permissions mode

**Files:** Modify `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`; Test `Tests/UI/test_mcp_workbench.py`, `Tests/UI/test_mcp_permissions_mode.py`

**Interfaces:** Consumes Task 2's `builtin_permission_rows`. Produces built-in `PermRow`s appended to the Permissions matrix.

**Read the spec's "three real gaps" before starting.** `_build_permission_rows` (~:1456) iterates `tools_by_server` built from the `tools: list[HubTool]` argument; a key absent from that list is never a candidate. `_resolve_effective_states` (~:1179) is shared with `_sync_tools_mode()` — **do not modify it.**

- [ ] **Step 1: Write the failing tests.** Use the existing Textual harness style in `Tests/UI/test_mcp_workbench.py` (`async with app.run_test() as pilot`). Assertions that must hold:
  - A built-in section appears in Permissions mode listing `calculator`/`get_current_datetime`, even with **no MCP servers configured** (the section must not depend on the MCP catalog).
  - Its server label is visibly distinct from the built-in MCP server's — assert the two labels differ and that `agent:builtin` rows are not grouped under `builtin:tldw_chatbook`.
  - An untagged built-in renders `Allow` (the floor), not `Ask` — proving `resolve_builtin_state` was used, not the MCP resolver.
  - A stored `deny` for a built-in renders `Off` with the `tool_override` marker.
  - `resolve_effective_state` is **not** called for `agent:builtin` — monkeypatch it to `pytest.fail` for that server key, or assert `effective_tool_states` was called only with MCP tools.

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement.** Add a sibling to `_resolve_effective_states` — do not extend it:

```python
    def _builtin_permission_rows(self) -> list:
        """This run's built-in tool rows, resolved by the BUILT-IN resolver.

        Deliberately NOT merged into `_resolve_effective_states()`: that
        method calls `effective_tool_states()`, which applies MCP semantics
        (ask-floor + hash check) and calls `store.mark_config_changed()` --
        a rug-pull marker `resolve_builtin_state` ignores. Routing built-ins
        through it would resolve them wrongly AND store an inert flag. See
        the design doc's spike findings for the failure this avoids.

        Fail-soft like every other service seam here: any failure yields an
        empty list rather than raising into a render pass.
        """
```

It reads the store payload via the same fail-soft `getattr`/`callable`/try-except pattern the other seams use, then calls `builtin_permission_rows(payload)`.

Then in `_sync_permissions_mode`, append a built-in section to the rows handed to the widget, after the MCP sections. Build `PermRow`s with the same `state_label` formatting the MCP path uses (`format_tool_state_label(effective)`) so markers stay consistent, a server-row label distinct from `builtin:tldw_chatbook` (suggested `Built-in (agent runtime)`), and orphaned rows marked in their label.

**Fail closed (Constraint 4):** if a `server_key` reaches row-building that is neither `agent:builtin` nor a live MCP catalog key, render it `deny`/unknown rather than letting it inherit a branch.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/UI/test_mcp_workbench.py Tests/UI/test_mcp_permissions_mode.py Tests/MCP/ -q`

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/
git commit -m "feat(mcp): surface agent:builtin tools in the Permissions matrix [TASK-627]"
```

---

### Task 4: Writing a persistent decision from the UI

**Files:** Modify `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py`; Test `Tests/UI/test_mcp_workbench.py`

**Interfaces:** Consumes Task 1's exemption. The existing `StateCycleRequested` → `on_mcp_permissions_mode_state_cycle_requested` handler must route built-in rows correctly.

The handler currently calls `service.set_tool_state(event.server_key, event.tool_name or "", event.new_state, tool=cycled_tool)` where `cycled_tool` is a `HubTool` found in the MCP catalog. A built-in row has **no** `HubTool` — that lookup will fail or pass `None`.

- [ ] **Step 1: Write the failing tests**
  - Cycling a built-in row to `allow` persists it, with **no** `tool=` argument and no hash written.
  - Cycling to `deny`, then clearing back to inherit, both work.
  - The change is reflected on the next render without a restart.
  - An MCP row's cycle path is **unchanged** — still passes its `HubTool` and still writes a hash for `allow`.
  - Cycling an **orphaned** built-in row to inherit clears the stored entry.

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement.** In the handler, branch on `event.server_key == BUILTIN_TOOL_SERVER_KEY`: skip the `HubTool` lookup and call `service.set_tool_state(server_key, tool_name, new_state)` with no `tool=`. Leave the MCP branch byte-identical. Add a comment naming Task 1's exemption as why the omitted `tool=` is safe here.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/UI/ Tests/MCP/ -q`

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/UI/MCP_Modules/mcp_workbench.py Tests/UI/
git commit -m "feat(mcp): persist allow/deny for built-in tools from the Permissions UI [TASK-627]"
```

---

### Task 5: Flag rows a bulk action could not apply to

**Files:** Modify `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py`, `tldw_chatbook/css/tldw_cli_modular.tcss`; Test `Tests/UI/test_console_mcp_approval.py`

`_set_all_batch_decisions` skips a row whose `legal_values` contains none of the bulk candidates, leaving it visually identical to an untouched row.

**This path is not currently reachable** — today's only narrowed rows (`approve_once`/`approve_session`/`deny`) accept a candidate from both bulk buttons. So the test must construct a row that genuinely excludes both, and the fix is a guard for future narrowings.

- [ ] **Step 1: Write the failing test.** Build a batch with a row whose `options` exclude every bulk candidate (e.g. `["always_allow"]`), click Approve-all, and assert that row carries a `needs-decision` class while an applied row does not. Then change that row's `Select` and assert the class clears.

- [ ] **Step 2: Run — verify fail.**

- [ ] **Step 3: Implement.** In `_set_all_batch_decisions`, when no candidate is legal for a row, add a `needs-decision` class to that row's container.

**There is currently no `Select.Changed` handler in this widget** (verified — `grep -n "Select.Changed\|on_select_changed"` returns nothing), so clearing the flag requires ADDING one: an `@on(Select.Changed)` handler that removes the class from the changed row's container. Keep it narrow — it must not disturb `_submit_batch_decisions`' existing read of `self._batch_selects`, which is how decisions are collected today.

Add a `.approval-row.needs-decision` rule to the modular TCSS near the existing `.approval-row*` rules (~:6435).

**Do not hand-edit the CSS bundle** — this repo regenerates it; edit the modular source and let the build produce the bundle.

- [ ] **Step 4: Run — verify pass.** `pytest Tests/UI/test_console_mcp_approval.py Tests/UI/test_chat_approval_card.py -q`

- [ ] **Step 5: Commit**
```bash
git add tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py tldw_chatbook/css/ Tests/UI/
git commit -m "feat(ui): flag approval rows a bulk action could not apply to [TASK-627]"
```

---

### Task 6: Backlog + docs

- [ ] **Step 1: Close TASK-627** — tick its ACs, add Implementation Notes, set status Done via `backlog task edit 627 -s Done --notes "..."` (or edit the file if the CLI is unavailable). Record that persistent built-in decisions are now available, unblocking TASK-545/P2 from session-scope-only.

- [ ] **Step 2: Update TASK-545's P2 section** to note the persistent-decision prerequisite is satisfied, and TASK-659's description to reference this UI as the permissions half of the agent settings surface.

- [ ] **Step 3: File the spike follow-ups** named in the spec — hard-floor category, permission modes, and the "model must never change its own permission posture" constraint. **Run an ID sweep first** against BOTH `origin/dev` and the working tree; note that a `task-634` duplicate already exists on dev (another session's renumber to 635 is in flight), so re-verify the range at merge time, not just at file time.

- [ ] **Step 4: Commit**
```bash
git add backlog/
git commit -m "chore(backlog): close TASK-627; file spike follow-ups"
```

---

## Post-Implementation

Run `pytest Tests/MCP/ Tests/UI/ Tests/Agents/ Tests/Chat/ -q`, then hand off to the final whole-branch review (opus) and superpowers:finishing-a-development-branch.
