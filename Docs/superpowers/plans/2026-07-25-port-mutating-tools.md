# Port mutating tools behind the gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register `write_file`/`create_note`/`update_note` in the agent runtime's built-in tool provider behind the permission gate, and tag them (plus the already-registered `read_file`/`list_directory`) so the gate's approval path becomes live for the first time.

**Architecture:** A built-in-only risk vocabulary (`BUILTIN_HIGH_RISK_TAGS`) is consulted by `resolve_builtin_state` alone, leaving MCP's `resolve_effective_state` byte-unchanged. Tools declare membership via the existing `Tool.risk_tags` property. Registration reuses the `[tools]` config gates that already govern these same tools on the legacy path.

**Tech Stack:** Python ≥3.11, pytest, existing `MCP/permission_store.py` + `Agents/builtin_tool_gate.py` machinery.

**Spec:** `Docs/superpowers/specs/2026-07-25-port-mutating-tools-design.md`

## Global Constraints

- **Do NOT modify `resolve_effective_state`.** MCP's resolver and its `HIGH_RISK_TAGS` flooring at `permission_store.py:653-660` must remain unchanged. Only `resolve_builtin_state` learns the new vocabulary.
- **`HIGH_RISK_TAGS` itself keeps its exact current value** — `frozenset({"mutates", "process"})`. Widen only the new built-in set.
- **All new `[tools]` gates default to `False`** (off). The default catalog after this change must be exactly `{"calculator", "get_current_datetime"}`.
- **Reuse the existing gate key names** — `write_file_enabled`, `create_note_enabled`, `update_note_enabled` — which already exist at `tool_executor.py:735/763/789`. Do not invent new keys.
- **Use the 3-argument `get_cli_setting("tools", key, False)` form.** The section-dict form (`get_cli_setting("tools")`) silently returns `{}` and would make every gate permanently off.
- **Note `user_id` comes from `load_settings()["USERS_NAME"]`** — never `get_cli_setting("general", "users_name", ...)`, which reads only TOML and diverges from `app.notes_user_id` when the `USERS_NAME` env var is set.
- Tests run from the worktree with: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest`
- Run pytest in the **foreground**. Do not background test runs.
- `git add` only the files listed in your task. Never `git add -A`.

---

### Task 1: Built-in risk vocabulary

**Files:**
- Modify: `tldw_chatbook/MCP/permission_store.py` (add constant after line 69; edit `resolve_builtin_state`'s flooring condition ~line 723-727 and its docstring ~line 690-692)
- Modify: `tldw_chatbook/Tools/tool_executor.py:42-55` (`Tool.risk_tags` docstring only)
- Test: `Tests/MCP/test_permission_store.py` (append to the existing "Task 2: GatedToolRef + resolve_builtin_state" section)

**Interfaces:**
- Consumes: existing `HIGH_RISK_TAGS`, `resolve_builtin_state(payload, tool)`, `GatedToolRef`.
- Produces: `BUILTIN_HIGH_RISK_TAGS: frozenset[str]` exported from `tldw_chatbook.MCP.permission_store`. Tasks 2 and 5 rely on the tag string `"reads"` being in it and on `"mutates"` remaining in it.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/MCP/test_permission_store.py`. The helpers `_ref` and `_payload` already exist in that file (defined around line 266) — reuse them, do not redefine.

```python
# -- P2 Task 1: built-in-only risk vocabulary ---------------------------------

from tldw_chatbook.MCP.permission_store import BUILTIN_HIGH_RISK_TAGS


def test_builtin_risk_set_is_a_strict_superset_of_the_mcp_set():
    """Built-ins floor on everything MCP does, plus reads."""
    assert HIGH_RISK_TAGS < BUILTIN_HIGH_RISK_TAGS
    assert "reads" in BUILTIN_HIGH_RISK_TAGS
    assert "reads" not in HIGH_RISK_TAGS


def test_mcp_high_risk_set_is_unchanged():
    """Pin the CONTENTS: widening this set would make remote MCP tools
    carrying the new tag start prompting, which P2 must not cause."""
    assert HIGH_RISK_TAGS == frozenset({"mutates", "process"})


def test_reads_tag_floors_an_inherited_builtin_allow_to_ask():
    eff = resolve_builtin_state({}, _ref(name="read_file", tags=("reads",)))
    assert eff.state == "ask"
    assert eff.risk_floored is True


def test_reads_tag_does_not_floor_an_mcp_tool():
    """The asymmetry is deliberate: only resolve_builtin_state learned the
    new tag. An MCP tool inheriting `allow` keeps it despite the tag."""
    from tldw_chatbook.MCP.permission_store import resolve_effective_state
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        server_key="local:x", server_label="x", source="local",
        name="t", description="d", input_schema=None, tags=("reads",),
        stale=False, executable=True,
    )
    eff = resolve_effective_state(_payload(global_default="allow"), tool)
    assert eff.state == "allow"
    assert eff.risk_floored is False


def test_mutates_still_floors_an_mcp_tool():
    """Negative control for the test above: MCP's own flooring still works,
    so a passing `test_reads_tag_does_not_floor_an_mcp_tool` proves the tag
    was not added to HIGH_RISK_TAGS -- not merely that flooring is broken."""
    from tldw_chatbook.MCP.permission_store import resolve_effective_state
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    tool = HubTool(
        server_key="local:x", server_label="x", source="local",
        name="t", description="d", input_schema=None, tags=("mutates",),
        stale=False, executable=True,
    )
    eff = resolve_effective_state(_payload(global_default="allow"), tool)
    assert eff.state == "ask"
    assert eff.risk_floored is True


def test_explicit_builtin_tool_override_beats_the_reads_floor():
    """An explicit user choice is still not floored -- same rule the
    `mutates` path already follows."""
    eff = resolve_builtin_state(
        _payload(tool_state="allow"), _ref(tags=("reads",))
    )
    assert eff.state == "allow"
    assert eff.origin == "tool_override"
    assert eff.risk_floored is False
```

`HIGH_RISK_TAGS` must be importable in that test module. Check whether it is already imported at the top of the file; if not, add it to the existing `from tldw_chatbook.MCP.permission_store import (...)` block at line 257.

`_payload(tool_state="allow")` keys its tool entry on the name `"calculator"`, and `_ref()` defaults to that same name — so leave `_ref`'s name defaulted in that last test.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_permission_store.py -k "reads or risk_set or high_risk_set" -v
```

Expected: FAIL — `ImportError: cannot import name 'BUILTIN_HIGH_RISK_TAGS'`.

- [ ] **Step 3: Add the constant**

In `tldw_chatbook/MCP/permission_store.py`, immediately after the `HIGH_RISK_TAGS` line (currently line 69):

```python
HIGH_RISK_TAGS = frozenset({"mutates", "process"})
#: Risk tags that floor an INHERITED ``allow`` to ``ask`` for in-process
#: built-ins. A superset of ``HIGH_RISK_TAGS``: built-ins additionally
#: treat filesystem reads as prompt-worthy, because an agent reading
#: arbitrary sandbox files is a disclosure risk even though it mutates
#: nothing. MCP deliberately keeps ``HIGH_RISK_TAGS`` -- widening the
#: shared set would make remote tools carrying ``"reads"`` start
#: prompting, which is not this phase's call to make.
BUILTIN_HIGH_RISK_TAGS = HIGH_RISK_TAGS | frozenset({"reads"})
```

- [ ] **Step 4: Point `resolve_builtin_state` at the new set**

In `resolve_builtin_state`, change ONLY the flooring condition (currently lines 723-727):

```python
    risk_floored = False
    if (
        origin != "tool_override"
        and state == "allow"
        and set(tool.tags) & BUILTIN_HIGH_RISK_TAGS
    ):
```

And in the same function's docstring (currently ~line 690-692), change:

```
    The high-risk floor is unchanged: an INHERITED ``allow`` (not an
    explicit tool override) whose tags intersect ``HIGH_RISK_TAGS`` is
    downgraded to ``ask`` with ``risk_floored=True``.
```

to:

```
    The high-risk floor: an INHERITED ``allow`` (not an explicit tool
    override) whose tags intersect ``BUILTIN_HIGH_RISK_TAGS`` is
    downgraded to ``ask`` with ``risk_floored=True``. That set is a
    superset of MCP's ``HIGH_RISK_TAGS`` -- built-ins additionally floor
    on ``"reads"``.
```

**Do not touch `resolve_effective_state`** (its identical-looking block is ~70 lines earlier, at 653-660). Verify with `git diff` that your change to the flooring condition appears exactly once.

- [ ] **Step 5: Update the `Tool.risk_tags` docstring**

In `tldw_chatbook/Tools/tool_executor.py`, the `risk_tags` property (line 42) currently documents `HIGH_RISK_TAGS` as the vocabulary, which this task makes false. Replace its docstring body:

```python
    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Risk classes for the permission gate, e.g. ``("mutates",)``.

        Concrete with an empty default so every existing subclass keeps
        working unchanged. For tools reached through the agent runtime
        the vocabulary is the permission store's
        ``BUILTIN_HIGH_RISK_TAGS`` (``mutates``/``process``/``reads``) --
        a tool tagged with one of those has an INHERITED ``allow``
        floored to ``ask`` by ``resolve_builtin_state``. MCP tools are
        resolved against the narrower ``HIGH_RISK_TAGS`` instead.
        Tools with no elevated risk leave this empty.

        Returns:
            A tuple of risk tag strings drawn from
            ``BUILTIN_HIGH_RISK_TAGS``; empty for a tool with no
            elevated risk.
        """
        return ()
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/MCP/test_permission_store.py -v
```

Expected: PASS, including every pre-existing test in the file (the MCP resolver must be unaffected).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/MCP/permission_store.py tldw_chatbook/Tools/tool_executor.py Tests/MCP/test_permission_store.py
git commit -m "feat: add built-in-only risk tag vocabulary

BUILTIN_HIGH_RISK_TAGS extends HIGH_RISK_TAGS with 'reads' and is
consulted only by resolve_builtin_state, leaving MCP's resolver
unchanged."
```

---

### Task 2: Tag the five tools

**Files:**
- Modify: `tldw_chatbook/Tools/file_operation_tools.py` (`ReadFileTool` ~53, `ListDirectoryTool` ~145, `WriteFileTool` ~309)
- Modify: `tldw_chatbook/Tools/note_management_tools.py` (`CreateNoteTool` ~15, `UpdateNoteTool` ~179)
- Test: `Tests/Agents/test_builtin_tool_risk_tags.py` (create)

**Interfaces:**
- Consumes: `BUILTIN_HIGH_RISK_TAGS` from Task 1; the existing `Tool.risk_tags` property.
- Produces: five tool classes whose `risk_tags` are non-empty. Tasks 3 and 5 depend on these exact values: `write_file`/`create_note`/`update_note` → `("mutates",)`; `read_file`/`list_directory` → `("reads",)`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Agents/test_builtin_tool_risk_tags.py`:

```python
"""TASK-545 P2: the ported tools must declare risk tags.

Until this lands, no shipped tool overrides `risk_tags`, so the built-in
gate's entire `ask` path is exercised only by tests. Tagging is what makes
the approval machinery live for real users.
"""

import pytest

from tldw_chatbook.MCP.permission_store import BUILTIN_HIGH_RISK_TAGS
from tldw_chatbook.Tools.file_operation_tools import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from tldw_chatbook.Tools.note_management_tools import CreateNoteTool, UpdateNoteTool


@pytest.mark.parametrize(
    "factory,expected",
    [
        (WriteFileTool, ("mutates",)),
        (CreateNoteTool, ("mutates",)),
        (UpdateNoteTool, ("mutates",)),
        (ReadFileTool, ("reads",)),
        (ListDirectoryTool, ("reads",)),
    ],
)
def test_tool_declares_expected_risk_tags(factory, expected):
    assert factory().risk_tags == expected


@pytest.mark.parametrize(
    "factory",
    [WriteFileTool, CreateNoteTool, UpdateNoteTool, ReadFileTool, ListDirectoryTool],
)
def test_every_declared_tag_is_in_the_builtin_vocabulary(factory):
    """A typo'd tag would silently never floor anything to `ask`."""
    tags = set(factory().risk_tags)
    assert tags, "tool declares no risk tags"
    assert tags <= BUILTIN_HIGH_RISK_TAGS, (
        f"unrecognized risk tags: {tags - BUILTIN_HIGH_RISK_TAGS}"
    )


def test_read_only_search_tool_stays_untagged():
    """Scope guard: SearchNotesTool is explicitly out of P2's scope, so it
    must not acquire tags and start prompting."""
    from tldw_chatbook.Tools.note_management_tools import SearchNotesTool

    assert SearchNotesTool().risk_tags == ()
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_builtin_tool_risk_tags.py -v
```

Expected: FAIL — every parametrized case asserting a non-empty tuple fails with `assert () == ('mutates',)`. `test_read_only_search_tool_stays_untagged` should PASS already.

- [ ] **Step 3: Add `risk_tags` to the three file tools**

In `tldw_chatbook/Tools/file_operation_tools.py`, add the property to each class immediately after its `parameters` property and before its `execute` method.

`ReadFileTool` (after the `parameters` property ending at line 80):

```python
    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Reading arbitrary sandbox files is a disclosure risk."""
        return ("reads",)
```

`ListDirectoryTool` (after the `parameters` property ending at line 184):

```python
    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Enumerating the sandbox discloses its structure."""
        return ("reads",)
```

`WriteFileTool` (after the `parameters` property ending at line 351):

```python
    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Creates, overwrites, or appends to files."""
        return ("mutates",)
```

- [ ] **Step 4: Add `risk_tags` to the two note tools**

In `tldw_chatbook/Tools/note_management_tools.py`, same placement — after `parameters`, before `execute`.

`CreateNoteTool` (after the `parameters` property ending at line 35):

```python
    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Inserts a note row into the user's database."""
        return ("mutates",)
```

`UpdateNoteTool` (after the `parameters` property ending at line 208):

```python
    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Mutates an existing note the user owns."""
        return ("mutates",)
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_builtin_tool_risk_tags.py Tests/Tools/ -v
```

Expected: PASS, including the pre-existing `Tests/Tools/test_file_tool_sandbox.py`.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Tools/file_operation_tools.py tldw_chatbook/Tools/note_management_tools.py Tests/Agents/test_builtin_tool_risk_tags.py
git commit -m "feat: tag file and note tools with risk classes

write_file/create_note/update_note as 'mutates'; the already-registered
read_file/list_directory as 'reads', closing a live gap where enabling
either gate produced silent unprompted filesystem reads."
```

---

### Task 3: Register the mutating tools and cover their names

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py:194-214` (the `BuiltinToolProvider.__init__` gate loop)
- Modify: `tldw_chatbook/Library/library_skills_state.py:39-76` (`_SHADOWED_BUILTIN_NAMES`)
- Test: `Tests/Agents/test_builtin_file_tools.py` (extend the existing file)

**Interfaces:**
- Consumes: Task 2's tagged tool classes; the existing `tools_config` fixture and `_names` helper already in `Tests/Agents/test_builtin_file_tools.py`.
- Produces: `BuiltinToolProvider` registers `write_file`, `create_note`, `update_note` when their `[tools]` gates are on. Task 5 depends on `BuiltinToolProvider().tool_for("write_file")` returning the tool when `write_file_enabled` is set.

**Context:** the loop already handles `read_file`/`list_directory`. The note tools live in a **different module** (`note_management_tools`) than the file tools, so the existing single-module loop cannot express them — the pair must become a triple carrying the module name.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Agents/test_builtin_file_tools.py`. The `tools_config` fixture and `_names` helper are already defined at the top of that file — reuse them.

```python
# -- P2: the mutating tools ---------------------------------------------------


def test_mutating_tools_absent_by_default(tools_config):
    """Default posture is unchanged: all three gates default to disabled."""
    names = _names(BuiltinToolProvider())
    assert "write_file" not in names
    assert "create_note" not in names
    assert "update_note" not in names
    assert names == {"calculator", "get_current_datetime"}


@pytest.mark.parametrize(
    "gate_key,tool_name",
    [
        ("write_file_enabled", "write_file"),
        ("create_note_enabled", "create_note"),
        ("update_note_enabled", "update_note"),
    ],
)
def test_mutating_tool_appears_when_its_gate_is_enabled(
    tools_config, gate_key, tool_name
):
    tools_config[gate_key] = True
    assert tool_name in _names(BuiltinToolProvider())


def test_each_mutating_gate_is_independent(tools_config):
    tools_config["write_file_enabled"] = True
    names = _names(BuiltinToolProvider())
    assert "write_file" in names
    assert "create_note" not in names
    assert "update_note" not in names


def test_registered_mutating_tools_carry_their_risk_tags(tools_config):
    """Registration must surface the SAME tagged classes -- an untagged
    duplicate would register fine and silently never prompt."""
    tools_config["write_file_enabled"] = True
    tools_config["create_note_enabled"] = True
    tools_config["update_note_enabled"] = True
    provider = BuiltinToolProvider()
    for name in ("write_file", "create_note", "update_note"):
        assert provider.tool_for(name).risk_tags == ("mutates",)


def test_all_gated_tool_names_are_covered_by_the_shadow_guard(tools_config):
    """Extends the task-584 guard to P2's names.

    The drift guard in Tests/Library builds a BuiltinToolProvider with
    DEFAULT config, so config-gated tools are structurally invisible to
    it. These names must therefore be pinned explicitly.
    """
    from tldw_chatbook.Library.library_skills_state import _SHADOWED_BUILTIN_NAMES

    for key in (
        "read_file_enabled",
        "list_directory_enabled",
        "write_file_enabled",
        "create_note_enabled",
        "update_note_enabled",
    ):
        tools_config[key] = True
    gated = _names(BuiltinToolProvider())
    assert gated <= _SHADOWED_BUILTIN_NAMES, (
        f"gated builtin tool names not covered: {gated - _SHADOWED_BUILTIN_NAMES}"
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_builtin_file_tools.py -v
```

Expected: the new tests FAIL (`write_file` etc. not in the catalog); the pre-existing tests in the file PASS.

- [ ] **Step 3: Extend the registration loop**

In `tldw_chatbook/Agents/tool_catalog.py`, replace the `for gate_key, factory_name in (...)` loop (currently lines 200-214) with a three-element form that carries the module, since the note tools live in a different module:

```python
        for gate_key, module_name, factory_name in (
            ("read_file_enabled", "file_operation_tools", "ReadFileTool"),
            ("list_directory_enabled", "file_operation_tools", "ListDirectoryTool"),
            # TASK-545 P2: the mutating tools. Same [tools] gate keys that
            # already govern them on the legacy ToolExecutor path
            # (tool_executor.py:735/763/789), same default-disabled posture.
            # These are tagged ("mutates",), so reaching them still costs an
            # approval -- registration makes them reachable, not automatic.
            ("write_file_enabled", "file_operation_tools", "WriteFileTool"),
            ("create_note_enabled", "note_management_tools", "CreateNoteTool"),
            ("update_note_enabled", "note_management_tools", "UpdateNoteTool"),
        ):
            try:
                from ..config import get_cli_setting

                if not get_cli_setting("tools", gate_key, False):
                    continue
                import importlib

                module = importlib.import_module(
                    f"..Tools.{module_name}", package=__package__
                )
                tool = getattr(module, factory_name)()
            except Exception:  # noqa: BLE001 — an unavailable tool is just absent
                continue
            self._tools[tool.name] = tool
```

Keep the existing task-584 comment block above the loop; add to it:

```python
        # task-584: surface the app's existing sandbox-rooted file tools to the
        # agent loop. They were registered on the global ToolExecutor but never
        # reachable from here, so retained script output -- deliberately written
        # under the file-tool sandbox root -- had no consumer. Behind the SAME
        # [tools] gates that already govern them, which default to DISABLED:
        # this changes reachability, not the default posture. TASK-545 P2 adds
        # the mutating tools on the same terms.
```

- [ ] **Step 4: Add the three names to the shadow set**

In `tldw_chatbook/Library/library_skills_state.py`, inside the `_SHADOWED_BUILTIN_NAMES` frozenset, after the existing `"read_file",` / `"list_directory",` entries (currently lines 73-74):

```python
        "read_file",
        "list_directory",
        # TASK-545 P2's mutating tools. Same rationale as the two above:
        # CONFIG-GATED, so the drift guard (which builds a
        # BuiltinToolProvider with default config) cannot see them. A skill
        # named `write_file` shadows a real builtin the moment a user turns
        # the gate on.
        "write_file",
        "create_note",
        "update_note",
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_builtin_file_tools.py Tests/Library/test_library_skills_state.py Tests/Agents/test_tool_catalog.py -v
```

Expected: PASS. `test_shadow_name_set_stays_in_sync_with_real_sources` must still pass — it asserts a **subset** relation (real names ⊆ set), so adding names cannot break it.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Library/library_skills_state.py Tests/Agents/test_builtin_file_tools.py
git commit -m "feat: register the mutating tools in BuiltinToolProvider

Behind the existing default-off [tools] gates, and added to the Library
shadow-name set (config-gated tools are invisible to its drift guard)."
```

---

### Task 4: Note tools resolve the real user

**Files:**
- Modify: `tldw_chatbook/Tools/note_management_tools.py` (add a module-level helper; then lines 69, 246, 261)
- Test: `Tests/Tools/test_note_tool_user_id.py` (create)

**Interfaces:**
- Consumes: `tldw_chatbook.config.load_settings`.
- Produces: `_resolve_user_id() -> str` at module scope in `note_management_tools`, monkeypatchable by tests.

**Context:** `app.py:3139-3140` sets `self.notes_user_id = settings.get("USERS_NAME", "default_tui_user")` **once** at init from `load_settings()`, and `config.py:826` resolves that as `os.getenv("USERS_NAME", <toml value>)`. The tools' hardcoded `"default_user"` happens to equal `config.py`'s own `default_users_name_fallback`, so this is a **no-op for unconfigured users** — it only fixes the case where a user set a name and their agent-created notes landed in a bucket the Notes UI never reads.

Note `SearchNotesTool` (line 87) also passes a `user_id`. It is **out of scope** — do not modify it. Changing where it reads from would alter which notes an existing user's searches return, which is a behavior change P2 did not design for.

- [ ] **Step 1: Write the failing test**

Create `Tests/Tools/test_note_tool_user_id.py`:

```python
"""TASK-545 P2: note tools must write as the real configured user.

Both tools hardcoded `user_id="default_user"` with a "Would be actual user
in production" comment, while the app resolves `notes_user_id` from
`load_settings()["USERS_NAME"]`. A user who set [general] users_name got
agent-created notes in a bucket their Notes UI never reads.
"""

import pytest

import tldw_chatbook.Tools.note_management_tools as nmt


class _FakeNotesService:
    """Captures the user_id every call was made with."""

    def __init__(self, **kwargs):
        _FakeNotesService.last = self
        self.calls = []

    def add_note(self, user_id, title, content):
        self.calls.append(("add_note", user_id))
        return "note-1"

    def get_note_by_id(self, user_id, note_id):
        self.calls.append(("get_note_by_id", user_id))
        return {"id": note_id, "version": 1}

    def update_note(self, user_id, note_id, update_data, expected_version):
        self.calls.append(("update_note", user_id))
        return True


@pytest.fixture
def fake_service(monkeypatch):
    monkeypatch.setattr(nmt, "NotesInteropService", _FakeNotesService)
    return _FakeNotesService


@pytest.mark.asyncio
async def test_create_note_uses_the_configured_user(monkeypatch, fake_service):
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")
    result = await nmt.CreateNoteTool().execute(title="t", content="c")
    assert "error" not in result
    assert fake_service.last.calls == [("add_note", "alice")]


@pytest.mark.asyncio
async def test_update_note_uses_the_configured_user_on_every_call(
    monkeypatch, fake_service
):
    """Both the existence check and the write must use the same id -- a
    mismatch would read one user's note and write another's."""
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")
    result = await nmt.UpdateNoteTool().execute(note_id="n1", title="t2")
    assert "error" not in result
    assert fake_service.last.calls == [
        ("get_note_by_id", "alice"),
        ("update_note", "alice"),
    ]


def test_resolver_reads_users_name_from_load_settings(monkeypatch):
    monkeypatch.setattr(nmt, "load_settings", lambda: {"USERS_NAME": "bob"})
    assert nmt._resolve_user_id() == "bob"


def test_resolver_honors_the_env_var_override(monkeypatch):
    """The real value is os.getenv("USERS_NAME", <toml>) resolved INSIDE
    load_settings -- reading TOML directly would diverge from
    app.notes_user_id and create a third bucket."""
    import tldw_chatbook.config as config_module

    monkeypatch.setenv("USERS_NAME", "env_user")
    settings = config_module.load_settings(force_reload=True)
    assert settings["USERS_NAME"] == "env_user"
    monkeypatch.setattr(nmt, "load_settings", lambda: settings)
    assert nmt._resolve_user_id() == "env_user"


def test_resolver_falls_back_when_settings_are_unavailable(monkeypatch):
    """A tool must never crash because config could not be read."""

    def boom():
        raise RuntimeError("config unavailable")

    monkeypatch.setattr(nmt, "load_settings", boom)
    assert nmt._resolve_user_id() == "default_user"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_note_tool_user_id.py -v
```

Expected: FAIL — `AttributeError: module ... has no attribute '_resolve_user_id'`.

- [ ] **Step 3: Add the resolver**

In `tldw_chatbook/Tools/note_management_tools.py`, extend the module-level imports (currently lines 10-12) and add the helper before `class CreateNoteTool`:

```python
from . import Tool
from ..Notes.Notes_Library import NotesInteropService
from ..config import USER_DB_BASE_DIR, load_settings

#: Matches config.py's own `default_users_name_fallback`, so an
#: unconfigured user sees no change from the previously hardcoded value.
_DEFAULT_USER_ID = "default_user"


def _resolve_user_id() -> str:
    """Return the user id notes should be written under.

    Reads ``load_settings()["USERS_NAME"]`` -- the SAME source
    ``app.notes_user_id`` comes from (``app.py:3139``). Deliberately not
    ``get_cli_setting("general", "users_name", ...)``: the real value is
    ``os.getenv("USERS_NAME", <toml value>)`` resolved inside
    ``load_settings`` (``config.py:826``), so a direct TOML read would
    diverge from the app whenever the env var is set and strand notes in
    a third bucket.

    Resolved per call rather than at construction time: a
    ``BuiltinToolProvider`` is built from four sites, two of which have no
    app access, and the tool classes take no constructor arguments.

    Returns:
        The configured user id, or ``"default_user"`` if settings cannot
        be read.
    """
    try:
        return load_settings().get("USERS_NAME") or _DEFAULT_USER_ID
    except Exception as e:  # noqa: BLE001 — a tool must not crash on config
        logger.warning(f"Could not resolve USERS_NAME, using default: {e}")
        return _DEFAULT_USER_ID
```

- [ ] **Step 4: Use it at all three call sites**

In `CreateNoteTool.execute`, replace the comment block and the `add_note` call (currently lines 56-72):

```python
        try:
            from ..config import chachanotes_db

            notes_service = NotesInteropService(
                base_db_directory=USER_DB_BASE_DIR,
                api_client_id="tool_executor",
                global_db_to_use=chachanotes_db,
            )

            note_id = notes_service.add_note(
                user_id=_resolve_user_id(),
                title=title,
                content=content,
            )
```

In `UpdateNoteTool.execute`, replace both `user_id="default_user"` occurrences (currently lines 246 and 261). Resolve **once** and reuse, so the existence check and the write cannot disagree:

```python
        try:
            from ..config import chachanotes_db

            notes_service = NotesInteropService(
                base_db_directory=USER_DB_BASE_DIR,
                api_client_id="tool_executor",
                global_db_to_use=chachanotes_db,
            )

            user_id = _resolve_user_id()

            # First, get the current note to check it exists
            current_note = notes_service.get_note_by_id(
                user_id=user_id, note_id=note_id
            )
```

and further down:

```python
            success = notes_service.update_note(
                user_id=user_id,
                note_id=note_id,
                update_data=update_data,
                expected_version=expected_version,
            )
```

Leave `SearchNotesTool` untouched.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_note_tool_user_id.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 6: Verify no hardcoded ids remain**

```bash
grep -n "default_user" tldw_chatbook/Tools/note_management_tools.py
```

Expected: matches only in `_DEFAULT_USER_ID`'s definition/comment and in `SearchNotesTool` (out of scope). Neither `CreateNoteTool` nor `UpdateNoteTool` may contain a literal `"default_user"`.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Tools/note_management_tools.py Tests/Tools/test_note_tool_user_id.py
git commit -m "fix: note tools write as the configured user

Resolve user_id from load_settings()[USERS_NAME] -- the same source
app.notes_user_id uses -- instead of a hardcoded default_user."
```

---

### Task 5: Prove the gate is live end to end

**Files:**
- Test: `Tests/Agents/test_builtin_gate_live_tools.py` (create)

**Interfaces:**
- Consumes: everything from Tasks 1-4. Uses `BuiltinToolProvider(gate=...)`, `BuiltinToolGate`, and the `_FakeService` pattern from `Tests/Agents/test_builtin_tool_gate.py`.
- Produces: nothing consumed downstream.

**Context:** This task adds no production code. Its purpose is to prove the machinery P1 built — which until now was reachable only by tests using a synthetic `_Mutating` tool — actually behaves correctly with the real ported tools. `BuiltinToolGate.check()` has an explicit comment saying its fail-closed branch is "unreachable in P1; P2's mutating tools make it live". These tests are that claim's evidence.

Read `Tests/Agents/test_builtin_tool_gate.py` first — reuse its `_FakeService`/`_FakePermissionStore` classes by importing them rather than redefining.

- [ ] **Step 1: Write the tests**

Create `Tests/Agents/test_builtin_gate_live_tools.py`:

```python
"""TASK-545 P2: the approval path with REAL ported tools.

P1's gate was exercised only by a synthetic `_Mutating` fixture, because no
shipped tool declared risk tags. These tests drive the same machinery with
the actual registered tools -- the first coverage that the feature works
for a user rather than for a test double.
"""

import threading

import pytest

from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
from tldw_chatbook.MCP.permission_store import BUILTIN_TOOL_SERVER_KEY

from Tests.Agents.test_builtin_tool_gate import _FakeService


@pytest.fixture
def tools_config(monkeypatch):
    """Drive the [tools] gates (mirrors test_builtin_file_tools.py)."""
    values = {}
    import tldw_chatbook.config as config_module

    def fake(section, key=None, default=None):
        if section != "tools" or not isinstance(key, str):
            return default
        return values.get(key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", fake)
    return values


@pytest.fixture
def all_gates_on(tools_config):
    for key in ("write_file_enabled", "create_note_enabled", "read_file_enabled"):
        tools_config[key] = True
    return tools_config


def _provider(service):
    return BuiltinToolProvider(gate=BuiltinToolGate(service=service))


def test_a_mutating_tool_is_refused_without_approval(all_gates_on):
    """The core of the phase: write_file cannot run unprompted."""
    provider = _provider(_FakeService())
    result = provider.invoke("builtin:write_file", {"file_path": "x", "content": "y"})
    assert result.ok is False
    assert "approval" in result.error


def test_a_reads_tool_is_refused_without_approval(all_gates_on):
    """Closes the live gap: enabling read_file no longer means silent reads."""
    provider = _provider(_FakeService())
    result = provider.invoke("builtin:read_file", {"file_path": "x"})
    assert result.ok is False
    assert "approval" in result.error


def test_an_untagged_tool_still_runs_unprompted(all_gates_on):
    """Regression guard: calculator must not start prompting."""
    provider = _provider(_FakeService())
    result = provider.invoke("builtin:calculator", {"expression": "1+1"})
    assert result.ok is True


def test_a_stamped_permit_lets_a_mutating_tool_run(all_gates_on, tmp_path):
    """The approval round trip: a per-turn permit reaches execution."""
    service = _FakeService()
    gate = BuiltinToolGate(service=service)
    gate.begin_turn()
    gate.stamp("write_file", "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    target = tmp_path / "out.txt"
    import tldw_chatbook.Tools.file_operation_tools as fot

    monkey = pytest.MonkeyPatch()
    monkey.setattr(fot, "_resolve_sandbox_config", lambda: str(tmp_path))
    try:
        result = provider.invoke(
            "builtin:write_file", {"file_path": "out.txt", "content": "hello"}
        )
    finally:
        monkey.undo()

    assert result.ok is True, result.error
    assert target.read_text(encoding="utf-8") == "hello"


def test_a_resolved_deny_beats_a_permitting_stamp(all_gates_on):
    """The property Qodo caught in P1, now with a real tool: `Off` is
    absolute. Built-ins have no catalog filtering, so invoke() is the only
    barrier -- a stamp must never shadow it."""
    payload = {
        "profiles": {
            "default": {
                "servers": {
                    BUILTIN_TOOL_SERVER_KEY: {
                        "tools": {"write_file": {"state": "deny"}}
                    }
                }
            }
        }
    }
    gate = BuiltinToolGate(service=_FakeService(payload=payload))
    gate.begin_turn()
    gate.stamp("write_file", "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    result = provider.invoke("builtin:write_file", {"file_path": "x", "content": "y"})
    assert result.ok is False
    assert "Off" in result.error


def test_a_refusal_is_a_result_never_an_exception(all_gates_on):
    """The pure loop must never see an exception from tool invocation."""
    provider = _provider(_FakeService())
    for tool_id, args in (
        ("builtin:write_file", {"file_path": "x", "content": "y"}),
        ("builtin:create_note", {"title": "t", "content": "c"}),
        ("builtin:read_file", {"file_path": "x"}),
    ):
        result = provider.invoke(tool_id, args)
        assert result.ok is False
        assert isinstance(result.error, str) and result.error


@pytest.mark.parametrize(
    "gate_key,tool_name,args",
    [
        ("create_note_enabled", "create_note", {"title": "t", "content": "c"}),
        ("update_note_enabled", "update_note", {"note_id": "n1", "title": "t2"}),
    ],
)
def test_note_tool_executes_on_a_worker_thread(
    tools_config, monkeypatch, gate_key, tool_name, args
):
    """The agent service invokes tools off the main thread, and these note
    tools have never run there. `asyncio.run` inside invoke() requires no
    running loop on that thread."""
    import tldw_chatbook.Tools.note_management_tools as nmt

    tools_config[gate_key] = True
    seen = {}

    class _FakeNotes:
        def __init__(self, **kwargs):
            pass

        def _record(self, user_id):
            seen["thread"] = threading.current_thread().name
            seen["user_id"] = user_id

        def add_note(self, user_id, title, content):
            self._record(user_id)
            return "note-1"

        def get_note_by_id(self, user_id, note_id):
            self._record(user_id)
            return {"id": note_id, "version": 1}

        def update_note(self, user_id, note_id, update_data, expected_version):
            self._record(user_id)
            return True

    monkeypatch.setattr(nmt, "NotesInteropService", _FakeNotes)
    monkeypatch.setattr(nmt, "_resolve_user_id", lambda: "alice")

    gate = BuiltinToolGate(service=_FakeService())
    gate.begin_turn()
    gate.stamp(tool_name, "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    box = {}

    def run():
        box["result"] = provider.invoke(f"builtin:{tool_name}", args)

    worker = threading.Thread(target=run, name="tool-worker")
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive(), "tool invocation hung on the worker thread"
    assert box["result"].ok is True, box["result"].error
    assert seen["thread"] == "tool-worker"
    assert seen["user_id"] == "alice"


def test_a_parents_approval_survives_a_nested_sub_agent_run(all_gates_on):
    """task-628's stamp_scope, now carrying a REAL tool.

    A spawned sub-agent shares the parent's gate instance and review
    closure, and the hook's first act is begin_turn() -- which clears every
    stamp. Without the scope, a parent's approved write_file becomes
    unusable the moment a sub-agent runs. P1/task-628 proved this with a
    synthetic tool; this is the first coverage with a tool a user can
    actually reach.
    """
    gate = BuiltinToolGate(service=_FakeService())
    gate.begin_turn()
    gate.stamp("write_file", "approve_once")
    provider = BuiltinToolProvider(gate=gate)

    with gate.stamp_scope():
        # Stand in for the child run: it clears the turn and records its own
        # verdicts on the SAME gate.
        gate.begin_turn()
        gate.stamp("read_file", "deny")

    # The parent's approval must be back, and the child's verdict gone.
    result = provider.invoke("builtin:write_file", {"file_path": "x", "content": "y"})
    assert result.ok is not False or "approval" not in (result.error or ""), (
        "parent's stamp was clobbered by the nested run"
    )
    assert gate._stamps.get("read_file") is None, "child's verdict leaked to the parent"
```

Note the stamp vocabulary: `_PERMITTING = {"approve_once", "approve_session", "always_allow"}` (`builtin_tool_gate.py:31`). `"allow"` is **not** a permitting stamp — `stamp()` accepts any string but only those three permit, so a stamp of `"allow"` on a `mutates` tool still fails closed. Use `"approve_once"`.

The `write_file` assertion above is deliberately loose about the sandbox: the call may still fail on path containment (a bare `x` outside the configured root), and this test is about the *stamp surviving*, not about writing. What it must never see is the "requires approval" refusal.

- [ ] **Step 2: Run the tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_builtin_gate_live_tools.py -v
```

Expected: PASS (9 tests — the worker-thread case is parametrized over both note tools).

If `test_a_stamped_permit_lets_a_mutating_tool_run` fails on sandbox containment, read `WriteFileTool.execute` and `_resolve_sandbox_config` in `tldw_chatbook/Tools/file_operation_tools.py` and adjust the monkeypatch target to match how that tool actually resolves its root — do **not** relax the assertion that the file was written.

If the cross-module import `from Tests.Agents.test_builtin_tool_gate import _FakeService` does not resolve, check whether `Tests/` is a package on the path (`Tests/Agents/__init__.py` exists, so it should be). If it genuinely cannot be imported, copy `_FakeService` and `_FakePermissionStore` into the new file with a comment naming the source — do not weaken the tests to avoid the dependency.

- [ ] **Step 3: Sabotage-verify the deny test**

A test that passes for the wrong reason is worthless. Temporarily move the `state.state == "deny"` check in `BuiltinToolGate.check()` (`tldw_chatbook/Agents/builtin_tool_gate.py`) to **after** the stamp checks, then:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_builtin_gate_live_tools.py::test_a_resolved_deny_beats_a_permitting_stamp -v
```

Expected: FAIL. Then **revert the sabotage** (`git checkout tldw_chatbook/Agents/builtin_tool_gate.py`) and re-run to confirm PASS. Report both outcomes.

- [ ] **Step 4: Run the full affected suites**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/ Tests/MCP/ Tests/Tools/ Tests/Library/test_library_skills_state.py -q
```

Expected: PASS. Any failure that also fails on a clean `origin/dev` checkout is a pre-existing baseline — verify before reporting it as such.

- [ ] **Step 5: Commit**

```bash
git add Tests/Agents/test_builtin_gate_live_tools.py
git commit -m "test: prove the builtin gate is live with the real ported tools

Covers refusal without approval, the stamped-permit round trip, deny
beating a stamp, result-not-exception, and worker-thread execution."
```

---

## Final verification

- [ ] Default posture unchanged: with no `[tools]` keys set, `BuiltinToolProvider().list_catalog()` yields exactly `calculator` and `get_current_datetime`.
- [ ] `git diff origin/dev -- tldw_chatbook/MCP/permission_store.py` shows no change inside `resolve_effective_state`.
- [ ] Update `backlog/tasks/task-545*.md` — mark the P2 acceptance criteria complete and add an `## Implementation Notes` section. Do **not** mark the task Done; P3 remains.
- [ ] File the three follow-ups from the spec: `UpdateNoteTool.expected_version` default-of-1; per-call `CharactersRAGDB` construction; surfacing the sandbox root. Sweep `backlog/tasks/` against `origin/dev` for ID collisions before assigning numbers.
