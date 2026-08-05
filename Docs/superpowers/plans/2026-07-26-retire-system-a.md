# Retire System A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the dead `ToolExecutor` (System A), repoint the Settings ▸ Tools switches at the agent runtime's real tool provider, repair the save path, and fix the `[database]` config bug in `quick_ingest()`.

**Architecture:** A single module-level table in `tool_catalog.py` becomes the one source of truth for which built-in tools are config-gateable. `BuiltinToolProvider.__init__` and the Settings UI both derive from it, so they cannot disagree. The Settings UI writes the same `[tools]` keys the provider reads, via a merging save.

**Tech Stack:** Python ≥3.11, Textual, pytest.

**Spec:** `Docs/superpowers/specs/2026-07-26-retire-system-a-design.md`

## Global Constraints

- **Task order matters.** Task 2 removes the Settings window's dependency on System A; Task 3 deletes System A. Doing 3 before 2 leaves the app broken. Each task must leave the tree working.
- **All `[tools]` gates default to `False`.** After every task, a `BuiltinToolProvider()` built with default config must expose exactly `{"calculator", "get_current_datetime"}`.
- **`save_settings_to_cli_config({"tools": {...}})` MERGES** — verified: keys not in the dict survive. Use it. Never use `save_setting_to_cli_config(section, None, dict)`; that shape raises `KeyError: 'None'` and is the bug being fixed.
- **Reset means OFF, not ON.** The old reset set every switch to `True`. Gated tools default to disabled, and several are `mutates` — reset must restore `False`.
- **Textual markup:** a `Label` containing `[reads]` is parsed as markup. Never put risk tags in square brackets in a widget label; use parentheses.
- Keep `Tool`, `CalculatorTool`, `DateTimeTool` in `Tools/tool_executor.py`. They are imported by `builtin_tool_gate`, `tool_catalog`, and `code_audit_tool`.
- Tests: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` — run from the worktree, in the **FOREGROUND**.
- `git add` only the files your task names. Never `git add -A`.

---

### Task 1: One source of truth for gateable built-ins

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (the `BuiltinToolProvider.__init__` gate loop, ~line 194-240)
- Test: `Tests/Agents/test_gateable_builtin_tools.py` (create)

**Interfaces:**
- Produces, all importable from `tldw_chatbook.Agents.tool_catalog`:
  - `class GateableTool(NamedTuple)` with fields `gate_key: str`, `module_name: str`, `factory_name: str`, `tool_name: str`
  - `gateable_builtin_tools() -> tuple[GateableTool, ...]`
  - `build_gateable_tool(entry: GateableTool) -> Any` — instantiates; **raises** on failure
  - `ALWAYS_ON_BUILTIN_NAMES: tuple[str, ...]`
- Task 2 consumes all four.

**Context:** the constructor currently inlines its gate table as a literal tuple in the `for` statement. This task lifts it out without changing behavior. The P2 test `Tests/Agents/test_builtin_file_tools.py::test_a_failed_registration_logs_instead_of_vanishing` pins that a failed registration still **logs** — preserve that.

- [ ] **Step 1: Write the failing test**

Create `Tests/Agents/test_gateable_builtin_tools.py`:

```python
"""TASK-545 P3: the UI and the runtime must agree on which tools exist.

Settings needs the full set of gateable tools INCLUDING disabled ones, which
no provider instance can supply (a provider lists only what its gates already
permit). Both now derive from one table.
"""

import pytest

from tldw_chatbook.Agents.tool_catalog import (
    ALWAYS_ON_BUILTIN_NAMES,
    BuiltinToolProvider,
    GateableTool,
    build_gateable_tool,
    gateable_builtin_tools,
)


@pytest.fixture
def tools_config(monkeypatch):
    values = {}
    import tldw_chatbook.config as config_module

    def fake(section, key=None, default=None):
        if section != "tools" or not isinstance(key, str):
            return default
        return values.get(key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", fake)
    return values


def test_every_gateable_tool_is_listed_even_when_its_gate_is_off(tools_config):
    """THE deadlock regression test.

    A provider built with all gates off exposes none of these; the UI must
    still be able to offer a switch for each.
    """
    listed = {e.tool_name for e in gateable_builtin_tools()}
    assert {
        "read_file",
        "list_directory",
        "write_file",
        "create_note",
        "update_note",
    } <= listed

    provider_names = {e.name for e in BuiltinToolProvider().list_catalog()}
    assert not (listed & provider_names), (
        "gates are off, so a provider must expose none of them -- if this "
        "fails the test is not proving what it claims"
    )


def test_declared_tool_name_matches_the_real_tool(tools_config):
    """A typo in the table would render a switch that saves a dead key."""
    for entry in gateable_builtin_tools():
        assert build_gateable_tool(entry).name == entry.tool_name


def test_gate_key_actually_enables_that_tool(tools_config):
    """The table's gate_key must be the key the constructor reads."""
    for entry in gateable_builtin_tools():
        tools_config.clear()
        tools_config[entry.gate_key] = True
        names = {e.name for e in BuiltinToolProvider().list_catalog()}
        assert entry.tool_name in names, (
            f"{entry.gate_key} did not enable {entry.tool_name}"
        )


def test_always_on_names_match_the_default_catalog(tools_config):
    assert set(ALWAYS_ON_BUILTIN_NAMES) == {
        e.name for e in BuiltinToolProvider().list_catalog()
    }


def test_build_gateable_tool_raises_rather_than_returning_none():
    """The constructor logs the reason, so failures must carry one."""
    bogus = GateableTool("x_enabled", "no_such_module", "NoSuchTool", "x")
    with pytest.raises(Exception):
        build_gateable_tool(bogus)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/test_gateable_builtin_tools.py -q
```

Expected: FAIL — `ImportError: cannot import name 'GateableTool'`.

- [ ] **Step 3: Add the table and helpers**

In `tldw_chatbook/Agents/tool_catalog.py`, add `NamedTuple` to the `typing` import, and insert above `class BuiltinToolProvider:`:

```python
class GateableTool(NamedTuple):
    """A built-in tool that a ``[tools]`` config flag turns on or off.

    Attributes:
        gate_key: The ``[tools]`` key that enables it (default False).
        module_name: Module under ``tldw_chatbook.Tools`` defining it.
        factory_name: Class name to instantiate.
        tool_name: The name the LLM calls it by.
    """

    gate_key: str
    module_name: str
    factory_name: str
    tool_name: str


#: Built-ins registered unconditionally -- no gate, cannot be turned off.
ALWAYS_ON_BUILTIN_NAMES: tuple[str, ...] = ("calculator", "get_current_datetime")

#: THE source of truth for config-gateable built-ins. Both
#: `BuiltinToolProvider.__init__` and the Settings UI derive from this, so
#: they cannot disagree about which tools exist. The UI needs entries for
#: tools whose gate is OFF -- which is exactly why it cannot ask a provider,
#: since a provider only lists what its gates already permit.
_GATEABLE_BUILTINS: tuple[GateableTool, ...] = (
    GateableTool(
        "read_file_enabled", "file_operation_tools", "ReadFileTool", "read_file"
    ),
    GateableTool(
        "list_directory_enabled",
        "file_operation_tools",
        "ListDirectoryTool",
        "list_directory",
    ),
    GateableTool(
        "write_file_enabled", "file_operation_tools", "WriteFileTool", "write_file"
    ),
    GateableTool(
        "create_note_enabled", "note_management_tools", "CreateNoteTool", "create_note"
    ),
    GateableTool(
        "update_note_enabled", "note_management_tools", "UpdateNoteTool", "update_note"
    ),
)


def gateable_builtin_tools() -> tuple[GateableTool, ...]:
    """Every config-gateable built-in, whether or not its gate is on.

    Returns:
        The full table, in registration order.
    """
    return _GATEABLE_BUILTINS


def build_gateable_tool(entry: GateableTool) -> Any:
    """Instantiate ``entry``'s tool class.

    Raises rather than returning ``None`` so callers can report *why* a tool
    is unavailable -- the registration loop logs the exception, and the
    Settings UI degrades the row.

    Args:
        entry: The table entry to construct.

    Returns:
        The instantiated ``Tool``.

    Raises:
        Exception: Whatever import or construction raised.
    """
    import importlib

    module = importlib.import_module(
        f"..Tools.{entry.module_name}", package=__package__
    )
    return getattr(module, entry.factory_name)()
```

- [ ] **Step 4: Point the constructor at the table**

Replace the `for gate_key, module_name, factory_name in (...)` loop body (keep the comment block above it) with:

```python
        for entry in _GATEABLE_BUILTINS:
            try:
                from ..config import get_cli_setting

                if not get_cli_setting("tools", entry.gate_key, False):
                    continue
                tool = build_gateable_tool(entry)
            except Exception as exc:  # noqa: BLE001 — an unavailable tool is just absent
                # Log rather than vanish silently. The gate-off path `continue`s
                # ABOVE this handler, so reaching here means the user asked for
                # the tool and it could not be built -- indistinguishable from
                # "gate is off" without this line. That is not hypothetical:
                # note_management_tools was unimportable on dev for an unknown
                # period (it imported a name that exists only inside a string
                # literal in config.py) and nothing surfaced it. The legacy
                # path logs the same failure (tool_executor.py:725/738/779/805).
                logger.warning(
                    f"Could not register builtin tool {entry.factory_name} "
                    f"(gate {entry.gate_key} is enabled): {exc}"
                )
                continue
            self._tools[tool.name] = tool
```

- [ ] **Step 5: Run the tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/ -q -p no:randomly
```

Expected: PASS, **including** `test_builtin_file_tools.py::test_a_failed_registration_logs_instead_of_vanishing` and `::test_a_disabled_gate_is_not_logged_as_a_failure`. If either fails you changed the logging behavior — fix it, do not edit those tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_gateable_builtin_tools.py
git commit -m "feat: single source of truth for gateable built-in tools

Lifts BuiltinToolProvider's inline gate table to a module constant and
exposes it, so the Settings UI can list tools whose gate is off."
```

---

### Task 2: Repoint the Settings UI and repair the save

**Files:**
- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py` (compose ~3236-3325, save `_save_tool_settings` ~4220-4318, reset `_reset_tool_settings` ~4320-4345)
- Test: `Tests/UI/test_settings_tools_section.py` (create)

**Interfaces:**
- Consumes Task 1's `gateable_builtin_tools()`, `build_gateable_tool()`, `ALWAYS_ON_BUILTIN_NAMES`, `GateableTool`.
- Produces: no new symbols. After this task, `Tools_Settings_Window.py` contains **zero** references to `get_tool_executor` / `reload_tool_executor`.

**Context:** three things are broken here and all three are fixed in this task — the list is derived from a dead executor, the save raises `KeyError: 'None'`, and six controls configure an executor about to be deleted. There is **no existing test coverage** for any of it.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_settings_tools_section.py`:

```python
"""TASK-545 P3: the Settings tools section must control the real runtime.

Before this, the section was dead in four directions: its config read
returned {}, its executor had no callers, its save raised KeyError: 'None',
and it had no tests.
"""

import pytest

from tldw_chatbook.Agents.tool_catalog import (
    BuiltinToolProvider,
    gateable_builtin_tools,
)


@pytest.fixture
def tools_config(monkeypatch):
    values = {}
    import tldw_chatbook.config as config_module

    def fake(section, key=None, default=None):
        if section != "tools" or not isinstance(key, str):
            return default
        return values.get(key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", fake)
    return values


def test_saving_a_gate_key_round_trips_to_the_provider(tmp_path, monkeypatch):
    """config -> save -> provider: the round trip the UI depends on.

    Drives the real save helper the UI uses, not a mock, because the bug
    being fixed was IN that call shape.
    """
    cfg = tmp_path / "config.toml"
    cfg.write_text('[general]\nusers_name = "t"\n', encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(cfg))

    import tldw_chatbook.config as config_module

    config_module.load_settings(force_reload=True)
    try:
        assert config_module.save_settings_to_cli_config(
            {"tools": {"write_file_enabled": True}}
        )
        config_module.load_settings(force_reload=True)
        names = {e.name for e in BuiltinToolProvider().list_catalog()}
        assert "write_file" in names
    finally:
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None


def test_saving_leaves_unrendered_tools_keys_alone(tmp_path, monkeypatch):
    """A save must never silently disable a hand-edited flag."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[general]\nusers_name = "t"\n\n[tools]\ncreate_note_enabled = true\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(cfg))

    import tldw_chatbook.config as config_module

    config_module.load_settings(force_reload=True)
    try:
        config_module.save_settings_to_cli_config({"tools": {"read_file_enabled": True}})
        config_module.load_settings(force_reload=True)
        assert config_module.get_cli_setting("tools", "create_note_enabled") is True
    finally:
        config_module._SETTINGS_CACHE = None
        config_module._SETTINGS_CACHE_SOURCE = None


def test_the_broken_save_shape_is_gone():
    """`save_setting_to_cli_config(section, None, dict)` raises KeyError:
    'None'. Pin that the UI no longer uses it."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert 'save_setting_to_cli_config("tools", None' not in src


def test_settings_window_no_longer_touches_system_a():
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert "get_tool_executor" not in src
    assert "reload_tool_executor" not in src


def test_orphaned_executor_controls_are_gone():
    """Timeout/worker/cache controls configured only the deleted executor."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    for widget_id in (
        "tool-timeout-input",
        "tool-max-workers-input",
        "tool-cache-enabled",
        "tool-cache-max-size-input",
        "tool-cache-ttl-input",
        "tool-cache-persist",
    ):
        assert widget_id not in src, f"{widget_id} still present"


def test_risk_tags_are_not_rendered_as_textual_markup():
    """A Label containing [reads] would be parsed as markup, not shown."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert '[{tags}]' not in src and '[{", ".join' not in src


def test_every_gateable_tool_gets_a_switch_id():
    """The compose loop must cover the whole table, not a subset."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/UI/Tools_Settings_Window.py").read_text(
        encoding="utf-8"
    )
    assert "gateable_builtin_tools()" in src
    assert 'f"tool-switch-{entry.tool_name}"' in src
```

- [ ] **Step 2: Run to verify failure**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_settings_tools_section.py -q
```

Expected: the four source-assertion tests FAIL; the two round-trip tests PASS (they exercise `config.py`, which is already correct).

- [ ] **Step 3: Replace the compose block**

In `_compose_tools()`, replace everything from `# Get available tools from the tool executor` through the end of the `Collapsible(title="Tool Configuration", ...)` block (up to but NOT including `# Save and reset buttons`) with:

```python
            from ..Agents.tool_catalog import (
                ALWAYS_ON_BUILTIN_NAMES,
                build_gateable_tool,
                gateable_builtin_tools,
            )

            yield Label("Available Tools", classes="settings-label")
            yield Static(
                "Enable tools for the agent. Tools with a risk tag always ask "
                "for approval before each call -- enabling one makes it "
                "reachable, not automatic.",
                classes="section-description",
            )

            for tool_name in ALWAYS_ON_BUILTIN_NAMES:
                with Horizontal(classes="tool-item"):
                    with Container(classes="tool-info"):
                        yield Label(tool_name, classes="tool-name")
                        yield Static(
                            "Always available", classes="tool-description"
                        )

            for entry in gateable_builtin_tools():
                try:
                    tool = build_gateable_tool(entry)
                    description = tool.description
                    tags = ", ".join(tool.risk_tags)
                except Exception as exc:  # noqa: BLE001 — degrade the row, not the screen
                    logger.warning(
                        f"Could not describe builtin tool {entry.factory_name}: {exc}"
                    )
                    description = "Unavailable on this system."
                    tags = ""

                # Parentheses, NOT brackets: Textual parses [reads] as markup.
                label = f"{entry.tool_name}  ({tags})" if tags else entry.tool_name
                is_enabled = bool(tools_config.get(entry.gate_key, False))

                with Horizontal(classes="tool-item"):
                    yield Switch(
                        value=is_enabled,
                        id=f"tool-switch-{entry.tool_name}",
                        classes="tool-switch",
                    )
                    with Container(classes="tool-info"):
                        yield Label(label, classes="tool-name")
                        yield Static(description, classes="tool-description")
```

Leave the `# Save and reset buttons` block and the `Tool Usage Statistics` collapsible as they are.

`logger` is already imported in this module (`from loguru import logger`, line 53) -- do not add an import.

- [ ] **Step 4: Replace the save method**

Replace the whole body of `_save_tool_settings` with:

```python
    async def _save_tool_settings(self) -> None:
        """Save Tool Settings to the configuration file."""
        try:
            from ..Agents.tool_catalog import gateable_builtin_tools
            from ..config import (
                load_cli_config_and_ensure_existence,
                save_settings_to_cli_config,
            )

            updates: dict = {}
            for entry in gateable_builtin_tools():
                try:
                    switch = self.query_one(f"#tool-switch-{entry.tool_name}", Switch)
                except Exception:  # noqa: BLE001 — a row that isn't mounted
                    continue
                updates[entry.gate_key] = switch.value

            # Merges: [tools] keys with no switch here are left untouched, so
            # a save can never silently disable a hand-edited flag. The old
            # `save_setting_to_cli_config("tools", None, dict)` call raised
            # KeyError: 'None' -- there is no section-replacement API.
            if save_settings_to_cli_config({"tools": updates}):
                self.app_instance.notify(
                    f"Tool Settings saved! ({len(updates)} settings)",
                    severity="information",
                )
                self.config_data = load_cli_config_and_ensure_existence()
            else:
                self.app_instance.notify(
                    "Failed to save Tool Settings", severity="error"
                )

        except Exception as e:
            self.app_instance.notify(
                f"Error saving Tool Settings: {e}", severity="error"
            )
```

No executor reload: `BuiltinToolProvider` is constructed per run and reads config at construction, so the next agent run picks the change up.

- [ ] **Step 5: Replace the reset method**

Replace the body of `_reset_tool_settings` with:

```python
    async def _reset_tool_settings(self) -> None:
        """Reset Tool Settings to defaults (every gated tool OFF)."""
        try:
            from ..Agents.tool_catalog import gateable_builtin_tools

            reset_count = 0
            for entry in gateable_builtin_tools():
                try:
                    switch = self.query_one(f"#tool-switch-{entry.tool_name}", Switch)
                except Exception:  # noqa: BLE001 — a row that isn't mounted
                    continue
                # Defaults are DISABLED. The previous implementation reset every
                # switch to True, which would now enable mutating tools.
                switch.value = False
                reset_count += 1

            self.app_instance.notify(
                f"Tool Settings reset to defaults ({reset_count} tools disabled). "
                "Save to apply.",
                severity="information",
            )
        except Exception as e:
            self.app_instance.notify(
                f"Error resetting Tool Settings: {e}", severity="error"
            )
```

- [ ] **Step 6: Run the tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_settings_tools_section.py Tests/Agents/ -q -p no:randomly
```

Expected: PASS. Also confirm the module still imports:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.UI.Tools_Settings_Window; print('imports OK')"
```

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Tools_Settings_Window.py Tests/UI/test_settings_tools_section.py
git commit -m "fix: Settings tools section controls the real runtime

Lists every gateable built-in (including disabled ones), repairs the save
that raised KeyError: 'None', resets to OFF rather than ON, and drops six
controls that configured only the dead executor."
```

---

### Task 3: Delete System A

**Files:**
- Modify: `tldw_chatbook/Tools/tool_executor.py` (delete `ToolResultCache` ~86-290, `ToolExecutor` ~291-519, `_global_executor`/`get_tool_executor`/`reload_tool_executor` ~652-847)
- Modify: `tldw_chatbook/Tools/__init__.py` (imports + `__all__`)
- Delete: `Tests/Tools/test_tool_cache_json.py`
- Test: `Tests/Tools/test_system_a_is_retired.py` (create)

**Interfaces:**
- Removes `ToolExecutor`, `ToolResultCache`, `get_tool_executor`, `reload_tool_executor` from `tldw_chatbook.Tools` and `tldw_chatbook.Tools.tool_executor`.
- `Tool`, `DateTimeTool`, `CalculatorTool` stay exactly where they are.

**Context:** verified before writing this plan — `execute_tool_calls` has zero production callers; MCP executes via its own `local_runtime_delegate`; after Task 2 the Settings window is the last referrer and no longer refers. Deleting `get_tool_executor` also removes the only call to `install_claude_code_hooks()`, which monkeypatched `WriteFileTool.execute` — a deliberate part of this change (spec §6).

- [ ] **Step 1: Write the failing test**

Create `Tests/Tools/test_system_a_is_retired.py`:

```python
"""TASK-545 P3: System A is gone; its live parts remain.

ToolExecutor had no execution path -- zero production callers of
execute_tool_calls, and its only referrers listed tools for a Settings
screen. What stayed is the half of the module that IS load-bearing.
"""

import pytest


def test_the_dead_symbols_are_gone():
    import tldw_chatbook.Tools.tool_executor as te

    for name in (
        "ToolExecutor",
        "ToolResultCache",
        "get_tool_executor",
        "reload_tool_executor",
    ):
        assert not hasattr(te, name), f"{name} should have been deleted"


def test_the_package_no_longer_exports_them():
    import tldw_chatbook.Tools as tools

    for name in (
        "ToolExecutor",
        "get_tool_executor",
        "reload_tool_executor",
    ):
        assert name not in tools.__all__
        with pytest.raises(AttributeError):
            getattr(tools, name)


def test_the_load_bearing_half_survives():
    """System B imports these; the gate imports Tool."""
    from tldw_chatbook.Tools.tool_executor import (
        CalculatorTool,
        DateTimeTool,
        Tool,
    )

    assert CalculatorTool().name == "calculator"
    assert DateTimeTool().name == "get_current_datetime"
    assert Tool.risk_tags is not None


def test_system_b_and_the_gate_still_import():
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate  # noqa: F401
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
    from tldw_chatbook.Tools.code_audit_tool import CodeAuditTool  # noqa: F401

    assert {e.name for e in BuiltinToolProvider().list_catalog()} == {
        "calculator",
        "get_current_datetime",
    }


def test_no_production_code_references_the_deleted_symbols():
    import pathlib

    offenders = []
    for path in pathlib.Path("tldw_chatbook").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in ("get_tool_executor", "reload_tool_executor", "ToolResultCache"):
            if name in text:
                offenders.append(f"{path}: {name}")
    assert not offenders, offenders


def test_opening_settings_no_longer_patches_write_file():
    """install_claude_code_hooks had exactly one caller: the deleted
    registration. WriteFileTool.execute must stay unpatched."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/Tools/tool_executor.py").read_text(
        encoding="utf-8"
    )
    assert "install_claude_code_hooks" not in src
```

- [ ] **Step 2: Run to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/test_system_a_is_retired.py -q
```

Expected: FAIL — the symbols still exist.

- [ ] **Step 3: Delete the dead classes and functions**

From `tldw_chatbook/Tools/tool_executor.py`, delete:
- `class ToolResultCache` in its entirety
- `class ToolExecutor` in its entirety
- the `_global_executor` module global, `get_tool_executor()`, and `reload_tool_executor()` at the end of the file

Keep `class Tool`, `class DateTimeTool`, `class CalculatorTool`. Remove imports that become unused (check `OrderedDict`, `ThreadPoolExecutor`, `Tuple`, etc. — run the import check in Step 5).

Update the module docstring to state what it now holds: the `Tool` ABC and the two always-on built-in tools.

**After editing, AST-check before anything else** — this file has several classes and a mis-anchored edit can pull a method out of its class:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import ast; ast.parse(open('tldw_chatbook/Tools/tool_executor.py').read()); print('AST OK')"
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "
import ast
t = ast.parse(open('tldw_chatbook/Tools/tool_executor.py').read())
print(sorted(n.name for n in t.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))))"
```

Expected top-level names: exactly `['CalculatorTool', 'DateTimeTool', 'Tool']`.

- [ ] **Step 4: Update the package exports**

In `tldw_chatbook/Tools/__init__.py`, drop `ToolExecutor`, `get_tool_executor`, `reload_tool_executor` from both the `from .tool_executor import (...)` block and `__all__`, leaving `Tool`, `DateTimeTool`, `CalculatorTool`. Update the module docstring lines that name the removed symbols (they appear around lines 14 and 21-22).

- [ ] **Step 5: Delete the cache's test file**

```bash
git rm Tests/Tools/test_tool_cache_json.py
```

It imports `ToolResultCache` and tests nothing else.

- [ ] **Step 6: Run the tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Tools/ Tests/Agents/ Tests/UI/test_settings_tools_section.py -q -p no:randomly
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('app imports OK')"
```

Expected: PASS. If any test outside these paths imports a deleted symbol, report it — do not re-add the symbol without saying so.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Tools/tool_executor.py tldw_chatbook/Tools/__init__.py Tests/Tools/test_system_a_is_retired.py
git commit -m "refactor: retire System A's tool executor

ToolExecutor/ToolResultCache/get_tool_executor had no execution path --
zero production callers. Keeps the Tool ABC and the two built-in tools
System B runs. Also removes the only install_claude_code_hooks() call,
which monkeypatched WriteFileTool.execute when Settings was opened."
```

---

### Task 4: Fix `quick_ingest()`'s dead `[database]` read (TASK-658)

**Files:**
- Modify: `tldw_chatbook/Local_Ingestion/local_file_ingestion.py:1192`
- Test: `Tests/Local_Ingestion/test_quick_ingest_db_path.py` (create; check whether `Tests/Local_Ingestion/` exists and follow the neighbouring layout if so)

**Interfaces:** none new.

**Context:** `get_cli_setting("database", {})` returns `{}` unconditionally — a bare section name with a non-string second argument lands in the default slot and `config.py` returns it before ever reading the section. So `quick_ingest()` ignores a configured `media_db_path` and writes to the hardcoded default. Independent of Tasks 1-3.

- [ ] **Step 1: Write the failing test**

```python
"""TASK-658: quick_ingest() ignored [database] media_db_path.

`get_cli_setting("database", {})` passes a non-string in the *key* slot for a
section name with no dot, and config.py returns the default before reading
anything. The configured path was silently discarded.
"""

import pytest


def test_configured_media_db_path_is_honored(tmp_path, monkeypatch):
    import tldw_chatbook.Local_Ingestion.local_file_ingestion as lfi

    configured = tmp_path / "configured_media.db"
    monkeypatch.setattr(
        lfi, "get_cli_setting", lambda *a, **k: None, raising=False
    )

    seen = {}

    class _FakeMediaDatabase:
        def __init__(self, db_path, client_id):
            seen["db_path"] = db_path

        def close_connection(self):
            pass

    monkeypatch.setattr(lfi, "MediaDatabase", _FakeMediaDatabase)
    monkeypatch.setattr(lfi, "ingest_local_file", lambda *a, **k: {"ok": True})

    import tldw_chatbook.config as config_module

    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(configured) if (section, key) == ("database", "media_db_path") else default
        ),
    )

    lfi.quick_ingest(tmp_path / "some_file.txt")
    assert seen["db_path"] == str(configured)


def test_fallback_applies_only_when_the_key_is_absent(tmp_path, monkeypatch):
    import tldw_chatbook.Local_Ingestion.local_file_ingestion as lfi
    import tldw_chatbook.config as config_module

    seen = {}

    class _FakeMediaDatabase:
        def __init__(self, db_path, client_id):
            seen["db_path"] = db_path

        def close_connection(self):
            pass

    monkeypatch.setattr(lfi, "MediaDatabase", _FakeMediaDatabase)
    monkeypatch.setattr(lfi, "ingest_local_file", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )

    lfi.quick_ingest(tmp_path / "some_file.txt")
    assert "tldw_cli_media_v2.db" in seen["db_path"]
```

If the fake-injection points don't match the real module (e.g. `MediaDatabase` is imported inside the function rather than at module scope), adapt the monkeypatch targets to what the code actually does and say so in your report — do **not** weaken the assertions.

- [ ] **Step 2: Run to verify it fails**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Local_Ingestion/test_quick_ingest_db_path.py -q
```

Expected: the first test FAILS (the configured path is ignored).

- [ ] **Step 3: Fix the call**

At `local_file_ingestion.py:1192`, replace:

```python
        db_config = get_cli_setting("database", {})
        db_path = db_config.get(
            "media_db_path", "~/.local/share/tldw_cli/tldw_cli_media_v2.db"
        )
```

with:

```python
        # Three-argument form. `get_cli_setting("database", {})` put a
        # non-string in the KEY slot, and for a section name with no dot
        # config.py returns the default before reading the section at all --
        # so the configured path was silently discarded.
        db_path = get_cli_setting(
            "database",
            "media_db_path",
            "~/.local/share/tldw_cli/tldw_cli_media_v2.db",
        )
```

- [ ] **Step 4: Run the tests**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Local_Ingestion/ -q -p no:randomly
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Local_Ingestion/local_file_ingestion.py Tests/Local_Ingestion/test_quick_ingest_db_path.py
git commit -m "fix: quick_ingest honors [database] media_db_path (TASK-658)"
```

---

### Task 5: Close the backlog and file the follow-ups

**Files:**
- Modify: `backlog/tasks/task-545 - *.md`, `backlog/tasks/task-547 - *.md`, `backlog/tasks/task-658 - *.md`
- Create: four new task files

**Context:** TASK-545 has been open across three phases; P3 completes it.

- [ ] **Step 1: Determine the next free IDs**

Backlog IDs in this repo collide constantly — `origin/dev` alone is **not** sufficient. Sweep every remote branch:

```bash
git fetch origin
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python - <<'PY'
import subprocess, re
branches = [b.strip() for b in subprocess.run(
    ['git','branch','-r','--format=%(refname:short)'],capture_output=True,text=True
).stdout.splitlines() if b.strip() and 'HEAD' not in b]
taken = set()
for b in branches:
    out = subprocess.run(['git','ls-tree','-r','--name-only',b,'backlog/'],
                         capture_output=True,text=True).stdout
    taken |= {int(m.group(1)) for m in re.finditer(r'task-(\d+)', out)}
print("next free:", max(taken)+1)
PY
```

Use five consecutive IDs starting there. Re-run this check immediately before merge.

- [ ] **Step 2: File the five follow-ups**

Copy the frontmatter shape from an existing file (e.g. `task-691 - *.md`): `id`, `title`, `status: To Do`, `assignee: []`, `created_date`, `labels`, `dependencies: [TASK-545]`, `priority`, then `## Description` and `## Acceptance Criteria` inside the `<!-- SECTION:DESCRIPTION:BEGIN -->` / `<!-- AC:BEGIN -->` markers.

1. **`[splash_screen]` config is ignored** — `Widgets/splash_screen.py:196` and `Widgets/settings_splash_screen_viewer.py:54` both call `get_cli_setting("splash_screen", <dict>)`, which returns the default unconditionally. Confirmed at runtime that a real config has a populated `[splash_screen]` dict that is discarded. `CLAUDE.md` documents this as a key configuration section, so every user who customized their splash screen has been silently ignored. Priority: medium.
2. **`[web_server]` config is ignored** — `Web_Server/serve.py:331` calls `get_cli_setting("web_server", default={})`; a bare section with no key returns the default. Confirmed the section exists and is populated in a real config. Priority: medium.
3. **TTS OpenAI backend reads nothing** — `TTS/backends/openai.py:96,105,114` call `get_cli_setting("openai_api")` / `("API")` / `("app_tts")` with no key, so all three always return `None`. Priority: medium.
4. **Rehome file-operation auditing** — `install_claude_code_hooks()` monkeypatched `WriteFileTool.execute`, and its only caller was System A's registration, so the patch landed only if the user opened the Settings screen. P3 deleted that caller. If auditing agent file writes is wanted, its home is the gate/provider seam every call already passes through (`BuiltinToolProvider.invoke`), not a side effect of instantiating a UI screen. `Tools/file_operation_hooks.py` and `Tools/code_audit_tool.py` still exist and are now unreferenced by any install path. Priority: low. Note `code_audit` itself is covered by TASK-694.
5. **Guard the `get_cli_setting` bug class** — make a bare section name with a non-string second argument (or no key) fail loudly instead of silently returning the default, and/or add a lint. This class had six known instances across five subsystems. Note in the description that `save_setting_to_cli_config(section, None, value)` has the same shape of defect (raises `KeyError: 'None'`), and that P3 removed its only caller. Priority: medium.

- [ ] **Step 3: Update TASK-658**

Check its ACs off, and record in an `## Implementation Notes` section that AC#4's sweep found four further instances, now filed as the tasks above.

- [ ] **Step 4: Update TASK-547**

Reword its ACs to match what shipped: the `get_cli_setting("tools", {})` call site was **deleted with System A** rather than repaired, and the intent — "enabling a `[tools]` flag actually enables that tool" — is satisfied on the live path via `BuiltinToolProvider` plus the repaired Settings save. Check them off and add `## Implementation Notes`. Mark Done.

- [ ] **Step 5: Close TASK-545**

Check off the P3 criteria, rewording the "System A's fate is decided and implemented" one to record what happened: its execution machinery was deleted, the `Tool` ABC and the two live tools were kept (deleting the file entirely was never possible), and no tool executes ungated in either system because there is now only one system. Add P3 to the `## Implementation Notes`, then:

```bash
backlog task edit 545 -s Done --notes "P3 complete: System A retired, Settings repointed at the agent runtime."
```

If the CLI misbehaves (it has assigned stale IDs and mangled `--ac` before), edit the file directly and verify with `backlog task 545 --plain`.

- [ ] **Step 6: Commit**

```bash
git add backlog/
git commit -m "docs: close TASK-545/547/658 and file the config bug-class follow-ups"
```

---

## Final verification

- [ ] `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Agents/ Tests/MCP/ Tests/Tools/ Tests/UI/ Tests/Local_Ingestion/ Tests/Library/test_library_skills_state.py -q -p no:randomly`
- [ ] `python -c "import tldw_chatbook.app"` succeeds.
- [ ] With no `[tools]` keys set, `BuiltinToolProvider().list_catalog()` yields exactly `calculator` and `get_current_datetime`.
- [ ] No production code references the deleted symbols:
      `grep -rn "get_tool_executor\|reload_tool_executor\|ToolResultCache" tldw_chatbook/` returns nothing.
      Note: two **comments** in `tool_catalog.py` (lines ~197, ~207) mention "ToolExecutor" while describing history. Those are correct and stay -- do not grep for the bare class name and do not delete the comments.
- [ ] Backlog duplicate check, both namespaces:
      `ls backlog/tasks | sed -nE 's/^(task-[0-9]+(\.[0-9]+)*) - .*\.md$/\1/p' | sort | uniq -d`
      `awk 'FNR==1{seen=0} /^id:/ && !seen {seen=1; print tolower($2)}' backlog/tasks/*.md | sort | uniq -d`
