"""task-584: the file tools must be reachable from the Agents runtime.

`BuiltinToolProvider` hardcoded Calculator + DateTime, so the app's existing
sandbox-rooted ReadFileTool/ListDirectoryTool were registered on the global
ToolExecutor but never surfaced to the agent loop. Retained script output is
written under the file-tool sandbox root precisely so those tools can reach it.

They stay behind the SAME config gates that already govern them, which default
to disabled — wiring them in changes reachability, not the default posture.
"""

import pytest

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider


@pytest.fixture
def tools_config(monkeypatch):
    """Drive the existing [tools] gates."""
    values = {}
    import tldw_chatbook.config as config_module

    def fake(section, key=None, default=None):
        if section != "tools" or not isinstance(key, str):
            return default
        return values.get(key, default)

    monkeypatch.setattr(config_module, "get_cli_setting", fake)
    return values


def _names(provider):
    return {entry.name for entry in provider.list_catalog()}


def test_file_tools_absent_by_default(tools_config):
    """Default posture is unchanged: the gates default to disabled."""
    names = _names(BuiltinToolProvider())
    assert "read_file" not in names
    assert "list_directory" not in names
    assert {"calculator", "get_current_datetime"} <= names


def test_read_file_appears_when_enabled(tools_config):
    tools_config["read_file_enabled"] = True
    assert "read_file" in _names(BuiltinToolProvider())


def test_list_directory_appears_when_enabled(tools_config):
    tools_config["list_directory_enabled"] = True
    assert "list_directory" in _names(BuiltinToolProvider())


def test_each_gate_is_independent(tools_config):
    tools_config["read_file_enabled"] = True
    names = _names(BuiltinToolProvider())
    assert "read_file" in names
    assert "list_directory" not in names


def test_gated_tool_names_are_covered_by_the_shadow_guard(tools_config):
    """The drift guard has a blind spot for config-gated tools.

    It constructs a BuiltinToolProvider with DEFAULT config, so a tool that
    only appears when a gate is enabled is invisible to it. These names must
    therefore be pinned explicitly: a skill named `read_file` shadows a real
    builtin the moment someone turns the gate on.
    """
    from tldw_chatbook.Library.library_skills_state import _SHADOWED_BUILTIN_NAMES

    tools_config["read_file_enabled"] = True
    tools_config["list_directory_enabled"] = True
    gated = _names(BuiltinToolProvider())
    assert gated <= _SHADOWED_BUILTIN_NAMES, (
        f"gated builtin tool names not covered: {gated - _SHADOWED_BUILTIN_NAMES}"
    )


@pytest.mark.asyncio
async def test_list_directory_does_not_follow_symlinks_out_of_the_sandbox(
    tmp_path, monkeypatch
):
    """Containment: a planted symlink must not enumerate outside the root.

    Surfacing this tool to the agent loop makes the escape reachable by an
    agent, so it is fixed here rather than left to the tool's own PR: a
    symlink created inside file_sandbox_root previously let the recursive
    walk list files anywhere on disk.
    """
    import tldw_chatbook.Tools.file_operation_tools as fot

    sandbox = tmp_path / "sandbox"
    outside = tmp_path / "outside"
    sandbox.mkdir()
    outside.mkdir()
    (outside / "OUTSIDE_SECRET.txt").write_text("x", encoding="utf-8")
    (sandbox / "escape").symlink_to(outside)
    (sandbox / "legit").mkdir()
    (sandbox / "legit" / "inside.txt").write_text("y", encoding="utf-8")

    monkeypatch.setattr(fot, "_resolve_sandbox_config", lambda: str(sandbox))
    listing = str(await fot.ListDirectoryTool().execute(directory_path=".", recursive=True))

    assert "OUTSIDE_SECRET" not in listing, "symlink escaped the sandbox root"
    assert "inside.txt" in listing, "legitimate nested listing must still work"
