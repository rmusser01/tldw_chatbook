"""task-584: the file tools must be reachable from the Agents runtime.

`BuiltinToolProvider` hardcoded Calculator + DateTime, so the app's existing
sandbox-rooted ReadFileTool/ListDirectoryTool were registered on the global
ToolExecutor but never surfaced to the agent loop. Retained script output is
written under the file-tool sandbox root precisely so those tools can reach it.

They stay behind the SAME config gates that already govern them, which default
to disabled — wiring them in changes reachability, not the default posture.

task-545 P2 later replaced the direct `[tools]` gating in
`BuiltinToolProvider.__init__` with pack resolution (see
`Agents/builtin_pack_config.py` and `Agents/builtin_packs/`). The two
`[tools]` flags tested here still work as a deprecated fallback, but the
owner ruled that they must grant exactly the tool each flag names and
nothing more -- see `test_read_file_flag_grants_only_read_file` and
`test_list_directory_flag_grants_only_list_directory` below. (An earlier
version of this restructuring let either flag enable the whole `files`
pack, which `test_either_legacy_flag_enables_the_whole_files_pack` used to
assert; that behaviour has been reversed and the test rewritten.)
"""

import pytest

from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider


@pytest.fixture
def tools_config(monkeypatch):
    """Drive the legacy [tools] gates through the pack-config surface.

    task-545 P2 moved enablement from these per-tool `[tools]` flags to
    `builtin_pack_config.enabled_packs()`, which still honours them as a
    deprecated fallback (see `builtin_pack_config._LEGACY_FILE_FLAGS`).
    Patching `builtin_pack_config.get_cli_setting` -- not
    `tldw_chatbook.config.get_cli_setting` -- is required: that module
    binds its own `get_cli_setting` name via `from ... import`, so a patch
    on the origin module would not be seen there.
    """
    values = {}
    import tldw_chatbook.Agents.builtin_pack_config as pack_config_module

    def fake(section, key=None, default=None):
        if section != "tools" or not isinstance(key, str):
            return default
        return values.get(key, default)

    monkeypatch.setattr(pack_config_module, "get_cli_setting", fake)
    return values


def _names(provider):
    return {entry.name for entry in provider.list_catalog()}


def test_file_tools_absent_by_default(tools_config):
    """Default posture is unchanged: the gates default to disabled."""
    names = _names(BuiltinToolProvider())
    assert "read_file" not in names
    assert "list_directory" not in names
    assert "glob_files" not in names
    assert "grep_files" not in names
    assert names == {"calculator", "get_current_datetime"}


def test_read_file_appears_when_enabled(tools_config):
    tools_config["read_file_enabled"] = True
    assert "read_file" in _names(BuiltinToolProvider())


def test_list_directory_appears_when_enabled(tools_config):
    tools_config["list_directory_enabled"] = True
    assert "list_directory" in _names(BuiltinToolProvider())


def test_read_file_flag_grants_only_read_file(tools_config):
    """Owner ruling: legacy flags grant exactly the tool they name.

    `read_file` and `list_directory` are now two of FOUR tools the `files`
    pack contributes (`glob_files`/`grep_files` joined in task-545's later
    work). Setting only `read_file_enabled` must not silently also grant
    the other three just because they all happen to live in the same
    pack -- that would hand an existing user tools they never asked for
    during what is framed as a compatibility migration. This replaces the
    old `test_either_legacy_flag_enables_the_whole_files_pack`, which
    asserted the opposite (now-reversed) behaviour.
    """
    tools_config["read_file_enabled"] = True
    names = _names(BuiltinToolProvider())
    assert "read_file" in names
    assert "list_directory" not in names
    assert "glob_files" not in names
    assert "grep_files" not in names


def test_list_directory_flag_grants_only_list_directory(tools_config):
    """Mirror of `test_read_file_flag_grants_only_read_file` for the other flag."""
    tools_config["list_directory_enabled"] = True
    names = _names(BuiltinToolProvider())
    assert "list_directory" in names
    assert "read_file" not in names
    assert "glob_files" not in names
    assert "grep_files" not in names


def test_both_legacy_flags_grant_both_named_tools_and_nothing_else(tools_config):
    """Both legacy flags set: exactly `read_file` + `list_directory`, still
    no `glob_files`/`grep_files` -- the legacy flags never had names for
    those two tools, so no combination of the old flags can grant them."""
    tools_config["read_file_enabled"] = True
    tools_config["list_directory_enabled"] = True
    names = _names(BuiltinToolProvider())
    assert "read_file" in names
    assert "list_directory" in names
    assert "glob_files" not in names
    assert "grep_files" not in names


def test_modern_pack_list_grants_all_four_files_tools(monkeypatch):
    """`[agent_tools] enabled_packs = ["files"]` stays unrestricted.

    Unlike the legacy per-tool `[tools]` fallback above, the modern
    pack-level config has no tool-level restriction concept at all -- it
    grants every tool the named pack contributes. `PackResolution.only_tools`
    must be `None` on this path (see `builtin_pack_config.resolve_enabled_packs`).
    """
    import tldw_chatbook.Agents.builtin_pack_config as pack_config_module

    def fake(section, key=None, default=None):
        if section == "agent_tools" and key == "enabled_packs":
            return ["files"]
        return default

    monkeypatch.setattr(pack_config_module, "get_cli_setting", fake)
    names = _names(BuiltinToolProvider())
    assert {"read_file", "list_directory", "glob_files", "grep_files"} <= names


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
