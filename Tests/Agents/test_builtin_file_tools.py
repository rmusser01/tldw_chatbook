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


# -- A failed registration must be diagnosable, not silent -------------------


def test_a_failed_registration_logs_instead_of_vanishing(tools_config):
    """An enabled gate whose tool cannot be built must say so.

    The loop swallows every exception, so before this a real breakage was
    indistinguishable from "the gate is off": no log, no error, the tool
    simply absent. Not hypothetical -- note_management_tools was
    unimportable on dev for an unknown period (it imported a name that
    exists only inside a string literal in config.py) and nothing
    surfaced it.
    """
    import sys
    from unittest.mock import patch

    from loguru import logger

    tools_config["create_note_enabled"] = True
    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        with patch.dict(
            sys.modules, {"tldw_chatbook.Tools.note_management_tools": None}
        ):
            names = _names(BuiltinToolProvider())
    finally:
        logger.remove(sink)

    assert "create_note" not in names, "a tool that failed to build must not register"
    assert any("CreateNoteTool" in m for m in messages), (
        f"registration failure was not logged; warnings seen: {messages}"
    )


def test_a_disabled_gate_is_not_logged_as_a_failure(tools_config):
    """Gate-off is the normal case and must stay quiet -- otherwise every
    default startup emits five warnings and the signal is worthless."""
    from loguru import logger

    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        names = _names(BuiltinToolProvider())
    finally:
        logger.remove(sink)

    assert names == {"calculator", "get_current_datetime"}
    assert messages == [], f"disabled gates must not warn; got: {messages}"


# -- glob_files/grep_files -----------------------------------------------


def test_glob_and_grep_absent_by_default(tools_config):
    """Default posture is unchanged: both new gates default to disabled."""
    names = _names(BuiltinToolProvider())
    assert "glob_files" not in names
    assert "grep_files" not in names


@pytest.mark.parametrize(
    "gate_key,tool_name",
    [
        ("glob_files_enabled", "glob_files"),
        ("grep_files_enabled", "grep_files"),
    ],
)
def test_glob_or_grep_appears_when_its_gate_is_enabled(tools_config, gate_key, tool_name):
    tools_config[gate_key] = True
    assert tool_name in _names(BuiltinToolProvider())


def test_glob_and_grep_gates_are_independent(tools_config):
    tools_config["glob_files_enabled"] = True
    names = _names(BuiltinToolProvider())
    assert "glob_files" in names
    assert "grep_files" not in names


def test_glob_and_grep_carry_the_reads_risk_tag(tools_config):
    tools_config["glob_files_enabled"] = True
    tools_config["grep_files_enabled"] = True
    provider = BuiltinToolProvider()
    for name in ("glob_files", "grep_files"):
        assert provider.tool_for(name).risk_tags == ("reads",)


def test_glob_and_grep_names_are_covered_by_the_shadow_guard(tools_config):
    """Extends the task-584/545 guard to the two newest gated names."""
    from tldw_chatbook.Library.library_skills_state import _SHADOWED_BUILTIN_NAMES

    tools_config["glob_files_enabled"] = True
    tools_config["grep_files_enabled"] = True
    gated = _names(BuiltinToolProvider())
    assert gated <= _SHADOWED_BUILTIN_NAMES, (
        f"gated builtin tool names not covered: {gated - _SHADOWED_BUILTIN_NAMES}"
    )
