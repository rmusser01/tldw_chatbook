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
    bogus = GateableTool(
        "x_enabled", "no_such_module", "NoSuchTool", "x", "Bogus", "Not real."
    )
    with pytest.raises(Exception):
        build_gateable_tool(bogus)


def test_every_gateable_tool_carries_display_copy():
    """task-32284: the table is THE copy source for every gate surface.

    The first-run wizard's Tools step and the MCP hub's Tool gates pane both
    render straight off these rows, so a row without copy ships a blank
    switch -- the failure mode CLAUDE.md's "New Tool" checklist warns about,
    which used to depend on remembering a second table in the wizard.
    """
    for entry in gateable_builtin_tools():
        assert entry.title.strip(), f"{entry.tool_name} has no title"
        assert entry.blurb.strip(), f"{entry.tool_name} has no blurb"
        assert entry.title != entry.tool_name, (
            f"{entry.tool_name}'s title is the raw id, not a human name"
        )


def test_read_class_blurbs_promise_a_prompt_before_every_run():
    """task-32282: read-class tools are risk-floored to `ask`, so their copy
    must say so -- byte-identical to the first-run wizard's sentence, since
    both surfaces render the same row.

    The three write-class rows (marked ``⚠``) do NOT carry it: they say what
    they change on disk, and their approval story is the card itself.

    task-32284 (Qodo #2594 #6): the sentence used to read "Asks you each
    time before running", which the approval card itself contradicts -- its
    "Approve for session" and "Always allow" buttons are exactly the longer
    scopes that stop the asking. The copy now names that escape hatch.
    """
    sentence = "Asks before running unless you approve a longer scope."
    per_call = {
        "read_file",
        "list_directory",
        "glob_files",
        "grep_files",
        "expand_document",
    }
    for entry in gateable_builtin_tools():
        if entry.tool_name in per_call:
            assert entry.blurb.endswith(f" {sentence}"), (
                f"{entry.tool_name} does not promise per-call approval"
            )
        else:
            assert sentence not in entry.blurb, (
                f"{entry.tool_name} is not read-class but claims per-call approval"
            )
    assert per_call <= {e.tool_name for e in gateable_builtin_tools()}
