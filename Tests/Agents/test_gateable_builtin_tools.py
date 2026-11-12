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


#: What every gateable blurb says about the approval card (task-32284's
#: sentence, kept by the TASK-34100.13 rebase onto dev 4b90eb860b).
_ASKS_SENTENCE = "Asks before running unless you approve a longer scope."


def test_every_blurb_says_how_often_the_card_asks():
    """task-32282/32284, then TASK-34100.13: every gateable built-in is
    risk-tagged, so the card asks before it runs, and the card's
    "Until Chatbook exits" is the longer scope that stops it asking. Every
    row says so, the write-class rows included (they used to say nothing),
    so the wizard row, the MCP Tool gates pane and the card agree. The test
    below ties the sentence to the code that makes it true.
    """
    for entry in gateable_builtin_tools():
        assert entry.blurb.endswith(f" {_ASKS_SENTENCE}"), entry.tool_name


class _EmptyPermissionStore:
    def load(self):
        return {}


class _SessionRecordingService:
    """The control-plane surface ``BuiltinToolGate`` reads: an empty store
    (built-in default ``allow``), no kill switch, and session approvals kept
    in a set so a second turn can see the first one's grant."""

    def __init__(self):
        self.permission_store = _EmptyPermissionStore()
        self.session = set()

    def get_kill_switch(self):
        return False

    def is_session_approved(self, server_key, tool_name, *, profile_id="default"):
        return (server_key, tool_name, profile_id) in self.session

    def approve_for_session(self, server_key, tool_name, *, profile_id="default"):
        self.session.add((server_key, tool_name, profile_id))


@pytest.mark.parametrize(
    "entry",
    [e for e in gateable_builtin_tools() if _ASKS_SENTENCE in e.blurb],
    ids=lambda e: e.tool_name,
)
def test_a_blurb_that_says_it_asks_belongs_to_a_tool_the_gate_floors_to_ask(entry):
    """Review TEF-5 (TASK-34100.13), re-tied on the rebase onto dev
    4b90eb860b: "Asks before running unless you approve a longer scope" is
    two claims, and each is checked against the code that enforces it.

    * "Asks before running": the tool's own ``risk_tags`` floor an inherited
      ``allow`` to ``ask`` (resolved through ``tool_ref`` against an empty
      permission store, whose built-in default is ``allow``), and the real
      review hook (``build_tool_review_hook``) raises an approval row for it.
    * "unless you approve a longer scope": that row offers a decision beyond
      this one call (``approve_session``, the card's "Until Chatbook exits"),
      the captured card view keeps it legal, and choosing it stops the next
      turn's hook from asking again.

    A tool that lost its tags, or a card that dropped the session choice,
    would keep a blurb that had become false; this fails for either."""
    from types import SimpleNamespace

    from tldw_chatbook.Agents.agent_models import ToolCall
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate, tool_ref
    from tldw_chatbook.Chat.approval_presentation import capture_approval_view
    from tldw_chatbook.Chat.console_chat_controller import build_tool_review_hook
    from tldw_chatbook.MCP.permission_store import (
        BUILTIN_DEFAULT_STATE,
        BUILTIN_HIGH_RISK_TAGS,
        resolve_builtin_state,
    )
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        DECISION_SCOPE_COPY,
    )

    tool = build_gateable_tool(entry)
    tags = set(tool.risk_tags)
    assert tags, f"{entry.tool_name} declares no risk tags"
    assert tags <= BUILTIN_HIGH_RISK_TAGS, tags - BUILTIN_HIGH_RISK_TAGS
    assert BUILTIN_DEFAULT_STATE == "allow"  # else {} below proves nothing
    effective = resolve_builtin_state({}, tool_ref(tool))
    assert (effective.state, effective.risk_floored) == ("ask", True)

    service = _SessionRecordingService()
    gate = BuiltinToolGate(service)
    rows = []

    def request(pending):
        rows.extend(pending)
        return {"call-1": "approve_session"}

    hook = build_tool_review_hook(
        gate, SimpleNamespace(tool_for=lambda name: tool), None, request
    )
    hook([ToolCall(entry.tool_name, {}, "call-1")], "run-tef5")
    assert [row.tool_name for row in rows] == [entry.tool_name]  # it asked

    view = capture_approval_view(
        rows, round_id="round-1", session_id="s", run_id="run-tef5", revision=1
    )
    legal = view.rows[0].legal_decisions
    assert "approve_once" in legal
    longer = [d for d in legal if d not in ("approve_once", "deny")]
    assert longer == ["approve_session"], legal
    assert "until Chatbook exits" in DECISION_SCOPE_COPY["approve_session"]

    rows.clear()
    hook([ToolCall(entry.tool_name, {}, "call-2")], "run-tef5")
    assert rows == []  # the longer scope stopped the asking
