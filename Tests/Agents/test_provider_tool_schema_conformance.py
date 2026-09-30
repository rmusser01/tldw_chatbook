"""Provider tool-schema conformance (TASK-33621.1, review finding G4-01).

OpenAI and Anthropic validate every function schema before the model runs.
Both refuse a combinator at the TOP level of a function's parameters:

* OpenAI: "schema must have type 'object' and not have
  'oneOf'/'anyOf'/'allOf'/'enum'/'const'/'not' at the top level."
* Anthropic: "input_schema does not support oneOf, allOf, or anyOf at the
  top level".

Three built-in local tools (``watchlists_update_collection_sources``,
``watchlists_check_sources``, ``todo_update``) declared exactly that, so every
default Console send to either provider came back HTTP 400 -- whatever the
prompt, whatever the model. Nested combinators inside ``properties`` are
accepted by both providers and are left alone.

The checker below is written independently of the production projection
helpers on purpose: a conformance test that asks the code under test whether
the code under test conforms proves nothing.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_chatbook.Agents.local_tool_provider as local_tool_provider
from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Agents.local_tool_provider import (
    ASK_USER_GATE_KEY,
    CHARACTER_TOOLS_GATE_KEY,
    WEB_DEEP_SEARCH_GATE_KEY,
    LocalToolExposure,
    LocalToolProvider,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Agents.native_tools import schemas_to_openai_tools
from tldw_chatbook.Agents.run_context import use_run_id
from tldw_chatbook.Agents.session_todo_store import SessionTodoStore
from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.Tools.watchlists_command_service import WatchlistsCommandService

#: Building a LocalToolProvider reads [tools] gates through the guarded config
#: loader; under the per-test profile redirect that admission fails closed
#: with RecoveryRequired("raw_source_selection_changed") before any schema is
#: built (the same signature that keeps most of test_local_tool_provider.py
#: red on dev). The repository's bootstrap_profile opt-in keeps the
#: collection-time sandbox profile, which is still isolated from the user's.
pytestmark = pytest.mark.bootstrap_profile

#: The keywords at least one of OpenAI/Anthropic refuses at the top level of a
#: function schema. Spelled out here rather than imported from production.
PROVIDER_FORBIDDEN_TOP_LEVEL = ("anyOf", "oneOf", "allOf", "enum", "const", "not")

ALLOW = EffectiveToolState(state="allow", origin="tool_override")


def provider_schema_violations(parameters: object) -> list[str]:
    """Return every reason a provider would refuse ``parameters`` up front."""
    if not isinstance(parameters, dict):
        return [f"parameters is {type(parameters).__name__}, not an object schema"]
    violations = []
    if parameters.get("type") != "object":
        violations.append(f"top-level type is {parameters.get('type')!r}, not 'object'")
    violations.extend(
        f"top-level {key}" for key in PROVIDER_FORBIDDEN_TOP_LEVEL if key in parameters
    )
    return violations


class _NoopExecutor:
    """Workspace executor stand-in: conformance never dispatches a tool."""

    def execute(self, operation: str, arguments: dict, *, intent: str) -> str:
        raise AssertionError(f"conformance test dispatched {operation}")


def _widest_local_provider(tmp_path: Path, monkeypatch) -> LocalToolProvider:
    """Build the LocalToolProvider with EVERY optional spec family registered.

    The config gates for ``web_deep_search``, ``ask_user`` and the character
    tools are switched on, and a todo store, ask callback, Watchlists command
    service and character service are all supplied, so no spec escapes the
    sweep by being gated off in the test profile.
    """
    enabled_gates = {
        WEB_DEEP_SEARCH_GATE_KEY,
        ASK_USER_GATE_KEY,
        CHARACTER_TOOLS_GATE_KEY,
    }
    real_get_cli_setting = local_tool_provider.get_cli_setting

    def get_cli_setting(section, key, default=None):
        if section == "tools" and key in enabled_gates:
            return True
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr(local_tool_provider, "get_cli_setting", get_cli_setting)
    character_service = SimpleNamespace(
        search=lambda _args: "{}",
        get=lambda _args: "{}",
        save=lambda _args: "{}",
        approval_summary=lambda args: dict(args),
    )
    return LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: ALLOW,
        todo_store=SessionTodoStore(),
        ask_user=lambda _questions: {},
        watchlists_command_service=_rule_service(),
        character_service=character_service,
        workspace_executor=_NoopExecutor(),
    )


def _rule_service(calls: list | None = None) -> WatchlistsCommandService:
    """A local-mode Watchlists command service whose storage records calls."""
    recorded = calls if calls is not None else []

    def record(name):
        def _call(*args, **kwargs):
            recorded.append((name, args, kwargs))
            raise AssertionError(f"{name} reached storage")

        return _call

    return WatchlistsCommandService(
        runtime_source_loader=lambda: "local",
        create_sources_batch=record("create_sources_batch"),
        create_collection=record("create_collection"),
        update_collection_sources=record("update_collection_sources"),
        accept_source_checks=record("accept_source_checks"),
        resolve_collection_sources=record("resolve_collection_sources"),
    )


# ---------------------------------------------------------------------------
# AC#3: every schema a provider can see conforms
# ---------------------------------------------------------------------------


def test_every_local_tool_schema_is_provider_conformant_at_its_source(
    tmp_path, monkeypatch
):
    provider = _widest_local_provider(tmp_path, monkeypatch)
    names = [entry.name for entry in provider.list_catalog()]
    # The sweep must actually include the three tools that broke every send,
    # plus the config-gated families, or a green run proves nothing.
    for expected in (
        "watchlists_update_collection_sources",
        "watchlists_check_sources",
        "todo_update",
        "web_deep_search",
        "ask_user",
    ):
        assert expected in names, f"{expected} missing from the conformance sweep"

    failures = {
        name: violations
        for name in names
        if (
            violations := provider_schema_violations(
                provider.load_schema(name).parameters
            )
        )
    }
    # Every exposure (Console and external MCP serving) reads the same spec
    # objects, but sweep them too so a future per-exposure schema cannot
    # slip past.
    for exposure in LocalToolExposure:
        for spec in provider.specs_for_exposure(exposure):
            violations = provider_schema_violations(spec.parameters)
            if violations:
                failures.setdefault(spec.name, violations)
    assert failures == {}


def test_every_local_tool_is_provider_conformant_on_the_wire(tmp_path, monkeypatch):
    provider = _widest_local_provider(tmp_path, monkeypatch)
    schemas = [provider.load_schema(entry.id) for entry in provider.list_catalog()]

    tools = schemas_to_openai_tools(schemas)

    assert [tool["function"]["name"] for tool in tools] == [s.name for s in schemas]
    failures = {
        tool["function"]["name"]: violations
        for tool in tools
        if (violations := provider_schema_violations(tool["function"]["parameters"]))
    }
    assert failures == {}


def test_every_builtin_tool_schema_is_provider_conformant_at_its_source():
    from tldw_chatbook.Agents.tool_catalog import (
        build_gateable_tool,
        gateable_builtin_tools,
    )
    from tldw_chatbook.Tools.tool_executor import CalculatorTool, DateTimeTool

    tools = [CalculatorTool(), DateTimeTool()]
    tools.extend(build_gateable_tool(entry) for entry in gateable_builtin_tools())
    assert len(tools) == 2 + len(gateable_builtin_tools())

    failures = {
        tool.name: violations
        for tool in tools
        if (violations := provider_schema_violations(tool.parameters))
    }
    assert failures == {}


def test_every_library_tool_is_provider_conformant_on_the_wire():
    # library_search_notes declares a top-level anyOf in the shared Library
    # contract (it is also served verbatim to external MCP clients); the
    # native projection is what keeps it from failing a Console send.
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    provider = LibraryToolProvider(SimpleNamespace(invoke=lambda *_a: {}))
    schemas = [provider.load_schema(entry.id) for entry in provider.list_catalog()]
    assert "library_search_notes" in {schema.name for schema in schemas}

    failures = {
        tool["function"]["name"]: violations
        for tool in schemas_to_openai_tools(schemas)
        if (violations := provider_schema_violations(tool["function"]["parameters"]))
    }
    assert failures == {}


class _MCPHub:
    """The hub seams ``MCPToolProvider.compose_catalog`` reads (real names)."""

    def __init__(self, tools: list[dict]) -> None:
        self._records = [
            {
                "profile_id": "thirdparty",
                "is_connected": True,
                "discovery_snapshot": {"tools": tools},
            }
        ]
        self.local_service = SimpleNamespace(get_inventory=lambda: {"tools": []})

    def get_kill_switch(self) -> bool:
        return False

    async def local_external_catalog(self) -> list[dict]:
        return self._records

    def effective_tool_states(self, tools: list[HubTool]):
        return {
            (tool.server_key, tool.name): EffectiveToolState(
                state="ask", origin="global_default"
            )
            for tool in tools
        }


#: One third-party MCP tool per forbidden top-level keyword, each otherwise a
#: perfectly ordinary object schema with a NESTED combinator that must survive.
_THIRD_PARTY_SCHEMAS = {
    "pick_any": {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
        "additionalProperties": False,
    },
    "pick_one": {
        "type": "object",
        "properties": {
            "a": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "b": {"type": "string"},
        },
        "oneOf": [{"required": ["a"]}, {"required": ["b"]}],
        "required": [],
    },
    "pick_all": {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "allOf": [{"required": ["a"]}],
    },
    "enum_top": {"type": "object", "properties": {}, "enum": [{}]},
    "const_top": {"type": "object", "properties": {}, "const": {}},
    "not_top": {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "not": {"required": ["a"]},
    },
    "untyped": {"properties": {"q": {"type": "string"}}, "required": ["q"]},
}


def test_mcp_bridged_tools_lose_top_level_combinators_at_projection():
    tools = [
        {"name": name, "description": f"{name} tool", "inputSchema": schema}
        for name, schema in _THIRD_PARTY_SCHEMAS.items()
    ]
    originals = copy.deepcopy(_THIRD_PARTY_SCHEMAS)
    loop = asyncio.new_event_loop()
    try:
        provider = MCPToolProvider(service=_MCPHub(tools), main_loop=loop)
        with use_run_id("conformance"):
            asyncio.run(provider.compose_catalog())
        schemas = [provider.load_schema(entry.id) for entry in provider.list_catalog()]
    finally:
        loop.close()
    assert len(schemas) == len(_THIRD_PARTY_SCHEMAS)

    projected = schemas_to_openai_tools(schemas)

    by_suffix = {}
    for tool in projected:
        parameters = tool["function"]["parameters"]
        name = tool["function"]["name"]
        assert provider_schema_violations(parameters) == [], name
        source = next(key for key in _THIRD_PARTY_SCHEMAS if name.endswith(key))
        by_suffix[source] = parameters
    # Everything a provider accepts is kept verbatim: properties (including a
    # NESTED oneOf), required and additionalProperties.
    assert by_suffix["pick_any"]["properties"] == originals["pick_any"]["properties"]
    assert by_suffix["pick_any"]["additionalProperties"] is False
    assert by_suffix["pick_one"]["properties"]["a"] == {
        "oneOf": [{"type": "string"}, {"type": "integer"}]
    }
    assert by_suffix["pick_one"]["required"] == []
    assert by_suffix["untyped"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    # The hub's own record is never edited: permission fingerprints
    # (definition_hash) and the fence-protocol rendering still see the
    # server's real definition.
    assert _THIRD_PARTY_SCHEMAS == originals
    assert [schema.parameters for schema in schemas] == list(originals.values())


def _project_third_party_mcp_tools(schemas: dict[str, dict]) -> dict[str, dict]:
    """Compose the real MCP bridge over ``schemas``; return projected functions."""
    tools = [
        {"name": name, "description": f"{name} tool", "inputSchema": schema}
        for name, schema in schemas.items()
    ]
    loop = asyncio.new_event_loop()
    try:
        provider = MCPToolProvider(service=_MCPHub(tools), main_loop=loop)
        with use_run_id("conformance"):
            asyncio.run(provider.compose_catalog())
        loaded = [provider.load_schema(entry.id) for entry in provider.list_catalog()]
    finally:
        loop.close()
    projected = {}
    for tool in schemas_to_openai_tools(loaded):
        name = tool["function"]["name"]
        projected[next(key for key in schemas if name.endswith(key))] = tool["function"]
    return projected


def test_projection_tells_the_model_the_either_or_rule_it_strips():
    # Review of TASK-33621.1: a stripped "give a or b" rule must not simply
    # vanish -- the model would treat every field as optional and waste a
    # turn on a call the server refuses. A rule that only lists which
    # arguments to give is restated in the description the provider sees.
    projected = _project_third_party_mcp_tools(_THIRD_PARTY_SCHEMAS)

    assert projected["pick_any"]["description"] == (
        "pick_any tool Argument rule: provide at least one of a or b."
    )
    assert projected["pick_one"]["description"] == (
        "pick_one tool Argument rule: provide exactly one of a or b."
    )
    assert projected["pick_all"]["description"] == (
        "pick_all tool Argument rule: provide a."
    )
    # A rule that is not a plain list of required arguments cannot be put
    # into words faithfully, so nothing is invented for it.
    for untouched in ("enum_top", "const_top", "not_top", "untyped"):
        assert projected[untouched]["description"] == f"{untouched} tool", untouched


@pytest.mark.parametrize(
    ("description", "parameters", "expected"),
    [
        (
            "Find a person.",
            {
                "type": "object",
                "properties": {},
                "anyOf": [{"required": ["first", "last"]}, {"required": ["email"]}],
            },
            "Find a person. Argument rule: provide at least one of "
            "(first and last) or email.",
        ),
        (
            "",
            {
                "type": "object",
                "properties": {},
                "oneOf": [
                    {"required": ["a"]},
                    {"required": ["b"]},
                    {"required": ["c"]},
                ],
            },
            "Argument rule: provide exactly one of a, b or c.",
        ),
        (
            "Mixed.",
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "anyOf": [{"required": ["a"]}, {"properties": {"a": {"minLength": 2}}}],
            },
            "Mixed.",
        ),
    ],
    ids=["grouped-alternative", "empty-description", "not-required-only"],
)
def test_projection_rule_sentence_only_restates_required_only_alternatives(
    description, parameters, expected
):
    schema = ToolSchema(
        id="x", name="x", description=description, parameters=parameters
    )

    [tool] = schemas_to_openai_tools([schema])

    assert tool["function"]["description"] == expected
    assert provider_schema_violations(tool["function"]["parameters"]) == []


def test_library_search_notes_keeps_its_selector_rule_on_the_wire():
    from tldw_chatbook.Agents.library_tool_provider import LibraryToolProvider

    provider = LibraryToolProvider(SimpleNamespace(invoke=lambda *_a: {}))
    schema = provider.load_schema("library_search_notes")
    [tool] = schemas_to_openai_tools([schema])

    assert "anyOf" not in tool["function"]["parameters"]
    assert tool["function"]["description"].endswith(
        "Argument rule: provide at least one of query, keyword, folder_id or folder."
    )


def test_projection_passes_conformant_schemas_through_unchanged():
    conformant = {
        "type": "object",
        "properties": {"mode": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
        "required": ["mode"],
        "additionalProperties": False,
    }
    schema = ToolSchema(id="x", name="x", description="x", parameters=conformant)

    [tool] = schemas_to_openai_tools([schema])

    assert tool["function"]["parameters"] == conformant
    assert json.dumps(tool, sort_keys=True) == json.dumps(
        {
            "type": "function",
            "function": {"name": "x", "description": "x", "parameters": conformant},
        },
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# AC#2: the either/or rules still hold, now in the handlers + descriptions
# ---------------------------------------------------------------------------


def _local_provider(tmp_path: Path, calls: list) -> LocalToolProvider:
    return LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: ALLOW,
        todo_store=SessionTodoStore(),
        watchlists_command_service=_rule_service(calls),
        workspace_executor=_NoopExecutor(),
    )


@pytest.mark.parametrize(
    ("tool", "phrases"),
    [
        (
            "watchlists_update_collection_sources",
            ("at least one of", "add_source_ids", "remove_source_ids"),
        ),
        ("watchlists_check_sources", ("exactly one of", "source_ids", "collection_id")),
        (
            "todo_update",
            ("at least one of", "content", "status", "activeForm", '"deleted"'),
        ),
    ],
)
def test_each_either_or_tool_states_its_rule_in_its_description(
    tmp_path, tool, phrases
):
    description = _local_provider(tmp_path, []).load_schema(tool).description

    for phrase in phrases:
        assert phrase in description, (tool, phrase, description)


@pytest.mark.parametrize(
    ("args", "rule"),
    [
        (
            {"collection_id": "local:watchlist:1"},
            "at least one of add_source_ids or remove_source_ids",
        ),
        (
            {
                "collection_id": "local:watchlist:1",
                "add_source_ids": [],
                "remove_source_ids": [],
            },
            "at least one of add_source_ids or remove_source_ids",
        ),
    ],
    ids=["neither-list", "both-lists-empty"],
)
def test_update_collection_sources_refuses_a_call_with_no_membership_change(
    tmp_path, args, rule
):
    calls: list = []
    with use_run_id("rules"):
        result = _local_provider(tmp_path, calls).invoke(
            "local:watchlists_update_collection_sources", args
        )

    payload = json.loads(result.content)
    assert payload["status"] == "invalid_argument"
    assert rule in payload["message"]
    assert calls == []


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"source_ids": ["local:subscription:1"], "collection_id": "local:watchlist:1"},
    ],
    ids=["neither", "both"],
)
def test_check_sources_refuses_anything_but_exactly_one_scope(tmp_path, args):
    calls: list = []
    with use_run_id("rules"):
        result = _local_provider(tmp_path, calls).invoke(
            "local:watchlists_check_sources", args
        )

    payload = json.loads(result.content)
    assert payload["status"] == "invalid_argument"
    assert "exactly one of source_ids or collection_id" in payload["message"]
    assert calls == []


def test_check_sources_does_not_blame_the_scope_rule_for_an_unknown_argument(
    tmp_path,
):
    calls: list = []
    with use_run_id("rules"):
        result = _local_provider(tmp_path, calls).invoke(
            "local:watchlists_check_sources",
            {"source_ids": ["local:subscription:1"], "force": True},
        )

    payload = json.loads(result.content)
    assert payload["status"] == "invalid_argument"
    assert "Rule:" not in payload["message"]
    assert calls == []


@pytest.mark.parametrize(
    ("args", "rule"),
    [
        (
            {"id": "1", "expected_version": 1},
            "at least one of content, status, or activeForm",
        ),
        (
            {"id": "1", "expected_version": 1, "status": "deleted", "content": "x"},
            'status "deleted" must be the only change',
        ),
        (
            {"id": "1", "expected_version": 1, "status": "deleted", "activeForm": None},
            'status "deleted" must be the only change',
        ),
    ],
    ids=["no-change", "delete-plus-content", "delete-plus-active-form"],
)
def test_todo_update_refuses_rule_violations_naming_the_rule(tmp_path, args, rule):
    store = SessionTodoStore()
    store.create(content="keep")
    before = store.export_snapshot()
    provider = LocalToolProvider(
        workspace_root=tmp_path,
        resolve_state=lambda _hub: ALLOW,
        todo_store=store,
        workspace_executor=_NoopExecutor(),
    )

    with use_run_id("rules"):
        result = provider.invoke("local:todo_update", args)

    assert not result.ok
    assert rule in result.error
    assert store.export_snapshot() == before


# ---------------------------------------------------------------------------
# AC#4 support: naming the rejected tool from the provider's own message
# ---------------------------------------------------------------------------

_SENT = schemas_to_openai_tools(
    [
        ToolSchema(id=f"t:{name}", name=name, description=name, parameters={})
        for name in ("calculator", "todo_update", "watchlists_check_sources")
    ]
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (  # OpenAI names the function (and its position) in the body.
            'Bad request to openai (Status 400). Detail: {"error": {"message": '
            "\"Invalid schema for function 'watchlists_check_sources': schema must "
            'have type \'object\'", "param": "tools[2].function.parameters"}}',
            "watchlists_check_sources",
        ),
        (  # Anthropic names only the position.
            'Bad request (400). Detail: {"type":"error","error":{"message":'
            '"tools.1.custom.input_schema: input_schema does not support oneOf"}}',
            "todo_update",
        ),
        (  # Gemini: the declaration index wins over the outer tools[0].
            'Invalid JSON payload received. Unknown name "not" at '
            "'tools[0].function_declarations[2].parameters'",
            "watchlists_check_sources",
        ),
        ("tools.7.custom.input_schema: out of range", None),
        ("Invalid schema for function 'not_sent': bad", None),
        ("model: claude-3-opus is not available", None),
    ],
    ids=[
        "openai",
        "anthropic",
        "gemini",
        "index-out-of-range",
        "unsent-name",
        "not-a-tool",
    ],
)
def test_rejected_tool_name_reports_only_a_tool_the_request_sent(message, expected):
    from tldw_chatbook.Agents.native_tools import rejected_tool_name

    assert rejected_tool_name(message, _SENT) == expected


def test_rejected_tool_name_refuses_positions_when_a_sent_entry_is_unnamed():
    from tldw_chatbook.Agents.native_tools import rejected_tool_name

    unnamed = [*_SENT, {"type": "function", "function": {"name": ""}}]

    assert rejected_tool_name("tools.1.custom.input_schema: bad", unnamed) is None
    assert rejected_tool_name("tools.1.custom.input_schema: bad", None) is None


def test_rejected_tool_name_still_matches_a_sent_name_beside_an_unnamed_entry():
    from tldw_chatbook.Agents.native_tools import rejected_tool_name

    unnamed = [{"type": "function", "function": {"name": ""}}, *_SENT]

    # A name is matched against what was sent, so it stays trustworthy...
    assert (
        rejected_tool_name("Invalid schema for function 'todo_update': bad", unnamed)
        == "todo_update"
    )
    # ...a position is not: the adapter may have dropped the unnamed entry.
    assert rejected_tool_name("tools.1.custom.input_schema: bad", unnamed) is None


def test_rejected_tool_name_treats_a_blank_name_as_unnamed():
    from tldw_chatbook.Agents.native_tools import rejected_tool_name

    # Anthropic's adapter drops a name that is empty after strip(), which
    # shifts every later tools.N by one.
    blank = [{"type": "function", "function": {"name": "   "}}, *_SENT]

    assert rejected_tool_name("tools.1.custom.input_schema: bad", blank) is None


_SENT_WITH_MCP = schemas_to_openai_tools(
    [
        ToolSchema(id=f"t:{name}", name=name, description=name, parameters={})
        for name in ("todo_update", "mcp__thirdparty__lookup")
    ]
)


@pytest.mark.parametrize(
    ("name", "advice", "absent"),
    [
        (
            "mcp__thirdparty__lookup",
            "Turn off the MCP server that provides it on the MCP screen",
            "one of Chatbook's own tools",
        ),
        (
            "todo_update",
            "one of Chatbook's own tools, so please report it",
            "Turn off the MCP server",
        ),
    ],
    ids=["mcp-tool", "chatbook-tool"],
)
def test_tool_rejection_copy_gives_advice_that_fits_the_tool_source(
    name, advice, absent
):
    from tldw_chatbook.Chat.console_provider_gateway import (
        _provider_error_copy_with_model_recovery,
    )

    copy = _provider_error_copy_with_model_recovery(
        "Provider error from openai: bad request. Status: 400.",
        model="gpt-4.1-mini",
        status_code=400,
        provider_message=f"Invalid schema for function '{name}': bad",
        tools=_SENT_WITH_MCP,
    )

    assert f"the tool definition for {name} before the model ran" in copy
    assert "choosing another model will not help" in copy
    assert advice in copy, copy
    assert absent not in copy, copy
    assert "Confirm the model is still available" not in copy
    assert "model picker" not in copy


def test_a_non_tool_bad_request_keeps_the_model_recovery_copy():
    from tldw_chatbook.Chat.console_provider_gateway import (
        _provider_error_copy_with_model_recovery,
    )

    base = "Provider error from anthropic: bad request. Status: 400."
    model_copy = _provider_error_copy_with_model_recovery(
        base,
        model="claude-3-opus",
        status_code=400,
        provider_message="Bad request (400). Detail: model: claude-3-opus not found",
        tools=_SENT,
    )
    unnamed_tool_copy = _provider_error_copy_with_model_recovery(
        base,
        model="claude-3-opus",
        status_code=400,
        provider_message="Bad request (400). Detail: tools.9.custom.input_schema: bad",
        tools=_SENT,
    )

    assert "Confirm the model is still available" in model_copy
    assert "one of the tool definitions sent with this request" in unnamed_tool_copy
    # Unnamed, the blame rests only on a marker in the provider's text, so the
    # copy does not state as fact that no other model could help.
    assert "choosing another model is unlikely to help" in unnamed_tool_copy
    assert "will not help" not in unnamed_tool_copy
    assert "Confirm the model is still available" not in unnamed_tool_copy
    assert "model picker" not in unnamed_tool_copy
