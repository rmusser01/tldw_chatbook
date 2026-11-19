"""Container validation at the untrusted MCP JSON import boundary."""

import json

import pytest

from tldw_chatbook.MCP.mcp_import import parse_mcp_servers_json


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("args", 42, id="args-number"),
        pytest.param("args", 0, id="args-zero"),
        pytest.param("args", True, id="args-true"),
        pytest.param("args", False, id="args-false"),
        pytest.param("args", "-y", id="args-string"),
        pytest.param("args", "", id="args-empty-string"),
        pytest.param("args", {"bad": "value"}, id="args-object"),
        pytest.param("args", {}, id="args-empty-object"),
        pytest.param("env", ["bad"], id="env-list"),
        pytest.param("env", [], id="env-empty-list"),
        pytest.param("env", 42, id="env-number"),
        pytest.param("env", 0, id="env-zero"),
        pytest.param("env", True, id="env-true"),
        pytest.param("env", False, id="env-false"),
        pytest.param("env", "bad", id="env-string"),
        pytest.param("env", "", id="env-empty-string"),
    ],
)
def test_rejects_malformed_field_containers(field, value):
    text = json.dumps({"mcpServers": {"docs": {"command": "npx", field: value}}})
    with pytest.raises(ValueError, match=rf'Entry "docs".*{field}'):
        parse_mcp_servers_json(text)


@pytest.mark.parametrize(
    "fields,args,placeholders,literals",
    [
        pytest.param({}, [], {}, {}, id="omitted"),
        pytest.param({"args": None, "env": None}, [], {}, {}, id="null"),
        pytest.param({"args": [], "env": {}}, [], {}, {}, id="empty"),
        pytest.param(
            {
                "args": ["-y", 42, False, None],
                "env": {"WORKSPACE": " $HOME ", "DEBUG": True, "PORT": 42},
            },
            ["-y", "42", "False", "None"],
            {"WORKSPACE": "$HOME"},
            {"DEBUG": "True", "PORT": "42"},
            id="coercion",
        ),
    ],
)
def test_preserves_supported_fields_and_item_coercion(
    fields, args, placeholders, literals
):
    text = json.dumps({"mcpServers": {"docs": {"command": "npx", **fields}}})
    [candidate] = parse_mcp_servers_json(text)
    assert candidate.args == args
    assert candidate.env_placeholders == placeholders
    assert candidate.env_literals == literals
