# Tests/MCP/test_mcp_import.py
from __future__ import annotations

import json

import pytest

from tldw_chatbook.MCP.mcp_import import ImportCandidate, parse_mcp_servers_json


def test_parses_command_args_env_and_placeholder_passthrough():
    text = json.dumps({"mcpServers": {"docs": {
        "command": "npx", "args": ["-y", "pkg"], "env": {"WORKSPACE": "$HOME"}}}})
    [candidate] = parse_mcp_servers_json(text)
    assert candidate.profile_id == "docs"
    assert candidate.args == ["-y", "pkg"]
    assert candidate.env_placeholders == {"WORKSPACE": "$HOME"}
    assert candidate.env_literals == {} and candidate.warnings == []


def test_secret_shaped_literal_becomes_placeholder_with_warning():
    text = json.dumps({"mcpServers": {"web": {
        "command": "npx", "env": {"API_KEY": "sk-live-123456"}}}})
    [candidate] = parse_mcp_servers_json(text)
    assert candidate.env_placeholders == {"API_KEY": "$API_KEY"}
    assert candidate.env_literals == {}
    assert any("export it before connecting" in w for w in candidate.warnings)


def test_safe_literal_survives_and_overwrite_warning():
    text = json.dumps({"mcpServers": {"docs": {"command": "npx", "env": {"DEBUG": "true"}}}})
    [candidate] = parse_mcp_servers_json(text, existing_ids={"docs"})
    assert candidate.env_literals == {"DEBUG": "true"}
    assert any("overwrite" in w for w in candidate.warnings)


def test_invalid_json_and_missing_key_raise():
    with pytest.raises(ValueError, match="Not valid JSON"):
        parse_mcp_servers_json("{nope")
    with pytest.raises(ValueError, match="mcpServers"):
        parse_mcp_servers_json(json.dumps({"servers": {}}))


def test_to_payload_uses_exact_store_keys():
    text = json.dumps({"mcpServers": {"docs": {"command": "npx"}}})
    [candidate] = parse_mcp_servers_json(text)
    assert set(candidate.to_payload()) == {
        "profile_id", "command", "args", "env_placeholders", "env_literals"}
