# Tests/MCP/test_mcp_import.py
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from tldw_chatbook.MCP.mcp_import import parse_mcp_servers_json


MCP_PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "MCP"
RUNTIME_IMPORTS = {
    MCP_PACKAGE / "server.py": ["mcp_unified.gateway"],
    MCP_PACKAGE / "gateway_runtime.py": ["mcp_unified.gateway"],
}


def test_parses_command_args_env_and_placeholder_passthrough():
    text = json.dumps(
        {
            "mcpServers": {
                "docs": {
                    "command": "npx",
                    "args": ["-y", "pkg"],
                    "env": {"WORKSPACE": "$HOME"},
                }
            }
        }
    )
    [candidate] = parse_mcp_servers_json(text)
    assert candidate.profile_id == "docs"
    assert candidate.args == ["-y", "pkg"]
    assert candidate.env_placeholders == {"WORKSPACE": "$HOME"}
    assert candidate.env_literals == {} and candidate.warnings == []


def test_secret_shaped_literal_becomes_placeholder_with_warning():
    text = json.dumps(
        {
            "mcpServers": {
                "web": {"command": "npx", "env": {"API_KEY": "sk-live-123456"}}
            }
        }
    )
    [candidate] = parse_mcp_servers_json(text)
    assert candidate.env_placeholders == {"API_KEY": "$API_KEY"}
    assert candidate.env_literals == {}
    assert any("export it before connecting" in w for w in candidate.warnings)


def test_safe_literal_survives_and_overwrite_warning():
    text = json.dumps(
        {"mcpServers": {"docs": {"command": "npx", "env": {"DEBUG": "true"}}}}
    )
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
        "profile_id",
        "command",
        "args",
        "env_placeholders",
        "env_literals",
    }


def _mcp_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    mcp_imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mcp_imports.extend(
                alias.name
                for alias in node.names
                if alias.name in {"mcp", "mcp_unified"}
                or alias.name.startswith(("mcp.", "mcp_unified."))
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"mcp", "mcp_unified"} or module.startswith(
                ("mcp.", "mcp_unified.")
            ):
                mcp_imports.append(module)
    return mcp_imports


def _assert_public_gateway_imports(source: str, expected: list[str]) -> None:
    assert _mcp_imports(source) == expected


def test_standalone_runtime_imports_only_the_public_mcp_unified_gateway() -> None:
    for path, expected in RUNTIME_IMPORTS.items():
        _assert_public_gateway_imports(path.read_text(encoding="utf-8"), expected)


def test_gateway_runtime_public_docstrings_use_google_sections() -> None:
    source = (MCP_PACKAGE / "gateway_runtime.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    runtime = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChatbookGatewayRuntime"
    )
    class_doc = ast.get_docstring(runtime) or ""
    assert "Args:" in class_doc

    public_methods = {
        node.name: node
        for node in runtime.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
        and not node.name.startswith("_")
        and not any(
            isinstance(decorator, ast.Name) and decorator.id == "overload"
            for decorator in node.decorator_list
        )
    }
    assert set(public_methods) == {
        "call_tool",
        "finalize",
        "get_prompt",
        "list_prompts",
        "list_resource_templates",
        "list_resources",
        "list_tools",
        "prompt",
        "read_resource",
        "register_local_tools",
        "resource",
        "tool",
    }
    for method_name, method in public_methods.items():
        doc = ast.get_docstring(method) or ""
        if method_name != "finalize":
            assert "Args:" in doc, method_name
        if method_name not in {"finalize", "register_local_tools"}:
            assert "Returns:" in doc, method_name


def test_public_gateway_import_contract_accepts_exact_public_module() -> None:
    _assert_public_gateway_imports(
        "from mcp_unified.gateway import GatewayLimits, serve_stdio",
        ["mcp_unified.gateway"],
    )


@pytest.mark.parametrize(
    "mutated_import",
    [
        "from mcp.server import stdio",
        "import mcp.types",
        "import mcp_unified",
        "from mcp_unified.gateway.protocol import Connection",
        "import mcp_unified.gateway._private",
    ],
)
def test_public_gateway_import_contract_kills_official_and_private_mutations(
    mutated_import: str,
) -> None:
    with pytest.raises(AssertionError):
        _assert_public_gateway_imports(mutated_import, [])
