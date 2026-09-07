"""Public dependency contract for the standalone MCP runtime."""

from __future__ import annotations

import inspect
from importlib.metadata import requires, version
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib

from packaging.requirements import Requirement
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_DEPENDENCY = "mcp-unified==0.2.1"
try:
    MCP_UNIFIED_AVAILABLE = find_spec("mcp_unified") is not None
except (ImportError, ValueError):
    MCP_UNIFIED_AVAILABLE = False


@pytest.mark.skipif(not MCP_UNIFIED_AVAILABLE, reason="mcp-unified extra not installed")
def test_mcp_unified_public_contract_is_exact_release() -> None:
    from mcp_unified.gateway import (
        GatewayApplicationError,
        GatewayCoreRuntime,
        GatewayLimits,
        GatewayRequestContext,
        GatewayResourceTemplateRuntime,
        GatewayToolExecutionError,
        serve_stdio,
    )

    assert version("mcp-unified") == "0.2.1"
    assert GatewayCoreRuntime
    assert GatewayResourceTemplateRuntime
    assert GatewayRequestContext
    assert GatewayApplicationError
    assert GatewayToolExecutionError
    assert GatewayLimits
    assert callable(serve_stdio)

    for method_name in (
        "list_tools",
        "call_tool",
        "list_resources",
        "read_resource",
        "list_prompts",
        "get_prompt",
    ):
        assert callable(getattr(GatewayCoreRuntime, method_name))
    assert callable(GatewayResourceTemplateRuntime.list_resource_templates)


@pytest.mark.skipif(not MCP_UNIFIED_AVAILABLE, reason="mcp-unified extra not installed")
def test_exact_runtime_release_supplies_gateway_jsonschema_dependency() -> None:
    dependencies = [Requirement(value) for value in requires("mcp-unified") or []]
    jsonschema_dependencies = [
        dependency
        for dependency in dependencies
        if dependency.name.lower() == "jsonschema" and dependency.marker is None
    ]

    assert len(jsonschema_dependencies) == 1
    assert str(jsonschema_dependencies[0].specifier) == "<5,>=4.23"


@pytest.mark.skipif(not MCP_UNIFIED_AVAILABLE, reason="mcp-unified extra not installed")
def test_mcp_unified_public_signatures_match_the_released_runtime() -> None:
    from mcp_unified.gateway import (
        GatewayApplicationError,
        GatewayLimits,
        GatewayToolExecutionError,
        serve_stdio,
    )

    serve_parameters = list(inspect.signature(serve_stdio).parameters.values())
    assert [(parameter.name, parameter.kind) for parameter in serve_parameters] == [
        ("runtime", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("input_stream", inspect.Parameter.KEYWORD_ONLY),
        ("output_stream", inspect.Parameter.KEYWORD_ONLY),
        ("limits", inspect.Parameter.KEYWORD_ONLY),
        ("metadata", inspect.Parameter.KEYWORD_ONLY),
    ]
    assert serve_parameters[0].default is inspect.Parameter.empty
    assert serve_parameters[1].default is None
    assert serve_parameters[2].default is None
    assert serve_parameters[3].default == GatewayLimits()
    assert serve_parameters[4].default is None

    application_error_parameters = list(
        inspect.signature(GatewayApplicationError).parameters.values()
    )
    assert [
        (parameter.name, parameter.kind) for parameter in application_error_parameters
    ] == [
        ("public_message", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("reason_code", inspect.Parameter.KEYWORD_ONLY),
        ("kind", inspect.Parameter.KEYWORD_ONLY),
    ]
    assert application_error_parameters[0].default is inspect.Parameter.empty
    assert application_error_parameters[1].default is inspect.Parameter.empty
    assert application_error_parameters[2].default == "application"

    tool_error_parameters = list(
        inspect.signature(GatewayToolExecutionError).parameters.values()
    )
    assert [
        (parameter.name, parameter.kind) for parameter in tool_error_parameters
    ] == [
        ("public_message", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ("reason_code", inspect.Parameter.KEYWORD_ONLY),
    ]
    assert tool_error_parameters[0].default is inspect.Parameter.empty
    assert tool_error_parameters[1].default is inspect.Parameter.empty


def test_mcp_optional_extras_pin_the_exact_runtime_release() -> None:
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    optional_dependencies = pyproject["project"]["optional-dependencies"]

    assert optional_dependencies["mcp"] == [MCP_DEPENDENCY]
    assert [
        dependency
        for dependency in optional_dependencies["all-tools"]
        if dependency.lower().startswith("mcp")
    ] == [MCP_DEPENDENCY]


def test_missing_mcp_unified_returns_false_without_importing_server() -> None:
    probe = textwrap.dedent(
        """
        import sys

        attempted = []

        class BlockMcpUnified:
            def find_spec(self, fullname, path=None, target=None):
                attempted.append(fullname)
                if fullname == "mcp_unified" or fullname.startswith("mcp_unified."):
                    raise ModuleNotFoundError(
                        "simulated missing mcp_unified", name=fullname
                    )
                return None

        sys.meta_path.insert(0, BlockMcpUnified())
        from tldw_chatbook import MCP

        assert MCP.is_mcp_available() is False
        assert "mcp_unified" in attempted
        assert "tldw_chatbook.MCP.server" not in sys.modules
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
