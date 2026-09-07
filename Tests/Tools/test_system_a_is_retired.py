"""TASK-545 P3: System A is gone; its live parts remain.

ToolExecutor had no execution path -- zero production callers of
execute_tool_calls, and its only referrers listed tools for a Settings
screen. What stayed is the half of the module that IS load-bearing.
"""

import pytest


def test_the_dead_symbols_are_gone():
    import tldw_chatbook.Tools.tool_executor as te

    for name in (
        "ToolExecutor",
        "ToolResultCache",
        "get_tool_executor",
        "reload_tool_executor",
    ):
        assert not hasattr(te, name), f"{name} should have been deleted"


def test_the_package_no_longer_exports_them():
    import tldw_chatbook.Tools as tools

    for name in (
        "ToolExecutor",
        "get_tool_executor",
        "reload_tool_executor",
    ):
        assert name not in tools.__all__
        with pytest.raises(AttributeError):
            getattr(tools, name)


def test_the_load_bearing_half_survives():
    """System B imports these; the gate imports Tool."""
    from tldw_chatbook.Tools.tool_executor import (
        CalculatorTool,
        DateTimeTool,
        Tool,
    )

    assert CalculatorTool().name == "calculator"
    assert DateTimeTool().name == "get_current_datetime"
    assert Tool.risk_tags is not None


def test_system_b_and_the_gate_still_import():
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate  # noqa: F401
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
    from tldw_chatbook.Tools.code_audit_tool import CodeAuditTool  # noqa: F401

    assert {e.name for e in BuiltinToolProvider().list_catalog()} == {
        "calculator",
        "get_current_datetime",
    }


def test_no_production_code_references_the_deleted_symbols():
    import pathlib

    offenders = []
    for path in pathlib.Path("tldw_chatbook").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in ("get_tool_executor", "reload_tool_executor", "ToolResultCache"):
            if name in text:
                offenders.append(f"{path}: {name}")
    assert not offenders, offenders


def test_opening_settings_no_longer_patches_write_file():
    """install_claude_code_hooks had exactly one caller: the deleted
    registration. WriteFileTool.execute must stay unpatched."""
    import pathlib

    src = pathlib.Path("tldw_chatbook/Tools/tool_executor.py").read_text(
        encoding="utf-8"
    )
    assert "install_claude_code_hooks" not in src
