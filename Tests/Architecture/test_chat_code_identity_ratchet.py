"""Chat must not grow code-identity checks or reach UI-private names (TASK-36944.9)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHAT = ROOT / "tldw_chatbook" / "Chat"
PREPARATION = CHAT / "console_configuration_preparation.py"

# Measured over tldw_chatbook/Chat/**/*.py (Attribute `__code__` + getattr(.., "__code__")):
# 60 at base 66155632ed, 56 at head (console_configuration_preparation.py 10 -> 6).
CODE_REFERENCE_PIN = 56


def _is_code_reference(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr == "__code__"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value == "__code__"
    )


def test_chat_code_identity_references_do_not_rise():
    count = sum(
        _is_code_reference(node)
        for path in CHAT.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    )
    assert count <= CODE_REFERENCE_PIN, (
        f"Chat has {count} __code__ references (pin {CODE_REFERENCE_PIN}); "
        "signal stock collaborators explicitly instead of comparing code objects."
    )


def test_console_configuration_preparation_imports_no_ui_private_names():
    tree = ast.parse(PREPARATION.read_text(encoding="utf-8"))
    private = [
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("tldw_chatbook.UI")
        for alias in node.names
        if alias.name.startswith("_")
    ]
    assert private == []
