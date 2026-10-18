"""``python -m tldw_chatbook.app`` runs the app from app.py's last main block (TASK-35302).

Under ``-m`` the app runs inside that block, so module-level code after it does
not execute until the app exits. ``_CONSOLE_SKILL_APP_SOURCE`` was once defined
there; every ``python -m tldw_chatbook.app`` start (each served web child among
them) then crashed when the deferred collections wiring read it.
"""

import ast
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _is_main_guard(node: ast.stmt) -> bool:
    """A top-level ``if __name__ == "__main__":``, either operand order."""
    test = getattr(node, "test", None)
    if not isinstance(node, ast.If) or not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    operands = (test.left, test.comparators[0])
    name = any(isinstance(o, ast.Name) and o.id == "__name__" for o in operands)
    main = any(isinstance(o, ast.Constant) and o.value == "__main__" for o in operands)
    return name and main


def test_nothing_at_module_level_follows_app_main_block():
    body = ast.parse((ROOT / "tldw_chatbook/app.py").read_text(encoding="utf-8")).body
    last = max(index for index, node in enumerate(body) if _is_main_guard(node))
    after = [f"line {node.lineno}" for node in body[last + 1 :]]
    assert not after, f"app.py runs code after its main block: {after}"
    assigned = {
        target.id
        for node in body[:last]
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_CONSOLE_SKILL_APP_SOURCE" in assigned


@pytest.mark.parametrize("record", [None, ("not", "the", "stock", "record")])
def test_collections_wiring_defers_without_the_stock_app_record(monkeypatch, record):
    # A stand-in app module: the reader must not reach app.py's import-time
    # config load, and app_collections_wiring loads only through its owner.
    import tldw_chatbook.app_service_wiring  # noqa: F401
    from tldw_chatbook import app_collections_wiring

    stand_in = types.ModuleType("tldw_chatbook.app")
    if record is not None:
        stand_in._CONSOLE_SKILL_APP_SOURCE = record
    monkeypatch.setitem(sys.modules, "tldw_chatbook.app", stand_in)
    assert app_collections_wiring._capture_deferred_collections_setup(object()) is None
