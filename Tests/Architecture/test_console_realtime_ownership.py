"""Realtime lifecycle belongs to its controller; chip pixels stay on Screen."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_realtime_lifecycle_has_one_controller_owner() -> None:
    controller = ROOT / "tldw_chatbook/UI/Console_Modules/realtime.py"
    assert controller.exists(), "Realtime lifecycle still belongs to ChatScreen"
    source = ast.parse(controller.read_text())
    owner = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "ConsoleRealtimeController"
    )
    methods = {
        n.name
        for n in owner.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {
        "_enter_console_realtime_loop",
        "_release_console_realtime_state",
        "_close_console_realtime_resources",
        "_console_realtime_marshal",
    } <= methods
    screen = ast.parse((ROOT / "tldw_chatbook/UI/Screens/chat_screen.py").read_text())
    screen_class = next(
        n for n in screen.body if isinstance(n, ast.ClassDef) and n.name == "ChatScreen"
    )
    screen_methods = {
        n.name
        for n in screen_class.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    lifecycle_methods = {name for name in methods if "realtime" in name}
    assert not lifecycle_methods & screen_methods
    assert "_repaint_console_realtime_chip" in screen_methods
    assert not any(
        isinstance(n, ast.Attribute)
        and n.attr in {"query_one", "query", "mount", "remove", "set_voice_status"}
        for n in ast.walk(owner)
    )
