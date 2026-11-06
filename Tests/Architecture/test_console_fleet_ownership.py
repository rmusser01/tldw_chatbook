"""Fleet policy belongs to a controller; screen identity stays with the DOM."""

import ast
from pathlib import Path

from Tests.Architecture.test_console_wave6_inventory import WAVE6_GROUPS

ROOT = Path(__file__).resolve().parents[2]


def test_fleet_policy_has_one_non_dom_owner():
    path = ROOT / "tldw_chatbook/UI/Console_Modules/fleet.py"
    assert path.exists(), "Fleet lifecycle still belongs to ChatScreen"
    owner = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ConsoleFleetLifecycleController"
    )
    methods = {
        n.name
        for n in owner.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # TASK-38988 review: the two DOM-reading probes the screen kept now live
    # on the controller too, reading through wired accessors (no DOM calls,
    # checked below), so the whole moved group belongs to the owner.
    moved = WAVE6_GROUPS["fleet"].moved
    assert moved <= methods
    screen = ast.parse((ROOT / "tldw_chatbook/UI/Screens/chat_screen.py").read_text())
    screen_class = next(
        n for n in screen.body if isinstance(n, ast.ClassDef) and n.name == "ChatScreen"
    )
    screen_methods = {
        n.name
        for n in screen_class.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not moved & screen_methods
    assert not any(
        isinstance(n, ast.Attribute)
        and n.attr in {"query_one", "query", "mount", "draft_text"}
        for n in ast.walk(owner)
    )
