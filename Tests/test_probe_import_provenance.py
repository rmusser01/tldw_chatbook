"""Prove navigation tests import every critical module from this checkout."""

from __future__ import annotations

from pathlib import Path

import tldw_chatbook
from tldw_chatbook.Chat import console_chat_controller, console_runtime
from tldw_chatbook.UI.Screens import chat_screen


def test_probe_imports_this_worktree():
    here = Path(__file__).resolve().parents[1]
    modules = (
        tldw_chatbook,
        console_runtime,
        console_chat_controller,
        chat_screen,
    )
    print(f"PROBE worktree root: {here}")
    for module in modules:
        imported = Path(module.__file__).resolve()
        print(f"PROBE imported {module.__name__} from: {imported}")
        assert imported.is_relative_to(here), (
            f"imported {imported} is NOT under {here} -- the editable install's "
            "foreign-worktree finder won"
        )
