"""Disposable HEAD-method control for neighboring failures."""

import ast
import subprocess

import pytest

from Tests.UI import test_library_file_notes_workspace as files
from Tests.UI import test_library_shell as shells
from tldw_chatbook.UI.Screens import library_screen as module


def restore_head(monkeypatch):
    text = subprocess.check_output(
        ["git", "show", "af5a26a2af:tldw_chatbook/UI/Screens/library_screen.py"],
        text=True,
    )
    tree = ast.parse(text)
    for name in (
        "_active_library_rail",
        "_sync_library_ordinary_rail_width_contract",
        "_library_focusable",
        "on_resize",
    ):
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        namespace = dict(vars(module))
        exec(  # noqa: S102 -- compile only the explicitly pinned local git source.
            compile(ast.Module(body=[node], type_ignores=[]), module.__file__, "exec"),
            namespace,
        )
        monkeypatch.setattr(module.LibraryScreen, name, namespace[name])


@pytest.mark.asyncio
async def test_width_control(monkeypatch):
    restore_head(monkeypatch)
    await shells.test_ordinary_rail_restores_custom_owner_after_collapse_and_adaptive_route(
        24
    )


@pytest.mark.asyncio
async def test_notes_control(monkeypatch, tmp_path):
    restore_head(monkeypatch)
    await files.test_notes_authority_round_trip_retains_both_workspaces(tmp_path)
