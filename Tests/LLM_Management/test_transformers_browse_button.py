"""The Transformers 'browse models directory' button must open a picker.

TASK-32811.1. The handler imported `FileOpen` from a top-level
`textual_fspicker` package that is not installed, so every press hit the
`except ImportError` and notified "File picker utility not available" -- and
it asked `FileOpen` for a `select_dirs` argument `FileOpen` does not accept,
so the import path could not have worked either. It uses the vendored
`SelectDirectory` now.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
    llm_management_events_transformers as handlers,
)
from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory


class _App:
    """A minimal app double: records push_screen and notify calls."""

    def __init__(self) -> None:
        self.pushed = []
        self.notifications = []

    async def push_screen(self, screen, callback=None):
        self.pushed.append((screen, callback))

    def notify(self, message, severity="information"):
        self.notifications.append((message, severity))


@pytest.mark.asyncio
async def test_browse_button_pushes_a_directory_picker():
    app = _App()
    await handlers.handle_transformers_browse_models_dir_button_pressed(None, app, None)

    assert not app.notifications, (
        f"the handler notified instead of opening a picker: {app.notifications}"
    )
    assert len(app.pushed) == 1, "the handler did not push a picker screen"
    screen, callback = app.pushed[0]
    assert isinstance(screen, SelectDirectory), (
        f"expected a vendored SelectDirectory, got {type(screen).__name__}"
    )
    assert callback is not None, "no callback wired to update the path input"


@pytest.mark.asyncio
async def test_the_handler_no_longer_touches_the_unvendored_package():
    """Guard the specific regression: the import must be the vendored one."""
    import ast
    from pathlib import Path

    source = Path(handlers.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "textual_fspicker":
            raise AssertionError(
                "still imports the unvendored top-level textual_fspicker"
            )
    assert "Third_Party.textual_fspicker import SelectDirectory" in source
