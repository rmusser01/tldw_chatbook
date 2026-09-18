"""Diagnostic pytest plugin: delay the replacement reader dropdown compose.

Write failure lifecycle state to /tmp/task-32628-mount-diagnosis.jsonl.
"""

import asyncio
import json
import traceback
from pathlib import Path

from textual.app import App
from textual.css.query import NoMatches
from textual.widgets import Select

shutdown = set()
original_shutdown = App._shutdown
original_mount = Select._on_mount


async def traced_shutdown(self):
    shutdown.add(id(self))
    return await original_shutdown(self)


def traced_mount(self, event):
    try:
        return original_mount(self, event)
    except NoMatches:
        app = self.app
        record = {
            "widget": repr(self),
            "shutdown_started": id(app) in shutdown,
            "is_attached": self.is_attached,
            "is_mounted": self.is_mounted,
            "children": [repr(node) for node in self.children],
            "app_running": app.is_running,
            "app_exiting": app._exit,
            "stack": traceback.format_exc(),
        }
        with Path("/tmp/task-32628-mount-diagnosis.jsonl").open("a") as target:
            target.write(json.dumps(record) + "\n")
        raise


App._shutdown = traced_shutdown
Select._on_mount = traced_mount

original_compose = Select._on_compose


async def traced_compose(self, event):
    if self.id == "prompt-editor-save-menu":
        state = getattr(self.app.screen, "_prompts_state", None)
        detail = getattr(state, "detail", None)
        if isinstance(detail, dict) and detail.get("name") == "Second prompt":
            await asyncio.sleep(0.2)
    return await original_compose(self, event)


Select._on_compose = traced_compose
