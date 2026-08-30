"""Messages published by the legacy ``ToolsSettingsWindow``.

Split out for task-24458. ``app.py`` needs ``IngestUiStyleChanged`` at class
BODY time, because it is the argument to an ``@on(...)`` decorator -- so the
import could not simply be deferred into a function. Importing the window to
get it pulled this chain onto the boot import path::

    app.py
      -> UI.Tools_Settings_Window
        -> Agents.local_tool_provider
          -> Tools.workspace_tool_executor
            -> Tools.{git,local,patch,virtual_cli}_tool_impls,
               Tools.workspace_tool_protocol, Tools.workspace_root_pin,
               Utils.filesystem_identity

...for a window that is DEPRECATED (TASK-1346), nav-unreachable, and whose
route resolves to the MCP screen. The message itself depends on nothing but
Textual, so moving it here severs the chain while leaving the window's own
behaviour untouched -- it re-exports this class as a class attribute, so both
``ToolsSettingsWindow.IngestUiStyleChanged`` and ``self.IngestUiStyleChanged``
still resolve exactly as before.
"""

from __future__ import annotations

from textual.message import Message

__all__ = ["IngestUiStyleChanged"]


class IngestUiStyleChanged(Message):
    """Request that the app refresh the active ingest view after a style change."""

    def __init__(self, new_style: str) -> None:
        super().__init__()
        self.new_style = new_style
