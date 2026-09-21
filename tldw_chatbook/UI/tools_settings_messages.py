"""Messages published by the legacy ``ToolsSettingsWindow``.

Split out for task-24458. It was originally kept out of the window module
because ``app.py`` referenced ``IngestUiStyleChanged`` at class-BODY time (as
the argument to an ``@on(...)`` decorator), so importing the window just to get
the class pulled this chain onto the boot import path::

    app.py
      -> UI.Tools_Settings_Window
        -> Agents.local_tool_provider
          -> Tools.workspace_tool_executor
            -> Tools.{git,local,patch,virtual_cli}_tool_impls,
               Tools.workspace_tool_protocol, Tools.workspace_root_pin,
               Utils.filesystem_identity

...for a window that is DEPRECATED (TASK-1346), nav-unreachable, and whose
route resolves to the MCP screen. That app.py ``@on`` handler was removed as
dead code in task-32810 (its ``#ingest-window`` target is composed nowhere, so
it only ever hit the ``QueryError`` branch), so app.py no longer imports this
class. The module stays split out regardless: the deprecated window still
re-exports the class as a class attribute and constructs it
(``self.IngestUiStyleChanged(...)``), so ``ToolsSettingsWindow.IngestUiStyleChanged``
and ``self.IngestUiStyleChanged`` keep resolving exactly as before, and the
window's own behaviour is untouched. The message itself depends on nothing but
Textual.
"""

from __future__ import annotations

from textual.message import Message

__all__ = ["IngestUiStyleChanged"]


class IngestUiStyleChanged(Message):
    """Request that the app refresh the active ingest view after a style change."""

    def __init__(self, new_style: str) -> None:
        super().__init__()
        self.new_style = new_style
