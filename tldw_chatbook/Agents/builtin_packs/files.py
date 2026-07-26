# tldw_chatbook/Agents/builtin_packs/files.py
"""The `files` pack: sandbox-rooted filesystem reads.

Wraps the existing implementations in ``Tools/file_operation_tools.py``
rather than reimplementing them. Those tools already confine every path to
``_tool_sandbox_root()``; this pack does not widen that (see the plan's
Global Constraints -- workspace-rooting is a separate, signed-off change).

The thin subclasses exist to satisfy the pack contract: every pack tool
constructs as ``cls(services=...)`` and its metadata properties never touch
services, so TASK-656's enumerator can describe them with ``services=None``.
"""

from __future__ import annotations

from typing import Any

from tldw_chatbook.Tools.file_operation_tools import ListDirectoryTool, ReadFileTool


class ReadFile(ReadFileTool):
    """`read_file`, constructed under the pack contract."""

    def __init__(self, services: Any | None = None) -> None:
        super().__init__()
        self.services = services


class ListDirectory(ListDirectoryTool):
    """`list_directory`, constructed under the pack contract."""

    def __init__(self, services: Any | None = None) -> None:
        super().__init__()
        self.services = services


#: Tool classes this pack contributes, in catalog order.
TOOLS: tuple[type, ...] = (ReadFile, ListDirectory)

#: Optional-dependency feature names required for this pack to appear.
#: Empty: the file tools use only the standard library.
REQUIRES: tuple[str, ...] = ()
