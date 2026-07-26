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

import re
from pathlib import Path

from tldw_chatbook.Agents.builtin_services import BuiltinToolServices

from tldw_chatbook.Tools.file_operation_tools import (
    ListDirectoryTool,
    ReadFileTool,
    _tool_sandbox_root,
    is_within,
)
from tldw_chatbook.Tools.tool_executor import Tool


class ReadFile(ReadFileTool):
    """`read_file`, constructed under the pack contract."""

    def __init__(self, services: BuiltinToolServices | None = None) -> None:
        super().__init__()
        self.services = services


class ListDirectory(ListDirectoryTool):
    """`list_directory`, constructed under the pack contract."""

    def __init__(self, services: BuiltinToolServices | None = None) -> None:
        super().__init__()
        self.services = services


#: Most matches either tool returns. Results also pass through the runtime's
#: own `max_tool_result_chars` cap, but bounding here keeps the JSON small
#: enough that the cap rarely has to cut mid-structure.
_MAX_MATCHES = 200

#: Most filesystem entries either tool will EXAMINE, independent of how many
#: match. Verified necessary: `Path.glob("../**/*")` does not raise -- it
#: happily yields ~1.4M paths from a temp dir. Since none of them pass the
#: containment check, a match-only bound never trips and the tool walks the
#: entire filesystem. This bound is what actually stops that.
_MAX_CANDIDATES = 20_000


def _rejects_traversal(pattern: str) -> bool:
    """Whether a glob pattern tries to leave the sandbox root.

    Checked before globbing rather than filtering afterwards: containment
    filtering alone still pays the cost of walking everything the pattern
    matched (see ``_MAX_CANDIDATES``).

    Args:
        pattern: A user- or model-supplied glob pattern.

    Returns:
        True when the pattern is absolute or contains a `..` component.
    """
    return pattern.startswith("/") or ".." in Path(pattern).parts


class GlobFiles(Tool):
    """`glob_files` -- path-pattern search inside the sandbox root."""

    def __init__(self, services: BuiltinToolServices | None = None) -> None:
        self.services = services

    @property
    def name(self) -> str:
        return "glob_files"

    @property
    def description(self) -> str:
        return (
            "Find files by path pattern inside the tool sandbox. Supports "
            "glob syntax including ** for recursive matches, e.g. '**/*.py'. "
            f"Returns at most {_MAX_MATCHES} paths."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '**/*.py'.",
                }
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs) -> dict:
        """Search for files under the sandbox root by glob pattern.

        Args:
            pattern: Glob pattern, e.g. ``"**/*.py"``. Absolute patterns and
                patterns containing a `..` component are refused up front.

        Returns:
            Dict with a `matches` list of absolute path strings, or an
            `error` string.
        """
        pattern = str(kwargs.get("pattern") or "").strip()
        if not pattern:
            return {"error": "pattern is required"}
        if _rejects_traversal(pattern):
            return {"error": "pattern must stay inside the sandbox root"}
        root = _tool_sandbox_root()
        try:
            candidates = root.glob(pattern)
        except (ValueError, NotImplementedError) as exc:
            return {"error": f"invalid pattern: {exc}"}
        matches = []
        for examined, path in enumerate(candidates, start=1):
            if len(matches) >= _MAX_MATCHES or examined > _MAX_CANDIDATES:
                break
            if path.is_file() and is_within(path, root):
                matches.append(str(path))
        return {"matches": sorted(matches)}


class GrepFiles(Tool):
    """`grep_files` -- content search inside the sandbox root."""

    def __init__(self, services: BuiltinToolServices | None = None) -> None:
        self.services = services

    @property
    def name(self) -> str:
        return "grep_files"

    @property
    def description(self) -> str:
        return (
            "Search file contents by regular expression inside the tool "
            "sandbox, optionally narrowed by a path glob. Returns matching "
            f"lines with their file and line number, at most {_MAX_MATCHES}."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python regular expression to search for.",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional path glob to narrow the search.",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs) -> dict:
        """Search file contents under the sandbox root by regular expression.

        Args:
            pattern: Python regular expression to search for.
            glob: Optional path glob narrowing which files are searched,
                e.g. ``"**/*.py"``. Defaults to ``"**/*"``. Absolute
                patterns and patterns containing a `..` component are
                refused up front.

        Returns:
            Dict with a `matches` list of `{path, line_number, line}`
            dicts, or an `error` string.
        """
        raw_pattern = str(kwargs.get("pattern") or "")
        if not raw_pattern:
            return {"error": "pattern is required"}
        try:
            regex = re.compile(raw_pattern)
        except re.error as exc:
            return {"error": f"invalid regular expression: {exc}"}

        root = _tool_sandbox_root()
        glob_pattern = str(kwargs.get("glob") or "**/*")
        if _rejects_traversal(glob_pattern):
            return {"error": "glob must stay inside the sandbox root"}
        try:
            candidates = root.glob(glob_pattern)
        except (ValueError, NotImplementedError) as exc:
            return {"error": f"invalid glob: {exc}"}

        matches: list[dict] = []
        # Deliberately NOT sorted(candidates): materialising and sorting the
        # generator defeats _MAX_CANDIDATES on a broad pattern.
        for examined, path in enumerate(candidates, start=1):
            if len(matches) >= _MAX_MATCHES or examined > _MAX_CANDIDATES:
                break
            if not path.is_file() or not is_within(path, root):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if len(matches) >= _MAX_MATCHES:
                    break
                if regex.search(line):
                    matches.append(
                        {
                            "path": str(path),
                            "line_number": number,
                            "line": line[:500],
                        }
                    )
        return {"matches": matches}


#: Tool classes this pack contributes, in catalog order.
TOOLS: tuple[type, ...] = (ReadFile, ListDirectory, GlobFiles, GrepFiles)

#: Optional-dependency feature names required for this pack to appear.
#: Empty: the file tools use only the standard library.
REQUIRES: tuple[str, ...] = ()
