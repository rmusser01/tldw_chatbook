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
from pathlib import Path, PureWindowsPath

from tldw_chatbook.Agents.builtin_services import BuiltinToolServices
from tldw_chatbook.Tools.base import Tool
from tldw_chatbook.Tools.file_operation_tools import (
    ListDirectoryTool,
    ReadFileTool,
    _tool_sandbox_root,
    is_within,
)
from tldw_chatbook.Utils.sensitive_paths import resolve_sensitive_context


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

#: Per-file byte cap for `grep_files`. Streaming the file line-by-line (see
#: `GrepFiles.execute`) already avoids the large peak allocation a whole-file
#: `read_text()` would cost, but a single pathological file with no newline
#: characters would still force one giant line to be buffered in full. This
#: bounds that worst case independent of `_MAX_CANDIDATES`/`_MAX_MATCHES`,
#: which bound the number of files/matches, not the size of any one file.
_MAX_GREP_FILE_BYTES = 5_000_000


def _rejects_traversal(pattern: str) -> bool:
    """Whether a glob pattern tries to leave the sandbox root.

    Checked before globbing rather than filtering afterwards: containment
    filtering alone still pays the cost of walking everything the pattern
    matched (see ``_MAX_CANDIDATES``).

    Args:
        pattern: A user- or model-supplied glob pattern.

    Returns:
        True when the pattern is absolute -- POSIX (``/etc/...``), Windows
        drive-letter (``C:\\...``), or Windows UNC (``\\\\server\\share\\...``)
        -- or contains a `..` component. Both absolute forms are checked
        regardless of the host OS: `Path(pattern).is_absolute()` alone only
        recognizes the form native to the platform actually running this
        process, so on a POSIX host a Windows drive-letter or UNC pattern
        would silently fail to be rejected here (`is_within` still guards
        every candidate either way, so this was a cost/consistency gap,
        never an escape).
    """
    return (
        Path(pattern).is_absolute()
        or PureWindowsPath(pattern).is_absolute()
        or ".." in Path(pattern).parts
    )


def _is_hidden_within(resolved: Path, root_resolved: Path) -> bool:
    """Whether a resolved candidate has a dot-prefixed component under root.

    Mirrors the hidden-component rule ``Utils.path_validation.validate_path``
    applies for `read_file`/`write_file` (its user-supplied-portion check),
    so `glob_files`/`grep_files` cannot surface a dotfile/dotdir -- e.g. a
    `.env` secret -- that those tools would refuse to touch directly.

    Applied here against an already-resolved path rather than by calling
    `validate_path` itself for each candidate. `validate_path` raises
    `ValueError` on rejection, but a glob candidate that merely fails to
    qualify is not an error for these tools -- it is simply skipped. It
    also carries per-call logging/timing overhead irrelevant to a candidate
    that came from `Path.glob()` rather than directly from user input:
    benchmarked over a 1,500-file sandbox tree, routing every candidate
    through `validate_path` cost ~46% more wall-clock than this inline
    check's ~20% over the `is_within`-only baseline (0.164ms -> 0.240ms vs.
    0.164ms -> 0.197ms per candidate; see the report for the full numbers).
    For a rule with no other behavioural difference in this context, that
    was worth avoiding.

    Args:
        resolved: The already-resolved candidate path (the return value of
            ``path.resolve()``, not the raw candidate from ``glob()``).
        root_resolved: The already-resolved sandbox root.

    Returns:
        True if any path component between `root_resolved` and `resolved`
        starts with `.`, or if `resolved` is not actually under
        `root_resolved`. The latter should not occur in practice --
        `is_within` is the real containment check and must always be
        called first -- but failing closed here costs nothing.
    """
    try:
        relative_parts = resolved.relative_to(root_resolved).parts
    except ValueError:
        return True
    return any(part.startswith(".") for part in relative_parts)


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
        try:
            root = _tool_sandbox_root()
        except OSError as exc:
            return {"error": f"sandbox root is not usable: {exc}"}
        try:
            candidates = root.glob(pattern)
        except (ValueError, NotImplementedError) as exc:
            return {"error": f"invalid pattern: {exc}"}
        matches: list[str] = []
        examined = 0
        # Resolved ONCE for this call and reused for every candidate below,
        # rather than letting `is_within` -> `is_sensitive_path` re-resolve
        # the sensitive-path set (11 config accessors) per candidate -- see
        # Utils.sensitive_paths.resolve_sensitive_context.
        sensitive_ctx = resolve_sensitive_context()
        root_resolved = root.resolve()
        while True:
            # `Path.glob()` validates lazily: a malformed pattern (e.g.
            # "**foo/*") doesn't raise at construction above, it raises on
            # the first `next()` here. Only the `next()` call is inside
            # this try -- `path.is_file()`/`is_within()` below run outside
            # it, so a ValueError from the loop body is never misreported
            # as an invalid pattern.
            try:
                path = next(candidates)
            except StopIteration:
                break
            except (ValueError, NotImplementedError) as exc:
                return {"error": f"invalid pattern: {exc}"}
            examined += 1
            if len(matches) >= _MAX_MATCHES or examined > _MAX_CANDIDATES:
                break
            if not path.is_file() or not is_within(path, root, context=sensitive_ctx):
                continue
            # A dotfile/dotdir must be invisible here even though it passed
            # `is_within` -- that call applies the credential/app-state
            # denylist, not the hidden-component rule `read_file`/`write_file`
            # enforce via `validate_path`. See `_is_hidden_within`.
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError):
                continue
            if _is_hidden_within(resolved, root_resolved):
                continue
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

        try:
            root = _tool_sandbox_root()
        except OSError as exc:
            return {"error": f"sandbox root is not usable: {exc}"}
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
        examined = 0
        # Resolved ONCE for this call and reused for every candidate below --
        # see the matching comment in GlobFiles.execute above.
        sensitive_ctx = resolve_sensitive_context()
        root_resolved = root.resolve()
        while True:
            # As in GlobFiles: `Path.glob()` validates lazily, so a bad
            # pattern raises here, on `next()`, not at the call above. Only
            # `next()` is inside this try -- the body below (is_file,
            # is_within, the streamed read, regex.search) runs outside it,
            # so a ValueError raised there is never misreported as a bad
            # glob.
            try:
                path = next(candidates)
            except StopIteration:
                break
            except (ValueError, NotImplementedError) as exc:
                return {"error": f"invalid glob: {exc}"}
            examined += 1
            if len(matches) >= _MAX_MATCHES or examined > _MAX_CANDIDATES:
                break
            if not path.is_file() or not is_within(path, root, context=sensitive_ctx):
                continue
            # A dotfile/dotdir must be unreadable here even though it passed
            # `is_within` -- see the matching comment in GlobFiles.execute
            # and `_is_hidden_within`.
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError):
                continue
            if _is_hidden_within(resolved, root_resolved):
                continue
            try:
                if path.stat().st_size > _MAX_GREP_FILE_BYTES:
                    continue
            except OSError:
                continue
            # Streamed line-by-line rather than `read_text()` + `splitlines()`
            # (which would materialize the whole file, and a second full
            # copy split into lines, in memory at once): one large file in
            # the sandbox previously forced a large peak allocation. The
            # per-file byte cap above still bounds the worst case for a
            # single pathological line with no newline.
            try:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for number, line in enumerate(fh, start=1):
                        if regex.search(line):
                            matches.append(
                                {
                                    "path": str(path),
                                    "line_number": number,
                                    "line": line.rstrip("\n")[:500],
                                }
                            )
                        if len(matches) >= _MAX_MATCHES:
                            break
            except OSError:
                continue
        return {"matches": matches}


#: Tool classes this pack contributes, in catalog order.
TOOLS: tuple[type, ...] = (ReadFile, ListDirectory, GlobFiles, GrepFiles)

#: Optional-dependency feature names required for this pack to appear.
#: Empty: the file tools use only the standard library.
REQUIRES: tuple[str, ...] = ()
