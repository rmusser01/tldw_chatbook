"""
File Operation Tools for LLM function calling.

These tools allow LLMs to perform safe file operations with proper validation.
"""

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path, PureWindowsPath
from typing import Dict, Any, Iterator

from loguru import logger

from . import Tool
from ..Utils.path_validation import validate_path_multi
from ..Utils.sensitive_paths import (
    SensitivePathContext,
    is_sensitive_path,
    refuses_new_directory_chain,
    resolve_sensitive_context,
)
from .workspace_file_roots import allowed_file_roots


def _resolve_sandbox_config() -> str:
    """Return the configured sandbox root string (indirection for tests)."""
    from ..config import get_cli_setting, get_user_data_dir

    default_root = str(get_user_data_dir() / "tool_sandbox")
    return get_cli_setting("tools", "file_sandbox_root", default_root) or default_root


def is_within(
    candidate: Path, root: Path, context: SensitivePathContext | None = None
) -> bool:
    """Return whether ``candidate`` resolves inside ``root`` and is not sensitive.

    Callers include ``ListDirectoryTool``'s containment-root selection and
    recursive-descent guard, and ``GlobFiles``/``GrepFiles`` below, which
    consult it on every candidate path before returning or reading it. It
    does not by itself cover a tool's own top-level target: ``ReadFileTool``,
    ``WriteFileTool`` and ``ListDirectoryTool`` each call
    ``Utils.sensitive_paths.is_sensitive_path`` directly on their target
    before touching the filesystem, independently of this function.

    Args:
        candidate: Path to test.
        root: The root it must stay under (the sandbox root, or one of the
            run's bound workspace folders).
        context: Optional pre-resolved ``SensitivePathContext`` (see
            ``Utils.sensitive_paths.resolve_sensitive_context``). Callers
            that test many candidates in one tool invocation (``GlobFiles``,
            ``GrepFiles``, the recursive-listing walk) should resolve this
            ONCE per invocation and pass it through here, rather than let
            every candidate re-resolve the sensitive-path set from scratch.
            Leave ``None`` for a one-off check -- that still enforces the
            denylist, just resolved fresh; it never means "nothing is
            sensitive".

    Returns:
        True only when the fully-resolved candidate is the root or below it
        AND is not a sensitive path.
    """
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except (OSError, RuntimeError):
        return False
    if is_sensitive_path(resolved, context=context):
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


def _tool_sandbox_root() -> Path:
    """Resolve + create the file-tool sandbox root.

    The file tools confine all reads/writes/listings under this directory.
    Defaults to ``<user data dir>/tool_sandbox``; override with
    ``[tools] file_sandbox_root`` in config.toml.
    """
    root = Path(_resolve_sandbox_config()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


class ReadFileTool(Tool):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Returns the file content as text."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8",
                },
            },
            "required": ["file_path"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Reading arbitrary sandbox files is a disclosure risk."""
        return ("reads",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Read a file's contents.

        Args:
            file_path: Path to the file
            encoding: File encoding (default: utf-8)

        Returns:
            Dictionary with file content or error
        """
        file_path = kwargs.get("file_path")
        if not file_path:
            return {"error": "No file path provided"}

        encoding = kwargs.get("encoding", "utf-8")

        try:
            # Validate the path against the sandbox plus any read-eligible
            # workspace folder roots bound to the run.
            validated_path = validate_path_multi(
                file_path,
                allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root()),
            )
            path = Path(validated_path)

            # Refuse credential, gate-state, and app-database paths outright,
            # regardless of the sandbox/workspace roots (see
            # Utils.sensitive_paths). This must run before any filesystem
            # access below.
            if is_sensitive_path(path):
                return {
                    "file_path": file_path,
                    "error": f"Refused: '{file_path}' is a protected path and cannot be read",
                }

            # Check if file exists
            if not path.exists():
                return {
                    "error": f"File not found: {file_path}",
                    "absolute_path": str(path.absolute()),
                }

            # Check if it's a file
            if not path.is_file():
                return {
                    "error": f"Path is not a file: {file_path}",
                    "path_type": "directory" if path.is_dir() else "other",
                }

            # Read the file
            content = path.read_text(encoding=encoding)

            # Get file info
            stat = path.stat()

            return {
                "file_path": str(path),
                "content": content,
                "size_bytes": stat.st_size,
                "encoding": encoding,
                "lines": len(content.splitlines()),
            }

        except UnicodeDecodeError as e:
            return {
                "file_path": file_path,
                "error": f"Unable to decode file with {encoding} encoding: {e}",
                "suggestion": "Try a different encoding like 'latin-1' or 'cp1252'",
            }
        except PermissionError:
            return {"file_path": file_path, "error": "Permission denied to read file"}
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"file_path": file_path, "error": f"Failed to read file: {str(e)}"}


class ListDirectoryTool(Tool):
    """Tool for listing directory contents."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List the contents of a directory. Returns file names, types, and basic info."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "The path to the directory to list",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with .)",
                    "default": False,
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List subdirectories recursively",
                    "default": False,
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for recursive listing",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["directory_path"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Enumerating the sandbox discloses its structure."""
        return ("reads",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            directory_path: Path to the directory
            include_hidden: Include hidden files
            recursive: List recursively
            max_depth: Max depth for recursion

        Returns:
            Dictionary with directory contents or error
        """
        directory_path = kwargs.get("directory_path", ".")
        include_hidden = kwargs.get("include_hidden", False)
        recursive = kwargs.get("recursive", False)
        max_depth = kwargs.get("max_depth", 2)

        try:
            # Validate the path against the sandbox plus any read-eligible
            # workspace folder roots bound to the run.
            sandbox_root = _tool_sandbox_root()
            read_roots = allowed_file_roots(write=False, sandbox_root=sandbox_root)
            validated_path = validate_path_multi(directory_path, read_roots)
            path = Path(validated_path)

            # Resolve the sensitive-path set ONCE for this call and reuse it
            # for the top-level check below, containment_root selection, and
            # every per-entry check the recursive walk makes -- not a fresh
            # resolution per entry (see
            # Utils.sensitive_paths.resolve_sensitive_context).
            sensitive_ctx = resolve_sensitive_context()

            # Refuse credential, gate-state, and app-database paths outright,
            # regardless of the sandbox/workspace roots (see
            # Utils.sensitive_paths). This must run before any filesystem
            # access below.
            if is_sensitive_path(path, context=sensitive_ctx):
                return {
                    "directory_path": directory_path,
                    "error": f"Refused: '{directory_path}' is a protected path and cannot be listed",
                }

            # The recursive-descent symlink guard below must compare against
            # whichever allowed root actually contains ``path`` — not always
            # the sandbox — otherwise a legitimately bound workspace folder
            # would silently refuse to recurse past its top level.
            containment_root = next(
                (root for root in read_roots if is_within(path, root, context=sensitive_ctx)),
                sandbox_root,
            )

            # Check if directory exists
            if not path.exists():
                return {
                    "error": f"Directory not found: {directory_path}",
                    "absolute_path": str(path.absolute()),
                }

            # Check if it's a directory
            if not path.is_dir():
                return {
                    "error": f"Path is not a directory: {directory_path}",
                    "path_type": "file" if path.is_file() else "other",
                }

            entries = []

            def list_dir_contents(dir_path: Path, current_depth: int = 0):
                """Recursively list directory contents."""
                if current_depth > max_depth:
                    return

                try:
                    for item in sorted(dir_path.iterdir()):
                        # Skip hidden files if not requested
                        if not include_hidden and item.name.startswith("."):
                            continue

                        # Refuse individual sensitive entries the same way the
                        # top-level target is refused above. The recursive
                        # descent guard below already stops the walk from
                        # entering a sensitive DIRECTORY (e.g. ~/.ssh), but a
                        # sensitive FILE sitting inside an otherwise-ordinary
                        # directory (this app's own config.toml,
                        # mcp_permissions.json, or a ChaChaNotes DB and its
                        # WAL/SHM sidecars) would still be listed by name and
                        # size without this per-entry check.
                        if is_sensitive_path(item, context=sensitive_ctx):
                            continue

                        # Get item info
                        try:
                            stat = item.stat()
                            entry = {
                                "name": item.name,
                                "path": str(item.relative_to(path)),
                                "type": "directory" if item.is_dir() else "file",
                                "size_bytes": stat.st_size if item.is_file() else None,
                                "depth": current_depth,
                            }
                            entries.append(entry)

                            # Recursively list subdirectories, but NEVER
                            # follow a symlink: a link planted inside an
                            # allowed root would otherwise let the walk
                            # enumerate files outside it, breaking the
                            # containment every other path here relies on.
                            # Belt-and-braces, the resolved child must still
                            # sit under the root that contains this listing.
                            if (
                                recursive
                                and item.is_dir()
                                and not item.is_symlink()
                                and current_depth < max_depth
                                and is_within(item, containment_root, context=sensitive_ctx)
                            ):
                                list_dir_contents(item, current_depth + 1)

                        except PermissionError:
                            entries.append(
                                {
                                    "name": item.name,
                                    "path": str(item.relative_to(path)),
                                    "type": "inaccessible",
                                    "error": "Permission denied",
                                    "depth": current_depth,
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Error accessing {item}: {e}")

                except PermissionError:
                    return {"error": f"Permission denied to list directory: {dir_path}"}

            # List the directory
            list_dir_contents(path)

            # Count types
            file_count = sum(1 for e in entries if e.get("type") == "file")
            dir_count = sum(1 for e in entries if e.get("type") == "directory")

            return {
                "directory_path": str(path),
                "total_entries": len(entries),
                "file_count": file_count,
                "directory_count": dir_count,
                "entries": entries[:100],  # Limit to first 100 entries
            }

        except PermissionError:
            return {
                "directory_path": directory_path,
                "error": "Permission denied to access directory",
            }
        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {e}")
            return {
                "directory_path": directory_path,
                "error": f"Failed to list directory: {str(e)}",
            }


class WriteFileTool(Tool):
    """Tool for writing content to files."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates the file if it doesn't exist. Use with caution."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
                "mode": {
                    "type": "string",
                    "description": "Write mode: 'overwrite' or 'append'",
                    "enum": ["overwrite", "append"],
                    "default": "overwrite",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8",
                },
                "create_directories": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist",
                    "default": False,
                },
            },
            "required": ["file_path", "content"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Creates, overwrites, or appends to files."""
        return ("mutates",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write
            mode: Write mode (overwrite or append)
            encoding: File encoding
            create_directories: Create parent dirs if needed

        Returns:
            Dictionary with success status or error
        """
        file_path = kwargs.get("file_path")
        content = kwargs.get("content")

        if not file_path:
            return {"error": "No file path provided"}
        if content is None:
            return {"error": "No content provided"}

        mode = kwargs.get("mode", "overwrite")
        encoding = kwargs.get("encoding", "utf-8")
        create_directories = kwargs.get("create_directories", False)

        try:
            # Validate the path against the sandbox plus any write-eligible
            # (rw) workspace folder roots bound to the run.
            validated_path = validate_path_multi(
                file_path,
                allowed_file_roots(write=True, sandbox_root=_tool_sandbox_root()),
            )
            path = Path(validated_path)

            # Refuse credential, gate-state, and app-database paths outright,
            # regardless of the sandbox/workspace roots (see
            # Utils.sensitive_paths). This must run before any filesystem
            # access below.
            if is_sensitive_path(path):
                return {
                    "file_path": file_path,
                    "error": f"Refused: '{file_path}' is a protected path and cannot be written",
                }

            # Check if we're overwriting an existing file
            file_exists = path.exists()

            # Create parent directories if requested
            if create_directories and not path.parent.exists():
                # TASK-849: `is_sensitive_path(path)` just above only ever
                # validated the FINAL file being written, never the new
                # directory levels `mkdir(parents=True)` is about to create
                # on the way there -- so nothing stopped an agent from
                # planting a directory at a name this app expects to use
                # for its own state file later (e.g. `search_history.db/`,
                # created here as a side effect of writing
                # `search_history.db/note.txt`, before the app ever
                # creates `search_history.db` itself as a SQLite file).
                # This app's later attempt to open it then fails outright --
                # a denial of service. See
                # `Utils.sensitive_paths.refuses_new_directory_chain` for
                # why this walks every not-yet-existing ancestor rather
                # than just `path.parent` itself.
                if refuses_new_directory_chain(path.parent):
                    return {
                        "file_path": file_path,
                        "error": (
                            f"Refused: creating '{path.parent}' would collide "
                            "with a protected path"
                        ),
                    }
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directories: {path.parent}")

            # Check if parent directory exists
            if not path.parent.exists():
                return {
                    "error": f"Parent directory does not exist: {path.parent}",
                    "suggestion": "Set create_directories=true to create it",
                }

            # Write the file
            if mode == "append" and file_exists:
                # Append mode
                with open(path, "a", encoding=encoding) as f:
                    f.write(content)
                action = "appended to"
            else:
                # Overwrite mode
                with open(path, "w", encoding=encoding) as f:
                    f.write(content)
                action = "created" if not file_exists else "overwritten"

            # Get file info after writing
            stat = path.stat()

            return {
                "file_path": str(path),
                "action": action,
                "size_bytes": stat.st_size,
                "encoding": encoding,
                "lines_written": len(content.splitlines()),
            }

        except PermissionError:
            return {"file_path": file_path, "error": "Permission denied to write file"}
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return {"file_path": file_path, "error": f"Failed to write file: {str(e)}"}


#: Most matches either GlobFiles or GrepFiles returns. Results also pass
#: through the agent runtime's own `max_tool_result_chars` cap, but bounding
#: here keeps the JSON small enough that the cap rarely has to cut mid-structure.
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

#: Length cap on the slice of each line actually handed to `regex.search`
#: in the grep worker (`_grep_worker.py`, spawned by `_run_grep_subprocess`
#: -- TASK-843). Python's `re` module has no match timeout, and a
#: catastrophic-backtracking pattern (e.g. `(a+)+$`) burns CPU
#: superlinearly in input length -- measured on this branch,
#: `re.compile(r'(a+)+$').search('a' * 30 + 'X')` alone took ~47s, and the
#: cost roughly doubles per additional character from there. Before this
#: cap, the search ran against the FULL line while only the *stored*
#: result was truncated (to 500 chars, below) -- and since
#: `_MAX_GREP_FILE_BYTES` bounds one *file*, not one *line*, that full
#: line could be up to ~5,000,000 characters for a file with no
#: newlines. This remains the FIRST line of defence -- cheap, and it
#: shrinks the ordinary (non-pathological) cost of every search -- but on
#: its own it does not make catastrophic backtracking fast, only smaller:
#: a sufficiently adversarial pattern run against even a
#: `_MAX_GREP_LINE_SEARCH_CHARS`-length slice can still be expensive (our
#: own repro above needed only 30 characters to already run ~47s). What
#: actually bounds the CPU a pathological pattern can consume AFTER the
#: tool call returns is `_run_grep_subprocess`/`_grep_worker.py`: the
#: search itself now runs in a separate, killable process rather than the
#: in-process worker THREAD `Agents/agent_service.py`'s
#: `_call_with_timeout` uses for every other tool -- a timed-out THREAD is
#: abandoned (Python cannot kill one), but a timed-out PROCESS is
#: `Popen.kill()`ed. See `_run_grep_subprocess`'s docstring for exactly
#: what that does and does not guarantee.
_MAX_GREP_LINE_SEARCH_CHARS = 500

#: Total number of lines `GrepFiles.execute` will read across ALL
#: candidate files in one invocation, independent of how many match or
#: how many files are examined. `_MAX_GREP_FILE_BYTES` bounds a single
#: file and `_MAX_CANDIDATES` bounds how many glob results are looked
#: at, but neither bounds the total number of *lines* actually streamed
#: and searched -- a large corpus of small-line files (each individually
#: under the per-file byte cap) can still add up to an enormous line
#: count. This bound stops that from extending one invocation's total
#: CPU exposure. It is a *different* concern from
#: `_MAX_GREP_LINE_SEARCH_CHARS` above: this one bounds the aggregate
#: cost of many ordinary (non-pathological) per-line searches, not the
#: cost of any single catastrophic one.
_MAX_GREP_LINES_SCANNED = 200_000

#: Wall-clock ceiling for the grep worker SUBPROCESS itself (TASK-843),
#: passed to `_run_grep_subprocess`. Deliberately shorter than
#: `GrepFiles.timeout_seconds` (20.0s): this is the timeout that actually
#: matters for bounding CPU, because unlike the run loop's own
#: thread-based `_call_with_timeout` (which ABANDONS a hung worker
#: thread), this one ends in `Popen.kill()` -- a subprocess genuinely can
#: be killed. Leaving ~2s of headroom below the outer 20.0s covers
#: candidate-gathering plus process spawn/teardown, so the kill fires (and
#: the child's CPU consumption actually stops) at or before the point the
#: agent is told the call failed, not sometime after.
_GREP_SUBPROCESS_TIMEOUT_SECONDS = 18.0


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
    that came from `Path.glob()` rather than directly from user input: an
    inline check like this one measures substantially cheaper per candidate
    than routing through `validate_path` (see the port report for numbers
    re-measured on this codebase). For a rule with no other behavioural
    difference in this context, that was worth avoiding.

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


def _sandbox_root_is_hidden(root: Path) -> bool:
    """Whether ``root``'s own final path component is dot-prefixed.

    Mirrors ``Utils.path_validation.validate_path``'s "hidden base
    directory" rejection (see that function's own comment): with
    ``[tools] file_sandbox_root`` itself dotted (e.g. ``~/.tldw_sandbox``),
    `read_file`/`write_file`/`list_directory` all refuse EVERY candidate,
    because each routes through `validate_path_multi` -> `validate_path`
    against this same root, and that check fires unconditionally once the
    root's own name starts with `.` -- independent of the candidate.
    `glob_files`/`grep_files` instead glob their roots directly and never
    pass through `validate_path` at all, so without this mirrored guard a
    dotted root INVERTS the hidden-file protection: the three siblings
    refuse everything while these two enumerate/read it normally
    (live-reproduced: `grep_files` returned a plain, non-hidden file's
    contents from inside a dotted root while `read_file` refused the
    identical path).

    TASK-850 generalizes this from a single sandbox root to the whole
    ``allowed_file_roots`` set: called once per root by
    ``GlobFiles.execute``/``GrepFiles.execute`` when building
    ``usable_roots``, so a dotted root is excluded from the search rather
    than checked only once against the sandbox. See those callers for how
    a dotted root's absence is handled when it leaves zero usable roots.

    Args:
        root: One of the resolved roots ``allowed_file_roots`` returned
            (the sandbox root, or one of the run's bound workspace
            folders) -- not necessarily the sandbox root itself.

    Returns:
        True if ``root``'s final path component starts with ``.``.
    """
    return root.name.startswith(".")


def _iter_candidates_across_roots(
    pattern: str,
    roots: tuple[Path, ...],
    sensitive_ctx: SensitivePathContext,
) -> Iterator[Path]:
    """Yield validated file candidates across every allowed root, once each.

    Shared by ``GlobFiles.execute`` and ``GrepFiles.execute`` (TASK-850):
    both previously globbed the tool sandbox root only, which was strictly
    narrower than -- and inconsistent with -- `read_file`/`write_file`/
    `list_directory`, all three of which already honour every root
    ``allowed_file_roots`` returns (the sandbox plus any workspace folder
    bound to the run). They now search the SAME root set, so an agent can
    no longer read a file by path that it cannot find by search.

    Every existing guard applies to every candidate from every root, not
    just the first: containment (``is_within``, which also applies the
    sensitive-path denylist) and the hidden-component rule
    (``_is_hidden_within``, mirroring ``validate_path``'s dotfile
    refusal). Callers must pre-filter ``roots`` down to the ones that pass
    the dotted-root rule (``_sandbox_root_is_hidden``) themselves -- this
    function assumes every entry in ``roots`` already qualifies; see
    ``GlobFiles.execute``/``GrepFiles.execute`` for where that filtering
    (and the "refuse the whole call only if none survive" decision)
    happens.

    ``_MAX_CANDIDATES`` bounds the TOTAL number of candidates pulled from
    the underlying ``Path.glob()`` iterators across ALL roots COMBINED,
    never per root -- otherwise N configured roots would multiply the
    worst-case walk by N. A candidate reachable through more than one root
    (e.g. a bound workspace folder that happens to nest the sandbox, or
    two bound folders that overlap) is yielded at most once, deduplicated
    by resolved identity.

    Args:
        pattern: A glob pattern, already checked by ``_rejects_traversal``.
        roots: Roots to search, in priority order. Every entry must
            already have passed ``_sandbox_root_is_hidden`` (i.e. none is
            itself dot-prefixed) -- this function does not check that
            again.
        sensitive_ctx: A ``SensitivePathContext`` resolved ONCE by the
            caller for this whole invocation (see
            ``Utils.sensitive_paths.resolve_sensitive_context``) and
            reused for every candidate from every root here -- never
            re-resolved per root or per candidate.

    Yields:
        Each qualifying candidate ``Path`` (as returned by ``glob()``,
        not pre-resolved), at most ``_MAX_CANDIDATES`` total across every
        root and never repeated.

    Raises:
        ValueError: The pattern is syntactically invalid for the root
            currently being searched. ``Path.glob()`` validates lazily --
            on the first ``next()`` pulled from it, not at construction --
            so this can surface from a ``next()`` call deep inside
            iteration; it propagates straight out of this generator so
            callers can distinguish a bad pattern from a legitimate empty
            result.
        NotImplementedError: Same lazy-validation timing, for a pattern
            form ``pathlib`` does not support.
    """
    seen_resolved: set[Path] = set()
    examined = 0
    for root in roots:
        if examined >= _MAX_CANDIDATES:
            return
        root_resolved = root.resolve()
        candidates = root.glob(pattern)
        while examined < _MAX_CANDIDATES:
            try:
                path = next(candidates)
            except StopIteration:
                break
            examined += 1
            if not path.is_file() or not is_within(path, root, context=sensitive_ctx):
                continue
            # A dotfile/dotdir must be invisible here even though it
            # passed `is_within` -- see `_is_hidden_within`.
            try:
                resolved = path.resolve()
            except (OSError, RuntimeError):
                continue
            if _is_hidden_within(resolved, root_resolved):
                continue
            if resolved in seen_resolved:
                continue
            seen_resolved.add(resolved)
            yield path


class GlobFiles(Tool):
    """`glob_files` -- path-pattern search across the sandbox and workspace roots."""

    @property
    def name(self) -> str:
        return "glob_files"

    @property
    def description(self) -> str:
        return (
            "Find files by path pattern inside the tool sandbox and any "
            "bound workspace folders. Supports glob syntax including ** "
            "for recursive matches, e.g. '**/*.py'. "
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

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Enumerating sandbox paths by pattern is a disclosure risk."""
        return ("reads",)

    async def execute(self, **kwargs) -> dict:
        """Search for files by glob pattern across every allowed root.

        TASK-850: previously globbed the tool sandbox root only -- strictly
        narrower than, and inconsistent with, `read_file`/`write_file`/
        `list_directory`, which already honour every root
        `allowed_file_roots` returns (the sandbox plus any workspace
        folder bound to the current run). Now searches that SAME root
        set: each root is globbed and results are merged, with
        `_MAX_CANDIDATES`/`_MAX_MATCHES` still applied GLOBALLY across all
        roots combined, not per root -- so N configured roots do not
        multiply the worst-case walk by N (see
        `_iter_candidates_across_roots`).

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
            roots = allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root())
            # Dotted-root rule, extended to a ROOT SET (TASK-850): with a
            # single root (the sandbox alone, the sandbox-only
            # configuration), a dotted root refuses the WHOLE call, exactly
            # as before -- see _sandbox_root_is_hidden. With several roots,
            # each is checked independently here and a dotted one is simply
            # excluded from `usable_roots` rather than failing every other,
            # still-valid root's results too; the call is refused outright
            # only when NONE survive that filter (which is exactly the
            # single-dotted-root case, so sandbox-only behavior is
            # unchanged).
            usable_roots = tuple(
                root for root in roots if not _sandbox_root_is_hidden(root)
            )
            if not usable_roots:
                return {"error": "Access to hidden files/directories is not allowed"}
            # Resolved ONCE for this call and reused for every candidate
            # from every root, rather than letting `is_within` ->
            # `is_sensitive_path` re-resolve the sensitive-path set (11
            # config accessors) per candidate -- see
            # Utils.sensitive_paths.resolve_sensitive_context.
            sensitive_ctx = resolve_sensitive_context()
            matches: list[str] = []
            try:
                for path in _iter_candidates_across_roots(
                    pattern, usable_roots, sensitive_ctx
                ):
                    matches.append(str(path))
                    if len(matches) >= _MAX_MATCHES:
                        break
            except (ValueError, NotImplementedError) as exc:
                return {"error": f"invalid pattern: {exc}"}
            return {"matches": sorted(matches)}
        except OSError as exc:
            return {"error": f"sandbox root is not usable: {exc}"}
        except Exception as exc:
            # Same never-raise contract as read_file/write_file/
            # list_directory's own outer catch-all (finding 6, substrate
            # review): without this, an unanticipated exception -- e.g.
            # `Path.expanduser()`'s `RuntimeError` when HOME can't be
            # determined -- would escape `execute()` entirely, relying
            # solely on `BuiltinToolProvider.invoke`'s own catch-all to
            # keep the run alive.
            logger.error(f"Error globbing pattern {pattern!r}: {exc}")
            return {"error": f"Failed to glob files: {exc}"}


#: Absolute path to the standalone worker module `_run_grep_subprocess`
#: spawns (TASK-843). Resolved relative to this file (not a package import)
#: so it works identically whether `tldw_chatbook` is installed editable
#: or as a built wheel -- see `_grep_worker.py`'s own docstring for why it
#: is a plain script with no import of this package at all.
_GREP_WORKER_SCRIPT = str(Path(__file__).with_name("_grep_worker.py"))


def _run_grep_subprocess(
    pattern: str,
    file_paths: list[str],
    *,
    max_matches: int,
    max_line_search_chars: int,
    max_lines_scanned: int,
    max_file_bytes: int,
    timeout_seconds: float,
) -> dict:
    """Run the regex-vs-file-content search in a killable child process.

    TASK-843: completes the catastrophic-backtracking mitigation
    `_MAX_GREP_LINE_SEARCH_CHARS`/`_MAX_GREP_LINES_SCANNED`/
    `GrepFiles.timeout_seconds` left open by themselves. Those bound the
    WORST CASE to a small, finite amount of work, but none of them can
    stop a pathological pattern's CPU burn once it starts: Python's `re`
    has no match timeout, and `Agents/agent_service.py`'s
    `_call_with_timeout` (the run loop's own per-tool-call ceiling)
    ABANDONS its worker thread on timeout rather than killing it --
    Python cannot forcibly kill a thread, so that CPU burn outlives the
    agent's own timeout report, and repeated calls accumulate. A
    subprocess is different: `Popen.kill()` sends SIGKILL (POSIX) /
    calls `TerminateProcess` (Windows), and the OS enforces that
    unconditionally, regardless of what the process is doing.

    Why a subprocess rather than the third-party `regex` module (which
    supports `timeout=`): `regex` is not a direct OR declared optional
    dependency of this project (checked against `pyproject.toml`) -- it is
    only present at all, transitively, through unrelated optional extras
    (`nltk`/`transformers`/`dateparser`, pulled in by the RAG/embeddings
    extras). `grep_files` is a core built-in, reachable with no extras
    installed; depending on `regex` here would make a currently-optional,
    transitive package a hard requirement for the base install. A
    subprocess adds no new dependency and works with the stdlib `re`
    already in use.

    This function is a BLOCKING call (`Popen.communicate(timeout=...)`);
    `GrepFiles.execute` runs it via `asyncio.to_thread` rather than
    awaiting it directly, so it never blocks the event loop.

    What this DOES guarantee: once `timeout_seconds` elapses, the child
    process is killed and its CPU consumption stops -- the search cannot
    keep burning CPU past that deadline the way the in-process,
    thread-based mitigation could. The worker also self-limits its own
    CPU time via `resource.setrlimit(RLIMIT_CPU, ...)` on POSIX (see
    `_grep_worker.py`), so an ORPHANED worker -- e.g. if this process
    itself were killed before ever reaching the `kill()` call below --
    still self-terminates rather than running unbounded.

    What this does NOT guarantee: the child can still burn real CPU for
    up to `timeout_seconds` before it is killed (down from unbounded
    before this task, but not zero), and every `grep_files` call now pays
    a small, fixed process spawn/teardown cost (~15-20ms locally with the
    `-S` flag below, negligible against the ~18s ceiling) whether the
    pattern is pathological or not. `RLIMIT_CPU` is POSIX-only; the
    `communicate(timeout=)` + `kill()` path is the guarantee that holds on
    every platform, including Windows.

    Args:
        pattern: The regular expression to search for. `GrepFiles.execute`
            already validates (compiles) it before ever calling this
            function, so a malformed pattern should never reach here --
            the worker re-validates anyway rather than trusting that.
        file_paths: Absolute paths to search, in order. Every path here
            MUST already be fully validated by the caller (containment,
            the sensitive-path denylist, the hidden-component rule) --
            neither this function nor the worker it spawns performs any
            of that validation, and must never be handed a path the
            caller has not already cleared.
        max_matches: Same bound as `_MAX_MATCHES` -- stop once this many
            matches are found.
        max_line_search_chars: Same bound as `_MAX_GREP_LINE_SEARCH_CHARS`.
        max_lines_scanned: Same bound as `_MAX_GREP_LINES_SCANNED`.
        max_file_bytes: Same bound as `_MAX_GREP_FILE_BYTES`.
        timeout_seconds: Wall-clock ceiling for the whole subprocess call
            (typically `_GREP_SUBPROCESS_TIMEOUT_SECONDS`).

    Returns:
        Dict with a `matches` list of `{path, line_number, line}` dicts on
        success, or an `error` string -- never raises.
    """
    request = json.dumps(
        {
            "pattern": pattern,
            "file_paths": file_paths,
            "max_matches": max_matches,
            "max_line_search_chars": max_line_search_chars,
            "max_lines_scanned": max_lines_scanned,
            "max_file_bytes": max_file_bytes,
        }
    )
    try:
        proc = subprocess.Popen(
            # "-S": skip `site` initialization. The worker imports only
            # stdlib (`json`/`re`/`sys`/`pathlib`, optionally `resource`),
            # none of which needs site-packages, and this measurably
            # shrinks interpreter startup (~15ms vs ~20ms, locally) paid
            # on every single grep_files call.
            [sys.executable, "-S", _GREP_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return {"error": f"could not start grep worker process: {exc}"}

    try:
        stdout, stderr = proc.communicate(input=request, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # The ONE line that actually bounds a pathological pattern's CPU
        # exposure past this call's return: a thread cannot be killed, a
        # process can.
        proc.kill()
        try:
            proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.error(f"grep worker (pid {proc.pid}) did not exit even after SIGKILL")
        return {
            "error": (
                f"grep search timed out after {timeout_seconds:g}s and was "
                "terminated"
            )
        }

    if proc.returncode != 0:
        return {
            "error": (
                f"grep worker failed (exit {proc.returncode}): "
                f"{(stderr or '').strip()[:500]}"
            )
        }
    try:
        parsed = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return {"error": "grep worker produced malformed output"}
    if not isinstance(parsed, dict):
        return {"error": "grep worker produced malformed output"}
    return parsed


class GrepFiles(Tool):
    """`grep_files` -- content search across the sandbox and workspace roots."""

    @property
    def name(self) -> str:
        return "grep_files"

    @property
    def description(self) -> str:
        return (
            "Search file contents by regular expression inside the tool "
            "sandbox and any bound workspace folders, optionally narrowed "
            "by a path glob. Returns matching lines with their file and "
            f"line number, at most {_MAX_MATCHES}."
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

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Reading arbitrary sandbox file contents is a disclosure risk."""
        return ("reads",)

    @property
    def timeout_seconds(self) -> float:
        """Overall per-call ceiling for the run loop's own accounting.

        A legitimate search over the candidate bound (`_MAX_CANDIDATES`
        files, `_MAX_GREP_LINES_SCANNED` lines total) is comfortably fast:
        locally measured, 200,000 lines against a realistic non-pathological
        pattern complete in well under a second, plus the search
        subprocess's own small, fixed spawn cost (TASK-843; see
        `_run_grep_subprocess`). 20s leaves generous headroom above that
        for a slower disk or a loaded system, while still being far
        tighter than the run's own default (`RunBudget.max_tool_call_
        seconds`, 300s at defaults) -- so a pathological call is reported
        back to the agent as timed out much sooner.

        Unlike before TASK-843, a pathological pattern's CPU burn no
        longer keeps running unbounded past this deadline: the actual
        search now runs in a subprocess bounded by its OWN, shorter
        internal ceiling (`_GREP_SUBPROCESS_TIMEOUT_SECONDS`), which ends
        in `Popen.kill()` rather than `_call_with_timeout`
        (`Agents/agent_service.py`) abandoning a worker thread it cannot
        actually kill. This property's value still governs when the run
        loop itself gives up and reports failure; see
        `_run_grep_subprocess`'s docstring for exactly what does and does
        not stop once that happens.

        Returns:
            20.0 seconds.
        """
        return 20.0

    async def execute(self, **kwargs) -> dict:
        """Search file contents across every allowed root by regular expression.

        TASK-850: previously searched the tool sandbox root only; now
        searches every root `allowed_file_roots` returns (the sandbox plus
        any workspace folder bound to the run), the same root set
        `read_file`/`write_file`/`list_directory` already honour --
        merged via `_iter_candidates_across_roots`, with
        `_MAX_CANDIDATES` applied globally across all roots, not per root.

        TASK-843: the supplied `pattern` is compiled with Python's `re`,
        which has no match timeout. Each candidate file's content is only
        ever searched a length-capped slice at a time
        (`_MAX_GREP_LINE_SEARCH_CHARS`), and the total number of lines
        read across the whole call is capped
        (`_MAX_GREP_LINES_SCANNED`) -- both bound the WORST CASE to a
        small, finite amount of work, and remain in place as the cheap
        first line of defence. What actually stops a pathological
        pattern's CPU burn from continuing after this call returns is
        that the search itself now runs in a separate, killable
        subprocess (`_run_grep_subprocess`) rather than in this process --
        see that function's docstring for exactly what it does and does
        not guarantee.

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
            re.compile(raw_pattern)
        except re.error as exc:
            return {"error": f"invalid regular expression: {exc}"}

        try:
            glob_pattern = str(kwargs.get("glob") or "**/*")
            if _rejects_traversal(glob_pattern):
                return {"error": "glob must stay inside the sandbox root"}
            roots = allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root())
            # Dotted-root rule, extended to a ROOT SET (TASK-850) -- see the
            # matching comment in GlobFiles.execute.
            usable_roots = tuple(
                root for root in roots if not _sandbox_root_is_hidden(root)
            )
            if not usable_roots:
                return {"error": "Access to hidden files/directories is not allowed"}
            # Resolved ONCE for this call and reused for every candidate
            # from every root -- see the matching comment in
            # GlobFiles.execute above.
            sensitive_ctx = resolve_sensitive_context()

            # Candidate discovery (containment, sensitivity, hidden-component,
            # _MAX_CANDIDATES) happens HERE, in-process -- none of it runs the
            # user-supplied regex, so none of it needs the subprocess
            # boundary below. Only the actual content search does.
            candidate_paths: list[str] = []
            try:
                for path in _iter_candidates_across_roots(
                    glob_pattern, usable_roots, sensitive_ctx
                ):
                    candidate_paths.append(str(path))
            except (ValueError, NotImplementedError) as exc:
                return {"error": f"invalid glob: {exc}"}

            if not candidate_paths:
                return {"matches": []}

            result = await asyncio.to_thread(
                _run_grep_subprocess,
                raw_pattern,
                candidate_paths,
                max_matches=_MAX_MATCHES,
                max_line_search_chars=_MAX_GREP_LINE_SEARCH_CHARS,
                max_lines_scanned=_MAX_GREP_LINES_SCANNED,
                max_file_bytes=_MAX_GREP_FILE_BYTES,
                timeout_seconds=_GREP_SUBPROCESS_TIMEOUT_SECONDS,
            )
            if "error" in result:
                return result
            return {"matches": result.get("matches", [])}
        except OSError as exc:
            return {"error": f"sandbox root is not usable: {exc}"}
        except Exception as exc:
            # Same never-raise contract as read_file/write_file/
            # list_directory's own outer catch-all (finding 6, substrate
            # review): without this, an unanticipated exception -- e.g.
            # `Path.expanduser()`'s `RuntimeError` when HOME can't be
            # determined -- would escape `execute()` entirely, relying
            # solely on `BuiltinToolProvider.invoke`'s own catch-all to
            # keep the run alive.
            logger.error(f"Error grepping pattern {raw_pattern!r}: {exc}")
            return {"error": f"Failed to grep files: {exc}"}
