"""
File Operation Tools for LLM function calling.

These tools allow LLMs to perform safe file operations with proper validation.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PureWindowsPath
from typing import Dict, Any, Iterator, Mapping

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


#: Tool name -> (path argument key, write access required). Read/write mirror
#: exactly what each tool itself passes to `allowed_file_roots(write=...)`
#: below (see `ReadFileTool`/`ListDirectoryTool`/`WriteFileTool.execute`).
#: Consulted by `path_precheck_failed` (TASK-1231/F3 AC2) -- the approval
#: card's pre-flight check -- so the two never drift: if a tool's own
#: `write=` argument ever changes, this mapping must change with it.
_FILE_TOOL_PATH_ARGS: dict[str, tuple[str, bool]] = {
    "read_file": ("file_path", False),
    "list_directory": ("directory_path", False),
    "write_file": ("file_path", True),
}


def path_precheck_failed(tool_name: str, args: Mapping[str, Any] | None) -> bool:
    """Whether ``tool_name``'s path argument in ``args`` will fail the roots check.

    TASK-1231/F3 AC2: on an unbound (Default) workspace, an approved
    `read_file`/`list_directory`/`write_file` call used to fail invisibly
    AFTER the user already approved it -- the approval card had no way to
    know the call was doomed. This lets `console_chat_controller.
    build_tool_review_hook` pre-flight that same check at card-build time so
    the row can WARN the user instead. It is a warning only: the user can
    still approve, and the call then fails exactly as before with
    `validate_path_multi`'s own (now recovery-route-bearing) error --
    this function must never be used to auto-deny or otherwise gate
    dispatch. The real, authoritative enforcement remains `ReadFileTool`/
    `ListDirectoryTool`/`WriteFileTool.execute`'s own `validate_path_multi`
    call.

    Fails closed to ``False`` (no warning shown) on any unexpected error --
    a pre-flight check that itself breaks must never block or corrupt the
    approval flow; the tool's own enforcement at dispatch time is unaffected
    either way.

    Args:
        tool_name: The builtin tool's dispatch name (``ToolCall.name`` /
            ``MCPPendingCall.llm_name`` for a builtin row).
        args: The call's raw arguments, as the model supplied them.

    Returns:
        ``True`` only for a known file tool whose path argument is a
        non-empty string that `validate_path_multi` would currently reject
        against this run's `allowed_file_roots`. ``False`` for every other
        tool name (including every non-file builtin and every MCP tool), a
        missing/blank path argument, or an unexpected error while checking.
    """
    spec = _FILE_TOOL_PATH_ARGS.get(tool_name)
    if spec is None:
        return False
    arg_name, write = spec
    path_value = args.get(arg_name) if isinstance(args, Mapping) else None
    if not isinstance(path_value, str) or not path_value.strip():
        return False
    try:
        validate_path_multi(
            path_value,
            allowed_file_roots(write=write, sandbox_root=_tool_sandbox_root()),
        )
    except ValueError:
        return True
    except Exception:  # noqa: BLE001 -- a broken pre-flight must never break approval
        logger.opt(exception=True).warning(
            "path_precheck_failed: unexpected error checking {!r} for {!r}; "
            "not warning (the tool's own validate_path_multi still enforces "
            "this at dispatch time)",
            tool_name,
            path_value,
        )
        return False
    return False


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

#: Wall-clock ceiling for the ENTIRE grep search phase (TASK-843; widened
#: by the follow-up hardening review's Finding 1 -- see `_run_grep_search`).
#: Deliberately shorter than `GrepFiles.timeout_seconds` (20.0s): this is
#: the timeout that actually matters for bounding CPU, because unlike the
#: run loop's own thread-based `_call_with_timeout` (which ABANDONS a hung
#: worker thread), every subprocess spawned within this budget ends in
#: `Popen.kill()` -- a subprocess genuinely can be killed.
#:
#: What this covers changed with Finding 1: candidate discovery and the
#: regex search used to be sequential phases -- discovery ran to
#: completion, in-process, un-timed, THEN one subprocess call got this
#: whole budget. That let a slow discovery phase (a large, high-hit-rate
#: tree; see `_run_grep_search`'s docstring for the measured regression)
#: eat into time this constant never accounted for, pushing the REAL
#: worst-case wall-clock past `GrepFiles.timeout_seconds` and making the
#: "kill fires before the agent is told the call failed" guarantee false.
#: `_run_grep_search` now starts this deadline before pulling the first
#: candidate and re-derives the remaining budget before every subsequent
#: batch/subprocess call, so discovery time and every batch's subprocess
#: wait are drawn from the SAME 18.0s window rather than the former
#: (discovery) + 18.0s (search). Leaving ~2s of headroom below the outer
#: 20.0s still covers process spawn/teardown for whichever batch is in
#: flight when the deadline is reached, so the kill (or the timeout-error
#: return before ever starting another batch) fires at or before the
#: point the agent is told the call failed, not sometime after -- for the
#: aggregate of however many batches this call actually needed, not just
#: one.
_GREP_SUBPROCESS_TIMEOUT_SECONDS = 18.0

#: Initial number of candidates `_run_grep_search` pulls from
#: `_iter_candidates_across_roots` before the FIRST subprocess call
#: (Finding 1, follow-up hardening review). Chosen small enough that an
#: ordinary, high-hit-rate search's enumeration cost before that first
#: call stays close to what the old in-process, single-phase
#: implementation paid before its early-break -- at the reviewer's
#: measured ~0.32ms/candidate, 256 candidates costs ~82ms, comfortably in
#: the same ballpark as the ~0.1s the old early-break stopped at (~200
#: files). Doubles on each subsequent round, up to
#: `_GREP_MAX_CANDIDATE_BATCH_SIZE` -- see that constant for why a fixed
#: small size is not used throughout.
_GREP_INITIAL_CANDIDATE_BATCH_SIZE = 256

#: Ceiling `_run_grep_search`'s per-round batch size grows to (doubling
#: from `_GREP_INITIAL_CANDIDATE_BATCH_SIZE`) before it stops growing
#: further. Exists so a pattern with few or zero hits -- which must still
#: examine candidates up to `_MAX_CANDIDATES` (20,000) to confirm that --
#: does not pay for a large number of small, separately-spawned
#: subprocess calls to get there: capped at this size, covering
#: `_MAX_CANDIDATES` costs on the order of a handful of subprocess spawns
#: (doubling from 256: 256, 512, 1024, 2048, 4096, 4096, ... -- roughly 8
#: rounds to reach 20,000), not one call per few hundred candidates.
_GREP_MAX_CANDIDATE_BATCH_SIZE = 4096


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

    Finding 5 (follow-up hardening review): that global bound is consumed
    root-by-root IN ORDER, not split fairly across roots -- ``roots[0]``'s
    candidates are drained (up to ``_MAX_CANDIDATES``) before ``roots[1]``
    is ever globbed at all. A first root that alone holds
    ``_MAX_CANDIDATES`` or more matching entries therefore starves every
    later root completely, and the caller cannot distinguish that from
    those later roots genuinely containing no matches -- both look like
    "no results from root N" from the outside. This is judged correct,
    not a bug to fix: the bound must stay global (see above), and there is
    no principled way to divide a single global budget fairly across an
    arbitrary number of roots without either wasting it on roots with
    nothing to find or starving one that has plenty. A caller that needs a
    specific root's results reliably included should narrow the search
    (e.g. a ``glob``/``pattern`` scoped under that root specifically)
    rather than rely on root ordering.

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
#:
#: ``.resolve()`` is load-bearing, not cosmetic: ``__file__`` is not
#: guaranteed absolute under every loader, and the worker is spawned with
#: ``cwd=_GREP_WORKER_CWD`` (the temp dir) rather than inheriting the
#: parent's. A relative script path would therefore be resolved against the
#: temp dir and fail to open, breaking `grep_files` outright -- the cwd
#: hardening below turned a latent portability wart into a hard failure.
_GREP_WORKER_SCRIPT = str(Path(__file__).resolve().with_name("_grep_worker.py"))

#: Working directory for the grep worker subprocess (Finding 3, follow-up
#: hardening review). Every path the worker ever touches is one of the
#: already-fully-resolved, ABSOLUTE candidates `_iter_candidates_across_roots`
#: yielded (every root -- the sandbox and every bound workspace folder --
#: is itself absolute; see `_tool_sandbox_root`/`allowed_file_roots`), so
#: the worker has no legitimate use for the PARENT's actual working
#: directory. Inheriting it for free (the default when `cwd=` is omitted)
#: would still expose that directory's identity to a process whose only
#: job is running a possibly-adversarial regex, for no benefit -- a
#: neutral, always-present directory costs nothing to pass explicitly
#: instead.
_GREP_WORKER_CWD = tempfile.gettempdir()


def _grep_worker_env() -> Dict[str, str]:
    """Minimal environment for the grep worker subprocess (Finding 3).

    Inheriting the parent's full environment (the default when `env=` is
    omitted from `subprocess.Popen`) costs nothing under this worker's
    existing trust model -- it only ever runs a regex against paths the
    PARENT has already fully validated, and a module shadow-attack via a
    leaked `sys.path` entry would require source-tree write access, i.e.
    already-complete compromise. It is still a needless surface: a probe
    confirmed a secret set in the parent's environment (e.g.
    `MY_FAKE_API_KEY=sk-...`) is visible verbatim inside the child. The
    worker imports nothing beyond the standard library and never shells
    out, so it needs none of the parent's environment to do its job.

    Returns:
        A dict containing `PATH` (needed for the interpreter's own
        startup/library resolution, not for anything the worker itself
        invokes) plus, on Windows only, `SystemRoot` -- `subprocess.Popen`
        with a genuinely empty `env={}` can fail outright on Windows,
        since several CRT/socket-initialization paths read it
        unconditionally regardless of what the child program needs.
    """
    env: Dict[str, str] = {}
    path = os.environ.get("PATH")
    if path:
        env["PATH"] = path
    if sys.platform == "win32":
        system_root = os.environ.get("SystemRoot")
        if system_root:
            env["SystemRoot"] = system_root
    return env


def _validated_grep_worker_payload(parsed: dict) -> dict:
    """Validate the shape of a successfully-parsed worker JSON payload.

    Finding 4 (follow-up hardening review): `_run_grep_subprocess`
    previously returned `parsed` straight from `json.loads(stdout)` once
    it was confirmed to be *a dict* -- but never confirmed `matches` was
    actually a list, or that `lines_scanned` was actually an int. A worker
    emitting `{"matches": "not-a-list"}` (a bug, not plausibly an attack:
    both ends of this pipe -- `_grep_worker.py` and this module -- are
    code this project owns) would propagate that shape straight through
    to `GrepFiles.execute` and on to the agent, instead of the documented
    `{"path": str, "line_number": int, "line": str}` per-match shape.

    Args:
        parsed: A JSON value already confirmed to be a `dict` (see the
            caller), with no `"error"` key -- i.e. the worker's claimed
            success path.

    Returns:
        `parsed` unchanged if `matches` is a list of well-formed
        `{"path": str, "line_number": int, "line": str}` dicts and
        `lines_scanned` is an int; otherwise a fresh
        `{"error": "grep worker produced malformed output"}` dict --
        never raises.
    """
    matches = parsed.get("matches")
    if not isinstance(matches, list):
        return {"error": "grep worker produced malformed output"}
    for entry in matches:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("line_number"), int)
            or not isinstance(entry.get("line"), str)
        ):
            return {"error": "grep worker produced malformed output"}
    if not isinstance(parsed.get("lines_scanned"), int):
        return {"error": "grep worker produced malformed output"}
    return parsed


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
    `-S`/`-P` flags below, negligible against the ~18s ceiling) whether
    the pattern is pathological or not. `RLIMIT_CPU` is POSIX-only; the
    `communicate(timeout=)` + `kill()` path is the guarantee that holds on
    every platform, including Windows.

    Note (Finding 1, follow-up hardening review): `GrepFiles.execute` no
    longer calls this once with every candidate -- `_run_grep_search`
    calls it once per BATCH, with a shrinking `timeout_seconds` and a
    shrinking `max_matches`/`max_lines_scanned` reflecting the budget
    already spent by earlier batches. Nothing about this function's own
    guarantee changes: every individual call it makes is still bounded
    and killable exactly as documented above, independent of how many
    times it is called in one `grep_files` invocation.

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
        max_matches: Stop once this many matches are found from THIS
            batch -- the caller (`_run_grep_search`) passes the REMAINING
            budget out of `_MAX_MATCHES`, not always the same constant.
        max_line_search_chars: Same bound as `_MAX_GREP_LINE_SEARCH_CHARS`.
        max_lines_scanned: Same per-batch-remaining-budget treatment as
            `max_matches`, out of `_MAX_GREP_LINES_SCANNED`.
        max_file_bytes: Same bound as `_MAX_GREP_FILE_BYTES`.
        timeout_seconds: Wall-clock ceiling for THIS subprocess call --
            the caller passes whatever remains of the overall search
            deadline (see `_GREP_SUBPROCESS_TIMEOUT_SECONDS`), not always
            that same constant.

    Returns:
        Dict with a `matches` list of `{path, line_number, line}` dicts
        and a `lines_scanned` int on success, or an `error` string --
        never raises. The success shape is validated (Finding 4; see
        `_validated_grep_worker_payload`) before being returned, so a
        worker emitting a malformed payload (e.g. `matches` not actually a
        list) surfaces as an error dict here rather than propagating that
        shape onward.
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
            # "-P" (Finding 3, follow-up hardening review; Python >=3.11,
            # matching this project's floor): don't prepend the script's
            # own directory to `sys.path`. Without it, a probe confirmed
            # `sys.path[0]` inside the worker is `Tools/` -- this
            # project's own source directory -- for no reason the worker
            # needs; planting a shadow module there to exploit that would
            # already require source-tree write access (full compromise
            # on its own), so this is not an escalation, just a needless
            # surface removed for free.
            [sys.executable, "-S", "-P", _GREP_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Explicit `cwd=`/`env=` (Finding 3): see `_GREP_WORKER_CWD`/
            # `_grep_worker_env` for why the worker needs neither the
            # parent's actual working directory nor its environment.
            cwd=_GREP_WORKER_CWD,
            env=_grep_worker_env(),
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
    if "error" in parsed:
        error_value = parsed.get("error")
        if not isinstance(error_value, str):
            return {"error": "grep worker produced malformed output"}
        return parsed
    return _validated_grep_worker_payload(parsed)


async def _run_grep_search(
    pattern: str,
    candidates: Iterator[Path],
    *,
    max_matches: int,
    max_line_search_chars: int,
    max_lines_scanned: int,
    max_file_bytes: int,
    deadline_seconds: float,
) -> dict:
    """Search ``candidates`` for ``pattern``, streaming batches to killable subprocesses.

    Finding 1 (follow-up hardening review). TASK-843 moved the regex
    search into a killable child process, but `GrepFiles.execute` still
    drained `candidates` (bounded by `_MAX_CANDIDATES`, up to 20,000)
    all the way to completion, in-process, BEFORE ever spawning that
    subprocess -- undoing the early-break the pre-subprocess implementation
    had (`len(matches) >= _MAX_MATCHES` / `lines_scanned >=
    _MAX_GREP_LINES_SCANNED`, checked during enumeration itself). That
    made an ordinary, HIGH-HIT-RATE search over a large tree pay for every
    candidate the match/line budget never needed just to reach the point
    of spawning a subprocess at all -- measured: a 5,000-file tree with a
    pattern matching every file went from the old ~0.1s (in-process,
    early-broken after ~200 files) to ~1.6s (all 5,000 candidates
    enumerated first).

    This function restores that early exit without giving up the
    subprocess boundary TASK-843 added: it pulls candidates from
    ``candidates`` (already fully validated -- containment, the
    sensitive-path denylist, the hidden-component rule, and
    ``_MAX_CANDIDATES`` -- by ``_iter_candidates_across_roots``) in
    GROWING batches, starting at ``_GREP_INITIAL_CANDIDATE_BATCH_SIZE`` and
    doubling up to ``_GREP_MAX_CANDIDATE_BATCH_SIZE``, running each batch
    through its OWN call to ``_run_grep_subprocess`` with whatever
    match/line/time budget remains, and stopping -- WITHOUT pulling
    another candidate -- the moment that budget is satisfied. A small
    initial batch keeps a high-hit-rate search's pre-first-call
    enumeration cost close to what the old in-process early-break paid;
    growing it on later rounds keeps a rare/zero-hit search (which must
    still examine candidates up to ``_MAX_CANDIDATES`` to confirm that)
    from paying for a large number of small, separately-spawned
    subprocesses to get there.

    Killability is unchanged: every batch is still searched by its own
    fully killable child process (see ``_run_grep_subprocess``) -- this
    function only changes HOW MANY candidates are handed to it and in how
    many calls, never how a batch is searched once handed off. A
    catastrophic-backtracking pattern inside any one batch is bounded and
    killed exactly as before, independent of batching.

    The wall-clock deadline (``deadline_seconds``) starts the moment this
    function is entered -- i.e. BEFORE the first candidate is even pulled,
    not after candidate discovery finishes. This is what makes
    ``_GREP_SUBPROCESS_TIMEOUT_SECONDS``'s ~2s headroom below
    ``GrepFiles.timeout_seconds`` an honest bound again: discovery time
    and every batch's subprocess wait are drawn from the SAME window, so
    the aggregate -- however it is spent across batches -- cannot exceed
    it. The deadline is re-checked before pulling each batch and again
    before spawning each subprocess call; between those checks, pulling
    one batch of up to the current (capped) batch size from an
    already-validated iterator is itself bounded, so the worst-case
    overshoot before a check catches it is small and fixed, not unbounded.

    Args:
        pattern: The already-`re.compile`-checked pattern to search for.
        candidates: Iterator of already-validated candidate paths (see
            ``_iter_candidates_across_roots``). This function performs
            NONE of that validation itself and trusts every path it pulls
            completely.
        max_matches: Same bound as ``_MAX_MATCHES`` -- the TOTAL across
            every batch, not per batch.
        max_line_search_chars: Same bound as ``_MAX_GREP_LINE_SEARCH_CHARS``.
        max_lines_scanned: Same bound as ``_MAX_GREP_LINES_SCANNED`` --
            the TOTAL across every batch, not per batch.
        max_file_bytes: Same bound as ``_MAX_GREP_FILE_BYTES``.
        deadline_seconds: Wall-clock budget for this ENTIRE search phase
            -- candidate discovery interleaved with every batch's
            subprocess call, together -- starting now, not after
            discovery. Typically ``_GREP_SUBPROCESS_TIMEOUT_SECONDS``.

    Returns:
        Dict with a `matches` list (capped at `max_matches`) on success,
        or an `error` string -- never raises for a worker-level failure
        (see ``_run_grep_subprocess``). A syntactically invalid glob
        pattern can still raise ``ValueError``/``NotImplementedError`` out
        of ``next(candidates)`` (``Path.glob()``'s lazy validation) --
        exactly as before this function existed -- and callers must still
        handle that themselves.
    """
    deadline = time.monotonic() + deadline_seconds
    all_matches: list[dict] = []
    lines_scanned_total = 0
    batch_size = _GREP_INITIAL_CANDIDATE_BATCH_SIZE

    while len(all_matches) < max_matches and lines_scanned_total < max_lines_scanned:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "error": (
                    f"grep search timed out after {deadline_seconds:g}s and "
                    "was terminated"
                )
            }

        batch: list[str] = []
        for _ in range(batch_size):
            try:
                batch.append(str(next(candidates)))
            except StopIteration:
                break
        if not batch:
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "error": (
                    f"grep search timed out after {deadline_seconds:g}s and "
                    "was terminated"
                )
            }

        result = await asyncio.to_thread(
            _run_grep_subprocess,
            pattern,
            batch,
            max_matches=max_matches - len(all_matches),
            max_line_search_chars=max_line_search_chars,
            max_lines_scanned=max_lines_scanned - lines_scanned_total,
            max_file_bytes=max_file_bytes,
            timeout_seconds=remaining,
        )
        if "error" in result:
            return result
        all_matches.extend(result["matches"])
        lines_scanned_total += result["lines_scanned"]

        if len(batch) < batch_size:
            # The iterator ran out mid-batch -- no candidates remain, so
            # there is no point looping again only to find that out.
            break
        batch_size = min(batch_size * 2, _GREP_MAX_CANDIDATE_BATCH_SIZE)

    return {"matches": all_matches[:max_matches]}


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
        pattern complete in well under a second, plus however many search
        subprocesses that takes (TASK-843/Finding 1; see
        `_run_grep_search`) -- each with its own small, fixed spawn cost
        (`_run_grep_subprocess`). 20s leaves generous headroom above that
        for a slower disk or a loaded system, while still being far
        tighter than the run's own default (`RunBudget.max_tool_call_
        seconds`, 300s at defaults) -- so a pathological call is reported
        back to the agent as timed out much sooner.

        Unlike before TASK-843, a pathological pattern's CPU burn no
        longer keeps running unbounded past this deadline: the actual
        search now runs in one or more subprocesses (`_run_grep_search`
        batches candidate discovery and the search together; see its
        docstring), bounded by their OWN, shorter internal ceiling
        (`_GREP_SUBPROCESS_TIMEOUT_SECONDS`, counted from before the first
        candidate is even pulled), which ends in `Popen.kill()` rather
        than `_call_with_timeout` (`Agents/agent_service.py`) abandoning a
        worker thread it cannot actually kill. This property's value
        still governs when the run loop itself gives up and reports
        failure; see `_run_grep_subprocess`'s docstring for exactly what
        does and does not stop once that happens.

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
        that the search itself now runs in one or more separate, killable
        subprocesses (`_run_grep_subprocess`) rather than in this process --
        see that function's docstring for exactly what it does and does
        not guarantee.

        Finding 1 (follow-up hardening review): candidate discovery
        (`_iter_candidates_across_roots`) and the search are STREAMED
        together via `_run_grep_search` rather than fully separated into
        "discover everything, then search everything" -- draining
        discovery all the way to `_MAX_CANDIDATES` before ever spawning a
        subprocess made an ordinary, high-hit-rate search over a large
        tree pay for candidates the match budget never needed. See
        `_run_grep_search`'s docstring for exactly how batching restores
        that early exit without giving up killability.

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
            # _MAX_CANDIDATES) and the search are STREAMED together, in
            # growing batches, by `_run_grep_search` (Finding 1, follow-up
            # hardening review) -- neither discovery nor the deadline that
            # bounds it waits for the other to fully finish first. See that
            # function's docstring for exactly why and how.
            candidates = _iter_candidates_across_roots(
                glob_pattern, usable_roots, sensitive_ctx
            )
            try:
                return await _run_grep_search(
                    raw_pattern,
                    candidates,
                    max_matches=_MAX_MATCHES,
                    max_line_search_chars=_MAX_GREP_LINE_SEARCH_CHARS,
                    max_lines_scanned=_MAX_GREP_LINES_SCANNED,
                    max_file_bytes=_MAX_GREP_FILE_BYTES,
                    deadline_seconds=_GREP_SUBPROCESS_TIMEOUT_SECONDS,
                )
            except (ValueError, NotImplementedError) as exc:
                return {"error": f"invalid glob: {exc}"}
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
