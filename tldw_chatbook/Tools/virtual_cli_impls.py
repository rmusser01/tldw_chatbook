"""Structured read-only virtual CLI over existing workspace and Git cores."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, cast

from .git_tool_impls import git_blame, git_branches, git_diff, git_log, git_status
from .local_tool_impls import (
    glob_files,
    grep_files,
    list_directory,
    read_file,
    stat_path,
)
from .workspace_tool_executor import WorkspaceToolExecutor

VIRTUAL_CLI_COMMANDS = (
    "ls",
    "cat",
    "grep",
    "find",
    "stat",
    "git_status",
    "git_diff",
    "git_log",
    "git_blame",
    "git_branches",
)
VirtualCliCommand = Literal[
    "ls",
    "cat",
    "grep",
    "find",
    "stat",
    "git_status",
    "git_diff",
    "git_log",
    "git_blame",
    "git_branches",
]

MAX_ARGV_ITEMS = 64
MAX_ARG_BYTES = 4 * 1024
MAX_ARGV_BYTES = 16 * 1024


class VirtualCliArgumentError(ValueError):
    """Invalid command or argv for the virtual CLI."""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise VirtualCliArgumentError(f"invalid arguments: {message}")


def _parser() -> _ArgumentParser:
    return _ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)


def _integer_at_least(minimum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("must be an integer") from exc
        if parsed < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return parsed

    return parse


def _build_parsers() -> dict[str, _ArgumentParser]:
    parsers: dict[str, _ArgumentParser] = {}

    ls_parser = _parser()
    ls_parser.add_argument("path", nargs="?", default=".")
    parsers["ls"] = ls_parser

    cat_parser = _parser()
    cat_parser.add_argument("path")
    cat_parser.add_argument("--offset", type=_integer_at_least(1), default=1)
    cat_parser.add_argument("--limit", type=_integer_at_least(0))
    parsers["cat"] = cat_parser

    grep_parser = _parser()
    grep_parser.add_argument("pattern")
    grep_parser.add_argument("--mode", choices=("content", "files", "count"), default="content")
    parsers["grep"] = grep_parser

    find_parser = _parser()
    find_parser.add_argument("pattern")
    parsers["find"] = find_parser

    stat_parser = _parser()
    stat_parser.add_argument("path")
    parsers["stat"] = stat_parser

    status_parser = _parser()
    status_parser.add_argument("path", nargs="?", default=".")
    parsers["git_status"] = status_parser

    diff_parser = _parser()
    diff_parser.add_argument("--staged", action="store_true")
    diff_parser.add_argument("--range", dest="commit_range")
    diff_parser.add_argument("--path")
    diff_parser.add_argument("--stat", action="store_true")
    parsers["git_diff"] = diff_parser

    log_parser = _parser()
    log_parser.add_argument("--count", type=_integer_at_least(1), default=20)
    log_parser.add_argument("--path")
    parsers["git_log"] = log_parser

    blame_parser = _parser()
    blame_parser.add_argument("path")
    blame_parser.add_argument("--start", type=_integer_at_least(1))
    blame_parser.add_argument("--end", type=_integer_at_least(1))
    parsers["git_blame"] = blame_parser

    parsers["git_branches"] = _parser()
    return parsers


_PARSERS = _build_parsers()


@dataclass(frozen=True, slots=True)
class VirtualCliRequest:
    """One validated virtual command invocation."""

    command: VirtualCliCommand
    argv: tuple[str, ...]


def validate_request(command: str, argv: Sequence[str]) -> VirtualCliRequest:
    """Validate the structured outer request without parsing shell text."""
    if not isinstance(command, str) or command not in VIRTUAL_CLI_COMMANDS:
        raise VirtualCliArgumentError(f"unknown virtual CLI command: {command!r}")
    if not isinstance(argv, (list, tuple)):
        raise VirtualCliArgumentError("argv must be an array of strings")
    if len(argv) > MAX_ARGV_ITEMS:
        raise VirtualCliArgumentError(f"argv exceeds {MAX_ARGV_ITEMS} items")
    total = 0
    normalized: list[str] = []
    for item in argv:
        if not isinstance(item, str):
            raise VirtualCliArgumentError("argv must contain only strings")
        if "\x00" in item:
            raise VirtualCliArgumentError("argv must not contain NUL")
        size = len(item.encode("utf-8"))
        if size > MAX_ARG_BYTES:
            raise VirtualCliArgumentError(f"argv item exceeds {MAX_ARG_BYTES} UTF-8 bytes")
        total += size
        normalized.append(item)
    if total > MAX_ARGV_BYTES:
        raise VirtualCliArgumentError(f"argv exceeds {MAX_ARGV_BYTES} UTF-8 bytes")
    return VirtualCliRequest(cast(VirtualCliCommand, command), tuple(normalized))


def parse_request(command: str, argv: Sequence[str]) -> tuple[VirtualCliRequest, argparse.Namespace]:
    """Validate the outer request and its command-specific argv grammar."""
    request = validate_request(command, argv)
    try:
        parsed = _PARSERS[request.command].parse_args(list(request.argv))
    except (argparse.ArgumentError, TypeError, ValueError) as exc:
        if isinstance(exc, VirtualCliArgumentError):
            raise
        raise VirtualCliArgumentError(f"invalid arguments: {exc}") from exc
    return request, parsed


class VirtualCliRegistry:
    """Parse fixed argv forms and dispatch through a pinned executor by default."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        workspace_executor: WorkspaceToolExecutor | None = None,
    ) -> None:
        self._root = Path(workspace_root).resolve()
        self._workspace_executor = workspace_executor

    @property
    def workspace_root(self) -> Path:
        return self._root

    def execute(self, command: str, argv: Sequence[str]) -> str:
        request, args = parse_request(command, argv)
        if self._workspace_executor is not None:
            operation, arguments = _leased_operation(request.command, args)
            return self._workspace_executor.execute(
                operation,
                arguments,
                intent="read",
            )

        if request.command == "ls":
            return list_directory(args.path, workspace_root=self._root)
        if request.command == "cat":
            return read_file(
                args.path,
                workspace_root=self._root,
                offset=args.offset,
                limit=args.limit,
            )
        if request.command == "grep":
            return grep_files(args.pattern, workspace_root=self._root, mode=args.mode)
        if request.command == "find":
            return glob_files(args.pattern, workspace_root=self._root)
        if request.command == "stat":
            return stat_path(args.path, workspace_root=self._root)
        if request.command == "git_status":
            return git_status(self._root, args.path)
        if request.command == "git_diff":
            return git_diff(
                self._root,
                staged=args.staged,
                commit_range=args.commit_range,
                path=args.path,
                stat=args.stat,
            )
        if request.command == "git_log":
            return git_log(self._root, count=args.count, path=args.path)
        if request.command == "git_blame":
            return git_blame(
                self._root,
                args.path,
                start_line=args.start,
                end_line=args.end,
            )
        return git_branches(self._root)


def _leased_operation(
    command: VirtualCliCommand,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    """Project one parsed virtual command onto the closed executor protocol."""
    if command == "ls":
        return "fs_list", {"path": args.path}
    if command == "cat":
        arguments = {"path": args.path, "offset": args.offset}
        if args.limit is not None:
            arguments["limit"] = args.limit
        return "fs_read", arguments
    if command == "grep":
        return "fs_grep", {"pattern": args.pattern, "mode": args.mode}
    if command == "find":
        return "fs_glob", {"pattern": args.pattern}
    if command == "stat":
        return "stat_path", {"path": args.path}
    if command == "git_status":
        return "git_status", {"path": args.path}
    if command == "git_diff":
        arguments = {"staged": args.staged, "stat": args.stat}
        if args.commit_range is not None:
            arguments["commit_range"] = args.commit_range
        if args.path is not None:
            arguments["path"] = args.path
        return "git_diff", arguments
    if command == "git_log":
        arguments = {"count": args.count}
        if args.path is not None:
            arguments["path"] = args.path
        return "git_log", arguments
    if command == "git_blame":
        arguments = {"path": args.path}
        if args.start is not None:
            arguments["start_line"] = args.start
        if args.end is not None:
            arguments["end_line"] = args.end
        return "git_blame", arguments
    return "git_branches", {}
