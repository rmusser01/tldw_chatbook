from pathlib import Path

import pytest

from tldw_chatbook.Tools import virtual_cli_impls
from tldw_chatbook.Tools.local_tool_impls import LocalToolError


def _registry(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return virtual_cli_impls.VirtualCliRegistry(workspace), workspace


def test_virtual_cli_command_set_is_fixed_and_read_only():
    assert virtual_cli_impls.VIRTUAL_CLI_COMMANDS == (
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


@pytest.mark.parametrize(
    ("command", "argv", "expected"),
    (
        ("ls", ["docs"], ("list_directory", "docs")),
        ("cat", ["a.txt", "--offset", "2", "--limit", "4"], ("read_file", "a.txt", 2, 4)),
        ("grep", ["needle", "--mode", "files"], ("grep_files", "needle", "files")),
        ("find", ["**/*.py"], ("glob_files", "**/*.py")),
        ("stat", ["a.txt"], ("stat_path", "a.txt")),
        ("git_status", ["src"], ("git_status", "src")),
        (
            "git_diff",
            ["--staged", "--range", "HEAD~1..HEAD", "--path", "a.py", "--stat"],
            ("git_diff", True, "HEAD~1..HEAD", "a.py", True),
        ),
        ("git_log", ["--count", "7", "--path", "src"], ("git_log", 7, "src")),
        ("git_blame", ["a.py", "--start", "2", "--end", "5"], ("git_blame", "a.py", 2, 5)),
        ("git_branches", [], ("git_branches",)),
    ),
)
def test_virtual_cli_dispatches_each_command_directly(
    tmp_path, monkeypatch, command, argv, expected
):
    registry, _workspace = _registry(tmp_path)
    seen = []

    def fake(name):
        def call(*args, **kwargs):
            normalized = tuple(value for value in args if not isinstance(value, Path))
            normalized += tuple(
                kwargs[key]
                for key in (
                    "offset",
                    "limit",
                    "mode",
                    "staged",
                    "commit_range",
                    "count",
                    "path",
                    "stat",
                    "start_line",
                    "end_line",
                )
                if key in kwargs
            )
            seen.append((name, *normalized))
            return name

        return call

    for name in (
        "list_directory",
        "read_file",
        "grep_files",
        "glob_files",
        "stat_path",
        "git_status",
        "git_diff",
        "git_log",
        "git_blame",
        "git_branches",
    ):
        monkeypatch.setattr(virtual_cli_impls, name, fake(name))

    assert registry.execute(command, argv) == expected[0]
    assert seen == [expected]


@pytest.mark.parametrize(
    ("command", "argv"),
    (
        ("rm", []),
        ("cat", []),
        ("cat", ["a", "b"]),
        ("ls", [".", "extra"]),
        ("grep", ["x", "--mo", "files"]),
        ("grep", ["x", "--mode", "unknown"]),
        ("find", []),
        ("stat", ["a", "extra"]),
        ("git_status", [".", "extra"]),
        ("git_diff", ["--unknown"]),
        ("git_log", ["--count", "zero"]),
        ("git_blame", ["a.py", "--start", "0"]),
        ("git_branches", ["extra"]),
    ),
)
def test_virtual_cli_rejects_unknown_commands_and_argv_forms(tmp_path, command, argv):
    registry, _workspace = _registry(tmp_path)
    with pytest.raises(virtual_cli_impls.VirtualCliArgumentError):
        registry.execute(command, argv)


@pytest.mark.parametrize(
    "argv",
    (
        "a.txt",
        [1],
        ["x"] * 65,
        ["x" * 4097],
        ["x" * 4096] * 5,
        ["bad\x00name"],
    ),
)
def test_virtual_cli_rejects_non_array_and_oversized_argv(tmp_path, argv):
    registry, _workspace = _registry(tmp_path)
    with pytest.raises(virtual_cli_impls.VirtualCliArgumentError):
        registry.execute("cat", argv)


def test_shell_metacharacters_are_literal_path_text(tmp_path):
    registry, workspace = _registry(tmp_path)
    literal_name = "note; echo NOT_A_SHELL.txt"
    (workspace / literal_name).write_text("literal", encoding="utf-8")

    assert registry.execute("cat", [literal_name]) == "1\tliteral"


def test_virtual_cli_reuses_workspace_confinement(tmp_path):
    registry, _workspace = _registry(tmp_path)
    with pytest.raises(LocalToolError, match="outside the workspace root"):
        registry.execute("stat", ["../outside.txt"])
