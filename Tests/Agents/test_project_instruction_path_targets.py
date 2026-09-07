"""Structural tool-call target mapping for nested project instructions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tldw_chatbook.Agents.local_tool_provider import LocalToolProvider
from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider, ToolPathTarget
from tldw_chatbook.Tools.local_tool_impls import LocalToolError


def _local_provider(root: Path, *, todo_store: list | None = None) -> LocalToolProvider:
    return LocalToolProvider(workspace_root=root, todo_store=todo_store)


def _target(path: Path, kind: str) -> ToolPathTarget:
    return ToolPathTarget(path=path.resolve(), kind=kind)


@pytest.mark.parametrize("name", ["fs_read", "fs_write", "fs_edit"])
def test_local_file_tools_report_exact_targets(tmp_path, name):
    provider = _local_provider(tmp_path)
    expected = tmp_path / "src" / "pkg" / "module.py"

    assert provider.path_targets(f"local:{name}", {"path": "src/pkg/module.py"}) == (
        _target(expected, "exact"),
    )


def test_local_list_reports_directory_target(tmp_path):
    provider = _local_provider(tmp_path)

    assert provider.path_targets("local:fs_list", {"path": "src/pkg"}) == (
        _target(tmp_path / "src" / "pkg", "directory"),
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("fs_glob", {"pattern": "src/deep/**/*.py"}),
        ("fs_grep", {"pattern": "src/deep/secret"}),
    ],
)
def test_local_search_tools_never_infer_scope_from_patterns(tmp_path, name, args):
    provider = _local_provider(tmp_path)

    assert provider.path_targets(f"local:{name}", args) == (
        _target(tmp_path, "directory"),
    )


PATCH = """\
--- a/src/one.py
+++ b/src/one.py
@@ -1 +1 @@
-old
+new
--- /dev/null
+++ b/lib/new.py
@@ -0,0 +1 @@
+created
"""


@pytest.mark.parametrize("dry_run", [False, True])
def test_patch_reports_every_create_and_modify_target_for_real_and_dry_run(
    tmp_path, dry_run
):
    provider = _local_provider(tmp_path)

    assert provider.path_targets(
        "local:fs_patch", {"diff": PATCH, "dry_run": dry_run}
    ) == (
        _target(tmp_path / "src" / "one.py", "exact"),
        _target(tmp_path / "lib" / "new.py", "exact"),
    )


@pytest.mark.parametrize(
    ("diff_text", "reason"),
    [
        ("not a diff", "invalid_diff"),
        (
            "--- a/old.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n",
            "delete_not_supported",
        ),
        (
            "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-old\n+new\n",
            "rename_not_supported",
        ),
    ],
)
def test_patch_preflight_preserves_parser_errors(tmp_path, diff_text, reason):
    provider = _local_provider(tmp_path)

    with pytest.raises(LocalToolError, match=reason):
        provider.path_targets("local:fs_patch", {"diff": diff_text})


def _git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "live.py").write_text("pass\n", encoding="utf-8")
    return root


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("git_branches", {}),
        ("git_diff", {}),
        ("git_log", {}),
        ("git_status", {}),
        ("git_status", {"path": "src/pkg/live.py"}),
    ],
)
def test_unfiltered_git_operations_report_only_repository_root(tmp_path, name, args):
    root = _git_repo(tmp_path)
    provider = _local_provider(root)

    assert provider.path_targets(f"local:{name}", args) == (
        _target(root, "repository"),
    )


@pytest.mark.parametrize("name", ["git_diff", "git_log"])
@pytest.mark.parametrize(
    ("path", "scope"),
    [
        ("src/pkg", "src/pkg"),
        ("src/pkg/live.py", "src/pkg"),
        ("src/pkg/deleted.py", "src/pkg"),
    ],
)
def test_path_filtered_git_operations_report_repository_through_target_scope(
    tmp_path, name, path, scope
):
    root = _git_repo(tmp_path)
    provider = _local_provider(root)
    args = {
        "path": path,
        "commit_range": "HEAD~1..HEAD",
        "staged": True,
        "stat": True,
    }

    assert provider.path_targets(f"local:{name}", args) == (
        _target(root / scope, "repository"),
    )


def test_git_blame_reports_repository_through_file_parent(tmp_path):
    root = _git_repo(tmp_path)
    provider = _local_provider(root)

    assert provider.path_targets("local:git_blame", {"path": "src/pkg/live.py"}) == (
        _target(root / "src" / "pkg", "repository"),
    )


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("web_fetch", {"url": "https://example.com/deep/file"}),
        ("web_search", {"query": "src/pkg"}),
        ("web_crawl", {"url": "https://example.com/src/pkg"}),
        ("todo_write", {"todos": []}),
        ("spawn_subagent", {"task": "inspect src/pkg"}),
        ("run_skill_script", {"script": "cat src/pkg/file"}),
        ("process", {"command": "cat src/pkg/file"}),
        ("mcp_tool", {"path": "src/pkg/file"}),
    ],
)
def test_opaque_and_nonlocal_tools_report_no_targets(tmp_path, name, args):
    provider = _local_provider(tmp_path, todo_store=[])

    assert provider.path_targets(f"local:{name}", args) == ()


def _enable_builtin_file_tools(monkeypatch):
    import tldw_chatbook.config as config

    enabled = {
        "read_file_enabled",
        "list_directory_enabled",
        "write_file_enabled",
    }
    monkeypatch.setattr(
        config,
        "get_cli_setting",
        lambda section, key, default=None: (
            key in enabled if section == "tools" else default
        ),
    )


@pytest.mark.parametrize(
    ("name", "argument", "kind"),
    [
        ("read_file", "file_path", "exact"),
        ("write_file", "file_path", "exact"),
        ("list_directory", "directory_path", "directory"),
    ],
)
def test_builtin_file_tools_map_selected_binding_targets(
    monkeypatch, tmp_path, name, argument, kind
):
    _enable_builtin_file_tools(monkeypatch)
    selected = tmp_path / "selected"
    selected.mkdir()
    target = selected / "src" / ("pkg" if kind == "directory" else "file.py")
    monkeypatch.setattr(
        "tldw_chatbook.Tools.file_operation_tools.allowed_file_roots",
        lambda **_kwargs: (selected,),
    )
    provider = BuiltinToolProvider(instruction_root=selected)

    assert provider.path_targets(f"builtin:{name}", {argument: str(target)}) == (
        _target(target, kind),
    )


@pytest.mark.parametrize(
    ("name", "argument"),
    [
        ("read_file", "file_path"),
        ("write_file", "file_path"),
        ("list_directory", "directory_path"),
    ],
)
def test_builtin_other_authorized_binding_is_outside_instruction_scope(
    monkeypatch, tmp_path, name, argument
):
    _enable_builtin_file_tools(monkeypatch)
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    selected.mkdir()
    other.mkdir()
    target = other / "nested" / "file.py"
    monkeypatch.setattr(
        "tldw_chatbook.Tools.file_operation_tools.allowed_file_roots",
        lambda **_kwargs: (selected, other),
    )
    provider = BuiltinToolProvider(instruction_root=selected)

    assert provider.path_targets(f"builtin:{name}", {argument: str(target)}) == (
        _target(target, "outside"),
    )


def test_builtin_file_tools_report_no_targets_when_instruction_scope_disabled(
    monkeypatch, tmp_path
):
    _enable_builtin_file_tools(monkeypatch)
    provider = BuiltinToolProvider(instruction_root=None)

    assert (
        provider.path_targets(
            "builtin:read_file", {"file_path": str(tmp_path / "file.py")}
        )
        == ()
    )


def test_config_disabled_builtin_reports_no_targets(monkeypatch, tmp_path):
    import tldw_chatbook.config as config

    monkeypatch.setattr(config, "get_cli_setting", lambda *_args, **_kwargs: False)
    provider = BuiltinToolProvider(instruction_root=tmp_path)

    assert (
        provider.path_targets(
            "builtin:read_file", {"file_path": str(tmp_path / "file.py")}
        )
        == ()
    )
