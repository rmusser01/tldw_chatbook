"""Recognizable shell deletion targets, without executing the commands."""

import os
from pathlib import Path

import pytest

from tldw_chatbook.Tools.workspace_shell_guard import workspace_deletion_violation


@pytest.fixture
def bound_roots(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "bound project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "code.py").touch()
    (root / ".keep").touch()
    other_root = tmp_path / "container" / "second binding"
    other_root.mkdir(parents=True)
    return root, other_root


@pytest.mark.parametrize(
    "dialect,command,violation",
    [
        ("bash", 'rm -r -- "{root}"', "workspace-folder-delete"),
        ("bash", "rmdir .", "workspace-folder-delete"),
        ("bash", "rm -r ../container", "workspace-folder-delete"),
        ("bash", "rm -r ../container/*", "workspace-folder-delete"),
        ("bash", "rm -r ./*", "workspace-folder-wipe"),
        ("bash", "rm -r src .keep", "workspace-folder-wipe"),
        ("bash", "rm -r src; rm .keep", "workspace-folder-wipe"),
        ("bash", "rm -r src\nrm .keep", "workspace-folder-wipe"),
        (
            "bash",
            'cd "../container/second binding" && rm -r .',
            "workspace-folder-delete",
        ),
        ("bash", "find . -mindepth 1 -delete", "workspace-folder-wipe"),
        ("bash", "find ../container -mindepth 1 -delete", "workspace-folder-delete"),
        (
            "powershell",
            "Remove-Item -Recurse -Force -LiteralPath '{root}'",
            "workspace-folder-delete",
        ),
        ("powershell", "rm -r -Path '*'", "workspace-folder-wipe"),
        (
            "powershell",
            "Remove-Item -LiteralPath 'src','.keep' -Recurse",
            "workspace-folder-wipe",
        ),
        (
            "powershell",
            "Get-ChildItem -Force | Remove-Item -Recurse -Force",
            "workspace-folder-wipe",
        ),
        (
            "powershell",
            "Set-Location -LiteralPath '..'; Remove-Item -Recurse -Path 'bound project/*'",
            "workspace-folder-wipe",
        ),
        ("cmd", 'RD /S /Q "{root}"', "workspace-folder-delete"),
        ("cmd", 'cd /d "{root}" && del /s /q *.*', "workspace-folder-wipe"),
        ("cmd", 'del /q "{root}"', "workspace-folder-wipe"),
        ("cmd", 'erase /q "*.*"', "workspace-folder-wipe"),
        ("cmd", "rd /s /q src & erase /q .keep", "workspace-folder-wipe"),
    ],
)
def test_protected_workspace_deletions(bound_roots, dialect, command, violation):
    root, _ = bound_roots

    assert (
        workspace_deletion_violation(
            command.format(root=root.as_posix()),
            dialect=dialect,
            cwd=root,
            protected_roots=bound_roots,
        )
        == violation
    )


@pytest.mark.parametrize(
    "dialect,command",
    [
        ("bash", "rm -r src"),
        ("bash", "rm .keep"),
        ("bash", "rm -r src || rm .keep"),
        ("bash", "rm src; rm .keep"),
        ("bash", 'rm -r "../bound project copy"'),
        ("bash", "rm -r './*'"),
        ("bash", r"rm -r ./\*"),
        ("bash", "find . -name '*.tmp' -delete"),
        ("bash", "find . -mindepth 2 -delete"),
        ("bash", "printf '%s' 'text; rm -r .; more text'"),
        ("bash", "printf '%s' text # note; rm -r ."),
        ("powershell", "Remove-Item -LiteralPath '*' -Recurse"),
        ("powershell", "Remove-Item -Path . -Recurse -WhatIf"),
        ("powershell", "Remove-Item -Path '*' -Filter '*.tmp'"),
        ("powershell", "Get-ChildItem -Filter '*.tmp' | Remove-Item -Recurse"),
        ("powershell", "Write-Output 'text; Remove-Item -Recurse .; more text'"),
        ("cmd", "del /q .keep"),
        ("cmd", "del /a:h *.*"),
        ("cmd", 'echo "text & rd /s /q . & more text"'),
    ],
)
def test_targeted_deletions_and_quoted_data_are_allowed(bound_roots, dialect, command):
    root, _ = bound_roots

    assert (
        workspace_deletion_violation(
            command, dialect=dialect, cwd=root, protected_roots=bound_roots
        )
        is None
    )


@pytest.mark.parametrize(
    "dialect,command",
    [
        ("bash", "rm -- .keep"),
        ("powershell", "Remove-Item -LiteralPath '.keep' -Force"),
        ("cmd", "del /q .keep"),
    ],
)
def test_removing_last_hidden_child_is_a_wipe(tmp_path, dialect, command):
    root = tmp_path / "single-child"
    root.mkdir()
    (root / ".keep").touch()

    assert (
        workspace_deletion_violation(
            command, dialect=dialect, cwd=root, protected_roots=(root,)
        )
        == "workspace-folder-wipe"
    )


def test_missing_binding_still_protects_its_location(tmp_path):
    missing = tmp_path / "missing"

    assert (
        workspace_deletion_violation(
            "rm -r missing", dialect="bash", cwd=tmp_path, protected_roots=(missing,)
        )
        == "workspace-folder-delete"
    )


def test_link_traversal_protects_binding_but_unlinking_alias_is_allowed(
    tmp_path, bound_roots
):
    root, _ = bound_roots
    link = tmp_path / "shortcut"
    try:
        if os.name == "nt":
            import _winapi

            _winapi.CreateJunction(str(root), str(link))
        else:
            link.symlink_to(root, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory links unavailable: {exc}")

    options = dict(dialect="bash", cwd=tmp_path, protected_roots=bound_roots)
    assert (
        workspace_deletion_violation("rm -r shortcut/*", **options)
        == "workspace-folder-wipe"
    )
    assert workspace_deletion_violation("rm -r shortcut", **options) is None


def test_windows_path_case_and_separators_resolve_to_binding(bound_roots):
    if os.name != "nt":
        pytest.skip("Native Windows path semantics")
    root, _ = bound_roots

    assert (
        workspace_deletion_violation(
            f'RD /S "{str(root).swapcase()}"',
            dialect="cmd",
            cwd=root.parent,
            protected_roots=bound_roots,
        )
        == "workspace-folder-delete"
    )
