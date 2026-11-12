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
    (root / "src" / "cache").mkdir()
    (root / ".keep").touch()
    other_root = tmp_path / "container" / "second binding"
    other_root.mkdir(parents=True)
    return root, other_root


@pytest.mark.parametrize(
    "dialect,command,violation",
    [
        ("bash", 'rm -r -- "{root}"', "workspace-folder-delete"),
        ("bash", "rmdir .", "workspace-folder-delete"),
        ("bash", "rm -r . 2>/dev/null", "workspace-folder-delete"),
        ("bash", "rm -r ../container 2>&1", "workspace-folder-delete"),
        ("bash", 'rmdir -p "{root}/src/cache"', "workspace-folder-delete"),
        ("bash", "rm -r ../container", "workspace-folder-delete"),
        ("bash", "rm -r ../container/*", "workspace-folder-delete"),
        ("bash", "rm -r ./*", "workspace-folder-wipe"),
        ("bash", "rm -r ./**", "workspace-folder-wipe"),
        ("bash", "rm -r src .keep", "workspace-folder-wipe"),
        ("bash", "rm -r src; rm .keep", "workspace-folder-wipe"),
        ("bash", "rm -r src; echo done; rm .keep", "workspace-folder-wipe"),
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
        ("powershell", "rm -r -Path '*' -Force:$true", "workspace-folder-wipe"),
        (
            "powershell",
            "& 'Remove-Item' -Recurse -LiteralPath '{root}'",
            "workspace-folder-delete",
        ),
        (
            "powershell",
            "Remove-Item -Recurse:$true -Force:$false . > 'log.txt'",
            "workspace-folder-delete",
        ),
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
        ("cmd", 'rd /s /q "{root}" 2>nul', "workspace-folder-delete"),
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
        ("bash", "rm -r src > ."),
        ("bash", "rmdir -p src/cache"),
        ("bash", "rm .keep"),
        ("bash", "rm -r src || rm .keep"),
        ("bash", "rm -r src; echo done > retained.txt; rm .keep"),
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
        ("powershell", "Remove-Item -Path . -Recurse -WhatIf:$true"),
        ("powershell", "Remove-Item -Recurse src *> 'log.txt'"),
        ("powershell", "<# note\nRemove-Item -Recurse .\n#>\nWrite-Output normal"),
        ("powershell", "Remove-Item -Path '*' -Filter '*.tmp'"),
        ("powershell", "Get-ChildItem -Filter '*.tmp' | Remove-Item -Recurse"),
        ("powershell", "Write-Output 'text; Remove-Item -Recurse .; more text'"),
        ("powershell", "'Remove-Item'"),
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


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("true", None),
        ("false", "workspace-folder-wipe"),
        ("probe", "workspace-folder-wipe"),
    ],
)
def test_fallback_location_respects_command_reachability(bound_roots, status, expected):
    root, _ = bound_roots
    sibling = root.parent / "unbound project"
    sibling.mkdir()
    assert (
        workspace_deletion_violation(
            f'{status} || cd "{root.as_posix()}"; rm -r ./*',
            dialect="bash",
            cwd=sibling,
            protected_roots=(root,),
        )
        == expected
    )


def test_skipped_fallback_leaves_bound_working_directory(bound_roots):
    root, _ = bound_roots
    sibling = root.parent / "unbound project"
    sibling.mkdir()
    assert (
        workspace_deletion_violation(
            f'true || cd "{sibling.as_posix()}"; rm -r ./*',
            dialect="bash",
            cwd=root,
            protected_roots=(root,),
        )
        == "workspace-folder-wipe"
    )


@pytest.mark.parametrize("has_sibling", [False, True])
def test_parent_pruning_can_remove_the_last_child(tmp_path, has_sibling):
    root = tmp_path / "workspace"
    (root / "parent" / "leaf").mkdir(parents=True)
    if has_sibling:
        (root / "parent" / "retained").touch()
    assert workspace_deletion_violation(
        "rmdir -p parent/leaf",
        dialect="bash",
        cwd=root,
        protected_roots=(root,),
    ) == (None if has_sibling else "workspace-folder-wipe")


def test_cmd_location_accepts_other_drive_letter_casing(bound_roots):
    if os.name != "nt":
        pytest.skip("Native Windows path semantics")
    root, _ = bound_roots
    other_casing = root.as_posix()[0].swapcase() + root.as_posix()[1:]
    assert (
        workspace_deletion_violation(
            f'cd "{other_casing}" && del /q *.*',
            dialect="cmd",
            cwd=root.parent,
            protected_roots=(root,),
        )
        == "workspace-folder-wipe"
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


@pytest.mark.parametrize("suffix", ["", "."])
def test_windows_path_alias_resolves_to_binding(bound_roots, suffix):
    if os.name != "nt":
        pytest.skip("Native Windows path semantics")
    root, _ = bound_roots

    assert (
        workspace_deletion_violation(
            f'RD /S "{str(root).swapcase()}{suffix}"',
            dialect="cmd",
            cwd=root.parent,
            protected_roots=bound_roots,
        )
        == "workspace-folder-delete"
    )
