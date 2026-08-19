# Tests/Notes/test_git_network_pin_environment_contract.py
"""TASK-18609 diagnostic: name the unsafe filesystem edge on CI runners.

17 `test_file_notes_git_push_service` tests fail on both GitHub runners with
`NetworkContextError: invalid_executable` / `unsafe_filesystem` from
`_pin_executable` and `_capture_safe_ancestors`, while passing on developer
macOS and Linux machines. The validators walk EVERY ancestor of the pinned
executable/tmp file up to `/` and require: owner in {euid, 0}, no
group/other write unless root-owned AND sticky, regular file with exec bit,
and nlink == 1 for non-root files. Something in the runner filesystem layout
violates one of those predicates; which one is not observable from the
current failure messages.

This canary re-runs the same predicates and, on failure, prints the FULL
stat table (path, uid, mode, nlink, sticky) for every candidate and every
ancestor, so the next CI run names the exact directory and predicate. It
fails exactly where the real tests already fail, and passes everywhere they
pass -- it adds no new red.
"""

from __future__ import annotations

import os
import shutil
import stat as stat_module
import sys
from pathlib import Path

import pytest

from tldw_chatbook.Notes import file_notes_git_network as git_network


def _describe(path: Path, metadata: os.stat_result | None = None) -> str:
    try:
        info = metadata or path.stat(follow_symlinks=False)
    except OSError as exc:
        return f"{path}: <stat failed: {exc}>"
    return (
        f"{path}: uid={info.st_uid} gid={info.st_gid} "
        f"mode={stat_module.S_IMODE(info.st_mode):o} "
        f"nlink={info.st_nlink} "
        f"sticky={bool(info.st_mode & stat_module.S_ISVTX)} "
        f"reg={stat_module.S_ISREG(info.st_mode)} "
        f"dir={stat_module.S_ISDIR(info.st_mode)}"
    )


def _table(path: Path) -> str:
    lines = [_describe(path)]
    for ancestor in path.parents:
        lines.append(_describe(ancestor))
    return "\n".join(lines)


@pytest.mark.parametrize(
    "label,value,search_path",
    [
        ("git-from-defpath", shutil.which("git", path=os.defpath), os.defpath),
        ("git-usr-bin", "/usr/bin/git", os.defpath),
        ("ssh-from-defpath", shutil.which("ssh", path=os.defpath), os.defpath),
        ("ssh-usr-bin", "/usr/bin/ssh", os.defpath),
    ],
)
def test_runner_pinnable_executables_pass_the_safety_predicates(
    label: str, value: str | None, search_path: str
) -> None:
    if value is None:
        pytest.skip(f"{label}: not present on this host")
    try:
        git_network._pin_executable(value, search_path)
    except git_network.NetworkContextError:
        resolved = Path(value).resolve(strict=False)
        pytest.fail(
            f"{label}: _pin_executable({value!r}) rejected on this host.\n"
            f"stat table of the resolved candidate and ancestors:\n"
            f"{_table(resolved)}\n"
            f"euid={os.geteuid()} TMPDIR={os.environ.get('TMPDIR')!r} "
            f"python={sys.version.split()[0]}"
        )


def test_pytest_tmp_root_passes_the_ancestor_predicates(tmp_path: Path) -> None:
    """The agent-socket pins live under pytest's tmp; name any bad ancestor."""
    probe = tmp_path / "pin-probe"
    probe.write_text("probe", encoding="utf-8")
    try:
        git_network._capture_safe_ancestors(probe.resolve())
    except git_network.NetworkContextError:
        pytest.fail(
            "pytest tmp root rejected by _capture_safe_ancestors on this "
            f"host.\nstat table:\n{_table(probe.resolve())}\n"
            f"euid={os.geteuid()} TMPDIR={os.environ.get('TMPDIR')!r}"
        )
