"""B27: the grep/list tools resolve the workspace once, not per entry.

``_relative_target_is_safe`` called ``workspace.resolve()`` on every entry of
a grep/glob/list walk -- one resolution syscall chain per file. The hoisted
form resolves the workspace once per operation and passes it in; decisions
must be byte-identical over a path list covering plain files, nested dirs,
sensitive paths, symlinked entries, and traversal-shaped names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Tools.local_tool_impls import (
    _grep_relative_files,
    _relative_target_is_safe,
)
from tldw_chatbook.Utils.sensitive_paths import sensitive_exclusions_under


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src/deep").mkdir(parents=True)
    (root / "src/app.py").write_text("needle one\n")
    (root / "src/deep/lib.py").write_text("needle two\n")
    for index in range(200):
        (root / f"file-{index:03}.txt").write_text(f"needle {index}\n")
    (root / ".env").write_text("secret=1\n")
    (root / "notes copy.txt").write_text("needle notes\n")
    return root


def test_safe_decisions_identical_with_precomputed_workspace(
    workspace: Path,
) -> None:
    """The hoisted resolve must not change a single accept/reject decision."""
    exclusions = sensitive_exclusions_under(workspace, None)
    resolved_workspace = workspace.resolve()
    candidates: list[tuple[Path, bool]] = [
        (Path("src/app.py"), False),
        (Path("src/deep/lib.py"), False),
        (Path(".env"), False),  # sensitive -> reject
        (Path("notes copy.txt"), False),
        (Path("../outside.txt"), False),  # traversal shape
        (Path("missing.py"), False),  # absent -> still "safe" lexically
        (Path("src"), True),
        (Path("src/deep"), True),
        (Path("."), True),
        (Path(""), True),
    ]
    # symlinked file inside the tree
    link = workspace / "link-to-app"
    try:
        link.symlink_to(workspace / "src/app.py")
        candidates.append((Path("link-to-app"), False))
    except OSError:  # pragma: no cover - platform without symlink support
        pass

    for relative, is_directory in candidates:
        with_resolve = _relative_target_is_safe(
            relative,
            workspace,
            exclusions,
            is_directory=is_directory,
        )
        hoisted = _relative_target_is_safe(
            relative,
            workspace,
            exclusions,
            is_directory=is_directory,
            resolved_workspace=resolved_workspace,
        )
        assert with_resolve == hoisted, (
            f"decision changed for {relative!r}: {with_resolve} != {hoisted}"
        )


def test_grep_resolves_workspace_once_per_operation(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exclusions = sensitive_exclusions_under(workspace, None)
    resolve_calls = {"count": 0}
    real_resolve = Path.resolve

    def counting_resolve(self, *args, **kwargs):
        if self == workspace:
            resolve_calls["count"] += 1
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counting_resolve)

    output = _grep_relative_files(
        "needle",
        workspace=workspace,
        mode="files",
        max_results=10,
        sensitive_exclusions=exclusions,
    )

    # mtimes decide the top-10 order; just require real matches surfaced
    assert "file-016.txt" in output
    assert not output.startswith("(no matches")
    assert resolve_calls["count"] <= 1, (
        f"grep resolved the workspace {resolve_calls['count']} times for "
        "~203 entries; the resolution must be hoisted out of the per-entry "
        "safety check"
    )
