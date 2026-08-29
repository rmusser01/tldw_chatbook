from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.Tools.workspace_root_pin import (
    WorkspaceRootPinError,
    pin_workspace_root,
)
from tldw_chatbook.Utils.filesystem_identity import DirectoryChain, capture_directory_chain


def _pre_pin_child(locator: str, chain: DirectoryChain, output: Any) -> None:
    try:
        with pin_workspace_root(Path(locator), chain):
            output.put(("pinned", ""))
    except WorkspaceRootPinError as error:
        output.put(("refused", str(error)))


def _post_pin_child(
    locator: str,
    chain: DirectoryChain,
    ready: Any,
    resume: Any,
    output: Any,
) -> None:
    try:
        with pin_workspace_root(Path(locator), chain) as pinned:
            ready.set()
            if not resume.wait(5):
                raise RuntimeError("test barrier timed out")
            output.put(
                (
                    "read",
                    pinned.relative_path("sentinel.txt").read_text(encoding="utf-8"),
                )
            )
    except BaseException as error:
        output.put(("error", type(error).__name__))


def _spawn_context() -> multiprocessing.context.BaseContext:
    return multiprocessing.get_context("spawn")


def _join_child(process: Any) -> None:
    process.join(10)
    if process.is_alive():
        process.kill()
        process.join(5)
        pytest.fail("root-pin child did not exit")
    assert process.exitcode == 0


def test_root_replacement_before_pin_is_refused_by_identity(tmp_path: Path) -> None:
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "sentinel.txt").write_text("A", encoding="utf-8")
    chain = capture_directory_chain(locator)

    retained_a = tmp_path / "retained-a"
    locator.rename(retained_a)
    locator.mkdir()
    (locator / "sentinel.txt").write_text("B", encoding="utf-8")

    context = _spawn_context()
    output = context.Queue()
    process = context.Process(
        target=_pre_pin_child,
        args=(str(locator), chain, output),
    )
    process.start()
    _join_child(process)

    assert output.get(timeout=2) == ("refused", "workspace root identity mismatch")


def test_root_replacement_after_pin_never_redirects_relative_io(tmp_path: Path) -> None:
    locator = tmp_path / "workspace"
    locator.mkdir()
    (locator / "sentinel.txt").write_text("A", encoding="utf-8")
    replacement_b = tmp_path / "replacement-b"
    replacement_b.mkdir()
    (replacement_b / "sentinel.txt").write_text("B", encoding="utf-8")
    chain = capture_directory_chain(locator)

    context = _spawn_context()
    ready = context.Event()
    resume = context.Event()
    output = context.Queue()
    process = context.Process(
        target=_post_pin_child,
        args=(str(locator), chain, ready, resume, output),
    )
    process.start()
    assert ready.wait(5), "root-pin child did not reach the barrier"

    retained_a = tmp_path / "retained-a"
    replacement_refused = False
    try:
        os.replace(locator, retained_a)
        os.replace(replacement_b, locator)
    except OSError:
        replacement_refused = True
        if retained_a.exists() and not locator.exists():
            os.replace(retained_a, locator)
    finally:
        resume.set()

    _join_child(process)
    outcome = output.get(timeout=2)
    assert outcome == ("read", "A")
    if os.name == "nt":
        assert replacement_refused, "Windows should lock the retained current directory"


def test_relative_path_rejects_absolute_and_parent_paths(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    chain = capture_directory_chain(root)

    with pin_workspace_root(root, chain) as pinned:
        with pytest.raises(WorkspaceRootPinError, match="relative path"):
            pinned.relative_path(str(tmp_path / "outside.txt"))
        with pytest.raises(WorkspaceRootPinError, match="relative path"):
            pinned.relative_path("../outside.txt")
