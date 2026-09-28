"""Session loopback: the real bundle through the real bootstrap + loader.

No ssh: ``run_session_loopback`` spawns ``python -I -c <bootstrap>``
locally, feeds the stage-1 loader, answers ``NEED`` with the bundle,
asserts ``READY <stamp>``, then drives the bundle's ``serve_session``
fork-server over binary frames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.Tools.test_remote_worker_bundle import _python_310_interpreter
from tldw_chatbook.Tools.remote_workspace_executor import run_session_loopback


def _ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    root.mkdir()
    (root / "a.txt").write_text("alpha\n")
    (root / "big.txt").write_text("a" * 60000 + "b\n")
    return root


def test_ping_then_reads_share_one_session(tmp_path: Path) -> None:
    root = _ws(tmp_path)
    out = run_session_loopback(
        root,
        [{"op": "ping"}, {"op": "fs_read", "path": "a.txt"}, {"op": "fs_list", "path": "."}],
    )
    assert out[1][-2]["outcome"] == "success" and "alpha" in out[1][-2]["result"]
    assert all(frames[-1]["status"] == (0, None) for frames in out.values())


def test_catastrophic_regex_dies_alone(tmp_path: Path) -> None:
    root = _ws(tmp_path)
    out = run_session_loopback(
        root,
        [
            {"op": "fs_grep", "pattern": "(a+)+$", "budget": 2},
            {"op": "fs_read", "path": "a.txt"},
        ],
    )
    exit_code, signal_no = out[0][-1]["status"]
    assert exit_code == 75 or signal_no is not None
    assert out[1][-2]["outcome"] == "success"


def test_pin_failure_leaves_session_up(tmp_path: Path) -> None:
    root = _ws(tmp_path)
    out = run_session_loopback(
        root,
        [
            {"op": "fs_read", "path": "a.txt", "stale_identity": True},
            {"op": "fs_read", "path": "a.txt"},
        ],
    )
    assert out[0][-2]["code"] == "root_pin_failed"
    assert out[1][-2]["outcome"] == "success"


def test_cache_hit_on_second_session(tmp_path: Path) -> None:
    root = _ws(tmp_path)
    cache = tmp_path / "run"
    cache.mkdir(mode=0o700)
    run_session_loopback(root, [{"op": "ping"}], cache_dir=cache)
    assert any((cache / "tldw-worker").iterdir())
    # Must not ask for the bundle again: the harness asserts READY first.
    run_session_loopback(root, [{"op": "ping"}], cache_dir=cache)


def test_session_executes_on_python_310_floor(tmp_path: Path) -> None:
    """Bootstrap + loader + serve + bundle EXECUTE under a real 3.10."""
    interpreter = _python_310_interpreter()
    if interpreter is None:
        pytest.skip("no 3.10 interpreter — CI must install one (python3.10 on PATH)")
    root = _ws(tmp_path)
    out = run_session_loopback(
        root, [{"op": "ping"}, {"op": "fs_read", "path": "a.txt"}], python=interpreter
    )
    assert out[1][-2]["outcome"] == "success" and "alpha" in out[1][-2]["result"]
    assert all(frames[-1]["status"] == (0, None) for frames in out.values())
