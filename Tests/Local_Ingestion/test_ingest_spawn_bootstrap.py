"""Regression guards for spawn-safe supported CLI bootstraps."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _python_env(*, pythonpath: str) -> dict[str, str]:
    """Return an environment that imports this checkout first."""

    return {**os.environ, "PYTHONPATH": pythonpath}


def test_importing_cli_does_not_import_app() -> None:
    """The console-script target must be safe to import in a spawn child."""

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tldw_chatbook.cli; "
            "assert 'tldw_chatbook.app' not in sys.modules",
        ],
        cwd=REPO_ROOT,
        env=_python_env(pythonpath=str(REPO_ROOT)),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_cli_returns_delegated_runner_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lightweight wrapper must preserve the app runner's return value."""

    from tldw_chatbook import cli

    expected = object()
    fake_app = types.ModuleType("tldw_chatbook.app")
    fake_app.main_cli_runner = lambda: expected
    monkeypatch.setitem(sys.modules, "tldw_chatbook.app", fake_app)

    assert cli.main_cli_runner() is expected


def test_module_launcher_uses_delegated_value_as_exit_code(tmp_path: Path) -> None:
    """``python -m`` must transparently propagate a delegated CLI exit code."""

    hook_dir = tmp_path / "exit_hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import sys
            import types

            fake_app = types.ModuleType("tldw_chatbook.app")
            fake_app.main_cli_runner = lambda: 7
            sys.modules["tldw_chatbook.app"] = fake_app
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "tldw_chatbook"],
        cwd=REPO_ROOT,
        env=_python_env(
            pythonpath=os.pathsep.join((str(hook_dir), str(REPO_ROOT)))
        ),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 7, result.stderr


def test_module_launcher_spawn_child_starts_without_heavy_app_imports(
    tmp_path: Path,
) -> None:
    """A real spawn child re-imports the supported package launcher lightly."""

    hook_dir = tmp_path / "bootstrap_hook"
    hook_dir.mkdir()
    (hook_dir / "sitecustomize.py").write_text(
        textwrap.dedent(
            """
            import json
            import multiprocessing
            import sys
            import types


            def child_target(queue):
                guarded = ("tldw_chatbook.app", "torch", "transformers")
                queue.put([name for name in guarded if name in sys.modules])


            def parent_main_cli_runner():
                context = multiprocessing.get_context("spawn")
                queue = context.Queue()
                child = context.Process(target=child_target, args=(queue,))
                child.start()
                try:
                    loaded = queue.get(timeout=15)
                    child.join(timeout=15)
                    if child.is_alive():
                        raise RuntimeError("spawn child did not exit")
                    if child.exitcode != 0:
                        raise RuntimeError(f"spawn child exited {child.exitcode}")
                    print(json.dumps({"loaded": loaded}))
                finally:
                    if child.is_alive():
                        child.terminate()
                        child.join(timeout=5)
                    queue.close()
                    queue.join_thread()


            if "--multiprocessing-fork" not in sys.argv:
                fake_app = types.ModuleType("tldw_chatbook.app")
                fake_app.main_cli_runner = parent_main_cli_runner
                sys.modules["tldw_chatbook.app"] = fake_app
            """
        ),
        encoding="utf-8",
    )

    pythonpath = os.pathsep.join((str(hook_dir), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-m", "tldw_chatbook"],
        cwd=REPO_ROOT,
        env=_python_env(pythonpath=pythonpath),
        text=True,
        capture_output=True,
        timeout=45,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["loaded"] == []
