"""Isolation checks for the TASK-22033 live evidence driver."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _driver_paths() -> tuple[Path, Path, Path]:
    root = Path(__file__).resolve().parents[2]
    evidence_dir = root / "Docs/superpowers/reviews/evidence/task-22033"
    return root, evidence_dir / "task22033_live_matrix.py", evidence_dir


def _isolated_env(root: Path, scratch: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                part
                for part in (str(root), env.get("PYTHONPATH", ""))
                if part
            ),
            "TASK22033_SCRATCH_ROOT": str(scratch),
            "TASK22033_DATA_DIR": str(scratch / "prompt-data"),
            "XDG_CONFIG_HOME": str(scratch / "xdg-config"),
            "XDG_DATA_HOME": str(scratch / "xdg-data"),
            "XDG_CACHE_HOME": str(scratch / "xdg-cache"),
            "TLDW_CONFIG_PATH": str(scratch / "config/config.toml"),
            "TLDW_TEST_MODE": "1",
            "TLDW_DISABLE_MODEL_CATALOG_NETWORK": "1",
        }
    )
    return env


def test_prompt_evidence_driver_rejects_xdg_path_outside_scratch(tmp_path) -> None:
    root, driver, _evidence_dir = _driver_paths()
    scratch = tmp_path / "scratch"
    outside_cache = tmp_path / "outside-cache"
    env = _isolated_env(root, scratch)
    env["XDG_CACHE_HOME"] = str(outside_cache)

    result = subprocess.run(
        [sys.executable, str(driver), "preflight"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert "must be contained by TASK22033_SCRATCH_ROOT" in (
        result.stdout + result.stderr
    )
    assert not outside_cache.exists()


@pytest.mark.parametrize("selector", ["unknown", "<script>"])
def test_prompt_evidence_driver_rejects_invalid_journey_selector(
    tmp_path, selector: str
) -> None:
    root, driver, _evidence_dir = _driver_paths()
    scratch = tmp_path / "scratch"

    result = subprocess.run(
        [sys.executable, str(driver), selector],
        cwd=root,
        env=_isolated_env(root, scratch),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert "journey selector" in (result.stdout + result.stderr)


def test_prompt_evidence_driver_replaces_existing_external_data_config(
    tmp_path,
) -> None:
    root, driver, _evidence_dir = _driver_paths()
    scratch = tmp_path / "scratch"
    config_path = scratch / "config/config.toml"
    outside_data = tmp_path / "outside-data"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        f'[paths]\ndata_dir = "{outside_data.as_posix()}"\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(driver), "unknown"],
        cwd=root,
        env=_isolated_env(root, scratch),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode != 0
    assert config_path.read_text(encoding="utf-8") == (
        f'[paths]\ndata_dir = "{(scratch / "app-data").as_posix()}"\n'
    )
    assert not outside_data.exists()


def test_prompt_evidence_host_closes_database_after_failure(monkeypatch) -> None:
    _root, _driver, evidence_dir = _driver_paths()
    runner_path = evidence_dir / "task22033_live_matrix_runner.py"
    module_name = "task22033_live_matrix_runner_test"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class FakeDatabase:
        closed = False

        def close(self) -> None:
            self.closed = True

    database = FakeDatabase()

    class FakeHarness:
        def __init__(self, _app) -> None:
            pass

        @asynccontextmanager
        async def run_test(self, *, size):
            yield SimpleNamespace(size=size)

    monkeypatch.setattr(runner, "LibraryProductionCSSHarness", FakeHarness)

    async def fail_inside_host() -> None:
        async with runner._run_seeded_host(
            SimpleNamespace(), database, size=(80, 24)
        ):
            raise RuntimeError("journey failed")

    with pytest.raises(RuntimeError, match="journey failed"):
        asyncio.run(fail_inside_host())

    assert database.closed is True
