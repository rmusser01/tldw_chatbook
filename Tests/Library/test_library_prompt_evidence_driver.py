"""Isolation checks for the TASK-22033 live evidence driver."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_prompt_evidence_driver_rejects_xdg_path_outside_scratch(tmp_path) -> None:
    root = Path(__file__).resolve().parents[2]
    driver = (
        root
        / "Docs/superpowers/reviews/evidence/task-22033/task22033_live_matrix.py"
    )
    scratch = tmp_path / "scratch"
    outside_cache = tmp_path / "outside-cache"
    env = os.environ.copy()
    env.update(
        {
            "TASK22033_SCRATCH_ROOT": str(scratch),
            "TASK22033_DATA_DIR": str(scratch / "prompt-data"),
            "XDG_CONFIG_HOME": str(scratch / "xdg-config"),
            "XDG_DATA_HOME": str(scratch / "xdg-data"),
            "XDG_CACHE_HOME": str(outside_cache),
            "TLDW_CONFIG_PATH": str(scratch / "config/config.toml"),
            "TLDW_TEST_MODE": "1",
            "TLDW_DISABLE_MODEL_CATALOG_NETWORK": "1",
        }
    )

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
