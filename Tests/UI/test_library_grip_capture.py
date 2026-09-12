"""Reproduction contracts for the Library reader grip capture utility."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


CAPTURE_SCRIPT = (
    Path(__file__).parents[2]
    / "Docs"
    / "superpowers"
    / "qa"
    / "library-reader-grip-polish-2026-09"
    / "capture_grips.py"
)


def test_capture_bootstrap_uses_platform_temp_root_without_import_side_effects(
    tmp_path: Path,
) -> None:
    """Importing the bootstrap selects TMPDIR without creating its workspace."""
    environment = os.environ.copy()
    environment["TMPDIR"] = str(tmp_path)
    probe = (
        "import runpy; "
        f"namespace = runpy.run_path({str(CAPTURE_SCRIPT)!r}, run_name='qa_import'); "
        "print(namespace['QA_ROOT'])"
    )

    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    expected_root = tmp_path / "tldw-chatbook-library-reader-grip-polish-qa"
    assert Path(result.stdout.strip().splitlines()[-1]) == expected_root
    assert not expected_root.exists()
