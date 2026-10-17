"""TASK-33641.8: a pytest run started at the repo root never reaches Docs/.

An empty argument or '.' is still an argument, so ``testpaths`` does not
apply and collection starts at the root. Docs/ holds tracked test-named files
no conftest covers: one calls a paid provider with the configured key, one
rewrites a tracked test file when imported. ``--ignore=Docs`` in pyproject's
addopts is the guard. These runs disable the python plugin and every conftest,
so nothing under Docs/ is imported even if the guard is gone; a recorder
plugin notes every file the collection visits there instead. Without the
guard both root runs listed all 17,093 files under Docs/.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPLICIT_DOCS_FILE = "Docs/Development/test_logits_openai.py"
_RECORDER = """
import os
def pytest_collect_file(file_path, parent):
    relative = file_path.relative_to(parent.config.rootpath)
    if relative.parts[0] == "Docs":
        with open(os.environ["DOCS_RECORD"], "a", encoding="utf-8") as record:
            record.write(f"{relative.as_posix()}\\n")
"""


def _docs_files_visited(tmp_path: Path, *args: str) -> list[str]:
    (tmp_path / "docs_recorder.py").write_text(_RECORDER, encoding="utf-8")
    record = tmp_path / "visited.txt"
    environment = {
        name: value
        for name, value in os.environ.items()
        if name != "PYTEST_ADDOPTS" and not name.startswith("PYTEST_XDIST")
    }
    environment.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(tmp_path),
        DOCS_RECORD=str(record),
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--noconftest",
            "-p",
            "no:python",
            "-p",
            "no:cacheprovider",
            "-p",
            "docs_recorder",
            *args,
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if not record.exists():
        return []
    return record.read_text(encoding="utf-8").splitlines()


@pytest.mark.parametrize("argument", ["", "."], ids=["empty-argument", "dot"])
def test_a_root_run_visits_nothing_under_docs(tmp_path: Path, argument: str) -> None:
    assert _docs_files_visited(tmp_path, argument) == []


def test_an_explicit_path_under_docs_is_still_collected(tmp_path: Path) -> None:
    # Collection lists each directory on the way down, so siblings show too.
    assert EXPLICIT_DOCS_FILE in _docs_files_visited(tmp_path, EXPLICIT_DOCS_FILE)
