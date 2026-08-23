"""Import-closure guard: booting the app must not execute Chunking (TASK-21102).

~15k LOC of ``tldw_chatbook/Chunking`` (the shim + 28/38 vendored engine
modules, a real ``import langdetect`` attempt, an nltk ``find_spec`` path
scan, and the Internal_Prompts package) used to execute at
``import tldw_chatbook.app`` through six eager entry points, the first being
``Local_Ingestion/local_file_ingestion.py`` importing ``ENGINE_VERSION`` --
a string literal. The pin now lives in the stdlib-only
``tldw_chatbook/chunking_engine_version.py`` (outside the Chunking package,
because importing ANY ``tldw_chatbook.Chunking.*`` submodule executes the
package ``__init__`` and with it the whole engine), and every boot-path
importer was converted to lazy/function-local access.

Subprocess-isolated for the same reason as
``test_extras_import_closure.py`` (TASK-21104), whose pattern this file
follows: ``sys.modules`` is process-global, so an earlier test in the
session that legitimately imported Chunking would false-fail (or a
pre-imported app would false-pass) an in-process check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the app import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

    Returns:
        The completed process (never raises on nonzero exit).
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


_CHUNKING_CLOSURE_SNIPPET = """
import sys

import tldw_chatbook.app  # noqa: F401

resident = sorted(
    m for m in sys.modules
    if (m == "tldw_chatbook.Chunking" or m.startswith("tldw_chatbook.Chunking."))
    and sys.modules[m] is not None
)
assert "tldw_chatbook.Chunking" not in sys.modules, resident
assert not resident, f"Chunking modules resident after app import: {resident}"
assert "langdetect" not in sys.modules, "langdetect resident after app import"

# Anti-vacuity: the converted entry-point modules must still be part of the
# app's import closure. If one of them leaves the closure entirely, this
# guard would otherwise pass without testing the conversion at all.
for expected in (
    "tldw_chatbook.Local_Ingestion.local_file_ingestion",
    "tldw_chatbook.Library.ingest_preflight",
    "tldw_chatbook.Library.web_clip_request",
    "tldw_chatbook.RAG_Search",
    "tldw_chatbook.RAG_Admin.local_rag_admin_service",
    "tldw_chatbook.Media.local_media_reading_service",
):
    assert expected in sys.modules, f"expected closure member missing: {expected}"

print("CHUNKING_CLOSURE_OK")
"""


def test_app_import_does_not_execute_chunking(tmp_path: Path) -> None:
    """No ``tldw_chatbook.Chunking*`` module (nor langdetect) after app import.

    Regression guard for the TASK-21102 defect: before the fix, 39+ Chunking
    modules (shim + vendored engine) executed during ``import
    tldw_chatbook.app`` via six entry points, so this subprocess failed on
    the residency assertion.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _CHUNKING_CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        "import tldw_chatbook.app must not execute the Chunking package:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "CHUNKING_CLOSURE_OK" in result.stdout


_ENGINE_VERSION_PIN_SNIPPET = """
import sys

from tldw_chatbook.chunking_engine_version import ENGINE_VERSION

assert "tldw_chatbook.Chunking" not in sys.modules, (
    "importing the pin module must not execute the Chunking package"
)

from tldw_chatbook.Chunking.Chunk_Lib import ENGINE_VERSION as shim_version

assert ENGINE_VERSION is shim_version, (ENGINE_VERSION, shim_version)
print("ENGINE_VERSION_PIN_OK:" + ENGINE_VERSION)
"""


def test_engine_version_pin_is_chunking_free_and_single_sourced(
    tmp_path: Path,
) -> None:
    """The pin module imports without Chunking, and Chunk_Lib re-exports it.

    Two properties in one subprocess: reading ``ENGINE_VERSION`` (all the
    ingestion persist seam needs at import time) must not execute the
    Chunking package, and the shim's ``ENGINE_VERSION`` must be the SAME
    object -- one source of truth, no second copy that can drift from the
    vendored-engine pin.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _ENGINE_VERSION_PIN_SNIPPET)
    assert result.returncode == 0, (
        f"engine-version pin check failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "ENGINE_VERSION_PIN_OK:parity-1@385afa95" in result.stdout
