"""Import-weight regression guards for the local ingestion parse chain.

These tests spawn a real subprocess (mirroring
``Tests/Performance/test_app_startup_performance.py``) so the assertions
reflect a cold import rather than whatever happens to already be cached in
``sys.modules`` for the pytest worker. The module-absence assertions are the
real guard: OCR backends (docling/torch/transformers/torchvision/paddle) and
the analysis chain (nltk, via Summarization_General_Lib -> Chunk_Lib) must
not be imported just because a file was parsed -- they should only load when
an OCR backend or an LLM analysis call is actually invoked. The wall-clock
bounds are generous on purpose: they only catch catastrophic regressions
without flaking slow CI.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Top-level module names that must NOT be present in sys.modules just from
# parsing a file. Any of these appearing means something re-introduced an
# eager heavy import into the parse chain.
HEAVY_GUARDS = ("torch", "docling", "nltk", "torchvision", "transformers")


def _run_isolated_python(
    tmp_path: Path,
    code: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet with isolated Chatbook config/data directories."""

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
    if extra_env:
        env.update(extra_env)

    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_file_ingestion_import_excludes_heavy_deps(tmp_path: Path) -> None:
    """Importing the parse entry point must not drag in OCR/analysis deps.

    Regression guard for the F3 import-chain slimming: docling (and its
    torch/transformers/torchvision transitive deps) previously loaded at
    OCR backend *registration* time, and nltk (via Summarization_General_Lib)
    loaded just from importing the PDF/Document/Image/EPUB processing libs
    -- neither actually requires OCR or analysis to run.
    """

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys
        import time

        t0 = time.time()
        import tldw_chatbook.Local_Ingestion.local_file_ingestion  # noqa: F401
        dt = time.time() - t0

        guards = ("torch", "docling", "nltk", "torchvision", "transformers")
        loaded = [m for m in sys.modules if m.split(".")[0] in guards]
        print(json.dumps({"dt": dt, "loaded": sorted(set(m.split(".")[0] for m in loaded))}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["loaded"] == [], (
        f"heavy modules leaked into sys.modules just from importing "
        f"local_file_ingestion: {payload['loaded']}"
    )
    # Generous bound -- only meant to catch catastrophic regressions, not to
    # pin down exact timing on slower/loaded CI boxes.
    assert payload["dt"] < 5.0, f"local_file_ingestion import took {payload['dt']:.2f}s"


def test_ocr_backends_import_alone_excludes_heavy_deps(tmp_path: Path) -> None:
    """OCR_Backends itself must defer backend deps past module import.

    Finer-grained guard than the full parse-chain test above: anything that
    imports OCR_Backends directly (not just through local_file_ingestion)
    must not eagerly pull docling/torch/transformers just from constructing
    the module-level ``ocr_manager`` singleton.
    """

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys

        import tldw_chatbook.Local_Ingestion.OCR_Backends as ocr_backends

        guards = ("torch", "docling", "nltk", "torchvision", "transformers", "paddle", "paddleocr")
        loaded = [m for m in sys.modules if m.split(".")[0] in guards]
        print(json.dumps({
            "loaded": sorted(set(m.split(".")[0] for m in loaded)),
            "has_manager": ocr_backends.ocr_manager is not None,
        }))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["loaded"] == [], (
        f"heavy OCR backend deps leaked into sys.modules just from "
        f"importing OCR_Backends: {payload['loaded']}"
    )
    assert payload["has_manager"] is True


def test_app_import_excludes_docling_and_torchvision(tmp_path: Path) -> None:
    """App boot must not eagerly load docling/torchvision via the ingest chain.

    ``tldw_chatbook.app`` still ends up importing torch/transformers/nltk
    through unrelated subsystems (``Web_Scraping.Article_Extractor_Lib``
    imports ``Summarization_General_Lib`` eagerly at module level, and
    ``Utils.optional_deps`` performs an eager torch/transformers dependency
    probe) -- both out of scope for this ingestion import-chain slimming
    pass. docling and torchvision, however, were ONLY ever reachable through
    the ingest OCR chain (OCR_Backends -> Document_Processing_Lib), so their
    absence here is a directly attributable, verifiable regression guard.
    """

    result = _run_isolated_python(
        tmp_path,
        """
        import json
        import sys
        import time

        t0 = time.time()
        import tldw_chatbook.app  # noqa: F401
        dt = time.time() - t0

        guards = ("docling", "torchvision")
        loaded = [m for m in sys.modules if m.split(".")[0] in guards]
        print(json.dumps({"dt": dt, "loaded": sorted(set(m.split(".")[0] for m in loaded))}))
        """,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    assert payload["loaded"] == [], (
        f"docling/torchvision leaked into sys.modules just from importing "
        f"tldw_chatbook.app: {payload['loaded']}"
    )


def test_settings_modules_import_without_reading_config(tmp_path: Path) -> None:
    """TASK-36945.3 AC#3: importing the modules that used to read settings at
    import time (Document/PDF processing libs, optional_deps, rag_service, app)
    must not read settings or open the config file.

    ``tldw_chatbook.config`` itself loads the config on import, and other
    transitive imports (Chunk_Lib) read settings too, so ``tldw_chatbook.config``
    is imported first and the recorder counts only reads made from module-level
    code (or a function called directly from it) of the five modules.
    """

    result = _run_isolated_python(
        tmp_path,
        f"""
        import builtins, io, json, sys
        import tldw_chatbook.config as cfg

        FIVE = ("Document_Processing_Lib.py", "PDF_Processing_Lib.py",
                "optional_deps.py", "rag_service.py", "tldw_chatbook/app.py")
        def from_five():
            f = sys._getframe(2)
            while f is not None and ("importlib" in f.f_code.co_filename
                                     or f.f_code.co_filename.endswith("tldw_chatbook/config.py")):
                f = f.f_back
            return f is not None and f.f_code.co_filename.endswith(FIVE)

        reads = []
        for name in ("load_settings", "get_cli_setting", "get_cli_config_value",
                     "load_cli_config_and_ensure_existence"):
            orig = getattr(cfg, name, None)
            if orig is None:
                continue
            def make(orig, name):
                def wrapper(*a, **k):
                    if from_five():
                        reads.append(name)
                    return orig(*a, **k)
                return wrapper
            setattr(cfg, name, make(orig, name))

        real_open = builtins.open
        def rec_open(file, *a, **k):
            if str(file).endswith("config.toml") and from_five():
                reads.append("open:" + str(file))
            return real_open(file, *a, **k)
        builtins.open = rec_open
        io.open = rec_open

        import tldw_chatbook.Local_Ingestion.Document_Processing_Lib
        import tldw_chatbook.Local_Ingestion.PDF_Processing_Lib
        import tldw_chatbook.Utils.optional_deps
        import tldw_chatbook.RAG_Search.simplified.rag_service
        import tldw_chatbook.app
        print(json.dumps(reads))
        """,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []
