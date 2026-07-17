# Tests/Utils/test_optional_import_deferral.py
"""
Regression tests for task-257: defer optional-feature imports (chromadb,
the web-search chain [playwright/trafilatura], and PDF/document/ebook
per-format processors) out of `import tldw_chatbook.app`'s hot path.

Before this task, a plain `import tldw_chatbook.app` eagerly paid for three
chains (~550ms combined, see Docs/Design/2026-07-16-performance-audit.md
S2 C2) even when the corresponding feature was never used in the session:

1. `Embeddings/Chroma_Lib.py` did module-scope `get_safe_import('chromadb')`
   (a REAL import), reachable via `RAG_Admin/local_rag_admin_service.py`
   which `app.py` imports unconditionally (~154ms: chromadb -> OTel -> gRPC
   -> protobuf). Initially deferred into `ChromaDBManager.__init__`;
   task-248 then removed `Chroma_Lib.py` entirely (the surviving chromadb
   consumer, `RAG_Search/simplified/vector_store.py`, already imports
   chromadb lazily in its `client` property), so only the subprocess
   sys.modules guard below still covers this chain.
2. `Tools/__init__.py` eagerly imported `WebSearchTool` (and other optional
   tool classes), which pulled in `Web_Scraping/Article_Extractor_Lib.py`'s
   module-scope playwright + trafilatura imports (~197ms), even though
   `web_search_enabled` defaults to False. `Tools/__init__.py` now uses a
   PEP 562 `__getattr__` lazy re-export layer, and Article_Extractor_Lib's
   playwright/trafilatura imports are deferred behind
   `_ensure_playwright()`/`_ensure_trafilatura()` helpers (mirroring that
   file's pre-existing pandas `find_spec` deferral).
3. `app.py` imports `Local_Ingestion.local_file_ingestion` directly (for
   `classify_ingest_source`/`persist_parsed_media`), which bypasses
   `Local_Ingestion`'s own lazy `__init__.py` -- standard Python import
   semantics run `local_file_ingestion.py`'s module body regardless of
   which name was requested. Its top-of-file imports of
   `process_pdf`/`process_document`/`process_ebook`/`LocalAudioProcessor`/
   `LocalVideoProcessor` (~230ms: pymupdf/onnxruntime + Document/ebook
   stack) are now deferred behind module-level `_ensure_*()` helpers called
   from `parse_local_file_for_ingest()`'s per-`file_type` branches.

This file has two kinds of coverage:
  - Subprocess-based `sys.modules` assertions after a plain
    `import tldw_chatbook.app`, in an isolated HOME/XDG sandbox (never the
    real `~/.config/tldw_cli`).
  - In-process functional smoke tests proving each chain still works when
    actually used, plus one graceful-absence (simulated missing dependency)
    case.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The five sys.modules names named in the task-257 AC. None of these may be
# resident after a plain `import tldw_chatbook.app`.
DEFERRED_MODULES = (
    "chromadb",
    "playwright",
    "trafilatura",
    "pymupdf",
    "fitz",
    "dateparser",
)


def _run_isolated_python(tmp_path: Path, code: str) -> "subprocess.CompletedProcess[str]":
    """Run a Python snippet in a fresh interpreter with isolated config/data
    dirs -- NEVER the real `~/.config/tldw_cli`.

    A fresh interpreter is required because `sys.modules` is process-global:
    once chromadb/playwright/etc. are imported by anything else (e.g. an
    earlier test in the same pytest session), they would stay cached
    in-process and this guard would give a false pass.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's
            HOME/XDG_CONFIG_HOME/XDG_DATA_HOME.
        code: Python source to run via `python -c`.
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

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


# --- 1. sys.modules assertions after `import tldw_chatbook.app` -------------

_MEASURE_SNIPPET = """
import json
import sys

import tldw_chatbook.app

deferred = {deferred_modules!r}
loaded = sorted(m for m in deferred if m in sys.modules)
print(json.dumps({{"loaded": loaded}}))
""".format(deferred_modules=DEFERRED_MODULES)


def test_app_import_does_not_load_any_deferred_module(tmp_path: Path) -> None:
    """Plain `import tldw_chatbook.app` must not load chromadb, playwright,
    trafilatura, pymupdf/fitz, or dateparser.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _MEASURE_SNIPPET)
    assert result.returncode == 0, (
        f"import tldw_chatbook.app failed in isolated subprocess:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    last_line = result.stdout.strip().splitlines()[-1]
    payload = json.loads(last_line)
    assert payload["loaded"] == [], (
        f"import tldw_chatbook.app eagerly loaded deferred modules: {payload['loaded']}"
    )


@pytest.mark.parametrize("module_name", DEFERRED_MODULES)
def test_app_import_does_not_load_module_individually(tmp_path: Path, module_name: str) -> None:
    """Per-module variant of the guard above, so a regression in any single
    chain produces an obviously-scoped failure message.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
        module_name: one of DEFERRED_MODULES.
    """
    result = _run_isolated_python(tmp_path, _MEASURE_SNIPPET)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert module_name not in payload["loaded"], (
        f"import tldw_chatbook.app eagerly loaded {module_name!r}"
    )


# --- 2. Functional smoke: chromadb chain ------------------------------------
# (The ChromaDBManager construction/gate smoke tests that lived here were
# removed with `Embeddings/Chroma_Lib.py` itself in task-248; the surviving
# chromadb chain -- vector_store.ChromaVectorStore's lazy `client` property --
# is covered by Tests/RAG/simplified/test_vector_stores.py and the
# sys.modules guards in section 1.)


# --- 3. Functional smoke: web-search chain (Tools.__getattr__) --------------


def test_tools_getattr_resolves_web_search_tool() -> None:
    """`tldw_chatbook.Tools.WebSearchTool` must still resolve via the PEP 562
    `__getattr__` lazy layer, from its real defining submodule."""
    import tldw_chatbook.Tools as Tools

    web_search_tool_cls = Tools.WebSearchTool
    assert web_search_tool_cls is not None
    assert web_search_tool_cls.__module__ == "tldw_chatbook.Tools.web_search_tool"
    assert web_search_tool_cls.__name__ == "WebSearchTool"
    # Cached on the module after first resolution (no repeated __getattr__).
    assert "WebSearchTool" in vars(Tools)


def test_tools_dunder_all_names_all_resolve() -> None:
    """Every name in Tools.__all__ (the lazy re-export surface) must resolve
    with no exceptions -- regression net against a stale `_SUBMODULE_BY_NAME`
    mapping."""
    import tldw_chatbook.Tools as Tools

    failed = []
    for name in Tools.__all__:
        try:
            getattr(Tools, name)
        except Exception as exc:  # pragma: no cover - failure path
            failed.append((name, repr(exc)))
    assert not failed, f"{len(failed)} of {len(Tools.__all__)} names failed to resolve: {failed}"


def test_tools_unknown_attribute_raises_attribute_error() -> None:
    """Unknown attributes must raise AttributeError, per PEP 562."""
    import tldw_chatbook.Tools as Tools

    with pytest.raises(AttributeError, match="tldw_chatbook.Tools"):
        getattr(Tools, "NoSuchToolNameXYZ")


def test_scrape_article_degrades_gracefully_when_playwright_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graceful-absence case: simulate a broken/partial playwright install
    (module directory present so `find_spec` succeeds, but the real import
    raises) via `builtins.__import__`, matching the existing
    `TestParseCsvUrlsPandasGuard` pattern for pandas in
    Tests/Web_Scraping/test_article_extractor.py. `scrape_article()` must
    degrade to its existing "Playwright not installed" error dict rather
    than raising or crashing.

    Args:
        monkeypatch: pytest fixture; used to force `PLAYWRIGHT_AVAILABLE`
            True and to make `import playwright...` raise ImportError.
    """
    import tldw_chatbook.Web_Scraping.Article_Extractor_Lib as mod

    monkeypatch.setattr(mod, "PLAYWRIGHT_AVAILABLE", True)
    monkeypatch.setattr(mod, "async_playwright", None)
    monkeypatch.setattr(mod, "sync_playwright", None)

    real_import = builtins.__import__

    def _fail_playwright(name, *args, **kwargs):
        if name == "playwright" or name.startswith("playwright."):
            raise ImportError("simulated broken playwright install")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_playwright)

    result = asyncio.run(mod.scrape_article("https://example.com/some-article"))

    assert result["extraction_successful"] is False
    assert result["error"] == "Playwright not installed"


# --- 4. Functional smoke: PDF/document processors (dispatch-time import) ----


def test_parse_local_file_for_ingest_processes_plaintext_without_optional_deps(tmp_path: Path) -> None:
    """The plaintext dispatch branch needs none of the five deferred
    per-format processors (process_pdf/process_document/process_ebook/
    LocalAudioProcessor/LocalVideoProcessor) -- it must work standalone.

    Note: this does NOT also assert `'pymupdf' not in sys.modules` here --
    that would be order-dependent within a shared pytest process (an
    unrelated earlier test in the same session may have already imported
    pymupdf for its own reasons, e.g. Tests/Utils/test_optional_deps.py's
    dependency-check tests). That exact assertion is covered correctly, in
    isolation, by test_app_import_does_not_load_any_deferred_module above.

    Args:
        tmp_path: pytest fixture; holds the small .txt fixture file.
    """
    from tldw_chatbook.Local_Ingestion.local_file_ingestion import parse_local_file_for_ingest

    source = tmp_path / "note.txt"
    source.write_text("Deferred-import smoke test content.", encoding="utf-8")

    payload = parse_local_file_for_ingest(str(source), {"perform_analysis": False})

    assert payload["content"] == "Deferred-import smoke test content."


def test_parse_local_file_for_ingest_pdf_branch_still_resolves_process_pdf(tmp_path: Path) -> None:
    """The pdf dispatch branch must still resolve and call the real
    `process_pdf` via `_ensure_process_pdf()` when a `.pdf` file is routed
    to it -- proving the deferral didn't break the wiring, without needing
    a real PDF fixture (process_pdf itself is monkeypatched, matching the
    existing `test_process_pdf_error_key_presence_vs_truthiness`-style
    pattern in Tests/Local_Ingestion/test_ingest_parse_worker.py).

    Args:
        tmp_path: pytest fixture; holds the (stub-content) .pdf fixture file.
    """
    import tldw_chatbook.Local_Ingestion.local_file_ingestion as lfi

    source = tmp_path / "doc.pdf"
    source.write_bytes(b"%PDF-1.4 stub bytes, never actually parsed (process_pdf is mocked).")

    calls = []

    def _fake_process_pdf(**kwargs):
        calls.append(kwargs)
        return {
            "status": "Success",
            "content": "Fake PDF content.",
            "title": "Fake title",
            "author": "Fake author",
            "keywords": [],
            "chunks": [],
            "analysis": "",
            "metadata": {},
            "error": None,
        }

    # local_file_ingestion.process_pdf starts as the module-level `None`
    # placeholder; _ensure_process_pdf() must not clobber an already-bound
    # (here, test-provided) value -- same contract exercised by
    # Tests/Local_Ingestion/test_ingest_parse_worker.py's monkeypatch of
    # this exact attribute.
    original = lfi.process_pdf
    lfi.process_pdf = _fake_process_pdf
    try:
        payload = lfi.parse_local_file_for_ingest(str(source), {"perform_analysis": False})
    finally:
        lfi.process_pdf = original

    assert len(calls) == 1
    assert payload["content"] == "Fake PDF content."


# --- 5. Graceful-absence: simulated missing chromadb -------------------------
# (Removed with `Embeddings/Chroma_Lib.py` in task-248; the graceful-absence
# behavior of the surviving chain is ChromaVectorStore.client's ImportError
# message, exercised by the RAG vector-store tests.)
