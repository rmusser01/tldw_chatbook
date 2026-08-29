"""The bundled tiktoken reader installs lazily, and nothing forgets it (TASK-24305).

`tldw_chatbook/__init__.py` used to call `install_tiktoken_runtime()` at package
import, which meant `import tiktoken.load` -- 19.6-29.1 ms measured -- on the
first line of every cold start, whether or not the session ever tokenised. All
four boot budget guards count `tldw_chatbook.*` modules only, so third-party
import weight was invisible to every one of them; profiling was the only way to
see it.

The shim only has to be in place before the first `get_encoding()`, so it is
armed at the call sites that tokenise. That trades one eager cost for a
distributed obligation, and the last test here is what keeps the obligation
from being quietly dropped by a new call site.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tldw_chatbook"

#: Modules allowed to import tiktoken without arming the runtime themselves.
#: Only the shim, which IS the arming mechanism.
_ARMING_EXEMPT = {PACKAGE_ROOT / "Utils" / "tiktoken_runtime.py"}


def test_importing_the_package_does_not_import_tiktoken() -> None:
    """A cold start pays nothing for a library it may never use.

    Run in a subprocess: this test session has almost certainly imported
    tiktoken already, so an in-process `sys.modules` check would be vacuous.
    """
    code = (
        "import sys, os, tempfile\n"
        "os.environ['HOME'] = tempfile.mkdtemp()\n"
        "os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp()\n"
        "os.environ['TLDW_TEST_MODE'] = '1'\n"
        "import tldw_chatbook\n"
        "import tldw_chatbook.app\n"
        "print('TIKTOKEN' if 'tiktoken' in sys.modules else 'CLEAN')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "CLEAN", (
        "importing tldw_chatbook.app pulled tiktoken into sys.modules; the "
        "eager import is back on the cold-start path (TASK-24305)."
    )


def test_the_bundled_tables_are_in_force_before_the_first_encoding() -> None:
    """Arming installs the offline bundle, and an encoding resolves from it."""
    code = (
        "import os, sys, tempfile\n"
        "os.environ['HOME'] = tempfile.mkdtemp()\n"
        "os.environ['XDG_CONFIG_HOME'] = tempfile.mkdtemp()\n"
        "os.environ['TLDW_TEST_MODE'] = '1'\n"
        "os.environ.pop('TIKTOKEN_CACHE_DIR', None)\n"
        "os.environ.pop('DATA_GYM_CACHE_DIR', None)\n"
        "from tldw_chatbook.Utils.tiktoken_runtime import ensure_tiktoken_runtime\n"
        "ensure_tiktoken_runtime()\n"
        "import tiktoken\n"
        "enc = tiktoken.get_encoding('cl100k_base')\n"
        "bundled = 'tldw_chatbook' in os.environ.get('TIKTOKEN_CACHE_DIR', '')\n"
        "print('OK' if enc.encode('hello world') and bundled else 'BAD')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )

    if "ModuleNotFoundError" in result.stderr and "tiktoken" in result.stderr:
        pytest.skip("tiktoken is not installed in this environment")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "OK", (
        "the bundled offline table reader was not in force for the first "
        "encoding; lazy arming has broken the offline guarantee."
    )


def test_arming_is_idempotent_under_concurrent_first_use() -> None:
    """Several workers racing into their first encode install exactly once."""
    import threading

    from tldw_chatbook.Utils import tiktoken_runtime

    calls = {"count": 0}
    real_install = tiktoken_runtime.install_tiktoken_runtime

    def counted() -> None:
        calls["count"] += 1
        real_install()

    original_installed = tiktoken_runtime._INSTALLED
    tiktoken_runtime.install_tiktoken_runtime = counted  # type: ignore[assignment]
    tiktoken_runtime._INSTALLED = False
    try:
        start = threading.Barrier(8)

        def worker() -> None:
            start.wait()
            tiktoken_runtime.ensure_tiktoken_runtime()

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        tiktoken_runtime.install_tiktoken_runtime = real_install  # type: ignore[assignment]
        tiktoken_runtime._INSTALLED = original_installed

    assert calls["count"] == 1, (
        f"{calls['count']} installs across 8 concurrent first-uses; arming is "
        "not idempotent under a race."
    )


def _modules_importing_tiktoken() -> list[Path]:
    """Every package module that imports tiktoken, by AST rather than grep."""
    found: list[Path] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "tiktoken" or alias.name.startswith("tiktoken.")
                for alias in node.names
            ):
                found.append(path)
                break
            if isinstance(node, ast.ImportFrom) and (
                node.module == "tiktoken"
                or (node.module or "").startswith("tiktoken.")
            ):
                found.append(path)
                break
    return found


def test_every_tiktoken_importer_arms_the_bundled_runtime() -> None:
    """A new tokenising module cannot silently skip the offline bundle.

    Deferring the install traded one eager import for an obligation spread
    across call sites. Without this guard, a future module that imports
    tiktoken directly would reach upstream for its tables -- a network read
    where there used to be none -- and nothing would say so.
    """
    offenders: list[str] = []
    for path in _modules_importing_tiktoken():
        if path in _ARMING_EXEMPT:
            continue
        source = path.read_text(encoding="utf-8")
        if "ensure_tiktoken_runtime" not in source:
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "these modules import tiktoken without arming the bundled offline "
        f"table reader: {offenders}. Call "
        "`tldw_chatbook.Utils.tiktoken_runtime.ensure_tiktoken_runtime()` "
        "before the first encoding is fetched (TASK-24305)."
    )


def test_the_package_entrypoint_no_longer_installs_eagerly() -> None:
    """The regression is a one-line revert, so name it explicitly."""
    source = (PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    eager_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.endswith("install_tiktoken_runtime")
    ]

    assert not eager_calls, (
        "tldw_chatbook/__init__.py calls install_tiktoken_runtime() at import "
        "time again; that is 19.6-29.1 ms on every cold start, invisible to "
        "all four boot budgets because they count tldw_chatbook modules only."
    )
