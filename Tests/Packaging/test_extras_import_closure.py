"""Import-closure guards for extras-gated packages (TASK-21104).

`Subscriptions/monitoring_engine.py` used to do a module-level
`from bs4 import BeautifulSoup` while `beautifulsoup4` exists only under
`[project.optional-dependencies]` in pyproject.toml. The module is imported
eagerly at boot (app.py -> Scheduling/scheduler/handlers ->
watchlist_check_handler -> monitoring_engine), so an install without the
extra could not `import tldw_chatbook.app` at all, and every configured
install paid the bs4+soupsieve import at boot. (In practice the crash was
masked on healthy installs because the CORE dependency `markdownify` pins
`beautifulsoup4<5,>=4.9` transitively -- but nothing in the project's own
dependency declarations guarantees bs4, so boot must not rely on it.)

Two guards, so the NEXT unguarded optional import is caught at PR time:

* `test_app_imports_with_bs4_absent` simulates a bs4-less install with a
  meta-path blocker in a fresh subprocess and asserts the app still imports.
* `test_app_import_closure_excludes_extras_only_packages` asserts that a
  plain `import tldw_chatbook.app` loads NONE of an explicit list of
  extras-only packages.

Subprocesses are used because `sys.modules` is process-global: an earlier
test in the session may already have imported bs4, which would false-pass
an in-process closure check (same rationale as
`Tests/Performance/test_app_import_weight.py`, whose isolation pattern this
file follows).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

# Extras-only packages that must NEVER be resident after a plain
# `import tldw_chatbook.app`. Every entry is (pyproject/pip name, importable
# module name) and was verified extras-only against pyproject.toml's
# [project.optional-dependencies] at the time of writing (none appear in the
# core `dependencies` list). soupsieve is bs4's own hard dependency, listed
# so a sneaky `import soupsieve` cannot ride in either.
#
# Deliberately EXCLUDED, with reasons:
# * defusedxml, pillow (PIL), rich-pixels, textual-image -- CORE deps in
#   pyproject `dependencies`; they are legitimately resident at boot even
#   though optional_deps.py also tracks their availability.
# * torch, transformers, nltk, scipy, sklearn, pandas-as-heavy-dep --
#   already guarded by Tests/Performance/test_app_import_weight.py
#   (task 163); pandas stays here too because it is ALSO extras-only.
EXTRAS_ONLY_MODULES: tuple[tuple[str, str], ...] = (
    ("beautifulsoup4", "bs4"),
    ("soupsieve (beautifulsoup4 dep)", "soupsieve"),
    ("lxml", "lxml"),
    ("pandas", "pandas"),
    ("playwright", "playwright"),
    ("trafilatura", "trafilatura"),
    ("aiohttp", "aiohttp"),
    ("chromadb", "chromadb"),
    ("pymupdf", "fitz"),
    ("pymupdf4llm", "pymupdf4llm"),
    ("ebooklib", "ebooklib"),
    ("html2text", "html2text"),
    ("markdown", "markdown"),
    ("schedule", "schedule"),
    ("feedparser", "feedparser"),
    ("textual-serve", "textual_serve"),
    ("mcp-unified", "mcp"),
)


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

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


_BS4_ABSENT_SNIPPET = """
import importlib.abc
import sys


class _Blocker(importlib.abc.MetaPathFinder):
    BLOCKED = ("bs4", "soupsieve")

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"blocked by test: {fullname}")
        return None


sys.meta_path.insert(0, _Blocker())

import tldw_chatbook.app

leaked = sorted(
    m for m in sys.modules
    if m.split(".")[0] in _Blocker.BLOCKED and sys.modules[m] is not None
)
assert not leaked, f"blocked modules leaked into sys.modules: {leaked}"
print("BS4_ABSENT_IMPORT_OK")
"""


def test_app_imports_with_bs4_absent(tmp_path: Path) -> None:
    """A base (no-extras) install must be able to import tldw_chatbook.app.

    Regression guard for the TASK-21104 defect: before the fix this failed
    with `ImportError: blocked by test: bs4` raised from
    `Subscriptions/monitoring_engine.py`'s module-level bs4 import, reached
    via app.py's eager scheduler-handler imports.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _BS4_ABSENT_SNIPPET)
    assert result.returncode == 0, (
        "import tldw_chatbook.app must survive beautifulsoup4 being absent, "
        f"but the bs4-blocked subprocess failed:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "BS4_ABSENT_IMPORT_OK" in result.stdout


_CLOSURE_SNIPPET = (
    "import json, sys\n"
    "import tldw_chatbook.app\n"
    f"candidates = {sorted(module for _pip_name, module in EXTRAS_ONLY_MODULES)!r}\n"
    "resident = sorted(m for m in candidates if sys.modules.get(m) is not None)\n"
    "print('CLOSURE:' + json.dumps(resident))\n"
)


def test_app_import_closure_excludes_extras_only_packages(tmp_path: Path) -> None:
    """No extras-only package may be resident after `import tldw_chatbook.app`.

    Catches the NEXT unguarded optional import at PR time: any module-scope
    `import <extras-only package>` reachable from the app's import closure
    turns up in this list. For packages not installed in the running
    environment the check is trivially true (they could not have imported),
    which is the honest scope: the guard bites exactly on the dev/CI
    environments that have extras installed, where an unguarded import would
    otherwise go unnoticed until a base install crashed.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        f"import tldw_chatbook.app failed in isolated subprocess:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    closure_lines = [
        line for line in result.stdout.splitlines() if line.startswith("CLOSURE:")
    ]
    assert closure_lines, f"closure marker missing from stdout: {result.stdout!r}"
    resident = json.loads(closure_lines[-1][len("CLOSURE:") :])
    assert resident == [], (
        "import tldw_chatbook.app eagerly imported extras-only packages "
        f"{resident}; either guard the import (see "
        "Subscriptions/monitoring_engine.py's _require_beautifulsoup for the "
        "pattern) or promote the package to core dependencies AND remove it "
        "from EXTRAS_ONLY_MODULES with a comment."
    )


def test_extract_text_from_html_degrades_actionably_without_bs4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With bs4 missing, HTML extraction raises an actionable install hint.

    The monitors' per-check exception handling records the error against the
    subscription, so this message is what the user sees -- it must name the
    package and the exact install command, not be a bare ImportError or a
    silent no-op.

    Args:
        monkeypatch: pytest fixture; poisons sys.modules['bs4'] (None makes
            any `import bs4` raise) and restores it afterwards.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor

    monkeypatch.setitem(sys.modules, "bs4", None)

    with pytest.raises(ImportError) as excinfo:
        ContentExtractor.extract_text_from_html("<p>hello</p>")

    message = str(excinfo.value)
    assert "beautifulsoup4" in message
    assert "pip install tldw_chatbook[subscriptions]" in message


def test_extract_text_from_html_works_with_bs4_present() -> None:
    """The lazy resolution must still parse HTML once bs4 is importable."""
    pytest.importorskip("bs4")
    from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor

    text = ContentExtractor.extract_text_from_html(
        "<p>hello <b>world</b></p><script>ignored()</script>"
    )
    assert text == "hello world"
    assert "ignored" not in text
