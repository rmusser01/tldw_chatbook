"""TASK-1221: the Watchlists dependency gate must name packages the code imports.

`ingest_capabilities._is_installed` requires **every** package in a feature's
`package_dependencies` to import before it reports the feature available. So a
package listed there that nothing imports is not harmless documentation — it is a
package whose absence disables Watchlists for no reason, and tells the user to
install something the feature never uses.

The check resolves imports with an AST walk rather than a text search. A bare-name
grep for `markdown` matches the comment "Extract cells from markdown table row" in
`Utils/file_extraction.py`, which is exactly how this went unnoticed: the name is
present in the tree, the import is not.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tldw_chatbook.Utils.optional_deps import OPTIONAL_FEATURES

#: PyPI name -> the module name actually imported, where they differ.
_PYPI_TO_IMPORT = {
    "beautifulsoup4": "bs4",
    "mcp[cli]": "mcp",
}

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"


def _imported_top_level_modules() -> set[str]:
    """Every top-level module name imported anywhere under tldw_chatbook/.

    Walks the AST so that only genuine ``import x`` / ``from x import y``
    statements count -- including ones nested inside functions or ``try`` blocks,
    which is where optional dependencies usually live.
    """
    imported: set[str] = set()
    for path in _PACKAGE_ROOT.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                # `level > 0` is a relative import -- never an external package.
                if node.level == 0 and node.module:
                    imported.add(node.module.split(".")[0])
    return imported


@pytest.fixture(scope="module")
def imported_modules() -> set[str]:
    return _imported_top_level_modules()


def test_the_ast_walk_actually_finds_imports(imported_modules: set[str]) -> None:
    """Guard the guard: an empty or broken walk would pass everything below."""
    assert "loguru" in imported_modules
    assert "httpx" in imported_modules
    assert len(imported_modules) > 50


def test_subscriptions_gate_lists_only_imported_packages(
    imported_modules: set[str],
) -> None:
    """Every package gating Watchlists must be imported somewhere.

    Before TASK-1221 this failed on `markdown`, `schedule` and `feedparser`:
    `FeedMonitor` parses feeds with `defusedxml`/`ElementTree` plus `bs4`, never
    feedparser, and the other two belonged to the retired briefing and scheduler
    modules.
    """
    subscriptions = OPTIONAL_FEATURES["subscriptions"]
    unimported = [
        package
        for package in subscriptions.package_dependencies
        if _PYPI_TO_IMPORT.get(package, package) not in imported_modules
    ]
    assert not unimported, (
        f"the subscriptions gate requires {unimported}, which nothing imports; "
        "a user missing any of them is told Watchlists is unavailable and asked "
        "to install a package the feature never uses"
    )


def test_the_gate_still_requires_what_watchlists_genuinely_needs(
    imported_modules: set[str],
) -> None:
    """Narrowing the gate must not empty it.

    `monitoring_engine.py` imports `bs4` at module level with no fallback, so
    losing it breaks `WatchlistCheckHandler` outright.
    """
    required = set(OPTIONAL_FEATURES["subscriptions"].package_dependencies)
    assert "beautifulsoup4" in required
    assert "bs4" in imported_modules


def test_gate_does_not_require_packages_watchlists_degrades_without() -> None:
    """`cryptography` and `defusedxml` both have working fallbacks.

    `cryptography` is absent from this project's own dev environment while
    Watchlists runs fine, so gating on it would mark a working install
    unavailable -- the same defect this task fixes, one package over. Its
    absence is a security degradation (plaintext credentials) that security.py
    already warns about at the point of use.
    """
    required = set(OPTIONAL_FEATURES["subscriptions"].package_dependencies)
    assert "cryptography" not in required
    assert "defusedxml" not in required


def test_watchlists_available_without_the_three_unused_packages(monkeypatch) -> None:
    """AC#5: an environment with bs4 and cryptography but none of the three.

    Simulated by making the removed packages un-findable and re-probing, since
    they may well be installed in the test environment.
    """
    import importlib.util

    from tldw_chatbook.Library import ingest_capabilities

    ingest_capabilities._INSTALLED_PROBE_CACHE.clear()
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"markdown", "schedule", "feedparser"}:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(ingest_capabilities.importlib.util, "find_spec", fake_find_spec)
    try:
        assert ingest_capabilities._probe_installed("subscriptions") is True, (
            "Watchlists reported unavailable without markdown/schedule/feedparser, "
            "none of which it uses"
        )
    finally:
        ingest_capabilities._INSTALLED_PROBE_CACHE.clear()


def test_missing_dependency_alert_names_only_gate_packages() -> None:
    """The alert and the gate must not be able to drift apart.

    The alert previously hardcoded its own list, which named three packages the
    gate did not require and omitted the one that breaks the feature.
    """
    import inspect

    from tldw_chatbook.Utils import widget_helpers

    source = inspect.getsource(widget_helpers.alert_subscriptions_not_available)
    assert 'OPTIONAL_FEATURES["subscriptions"].package_dependencies' in source, (
        "the alert should read the gate's packages, not repeat them"
    )
    for stale in ("markdown", "schedule", "feedparser"):
        assert f'"{stale}"' not in source.split("(TASK-1221)")[-1], (
            f"{stale} is still named as a runtime dependency in the alert"
        )
