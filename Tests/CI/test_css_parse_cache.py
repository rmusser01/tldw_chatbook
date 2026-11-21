"""The suite's global CSS parse cache still hits, and keeps few variable sets.

`Tests/UI/css_cache.py` sits in front of Textual's per-instance parse cache so
the ~1,300 app mounts of a run parse each stylesheet once (task-1459). Its key
includes the stylesheet variables, and unbounded it kept one full parse per
theme: after the theme-contrast node's 89 themes, ~10.6M objects, and every
later per-test `gc.collect()` took ~5.5 s (TASK-42144). It now keeps the
latest `_VARIABLE_SETS_KEPT` variable sets.
"""

from __future__ import annotations

import uuid

import pytest
import textual.css.stylesheet as stylesheet_module
from textual.css.stylesheet import Stylesheet

from Tests.UI import css_cache


@pytest.fixture
def parses(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every real Textual parse behind the installed global cache."""
    if not css_cache.install():
        pytest.skip("TLDW_TEST_CSS_CACHE=0 turns the cache off")
    calls: list[str] = []
    real_parse = stylesheet_module.parse

    def spy(scope, css, *args, **kwargs):
        calls.append(css)
        return real_parse(scope, css, *args, **kwargs)

    monkeypatch.setattr(stylesheet_module, "parse", spy)
    return calls


def _css() -> str:
    """CSS no earlier test in the process can have parsed."""
    return f"Static.c{uuid.uuid4().hex} {{ color: $accent; }}"


def _parse(css: str, accent: str) -> None:
    # A fresh Stylesheet each time: Textual's own cache is per instance, so
    # only the global cache can answer a repeat.
    Stylesheet(variables={"accent": accent})._parse_rules(css, ("t.tcss", ""))


def test_a_repeat_with_the_same_variables_hits_the_cache(parses: list[str]) -> None:
    css = _css()
    _parse(css, "#ff0000")
    _parse(css, "#ff0000")
    assert len(parses) == 1
    _parse(css, "#00ff00")
    assert len(parses) == 2, "other variables must not reuse the parse"


def test_only_the_latest_variable_sets_are_kept(parses: list[str]) -> None:
    css = _css()
    accents = [f"#0000{i:02x}" for i in range(css_cache._VARIABLE_SETS_KEPT + 1)]
    for accent in accents:
        _parse(css, accent)
    assert len(parses) == len(accents)
    _parse(css, accents[-1])
    assert len(parses) == len(accents), "the newest variable set is kept"
    _parse(css, accents[0])
    assert len(parses) == len(accents) + 1, "the oldest variable set is dropped"
