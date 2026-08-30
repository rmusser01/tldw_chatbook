"""Guards for the ordered-candidate fast path on ``Stylesheet.apply``.

The fast path (``tldw_chatbook/Utils/textual_css_fastpath.py``) exists because
upstream ``apply`` recovers source order with
``filter(limit_rules.__contains__, reversed(self.rules))`` -- a walk of the
entire rule list on every call, which made styling one node cost
``O(total rules)``. On dev ``bc1e26ce60`` that was 7,335,029 ``RuleSet.__hash__``
calls in a single Console screen switch -- ~4,400 per apply over 1,667 applies,
of which 4,324 are the scan itself -- and CSS matching was the #1 sampled frame
during 399 ms event-loop stalls.

Two things have to stay true for that optimisation to be safe, and each has a
test here:

1. **It must not change what styling means.** ``test_fastpath_computes_identical
   _styles_for_every_node`` applies the stylesheet to every node of several real
   screens both ways and compares the resulting rule maps. This is the test that
   would catch a specificity or tie-breaking regression.
2. **It must keep matching the upstream implementation it delegates to.** The
   fast path hands upstream a pre-narrowed ``_rules`` list and relies on
   upstream reversing it. If upstream's candidate-selection line changes, that
   contract is void -- ``test_upstream_apply_still_has_the_shape_the_fastpath
   _assumes`` fails loudly on a Textual upgrade instead of letting styling drift
   silently.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from textual.css.stylesheet import Stylesheet


def _scratch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home, data, config = tmp_path / "home", tmp_path / "data", tmp_path / "config"
    for sub in (home, data, config):
        sub.mkdir(parents=True, exist_ok=True)
    config_file = config / "tldw_cli" / "config.toml"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[first_run]\nsetup_completed = true\n\n[splash_screen]\nenabled = false\n"
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_file))
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "css_fastpath")


def test_upstream_apply_still_has_the_shape_the_fastpath_assumes() -> None:
    """Pin the two upstream behaviours the delegation depends on.

    The fast path narrows ``self._rules`` to the candidate set in ascending
    source order and lets upstream do the rest. That is only equivalent while
    upstream (a) derives its candidates from ``rules_map`` keyed on
    ``node._selector_names``, and (b) recovers order by reversing
    ``self.rules``. Both appear as literal source below; if a Textual upgrade
    changes either, this test fails and the delegation must be re-reviewed.
    """
    source = inspect.getsource(
        getattr(Stylesheet.apply, "__wrapped__", Stylesheet.apply)
    )
    normalised = " ".join(source.split())

    assert "rules_map.keys() & node._selector_names" in normalised, (
        "Upstream Stylesheet.apply no longer derives candidates from "
        "rules_map & node._selector_names. The fast path in "
        "tldw_chatbook/Utils/textual_css_fastpath.py pre-computes exactly that "
        "set and must be re-reviewed against the new implementation."
    )
    assert "filter(limit_rules.__contains__, reversed(self.rules))" in normalised, (
        "Upstream Stylesheet.apply no longer recovers rule order by reversing "
        "self.rules. The fast path hands it a pre-ordered narrowed list and "
        "relies on that reversal; re-review "
        "tldw_chatbook/Utils/textual_css_fastpath.py before shipping."
    )


@pytest.mark.ui
@pytest.mark.asyncio
async def test_fastpath_computes_identical_styles_for_every_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every node on every toured screen resolves to the same rules both ways.

    This is the fidelity check. A specificity or source-order regression in the
    fast path shows up here as a differing rule map, naming the node.
    """
    _scratch_env(monkeypatch, tmp_path)
    from tldw_chatbook.Utils.textual_css_fastpath import (
        install_stylesheet_fastpath,
        is_installed,
    )
    from tldw_chatbook.app import TldwCli

    install_stylesheet_fastpath()
    assert is_installed(), "fast path failed to install"

    patched_apply = Stylesheet.apply
    upstream_apply = getattr(patched_apply, "__wrapped__", None)
    assert upstream_apply is not None, "fast path did not record the upstream apply"

    app = TldwCli()
    async with app.run_test(size=(170, 48)) as pilot:
        for _ in range(20):
            await asyncio.sleep(0.05)
            await pilot.pause()

        mismatches: list[str] = []
        checked = 0
        for key, expected in (
            ("ctrl+2", "ChatScreen"),
            ("ctrl+3", "LibraryScreen"),
            ("f9", "SettingsScreen"),
        ):
            await pilot.press(key)
            for _ in range(40):
                await pilot.pause()
                if type(pilot.app.screen).__name__ == expected:
                    break
            for _ in range(6):
                await asyncio.sleep(0.05)
                await pilot.pause()

            stylesheet = pilot.app.stylesheet
            for node in list(pilot.app.screen.query("*")):
                upstream_apply(stylesheet, node, animate=False)
                expected_rules = dict(node.styles.base.get_rules())
                patched_apply(stylesheet, node, animate=False)
                actual_rules = dict(node.styles.base.get_rules())
                checked += 1
                if expected_rules != actual_rules:
                    differing = {
                        name
                        for name in expected_rules.keys() | actual_rules.keys()
                        if expected_rules.get(name) != actual_rules.get(name)
                    }
                    mismatches.append(
                        f"{expected}: {type(node).__name__}"
                        f"#{node.id or '-'} differs on {sorted(differing)}"
                    )

        assert checked > 500, (
            f"only {checked} nodes compared -- the tour did not build real "
            "screens, so a passing result would prove nothing"
        )
        assert not mismatches, (
            f"{len(mismatches)} of {checked} nodes resolved to different styles "
            "under the fast path:\n" + "\n".join(mismatches[:20])
        )
