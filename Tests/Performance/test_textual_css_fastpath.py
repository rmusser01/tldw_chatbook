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


def test_ancestor_requirements_only_claims_what_it_can_prove() -> None:
    """The rejection filter must be conservative in every ambiguous case.

    `rules_map` keys a rule under its RIGHTMOST selector, so `#panel Button`
    is a candidate for every Button in the app. The fast path rejects such a
    rule when the node has no ancestor carrying the leading name -- which is
    only sound if "no requirement" is reported for every shape the parser can
    produce that does NOT imply an ancestor.

    Each case below is a shape that previously either cost performance (a
    real requirement reported as None) or would corrupt styling (a
    non-requirement reported as a requirement). The second kind is the
    dangerous one: `Button.foo` is ONE compound, so demanding an ancestor
    `.foo` would reject the very node that matches.
    """
    from textual.css.parse import parse_selectors
    from textual.css.model import RuleSet
    from tldw_chatbook.Utils.textual_css_fastpath import _ancestor_requirements

    def requirements_for(css_selector: str):
        rule = RuleSet(list(parse_selectors(css_selector)))
        rule._post_parse()
        return _ancestor_requirements(rule)

    # An ancestor IS implied -- these are the ones worth rejecting.
    assert requirements_for("#panel Button") == ("#panel",)
    assert requirements_for(".message-actions Button") == (".message-actions",)
    assert requirements_for("#panel > Button") == ("#panel",)
    assert requirements_for("#panel Button:disabled") == ("#panel",)

    # No ancestor implied -- must be None, or styling breaks.
    assert requirements_for("Button") == (None,)
    assert requirements_for("#panel") == (None,)
    assert requirements_for("Button.foo") == (None,), (
        "one compound: the class is on the SUBJECT, not an ancestor"
    )
    assert requirements_for("Button#thing.foo") == (None,)
    # These two are the shapes that actually exercise the one-compound
    # guard: a single compound LED by an id/class. `Button.foo` above does
    # not -- its first selector is a TYPE, so it reports None either way,
    # and it passed even with the guard deleted.
    assert requirements_for("#thing.foo") == (None,), (
        "single compound led by an id: the id is on the SUBJECT; demanding "
        "it of an ancestor rejects the very node that matches"
    )
    assert requirements_for(".foo.bar") == (None,)
    assert requirements_for("Widget Button") == (None,), (
        "a leading TYPE needs MRO matching, which this filter does not do"
    )

    # A rule matches if ANY selector set matches, so a comma-separated rule is
    # only rejectable when EVERY set states an unmet requirement.
    assert requirements_for("#panel Button, Button") == ("#panel", None)
    assert requirements_for("#a Button, #b Button") == ("#a", "#b")

    # A rule with no selector sets states nothing, so it must be KEPT. The
    # rejection loop iterates the requirements and appends on the first
    # keepable one -- with an empty tuple it never appends, which would drop
    # the rule silently. Rejection has to be a decision, not a fall-through.
    from tldw_chatbook.Utils.textual_css_fastpath import _ordered_candidates

    empty = RuleSet([])
    empty._post_parse()
    assert _ancestor_requirements(empty) == ()


@pytest.mark.ui
@pytest.mark.asyncio
async def test_filter_follows_a_class_added_to_an_ancestor_at_runtime() -> None:
    """A rule scoped under `.foo` must apply the moment an ancestor gains it.

    The ancestor-name set is recomputed on every apply rather than cached,
    because ancestors gain and lose classes constantly at runtime
    (`-active`, `hidden`, ...). Caching it is the obvious "optimisation"
    here, and it would silently drop styles that have just become
    applicable -- a bug that no static fixture would catch, because it only
    appears after a class toggle.

    Asserts both directions: the scoped rule becomes a candidate when the
    ancestor gains the class, and stops being one when it loses it.
    """
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Utils.textual_css_fastpath import _ordered_candidates

    app = _build_test_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        for _ in range(60):
            await pilot.pause(0.05)
            if len(list(app.screen.walk_children(with_self=True))) > 50:
                break
        nodes = list(app.screen.walk_children(with_self=True))
        assert len(nodes) > 50, (
            f"only {len(nodes)} nodes -- still on the splash screen, so a "
            "passing result would prove nothing"
        )

        target = next(
            (
                node
                for node in nodes
                if node.parent is not None
                and getattr(node, "_selector_names", None)
            ),
            None,
        )
        assert target is not None, "no node with a parent to decorate"
        parent = target.parent
        marker = "tldw-fastpath-runtime-marker"

        stylesheet = app.stylesheet
        stylesheet.add_source(
            f".{marker} {type(target).__name__} {{ color: red; }}",
            read_from=("test_filter_follows_runtime_class", ""),
        )
        stylesheet.parse()

        without = _ordered_candidates(stylesheet, target)
        parent.add_class(marker)
        with_class = _ordered_candidates(stylesheet, target)
        parent.remove_class(marker)
        after = _ordered_candidates(stylesheet, target)

        assert len(with_class) == len(without) + 1, (
            "adding the class to an ANCESTOR did not make the scoped rule a "
            f"candidate ({len(without)} -> {len(with_class)}); the filter is "
            "treating ancestor names as static and would drop styles that "
            "become applicable at runtime"
        )
        assert len(after) == len(without), (
            f"removing the ancestor class did not withdraw the rule "
            f"({len(with_class)} -> {len(after)})"
        )


#: TASK-25810 ratchet: ancestor-scoped rules whose SUBJECT is a bare common
#: type. Textual indexes each rule under its rightmost selector only, so
#: every one of these is a candidate for every widget of that type in the
#: app -- `#panel Button` costs all ~110 live Buttons a full selector
#: evaluation (the ancestor filter rejects most cheaply, but the candidate
#: set is still built). Measured 2026-08-30: these rules were 93% of all
#: per-node candidate work on a 502-node Console.
#:
#: NEVER RAISE this constant (ADR-097's ratchet discipline; the CSS byte
#: budget's history is three cycles of silent regrowth). On a breach:
#: re-key the new rule -- give its subject a class carried only by the
#: intended widgets (`#panel Button` -> `Button.panel-action`) -- instead of
#: widening the budget. When re-keying work lands, LOWER it to the new count.
#:
#: Pinned 2026-08-31 at measured 274 + 10 slack (ADR-097's convention:
#: enough headroom that one ordinary PR does not red the build, little
#: enough that regrowth forces the re-keying conversation). TIGHTENED
#: 2026-09-01 after the TASK-25812 split merged (`b62407e258`) and its
#: vocabulary pinning reorganised ten of these rules away: measured 264,
#: re-pinned at 264 + 10 so the freed headroom is banked, per the same
#: convention.
MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES = 274

#: Anti-vacuity floor: the census walking a hollow stylesheet (bundle
#: missing, parse failure swallowed) must fail loudly, not pass at zero.
MIN_ANCESTOR_SCOPED_BARE_TYPE_RULES = 150

#: The type keys that are worth guarding: common enough that one scoped rule
#: taxes dozens-to-hundreds of live nodes. (`Widget` is excluded -- its one
#: rule is upstream's, and rare bespoke types are self-limiting.)
_GUARDED_TYPE_KEYS = (
    "Button",
    "Static",
    "Input",
    "Select",
    "Checkbox",
    "Vertical",
    "Horizontal",
    "VerticalScroll",
)


@pytest.mark.ui
@pytest.mark.asyncio
async def test_ancestor_scoped_bare_type_rule_count_is_a_ratchet() -> None:
    """New CSS must not grow the bare-type-subject candidate tax.

    Counts, in the app's real parsed stylesheet, the rules that are (a)
    indexed under one of the common bare type keys and (b) ancestor-scoped
    (some selector set has a descendant/child hop, so the type is the
    subject and the scope rides an ancestor). These are exactly the rules
    the rightmost-selector index over-distributes.

    Counting the PARSED stylesheet rather than grepping .tcss text is
    deliberate: a 2026-08-29 dead-CSS sweep built on a text regex
    mis-parsed every selector and would have deleted live CSS. The parser
    is the only honest tokenizer for its own syntax.
    """
    from collections import Counter

    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        rules_map = app.stylesheet.rules_map

        per_key: Counter = Counter()
        offenders: list[str] = []
        counted: set[int] = set()
        for key in _GUARDED_TYPE_KEYS:
            for rule in rules_map.get(key, ()):
                if id(rule) in counted:
                    continue  # a rule can index under several names
                if any(
                    len(sset.selectors) > 1 for sset in rule.selector_set
                ):
                    counted.add(id(rule))
                    per_key[key] += 1
                    if len(offenders) < 400:
                        offenders.append(f"[{key}] {rule.selectors}")

        total = sum(per_key.values())
        breakdown = ", ".join(
            f"{key}={count}" for key, count in per_key.most_common()
        )

        print(f"\n[census] total={total} ({breakdown})")
        assert total >= MIN_ANCESTOR_SCOPED_BARE_TYPE_RULES, (
            f"census found only {total} ancestor-scoped bare-type rules "
            f"({breakdown}) -- below the anti-vacuity floor "
            f"({MIN_ANCESTOR_SCOPED_BARE_TYPE_RULES}). The census is walking "
            "a hollow stylesheet, not a real boot; a passing ratchet here "
            "would prove nothing."
        )
        assert total <= MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES, (
            f"{total} ancestor-scoped bare-type-subject rules "
            f"(ratchet limit {MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES}; "
            f"{breakdown}).\n"
            "Each of these is a style-apply candidate for EVERY widget of "
            "its type in the app, because Textual indexes rules by their "
            "rightmost selector only. Do not raise the constant (see the "
            "constant's comment and ADR-097's ratchet discipline): re-key "
            "the new rule instead -- give its subject a class carried only "
            "by the intended widgets, e.g. `#panel Button` -> "
            "`Button.panel-action` plus that class in compose().\n"
            "Newest offenders are easiest to find with: git diff on "
            "tldw_chatbook/css/ for selectors ending in a bare "
            f"{'/'.join(_GUARDED_TYPE_KEYS[:3])}... type.\n"
            "Sample of current offenders:\n  "
            + "\n  ".join(offenders[:15])
        )
