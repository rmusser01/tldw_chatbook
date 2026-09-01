"""Ordered-candidate fast path for Textual's ``Stylesheet.apply``.

Upstream ``Stylesheet.apply`` narrows the candidate rules through ``rules_map``
-- cheap, keyed on the node's own selector names -- and then recovers source
order by throwing that narrowing away::

    rules = list(filter(limit_rules.__contains__, reversed(self.rules)))

That final step walks the *entire* rule list on every call, so styling one node
is ``O(total rules in the app)`` rather than ``O(rules that could match this
node)``.

Measured on dev ``bc1e26ce60`` (4,324 global rules), 2026-08-29 holistic perf
review:

* 0.52 ms per single-node ``apply``
* 240 ms for one full-screen ``update_styles`` over the Console's 500 widgets
* **7,335,029** ``RuleSet.__hash__`` calls in a single Console screen switch:
  ~4,400 per apply over 1,667 applies, of which 4,324 are the full-list scan
  and ~76 are the candidate-set construction (each candidate is hashed on
  insertion). The scan is the dominant term, not the only one.

Stack sampling during observed event-loop stalls (worst 399 ms, on screen
switching) ranked ``textual/css/model.py:__hash__`` the #1 frame.

This module deliberately keeps upstream's ``apply`` as the single source of
truth for what styling *means*. It replaces only how the ordered candidate list
is built: candidates are sorted by a cached position index instead of recovered
by a full scan. Interleaved A/B on the Console, two rounds:

===========================  =========  ==========  ======
Metric                       upstream   fast path   delta
===========================  =========  ==========  ======
``apply()`` per node         0.52 ms    0.38 ms     -27%
``update_styles(screen)``    241 ms     156 ms      -35%
Console screen switch        1.93 s     1.67 s      -260 ms
===========================  =========  ==========  ======

``Tests/Performance/test_textual_css_fastpath.py`` pins the upstream
implementation this delegation assumes, so a Textual upgrade fails loudly here
rather than silently changing styling behaviour.

Second optimisation -- ancestor rejection (2026-08-30 holistic perf review)
--------------------------------------------------------------------------

``rules_map`` keys a rule under its **rightmost** selector only
(``RuleSet._post_parse``: ``selector_set.selectors[-1]``). So
``#prompt-variables-actions Button:disabled`` is filed under ``Button`` and
becomes a candidate for *every* ``Button`` in the app, each of which runs full
selector matching before rejecting it. Measured on a 502-node Console:

* ``Button`` carries 188 rules against 110 live buttons = 20,680 candidate
  considerations, **71% of the whole screen's candidate work**
* across the common type keys, **93%** of candidate work comes from
  ancestor-scoped rules (the attribution ceiling -- for their TARGET widgets
  those rules do match; the measured cannot-match share is the 47% below)
* **47%** of all candidates are rejectable by checking one cheap thing: does
  this node have an ancestor carrying the rule's leading ``#id``/``.class``?

Interleaved A/B (four pairs, filter toggled in place, median of five
full-screen ``stylesheet.update`` calls per arm):

===========================  =========  ==========  ======
Metric                       no filter  filter      delta
===========================  =========  ==========  ======
``update(screen)``, 502 nodes  105.0 ms   66.2 ms    -37%
===========================  =========  ==========  ======

Ranges did not overlap (103.5-108.3 vs 64.8-66.8).

The filter is conservative by construction: a rule survives unless **every**
one of its selector sets states a requirement that is unmet, and any shape
that cannot be decided cheaply reports "no requirement" (a leading ``TYPE``
selector needs MRO matching, which is the cost this avoids). Both the
per-node fidelity tour and a unit test over the parser shapes pin it, and
both were mutation-tested: deleting the one-compound guard fails them.
"""

from __future__ import annotations

from typing import Any

from textual.css.stylesheet import Stylesheet

__all__ = ["install_stylesheet_fastpath", "is_installed"]

#: Marks the patched callable so installation is idempotent and detectable.
_MARKER = "_tldw_ordered_candidate_fastpath"

#: Per-stylesheet cache: ``(rules_map, {id(rule): position}, {id(rule):
#: requirements})``. Keyed on the ``rules_map`` *object*, of which we hold a
#: strong reference: Textual sets ``_rules_map = None`` on every path that
#: changes ``_rules`` (``parse``, ``reparse``, ``add_source``), so a surviving
#: identity match proves the rule list is unchanged. Holding the reference
#: also stops ``id()`` reuse from aliasing a freed map onto a new one -- which
#: matters doubly now that the cache is keyed by ``id(rule)`` for two things.
_CACHE_ATTR = "_tldw_rule_position_index"


def _ancestor_requirements(rule: Any) -> tuple:
    """Names an ANCESTOR must carry for each of a rule's selector sets.

    Textual indexes a rule under its rightmost selector only, so
    ``#prompt-variables-actions Button`` is a candidate for every ``Button``
    in the app even though it can only match one panel's buttons. This
    recovers the leading requirement so those can be rejected before upstream
    runs full selector matching on them.

    Returns:
        One entry per selector set: ``"#id"`` / ``".class"`` when that set
        requires an ancestor carrying it, or ``None`` when no requirement can
        be established cheaply. A rule matches if ANY of its sets match, so it
        is only rejectable when EVERY entry is a requirement that is unmet.

        ``None`` is returned for a leading TYPE selector on purpose: matching
        a type against an ancestor means walking its MRO, which is what this
        filter exists to avoid, and a wrong answer here is a styling bug.
    """
    from textual.css.model import CombinatorType, SelectorType

    requirements = []
    for selector_set in rule.selector_set:
        selectors = selector_set.selectors
        if len(selectors) < 2:
            # Subject only -- says nothing about ancestors.
            requirements.append(None)
            continue
        if not any(
            selector.combinator in (CombinatorType.DESCENDENT, CombinatorType.CHILD)
            for selector in selectors[1:]
        ):
            # One compound (e.g. `Button.foo#bar`): the id/class is on the
            # SUBJECT, not an ancestor. Requiring an ancestor would wrongly
            # reject the node that actually matches.
            requirements.append(None)
            continue
        first = selectors[0]
        if first.type == SelectorType.ID:
            requirements.append(f"#{first.name}")
        elif first.type == SelectorType.CLASS:
            requirements.append(f".{first.name}")
        else:
            requirements.append(None)
    return tuple(requirements)


def _ancestor_names(node: Any) -> set:
    """Every ``#id`` and ``.class`` carried by this node's ancestors.

    Recomputed per apply rather than cached: an ancestor's classes change at
    runtime (``-active``, ``hidden``, ...), and a stale set here would reject
    rules that have just become applicable.
    """
    names = set()
    for ancestor in node.ancestors:
        ancestor_id = getattr(ancestor, "id", None)
        if ancestor_id:
            names.add(f"#{ancestor_id}")
        classes = getattr(ancestor, "classes", None)
        if classes:
            for class_name in classes:
                names.add(f".{class_name}")
    return names


def _ordered_candidates(stylesheet: Stylesheet, node: Any) -> list | None:
    """Return this node's candidate rules in source order, or ``None``.

    ``None`` means "no fast path available for this node" and the caller must
    fall back to upstream's own scan.
    """
    try:
        selector_names = node._selector_names
    except AttributeError:
        return None

    # Touch ``.rules`` first: the property parses on demand, and parsing
    # replaces ``_rules`` wholesale. Doing it before the swap below keeps the
    # swap from being undone underneath us.
    rules = stylesheet.rules
    rules_map = stylesheet.rules_map

    cached = getattr(stylesheet, _CACHE_ATTR, None)
    if cached is not None and cached[0] is rules_map:
        index = cached[1]
        requirements = cached[2]
    else:
        index = {id(rule): position for position, rule in enumerate(rules)}
        requirements = {}
        setattr(stylesheet, _CACHE_ATTR, (rules_map, index, requirements))

    candidates = {
        rule for name in rules_map.keys() & selector_names for rule in rules_map[name]
    }
    if not candidates:
        return []

    # Reject candidates whose leading compound names an ancestor this node
    # does not have. `rules_map` keys on the RIGHTMOST selector only, so a
    # rule scoped to one panel is a candidate for every widget of that type
    # in the app -- measured at 47% of all candidates on a 502-node Console,
    # and 50% of a Button's. Rejecting here skips upstream's full selector
    # evaluation for them.
    #
    # Conservative by construction: a rule survives unless EVERY one of its
    # selector sets states a requirement that is unmet, and any set we cannot
    # decide cheaply reports `None` (= keep). Correctness is pinned by
    # `test_fastpath_computes_identical_styles_for_every_node`.
    ancestor_names = _ancestor_names(node)
    surviving = []
    for rule in candidates:
        rule_id = id(rule)
        rule_requirements = requirements.get(rule_id)
        if rule_requirements is None:
            rule_requirements = _ancestor_requirements(rule)
            requirements[rule_id] = rule_requirements
        if not rule_requirements:
            # No selector sets at all: nothing is known, so keep it. Without
            # this the loop below never appends and the rule is silently
            # DROPPED -- rejection must always be something we decided, never
            # something that fell through.
            surviving.append(rule)
            continue
        for requirement in rule_requirements:
            if requirement is None or requirement in ancestor_names:
                surviving.append(rule)
                break
    if not surviving:
        return []
    # Ascending source order. Upstream reverses it again inside ``apply``,
    # which is what preserves "later rule wins" tie-breaking.
    surviving.sort(key=lambda rule: index[id(rule)])
    return surviving


def install_stylesheet_fastpath() -> bool:
    """Install the fast path on ``Stylesheet.apply``. Idempotent.

    Returns:
        ``True`` if this call installed the patch, ``False`` if it was already
        installed by an earlier call.
    """
    upstream = Stylesheet.apply
    if getattr(upstream, _MARKER, False):
        return False

    def apply(
        self: Stylesheet,
        node: Any,
        *,
        animate: bool = False,
        cache: dict | None = None,
    ) -> None:
        ordered = _ordered_candidates(self, node)
        if ordered is None:
            return upstream(self, node, animate=animate, cache=cache)

        # Hand upstream a rule list that is already narrowed and ordered. Its
        # own ``filter(limit_rules.__contains__, reversed(self.rules))`` then
        # reduces to "reverse this list", because every element is a
        # candidate -- same result, without the O(all rules) walk.
        saved_rules = self._rules
        try:
            self._rules = ordered
            return upstream(self, node, animate=animate, cache=cache)
        finally:
            self._rules = saved_rules

    setattr(apply, _MARKER, True)
    setattr(apply, "__wrapped__", upstream)
    Stylesheet.apply = apply  # type: ignore[method-assign]
    return True


def is_installed() -> bool:
    """Whether the fast path is currently installed on ``Stylesheet.apply``."""
    return bool(getattr(Stylesheet.apply, _MARKER, False))
