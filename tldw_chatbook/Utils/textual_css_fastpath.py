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

The original filter used only the leading ancestor ID/class. TASK-35611 extends
that necessary-condition check to all static ancestor and subject names, including
Textual's CSS inheritance types. Every comma alternative must be impossible before
a rule is rejected; universal and unknown selectors impose no cheap requirement.
Upstream still checks compound co-location, ancestry order and all dynamic state.

Candidate reuse (TASK-35610): a 1024-entry LRU per parsed stylesheet reuses
ordered candidates for identical live subject/ancestor names. It retains no
nodes or computed styles. Class/id/ancestry changes alter the key; stylesheet
reparsing replaces the cache. Upstream still matches pseudo-classes and order,
resolves values, and processes recursive component styles on every call.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from textual.css.stylesheet import Stylesheet

__all__ = ["install_stylesheet_fastpath", "is_installed"]

#: Marks the patched callable so installation is idempotent and detectable.
_MARKER = "_tldw_ordered_candidate_fastpath"

#: Per-stylesheet cache: ``(rules_map, {id(rule): position}, {id(rule):
#: requirements}, bounded candidate LRU)``. Keyed on the ``rules_map`` *object*,
#: of which we hold a
#: strong reference: Textual sets ``_rules_map = None`` on every path that
#: changes ``_rules`` (``parse``, ``reparse``, ``add_source``), so a surviving
#: identity match proves the rule list is unchanged. Holding the reference
#: also stops ``id()`` reuse from aliasing a freed map onto a new one -- which
#: matters doubly now that the cache is keyed by ``id(rule)`` for two things.
_CACHE_ATTR = "_tldw_rule_position_index"
# The measured Console has 523 subject/ancestor shapes, plus virtual components;
# 256 churns during a single screen update. Keep room for one full live screen.
_MAX_CANDIDATE_KEYS = 1024


def _static_requirements(rule: Any) -> tuple:
    """Return necessary ancestor/subject names for each selector alternative.

    Names before the final child/descendant hop belong to ancestors; the final
    compound belongs to the subject. Presence is only necessary, not sufficient:
    upstream still checks which ancestor, ordering, pseudo-state and specificity.
    Universal and unknown selectors impose no cheap requirement.
    """
    from textual.css.model import CombinatorType, SelectorType

    prefixes = {SelectorType.ID: "#", SelectorType.CLASS: ".", SelectorType.TYPE: ""}
    requirements = []
    for selector_set in rule.selector_set:
        selectors = selector_set.selectors
        subject_start = 0
        for index, selector in enumerate(selectors[1:], 1):
            if selector.combinator in (CombinatorType.DESCENDENT, CombinatorType.CHILD):
                subject_start = index
        ancestors, subject = set(), set()
        for index, selector in enumerate(selectors):
            prefix = prefixes.get(selector.type)
            if prefix is not None:
                target = ancestors if index < subject_start else subject
                target.add(f"{prefix}{selector.name}")
        requirements.append((frozenset(ancestors), frozenset(subject)))
    return tuple(requirements)


def _ancestor_names(node: Any) -> set:
    """Every type, ``#id`` and ``.class`` carried by this node's ancestors.

    Recomputed per apply rather than cached: an ancestor's classes change at
    runtime (``-active``, ``hidden``, ...), and a stale set here would reject
    rules that have just become applicable.
    """
    names = set()
    for ancestor in node.ancestors:
        # Use the matcher's CSS inheritance names, not Python's full MRO.
        names.update(getattr(ancestor, "_css_type_names", ()))
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
        candidate_cache = cached[3]
    else:
        index = {id(rule): position for position, rule in enumerate(rules)}
        requirements = {}
        candidate_cache = OrderedDict()
        setattr(
            stylesheet, _CACHE_ATTR, (rules_map, index, requirements, candidate_cache)
        )

    # Cache only candidate selection, never resolved styles. Live selector and
    # ancestor names witness class/id/parent changes; upstream still evaluates
    # pseudo-classes, order, specificity and theme values on every application.
    # Keys contain strings, not DOM nodes, and the LRU bounds dynamic IDs.
    ancestor_names = _ancestor_names(node)
    key = (frozenset(selector_names), frozenset(ancestor_names))
    if key in candidate_cache:
        candidate_cache.move_to_end(key)
        return candidate_cache[key]

    candidates = {
        rule for name in rules_map.keys() & selector_names for rule in rules_map[name]
    }

    # Each comma alternative must be statically impossible before dropping the
    # rule. An unrelated bare subject in one alternative must not keep a scoped
    # rule alive for every node of another alternative's type.
    surviving = []
    for rule in candidates:
        rule_id = id(rule)
        rule_requirements = requirements.get(rule_id)
        if rule_requirements is None:
            rule_requirements = _static_requirements(rule)
            requirements[rule_id] = rule_requirements
        if not rule_requirements:
            # No selector sets at all: nothing is known, so keep it. Without
            # this the loop below never appends and the rule is silently
            # DROPPED -- rejection must always be something we decided, never
            # something that fell through.
            surviving.append(rule)
            continue
        for ancestors, subject in rule_requirements:
            if ancestors <= ancestor_names and subject <= selector_names:
                surviving.append(rule)
                break
    # Ascending source order. Upstream reverses it again inside ``apply``,
    # which is what preserves "later rule wins" tie-breaking.
    surviving.sort(key=lambda rule: index[id(rule)])
    candidate_cache[key] = surviving
    if len(candidate_cache) > _MAX_CANDIDATE_KEYS:
        candidate_cache.popitem(last=False)
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
