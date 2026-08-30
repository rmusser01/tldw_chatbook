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
* **7,335,029** ``RuleSet.__hash__`` calls in a single Console screen switch,
  which is exactly 1,667 applies x 4,324 rules -- the scan

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
"""

from __future__ import annotations

from typing import Any

from textual.css.stylesheet import Stylesheet

__all__ = ["install_stylesheet_fastpath", "is_installed"]

#: Marks the patched callable so installation is idempotent and detectable.
_MARKER = "_tldw_ordered_candidate_fastpath"

#: Per-stylesheet cache: ``(rules_map, {id(rule): position})``. Keyed on the
#: ``rules_map`` *object*, of which we hold a strong reference: Textual sets
#: ``_rules_map = None`` on every path that changes ``_rules`` (``parse``,
#: ``reparse``, ``add_source``), so a surviving identity match proves the rule
#: list is unchanged. Holding the reference also stops ``id()`` reuse from
#: aliasing a freed map onto a new one.
_CACHE_ATTR = "_tldw_rule_position_index"


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
    else:
        index = {id(rule): position for position, rule in enumerate(rules)}
        setattr(stylesheet, _CACHE_ATTR, (rules_map, index))

    candidates = {
        rule for name in rules_map.keys() & selector_names for rule in rules_map[name]
    }
    if not candidates:
        return []
    # Ascending source order. Upstream reverses it again inside ``apply``,
    # which is what preserves "later rule wins" tie-breaking.
    return sorted(candidates, key=lambda rule: index[id(rule)])


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
