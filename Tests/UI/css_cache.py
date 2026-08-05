# css_cache.py
# Description: Session-global Textual CSS parse cache for tests (task-1459).
#
# Every TldwCli mount constructs fresh Stylesheet instances, and Textual's
# parse cache is per-instance (LRUCache(64) on self._parse_cache) — so each of
# the ~1,300 app mounts in a full run re-parses the same ~14 CSS blobs,
# including the 14,600-line app bundle. Measured on this harness: 0.12-0.15s
# of pure parsing per mount, 22-37% of total mount cost.
#
# This wrapper adds a process-global cache in FRONT of the per-instance one.
# Textual's own cache key omits the stylesheet's variables (safe per-instance
# because set_variables() clears the cache); a global cache MUST include them,
# so the key adds a fingerprint of self._variables. The cached RuleSet list is
# copied at both boundaries: Textual may mutate that container while building
# one Stylesheet, so sharing the same list with the next app is unsafe.
#
# Escape hatch: TLDW_TEST_CSS_CACHE=0 disables installation entirely.

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from textual.css.parse import RuleSet

_GLOBAL_PARSE_CACHE: dict[tuple[Any, ...], "list[RuleSet]"] = {}
_installed = False


def install() -> bool:
    """Install the global parse cache onto ``Stylesheet._parse_rules``.

    Idempotent; a no-op when ``TLDW_TEST_CSS_CACHE=0``. Called lazily from the
    root conftest once ``textual.css.stylesheet`` is already imported, so
    sessions that never touch Textual pay nothing.

    Returns:
        True if the cache is installed after this call.
    """
    global _installed
    if _installed:
        return True
    if os.environ.get("TLDW_TEST_CSS_CACHE", "1") == "0":
        return False

    from textual.css.stylesheet import Stylesheet

    original_parse_rules = Stylesheet._parse_rules

    def cached_parse_rules(
        self,
        css: str,
        read_from,
        is_default_rules: bool = False,
        tie_breaker: int = 0,
        scope: str = "",
    ):
        # Variables change what tokens like $accent resolve to, so they are
        # part of parse identity even though Textual's per-instance key omits
        # them (per-instance safety relies on set_variables() clearing the
        # cache — a guarantee that does not span instances).
        variables_fingerprint = tuple(sorted(self._variables.items()))
        key = (
            css,
            read_from,
            is_default_rules,
            tie_breaker,
            scope,
            variables_fingerprint,
        )
        cached = _GLOBAL_PARSE_CACHE.get(key)
        if cached is not None:
            return list(cached)
        rules = original_parse_rules(
            self, css, read_from, is_default_rules, tie_breaker, scope
        )
        _GLOBAL_PARSE_CACHE[key] = list(rules)
        return rules

    Stylesheet._parse_rules = cached_parse_rules
    _installed = True
    return True


def cache_info() -> tuple[bool, int]:
    """Return (installed, cached-entry count) for diagnostics."""
    return _installed, len(_GLOBAL_PARSE_CACHE)
