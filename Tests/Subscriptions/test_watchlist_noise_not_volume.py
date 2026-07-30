"""TASK-1362: suppress noise, not changes.

Spec: Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_selectors_strip_noise_but_not_cookie_recipes():
    """Every default line must do something; none may eat the payload.

    Proven during spec review: `[class*="cookie"]` matches
    `class="cookie-recipe-card"` and strips a cookie RECIPE, and
    `<input value=...>` never reaches `get_text()` at all. The default set
    was narrowed accordingly; this pins both properties.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor
    from tldw_chatbook.Subscriptions.noise_defaults import DEFAULT_IGNORE_SELECTORS

    html = (
        '<div class="cookie-consent-banner">We use cookies</div>'
        '<div class="ad">BUY NOW</div>'
        '<span class="view-count">123 views</span>'
        '<span class="timestamp">12:01</span>'
        '<div class="cookie-recipe-card">Best cookie recipe</div>'
        '<time datetime="2026-07-29">Release date 2026-07-29</time>'
        "<p>real content</p>"
    )
    out = ContentExtractor.extract_text_from_html(
        html, list(DEFAULT_IGNORE_SELECTORS)
    )
    for noise in ("We use cookies", "BUY NOW", "123 views", "12:01"):
        assert noise not in out
    assert "Best cookie recipe" in out
    assert "Release date 2026-07-29" in out
    assert "real content" in out


def test_fingerprint_ignores_cosmetic_selector_edits():
    """Reordering, blank lines and duplicates must not re-baseline.

    A too-small fixture makes a missing ``sorted()`` invisible: CPython set
    iteration order is fully determined by content (independent of
    insertion order) whenever no two items land in the same hash-table
    slot, and for a handful of items that "no collision at all" case is
    common (empirically ~50-60% of hash seeds for 2-6 items, still ~10-25%
    for a dozen or two, since the table itself grows with item count and
    keeps the collision odds roughly constant across small sizes). When
    that happens, *every* reordering of those items lands back in the same
    slots, so comparing against one shuffle -- or even several -- still
    passes by luck when ``sorted()`` is gone; more shuffles do not help
    because it is one binary per-seed property of the whole item set, not
    independent bad luck per comparison. What actually drives the
    probability down is enough items that at least one pairwise collision
    is close to certain: ~50 items pushes it below 1e-4. Verified directly:
    dropping ``sorted()`` was RED under every one of ``PYTHONHASHSEED``
    0-19 with this fixture, vs. failing to catch it under roughly half of
    those seeds with 2 items and a third of them with 12-20 items.
    """
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    selectors = tuple(f".noise-selector-{i}" for i in range(50))
    forward = "\n".join(selectors) + "\n\n" + selectors[0]  # trailing blank + dup
    reordered = "\n".join(reversed(selectors))

    assert extraction_fingerprint(forward, "auto") == extraction_fingerprint(
        reordered, "auto"
    )


def test_fingerprint_changes_when_extraction_actually_changes():
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    base = extraction_fingerprint(".ad", "auto")
    assert extraction_fingerprint(".ad\n.sponsored", "auto") != base
    assert extraction_fingerprint(".ad", "raw") != base
    assert extraction_fingerprint(None, "auto") != base
    # None and "" must normalize identically (str(x or "")): a form
    # round-trip that turns a NULL into an empty string must not silently
    # re-baseline every source's fingerprint.
    assert extraction_fingerprint(None, "auto") == extraction_fingerprint("", "auto")
