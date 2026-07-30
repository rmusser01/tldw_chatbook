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
    """Reordering, blank lines and duplicates must not re-baseline."""
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    a = extraction_fingerprint(".ad\n.timestamp\n\n.ad", "auto")
    b = extraction_fingerprint(".timestamp\n.ad", "auto")
    assert a == b


def test_fingerprint_changes_when_extraction_actually_changes():
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    base = extraction_fingerprint(".ad", "auto")
    assert extraction_fingerprint(".ad\n.sponsored", "auto") != base
    assert extraction_fingerprint(".ad", "raw") != base
    assert extraction_fingerprint(None, "auto") != base
