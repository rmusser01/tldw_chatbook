"""The default noise selectors and the extraction fingerprint (TASK-1362).

Selector semantics (verified, documented, NOT changed): newlines separate
independent rules; a comma within a line is a CSS selector group and matches
every branch. Splitting on commas would break `:is(.a, .b)` and
`[data-x="a,b"]`.

Two obvious-looking lines are deliberately absent from the default set.
CSRF/session-token inputs: `<input value=...>` contributes nothing to
`get_text()`, so a token selector strips nothing and only teaches users that
dead lines are normal. The broad `[class*="cookie"]`: it matches
`class="cookie-recipe-card"` and strips a cookie RECIPE -- substring
selectors are narrowed to consent-banner forms for exactly this reason.
Likewise `time[datetime]` is excluded: a release date lives in exactly that
element, and dates are often the payload being watched.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache

DEFAULT_IGNORE_SELECTORS: tuple[str, ...] = (
    '[class*="cookie-consent"], [class*="cookie-banner"], '
    '[id*="cookie-consent"], .cc-banner',
    '[class*="consent-manager"]',
    ".ad, .ads, .advertisement",
    '.sponsored, [class*="sponsored-"]',
    '.view-count, .views, [class*="viewcount"]',
    ".timestamp",
)


def default_ignore_selectors_text() -> str:
    """The default set as the newline-joined text a form field holds.

    Returns:
        One rule per line, in the documented order.
    """
    return "\n".join(DEFAULT_IGNORE_SELECTORS)


def extraction_fingerprint(
    ignore_selectors: str | None, extraction_method: str | None
) -> str:
    """A stable hash of the settings that shape extracted text.

    Snapshots store text extracted under the settings in force at capture
    time, so comparing a snapshot across a settings change is meaningless --
    the check must re-baseline instead (spec §3). Normalization: lines are
    stripped, empties dropped, duplicates removed and the result SORTED, so
    cosmetic reordering does not re-baseline.

    Args:
        ignore_selectors: The raw newline-separated selector text, or None.
        extraction_method: The subscription's extraction method. It is
            canonicalized to the EFFECTIVE mode the fetch actually runs --
            ``"html"`` for ``"full"``/``"auto"``, ``"raw"`` for anything else
            including ``None`` -- before hashing. See below.

    Returns:
        A hex digest; equal iff extraction behaviour is equal.
    """
    lines = sorted(
        {s.strip() for s in str(ignore_selectors or "").splitlines() if s.strip()}
    )
    # Hash the EFFECTIVE extraction mode, not the literal method string
    # (whole-branch fix F2, one step further than Minor 7). The one branch this
    # hash exists to model is in `URLMonitor._fetch_url_content`:
    #
    #     if extraction_method == "full" or extraction_method == "auto":
    #         text = ContentExtractor.extract_text_from_html(...)   # -> "html"
    #     else:
    #         text = response.text                                 # -> "raw"
    #
    # So `"full"` and `"auto"` are the SAME extraction -- HTML text with
    # `ignore_selectors` applied -- and hashing the literal strings split them.
    # Switching a source between the two therefore invalidated its snapshot and
    # burned a whole diff window (the next check compares against nothing and
    # reports nothing) for a change that alters no extracted byte.
    #
    # Everything else, `None` very much included, falls through to the raw
    # response body where `ignore_selectors` are never applied at all. That is
    # the Minor 7 fix and it is preserved: an explicit `None` (what a DB row
    # with a NULL `extraction_method` hands us) must NOT collide with an
    # HTML-extraction source carrying the same selectors.
    #
    # The engine's own `.get("extraction_method", "auto")` default covers the
    # other direction -- an ABSENT key really does mean "auto" -- and
    # `check_url` passes that same default in, so absent and NULL stay
    # distinguishable here.
    effective_mode = "html" if extraction_method in ("full", "auto") else "raw"
    payload = {"selectors": lines, "method": effective_mode}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@lru_cache(maxsize=1)
def selector_parse_errors() -> tuple[type[BaseException], ...]:
    """The exception family a malformed CSS selector raises, for `except`.

    One source of truth so the extraction guard and the two save-path
    validators can never drift apart. Empirically probed against the pinned
    soupsieve (2.9.1) with `BeautifulSoup.select`:

    * ``soupsieve.util.SelectorSyntaxError`` -- almost everything a user can
      mistype: ``div[``, ``!!!``, ``div >``, ``]]``, ``@media print``, an
      unknown pseudo-class, and (relevant here) a blank selector.
    * ``NotImplementedError`` -- pseudo-ELEMENTS (``::before``) and at-rules.
      A perfectly ordinary thing to paste out of a stylesheet, and not a
      syntax error at all, so it needs naming separately.
    * ``ValueError`` -- soupsieve's pseudo-class nesting limit (8192). Not
      reachable through the 4000-character form fields, included because the
      cost of an unguarded selector error is the entire check.

    Deliberately NOT included: ``AttributeError``/``KeyError``/``TypeError``.
    `select()` raises ``TypeError`` when handed a non-`Tag`, which is a bug in
    our code, not in the user's selector, and must keep propagating.

    Returns:
        A tuple usable directly in an `except` clause. Degrades to the two
        builtins if soupsieve is absent (no bs4 => no selector work at all).
    """
    try:
        from soupsieve import SelectorSyntaxError
    except ImportError:  # pragma: no cover - bs4 is the [subscriptions] extra
        return (NotImplementedError, ValueError)
    return (SelectorSyntaxError, NotImplementedError, ValueError)


def first_invalid_selector(ignore_selectors: str | None) -> str | None:
    """The first line of `ignore_selectors` that CSS cannot parse, if any.

    Validation for the save paths, so a typo is refused at the keyboard while
    the user still has the text in front of them, instead of silently costing
    them every subsequent check of that source. Lines are taken exactly as the
    field holds them: split on NEWLINES only, never on commas (a comma inside
    a line is a CSS selector group -- `:is(.a, .b)` is one valid rule).
    Blank lines are skipped, and empty text is valid: clearing the field is a
    deliberate "watch everything on this page".

    Args:
        ignore_selectors: The raw newline-separated selector text, or None.

    Returns:
        The offending line, stripped, or None if every non-empty line parses.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:  # pragma: no cover - bs4 is the [subscriptions] extra
        return None

    # A minimal document: `select()` has to parse the selector to run at all,
    # and matching nothing is the point -- this asks "is it valid CSS?", not
    # "does it hit anything?" (a rule that matches nothing today is legitimate;
    # the page may grow the element tomorrow).
    probe = BeautifulSoup("", "html.parser")
    for line in str(ignore_selectors or "").splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            probe.select(candidate)
        except selector_parse_errors():
            return candidate
    return None


def invalid_selector_message(selector: str) -> str:
    """The toast copy for a rejected selector line.

    Shared by both save paths (`SourcesPane`'s create form and the Inspector's
    editor) so the two cannot drift into saying different things about the same
    refusal, and so a test can pin one string.

    Args:
        selector: The offending line, as the user typed it.

    Returns:
        A message naming the line and what to do about it.
    """
    return (
        f"Ignore rule is not valid CSS and was not saved: {selector} — "
        "fix or remove that line."
    )
