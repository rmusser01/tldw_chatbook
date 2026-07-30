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
        extraction_method: The subscription's extraction method, or None
            (normalized to "auto", the code's effective default).

    Returns:
        A hex digest; equal iff extraction behaviour is equal.
    """
    lines = sorted(
        {s.strip() for s in str(ignore_selectors or "").splitlines() if s.strip()}
    )
    payload = {"selectors": lines, "method": (extraction_method or "auto")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
