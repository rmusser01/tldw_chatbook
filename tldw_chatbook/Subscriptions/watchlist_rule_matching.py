"""The one haystack watchlist filters and content-alert rules match against.

Fix round 1, Important #3. ``WatchlistFilterService`` and
``WatchlistContentAlertService`` each built their own copy of the same
``" ".join(...)`` over ``("title", "summary", "content", "author")``, and
``_apply_filters_and_alerts`` runs *before* persistence -- so when TASK-1343
made a site change's ``content`` a diff rather than the whole page, both
services silently narrowed from "matches anywhere on the page" to "matches a
changed segment plus one line of context".

That is a behaviour change a user would experience as an alert that had been
firing for months quietly stopping, with nothing in the UI to explain it. So
the producer keeps the full page text on the item under
:data:`RULE_MATCH_TEXT_KEY` -- a display-and-storage concern (the diff) and a
matching concern (the page) are simply not the same text -- and the haystack
lives here once, so the two services cannot drift apart again.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Full captured text for rule matching, when ``content`` is not it.
#:
#: Set by ``URLMonitor.check_url``, whose ``content`` is a diff. Deliberately
#: NOT a persisted column: ``persist_subscription_item`` reads a fixed key set
#: and ignores this one, and the same text is already durable in
#: ``url_snapshots``. Feed and API items do not set it -- their ``content``
#: *is* the captured body -- so the fallback below is the common path.
RULE_MATCH_TEXT_KEY = "rule_match_text"


def build_rule_haystack(item: Mapping[str, Any]) -> str:
    """Build the lowercased text a filter or alert rule is matched against.

    Args:
        item: A raw fetched item, before persistence.

    Returns:
        ``title``, ``summary``, the item's body and ``author`` joined by single
        spaces and lowercased. The body is :data:`RULE_MATCH_TEXT_KEY` when the
        producer supplied it, otherwise ``content``; it *replaces* ``content``
        rather than adding to it, so that a phrase which was only ever in the
        text the previous check removed does not start matching.
    """
    body = item.get(RULE_MATCH_TEXT_KEY)
    if body is None:
        body = item.get("content")
    parts = [
        str(item.get("title") or ""),
        str(item.get("summary") or ""),
        str(body or ""),
        str(item.get("author") or ""),
    ]
    return " ".join(parts).lower()
