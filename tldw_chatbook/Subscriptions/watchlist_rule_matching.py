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

#: The segment of a site change's text that is newly present, for a rule
#: scoped to "appeared" (TASK-1363).
#:
#: Set by ``URLMonitor.check_url`` (``added_and_removed_text``), from the same
#: diff the reader's change pane already shows. Matching-only, like
#: :data:`RULE_MATCH_TEXT_KEY`: not a persisted column (``persist_subscription_item``
#: reads a fixed key set and ignores this one), and not set at all by the feed
#: or API producers, whose items have no "previous version" to diff against.
RULE_MATCH_ADDED_TEXT_KEY = "rule_match_added_text"

#: The segment of a site change's text that is no longer present, for a rule
#: scoped to "disappeared" (TASK-1363). Same provenance and non-persistence as
#: :data:`RULE_MATCH_ADDED_TEXT_KEY`, above.
RULE_MATCH_REMOVED_TEXT_KEY = "rule_match_removed_text"


def _page_wide_haystack(item: Mapping[str, Any]) -> str:
    """The original, page-wide haystack: title/summary/body/author, lowercased.

    This is the whole of ``scope="anywhere"`` and also the fallback a wholly
    new item (no diff to narrow against) uses under ``scope="appeared"`` --
    kept as one function so the two paths cannot drift apart, and so a rule
    with no ``scope`` at all produces byte-identical output to before this
    parameter existed.
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


def build_rule_haystack(item: Mapping[str, Any], scope: str = "anywhere") -> str:
    """Build the lowercased text a filter or alert rule is matched against.

    Three scopes (TASK-1363), chosen per rule and passed in by the caller --
    ``build_rule_haystack`` itself has no notion of "the rule", only of the
    item and which slice of it to search:

    * ``"anywhere"`` (the default, and what any unrecognized value falls back
      to): the whole page -- ``title``/``summary``/body/``author``, where the
      body is :data:`RULE_MATCH_TEXT_KEY` when the producer supplied it,
      otherwise ``content``. Unchanged from before this parameter existed, so
      an existing rule with no ``scope`` key, and :class:`WatchlistFilterService`
      (which never passes ``scope`` at all, because a narrowed *exclude*
      filter could admit an item the user told the app to drop), keep matching
      the whole page exactly as before.
    * ``"appeared"``: only the text a site change newly introduced --
      :data:`RULE_MATCH_ADDED_TEXT_KEY` in place of the body, alongside
      ``title``/``summary``/``author`` (a rule can still match a phrase in the
      title of the item that reported the change). When that key is absent --
      a feed or API item, which has no "previous version" to diff against --
      the *entire* new item just appeared, so this falls back to the page-wide
      haystack rather than matching nothing.
    * ``"disappeared"``: only :data:`RULE_MATCH_REMOVED_TEXT_KEY`, and nothing
      else -- not ``title``/``summary``/``author``, because those describe the
      item as it now stands, not text that disappeared. When the key is
      absent, nothing is known to have disappeared, so the haystack is empty
      and the rule can never match (rather than silently falling back to the
      whole page, which would defeat the point of scoping to what left).

    Args:
        item: A raw fetched item, before persistence.
        scope: One of ``"anywhere"``, ``"appeared"``, ``"disappeared"``. Any
            other value (including an absent/``None`` scope) is treated as
            ``"anywhere"``.

    Returns:
        The scoped haystack, lowercased.
    """
    normalized_scope = str(scope or "anywhere").lower()

    if normalized_scope == "disappeared":
        if RULE_MATCH_REMOVED_TEXT_KEY not in item:
            return ""
        return str(item.get(RULE_MATCH_REMOVED_TEXT_KEY) or "").lower()

    if normalized_scope == "appeared":
        if RULE_MATCH_ADDED_TEXT_KEY not in item:
            return _page_wide_haystack(item)
        parts = [
            str(item.get("title") or ""),
            str(item.get("summary") or ""),
            str(item.get(RULE_MATCH_ADDED_TEXT_KEY) or ""),
            str(item.get("author") or ""),
        ]
        return " ".join(parts).lower()

    # "anywhere" and any unrecognized scope value (safe default).
    return _page_wide_haystack(item)
