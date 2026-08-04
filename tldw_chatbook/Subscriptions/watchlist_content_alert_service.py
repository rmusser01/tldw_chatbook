from __future__ import annotations

import re
from typing import Any, Mapping

from .watchlist_rule_matching import build_rule_haystack


class WatchlistContentAlertService:
    """Evaluate per-item content-alert rules."""

    def evaluate(
        self,
        item: Mapping[str, Any],
        rules: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        matched: list[dict[str, Any]] = []
        # There are only three distinct scopes, so a per-scope cache builds each
        # haystack at most once even when many rules share a scope (Qodo, the
        # page-wide join over a large page is not free) -- keyed by the
        # NORMALIZED scope so the whole `str(... or "anywhere").lower()` family
        # collapses to one key.
        haystack_by_scope: dict[str, str] = {}
        for rule in rules:
            conditions = dict(rule.get("conditions") or {})
            pattern = str(conditions.get("pattern") or "")
            if not pattern:
                continue
            rule_type = str(conditions.get("type") or "keyword").lower()
            # Shared with `WatchlistFilterService` so the two cannot drift.
            # Page-scoped ("anywhere") by default for a site change -- see
            # `watchlist_rule_matching` -- with a per-rule opt-in (TASK-1363)
            # to narrow to just the text a change added or removed.
            scope = str(conditions.get("scope") or "anywhere").lower()
            if scope not in haystack_by_scope:
                haystack_by_scope[scope] = build_rule_haystack(item, scope=scope)
            haystack = haystack_by_scope[scope]
            is_match = False
            if rule_type == "keyword":
                is_match = pattern.lower() in haystack
            elif rule_type == "regex":
                try:
                    is_match = bool(re.search(pattern, haystack, re.IGNORECASE))
                except re.error:
                    is_match = False
            if is_match:
                matched.append({
                    "rule_id": rule.get("id"),
                    "rule_name": rule.get("name"),
                    "severity": rule.get("severity", "warning"),
                    "message": f"Alert '{rule.get('name')}' matched item: {item.get('title') or item.get('url')}",
                    "notification_payload": {
                        "kind": "watchlist_content_alert",
                        "source_domain": "watchlists",
                        "source_entity_kind": "watchlist_item",
                        "source_entity_id": str(item.get("id") or ""),
                        "rule_id": str(rule.get("id")),
                        "dedupe_key": f"watchlist-content-alert:{rule.get('id')}:{item.get('id')}",
                    },
                })
        return matched
