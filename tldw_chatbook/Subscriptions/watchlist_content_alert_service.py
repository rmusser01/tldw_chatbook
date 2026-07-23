from __future__ import annotations

import re
from typing import Any, Mapping


class WatchlistContentAlertService:
    """Evaluate per-item content-alert rules."""

    def evaluate(
        self,
        item: Mapping[str, Any],
        rules: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        matched: list[dict[str, Any]] = []
        haystack = " ".join(
            str(item.get(key) or "") for key in ("title", "summary", "content", "author")
        ).lower()
        for rule in rules:
            conditions = dict(rule.get("conditions") or {})
            pattern = str(conditions.get("pattern") or "")
            if not pattern:
                continue
            rule_type = str(conditions.get("type") or "keyword").lower()
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
