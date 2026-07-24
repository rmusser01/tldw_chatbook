from __future__ import annotations

import re
from typing import Any, Mapping


class WatchlistFilterService:
    """Evaluate source-level include/exclude/flag filters against candidate items."""

    VALID_ACTIONS = frozenset({"include", "exclude", "flag"})

    def evaluate(
        self,
        items: list[dict[str, Any]],
        filters: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        active_filters = [
            f for f in filters
            if f.get("action") in self.VALID_ACTIONS and f.get("is_active", True)
        ]
        active_filters.sort(key=lambda f: int(f.get("priority") or 0))

        require_include = any(f.get("is_include_required") for f in active_filters)

        results: list[dict[str, Any]] = []
        for item in items:
            decision: str | None = None
            matched_filter_id: int | None = None
            for rule in active_filters:
                if self._matches(item, rule):
                    decision = str(rule["action"])
                    matched_filter_id = rule.get("id")
                    break
            if decision is None and require_include:
                decision = "exclude"
            enriched = dict(item)
            enriched["filter_decision"] = decision or "include"
            enriched["matched_filter_id"] = matched_filter_id
            results.append(enriched)
        return results

    @staticmethod
    def _matches(item: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
        conditions = dict(rule.get("conditions") or {})
        rule_type = str(conditions.get("type") or "keyword").lower()
        mode = str(conditions.get("mode") or "contains").lower()
        pattern = str(conditions.get("pattern") or "")
        if not pattern:
            return False

        haystack = " ".join(
            str(item.get(key) or "") for key in ("title", "summary", "content", "author")
        ).lower()

        if rule_type == "keyword":
            needle = pattern.lower()
            if mode == "contains":
                return needle in haystack
            if mode == "starts_with":
                return haystack.startswith(needle)
            if mode == "ends_with":
                return haystack.endswith(needle)
            return needle in haystack

        if rule_type == "regex":
            try:
                return bool(re.search(pattern, haystack, re.IGNORECASE))
            except re.error:
                return False

        return False
