# Watchlists TUI Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the placeholder `watchlists_collections` destination shell with a full three-pane TUI management screen supporting local and server backends, with Overview, Sources, Items, Runs, and Alert Rules sections.

**Architecture:** Reuse the existing `WatchlistScopeService` for backend routing; extend `SubscriptionsDB` and `LocalWatchlistsService` for local scraped-items, source-level filters, and content alerts; decompose the UI into a thin shell plus focused pane/controller modules under `tldw_chatbook/UI/Watchlists_Modules`.

**Tech Stack:** Python 3.11+, Textual, SQLite, pydantic, httpx, pytest.

**Spec:** [docs/superpowers/specs/2026-07-18-watchlists-tui-screen-design.md](../specs/2026-07-18-watchlists-tui-screen-design.md)  
**ADR:** [backlog/decisions/018-watchlists-tui-screen.md](../../../backlog/decisions/018-watchlists-tui-screen.md)

---

## File structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/DB/Subscriptions_DB.py` | Idempotent schema migration for new columns/indexes and widened `subscription_filters` CHECK constraint. |
| `tldw_chatbook/Subscriptions/watchlist_filter_service.py` | Evaluate source-level `include`/`exclude`/`flag` filters against candidate items. |
| `tldw_chatbook/Subscriptions/watchlist_content_alert_service.py` | Evaluate keyword/regex content-alert rules per item and build notification payloads. |
| `tldw_chatbook/Subscriptions/watchlist_preview_service.py` | Dry-run fetch for a source without persistence (local backend). |
| `tldw_chatbook/Subscriptions/watchlist_opml_service.py` | Parse and serialize OPML for local bulk import/export. |
| `tldw_chatbook/Subscriptions/local_watchlists_service.py` | Extended to honor `active` on create, apply filters/content alerts, upsert items. |
| `tldw_chatbook/UI/Watchlists_Modules/__init__.py` | Package marker. |
| `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py` | Backend authority switching, policy enforcement, operation routing, error handling. |
| `tldw_chatbook/UI/Watchlists_Modules/watchlists_navigator.py` | Left-rail section list widget. |
| `tldw_chatbook/UI/Watchlists_Modules/overview_pane.py` | Dashboard cards and recent-failed-runs list. |
| `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py` | Source list, create/edit form, filter editor, OPML, preview/check-now. |
| `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py` | Run list and full run inspector. |
| `tldw_chatbook/UI/Watchlists_Modules/items_pane.py` | Item reader with smart counts and batch actions. |
| `tldw_chatbook/UI/Watchlists_Modules/alert_rules_pane.py` | Health + content alert rule editor and inbox. |
| `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` | Thin shell composing rail + workbench + inspector; message routing. |
| `tldw_chatbook/UI/Screens/subscription_screen.py` | Deleted. |
| `tldw_chatbook/app.py` | Remove subscription-specific handlers; update handoff staging. |
| `tldw_chatbook/Constants.py` | Review/remove `TAB_SUBSCRIPTIONS` usage. |
| `tldw_chatbook/UI/Navigation/screen_registry.py` | Remove `subscriptions` route. |
| `tldw_chatbook/UI/Navigation/shell_destinations.py` | Keep `subscriptions` as alias for `watchlists_collections`. |
| `tldw_chatbook/UI/Workbench/route_inventory.py` | Update owner mapping. |

---

## Phase 1: Foundation (local schema + services + controller)

### Task 1: Add Watchlists schema migration to SubscriptionsDB

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py`
- Test: `Tests/DB/test_subscriptions_db.py` (create if missing; otherwise extend)

**Goal:** Idempotently add `queued_for_briefing`, `run_id`, `alert_matches` to `subscription_items`; add `priority`, `is_include_required` to `subscription_filters`; widen the `action` CHECK constraint; add indexes.

- [ ] **Step 1: Write failing test**

```python
# Tests/DB/test_subscriptions_db.py
import pytest
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

@pytest.fixture
def db(tmp_path):
    return SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")


def test_watchlists_columns_exist(db):
    cursor = db.conn.cursor()
    cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_items)")}
    assert "queued_for_briefing" in cols
    assert "run_id" in cols
    assert "alert_matches" in cols

    cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_filters)")}
    assert "priority" in cols
    assert "is_include_required" in cols


def test_subscription_filters_action_constraint_allows_include(db):
    cursor = db.conn.cursor()
    cursor.execute(
        "INSERT INTO subscription_filters (subscription_id, name, conditions, action) VALUES (?, ?, ?, ?)",
        (1, "include ai", "{}", "include"),
    )
    db.conn.commit()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest Tests/DB/test_subscriptions_db.py::test_watchlists_columns_exist -v
```
Expected: FAIL — columns missing.

- [ ] **Step 3: Implement `_ensure_watchlists_schema`**

Add near the end of `_initialize_schema` in `tldw_chatbook/DB/Subscriptions_DB.py`:

```python
    def _ensure_watchlists_schema(self):
        """Idempotent migration for watchlists screen schema additions."""
        with closing(self._get_connection()) as conn:
            cursor = conn.cursor()

            # Add columns to subscription_items
            items_cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_items)")}
            if "queued_for_briefing" not in items_cols:
                cursor.execute("ALTER TABLE subscription_items ADD COLUMN queued_for_briefing BOOLEAN DEFAULT 0")
            if "run_id" not in items_cols:
                cursor.execute("ALTER TABLE subscription_items ADD COLUMN run_id INTEGER")
            if "alert_matches" not in items_cols:
                cursor.execute("ALTER TABLE subscription_items ADD COLUMN alert_matches TEXT")

            # Add columns to subscription_filters
            filters_cols = {row[1] for row in cursor.execute("PRAGMA table_info(subscription_filters)")}
            if "priority" not in filters_cols:
                cursor.execute("ALTER TABLE subscription_filters ADD COLUMN priority INTEGER DEFAULT 0")
            if "is_include_required" not in filters_cols:
                cursor.execute("ALTER TABLE subscription_filters ADD COLUMN is_include_required BOOLEAN DEFAULT 0")

            # Widen CHECK constraint on subscription_filters.action
            existing_check = None
            for row in cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='subscription_filters'"):
                existing_check = row[0]
            if existing_check and "include" not in existing_check:
                cursor.execute("""
                    CREATE TABLE subscription_filters_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subscription_id INTEGER,
                        name TEXT NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        conditions TEXT NOT NULL,
                        action TEXT NOT NULL CHECK(action IN ('auto_ingest','auto_ignore','tag','priority','notify','include','exclude','flag')),
                        action_params TEXT,
                        priority INTEGER DEFAULT 0,
                        is_include_required BOOLEAN DEFAULT 0,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (subscription_id) REFERENCES subscriptions(id) ON DELETE CASCADE
                    )
                """)
                cursor.execute("""
                    INSERT INTO subscription_filters_new
                        (id, subscription_id, name, is_active, conditions, action, action_params, priority, is_include_required, created_at, updated_at)
                    SELECT id, subscription_id, name, is_active, conditions, action, action_params, priority, is_include_required, created_at, updated_at
                    FROM subscription_filters
                """)
                cursor.execute("DROP TABLE subscription_filters")
                cursor.execute("ALTER TABLE subscription_filters_new RENAME TO subscription_filters")
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_subscription_filters_timestamp
                    AFTER UPDATE ON subscription_filters
                    BEGIN
                        UPDATE subscription_filters SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
                    END
                """)

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscription_items_run_id ON subscription_items(run_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_subscription_items_queued ON subscription_items(queued_for_briefing, status)")
            conn.commit()
```

Call `_ensure_watchlists_schema()` at the end of `_initialize_schema`.

- [ ] **Step 4: Run tests**

```bash
pytest Tests/DB/test_subscriptions_db.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py Tests/DB/test_subscriptions_db.py
git commit -m "feat(watchlists): extend SubscriptionsDB schema for items, filters, and content alerts"
```

---

### Task 2: Add local filter evaluation service

**Files:**
- Create: `tldw_chatbook/Subscriptions/watchlist_filter_service.py`
- Test: `Tests/Subscriptions/test_watchlist_filter_service.py`

**Goal:** Given a list of candidate items and filter rows, return each item with a decision (`include`, `exclude`, `flag`) and matched filter.

- [ ] **Step 1: Write failing test**

```python
# Tests/Subscriptions/test_watchlist_filter_service.py
import pytest
from tldw_chatbook.Subscriptions.watchlist_filter_service import WatchlistFilterService

@pytest.fixture
def service():
    return WatchlistFilterService()


def test_keyword_include(service):
    items = [{"title": "AI news", "summary": "", "content": ""}]
    filters = [
        {"id": 1, "priority": 1, "action": "include", "conditions": {"type": "keyword", "mode": "contains", "pattern": "AI"}, "is_include_required": False}
    ]
    result = service.evaluate(items, filters)
    assert result[0]["filter_decision"] == "include"


def test_exclude_wins_over_include(service):
    items = [{"title": "AI news", "summary": "", "content": ""}]
    filters = [
        {"id": 1, "priority": 1, "action": "include", "conditions": {"type": "keyword", "pattern": "AI"}, "is_include_required": False},
        {"id": 2, "priority": 0, "action": "exclude", "conditions": {"type": "keyword", "pattern": "AI"}, "is_include_required": False},
    ]
    result = service.evaluate(items, filters)
    # Lower priority number evaluated first; exclude wins.
    assert result[0]["filter_decision"] == "exclude"
```

- [ ] **Step 2: Run test**

```bash
pytest Tests/Subscriptions/test_watchlist_filter_service.py -v
```
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement service**

```python
# tldw_chatbook/Subscriptions/watchlist_filter_service.py
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
```

- [ ] **Step 4: Run tests**

```bash
pytest Tests/Subscriptions/test_watchlist_filter_service.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_filter_service.py Tests/Subscriptions/test_watchlist_filter_service.py
git commit -m "feat(watchlists): add local filter evaluation service"
```

---

### Task 3: Add local content-alert evaluation service

**Files:**
- Create: `tldw_chatbook/Subscriptions/watchlist_content_alert_service.py`
- Test: `Tests/Subscriptions/test_watchlist_content_alert_service.py`

**Goal:** Given an item and content-alert rules, return matched rule IDs and notification payloads.

- [ ] **Step 1: Write failing test**

```python
# Tests/Subscriptions/test_watchlist_content_alert_service.py
import pytest
from tldw_chatbook.Subscriptions.watchlist_content_alert_service import WatchlistContentAlertService

@pytest.fixture
def service():
    return WatchlistContentAlertService()


def test_keyword_match(service):
    rules = [
        {"id": 1, "name": "AI alert", "severity": "warning", "conditions": {"type": "keyword", "pattern": "AI"}}
    ]
    matches = service.evaluate({"title": "AI news", "summary": "", "content": ""}, rules)
    assert len(matches) == 1
    assert matches[0]["rule_id"] == 1
    assert matches[0]["severity"] == "warning"
```

- [ ] **Step 2: Run test**

```bash
pytest Tests/Subscriptions/test_watchlist_content_alert_service.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement service**

Reuse the matching logic from `watchlist_filter_service`. Provide a method that returns matched rules with message + payload.

```python
# tldw_chatbook/Subscriptions/watchlist_content_alert_service.py
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
```

- [ ] **Step 4: Run tests**

```bash
pytest Tests/Subscriptions/test_watchlist_content_alert_service.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_content_alert_service.py Tests/Subscriptions/test_watchlist_content_alert_service.py
git commit -m "feat(watchlists): add local content-alert evaluation service"
```

---

### Task 4: Extend LocalWatchlistsService

**Files:**
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py`
- Test: `Tests/Subscriptions/test_local_watchlists_service.py` (extend)

**Goal:** Honor `active` on create, evaluate filters/content alerts during `execute_run`, upsert items into `subscription_items`, and return run stats.

- [ ] **Step 1: Write failing tests**

Add to `Tests/Subscriptions/test_local_watchlists_service.py`:

```python
import pytest
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService


@pytest.fixture
def local_service(tmp_path):
    db = SubscriptionsDB(str(tmp_path / "subs.db"), client_id="test")
    return LocalWatchlistsService(db_factory=lambda: db)


@pytest.mark.asyncio
async def test_create_source_honors_inactive(local_service):
    result = await local_service.create_source({"name": "Inactive", "source_type": "rss", "url": "http://example.com/feed", "active": False})
    assert result["active"] is False


@pytest.mark.asyncio
async def test_execute_run_persists_items_and_evaluates_filters(local_service):
    source = await local_service.create_source({"name": "Test", "source_type": "rss", "url": "http://example.com/feed"})
    run = await local_service.launch_run(source_id=source["source_id"])
    # Stub executor injection omitted for brevity; see implementation notes.
```

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Subscriptions/test_local_watchlists_service.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement changes**

1. In `create_source`, accept `active` from payload:
   ```python
   source_id = db.add_subscription(
       ...
       is_active=bool(payload.get("active", True)),
       ...
   )
   ```
2. Inject `filter_service` and `content_alert_service` via constructor defaults.
3. In `execute_run`, after fetching items:
   - Apply filters.
   - Evaluate content-alert rules for the source (`subscription_filters` where `action='notify'`).
   - Upsert items using `INSERT ... ON CONFLICT(subscription_id, url, content_hash) DO UPDATE`.
   - Store `run_id` and `alert_matches` JSON.
4. Keep existing health-alert evaluation.

- [ ] **Step 4: Run tests**

```bash
pytest Tests/Subscriptions/test_local_watchlists_service.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/local_watchlists_service.py Tests/Subscriptions/test_local_watchlists_service.py
git commit -m "feat(watchlists): extend local watchlist service with filters, content alerts, and item upsert"
```

---

### Task 5: Add local preview service

**Files:**
- Create: `tldw_chatbook/Subscriptions/watchlist_preview_service.py`
- Test: `Tests/Subscriptions/test_watchlist_preview_service.py`

**Goal:** Fetch a source once without persistence and return candidate items.

- [ ] **Step 1: Write failing test**

```python
# Tests/Subscriptions/test_watchlist_preview_service.py
import pytest
from tldw_chatbook.Subscriptions.watchlist_preview_service import WatchlistPreviewService

@pytest.mark.asyncio
async def test_preview_returns_items_or_empty():
    svc = WatchlistPreviewService()
    result = await svc.preview({"type": "rss", "source": "http://example.com/feed"})
    assert "items" in result
```

- [ ] **Step 2: Run test**

```bash
pytest Tests/Subscriptions/test_watchlist_preview_service.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/Subscriptions/watchlist_preview_service.py
from __future__ import annotations

from typing import Any, Mapping

from .local_watchlists_service import LocalWatchlistsService


class WatchlistPreviewService:
    """Dry-run fetch a source without persisting anything."""

    async def preview(self, source_config: Mapping[str, Any]) -> dict[str, Any]:
        # Re-use the existing fetchers from LocalWatchlistsService via a temporary executor.
        service = LocalWatchlistsService(db_factory=lambda: None, run_executor=self._no_op_executor)
        items = await service._default_run_executor(source_config, db=None)
        return {"items": items.get("items", []), "log_text": "Preview completed."}

    async def _no_op_executor(self, subscription: Mapping[str, Any]) -> dict[str, Any]:
        return {"items": []}
```

Use the actual `FeedMonitor` / `URLMonitor` imports from `local_watchlists_service` instead of the no-op.

- [ ] **Step 4: Run tests**

```bash
pytest Tests/Subscriptions/test_watchlist_preview_service.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_preview_service.py Tests/Subscriptions/test_watchlist_preview_service.py
git commit -m "feat(watchlists): add local source preview service"
```

---

### Task 6: Add local OPML import/export service

**Files:**
- Create: `tldw_chatbook/Subscriptions/watchlist_opml_service.py`
- Test: `Tests/Subscriptions/test_watchlist_opml_service.py`
- Fixture: `Tests/fixtures/watchlists/sample.opml`

**Goal:** Parse OPML outlines into source create payloads and serialize sources back to OPML.

- [ ] **Step 1: Write failing test**

```python
# Tests/Subscriptions/test_watchlist_opml_service.py
import pytest
from tldw_chatbook.Subscriptions.watchlist_opml_service import WatchlistOpmlService


def test_parse_opml():
    xml = '''<?xml version="1.0"?><opml version="2.0"><body><outline text="Tech" title="Tech"><outline text="AI" title="AI" type="rss" xmlUrl="http://example.com/ai"/></outline></body></opml>'''
    svc = WatchlistOpmlService()
    items = svc.parse(xml)
    assert len(items) == 1
    assert items[0]["url"] == "http://example.com/ai"
    assert items[0]["source_type"] == "rss"


def test_export_opml():
    svc = WatchlistOpmlService()
    xml = svc.export([
        {"name": "AI", "url": "http://example.com/ai", "source_type": "rss"}
    ])
    assert "http://example.com/ai" in xml
```

- [ ] **Step 2: Run test**

```bash
pytest Tests/Subscriptions/test_watchlist_opml_service.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

Use Python stdlib `xml.etree.ElementTree`.

```python
# tldw_chatbook/Subscriptions/watchlist_opml_service.py
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any


class WatchlistOpmlService:
    """Minimal OPML import/export for watchlist sources."""

    def parse(self, xml_text: str) -> list[dict[str, Any]]:
        root = ET.fromstring(xml_text)
        items: list[dict[str, Any]] = []
        for outline in root.iter("outline"):
            url = outline.get("xmlUrl") or outline.get("htmlUrl")
            if not url:
                continue
            source_type = outline.get("type", "rss").lower()
            if source_type not in {"rss", "site", "forum"}:
                source_type = "rss"
            items.append({
                "name": outline.get("text") or outline.get("title") or "Untitled",
                "url": url,
                "source_type": source_type,
            })
        return items

    def export(self, sources: list[dict[str, Any]]) -> str:
        root = ET.Element("opml", {"version": "2.0"})
        body = ET.SubElement(root, "body")
        for source in sources:
            outline = ET.SubElement(body, "outline", {
                "text": str(source.get("name") or "Untitled"),
                "title": str(source.get("name") or "Untitled"),
                "type": str(source.get("source_type") or "rss"),
                "xmlUrl": str(source.get("url") or ""),
            })
        return ET.tostring(root, encoding="unicode")
```

- [ ] **Step 4: Run tests**

```bash
pytest Tests/Subscriptions/test_watchlist_opml_service.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_opml_service.py Tests/Subscriptions/test_watchlist_opml_service.py Tests/fixtures/watchlists/sample.opml
git commit -m "feat(watchlists): add local OPML import/export service"
```

---

### Task 7: Create WatchlistsBackendController

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py`
- Test: `Tests/Watchlists/test_watchlists_backend_controller.py`

**Goal:** Encapsulate backend switching, policy enforcement, operation routing, and error-to-recovery conversion.

- [ ] **Step 1: Write failing test**

```python
# Tests/Watchlists/test_watchlists_backend_controller.py
import pytest
from tldw_chatbook.UI.Watchlists_Modules.watchlists_backend_controller import WatchlistsBackendController


class FakeScopeService:
    async def list_watch_items(self, *, runtime_backend, **kwargs):
        return [{"id": 1, "title": "Source"}]


def test_controller_normalizes_backend():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    assert ctrl._normalize_backend("server") == "server"
    assert ctrl._normalize_backend(None) == "local"


@pytest.mark.asyncio
async def test_list_sources_routes_to_scope_service():
    ctrl = WatchlistsBackendController(app_instance=None, scope_service=FakeScopeService(), server_service=None)
    items = await ctrl.list_sources(runtime_backend="local")
    assert len(items) == 1
```

- [ ] **Step 2: Run test**

```bash
pytest Tests/Watchlists/test_watchlists_backend_controller.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py
from __future__ import annotations

import inspect
from typing import Any


class WatchlistsBackendController:
    """Route watchlist operations to the active local/server authority."""

    def __init__(
        self,
        *,
        app_instance: Any,
        scope_service: Any,
        server_service: Any,
        notification_dispatch_service: Any = None,
    ) -> None:
        self.app_instance = app_instance
        self.scope_service = scope_service
        self.server_service = server_service
        self.notification_dispatch_service = notification_dispatch_service

    @staticmethod
    def _normalize_backend(runtime_backend: Any) -> str:
        return str(runtime_backend or "local").strip().lower() or "local"

    async def _maybe_await(self, value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    async def list_sources(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_watch_items(runtime_backend=backend, **kwargs)
        )
        return [dict(item) for item in list(result or [])]

    async def create_source(self, *, runtime_backend: str | None = None, payload: dict[str, Any]) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.create_watch_item(runtime_backend=backend, payload=payload)
        )
        return dict(result)

    async def update_source(self, *, runtime_backend: str | None = None, item_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.update_watch_item(runtime_backend=backend, item_id=item_id, payload=payload)
        )
        return dict(result)

    async def delete_source(self, *, runtime_backend: str | None = None, item_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.delete_watch_item(runtime_backend=backend, item_id=item_id)
        )
        return dict(result)

    async def list_runs(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_runs(runtime_backend=backend, **kwargs)
        )
        return [dict(item) for item in list(result or [])]

    async def get_run(self, *, runtime_backend: str | None = None, run_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.get_run(runtime_backend=backend, run_id=run_id)
        )
        return dict(result)

    async def observe_run(self, *, runtime_backend: str | None = None, run_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.observe_run(runtime_backend=backend, run_id=run_id)
        )
        return dict(result)

    async def cancel_run(self, *, runtime_backend: str | None = None, run_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.cancel_run(runtime_backend=backend, run_id=run_id)
        )
        return dict(result)

    async def launch_run(self, *, runtime_backend: str | None = None, source_id: Any = None) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.launch_run(runtime_backend=backend, source_id=source_id)
        )
        return dict(result)

    async def list_alert_rules(self, *, runtime_backend: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.list_alert_rules(runtime_backend=backend, **kwargs)
        )
        return [dict(item) for item in list(result or [])]

    async def save_alert_rule(self, *, runtime_backend: str | None = None, payload: dict[str, Any]) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.save_alert_rule(runtime_backend=backend, payload=payload)
        )
        return dict(result)

    async def delete_alert_rule(self, *, runtime_backend: str | None = None, rule_id: Any) -> dict[str, Any]:
        backend = self._normalize_backend(runtime_backend)
        result = await self._maybe_await(
            self.scope_service.delete_alert_rule(runtime_backend=backend, rule_id=rule_id)
        )
        return dict(result)

    def list_unsupported_capabilities(self, *, runtime_backend: str | None = None) -> list[dict[str, Any]]:
        backend = self._normalize_backend(runtime_backend)
        method = getattr(self.scope_service, "list_unsupported_capabilities", None)
        if callable(method):
            return list(method(runtime_backend=backend))
        return []
```

Extend with preview/check-now/OPML routing once the service helpers exist.

- [ ] **Step 4: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_backend_controller.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py Tests/Watchlists/test_watchlists_backend_controller.py
git commit -m "feat(watchlists): add backend controller for watchlists screen"
```

---

## Phase 2: Slice 1A UI — Overview, Sources, Runs

### Task 8: Create `Watchlists_Modules` package and shared navigator widget

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/__init__.py`
- Create: `tldw_chatbook/UI/Watchlists_Modules/watchlists_navigator.py`

**Goal:** Left-rail widget with Overview, Sources, Items, Runs, Rules buttons.

- [ ] **Step 1: Implement**

```python
# tldw_chatbook/UI/Watchlists_Modules/watchlists_navigator.py
from __future__ import annotations

from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button


class SectionSelected(Message):
    def __init__(self, section_id: str) -> None:
        self.section_id = section_id
        super().__init__()


class WatchlistsNavigator(Vertical):
    SECTIONS = [
        ("overview", "Overview"),
        ("sources", "Sources"),
        ("items", "Items"),
        ("runs", "Runs"),
        ("rules", "Rules"),
    ]

    def compose(self):
        for section_id, label in self.SECTIONS:
            yield Button(label, id=f"nav-{section-id}", classes="watchlists-nav-button")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        section_id = str(event.button.id).replace("nav-", "")
        self.post_message(SectionSelected(section_id))
```

- [ ] **Step 2: Basic test**

```bash
pytest Tests/Watchlists/test_watchlists_navigator.py -v
```

Create a minimal pilot test that asserts all five buttons exist.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/__init__.py tldw_chatbook/UI/Watchlists_Modules/watchlists_navigator.py Tests/Watchlists/test_watchlists_navigator.py
git commit -m "feat(watchlists): add watchlists navigator widget"
```

---

### Task 9: Rewrite the Watchlists screen shell

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: `Tests/UI/test_watchlists_destination_shell.py` (rewrite from existing tests if needed)

**Goal:** Keep public contract (route, class name, stable selectors) but replace internals with shell + panes + controller.

- [ ] **Step 1: Implement shell**

Rewrite `watchlists_collections_screen.py` to:
- Keep `class WatchlistsCollectionsScreen(BaseAppScreen)`.
- Add reactive state: `active_section`, `runtime_backend`, `selected_source`, `selected_run`, `recovery_state`.
- Compose: header bar with backend `Select`, left `WatchlistsNavigator`, middle `WatchlistsWorkbench` container, right `WatchlistsInspector` container.
- Instantiate controller in `__init__` from `app.watchlist_scope_service` etc.
- Handle `SectionSelected`, load pane into middle container.
- Preserve existing stable selectors (`watchlists-collections-title`, `wc-attach-to-console`, etc.) for backward compatibility where possible.

- [ ] **Step 2: Run destination tests**

```bash
pytest Tests/UI/test_watchlists_destination_shell.py -v
```
Expected: PASS after updating tests to the new layout.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/UI/test_watchlists_destination_shell.py
git commit -m "feat(watchlists): rewrite watchlists screen shell with navigator and panes"
```

---

### Task 10: Implement Overview pane

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/overview_pane.py`
- Test: `Tests/Watchlists/test_watchlists_overview_pane.py`

**Goal:** Show summary cards, health alerts, and recent failed runs.

- [ ] **Step 1: Implement**

A Textual `Vertical`/`Grid` of `Static` widgets and a `DataTable` for recent failed runs. Reactive data drives updates.

```python
# tldw_chatbook/UI/Watchlists_Modules/overview_pane.py
from textual.containers import Grid, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Static


class OverviewPane(Vertical):
    data = reactive({}, recompose=True)

    def compose(self):
        with Grid(id="watchlists-overview-grid"):
            yield Static("Feeds: -", id="overview-feeds")
            yield Static("Updates: -", id="overview-updates")
            yield Static("Activity: -", id="overview-activity")
        yield Static("Recent failed runs", classes="pane-title")
        yield DataTable(id="overview-failed-runs")

    def watch_data(self):
        table = self.query_one("#overview-failed-runs", DataTable)
        table.clear()
        for run in self.data.get("failed_runs", []):
            table.add_row(run.get("source_title"), run.get("status"), run.get("error_msg"))
```

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_overview_pane.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/overview_pane.py Tests/Watchlists/test_watchlists_overview_pane.py
git commit -m "feat(watchlists): add overview dashboard pane"
```

---

### Task 11: Implement Sources pane

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`
- Test: `Tests/Watchlists/test_watchlists_sources_pane.py`

**Goal:** Source list with search/type filter, DataTable, create/edit form, actions.

- [ ] **Step 1: Implement**

Build a pane with:
- Top bar: `Input` for search, `Select` for type, `Button` New Source.
- `DataTable` for source list columns: Name, Type, Status, Last scraped, Active, Actions.
- Selection updates `selected_source` reactive and posts a message to the shell.
- Create/edit uses a modal or inline form with fields: name, URL, source_type, active, tags, extraction rules.
- Actions wired through the controller.

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_sources_pane.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/sources_pane.py Tests/Watchlists/test_watchlists_sources_pane.py
git commit -m "feat(watchlists): add sources management pane"
```

---

### Task 12: Implement Runs pane

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py`
- Test: `Tests/Watchlists/test_watchlists_runs_pane.py`

**Goal:** Run list and run inspector (stats, items, logs).

- [ ] **Step 1: Implement**

Build a pane with:
- `DataTable` for runs: Source/Job, Status, Started, Duration, Found, Processed, Filtered, Errors, Actions.
- Selection shows run detail in the right inspector:
  - `Static` for stats.
  - `DataTable` for items.
  - `Static` for log text.
- Polling worker refreshes run detail while status is `running`.
- Cancel and Re-run actions call controller.

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_runs_pane.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/runs_pane.py Tests/Watchlists/test_watchlists_runs_pane.py
git commit -m "feat(watchlists): add runs list and inspector pane"
```

---

### Task 13: Wire screen messages and inspector pane

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Create: `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`
- Test: `Tests/UI/test_watchlists_inspector.py`

**Goal:** Selecting an entity updates the right inspector with context-aware actions.

- [ ] **Step 1: Implement inspector pane**

A dynamic `Vertical` that inspects `selected_entity` and renders action buttons: Preview, Check now, Stage in Console, Delete.

- [ ] **Step 2: Handle messages in shell**

The shell listens for messages from panes (e.g., `SourceSelected`, `RunSelected`) and updates reactive state, which re-composes the inspector.

- [ ] **Step 3: Run tests**

```bash
pytest Tests/UI/test_watchlists_inspector.py -v
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/UI/test_watchlists_inspector.py
git commit -m "feat(watchlists): wire pane selection to inspector actions"
```

---

### Task 14: App wiring and route cleanup

**Files:**
- Modify: `tldw_chatbook/Constants.py`
- Modify: `tldw_chatbook/UI/Navigation/screen_registry.py`
- Modify: `tldw_chatbook/UI/Workbench/route_inventory.py`
- Modify: `tldw_chatbook/app.py`
- Delete: `tldw_chatbook/UI/SubscriptionWindow.py`
- Test: `Tests/UI/test_screen_navigation.py`, `Tests/UI/test_shell_destinations.py`

**Goal:** Retire `SubscriptionWindow`, fold `subscriptions` route into the new screen.

- [ ] **Step 1: Update Constants**

Remove `TAB_SUBSCRIPTIONS` from `ALL_TABS` and `TAB_DISPLAY_LABELS`, or repurpose the tab label to point to Watchlists. Keep the constant itself if other code references it, but update `ALL_TABS` so the tab bar does not show a separate Subscriptions tab.

- [ ] **Step 2: Update screen_registry**

Remove the `subscriptions` route entry; keep the alias in `screen_registry._SCREEN_ALIASES` mapping `subscriptions -> watchlists_collections`.

- [ ] **Step 3: Update shell_destinations**

Already maps `subscriptions` to `watchlists_collections`; no change needed.

- [ ] **Step 4: Update route_inventory**

Ensure `subscriptions` and `subscription` owners map to `watchlists_collections`.

- [ ] **Step 5: Update app.py**

- Remove the `TAB_SUBSCRIPTIONS` event handler map entry around line 4347.
- Update `_stage_subscription_watchlist_run_context` to set pending state on the new screen (e.g., `pending_watchlists_section = "runs"`, `pending_watchlists_run_id`).
- Remove import of `SubscriptionWindow`.

- [ ] **Step 6: Delete SubscriptionWindow**

```bash
rm tldw_chatbook/UI/SubscriptionWindow.py
```

- [ ] **Step 7: Run navigation tests**

```bash
pytest Tests/UI/test_screen_navigation.py Tests/UI/test_shell_destinations.py -v
```
Expected: PASS after test updates.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Constants.py tldw_chatbook/UI/Navigation/screen_registry.py tldw_chatbook/UI/Workbench/route_inventory.py tldw_chatbook/app.py
git rm tldw_chatbook/UI/SubscriptionWindow.py
git commit -m "feat(watchlists): retire SubscriptionWindow and fold subscriptions route into Watchlists"
```

---

### Task 15: Slice 1A integration regression

**Files:** all Slice 1A files.

**Goal:** Verify the screen works end-to-end locally.

- [ ] **Step 1: Run focused test suite**

```bash
pytest Tests/Watchlists Tests/UI/test_watchlists_destination_shell.py Tests/UI/test_watchlists_sources.py Tests/UI/test_watchlists_runs.py Tests/UI/test_watchlists_inspector.py -v
```
Expected: PASS.

- [ ] **Step 2: Manual smoke**

```bash
python3 -m tldw_chatbook.app
```
Navigate to Watchlists, confirm Overview/Sources/Runs render and backend toggle works.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(watchlists): complete Slice 1A integration"
```

---

## Phase 3: Slice 1B UI — Items, Rules, filters, OPML, preview, Check now

### Task 16: Extend controller with preview/check-now/OPML

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py`
- Test: `Tests/Watchlists/test_watchlists_backend_controller.py` (extend)

**Goal:** Route preview, check-now, OPML, and item/rule operations.

- [ ] **Step 1: Add methods**

```python
async def preview_source(self, *, runtime_backend, source_config):
    backend = self._normalize_backend(runtime_backend)
    if backend == "local":
        from ...Subscriptions.watchlist_preview_service import WatchlistPreviewService
        return await WatchlistPreviewService().preview(source_config)
    # server
    client = self.server_service._require_client()
    return await client.test_watchlist_source(source_config)

async def check_now_source(self, *, runtime_backend, source_id):
    backend = self._normalize_backend(runtime_backend)
    if backend == "local":
        return await self.launch_run(runtime_backend=backend, source_id=source_id)
    client = self.server_service._require_client()
    return await client.check_watchlist_sources_now([int(source_id)])

async def import_opml(self, *, runtime_backend, xml_text):
    backend = self._normalize_backend(runtime_backend)
    if backend == "local":
        from ...Subscriptions.watchlist_opml_service import WatchlistOpmlService
        items = WatchlistOpmlService().parse(xml_text)
        created = []
        for item in items:
            created.append(await self.create_source(runtime_backend=backend, payload=item))
        return {"items": created, "total": len(items), "created": len(created)}
    client = self.server_service._require_client()
    return await client.import_watchlist_sources(xml_text)

async def export_opml(self, *, runtime_backend):
    backend = self._normalize_backend(runtime_backend)
    if backend == "local":
        sources = await self.list_sources(runtime_backend=backend)
        from ...Subscriptions.watchlist_opml_service import WatchlistOpmlService
        return WatchlistOpmlService().export(sources)
    client = self.server_service._require_client()
    return await client.export_watchlist_sources()
```

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_backend_controller.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py Tests/Watchlists/test_watchlists_backend_controller.py
git commit -m "feat(watchlists): extend backend controller with preview, check-now, and OPML"
```

---

### Task 17: Add filter editor to Sources pane

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`
- Test: `Tests/Watchlists/test_watchlists_sources_pane.py` (extend)

**Goal:** Edit `include`/`exclude`/`flag` filters in priority order.

- [ ] **Step 1: Implement filter editor**

Add a collapsible section or modal with a `DataTable` of filters and a form: type (keyword/regex), pattern, action, priority.
On save, insert/update a `subscription_filters` row via `SubscriptionsDB` directly or through a new controller method.

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_sources_pane.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/sources_pane.py Tests/Watchlists/test_watchlists_sources_pane.py
git commit -m "feat(watchlists): add source-level filter editor"
```

---

### Task 18: Add preview and Check now actions to Sources pane

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`
- Modify: `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`
- Test: `Tests/Watchlists/test_watchlists_sources_pane.py`

**Goal:** User can dry-run a source and trigger a real run.

- [ ] **Step 1: Wire actions**

- "Preview" calls `controller.preview_source(source_config)` and shows candidate items in a modal/Static.
- "Check now" calls `controller.check_now_source(source_id)` and switches to Runs section.

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_sources_pane.py::test_preview_action Tests/Watchlists/test_watchlists_sources_pane.py::test_check_now_action -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/sources_pane.py tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py Tests/Watchlists/test_watchlists_sources_pane.py
git commit -m "feat(watchlists): add source preview and check-now actions"
```

---

### Task 19: Add OPML import/export UI

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py`
- Test: `Tests/Watchlists/test_watchlists_sources_pane.py`

**Goal:** Buttons to import OPML from clipboard/file and export to clipboard.

- [ ] **Step 1: Implement**

Use Textual `Input` (paste XML) or a file picker if available. Call controller `import_opml` / `export_opml`.

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_sources_pane.py -k opml -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/sources_pane.py Tests/Watchlists/test_watchlists_sources_pane.py
git commit -m "feat(watchlists): add OPML import/export actions"
```

---

### Task 20: Implement Items pane

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/items_pane.py`
- Test: `Tests/Watchlists/test_watchlists_items_pane.py`

**Goal:** Global item reader with smart counts, batch review, queue for briefing.

- [ ] **Step 1: Implement**

Build a three-column pane:
- Left: source list filter.
- Middle: `DataTable` of items with smart filters (All/Today/Unread/Queued/Alert matches) and counts.
- Right inspector: item preview, mark reviewed, queue toggle, open external, discuss in Console.
Batch controls appear on selection.
Use `subscription_items` queries via controller.

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_items_pane.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/items_pane.py Tests/Watchlists/test_watchlists_items_pane.py
git commit -m "feat(watchlists): add items reader pane"
```

---

### Task 21: Implement Alert Rules pane

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/alert_rules_pane.py`
- Test: `Tests/Watchlists/test_watchlists_alert_rules_pane.py`

**Goal:** Health + content alert rule editor and alert inbox.

- [ ] **Step 1: Implement**

- Top: capability banner when server backend is active (content rules local-only).
- Rule list with toggle, edit, delete.
- Inline form: name, kind (health/content), condition type/pattern, severity, source scope.
- Health rules use `WatchlistScopeService.save_alert_rule`.
- Content rules write to `subscription_filters` with `action='notify'` via a small DB helper or controller method.
- Alert inbox: list of recent alert notifications filtered to `source_domain='watchlists'`.

- [ ] **Step 2: Run tests**

```bash
pytest Tests/Watchlists/test_watchlists_alert_rules_pane.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/alert_rules_pane.py Tests/Watchlists/test_watchlists_alert_rules_pane.py
git commit -m "feat(watchlists): add alert rules and inbox pane"
```

---

### Task 22: Final integration and full regression

**Files:** all changed files.

**Goal:** All tests pass and the screen is usable.

- [ ] **Step 1: Run full watchlists test suite**

```bash
pytest Tests/Watchlists Tests/UI/test_watchlists_*.py -v
```
Expected: PASS.

- [ ] **Step 2: Run related tests**

```bash
pytest Tests/Subscriptions/test_local_watchlists_service.py Tests/Subscriptions/test_watchlist_filter_service.py Tests/Subscriptions/test_watchlist_content_alert_service.py Tests/DB/test_subscriptions_db.py -v
```
Expected: PASS.

- [ ] **Step 3: Manual smoke**

```bash
python3 -m tldw_chatbook.app
```
Walk through all five sections, toggle backend, create a source, check now, view run, view items, create a rule.

- [ ] **Step 4: Update ADR / spec if needed**

If any deviations occurred during implementation, update the spec and ADR.

- [ ] **Step 5: Final commit**

```bash
git commit -m "feat(watchlists): complete Slice 1B and full Watchlists screen"
```

---

## Plan review

After completing this plan, run a final review:

```bash
# Verify every new file is imported and lint-clean
python -m compileall tldw_chatbook/UI/Watchlists_Modules tldw_chatbook/Subscriptions/watchlist_*.py
pytest Tests/Watchlists Tests/UI/test_watchlists_*.py -q
```

**Note:** The spec review subagent hit an API quota before final approval. This plan should be reviewed by a human or a fresh subagent before execution. Pay special attention to:
- Schema migration idempotency on existing user databases.
- The `subscription_filters` action CHECK constraint migration.
- Server backend content-alert rule gap (local-only in first slice).
