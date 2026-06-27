# Console Grouped Conversation Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Console rail's active-workspace-only conversation list with a grouped all-workspaces browser containing `Starred`, `Workspaces`, and `Chats`, while preserving active workspace authority for new chats, staging, readiness, and tool eligibility.

**Architecture:** Add local-only conversation marks in the ChaChaNotes database behind a dedicated service, then build a pure grouped display-state model that the Console screen owns and the rail widget renders. `ChatScreen` remains the controller for query state, group collapse preferences, star/unstar actions, and row selection, including active-workspace switching before workspace-scoped resume.

**Tech Stack:** Python 3.11, Textual, SQLite/ChaChaNotes DB, pytest, existing Console mounted test harnesses.

---

## References

- Spec: `Docs/superpowers/specs/2026-06-27-console-grouped-conversation-browser-design.md`
- ADR: `backlog/decisions/010-console-conversation-local-marks.md`
- Existing boundary ADR: `backlog/decisions/005-console-workspace-server-readiness.md`
- Screenshot workflow: `Docs/superpowers/handoffs/2026-05-08-ui-screenshot-approval-workflow-handoff.md`

## ADR Check

ADR required: yes
ADR path: `backlog/decisions/010-console-conversation-local-marks.md`
Reason: This feature adds durable local metadata and explicitly excludes it from sync/server/mirror surfaces.

## Scope Guardrails

- Do not add `starred` or any local mark field to `conversations`.
- Do not create sync triggers for `conversation_local_marks`.
- Do not add stars to Sync v2 outbox envelopes, chat metadata mirror records, or server conversation payloads.
- Do not change workspace membership semantics.
- Do not change Default workspace safety policy.
- Do not implement remote-only workspace hydration.
- Keep the existing active workspace status, sync/server readiness, runtime, and handoff rows below the grouped browser.

## File Structure

- Create `tldw_chatbook/Chat/conversation_local_marks_service.py`
  - Owns local-only mark CRUD for durable conversation ids.
  - Uses the existing `db.transaction()` pattern and parameterized SQL.
  - Does not import sync or server services.

- Modify `tldw_chatbook/DB/ChaChaNotes_DB.py`
  - Bump schema version from 16 to 17.
  - Add `conversation_local_marks` table to the full schema path.
  - Add `_MIGRATE_V16_TO_V17_SQL`, `_migrate_from_v16_to_v17`, and migration mapping.
  - Do not create sync triggers for the marks table.

- Modify `tldw_chatbook/Chat/__init__.py`
  - Export `ConversationLocalMarksService`.

- Modify `tldw_chatbook/app.py`
  - Import `ConversationLocalMarksService`.
  - Wire `self.conversation_local_marks_service` beside `local_chat_conversation_service`.

- Create `tldw_chatbook/Workspaces/conversation_browser_state.py`
  - Pure dataclasses and pure builder for grouped browser state.
  - No Textual, DB, or app imports.
  - Handles grouping, sorting, search filtering, dedupe, default collapse, and row count/status copy.

- Modify `tldw_chatbook/Workspaces/display_state.py`
  - Add optional `conversation_browser` to `ConsoleWorkspaceContextState`.
  - Keep the old `conversation_section` field during this change so fallback and existing tests remain understandable.

- Modify `tldw_chatbook/Workspaces/__init__.py`
  - Export grouped browser dataclasses and builder helpers.

- Modify `tldw_chatbook/Widgets/Console/console_workspace_context.py`
  - Render `state.conversation_browser` when present.
  - Keep legacy `_conversation_section()` fallback for tests and transitional compatibility.
  - Emit button attributes needed by `ChatScreen`: `row_key`, `conversation_id`, `native_session_id`, `scope_type`, and `workspace_id`.

- Modify `tldw_chatbook/UI/Screens/chat_screen.py`
  - Replace active-workspace-only browser assembly with all-workspaces browser assembly.
  - Keep DOM IDs `#console-workspace-conversation-search`, `#console-workspace-conversation-search-clear`, and `#console-workspace-conversations` for CSS stability.
  - Own local query, async search token, collapse preferences, row selection, and star/unstar commands.

- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss`
  - Add styles for grouped headers, nested workspace groups, star buttons, row metadata, and bounded scrolling.

- Modify `tldw_chatbook/css/tldw_cli_modular.tcss`
  - Mirror the same Console rail styles if this compiled/modular file is still checked in and used by tests/app packaging.

- Create `Tests/Chat/test_conversation_local_marks_service.py`
  - Service, migration, and sync-log exclusion tests.

- Create `Tests/Workspaces/test_console_conversation_browser_state.py`
  - Pure grouping/search/collapse/dedupe tests.

- Modify `Tests/UI/test_console_workspace_context_rail.py`
  - Widget rendering and layout tests for grouped browser state.

- Modify `Tests/UI/test_console_native_chat_flow.py`
  - Mounted Console integration tests for all-workspace browsing, search, stars, collapse persistence, and row selection matrix.

- Modify `Tests/Chat/test_chat_conversation_service.py`
  - Regression that normalized conversation metadata does not expose local marks.

- Modify `Tests/Chat/test_chat_conversation_scope_service.py`
  - Regression that chat metadata mirror report inputs remain explicit conversation records and do not gain local marks.

---

### Task 1: Local Marks Storage And Service

**Files:**
- Create: `tldw_chatbook/Chat/conversation_local_marks_service.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/Chat/__init__.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/Chat/test_conversation_local_marks_service.py`
- Test: `Tests/Chat/test_chat_conversation_service.py`

- [ ] **Step 1: Write failing service and schema tests**

Create `Tests/Chat/test_conversation_local_marks_service.py` with these tests:

```python
from __future__ import annotations

import json

from tldw_chatbook.Chat.conversation_local_marks_service import (
    ConversationLocalMarksService,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB


def _db(tmp_path):
    return CharactersRAGDB(str(tmp_path / "chacha.sqlite"), client_id="test-client")


def test_local_marks_table_exists_on_fresh_schema(tmp_path):
    db = _db(tmp_path)

    row = db.get_connection().execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        ("conversation_local_marks",),
    ).fetchone()

    assert row is not None


def test_star_unstar_is_idempotent_and_ordered(tmp_path):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    service.star_conversation("conv-a")
    service.star_conversation("conv-b")
    service.star_conversation("conv-a")

    assert service.is_starred("conv-a") is True
    assert service.is_starred("conv-b") is True
    assert service.list_marked_conversation_ids() == ("conv-a", "conv-b")

    service.unstar_conversation("conv-a")
    service.unstar_conversation("conv-a")

    assert service.is_starred("conv-a") is False
    assert service.list_marked_conversation_ids() == ("conv-b",)


def test_local_marks_tolerate_missing_conversations(tmp_path):
    db = _db(tmp_path)
    service = ConversationLocalMarksService(db)

    service.star_conversation("missing-conversation")

    assert service.is_starred("missing-conversation") is True
    assert service.list_marked_conversation_ids() == ("missing-conversation",)


def test_local_marks_do_not_create_sync_log_entries(tmp_path):
    db = _db(tmp_path)
    conversations = ChatConversationService(db)
    conversation_id = conversations.create_conversation(title="Sync Boundary")
    db.get_connection().execute("DELETE FROM sync_log")
    db.get_connection().commit()

    ConversationLocalMarksService(db).star_conversation(conversation_id)

    rows = db.get_connection().execute(
        "SELECT entity, entity_id, operation, payload FROM sync_log"
    ).fetchall()
    assert rows == []


def test_conversation_metadata_does_not_include_local_marks(tmp_path):
    db = _db(tmp_path)
    conversations = ChatConversationService(db)
    marks = ConversationLocalMarksService(db)
    conversation_id = conversations.create_conversation(title="Plain Metadata")

    marks.star_conversation(conversation_id)
    metadata = conversations.get_conversation_metadata(conversation_id)

    assert metadata is not None
    assert "starred" not in metadata
    assert "marks" not in metadata
    assert "local_marks" not in metadata
```

- [ ] **Step 2: Run the focused tests and verify they fail for the right reason**

Run:

```bash
pytest Tests/Chat/test_conversation_local_marks_service.py -q
```

Expected: fail with import error for `ConversationLocalMarksService` or missing `conversation_local_marks` table.

- [ ] **Step 3: Add ChaChaNotes schema version 17**

In `tldw_chatbook/DB/ChaChaNotes_DB.py`:

1. Change `_CURRENT_SCHEMA_VERSION = 16` to `_CURRENT_SCHEMA_VERSION = 17`.
2. Add this table to the full schema path before the final schema-version update:

```sql
CREATE TABLE IF NOT EXISTS conversation_local_marks (
  conversation_id TEXT NOT NULL,
  mark_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, mark_type)
);

CREATE INDEX IF NOT EXISTS idx_conversation_local_marks_type
  ON conversation_local_marks(mark_type, updated_at DESC, conversation_id);
```

3. Add:

```python
_MIGRATE_V16_TO_V17_SQL = """
CREATE TABLE IF NOT EXISTS conversation_local_marks (
  conversation_id TEXT NOT NULL,
  mark_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (conversation_id, mark_type)
);

CREATE INDEX IF NOT EXISTS idx_conversation_local_marks_type
  ON conversation_local_marks(mark_type, updated_at DESC, conversation_id);

UPDATE db_schema_version
   SET version = 17
 WHERE schema_name = 'rag_char_chat_schema'
   AND version = 16;
"""
```

4. Add `_migrate_from_v16_to_v17(self, conn)` following the surrounding migration method pattern.
5. Add `16: self._migrate_from_v16_to_v17` to `migration_steps`.
6. Confirm no `conversation_local_marks_*sync*` triggers are added.

- [ ] **Step 4: Implement `ConversationLocalMarksService`**

Create `tldw_chatbook/Chat/conversation_local_marks_service.py`:

```python
"""Local-only conversation organization marks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ConversationLocalMark:
    conversation_id: str
    mark_type: str
    created_at: str
    updated_at: str


class ConversationLocalMarksService:
    """Manage durable local-only marks for conversations."""

    STARRED = "starred"
    _ALLOWED_MARK_TYPES = frozenset({STARRED})

    def __init__(self, db: Any):
        self.db = db

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @classmethod
    def _mark_type(cls, mark_type: str | None) -> str:
        normalized = str(mark_type or cls.STARRED).strip().lower()
        if normalized not in cls._ALLOWED_MARK_TYPES:
            raise ValueError(f"Unsupported conversation mark_type: {mark_type!r}")
        return normalized

    @staticmethod
    def _conversation_id(conversation_id: str) -> str:
        normalized = str(conversation_id or "").strip()
        if not normalized:
            raise ValueError("conversation_id is required")
        return normalized

    def star_conversation(self, conversation_id: str) -> None:
        self.set_mark(conversation_id, self.STARRED)

    def unstar_conversation(self, conversation_id: str) -> None:
        self.clear_mark(conversation_id, self.STARRED)

    def is_starred(self, conversation_id: str) -> bool:
        return self.has_mark(conversation_id, self.STARRED)

    def set_mark(self, conversation_id: str, mark_type: str | None = None) -> None:
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        now = self._now()
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO conversation_local_marks (
                    conversation_id, mark_type, created_at, updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id, mark_type)
                DO UPDATE SET updated_at = excluded.updated_at
                """,
                (conversation_id, mark_type, now, now),
            )

    def clear_mark(self, conversation_id: str, mark_type: str | None = None) -> None:
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        with self.db.transaction() as conn:
            conn.execute(
                """
                DELETE FROM conversation_local_marks
                 WHERE conversation_id = ? AND mark_type = ?
                """,
                (conversation_id, mark_type),
            )

    def has_mark(self, conversation_id: str, mark_type: str | None = None) -> bool:
        conversation_id = self._conversation_id(conversation_id)
        mark_type = self._mark_type(mark_type)
        row = self.db.get_connection().execute(
            """
            SELECT 1
              FROM conversation_local_marks
             WHERE conversation_id = ? AND mark_type = ?
             LIMIT 1
            """,
            (conversation_id, mark_type),
        ).fetchone()
        return row is not None

    def list_marked_conversation_ids(
        self,
        mark_type: str | None = None,
        *,
        limit: int = 100,
    ) -> tuple[str, ...]:
        mark_type = self._mark_type(mark_type)
        safe_limit = max(1, int(limit))
        rows = self.db.get_connection().execute(
            """
            SELECT conversation_id
              FROM conversation_local_marks
             WHERE mark_type = ?
             ORDER BY updated_at DESC, conversation_id ASC
             LIMIT ?
            """,
            (mark_type, safe_limit),
        ).fetchall()
        return tuple(str(row["conversation_id"]) for row in rows)
```

- [ ] **Step 5: Export and wire the service**

In `tldw_chatbook/Chat/__init__.py`, import and add `ConversationLocalMarksService` and `ConversationLocalMark` to `__all__`.

In `tldw_chatbook/app.py`, import `ConversationLocalMarksService` and update `_wire_chat_conversation_services`:

```python
self.conversation_local_marks_service = (
    ConversationLocalMarksService(self.chachanotes_db)
    if getattr(self, "chachanotes_db", None) is not None
    else None
)
```

Place this next to `self.local_chat_conversation_service` because it has the same local DB ownership.

- [ ] **Step 6: Run focused tests**

Run:

```bash
pytest Tests/Chat/test_conversation_local_marks_service.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 1**

Run:

```bash
git add tldw_chatbook/DB/ChaChaNotes_DB.py tldw_chatbook/Chat/conversation_local_marks_service.py tldw_chatbook/Chat/__init__.py tldw_chatbook/app.py Tests/Chat/test_conversation_local_marks_service.py
git commit -m "feat: add local conversation marks service"
```

---

### Task 2: Pure Grouped Browser Display State

**Files:**
- Create: `tldw_chatbook/Workspaces/conversation_browser_state.py`
- Modify: `tldw_chatbook/Workspaces/display_state.py`
- Modify: `tldw_chatbook/Workspaces/__init__.py`
- Test: `Tests/Workspaces/test_console_conversation_browser_state.py`

- [ ] **Step 1: Write failing pure tests**

Create `Tests/Workspaces/test_console_conversation_browser_state.py` with tests covering:

```python
from __future__ import annotations

from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.models import DEFAULT_WORKSPACE_ID


def _row(
    key,
    title,
    *,
    scope_type="workspace",
    workspace_id="ws-a",
    workspace_label="Workspace A",
    starred=False,
    selected=False,
    source_kind="persisted",
):
    return ConsoleConversationBrowserInputRow(
        row_key=key,
        conversation_id=None if key.startswith("native:") else key,
        native_session_id=key.removeprefix("native:") if key.startswith("native:") else None,
        title=title,
        scope_type=scope_type,
        workspace_id=workspace_id,
        workspace_label=workspace_label,
        status="active" if selected else "workspace-thread",
        updated_label="1d",
        selected=selected,
        starred=starred,
        star_enabled=not key.startswith("native:"),
        source_kind=source_kind,
    )


def test_browser_groups_starred_workspaces_and_chats():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Workspace chat", starred=True),
            _row("conv-b", "Global chat", scope_type="global", workspace_id=None, workspace_label="Chats"),
            _row("conv-c", "Default chat", workspace_id=DEFAULT_WORKSPACE_ID, workspace_label="Default"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={},
        query="",
    )

    assert [section.section_id for section in state.sections] == ["starred", "workspaces", "chats"]
    assert state.sections[0].rows[0].row_key == "conv-a"
    assert state.sections[1].groups[0].group_id == "workspace:ws-a"
    assert [row.row_key for row in state.sections[2].rows] == ["conv-b", "conv-c"]
    assert state.sections[2].rows[0].scope_type == "global"
    assert state.sections[2].rows[1].workspace_id == DEFAULT_WORKSPACE_ID


def test_search_exposes_matching_rows_from_collapsed_groups_without_changing_preference():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Needle", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={"workspace:ws-b": True},
        query="needle",
    )

    workspace_section = next(section for section in state.sections if section.section_id == "workspaces")
    group = next(group for group in workspace_section.groups if group.group_id == "workspace:ws-b")
    assert group.collapsed is False
    assert group.preference_collapsed is True
    assert [row.title for row in group.rows] == ["Needle"]


def test_explicitly_expanded_inactive_workspace_group_is_remembered():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Alpha", workspace_id="ws-a", workspace_label="Workspace A"),
            _row("conv-b", "Beta", workspace_id="ws-b", workspace_label="Workspace B"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={"workspace:ws-b": False},
        query="",
    )

    workspace_section = next(section for section in state.sections if section.section_id == "workspaces")
    group = next(group for group in workspace_section.groups if group.group_id == "workspace:ws-b")
    assert group.collapsed is False
    assert group.preference_collapsed is False
    assert [row.title for row in group.rows] == ["Beta"]


def test_dedupe_is_within_normal_group_not_across_starred_section():
    state = build_console_conversation_browser_state(
        rows=(
            _row("conv-a", "Canonical", starred=True, source_kind="native"),
            _row("conv-a", "Duplicate", starred=True, source_kind="persisted"),
        ),
        active_workspace_id="ws-a",
        group_collapse_preferences={},
        query="",
    )

    starred = next(section for section in state.sections if section.section_id == "starred")
    workspaces = next(section for section in state.sections if section.section_id == "workspaces")
    assert [row.row_key for row in starred.rows] == ["conv-a"]
    assert [row.row_key for row in workspaces.groups[0].rows] == ["conv-a"]
```

Also add tests for:
- active workspace group is expanded by default
- inactive workspace groups are collapsed by default
- explicitly expanded inactive workspace groups stay expanded after refresh
- `Starred` is expanded by default
- `Chats` is expanded when it has rows
- native rows have `star_enabled=False`
- titles are plain strings and do not render markup control data
- capped groups expose hidden-count/status copy

- [ ] **Step 2: Run pure tests and verify they fail**

Run:

```bash
pytest Tests/Workspaces/test_console_conversation_browser_state.py -q
```

Expected: fail because `conversation_browser_state.py` does not exist.

- [ ] **Step 3: Implement browser state dataclasses**

Create `tldw_chatbook/Workspaces/conversation_browser_state.py` with these public constants and dataclasses:

```python
CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT = 75
CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT = 12

@dataclass(frozen=True)
class ConsoleConversationBrowserInputRow:
    row_key: str
    conversation_id: str | None
    native_session_id: str | None
    title: str
    scope_type: str
    workspace_id: str | None
    workspace_label: str
    status: str = ""
    updated_label: str = ""
    selected: bool = False
    starred: bool = False
    star_enabled: bool = True
    source_kind: str = "persisted"
    starred_sort: str = ""
    updated_sort: str = ""


@dataclass(frozen=True)
class ConsoleConversationBrowserRow:
    row_key: str
    conversation_id: str | None
    native_session_id: str | None
    title: str
    scope_type: str
    workspace_id: str | None
    workspace_label: str
    status: str
    updated_label: str
    selected: bool = False
    starred: bool = False
    star_enabled: bool = True
    source_kind: str = "persisted"


@dataclass(frozen=True)
class ConsoleConversationBrowserGroup:
    group_id: str
    label: str
    collapsed: bool
    rows: tuple[ConsoleConversationBrowserRow, ...]
    count: int
    hidden_count: int = 0
    preference_collapsed: bool = False
    empty_copy: str = ""


@dataclass(frozen=True)
class ConsoleConversationBrowserSection:
    section_id: str
    label: str
    collapsed: bool
    rows: tuple[ConsoleConversationBrowserRow, ...] = ()
    groups: tuple[ConsoleConversationBrowserGroup, ...] = ()
    count: int = 0
    hidden_count: int = 0
    empty_copy: str = ""


@dataclass(frozen=True)
class ConsoleConversationBrowserState:
    query: str
    sections: tuple[ConsoleConversationBrowserSection, ...]
    selected_summary: str
    status_copy: str = ""
    error_copy: str = ""
    marks_available: bool = True
    result_total_count: int | None = None
    result_limit: int = CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT
```

- [ ] **Step 4: Implement the pure builder**

Implement:

```python
def build_console_conversation_browser_state(
    *,
    rows: Iterable[ConsoleConversationBrowserInputRow],
    active_workspace_id: str | None,
    group_collapse_preferences: Mapping[str, bool] | None = None,
    query: str = "",
    marks_available: bool = True,
    error_copy: str = "",
    result_total_count: int | None = None,
    result_limit: int = CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT,
    group_row_limit: int = CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
) -> ConsoleConversationBrowserState:
    ...
```

Builder rules:
- Normalize `query` with `strip().lower()`.
- Match query against row title, workspace label, status, and scope copy.
- Deduplicate rows within each final normal group by `row_key`; first source wins.
- Do not dedupe a starred row out of its normal section. It can appear in `Starred` and `Workspaces` or `Chats`.
- `DEFAULT_WORKSPACE_ID` rows go to `Chats` but keep `scope_type="workspace"` and `workspace_id=DEFAULT_WORKSPACE_ID`.
- `scope_type="global"` rows go to `Chats` and keep `workspace_id=None`.
- Non-default workspace rows go under `Workspaces`, nested by `workspace_id`.
- Starred rows go under `Starred` only when `conversation_id` is not empty.
- Sort Starred by `starred_sort` descending, then `updated_sort` descending, then title.
- Sort workspace groups by active workspace first, then group latest `updated_sort` descending, then label.
- Sort rows selected first, then `updated_sort` descending, then title.
- Collapse defaults:
  - Use `group_collapse_preferences` as a tri-state map: missing key means use default; `True` means collapsed; `False` means expanded.
  - `Starred`: default expanded; key `"section:starred"` can override.
  - active workspace group: default expanded; key `"workspace:<id>"` can override.
  - inactive workspace groups: default collapsed; key `"workspace:<id>"` can explicitly expand or collapse.
  - `Chats`: default expanded when it has rows; key `"section:chats"` can override.
- When query is active, matching sections/groups render expanded but preserve `preference_collapsed` as the resolved saved/default collapsed value.
- Apply `group_row_limit` to visible rows per group or section and store `hidden_count`.
- Build `selected_summary` from the selected row, preferring `"{title} - {workspace_label}"`.
- Build `status_copy` as `"N matches"` for query results, plus `"Showing X of Y"` when capped. Use `result_total_count` when supplied; otherwise use the filtered row count.

- [ ] **Step 5: Attach the browser state to workspace context state**

In `tldw_chatbook/Workspaces/display_state.py`, import `ConsoleConversationBrowserState` under `TYPE_CHECKING` or directly if no cycle exists, and add this field to `ConsoleWorkspaceContextState`:

```python
conversation_browser: ConsoleConversationBrowserState | None = None
```

Keep `conversation_section` unchanged for now.

In `tldw_chatbook/Workspaces/__init__.py`, export:
- `CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT`
- `CONSOLE_CONVERSATION_BROWSER_RESULT_LIMIT`
- `ConsoleConversationBrowserGroup`
- `ConsoleConversationBrowserInputRow`
- `ConsoleConversationBrowserRow`
- `ConsoleConversationBrowserSection`
- `ConsoleConversationBrowserState`
- `build_console_conversation_browser_state`

- [ ] **Step 6: Run pure tests**

Run:

```bash
pytest Tests/Workspaces/test_console_conversation_browser_state.py -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add tldw_chatbook/Workspaces/conversation_browser_state.py tldw_chatbook/Workspaces/display_state.py tldw_chatbook/Workspaces/__init__.py Tests/Workspaces/test_console_conversation_browser_state.py
git commit -m "feat: add console conversation browser state"
```

---

### Task 3: Rail Rendering For Grouped Browser State

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_workspace_context_rail.py`

- [ ] **Step 1: Write failing widget rendering tests**

Add tests to `Tests/UI/test_console_workspace_context_rail.py` that build `ConsoleWorkspaceContextState(conversation_browser=...)` directly.

Required tests:
- `test_console_workspace_context_renders_grouped_conversation_browser`
- `test_console_workspace_context_keeps_status_rows_below_grouped_browser`
- `test_console_workspace_context_renders_disabled_star_for_unpersisted_native_session`
- `test_console_workspace_context_disables_star_controls_when_marks_unavailable`
- `test_console_workspace_context_search_controls_keep_stable_ids`
- `test_console_workspace_context_grouped_browser_styles_are_declared`

Assertions to include:

```python
assert _static_plain(console, "#console-conversation-browser-starred-title") == "Starred"
assert "Workspaces" in _visible_text(console)
assert "Chats" in _visible_text(console)
assert len(console.query("#console-workspace-conversation-search")) == 1
assert len(console.query("#console-workspace-conversations")) == 1
assert len(console.query(".console-conversation-star")) >= 1
assert len(console.query(".console-workspace-conversation-row")) >= 1
assert _static_plain(console, "#console-workspace-sync-label") == "Sync"
```

For marks unavailable, build a browser state with `marks_available=False`, assert rows still render, star buttons are disabled, and scoped warning copy such as `"Local stars unavailable"` is visible inside the rail.

- [ ] **Step 2: Run widget tests and verify they fail**

Run:

```bash
pytest Tests/UI/test_console_workspace_context_rail.py -k "grouped_conversation_browser or grouped_browser_styles or disabled_star or search_controls_keep_stable_ids" -q
```

Expected: fail because the tray still renders only `conversation_section`.

- [ ] **Step 3: Add grouped render branch**

In `ConsoleWorkspaceContextTray.compose()`:

1. Keep the heading, active workspace selector, change workspace button, and recovery copy unchanged.
2. Before the legacy `section = self._conversation_section()` block, branch:

```python
browser = self.state.conversation_browser
if browser is not None:
    yield from self._compose_conversation_browser(browser)
else:
    yield from self._compose_legacy_conversation_section()
```

3. Extract the existing legacy conversation-section render into `_compose_legacy_conversation_section()`.
4. Add `_compose_conversation_browser(browser)` that renders:
   - Header row with title `Conversations`.
   - Selected summary.
   - Existing search input id `console-workspace-conversation-search`.
   - Existing clear button id `console-workspace-conversation-search-clear`.
   - Existing scroll container id `console-workspace-conversations`.
   - Sections in state order.
   - Workspace groups under `Workspaces`.
   - Row button plus star button.

Use stable IDs:
- section title: `console-conversation-browser-{section_id}-title`
- section toggle: `console-conversation-browser-section-toggle-{section_id}`
- workspace group toggle: `console-conversation-browser-group-toggle-{index}`
- row button: `console-workspace-conversation-{index}`
- star button: `console-conversation-star-{index}`

The workspace group toggle id may remain index-based for Textual id safety, but the button must carry the stable group id as an attribute so sorting/filtering does not break persisted preferences:

```python
group_toggle.group_id = group.group_id
```

Use widget attributes:

```python
section_toggle.group_id = f"section:{section.section_id}"

row_button.row_key = row.row_key
row_button.conversation_id = row.conversation_id or row.row_key
row_button.native_session_id = row.native_session_id
row_button.scope_type = row.scope_type
row_button.workspace_id = row.workspace_id

star_button.row_key = row.row_key
star_button.conversation_id = row.conversation_id
star_button.starred = row.starred
```

Star labels:
- starred: `"*"` with tooltip `"Unstar conversation"`
- unstarred: `"☆"` is non-ASCII, so use `"+"` only if repo UI is ASCII-only. Prefer `"*"` for starred and `"."` for unstarred if avoiding Unicode. Tooltips must disambiguate.
- disabled native rows: disabled star button with tooltip `"Send or save this conversation before starring."`

Keep row title rendering through `_conversation_title()` and `_conversation_visible_title()` to avoid markup interpretation and long-title overflow.

- [ ] **Step 4: Add CSS for grouped browser**

In both TCSS files, add selectors:

```css
.console-conversation-browser-section-header {
    width: 100%;
    min-width: 0;
    height: 1;
    min-height: 1;
    layout: horizontal;
    align: left middle;
}

.console-conversation-browser-section-title {
    width: 1fr;
    min-width: 0;
}

.console-conversation-browser-group-header {
    width: 100%;
    min-width: 0;
    height: 1;
    min-height: 1;
    layout: horizontal;
    align: left middle;
}

.console-conversation-browser-group-title {
    width: 1fr;
    min-width: 0;
    color: $ds-text-muted;
}

.console-conversation-browser-row-line {
    width: 100%;
    min-width: 0;
    layout: horizontal;
}

.console-conversation-star {
    width: 3;
    min-width: 3;
    max-width: 3;
    height: 2;
    min-height: 2;
    margin: 0 1 1 0;
}
```

Keep `#console-workspace-conversations` bounded with `overflow-y: auto`.

- [ ] **Step 5: Run widget tests**

Run:

```bash
pytest Tests/UI/test_console_workspace_context_rail.py -q
```

Expected: pass.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_context.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_workspace_context_rail.py
git commit -m "feat: render grouped console conversation browser"
```

---

### Task 4: ChatScreen Browser Assembly, Search, Collapse, And Stars

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing mounted integration tests**

Add tests to `Tests/UI/test_console_native_chat_flow.py`:

- `test_console_conversation_browser_lists_all_workspace_groups`
- `test_console_conversation_browser_search_filters_all_groups`
- `test_console_conversation_browser_search_ignores_stale_results`
- `test_console_conversation_browser_group_collapse_persists_locally`
- `test_console_conversation_browser_starred_section_updates_from_row_action`
- `test_console_conversation_browser_keeps_starred_row_in_normal_group`
- `test_console_conversation_browser_marks_unavailable_keeps_browsing_enabled`
- `test_console_conversation_browser_long_list_keeps_readiness_rows_reachable`

Key setup:
- Use `_build_test_app()` and `ConsoleHarness(app)`.
- Create `ws-a`, `ws-b`, and active workspace `ws-a`.
- Link workspace memberships via `service.link_membership(...)`.
- Add native sessions through `store.ensure_session(title=..., workspace_id=...)`.
- For global persisted rows, install a fake local conversation service or scope service returning `scope_type="global"`.
- For stars, set `app.conversation_local_marks_service` to either the real service from Task 1 or a small fake with `star_conversation`, `unstar_conversation`, `is_starred`, and `list_marked_conversation_ids`.

Expected assertions:

```python
assert "Starred" in _visible_text(console)
assert "Workspaces" in _visible_text(console)
assert "Workspace A" in _visible_text(console)
assert "Workspace B" in _visible_text(console)
assert "Chats" in _visible_text(console)
assert "Global chat" in _visible_text(console)
assert "Storage" in _visible_text(console)
assert "Server handoff" in _visible_text(console)
```

For search:

```python
search = console.query_one("#console-workspace-conversation-search", Input)
search.focus()
await pilot.type("needle")
await pilot.pause(0.3)
assert "Needle in Workspace B" in _visible_text(console)
assert "Workspace B" in _visible_text(console)
```

- [ ] **Step 2: Run mounted tests and verify they fail**

Run:

```bash
pytest Tests/UI/test_console_native_chat_flow.py -k "conversation_browser" -q
```

Expected: fail because `ChatScreen` has not attached `conversation_browser`.

- [ ] **Step 3: Add browser state fields**

In `ChatScreen.__init__`, add these browser-wide fields and update all grouped-browser code to read from them instead of the old active-workspace-only `_console_workspace_conversation_*` search/cache fields:

```python
self._console_conversation_browser_query = ""
self._console_conversation_browser_search_timer: Any | None = None
self._console_conversation_browser_search_token = 0
self._console_conversation_browser_rows: tuple[ConsoleConversationBrowserInputRow, ...] = ()
self._console_conversation_browser_total: int | None = None
self._console_conversation_browser_error = ""
```

The existing DOM ids can stay `console-workspace-*`, but Python state for the grouped browser should use the new `_console_conversation_browser_*` names.

- [ ] **Step 4: Add row-source helpers**

Add these helpers to `ChatScreen`:

```python
def _console_browser_workspace_records(self) -> tuple[WorkspaceRecord, ...]:
    ...

def _console_browser_workspace_labels(self) -> dict[str, str]:
    ...

def _native_console_browser_rows(self) -> list[ConsoleConversationBrowserInputRow]:
    ...

def _membership_console_browser_rows(self) -> list[ConsoleConversationBrowserInputRow]:
    ...

def _persisted_console_browser_rows(self, query: str = "") -> tuple[list[ConsoleConversationBrowserInputRow], int | None, str]:
    ...

def _starred_console_conversation_ids(self) -> set[str]:
    ...

def _merge_console_browser_rows(self, *row_groups: Iterable[ConsoleConversationBrowserInputRow]) -> tuple[ConsoleConversationBrowserInputRow, ...]:
    ...
```

Implementation details:
- `workspace_records` comes from `workspace_registry_service.list_workspaces()`.
- Always include `DEFAULT_WORKSPACE_ID` if the registry has it or can create it through `ensure_default_workspace()`.
- Native rows come from every `ConsoleChatStore` session, not just the active workspace.
- Membership rows iterate `list_workspace_conversations(workspace_id)` for every workspace.
- Persisted rows use `self.app_instance.local_chat_conversation_service.list_conversations(...)` synchronously for:
  - global conversations: `scope_type="global", workspace_id=None`
  - each workspace: `scope_type="workspace", workspace_id=workspace_id`
- Use a conservative per-scope limit such as `25` and let the pure builder cap visible rows.
- Mark `star_enabled=False` when `conversation_id is None` or row key starts with `native:`.
- Apply star state from `ConversationLocalMarksService.list_marked_conversation_ids()`.
- Dedupe precedence: native rows first, membership rows second, persisted rows third.
- Default workspace rows use `workspace_label="Chats"` for display grouping but keep `workspace_id=DEFAULT_WORKSPACE_ID`.

- [ ] **Step 5: Attach browser state instead of active-workspace section**

Replace `_with_console_workspace_conversation_section(state)` usage in `_build_console_workspace_context_state()` with `_with_console_conversation_browser_state(state)`.

Add:

```python
def _with_console_conversation_browser_state(
    self,
    state: ConsoleWorkspaceContextState,
) -> ConsoleWorkspaceContextState:
    query = self._console_conversation_browser_query
    rows, total, error_copy = self._current_console_browser_rows(query)
    browser = build_console_conversation_browser_state(
        rows=rows,
        active_workspace_id=self._current_console_workspace_context().active_workspace_id,
        group_collapse_preferences=self._console_conversation_browser_collapse_preferences(),
        query=query,
        marks_available=getattr(self.app_instance, "conversation_local_marks_service", None) is not None,
        error_copy=error_copy or self._console_conversation_browser_error,
        result_total_count=total,
    )
    return replace(state, conversation_browser=browser)
```

- [ ] **Step 6: Update search handler**

Keep selector handler:

```python
@on(Input.Changed, "#console-workspace-conversation-search")
def on_console_workspace_conversation_search_changed(self, event: Input.Changed) -> None:
```

Change behavior:
- Store `event.value` in `_console_conversation_browser_query`.
- Increment `_console_conversation_browser_search_token`.
- Stop existing timer.
- Immediately rebuild from native and membership rows for fast feedback.
- Debounce persisted search with the existing worker group string `console-workspace-conversation-search`.
- Stale checks compare token and query only, not active workspace id.
- Clearing search restores group collapse preferences because the pure builder receives the same collapse preference mapping and empty query.

Add/replace async refresh helpers:

```python
async def _refresh_console_conversation_browser_search(self, query: str, token: int) -> None:
    ...

async def _refresh_console_conversation_browser_after_selection(self) -> None:
    ...
```

- [ ] **Step 7: Add collapse preference helpers**

Store preferences under `app_config["console"]["conversation_browser"]`:

```python
{
    "collapsed_groups": {
        "section:starred": False,
        "workspace:ws-a": False,
        "workspace:ws-b": True,
        "section:chats": False,
    }
}
```

Add:

```python
def _console_conversation_browser_config(self) -> dict[str, Any]:
    ...

def _console_conversation_browser_collapse_preferences(self) -> dict[str, bool]:
    ...

def _set_console_conversation_browser_group_collapsed(self, group_id: str, collapsed: bool) -> None:
    ...
```

`_console_conversation_browser_collapse_preferences()` must return the full mapping, not only collapsed ids, so explicit `False` entries can represent inactive workspace groups the user expanded.

Keep old `_console_workspace_conversations_collapsed()` only for legacy fallback or remove after tests are updated.

- [ ] **Step 8: Add button handling for group toggles and stars**

In `on_button_pressed`, add branches before the row-selection branch:

```python
if button_id and button_id.startswith("console-conversation-browser-section-toggle-"):
    ...

if button_id and button_id.startswith("console-conversation-browser-group-toggle-"):
    ...

if button_id and button_id.startswith("console-conversation-star-"):
    ...
```

Star handling:
- For section/group toggles, read `group_id` from `event.button.group_id` and persist that exact key.
- Read `conversation_id` from the star button.
- If missing, notify warning and do not crash.
- If current star state is true, call `unstar_conversation`.
- Otherwise call `star_conversation`.
- Rebuild browser state.
- On storage error, show warning and keep browsing usable.

- [ ] **Step 9: Run mounted browser tests**

Run:

```bash
pytest Tests/UI/test_console_native_chat_flow.py -k "conversation_browser" -q
```

Expected: pass.

- [ ] **Step 10: Commit Task 4**

Run:

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat: assemble console conversation browser"
```

---

### Task 5: Cross-Workspace Row Selection And Resume Semantics

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_native_chat_flow.py`

- [ ] **Step 1: Write failing row-selection matrix tests**

Add tests to `Tests/UI/test_console_native_chat_flow.py`:

- `test_console_browser_selecting_non_default_workspace_native_session_switches_active_workspace`
- `test_console_browser_selecting_non_default_workspace_persisted_row_switches_active_workspace_before_resume`
- `test_console_browser_selecting_default_native_session_switches_to_default_and_keeps_file_tools_disabled`
- `test_console_browser_selecting_default_persisted_row_switches_to_default_and_keeps_file_tools_disabled`
- `test_console_browser_selecting_global_persisted_row_preserves_active_workspace`

Each test must assert both active workspace and transcript/session outcome.

Example assertions:

```python
active = app.workspace_registry_service.get_active_workspace()
assert active is not None
assert active.workspace_id == "ws-b"
assert "Workspace B prompt" in _visible_text(console)
```

Default safety assertions:

```python
active = app.workspace_registry_service.get_active_workspace()
assert active is not None
assert active.workspace_id == DEFAULT_WORKSPACE_ID
assert "File tools: Off in Default workspace" in _visible_text(console)
```

Global row assertion:

```python
before = app.workspace_registry_service.get_active_workspace().workspace_id
await _click_console_workspace_conversation_for_id(console, pilot, "global-conv")
after = app.workspace_registry_service.get_active_workspace().workspace_id
assert after == before
```

- [ ] **Step 2: Run matrix tests and verify they fail**

Run:

```bash
pytest Tests/UI/test_console_native_chat_flow.py -k "selecting_non_default_workspace or selecting_default_ or selecting_global_persisted" -q
```

Expected: fail because row selection still assumes active workspace or switches after persisted load.

- [ ] **Step 3: Add row lookup and workspace activation helpers**

Add:

```python
def _find_console_browser_row(self, row_key: str) -> ConsoleConversationBrowserRow | None:
    ...

def _activate_console_workspace_for_browser_row(self, row: ConsoleConversationBrowserRow) -> None:
    ...
```

Rules for `_activate_console_workspace_for_browser_row`:
- If `row.scope_type == "global"`, do nothing.
- If `row.workspace_id` is empty, do nothing.
- If `row.workspace_id == CONSOLE_GLOBAL_WORKSPACE_ID`, do nothing.
- If `row.workspace_id == DEFAULT_WORKSPACE_ID`, call `workspace_registry_service.set_active_workspace(DEFAULT_WORKSPACE_ID)` and refresh the Console workspace context; do not enable file/runtime tools.
- For any other workspace id, call `workspace_registry_service.set_active_workspace(row.workspace_id)`.
- After changing active workspace, update `ConsoleChatStore.workspace_context` from `_current_console_workspace_context()`.

- [ ] **Step 4: Update persisted resume to accept target scope**

Change:

```python
async def _resume_console_workspace_conversation(self, conversation_id: str) -> bool:
```

to:

```python
async def _resume_console_workspace_conversation(
    self,
    conversation_id: str,
    *,
    target_scope_type: str | None = None,
    target_workspace_id: str | None = None,
) -> bool:
```

Use `target_workspace_id` when present for the restored session workspace. Fall back to persisted metadata, then active workspace. Update user-facing warnings from `"workspace conversation"` to `"saved conversation"` where the row may be global.

- [ ] **Step 5: Replace row-selection button branch**

In the `if button_id and button_id.startswith("console-workspace-conversation-")` branch:

1. Read `row_key` from `event.button.row_key`; fall back to `conversation_id` for legacy rows.
2. Find the current browser row by row key.
3. Activate workspace for the row before opening anything.
4. If `row.native_session_id` exists or `_console_session_id_for_workspace_conversation(row.row_key)` returns a session id:
   - switch native session
   - sync UI
   - focus composer
5. Else if `row.conversation_id` exists:
   - call `_resume_console_workspace_conversation(row.conversation_id, target_scope_type=row.scope_type, target_workspace_id=row.workspace_id)`
6. Else warn that the native row is unavailable.
7. Refresh browser search/results after selection.

- [ ] **Step 6: Run row-selection tests**

Run:

```bash
pytest Tests/UI/test_console_native_chat_flow.py -k "selecting_non_default_workspace or selecting_default_ or selecting_global_persisted" -q
```

Expected: pass.

- [ ] **Step 7: Commit Task 5**

Run:

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_native_chat_flow.py
git commit -m "fix: route console browser resume by row scope"
```

---

### Task 6: Sync Exclusion, Regression Sweep, And Visual Evidence

**Files:**
- Modify: `Tests/Chat/test_chat_conversation_scope_service.py`
- Modify: `Tests/Chat/test_server_chat_conversation_service.py`
- Add: `Docs/superpowers/qa/console-grouped-conversation-browser/2026-06-27-console-grouped-conversation-browser.md`
- Add: `Docs/superpowers/qa/console-grouped-conversation-browser/*.png`
- Modify: existing tests only if regressions reveal stale assumptions

- [x] **Step 1: Add explicit mirror and server payload exclusion regressions**

In `Tests/Chat/test_chat_conversation_scope_service.py`, add a test that verifies `record_sync_mirror_report()` passes only explicit records and does not consult marks:

```python
def test_scope_service_chat_metadata_mirror_report_does_not_add_local_marks():
    sync_scope = FakeSyncScopeService()
    service = ChatConversationScopeService(
        local_service=FakeConversationService(),
        server_service=FakeServerConversationService(),
        sync_scope_service=sync_scope,
    )

    service.record_sync_mirror_report(
        mode="server",
        server_profile_id="server-a",
        authenticated_principal_id="user-a",
        workspace_scope="workspace-1",
        local_records=[{"id": "conv-1", "title": "Plain"}],
        remote_records=[],
    )

    local_records = sync_scope.calls[0]["local_records"]
    assert local_records == [{"id": "conv-1", "title": "Plain"}]
    assert "starred" not in local_records[0]
    assert "marks" not in local_records[0]
```

In `Tests/Chat/test_server_chat_conversation_service.py`, add a test that verifies server create payloads do not accept or forward local marks:

```python
@pytest.mark.asyncio
async def test_server_chat_conversation_create_payload_does_not_forward_local_marks():
    client = FakeChatClient()
    service = ServerChatConversationService(client=client)

    await service.create_conversation(
        title="Server Plain",
        character_id=1,
        starred=True,
        local_marks={"starred": True},
        marks=["starred"],
    )

    call = next(call for call in client.calls if call[0] == "create_character_chat_session")
    request_data = call[1][0]
    payload = request_data.model_dump(exclude_none=True, mode="json")

    assert payload["title"] == "Server Plain"
    assert "starred" not in payload
    assert "local_marks" not in payload
    assert "marks" not in payload
```

If this fails because the current server schema rejects unknown fields before forwarding, preserve that behavior and assert the raised validation error mentions the unsupported local mark keys. The important boundary is that local marks never become server payload fields.

Add the equivalent update-path check:

```python
@pytest.mark.asyncio
async def test_server_chat_conversation_update_payload_does_not_forward_local_marks():
    client = FakeChatClient()
    service = ServerChatConversationService(client=client)

    await service.update_conversation(
        "conv-1",
        {
            "version": 3,
            "state": "resolved",
            "starred": True,
            "local_marks": {"starred": True},
            "marks": ["starred"],
        },
    )

    call = next(call for call in client.calls if call[0] == "update_chat_conversation")
    request_data = call[1][1]
    payload = request_data.model_dump(exclude_none=True, mode="json")

    assert payload == {"version": 3, "state": "resolved"}
    assert "starred" not in payload
    assert "local_marks" not in payload
    assert "marks" not in payload
```

This complements the Task 1 `sync_log` test.

- [x] **Step 2: Run focused regression suite**

Run:

```bash
pytest Tests/Chat/test_conversation_local_marks_service.py Tests/Workspaces/test_console_conversation_browser_state.py -q
pytest Tests/UI/test_console_workspace_context_rail.py -q
pytest Tests/UI/test_console_native_chat_flow.py -k "conversation_browser or workspace_conversation" -q
pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_conversation_service.py Tests/Chat/test_chat_conversation_scope_service.py Tests/Chat/test_server_chat_conversation_service.py -q
```

Expected: all pass.

- [x] **Step 3: Run broader safety tests**

Run:

```bash
pytest Tests/Chat Tests/Workspaces Tests/UI/test_console_workspace_context_rail.py Tests/UI/test_console_native_chat_flow.py -q
```

Expected: all pass. If unrelated pre-existing failures appear, document exact test names, command output summaries, and why they are unrelated before continuing.

- [x] **Step 4: Capture rendered visual evidence**

Follow `Docs/superpowers/handoffs/2026-05-08-ui-screenshot-approval-workflow-handoff.md`.

Capture actual rendered PNG evidence for:
- default grouped browser
- active workspace group expanded
- search result matching a collapsed inactive workspace
- starred quick-access section
- long grouped list with lower readiness/handoff rows reachable

Archive evidence under:

```text
Docs/superpowers/qa/console-grouped-conversation-browser/
```

If using textual-web, start the app with the repo's web serve path and capture the browser state. If textual-web is unavailable, use a running Textual app screenshot. Do not treat ASCII snapshots as visual approval.

- [x] **Step 5: Write QA evidence note**

Create `Docs/superpowers/qa/console-grouped-conversation-browser/2026-06-27-console-grouped-conversation-browser.md`:

```markdown
# Console Grouped Conversation Browser QA

Date: 2026-06-27

## Verification Commands

- `pytest Tests/Chat/test_conversation_local_marks_service.py Tests/Workspaces/test_console_conversation_browser_state.py -q`
- `pytest Tests/UI/test_console_workspace_context_rail.py -q`
- `pytest Tests/UI/test_console_native_chat_flow.py -k "conversation_browser or workspace_conversation" -q`
- `pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_conversation_service.py Tests/Chat/test_chat_conversation_scope_service.py Tests/Chat/test_server_chat_conversation_service.py -q`

## Screenshot Evidence

- Default grouped browser: `Docs/superpowers/qa/console-grouped-conversation-browser/default-grouped-browser.png`
- Active workspace expanded: `Docs/superpowers/qa/console-grouped-conversation-browser/active-workspace-expanded.png`
- Search across collapsed groups: `Docs/superpowers/qa/console-grouped-conversation-browser/search-collapsed-group-match.png`
- Starred section: `Docs/superpowers/qa/console-grouped-conversation-browser/starred-section.png`
- Long list bounded with readiness rows: `Docs/superpowers/qa/console-grouped-conversation-browser/long-list-readiness-reachable.png`

## Notes

- Stars are stored in `conversation_local_marks`.
- Stars are local-only and do not create `sync_log` rows.
- Default workspace conversations render under Chats but keep Default workspace scope.
```

Use those filenames unless the capture tooling requires different names; if so, record the actual PNG paths.

- [x] **Step 6: Self-review implementation**

Check:

```bash
git diff --stat
git diff -- tldw_chatbook/Chat/conversation_local_marks_service.py tldw_chatbook/Workspaces/conversation_browser_state.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Widgets/Console/console_workspace_context.py
```

Review for:
- accidental `starred` field in `conversations`
- sync trigger additions for marks
- server payload/mirror additions for marks
- active workspace changes on global row selection
- Default workspace file-tool readiness changing incorrectly
- browser overflow hiding status/readiness rows
- stale search token checks tied to active workspace

- [x] **Step 7: Commit Task 6**

Run:

```bash
git add Tests/Chat/test_chat_conversation_scope_service.py Tests/Chat/test_server_chat_conversation_service.py Docs/superpowers/qa/console-grouped-conversation-browser/2026-06-27-console-grouped-conversation-browser.md Docs/superpowers/qa/console-grouped-conversation-browser/*.png
git commit -m "test: verify console browser sync boundaries"
```

---

## Final Verification

Run before claiming implementation complete:

```bash
pytest Tests/Chat/test_conversation_local_marks_service.py Tests/Workspaces/test_console_conversation_browser_state.py -q
pytest Tests/UI/test_console_workspace_context_rail.py -q
pytest Tests/UI/test_console_native_chat_flow.py -k "conversation_browser or workspace_conversation" -q
pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_conversation_service.py Tests/Chat/test_chat_conversation_scope_service.py Tests/Chat/test_server_chat_conversation_service.py -q
```

Also run any lint/format command already standard for the branch if one is present in project metadata. If no fast lint command is configured, document that no repo lint command was found.

## Implementation Notes Template

Use this shape in the final task/PR notes:

```markdown
Implemented the grouped Console conversation browser in the `Convos & Workspaces` rail.

- Added local-only `conversation_local_marks` storage and `ConversationLocalMarksService`.
- Added pure grouped browser display state for `Starred`, `Workspaces`, and `Chats`.
- Updated Console rail rendering, search, collapse preferences, star/unstar actions, and row selection.
- Workspace-scoped row selection switches active workspace before resume; global rows preserve active workspace; Default rows preserve Default safety.
- Verified local marks do not create sync-log rows and are not included in conversation metadata, mirror report records, or server conversation payloads.

ADR: `backlog/decisions/010-console-conversation-local-marks.md`
QA evidence: `Docs/superpowers/qa/console-grouped-conversation-browser/2026-06-27-console-grouped-conversation-browser.md`
```
