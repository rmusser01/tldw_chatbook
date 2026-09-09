# Schedules Handoff — PR-0 + PR-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the two phase-0 fixes (reminder toast, visible New button) and the schema-v4 foundation (runs/results tables, model updates, slot keys, DB accessors) for the scheduled-task handoff program.

**Architecture:** PR-0 is two independent surface fixes. PR-1 is the storage floor for local automation execution: a forward-only v3→v4 migration following the house migration pattern (structural self-detection, per-connection applicability), Pydantic models mirroring tldw_server's run/result contracts, a byte-parity slot-key module, and DB accessors with prune-on-write and stale-run reconciliation. No execution, sync, or UI beyond the button lands here — that is PR-2+.

**Tech Stack:** Python ≥3.11, SQLite (WAL, fresh-connection-per-op), Pydantic, pytest, Textual 8.x.

**Spec:** `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` (§4 data model, §6.4/§7 constants referenced here). Read it before starting.

## Global Constraints

- Work in a **worktree off `origin/dev`** (`superpowers:using-git-worktrees`) — concurrent sessions mutate the main checkout. PR-0 and PR-1 are separate branches: `feat/schedules-toast-and-create-button`, `feat/schedules-schema-v4`.
- The checked-out venv is uv-managed with **no pip**: install with `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"` if pytest is missing. Always run tests as `.venv/bin/python -m pytest …`.
- The `timeout` shell command does not exist in this environment.
- **Never broad-pkill pytest** — scope any kill to your worktree.
- Schema version numbers collide across concurrent branches in this repo: **re-verify `_CURRENT_SCHEMA_VERSION` is still 3 on `origin/dev` at merge time**; if another branch claimed 4, renumber (v4→v5) with provenance in the migration docstring.
- Migrations are forward-only and must self-detect applicability structurally (presence of their column/table), never via `get_schema_version()` between steps — `:memory:` databases get a fresh empty DB per connection (see `_initialize_schema` docstring in `scheduled_tasks_db.py`).
- DB tests use a `tmp_path` file DB, never `:memory:` (fresh-connection-per-op makes `:memory:` see an empty DB every call).
- Never hand-edit the CSS bundle; edit source tcss and rebuild (Task 2, Step 4).

---

## PR-0 — branch `feat/schedules-toast-and-create-button`

### Task 1: Reminder toast — pass the app handle into dispatch

The only `dispatch()` caller in the repo that omits `app=` is `ReminderHandler`, so a fired reminder writes an inbox row and shows no toast. Fix with the zero-arg-getter pattern `BriefingJobHandler` already uses (`chachanotes_db_getter` precedent: the handler is constructed before app wiring completes, so it takes a getter resolved fresh per call, never a captured instance).

**Files:**
- Modify: `tldw_chatbook/Scheduling/scheduler/handlers/reminder_handler.py`
- Modify: `tldw_chatbook/app.py` (the `ReminderHandler(` construction site inside `_wire_watchlists_and_notifications_services` — find with `grep -n "ReminderHandler(" tldw_chatbook/app.py`)
- Test: `Tests/Scheduling/test_reminder_handler.py`

**Interfaces:**
- Produces: `ReminderHandler(dispatch_service, app_getter: Callable[[], Any] | None = None)`; `handle()` now calls `dispatch(app=…, category=…, …)`. PR-2's automation handler will reuse this exact getter.

- [ ] **Step 1: Update the four pinning tests and add two new ones**

The existing tests in `Tests/Scheduling/test_reminder_handler.py` assert the dispatch kwargs **without** `app=` — they currently pin the bug. Add `app=None` to every existing `assert_called_once_with(...)` (a handler constructed without a getter passes `app=None`), then add:

```python
@pytest.mark.asyncio
async def test_reminder_handler_passes_app_from_getter():
    app = object()
    service = Mock()
    handler = ReminderHandler(dispatch_service=service, app_getter=lambda: app)
    await handler.handle({"id": "5", "title": "T", "body": "B"})
    assert service.dispatch.call_args.kwargs["app"] is app


@pytest.mark.asyncio
async def test_reminder_handler_tolerates_getter_returning_none():
    service = Mock()
    handler = ReminderHandler(dispatch_service=service, app_getter=lambda: None)
    await handler.handle({"id": "6", "title": "T"})
    assert service.dispatch.call_args.kwargs["app"] is None
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_reminder_handler.py -v`
Expected: the two new tests FAIL (`TypeError: unexpected keyword argument 'app_getter'`); the four updated ones FAIL on the missing `app` kwarg.

- [ ] **Step 3: Implement**

In `reminder_handler.py`:

```python
from typing import Any, Callable


class ReminderHandler:
    """Dispatch a reminder notification for a scheduled task."""

    def __init__(
        self,
        dispatch_service: NotificationDispatchService,
        app_getter: Callable[[], Any] | None = None,
    ) -> None:
        self.dispatch_service = dispatch_service
        #: Zero-arg getter for the running app, resolved fresh per dispatch
        #: (BriefingJobHandler's chachanotes_db_getter discipline): the
        #: handler is constructed before app wiring completes, and
        #: dispatch() only attempts transient toast delivery when given a
        #: live app handle.
        self.app_getter = app_getter

    async def handle(self, task: dict[str, Any]) -> None:
        app = self.app_getter() if self.app_getter is not None else None
        self.dispatch_service.dispatch(
            app=app,
            category="reminder",
            title=task.get("title", "Reminder"),
            message=task.get("body") or "",
            source_entity_kind="scheduled_task",
            source_entity_id=task.get("id"),
        )
```

Keep `__call__` unchanged. In `app.py`, at the `ReminderHandler(` construction site:

```python
"reminder": ReminderHandler(
    dispatch_service=self.notification_dispatch_service,
    app_getter=lambda: self,
),
```

- [ ] **Step 4: Run the suite**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_reminder_handler.py Tests/Scheduling/test_scheduler_loop.py -v`
Expected: all PASS (the loop suite guards against signature regressions at the dispatch seam).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/scheduler/handlers/reminder_handler.py tldw_chatbook/app.py Tests/Scheduling/test_reminder_handler.py
git commit -m "fix(scheduling): reminders show a toast — pass the app handle into dispatch"
```

### Task 2: Visible New button in the Schedule Queue pane

The only create affordance is the `c` key in the footer. Add a primary button beside the "Schedule Queue" pane title wired to the existing `action_create_reminder`.

**Files:**
- Modify: `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (in `compose_content`, the queue list pane — anchor on `id="scheduling-list-title"`, not line numbers; dev has moved +799 lines since the last read)
- Modify: the scheduling feature tcss (locate: `grep -rn "scheduling-list-title" tldw_chatbook/css/`) and the rebuilt bundle
- Test: extend the workbench UI suite (locate: `grep -rl "SchedulesWorkbench" Tests/` — extend the suite that mounts the workbench; only if none mounts it, create `Tests/UI/test_schedules_new_button.py` reusing the harness/fixtures of the nearest workbench suite found by that grep)

**Interfaces:**
- Consumes: `SchedulesWorkbench.action_create_reminder()` (exists; pushes `ReminderForm`).
- Produces: `Button` id `scheduling-new-task` — PR-6's UI polish references it.

- [ ] **Step 1: Write the failing test**

In the located suite, using its existing mount fixture (same harness the suite's other tests use — do not invent a new one):

```python
async def test_new_button_exists_and_opens_create_form(<same fixture args the suite uses>):
    button = workbench.query_one("#scheduling-new-task", Button)
    assert "New" in str(button.label)
    pushed: list = []
    workbench.app.push_screen = lambda screen, callback=None: pushed.append(screen)
    button.press()
    await pilot.pause()
    assert pushed and type(pushed[0]).__name__ == "ReminderForm"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest <located suite> -k new_button -v`
Expected: FAIL — `NoMatches` for `#scheduling-new-task`.

- [ ] **Step 3: Implement**

In `compose_content`, replace the bare title yield in the queue list pane:

```python
with Horizontal(id="scheduling-list-header"):
    yield Static(
        "Schedule Queue",
        id="scheduling-list-title",
        classes="scheduling-column-title",
    )
    yield Button(
        "+ New",
        id="scheduling-new-task",
        variant="primary",
        tooltip="Schedule a new task (c).",
    )
```

Add the handler beside the other `@on(Button.Pressed, …)` handlers:

```python
@on(Button.Pressed, "#scheduling-new-task")
def _on_new_task_pressed(self, event: Button.Pressed) -> None:
    event.stop()
    self.action_create_reminder()
```

In the scheduling tcss source file found in Files:

```css
#scheduling-list-header {
    height: auto;
}
#scheduling-list-header #scheduling-list-title {
    width: 1fr;
}
#scheduling-new-task {
    width: auto;
    min-width: 9;
}
```

- [ ] **Step 4: Rebuild the CSS bundle and verify sync**

Run: `.venv/bin/python tldw_chatbook/css/build_css.py && .venv/bin/python tldw_chatbook/css/check_bundle_sync.py`
Expected: bundle regenerated, sync check passes. Never edit the bundle directly.

- [ ] **Step 5: Run the tests, then verify live**

Run: `.venv/bin/python -m pytest <located suite> -v`
Expected: PASS.
Then launch per `.claude/skills/verify` (tmux recipe, scratch `TLDW_CONFIG_PATH` profile), open Schedules, confirm the button renders beside the pane title and opens the form, and that the layout holds at 120 columns (the compact-workbench width class).

- [ ] **Step 6: Commit and open PR-0**

```bash
git add -A tldw_chatbook/UI/Screens/scheduling/ tldw_chatbook/css/ Tests/
git commit -m "feat(schedules): visible New button in the queue pane (UX F-07)"
```

Open the PR against `dev` covering Tasks 1–2.

---

## PR-1 — branch `feat/schedules-schema-v4` (off `origin/dev`, independent of PR-0)

### Task 3: Migration v3 → v4

**Files:**
- Create: `tldw_chatbook/Scheduling/db/migrations/v3_to_v4.py`
- Modify: `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (`_CURRENT_SCHEMA_VERSION` → 4; `_initialize_schema` imports and calls `migrate_v3_to_v4`; column registries — Step 3)
- Test: `Tests/Scheduling/test_migrations.py`, `Tests/Scheduling/test_schema.py` (extend, following each file's existing per-version test shape)

**Interfaces:**
- Produces: tables `automation_runs`, `automation_results`; new columns on `automation_definitions` (`disabled_lock_kind, disabled_reason, resolution_state, resolved_at, resolved_by, resolved_result_id, finding_policy, retention_policy, next_run_at, transfer_state`) and `reminder_tasks` (`transfer_state`). Tasks 4–7 and all later PRs consume these exact names.

- [ ] **Step 1: Write the failing tests**

Extend `Tests/Scheduling/test_migrations.py` (mirror the v2→v3 tests' shape in that file — tmp_path DB, never `:memory:`). Test-file imports used across Tasks 3/6/7: `from contextlib import closing`, `from datetime import datetime, timezone`, plus `ScheduledTasksDB` — add any that file lacks. The implementation in Task 6 needs `timedelta` added to `scheduled_tasks_db.py`'s datetime import if absent.

```python
def test_v4_creates_runs_and_results_tables(tmp_path):
    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    with closing(db._get_connection()) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    assert {"automation_runs", "automation_results"} <= tables
    assert db.get_schema_version() == 4


def test_v4_adds_definition_and_reminder_columns(tmp_path):
    db = ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")
    with closing(db._get_connection()) as conn:
        def_cols = {r[1] for r in conn.execute("PRAGMA table_info(automation_definitions)")}
        rem_cols = {r[1] for r in conn.execute("PRAGMA table_info(reminder_tasks)")}
    assert {
        "disabled_lock_kind", "disabled_reason", "resolution_state",
        "resolved_at", "resolved_by", "resolved_result_id",
        "finding_policy", "retention_policy", "next_run_at", "transfer_state",
    } <= def_cols
    assert "transfer_state" in rem_cols


def test_v4_preserves_existing_rows_and_is_idempotent(tmp_path):
    path = str(tmp_path / "s.db")
    db = ScheduledTasksDB(path, client_id="t")
    task_id = db.create_reminder_task(
        owner_id="local", title="keep me", schedule_kind="one_time",
        run_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
    )
    from tldw_chatbook.Scheduling.db.migrations.v3_to_v4 import migrate
    migrate(db)  # second application must be a no-op
    row = db.get_reminder_task(task_id)
    assert row is not None and row["title"] == "keep me"
    assert db.get_schema_version() == 4
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_migrations.py -k v4 -v`
Expected: FAIL — no `v3_to_v4` module, no tables, version 3.

- [ ] **Step 3: Implement the migration**

`v3_to_v4.py`, following `v2_to_v3.py` exactly (module docstring, `_MigrationCapableDB` protocol, structural self-detection, forward-only version stamp):

```python
"""Migration from schema version 3 to version 4.

Adds the local automation execution floor for the schedules-handoff
program (spec-2026-08-31-schedules-handoff-parity.md §4): the
``automation_runs`` and ``automation_results`` tables, the eight
reference-parity columns plus ``next_run_at``/``transfer_state`` on
``automation_definitions``, and ``transfer_state`` on ``reminder_tasks``.
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any, Protocol

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only

    class _MigrationCapableDB(Protocol):
        def _get_connection(self) -> Any: ...


_DEFINITION_COLUMNS_V4: tuple[tuple[str, str], ...] = (
    ("disabled_lock_kind", "TEXT"),
    ("disabled_reason", "TEXT"),
    ("resolution_state", "TEXT NOT NULL DEFAULT 'open'"),
    ("resolved_at", "TEXT"),
    ("resolved_by", "TEXT"),
    ("resolved_result_id", "TEXT"),
    ("finding_policy", "TEXT"),
    ("retention_policy", "TEXT"),
    ("next_run_at", "TEXT"),
    ("transfer_state", "TEXT"),
)

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS automation_runs (
    id TEXT PRIMARY KEY,
    server_id TEXT,
    owner_id TEXT NOT NULL,
    definition_id TEXT NOT NULL,
    definition_version INTEGER NOT NULL DEFAULT 1,
    trigger_reason TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'none',
    schedule_slot TEXT,
    scope_snapshot TEXT,
    finding_policy_snapshot TEXT,
    rag_request_snapshot TEXT,
    run_summary TEXT,
    evidence_summary TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    started_at TEXT,
    ended_at TEXT,
    UNIQUE (definition_id, definition_version, schedule_slot)
);
"""

_CREATE_RESULTS = """
CREATE TABLE IF NOT EXISTS automation_results (
    id TEXT PRIMARY KEY,
    server_id TEXT,
    owner_id TEXT NOT NULL,
    definition_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    answer TEXT,
    answer_mode TEXT NOT NULL DEFAULT 'none',
    confidence TEXT,
    source_refs TEXT,
    dedupe_key TEXT NOT NULL,
    visibility_destination TEXT,
    review_state TEXT NOT NULL DEFAULT 'unread',
    reviewed_at TEXT,
    reviewed_by TEXT,
    review_note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    UNIQUE (owner_id, dedupe_key)
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_automation_runs_owner_definition_created
    ON automation_runs (owner_id, definition_id, created_at);
CREATE INDEX IF NOT EXISTS idx_automation_runs_owner_status
    ON automation_runs (owner_id, status);
CREATE INDEX IF NOT EXISTS idx_automation_results_owner_review
    ON automation_results (owner_id, review_state, created_at);
CREATE INDEX IF NOT EXISTS idx_automation_definitions_owner_next_run
    ON automation_definitions (owner_id, next_run_at);
"""


def migrate(db: _MigrationCapableDB) -> None:
    """Apply the v3 -> v4 schema migration to ``db``. Idempotent."""
    with closing(db._get_connection()) as conn:
        existing = conn.execute(
            "PRAGMA table_info(automation_definitions)"
        ).fetchall()
        if not existing:
            # No automation_definitions on this connection: nothing to
            # migrate (same memory-correctness rule as v1_to_v2).
            return
        conn.execute(_CREATE_RUNS)
        conn.execute(_CREATE_RESULTS)
        conn.executescript(_INDEXES)

        def_cols = {row[1] for row in existing}
        for name, decl in _DEFINITION_COLUMNS_V4:
            if name not in def_cols:
                conn.execute(
                    f"ALTER TABLE automation_definitions ADD COLUMN {name} {decl}"
                )
        rem_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(reminder_tasks)")
        }
        if "transfer_state" not in rem_cols:
            conn.execute(
                "ALTER TABLE reminder_tasks ADD COLUMN transfer_state TEXT"
            )

        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        current_version = int(row[0]) if row and row[0] is not None else 0
        if current_version < 4:
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (4,))
        conn.commit()
    logger.debug(
        "Scheduling schema migrated to version 4 (automation runs/results)"
    )


def rollback(db: _MigrationCapableDB) -> None:
    """Revert to v3: drop the new tables and the added columns.

    Column removal uses ALTER TABLE DROP COLUMN (SQLite >=3.35; the
    Python >=3.11 floor bundles it). Deviation from v2_to_v3's
    table-recreate rollback is deliberate: ten columns across two tables
    make recreate disproportionate here.
    """
    with closing(db._get_connection()) as conn:
        conn.execute("DROP TABLE IF EXISTS automation_runs")
        conn.execute("DROP TABLE IF EXISTS automation_results")
        def_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(automation_definitions)")
        }
        for name, _decl in _DEFINITION_COLUMNS_V4:
            if name in def_cols:
                conn.execute(
                    f"ALTER TABLE automation_definitions DROP COLUMN {name}"
                )
        rem_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(reminder_tasks)")
        }
        if "transfer_state" in rem_cols:
            conn.execute("ALTER TABLE reminder_tasks DROP COLUMN transfer_state")
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (3,))
        conn.commit()
    logger.debug("Scheduling schema rolled back to version 3")
```

In `scheduled_tasks_db.py`: set `_CURRENT_SCHEMA_VERSION = 4`; in `_initialize_schema` add the import and call `migrate_v3_to_v4(self)` after `migrate_v2_to_v3(self)`; extend `_AUTOMATION_DEFINITION_COLUMNS` with the ten new column names; add `"finding_policy", "retention_policy"` to `_AUTOMATION_JSON_FIELDS`.

- [ ] **Step 4: Run the migration + schema + DB suites**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_migrations.py Tests/Scheduling/test_schema.py Tests/Scheduling/test_scheduled_tasks_db.py -v`
Expected: new tests PASS; any existing test asserting version 3 or column sets needs updating in this task (they pin the old schema deliberately — update the pinned values, nothing else).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/db/ Tests/Scheduling/test_migrations.py Tests/Scheduling/test_schema.py Tests/Scheduling/test_scheduled_tasks_db.py
git commit -m "feat(scheduling): schema v4 — automation runs/results tables + parity columns"
```

### Task 4: Model updates

**Files:**
- Modify: `tldw_chatbook/Scheduling/models.py`
- Test: `Tests/Scheduling/test_models.py`

**Interfaces:**
- Produces: `RunStatus`, `RunOutcome`, `ReviewState` enums; `AutomationRun`, `AutomationResult` models; `AutomationDefinition` gains ten fields; `AutomationPreview.validation_errors/warnings: list[dict[str, Any]]` + `risk_class`. Tasks 6–7 and PR-2/3 consume these names verbatim.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Scheduling/test_models.py`:

```python
def test_run_status_matches_server_vocabulary_plus_timed_out():
    assert {s.value for s in RunStatus} == {
        "queued", "running", "completed", "failed",
        "skipped", "cancelled", "timed_out",
    }


def test_run_outcome_and_review_state_match_server_literals():
    assert {o.value for o in RunOutcome} == {
        "finding", "no_match", "partial", "degraded", "none",
    }
    assert {r.value for r in ReviewState} == {"unread", "read", "dismissed"}


def test_automation_run_defaults():
    run = AutomationRun(
        owner_id="local", definition_id="d1",
        definition_version=1, trigger_reason="scheduled",
    )
    assert run.status == RunStatus.QUEUED
    assert run.outcome == RunOutcome.NONE
    assert run.schedule_slot is None


def test_automation_result_defaults_to_unread():
    result = AutomationResult(
        owner_id="local", definition_id="d1", run_id="r1",
        kind="finding", title="t", summary="s", dedupe_key="k",
    )
    assert result.review_state == ReviewState.UNREAD
    assert result.answer_mode == "none"


def test_definition_gains_parity_fields_with_defaults():
    d = AutomationDefinition(family=AutomationFamily.RECURRING_QUESTION, name="n")
    assert d.resolution_state == "open"
    assert d.finding_policy == {"preset": "balanced_findings"}
    assert d.retention_policy == {"mode": "default"}
    assert d.next_run_at is None and d.transfer_state is None


def test_preview_error_lists_hold_server_shaped_dicts():
    p = AutomationPreview(
        family=AutomationFamily.RECURRING_QUESTION,
        validation_errors=[{"field": "config.scope", "code": "scope_empty",
                            "message": "Scope must include at least one readable searchable source."}],
        warnings=[{"code": "source_unavailable", "source": "chats"}],
    )
    assert p.validation_errors[0]["code"] == "scope_empty"
    assert p.risk_class is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_models.py -v`
Expected: FAIL — imports of `RunStatus`/`AutomationRun`/etc. don't exist; the preview test fails Pydantic validation (`list[str]`).

- [ ] **Step 3: Implement in `models.py`**

```python
class RunStatus(str, Enum):
    """Automation run status — server literals plus local ``timed_out``.

    The server's normalized API folds timeouts into ``failed`` and keeps
    the truth in ``run_summary.legacy_status``; locally ``timed_out`` is
    first-class (TASK-18939 vocabulary, spec §4.1).
    """

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class RunOutcome(str, Enum):
    """Automation run outcome, orthogonal to status."""

    FINDING = "finding"
    NO_MATCH = "no_match"
    PARTIAL = "partial"
    DEGRADED = "degraded"
    NONE = "none"


class ReviewState(str, Enum):
    """Result review state."""

    UNREAD = "unread"
    READ = "read"
    DISMISSED = "dismissed"


class AutomationRun(BaseModel):
    """One local automation execution (server ``RunRow`` shape, spec §4.1)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_id: str | None = None
    owner_id: str = "local"
    definition_id: str
    definition_version: int = 1
    trigger_reason: str
    status: RunStatus = RunStatus.QUEUED
    outcome: RunOutcome = RunOutcome.NONE
    schedule_slot: str | None = None
    scope_snapshot: dict[str, Any] = Field(default_factory=dict)
    finding_policy_snapshot: dict[str, Any] = Field(default_factory=dict)
    rag_request_snapshot: dict[str, Any] = Field(default_factory=dict)
    run_summary: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    failure_reason: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class AutomationResult(BaseModel):
    """One automation result (server ``ResultRow`` shape, spec §4.2)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    server_id: str | None = None
    owner_id: str = "local"
    definition_id: str
    run_id: str
    kind: str
    title: str
    summary: str
    answer: Any | None = None
    answer_mode: str = "none"
    confidence: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    dedupe_key: str
    visibility_destination: dict[str, Any] = Field(default_factory=dict)
    review_state: ReviewState = ReviewState.UNREAD
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    review_note: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None
```

On `AutomationDefinition`, add:

```python
    disabled_lock_kind: str | None = None
    disabled_reason: str | None = None
    resolution_state: str = "open"
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolved_result_id: str | None = None
    finding_policy: dict[str, Any] = Field(
        default_factory=lambda: {"preset": "balanced_findings"}
    )
    retention_policy: dict[str, Any] = Field(
        default_factory=lambda: {"mode": "default"}
    )
    next_run_at: datetime | None = None
    transfer_state: str | None = None
```

On `AutomationPreview`, change the two list fields and add `risk_class`:

```python
    risk_class: str | None = None
    validation_errors: list[dict[str, Any]] | None = None
    warnings: list[dict[str, Any]] | None = None
```

- [ ] **Step 4: Run the model + service suites**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_models.py Tests/Scheduling/test_scheduling_service.py -v`
Expected: PASS (the service suite catches accidental `extra="forbid"` breakage from the new definition fields).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/models.py Tests/Scheduling/test_models.py
git commit -m "feat(scheduling): run/result models + definition parity fields (spec §4)"
```

### Task 5: Slot-key module

**Files:**
- Create: `tldw_chatbook/Scheduling/slot_keys.py`
- Test: `Tests/Scheduling/test_slot_keys.py`

**Interfaces:**
- Produces: `canonical_hash(payload: dict) -> str` (sha256 hex); `build_scheduled_run_idempotency_key(*, definition_id: str, definition_version: int, schedule_slot: str) -> str` (byte-identical to the server's `recurring_question_jobs.py` recipe, `"scheduled-task-rq:" + sha256`); `build_manual_run_idempotency_payload(*, definition_id: str) -> dict[str, str]`. PR-2's dispatch and PR-3's pending-mutation hashing consume all three.

- [ ] **Step 1: Write the failing tests**

```python
import hashlib
import json

from tldw_chatbook.Scheduling.slot_keys import (
    build_manual_run_idempotency_payload,
    build_scheduled_run_idempotency_key,
    canonical_hash,
)


def test_scheduled_key_matches_server_recipe_byte_for_byte():
    payload = {
        "definition_id": "d1",
        "definition_version": 3,
        "schedule_slot": "2026-09-01T09:00:00+00:00",
    }
    expected_digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    key = build_scheduled_run_idempotency_key(
        definition_id="d1",
        definition_version=3,
        schedule_slot="2026-09-01T09:00:00+00:00",
    )
    assert key == f"scheduled-task-rq:{expected_digest}"


def test_canonical_hash_is_key_order_independent():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_manual_payload_matches_server_shape():
    assert build_manual_run_idempotency_payload(definition_id="d9") == {
        "action": "create_manual_run",
        "definition_id": "d9",
        "trigger_reason": "manual",
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_slot_keys.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `slot_keys.py`**

```python
"""Deterministic idempotency keys for automation runs.

Byte-identical to tldw_server's ``recurring_question_jobs.py`` recipe so
a definition's slot identity survives handoff in either direction
(spec-2026-08-31-schedules-handoff-parity.md §7.2).
"""

from __future__ import annotations

import hashlib
import json


def canonical_hash(payload: dict) -> str:
    """Return a stable SHA-256 hex digest for a JSON-compatible payload."""
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_scheduled_run_idempotency_key(
    *, definition_id: str, definition_version: int, schedule_slot: str
) -> str:
    """Return the deterministic key for one scheduled slot (server parity)."""
    return "scheduled-task-rq:" + canonical_hash(
        {
            "definition_id": definition_id,
            "definition_version": definition_version,
            "schedule_slot": schedule_slot,
        }
    )


def build_manual_run_idempotency_payload(*, definition_id: str) -> dict[str, str]:
    """Return the idempotency payload for a manual run (server parity)."""
    return {
        "action": "create_manual_run",
        "definition_id": definition_id,
        "trigger_reason": "manual",
    }
```

- [ ] **Step 4: Run to verify PASS, then commit**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_slot_keys.py -v` → PASS.

```bash
git add tldw_chatbook/Scheduling/slot_keys.py Tests/Scheduling/test_slot_keys.py
git commit -m "feat(scheduling): slot-key module, byte-parity with server idempotency recipe"
```

### Task 6: Run accessors — create with slot dedupe + prune, update, list, reconcile

**Files:**
- Modify: `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (new section after the automation-definitions accessors; mirror `create_automation_definition`'s kwargs-validation and JSON-serialization discipline — read that method first, it is the house pattern)
- Test: `Tests/Scheduling/test_scheduled_tasks_db.py`

**Interfaces:**
- Consumes: Task 3's tables; Task 4's vocabularies (stored as plain strings).
- Produces (PR-2 dispatch consumes these exact signatures):
  - `create_automation_run(owner_id: str, definition_id: str, definition_version: int, trigger_reason: str, **kwargs) -> str | None` — returns the new run id, or **None when the slot UNIQUE dedupes** the insert.
  - `update_automation_run(run_id: str, **kwargs) -> bool`
  - `list_automation_runs(owner_id: str, definition_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]`
  - `reconcile_stale_automation_runs(older_than_seconds: float) -> int`
  - Class constant `_RUNS_RETAINED_PER_DEFINITION = 200`

- [ ] **Step 1: Write the failing tests**

```python
def _mk_db(tmp_path):
    return ScheduledTasksDB(str(tmp_path / "s.db"), client_id="t")


def test_create_run_and_slot_dedupe(tmp_path):
    db = _mk_db(tmp_path)
    first = db.create_automation_run(
        "local", "d1", 1, "scheduled",
        status="running", schedule_slot="2026-09-01T09:00:00+00:00",
    )
    assert first is not None
    duplicate = db.create_automation_run(
        "local", "d1", 1, "scheduled",
        status="running", schedule_slot="2026-09-01T09:00:00+00:00",
    )
    assert duplicate is None  # deduped, not raised
    two_manuals = [
        db.create_automation_run("local", "d1", 1, "manual", status="running")
        for _ in range(2)
    ]
    assert all(two_manuals)  # NULL slots never collide


def test_update_and_list_runs(tmp_path):
    db = _mk_db(tmp_path)
    run_id = db.create_automation_run("local", "d1", 1, "manual", status="running")
    assert db.update_automation_run(
        run_id, status="completed", outcome="finding",
        run_summary={"note": "ok"},
    )
    rows = db.list_automation_runs("local", definition_id="d1")
    assert rows[0]["status"] == "completed"
    assert rows[0]["run_summary"] == {"note": "ok"}  # JSON round-trips


def test_prune_keeps_newest_200_per_definition(tmp_path):
    db = _mk_db(tmp_path)
    for i in range(205):
        db.create_automation_run(
            "local", "d1", 1, "scheduled",
            status="completed", schedule_slot=f"slot-{i:04d}",
        )
    rows = db.list_automation_runs("local", definition_id="d1", limit=500)
    assert len(rows) == 200
    slots = {r["schedule_slot"] for r in rows}
    assert "slot-0204" in slots and "slot-0000" not in slots


def test_reconcile_marks_stale_running_as_interrupted(tmp_path):
    db = _mk_db(tmp_path)
    run_id = db.create_automation_run("local", "d1", 1, "manual", status="running")
    # Backdate created_at past the cutoff.
    with closing(db._get_connection()) as conn:
        conn.execute(
            "UPDATE automation_runs SET created_at = ? WHERE id = ?",
            ("2020-01-01T00:00:00+00:00", run_id),
        )
        conn.commit()
    reconciled = db.reconcile_stale_automation_runs(older_than_seconds=3600)
    assert reconciled == 1
    row = db.list_automation_runs("local", definition_id="d1")[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] == {"code": "interrupted"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_scheduled_tasks_db.py -k automation_run -v`
Expected: FAIL — `AttributeError: create_automation_run`.

- [ ] **Step 3: Implement**

Add to `ScheduledTasksDB` (registries near the existing ones, methods after the automation-definitions section):

```python
    _RUNS_RETAINED_PER_DEFINITION = 200

    _AUTOMATION_RUN_COLUMNS = {
        "server_id", "status", "outcome", "schedule_slot",
        "scope_snapshot", "finding_policy_snapshot", "rag_request_snapshot",
        "run_summary", "evidence_summary", "failure_reason",
        "updated_at", "started_at", "ended_at",
    }
    _AUTOMATION_RUN_JSON_FIELDS = {
        "scope_snapshot", "finding_policy_snapshot", "rag_request_snapshot",
        "run_summary", "evidence_summary", "failure_reason",
    }
```

Method bodies follow `create_automation_definition`'s structure exactly (kwargs via `_validate_kwargs`, JSON fields through `json.dumps`, datetimes through `_to_utc_iso`, reads through `_row_to_dict(json_fields=self._AUTOMATION_RUN_JSON_FIELDS)`), with the two behaviors specific to runs:

```python
    def create_automation_run(
        self,
        owner_id: str,
        definition_id: str,
        definition_version: int,
        trigger_reason: str,
        **kwargs: Any,
    ) -> str | None:
        """Insert a run; return its id, or None when the slot deduped it.

        Also prunes the definition's runs to the newest
        ``_RUNS_RETAINED_PER_DEFINITION`` (spec §4.1): an
        every-15-minutes definition would otherwise write ~35k rows/year.
        """
        self._validate_kwargs(kwargs, self._AUTOMATION_RUN_COLUMNS, "automation run")
        run_id = str(uuid.uuid4())
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        fields: dict[str, Any] = {
            "id": run_id,
            "owner_id": owner_id,
            "definition_id": definition_id,
            "definition_version": definition_version,
            "trigger_reason": trigger_reason,
            "status": "queued",
            "outcome": "none",
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            if key in self._AUTOMATION_RUN_JSON_FIELDS:
                fields[key] = json.dumps(value)
            elif isinstance(value, datetime):
                fields[key] = self._to_utc_iso(value)
            else:
                fields[key] = value
        columns = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        with self.transaction() as conn:
            try:
                conn.execute(
                    f"INSERT INTO automation_runs ({columns}) VALUES ({placeholders})",
                    list(fields.values()),
                )
            except sqlite3.IntegrityError:
                # The (definition, version, slot) UNIQUE fired: this slot
                # already ran. Dedupe is a result, not an error.
                return None
            conn.execute(
                """
                DELETE FROM automation_runs
                WHERE definition_id = ? AND id NOT IN (
                    SELECT id FROM automation_runs
                    WHERE definition_id = ?
                    ORDER BY created_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (definition_id, definition_id, self._RUNS_RETAINED_PER_DEFINITION),
            )
        return run_id

    def reconcile_stale_automation_runs(self, older_than_seconds: float) -> int:
        """Mark queued/running runs older than the cutoff as interrupted.

        Called at scheduler start (spec §4.1): an app killed mid-run must
        not leave a phantom in-flight run.
        """
        cutoff = self._to_utc_iso(
            datetime.now(timezone.utc) - timedelta(seconds=older_than_seconds)
        )
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE automation_runs
                SET status = 'failed',
                    failure_reason = ?,
                    ended_at = ?,
                    updated_at = ?
                WHERE status IN ('queued', 'running') AND created_at < ?
                """,
                (json.dumps({"code": "interrupted"}), now_iso, now_iso, cutoff),
            )
            return cursor.rowcount
```

`update_automation_run` and `list_automation_runs` are the definition accessors' update/list shapes pointed at `automation_runs` with the run registries (ordering: `ORDER BY created_at DESC, id DESC`).

- [ ] **Step 4: Run the DB suite**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_scheduled_tasks_db.py -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/db/scheduled_tasks_db.py Tests/Scheduling/test_scheduled_tasks_db.py
git commit -m "feat(scheduling): automation run accessors — slot dedupe, prune, stale reconcile"
```

### Task 7: Result accessors — create, list, review

**Files:**
- Modify: `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`
- Test: `Tests/Scheduling/test_scheduled_tasks_db.py`

**Interfaces:**
- Produces (PR-2 writes findings; PR-3 syncs; PR-6's inbox reads):
  - `create_automation_result(owner_id: str, definition_id: str, run_id: str, kind: str, title: str, summary: str, dedupe_key: str, **kwargs) -> str | None` — None when `(owner_id, dedupe_key)` dedupes.
  - `list_automation_results(owner_id: str, review_state: str | None = None, definition_id: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]`
  - `count_unread_results(owner_id: str) -> int`
  - `update_result_review(result_id: str, review_state: str, review_note: str | None = None, reviewed_by: str | None = None) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
def test_create_result_and_dedupe(tmp_path):
    db = _mk_db(tmp_path)
    rid = db.create_automation_result(
        "local", "d1", "r1", "finding", "Title", "Summary", "key-1",
        answer_mode="synthesized", answer={"text": "42"},
        source_refs=[{"source": "notes", "id": "n1"}],
    )
    assert rid is not None
    assert db.create_automation_result(
        "local", "d1", "r2", "finding", "Again", "S", "key-1"
    ) is None  # same (owner, dedupe_key)
    row = db.list_automation_results("local")[0]
    assert row["review_state"] == "unread"
    assert row["answer"] == {"text": "42"}
    assert row["source_refs"] == [{"source": "notes", "id": "n1"}]


def test_review_transitions_and_unread_count(tmp_path):
    db = _mk_db(tmp_path)
    rid = db.create_automation_result(
        "local", "d1", "r1", "finding", "T", "S", "k1"
    )
    db.create_automation_result("local", "d1", "r2", "failure", "F", "S", "k2")
    assert db.count_unread_results("local") == 2
    assert db.update_result_review(rid, "read", reviewed_by="local")
    assert db.count_unread_results("local") == 1
    assert db.list_automation_results("local", review_state="read")[0]["id"] == rid
    assert not db.update_result_review("missing", "dismissed")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest Tests/Scheduling/test_scheduled_tasks_db.py -k automation_result -v` → FAIL (`AttributeError`).

- [ ] **Step 3: Implement**

Registries:

```python
    _AUTOMATION_RESULT_COLUMNS = {
        "server_id", "answer", "answer_mode", "confidence", "source_refs",
        "visibility_destination", "review_state", "reviewed_at",
        "reviewed_by", "review_note", "updated_at",
    }
    _AUTOMATION_RESULT_JSON_FIELDS = {
        "answer", "confidence", "source_refs", "visibility_destination",
    }
```

`create_automation_result` mirrors Task 6's create (same INSERT/IntegrityError→None discipline; required fields `id, owner_id, definition_id, run_id, kind, title, summary, dedupe_key, review_state='unread', answer_mode='none', created_at, updated_at`; no prune). `update_result_review`:

```python
    def update_result_review(
        self,
        result_id: str,
        review_state: str,
        review_note: str | None = None,
        reviewed_by: str | None = None,
    ) -> bool:
        """Set a result's review state; returns False for an unknown id."""
        now_iso = self._to_utc_iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE automation_results
                SET review_state = ?, review_note = ?, reviewed_by = ?,
                    reviewed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (review_state, review_note, reviewed_by, now_iso, now_iso, result_id),
            )
            return cursor.rowcount > 0
```

`list_automation_results` follows the run list shape with optional `review_state`/`definition_id` filters; `count_unread_results` is a single `SELECT COUNT(*) … WHERE owner_id = ? AND review_state = 'unread'`.

- [ ] **Step 4: Run the full scheduling suite**

Run: `.venv/bin/python -m pytest Tests/Scheduling/ -v`
Expected: PASS across the module (reproduce any pre-existing dev failures with your changes stashed before attributing them — slice-1's notes recorded 2 such).

- [ ] **Step 5: Commit and open PR-1**

```bash
git add tldw_chatbook/Scheduling/db/scheduled_tasks_db.py Tests/Scheduling/test_scheduled_tasks_db.py
git commit -m "feat(scheduling): automation result accessors — dedupe, review, unread count"
```

Open the PR against `dev` covering Tasks 3–7. In the PR body, link the spec and note: schema number re-verified against `origin/dev` at merge (Global Constraints).

---

## After PR-1

PR-2 (local recurring_question execution) gets its own plan written against dev's state at that time — the validator/classifier ports, queue feed, spawn-shaped handler, and notification wiring consume the interfaces produced above (`create_automation_run` dedupe contract, `RunStatus.TIMED_OUT`, `build_scheduled_run_idempotency_key`, `ReminderHandler.app_getter`).
