# Supervisor Fleet PR 1 — Agent Definitions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Named, user-authored sub-agent definitions (name/instructions/tools/model) stored in AgentRuns_DB, selectable by the supervisor via a new optional `agent` parameter on `spawn_subagent`, editable in a new Settings ▸ Agents category.

**Architecture:** Pure `AgentDefinition` dataclass + validation in `agent_models.py`; storage + CRUD in `AgentRuns_DB.py` (CREATE-IF-NOT-EXISTS + idempotent-ALTER migration, version row 5); per-run schema build + spawn-closure resolution in `tool_catalog.py`/`agent_service.py`/`agent_runtime.py`; a bespoke Settings category rendering a dedicated `AgentsSettingsPanel` widget (immediate DB CRUD, no TOML draft). Spawn stays synchronous — this PR is purely additive; no `agent` argument ⇒ byte-identical behavior.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite (WAL, per-thread held connections), pytest + pytest-asyncio.

**Spec:** `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md` §4 (read it first; §3 invariants apply).

## Global Constraints

- Worktree: `.worktrees/supervisor-fleet`, branch `feat/supervisor-agent-fleet`. NEVER run git in the parent checkout. NEVER use `git stash` (repo-wide across 100+ worktrees). Push after every task.
- Run tests with the repo venv: `VIRTUAL_ENV=.venv uv pip install -e ".[dev]"` if pytest is missing; pytest is the ONLY python entry point (a bare `python -c "import tldw_chatbook..."` probe writes to the LIVE config).
- "no tests ran" = FAILED gate. A gate passes only on a read, nonzero passed-count.
- `agent_models.py` stays pure: stdlib imports only (dataclasses, typing, re, json, hashlib OK; no Textual/app/DB/I/O).
- **Identity contract:** `console_agent_bridge._is_subagent` (Chat/console_agent_bridge.py:903) detects sub-agent turns by PREFIX-matching the `agents.subagent_system` prompt. Definition instructions must be APPENDED after the base prompt, never prepended.
- Caps (spec §4, exact values): name slug `^[a-z][a-z0-9-]{0,63}$`; reserved names `general`, `subagent`; description ≤ 200 chars; instructions ≤ 16,000 chars, non-empty.
- Intersection, never union: a definition's tool list only narrows the child's inherited allow-list (spec §3 invariant 1).
- AgentRuns_DB has NO migration framework: new tables via `CREATE TABLE IF NOT EXISTS`, new columns via idempotent `PRAGMA table_info` + `ALTER TABLE` on every open, then `INSERT OR IGNORE INTO schema_version (version) VALUES (5)` (append-per-version; never UPDATE).
- Never hand-edit `css/tldw_cli_modular.tcss` (generated). This plan adds NO new CSS — reuse `settings-*` classes.
- Commit messages: conventional prefix + trailer `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 0: Backlog task hygiene

**Files:**
- Create: `backlog/tasks/` entries via CLI (never by hand)

**Interfaces:**
- Produces: a parent backlog task id + a PR-1 subtask id, both In Progress, referenced in later commit messages as context (not required in every message).

- [ ] **Step 1: Pick collision-safe IDs**

Run (from the worktree root):
```bash
git fetch origin dev --quiet
ls backlog/tasks/ | grep -oE 'task-[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -3
git ls-tree -r --name-only origin/dev -- backlog/tasks/ | grep -oE 'task-[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -3
```
Take the max of both lists and leapfrog with headroom (+20 or more — ten-plus collisions have occurred in this repo; Done tasks never move).

- [ ] **Step 2: Create parent + PR-1 task**

```bash
backlog task create "Supervisor agent fleet program" \
  -d "Named sub-agent definitions, background/parallel execution, steering, Console fleet panel. Spec: Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md" \
  -s "In Progress" --ac "All six PR-phase subtasks Done"
backlog task create "Fleet PR1: agent definitions (model, DB, spawn param, Settings editor)" \
  -p <PARENT_ID> -s "In Progress" \
  -d "Phase 1 of the fleet spec (§4): AgentDefinition + agent_definitions table + spawn 'agent' param + Settings ▸ Agents panel. Plan: Docs/superpowers/plans/2026-08-08-supervisor-fleet-pr1-agent-definitions.md" \
  --ac "Named definition resolves at spawn (prompt appended, tools intersected, model override)" \
  --ac "No-agent spawn is byte-identical to today" \
  --ac "Settings ▸ Agents CRUD works against the real DB" \
  --ac "Migration is idempotent on existing DB files" \
  --ac "User Guide updated (settings.md + console/agent-runs-and-tools.md)"
```
(Repeat `--ac` per criterion — a comma-joined `--ac` writes one run-on criterion.)

- [ ] **Step 3: Commit the task files**

```bash
git add backlog/tasks/ && git commit -m "chore: file fleet PR1 backlog tasks" && git push
```

---

### Task 1: `AgentDefinition` model, validation, fingerprint (pure)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_models.py` (append after `RunBudget`, before `AgentStep`)
- Test: `Tests/Agents/test_agent_models.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces (exact, later tasks import these from `tldw_chatbook.Agents.agent_models`):
  - `AgentDefinition` frozen dataclass: `name: str`, `description: str = ""`, `instructions: str = ""`, `tool_allowlist: tuple[str, ...] = ()`, `model: str = ""`, `enabled: bool = True`
  - `validate_agent_definition(defn: AgentDefinition) -> list[str]` (empty list = valid)
  - `definition_fingerprint(defn: AgentDefinition) -> str` (16 hex chars)
  - `definition_from_row(row: dict) -> AgentDefinition`
  - Constants: `AGENT_DEFINITION_NAME_PATTERN`, `AGENT_DEFINITION_RESERVED_NAMES`, `AGENT_DEFINITION_DESCRIPTION_MAX_CHARS = 200`, `AGENT_DEFINITION_INSTRUCTIONS_MAX_CHARS = 16_000`

- [ ] **Step 1: Write the failing tests** (append to `Tests/Agents/test_agent_models.py`)

```python
from tldw_chatbook.Agents.agent_models import (
    AgentDefinition,
    definition_fingerprint,
    definition_from_row,
    validate_agent_definition,
)


def _valid_definition(**overrides):
    base = dict(
        name="researcher",
        description="Searches and summarizes sources.",
        instructions="Research the task thoroughly. Cite sources.",
        tool_allowlist=("web_search",),
        model="",
        enabled=True,
    )
    base.update(overrides)
    return AgentDefinition(**base)


def test_valid_definition_passes():
    assert validate_agent_definition(_valid_definition()) == []


def test_name_must_be_slug():
    for bad in ("Researcher", "re searcher", "-x", "9x", "a" * 65, ""):
        assert validate_agent_definition(_valid_definition(name=bad)), bad


def test_reserved_names_rejected():
    for reserved in ("general", "subagent"):
        errors = validate_agent_definition(_valid_definition(name=reserved))
        assert any("reserved" in e for e in errors)


def test_description_and_instructions_caps():
    assert validate_agent_definition(_valid_definition(description="d" * 201))
    assert validate_agent_definition(_valid_definition(instructions="i" * 16_001))
    assert validate_agent_definition(_valid_definition(instructions="   "))


def test_fingerprint_covers_identity_fields_only():
    a = _valid_definition()
    assert definition_fingerprint(a) == definition_fingerprint(
        _valid_definition(description="different", enabled=False)
    )
    assert definition_fingerprint(a) != definition_fingerprint(
        _valid_definition(instructions="other text")
    )
    assert definition_fingerprint(a) != definition_fingerprint(
        _valid_definition(tool_allowlist=())
    )
    assert definition_fingerprint(a) != definition_fingerprint(
        _valid_definition(model="gpt-x")
    )
    assert len(definition_fingerprint(a)) == 16


def test_definition_from_row_round_trip():
    row = {
        "name": "critic",
        "description": "Reviews drafts.",
        "instructions": "Critique carefully.",
        "tool_allowlist": ["calculator"],
        "model": "m1",
        "enabled": 1,
    }
    defn = definition_from_row(row)
    assert defn == AgentDefinition(
        name="critic",
        description="Reviews drafts.",
        instructions="Critique carefully.",
        tool_allowlist=("calculator",),
        model="m1",
        enabled=True,
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_agent_models.py -v -k "definition or fingerprint"`
Expected: FAIL / ImportError ("cannot import name 'AgentDefinition'").

- [ ] **Step 3: Implement** (in `agent_models.py`; add `import hashlib`, `import json`, `import re` to the stdlib imports)

```python
#: Fleet spec §4: validation caps for user-authored agent definitions.
#: description rides the spawn tool's schema (re-sent every fence-model
#: turn); instructions ride every child model turn — both caps are cost
#: controls, not polish.
AGENT_DEFINITION_NAME_PATTERN = r"^[a-z][a-z0-9-]{0,63}$"
AGENT_DEFINITION_RESERVED_NAMES = frozenset({"general", "subagent"})
AGENT_DEFINITION_DESCRIPTION_MAX_CHARS = 200
AGENT_DEFINITION_INSTRUCTIONS_MAX_CHARS = 16_000


@dataclass(frozen=True)
class AgentDefinition:
    """A named, user-authored sub-agent template (fleet spec §4).

    ``instructions`` are APPENDED to the internal ``agents.subagent_system``
    prompt at spawn time — never a replacement (the base prompt is an
    identity contract: console_agent_bridge detects sub-agent turns by
    prefix-matching it). ``tool_allowlist`` only ever narrows the child's
    inherited allow-list (intersection, never union); empty means inherit.
    ``model`` overrides the parent's model on the SAME provider endpoint;
    empty means inherit.
    """

    name: str
    description: str = ""
    instructions: str = ""
    tool_allowlist: tuple[str, ...] = ()
    model: str = ""
    enabled: bool = True


def validate_agent_definition(defn: AgentDefinition) -> list[str]:
    """Return validation errors for ``defn``; empty list means valid."""
    errors: list[str] = []
    if not re.fullmatch(AGENT_DEFINITION_NAME_PATTERN, defn.name or ""):
        errors.append(
            "name must be a lowercase slug (a-z, 0-9, hyphens; starts with "
            "a letter; max 64 chars)"
        )
    if defn.name in AGENT_DEFINITION_RESERVED_NAMES:
        errors.append(f"name '{defn.name}' is reserved")
    if len(defn.description) > AGENT_DEFINITION_DESCRIPTION_MAX_CHARS:
        errors.append(
            f"description exceeds {AGENT_DEFINITION_DESCRIPTION_MAX_CHARS} chars"
        )
    if not defn.instructions.strip():
        errors.append("instructions must not be empty")
    if len(defn.instructions) > AGENT_DEFINITION_INSTRUCTIONS_MAX_CHARS:
        errors.append(
            f"instructions exceed {AGENT_DEFINITION_INSTRUCTIONS_MAX_CHARS} chars"
        )
    return errors


def definition_fingerprint(defn: AgentDefinition) -> str:
    """16-hex-char content hash of the fields that shape a child run.

    Covers instructions/tool_allowlist/model ONLY — the audit identity of
    what actually ran (spec §4). description/enabled are presentation.
    """
    payload = json.dumps(
        {
            "instructions": defn.instructions,
            "tool_allowlist": sorted(defn.tool_allowlist),
            "model": defn.model,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def definition_from_row(row: dict) -> AgentDefinition:
    """Build an ``AgentDefinition`` from an ``agent_definitions`` DB row
    (``tool_allowlist`` already JSON-decoded to a list by the DB layer)."""
    return AgentDefinition(
        name=row["name"],
        description=row["description"],
        instructions=row["instructions"],
        tool_allowlist=tuple(row["tool_allowlist"]),
        model=row["model"],
        enabled=bool(row["enabled"]),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_agent_models.py -v`
Expected: ALL PASS (new + pre-existing model tests).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/agent_models.py Tests/Agents/test_agent_models.py
git commit -m "feat: AgentDefinition model, validation, fingerprint" && git push
```

---

### Task 2: `agent_definitions` table + CRUD in AgentRuns_DB

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (`_initialize_schema` at :164; new methods after `create_run`)
- Test: `Tests/DB/test_agent_runs_db.py` (append)

**Interfaces:**
- Consumes: Task 1's `AgentDefinition`, `validate_agent_definition`.
- Produces (methods on `AgentRunsDB`):
  - `create_agent_definition(defn: AgentDefinition) -> str` (returns id; raises `ValueError` on validation failure or duplicate name)
  - `update_agent_definition(definition_id: str, defn: AgentDefinition) -> None` (same raises)
  - `soft_delete_agent_definition(definition_id: str) -> None`
  - `list_agent_definitions(enabled_only: bool = False) -> list[dict]` (excludes soft-deleted; rows carry `id`, `name`, `description`, `instructions`, `tool_allowlist` (decoded list), `model`, `enabled`, `created_at`, `updated_at`; ordered by name)
  - `get_agent_definition(definition_id: str) -> dict | None`

- [ ] **Step 1: Write the failing tests** (append to `Tests/DB/test_agent_runs_db.py`)

```python
from tldw_chatbook.Agents.agent_models import AgentDefinition


def _defn(**overrides):
    base = dict(
        name="researcher",
        description="Searches sources.",
        instructions="Research thoroughly.",
        tool_allowlist=("web_search",),
    )
    base.update(overrides)
    return AgentDefinition(**base)


def test_definition_crud_round_trip(db):
    definition_id = db.create_agent_definition(_defn())
    rows = db.list_agent_definitions()
    assert [r["name"] for r in rows] == ["researcher"]
    assert rows[0]["tool_allowlist"] == ["web_search"]
    db.update_agent_definition(definition_id, _defn(description="v2"))
    assert db.get_agent_definition(definition_id)["description"] == "v2"
    db.soft_delete_agent_definition(definition_id)
    assert db.list_agent_definitions() == []


def test_duplicate_name_raises_and_frees_after_soft_delete(db):
    definition_id = db.create_agent_definition(_defn())
    with pytest.raises(ValueError, match="already exists"):
        db.create_agent_definition(_defn())
    db.soft_delete_agent_definition(definition_id)
    db.create_agent_definition(_defn())  # name reusable after soft delete


def test_invalid_definition_rejected_at_db_boundary(db):
    with pytest.raises(ValueError, match="reserved"):
        db.create_agent_definition(_defn(name="subagent"))


def test_enabled_only_filter(db):
    db.create_agent_definition(_defn(name="on-agent"))
    db.create_agent_definition(_defn(name="off-agent", enabled=False))
    assert [r["name"] for r in db.list_agent_definitions(enabled_only=True)] == [
        "on-agent"
    ]
    assert len(db.list_agent_definitions()) == 2


def test_definitions_survive_reopen_and_migration_is_idempotent(tmp_path):
    path = tmp_path / "agent_runs.db"
    first = AgentRunsDB(path, client_id="test")
    first.create_agent_definition(_defn())
    first.close()
    second = AgentRunsDB(path, client_id="test")  # re-runs _initialize_schema
    assert [r["name"] for r in second.list_agent_definitions()] == ["researcher"]
    with second.connection() as conn:
        versions = {
            row[0]
            for row in conn.execute("SELECT version FROM schema_version").fetchall()
        }
    assert 5 in versions
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/DB/test_agent_runs_db.py -v -k definition`
Expected: FAIL with `AttributeError: ... 'create_agent_definition'`.

- [ ] **Step 3: Implement**

(a) Append to the `executescript` DDL in `_initialize_schema` (after the `change_snapshots` index, inside the same triple-quoted string):

```sql
                -- v5 (fleet spec §4, PR 1): user-authored agent
                -- definitions. DURABILITY NOTE: from v5 on this DB holds
                -- durable USER-AUTHORED CONTENT, not just run telemetry --
                -- any future "clear run history" feature must NOT treat
                -- the file as disposable.
                CREATE TABLE IF NOT EXISTS agent_definitions (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    instructions TEXT NOT NULL DEFAULT '',
                    tool_allowlist TEXT NOT NULL DEFAULT '[]',
                    model TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                -- Partial unique index: a live name is unique, but a
                -- soft-deleted row releases its name for re-creation.
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_definitions_name
                    ON agent_definitions(name) WHERE deleted = 0;
```

(b) After the existing `INSERT OR IGNORE ... VALUES (4)` execute near the end of `_initialize_schema`, add (append-per-version convention — never UPDATE):

```python
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (5)"
            )
```

(c) New methods (place after `create_run`; module needs `from tldw_chatbook.Agents.agent_models import AgentDefinition, validate_agent_definition` — import at top with the other local imports):

```python
    def create_agent_definition(self, defn: AgentDefinition) -> str:
        """Insert a definition; returns its id.

        Raises:
            ValueError: On validation failure, or a duplicate live name.
        """
        errors = validate_agent_definition(defn)
        if errors:
            raise ValueError("; ".join(errors))
        definition_id = uuid.uuid4().hex
        now = _now_iso()
        try:
            with self.transaction() as conn:
                conn.execute(
                    """INSERT INTO agent_definitions
                       (id, name, description, instructions, tool_allowlist,
                        model, enabled, deleted, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                    (
                        definition_id,
                        defn.name,
                        defn.description,
                        defn.instructions,
                        json.dumps(list(defn.tool_allowlist)),
                        defn.model,
                        1 if defn.enabled else 0,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"an agent named '{defn.name}' already exists"
            ) from exc
        return definition_id

    def update_agent_definition(
        self, definition_id: str, defn: AgentDefinition
    ) -> None:
        """Replace a definition's fields (same raises as create)."""
        errors = validate_agent_definition(defn)
        if errors:
            raise ValueError("; ".join(errors))
        try:
            with self.transaction() as conn:
                conn.execute(
                    """UPDATE agent_definitions
                       SET name = ?, description = ?, instructions = ?,
                           tool_allowlist = ?, model = ?, enabled = ?,
                           updated_at = ?
                       WHERE id = ? AND deleted = 0""",
                    (
                        defn.name,
                        defn.description,
                        defn.instructions,
                        json.dumps(list(defn.tool_allowlist)),
                        defn.model,
                        1 if defn.enabled else 0,
                        _now_iso(),
                        definition_id,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"an agent named '{defn.name}' already exists"
            ) from exc

    def soft_delete_agent_definition(self, definition_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE agent_definitions SET deleted = 1, updated_at = ? "
                "WHERE id = ?",
                (_now_iso(), definition_id),
            )

    def _definition_row_to_dict(self, row: sqlite3.Row) -> dict:
        data = {key: row[key] for key in row.keys()}
        data["tool_allowlist"] = json.loads(data["tool_allowlist"] or "[]")
        data.pop("deleted", None)
        return data

    def list_agent_definitions(self, enabled_only: bool = False) -> list[dict]:
        """Live (non-deleted) definitions ordered by name."""
        query = "SELECT * FROM agent_definitions WHERE deleted = 0"
        if enabled_only:
            query += " AND enabled = 1"
        query += " ORDER BY name"
        with self.connection() as conn:
            rows = conn.execute(query).fetchall()
        return [self._definition_row_to_dict(row) for row in rows]

    def get_agent_definition(self, definition_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM agent_definitions WHERE id = ? AND deleted = 0",
                (definition_id,),
            ).fetchone()
        return self._definition_row_to_dict(row) if row else None
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/DB/test_agent_runs_db.py Tests/Agents/test_agent_runs_db_connection_reuse.py -v`
Expected: ALL PASS (new + all pre-existing DB tests — the reuse test guards the connection discipline you just touched).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py Tests/DB/test_agent_runs_db.py
git commit -m "feat: agent_definitions table + CRUD (schema v5)" && git push
```

---

### Task 3: `agent_runs` audit columns + `create_run` params

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (`_initialize_schema` ALTER-guard block at :224-231; `create_run` at :410; the `agent_runs` DDL at :175)
- Test: `Tests/DB/test_agent_runs_db.py` (append)

**Interfaces:**
- Consumes: Task 2's migration block.
- Produces: `create_run(..., agent_definition: str | None = None, definition_fingerprint: str | None = None)`; run dicts from `get_run`/`list_runs` carry both keys (None when absent).

- [ ] **Step 1: Write the failing tests**

```python
def test_create_run_records_definition_audit_fields(db):
    run_id = db.create_run(
        conversation_id="c",
        agent_kind="subagent",
        task="t",
        parent_run_id=None,
        agent_definition="researcher",
        definition_fingerprint="abc123def4567890",
    )
    run = db.get_run(run_id)
    assert run["agent_definition"] == "researcher"
    assert run["definition_fingerprint"] == "abc123def4567890"


def test_create_run_definition_fields_default_none(db):
    run_id = db.create_run(conversation_id="c", agent_kind="primary")
    run = db.get_run(run_id)
    assert run["agent_definition"] is None
    assert run["definition_fingerprint"] is None


def test_agent_runs_columns_backfilled_on_old_file(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    # Simulate a pre-v5 file: the v4-era 12-column table, no new columns.
    conn.execute(
        """CREATE TABLE agent_runs (
               id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
               parent_run_id TEXT, agent_kind TEXT NOT NULL, task TEXT,
               status TEXT NOT NULL, steps TEXT NOT NULL DEFAULT '[]',
               result TEXT, budget TEXT, created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL, assistant_message_id TEXT)"""
    )
    conn.commit()
    conn.close()
    db = AgentRunsDB(path, client_id="test")  # open runs the ALTER guards
    with db.connection() as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(agent_runs)").fetchall()
        }
    assert {"agent_definition", "definition_fingerprint"} <= columns
```

(`import sqlite3` is already present in the test module; add it if not.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/DB/test_agent_runs_db.py -v -k "definition_audit or default_none or backfilled"`
Expected: FAIL (`TypeError: create_run() got an unexpected keyword argument` / missing column).

- [ ] **Step 3: Implement**

(a) In the `agent_runs` DDL (fresh-file path), add two columns after `assistant_message_id TEXT`:

```sql
                    assistant_message_id TEXT,
                    agent_definition TEXT,
                    definition_fingerprint TEXT
```

(b) In the existing-file ALTER-guard block (after the `assistant_message_id` guard, same `existing_columns` set):

```python
            # v4->v5 (fleet spec §4): definition audit identity on runs --
            # same idempotent-ALTER mechanism as above.
            if "agent_definition" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN agent_definition TEXT"
                )
            if "definition_fingerprint" not in existing_columns:
                conn.execute(
                    "ALTER TABLE agent_runs ADD COLUMN definition_fingerprint TEXT"
                )
```

(c) `create_run`: add keyword params `agent_definition: str | None = None, definition_fingerprint: str | None = None`; extend the INSERT column list with `agent_definition, definition_fingerprint`, the VALUES tuple with two more `?`, and the params tuple with both values (after `assistant_message_id`). Update the docstring Args accordingly.

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/DB/test_agent_runs_db.py -v`
Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/AgentRuns_DB.py Tests/DB/test_agent_runs_db.py
git commit -m "feat: definition audit columns on agent_runs" && git push
```

---

### Task 4: `build_spawn_schema`

**Files:**
- Modify: `tldw_chatbook/Agents/tool_catalog.py` (immediately after `SPAWN_TOOL_SCHEMA` at :42-59)
- Test: Create `Tests/Agents/test_build_spawn_schema.py`

**Interfaces:**
- Consumes: Task 1's `AgentDefinition`; existing `SPAWN_TOOL_SCHEMA`, `ToolSchema`.
- Produces: `build_spawn_schema(definitions: Sequence[AgentDefinition]) -> ToolSchema` — exported from `tldw_chatbook.Agents.tool_catalog`.

- [ ] **Step 1: Write the failing tests**

```python
"""build_spawn_schema: the spawn tool's per-run schema with named agents."""

from tldw_chatbook.Agents.agent_models import AgentDefinition
from tldw_chatbook.Agents.tool_catalog import SPAWN_TOOL_SCHEMA, build_spawn_schema

RESEARCHER = AgentDefinition(
    name="researcher",
    description="Searches and summarizes sources.",
    instructions="Research thoroughly.",
)
CRITIC = AgentDefinition(name="critic", instructions="Critique carefully.")


def test_no_definitions_returns_shipped_schema_object():
    # Identity, not equality: byte-identical behavior when no definitions
    # exist (spec §4 — phase 1 is purely additive).
    assert build_spawn_schema([]) is SPAWN_TOOL_SCHEMA


def test_definitions_add_optional_agent_enum_and_roster():
    schema = build_spawn_schema([RESEARCHER, CRITIC])
    assert schema.id == SPAWN_TOOL_SCHEMA.id
    assert schema.name == SPAWN_TOOL_SCHEMA.name
    props = schema.parameters["properties"]
    assert props["task"] == SPAWN_TOOL_SCHEMA.parameters["properties"]["task"]
    assert props["agent"]["enum"] == ["researcher", "critic"]
    # Prose roster for fence-protocol models (they read descriptions,
    # not enums): one "name — description" line each.
    assert "researcher — Searches and summarizes sources." in props["agent"]["description"]
    assert "- critic" in props["agent"]["description"]
    # agent stays OPTIONAL: required is untouched.
    assert schema.parameters["required"] == ["task"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/Agents/test_build_spawn_schema.py -v`
Expected: FAIL / ImportError ("cannot import name 'build_spawn_schema'").

- [ ] **Step 3: Implement** (in `tool_catalog.py`; add `AgentDefinition` to the existing `.agent_models` import block; `Sequence` from `collections.abc`)

```python
def build_spawn_schema(definitions: "Sequence[AgentDefinition]") -> ToolSchema:
    """The spawn tool's schema for THIS run.

    With no definitions, returns ``SPAWN_TOOL_SCHEMA`` itself (identity —
    byte-identical payloads for every pre-definition caller). With
    definitions, adds an OPTIONAL ``agent`` parameter carrying both an
    ``enum`` (native tool-calling) and a prose roster in the description
    (fence-protocol models read prose better than schema; this text rides
    every fence-model turn, which is why AgentDefinition.description is
    hard-capped).
    """
    if not definitions:
        return SPAWN_TOOL_SCHEMA
    roster = "\n".join(
        f"- {d.name} — {d.description}" if d.description else f"- {d.name}"
        for d in definitions
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": SPAWN_TOOL_SCHEMA.parameters["properties"]["task"],
            "agent": {
                "type": "string",
                "enum": [d.name for d in definitions],
                "description": (
                    "Optional: run the task as one of these named agents "
                    "(omit for a generic sub-agent):\n" + roster
                ),
            },
        },
        "required": ["task"],
    }
    return ToolSchema(
        id=SPAWN_TOOL_SCHEMA.id,
        name=SPAWN_TOOL_SCHEMA.name,
        description=SPAWN_TOOL_SCHEMA.description,
        parameters=parameters,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest Tests/Agents/test_build_spawn_schema.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Agents/tool_catalog.py Tests/Agents/test_build_spawn_schema.py
git commit -m "feat: build_spawn_schema with named-agent roster" && git push
```

---

### Task 5: Service + loop wiring (resolution at spawn)

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`__init__` ~:288; `run_turn` after `reset_catalog_cache()` at :1442; `_run_one` signature :508 and `create_run` call :521; `runtime_schemas.append(SPAWN_TOOL_SCHEMA)` at :543; the `spawn` closure at :669)
- Modify: `tldw_chatbook/Agents/agent_runtime.py` (spawn dispatch branch :686-713)
- Test: `Tests/Agents/test_agent_service.py` (append), `Tests/Agents/test_agent_runtime.py` (append)

**Interfaces:**
- Consumes: Tasks 1-4 (`definition_from_row`, `definition_fingerprint`, `build_spawn_schema`, `list_agent_definitions`, `create_run` params).
- Produces: the spawn closure signature `spawn(spawn_task, *, allowed_tools=None, agent=None)`; `_run_one(..., agent_definition=None, definition_fingerprint=None)`; loop passes `agent=` only when the model supplied it (existing `lambda task: ...` test doubles keep working).

- [ ] **Step 1: Write the failing service tests** (append to `Tests/Agents/test_agent_service.py`; imports: `from tldw_chatbook.Agents.agent_models import AgentDefinition` and add `definition_fingerprint` where used)

```python
RESEARCHER_DEFN = AgentDefinition(
    name="researcher",
    description="Searches and summarizes.",
    instructions="Always cite sources in your result.",
    tool_allowlist=("calculator",),
)


def _seed_definition(db, defn=RESEARCHER_DEFN):
    db.create_agent_definition(defn)


def test_named_spawn_appends_instructions_and_keeps_identity_prefix(db):
    _seed_definition(db)
    service, chat = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "compute 6*7", "agent": "researcher"}),
            "sub answer: 42",
            "done",
        ],
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "delegate"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert outcome.status == RUN_DONE
    child_system = chat.calls[1]["messages_payload"][0]
    assert child_system["role"] == "system"
    # IDENTITY CONTRACT: base subagent prompt stays the PREFIX
    # (console_agent_bridge._is_subagent prefix-matches it) ...
    assert child_system["content"].startswith(SUBAGENT_SYSTEM_PROMPT.split(".")[0])
    # ... and the definition's instructions are appended after it.
    assert "Always cite sources" in child_system["content"]


def test_named_spawn_intersects_allowlist_never_grants(db):
    _seed_definition(
        db,
        AgentDefinition(
            name="narrow",
            instructions="Do the task.",
            # calculator is in the parent set; forbidden_tool is not — the
            # definition can narrow to calculator but never grant extras.
            tool_allowlist=("calculator", "forbidden_tool"),
        ),
    )
    service, chat = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "narrow"}),
            "child done",
            "done",
        ],
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    # Channel-agnostic: disclosed schemas may ride the system prompt
    # (fence protocol) OR the tools= kwarg (native) — inspect the whole
    # provider call. (`import json` at module top if not present.)
    child_call = json.dumps(chat.calls[1], default=str)
    assert "calculator" in child_call
    assert "get_current_datetime" not in child_call  # narrowed away
    assert "forbidden_tool" not in child_call  # never granted


def test_named_spawn_model_override_same_endpoint(db):
    _seed_definition(
        db,
        AgentDefinition(
            name="cheap", instructions="Do it.", model="tiny-model"
        ),
    )
    service, chat = make_service(
        db,
        [fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "cheap"}), "ok", "done"],
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert chat.calls[1]["model"] == "tiny-model"
    assert chat.calls[0]["model"] == "test-model"
    assert chat.calls[1]["api_endpoint"] == "llama_cpp"


def test_unknown_agent_refused_without_burning_budget(db):
    _seed_definition(db)
    service, chat = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "nope"}),
            fence(SPAWN_TOOL_NAME, {"task": "t2", "agent": "researcher"}),
            "child ok",
            fence(SPAWN_TOOL_NAME, {"task": "t3", "agent": "researcher"}),
            "child ok 2",
            "done",
        ],
    )
    run_id, outcome = service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    # CFG default budget allows max_subagents=2: the unknown-agent refusal
    # must not have consumed a slot, so BOTH later spawns succeed.
    assert outcome.status == RUN_DONE
    assert db.count_subagent_runs("c") == 2
    # The refusal itself surfaced the roster to the model.
    refusal = chat.calls[1]["messages_payload"]
    assert any(
        "unknown agent 'nope'" in str(m.get("content", "")) for m in refusal
    )


def test_named_spawn_records_audit_fields(db):
    _seed_definition(db)
    service, _ = make_service(
        db,
        [
            fence(SPAWN_TOOL_NAME, {"task": "t", "agent": "researcher"}),
            "child ok",
            "done",
        ],
    )
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "go"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    child = next(
        r for r in db.list_runs("c") if r["agent_kind"] == "subagent"
    )
    assert child["agent_definition"] == "researcher"
    assert child["definition_fingerprint"] == definition_fingerprint(
        RESEARCHER_DEFN
    )


def test_definitions_load_once_per_turn_roster_in_protocol(db):
    _seed_definition(db)
    service, chat = make_service(db, ["no tools needed"])
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    # The spawn schema (with the roster) must reach the provider call —
    # via the system prompt (fence protocol) or the tools= kwarg (native);
    # inspect the whole call to stay channel-agnostic.
    assert "researcher" in json.dumps(chat.calls[0], default=str)


def test_no_definitions_spawn_unchanged(db):
    # Guard the identity path: with an empty definitions table the primary
    # system prompt must NOT mention an 'agent' parameter.
    service, chat = make_service(db, ["plain answer"])
    service.run_turn(
        conversation_id="c",
        messages=[{"role": "user", "content": "hi"}],
        config=CFG,
        api_endpoint="llama_cpp",
    )
    assert '"agent"' not in json.dumps(chat.calls[0], default=str)
```

- [ ] **Step 2: Write the failing loop test** (append to `Tests/Agents/test_agent_runtime.py`, alongside `test_spawn_result_and_budget` at :132 — reuse that test's `run()`/deps-builder helpers)

```python
def test_spawn_passes_agent_kwarg_only_when_present():
    seen = []

    def spawn(task, **kwargs):
        seen.append((task, kwargs))
        return ToolResult(ok=True, content="ok")

    outcome = run(
        replies=[
            fence("spawn_subagent", {"task": "plain"}),
            fence("spawn_subagent", {"task": "named", "agent": "researcher"}),
            "done",
        ],
        spawn=spawn,
    )
    assert outcome.status == RUN_DONE
    assert seen[0] == ("plain", {})
    assert seen[1] == ("named", {"agent": "researcher"})
```

(Adapt the two helper names to the module's actual builders — the file already constructs `LoopDeps` with an injected `spawn` at :39-62 and scripted fence replies; mirror `test_spawn_result_and_budget`'s arrangement exactly.)

- [ ] **Step 3: Run to verify failure**

Run: `pytest Tests/Agents/test_agent_service.py -v -k "named_spawn or unknown_agent or load_once or no_definitions" && pytest Tests/Agents/test_agent_runtime.py -v -k agent_kwarg`
Expected: FAIL (schema lacks `agent`; closure rejects kwarg).

- [ ] **Step 4: Implement**

(a) `agent_runtime.py` spawn branch — after `task = str(call.args.get("task", "")).strip()` (:694) add `agent_name = str(call.args.get("agent", "")).strip()`; include it in the STEP_SPAWN summary and pass it through conditionally (existing injected `lambda task: ...` doubles must keep working):

```python
                        task = str(call.args.get("task", "")).strip()
                        agent_name = str(call.args.get("agent", "")).strip()
```
```python
                            add(
                                STEP_SPAWN,
                                summary=(
                                    f"[{agent_name}] {task}"[:200]
                                    if agent_name
                                    else task[:200]
                                ),
                                tool_name=SPAWN_TOOL_NAME,
                                args=dict(call.args),
                            )
                            if agent_name:
                                result = deps.spawn(task, agent=agent_name)
                            else:
                                result = deps.spawn(task)
                            spawned += 1
```

(b) `agent_service.py` `__init__`: add `self._turn_definitions: list[AgentDefinition] = []` (import `AgentDefinition`, `definition_from_row`, `definition_fingerprint` from `.agent_models`; `build_spawn_schema` from `.tool_catalog`).

(c) `run_turn`, immediately after `self.registry.reset_catalog_cache()` (:1442):

```python
        # Fleet spec §4: definitions load ONCE per turn — the roster the
        # model sees in the spawn schema is exactly what resolves at spawn
        # time; Settings edits affect the NEXT turn, never an in-flight one.
        self._turn_definitions = [
            definition_from_row(row)
            for row in self.db.list_agent_definitions(enabled_only=True)
        ]
```

(d) `_run_one` (:543): replace `runtime_schemas.append(SPAWN_TOOL_SCHEMA)` with `runtime_schemas.append(build_spawn_schema(self._turn_definitions))`.

(e) `_run_one` signature: add `agent_definition: str | None = None, definition_fingerprint: str | None = None` keyword params; pass both through to `self.db.create_run(...)`.

(f) The `spawn` closure (:669): new signature and resolution — resolution runs BEFORE the budget increment so a typo never consumes a slot:

```python
        def spawn(
            spawn_task: str,
            *,
            allowed_tools: tuple[str, ...] | None = None,
            agent: str | None = None,
        ) -> ToolResult:
            nonlocal sub_agent_spawns
            # (existing Task-12 single-choke-point comment stays here)
            # Fleet spec §4: the skill path (allowed_tools override) and
            # the named-definition path are disjoint by construction —
            # skills never pass `agent`.
            assert not (agent and allowed_tools is not None)
            resolved = None
            if agent:
                resolved = next(
                    (d for d in self._turn_definitions if d.name == agent),
                    None,
                )
                if resolved is None:
                    available = (
                        ", ".join(d.name for d in self._turn_definitions)
                        or "none"
                    )
                    # Refused BEFORE the budget increment: a typo costs no
                    # sub-agent slot (mirrors the loop's empty-task refusal).
                    return ToolResult(
                        ok=False,
                        error=(
                            f"unknown agent '{agent}'; available: {available}"
                        ),
                    )
            if sub_agent_spawns >= config.budget.max_subagents:
                return ToolResult(ok=False, error="sub-agent budget exhausted")
            sub_agent_spawns += 1
```

then, where the child config is built (keep the existing allow-list derivation and comments; add the narrowing and overrides):

```python
            child_allowed_tools = (
                allowed_tools
                if allowed_tools is not None
                else tuple(
                    n
                    for n in config.allowed_tools
                    if n != SPAWN_TOOL_NAME
                    and not (
                        self.skill_runner is not None
                        and self.skill_runner.is_skill_tool(n)
                    )
                )
            )
            child_system_prompt = get_internal_prompt("agents.subagent_system")
            child_model = config.model
            if resolved is not None:
                # IDENTITY CONTRACT: console_agent_bridge._is_subagent
                # prefix-matches the base prompt — instructions APPEND,
                # never prepend (fleet spec §4 composition rule).
                child_system_prompt = (
                    child_system_prompt + "\n\n" + resolved.instructions
                )
                if resolved.model:
                    child_model = resolved.model
                if resolved.tool_allowlist:
                    # Intersection, never union (spec §3 invariant 1): the
                    # definition narrows the inherited set; unknown names
                    # drop out here and can never grant.
                    wanted = set(resolved.tool_allowlist)
                    child_allowed_tools = tuple(
                        n for n in child_allowed_tools if n in wanted
                    )
            child_config = AgentConfig(
                model=child_model,
                system_prompt=child_system_prompt,
                allowed_tools=child_allowed_tools,
                budget=clamp_child_budget(config.budget, remaining),
                native_tools=config.native_tools,
            )
```

and thread the audit identity into the child run (the existing `self._run_one(...)` call inside the `with scope:` block):

```python
                _child_id, child_outcome = self._run_one(
                    conversation_id=conversation_id,
                    messages=[{"role": "user", "content": spawn_task}],
                    config=child_config,
                    api_endpoint=api_endpoint,
                    should_cancel=should_cancel,
                    agent_kind=AGENT_KIND_SUBAGENT,
                    task=spawn_task,
                    parent_run_id=run_id,
                    agent_definition=(resolved.name if resolved else None),
                    definition_fingerprint=(
                        definition_fingerprint(resolved) if resolved else None
                    ),
                )
```

(The old inline `system_prompt=get_internal_prompt(...)` line inside `AgentConfig(...)` is replaced by `child_system_prompt` above — one composition site.)

- [ ] **Step 5: Run to verify pass — including the untouched-behavior battery**

Run: `pytest Tests/Agents/test_agent_service.py Tests/Agents/test_agent_runtime.py Tests/Agents/test_skill_tool_spawn.py Tests/Agents/test_agent_service_on_step.py -v`
Expected: ALL PASS — pre-existing spawn tests (`test_spawn_creates_linked_child_with_clean_context`, `test_child_cannot_spawn`, `test_subagent_result_is_capped`, skill-spawn suite) prove the no-`agent` path unchanged.

- [ ] **Step 6: Run the bridge regression battery**

Run: `pytest Tests/Chat/test_console_agent_bridge.py -v`
Expected: ALL PASS (the `_is_subagent` prefix detection and spawn-marker tests guard the identity contract end to end).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Agents/agent_service.py tldw_chatbook/Agents/agent_runtime.py Tests/Agents/
git commit -m "feat: spawn_subagent resolves named agent definitions" && git push
```

---

### Task 6: Settings ▸ Agents category + panel

**Files:**
- Create: `tldw_chatbook/Widgets/settings_agents_panel.py`
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py:33` area (new enum member)
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py` — the seven registration points listed in Step 3
- Modify: `Tests/UI/test_settings_configuration_hub.py:191` (count 24 → 25)
- Test: Create `Tests/UI/test_settings_agents_category.py`

**Interfaces:**
- Consumes: Task 2's CRUD methods; Task 1's `AgentDefinition`/`validate_agent_definition`; `RUNTIME_TOOL_NAMES` from `agent_models`; `ToolCatalogRegistry` + `BuiltinToolProvider` from `tool_catalog`.
- Produces: `AgentsSettingsPanel(app_instance, runs_db=None)` widget (injectable DB for tests); `SettingsCategoryId.AGENTS = "agents"`.

- [ ] **Step 1: Write the failing panel unit tests** (`Tests/UI/test_settings_agents_category.py`)

```python
"""Settings ▸ Agents: category registration + panel CRUD (fleet spec §4)."""

import pytest
from textual.app import App

from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Widgets.settings_agents_panel import AgentsSettingsPanel


@pytest.fixture()
def runs_db(tmp_path):
    return AgentRunsDB(tmp_path / "agent_runs.db", client_id="test")


class PanelHarness(App):
    def __init__(self, panel):
        super().__init__()
        self._panel = panel

    def compose(self):
        yield self._panel


@pytest.mark.asyncio
async def test_panel_creates_definition_via_form(runs_db):
    panel = AgentsSettingsPanel(app_instance=None, runs_db=runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        panel.query_one("#agents-name-input").value = "researcher"
        panel.query_one("#agents-description-input").value = "Searches sources."
        panel.query_one("#agents-instructions-area").text = "Cite sources."
        await pilot.click("#agents-save-button")
        await pilot.pause()
    rows = runs_db.list_agent_definitions()
    assert [r["name"] for r in rows] == ["researcher"]


@pytest.mark.asyncio
async def test_panel_surfaces_validation_error(runs_db):
    panel = AgentsSettingsPanel(app_instance=None, runs_db=runs_db)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        panel.query_one("#agents-name-input").value = "subagent"  # reserved
        panel.query_one("#agents-instructions-area").text = "x"
        await pilot.click("#agents-save-button")
        await pilot.pause()
        status = panel.query_one("#agents-status")
        # Rendered-geometry guard, not just DOM presence (Library-UAT
        # lesson: unbounded-width Statics are invisible to headless
        # queries while "present").
        assert status.region.width > 0
        assert "reserved" in status.renderable_text
    assert runs_db.list_agent_definitions() == []


@pytest.mark.asyncio
async def test_panel_without_db_shows_notice(tmp_path):
    panel = AgentsSettingsPanel(app_instance=None, runs_db=None)
    async with PanelHarness(panel).run_test(size=(120, 40)) as pilot:
        notice = panel.query_one("#agents-no-db-notice")
        assert notice.region.width > 0
```

(`renderable_text`: if the repo's Static exposes text differently, use `str(status.render())` — match whatever `Tests/UI/test_destination_shells.py::_visible_text` does.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest Tests/UI/test_settings_agents_category.py -v`
Expected: FAIL / ImportError (no `settings_agents_panel` module).

- [ ] **Step 3: Implement the panel** (`tldw_chatbook/Widgets/settings_agents_panel.py`)

```python
"""Settings ▸ Agents: CRUD editor for named sub-agent definitions.

Edits the AgentRuns DB directly (immediate CRUD) — unlike TOML-backed
Settings categories there is no draft/Save-with-`s` cycle; each Save/Delete
applies at once. Fleet spec §4.
"""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, ListItem, ListView, Static, Switch, TextArea

from tldw_chatbook.Agents.agent_models import (
    AgentDefinition,
    RUNTIME_TOOL_NAMES,
)
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

#: Soft ceiling before the status line warns about spawn-schema bloat
#: (spec §4: every enabled definition rides the spawn tool's schema).
ENABLED_DEFINITIONS_SOFT_CAP = 20


def _derive_runs_db(app_instance) -> AgentRunsDB | None:
    """Same derivation as UI/Console_Modules/agent.py:337 — the runs DB
    lives next to the ChaChaNotes file; a :memory: ChaChaNotes (tests,
    ephemeral) means no durable definitions store."""
    db = getattr(app_instance, "chachanotes_db", None)
    db_path = getattr(db, "db_path", None) if db is not None else None
    if not db_path or str(db_path) == ":memory:":
        return None
    return AgentRunsDB(Path(db_path).parent / "agent_runs.db")


class AgentsSettingsPanel(Vertical):
    """List + form editor over the agent_definitions table."""

    def __init__(self, app_instance, runs_db: AgentRunsDB | None = None, **kwargs):
        super().__init__(**kwargs)
        self._runs_db = runs_db if runs_db is not None else _derive_runs_db(app_instance)
        self._selected_id: str | None = None

    def compose(self) -> ComposeResult:
        if self._runs_db is None:
            yield Static(
                "Agent definitions need a saved (non-temporary) profile "
                "database; none is available in this session.",
                id="agents-no-db-notice",
                classes="settings-detail-row",
            )
            return
        yield Static(
            "Named sub-agents the Console supervisor can spawn. Changes "
            "apply immediately (stored in agent_runs.db, not config.toml) "
            "and take effect on the next reply.",
            classes="settings-detail-row",
        )
        yield ListView(id="agents-definition-list")
        with VerticalScroll(id="agents-form"):
            with Horizontal(classes="settings-input-row"):
                yield Static("Name", classes="settings-input-label")
                yield Input(
                    placeholder="researcher (lowercase slug)",
                    id="agents-name-input",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static("Description", classes="settings-input-label")
                yield Input(
                    placeholder="One line the supervisor reads (max 200 chars)",
                    id="agents-description-input",
                )
            yield Static("Instructions (appended to the sub-agent prompt)",
                         classes="settings-input-label")
            yield TextArea(id="agents-instructions-area")
            with Horizontal(classes="settings-input-row"):
                yield Static("Model override", classes="settings-input-label")
                yield Input(
                    placeholder="empty = parent's model (same provider)",
                    id="agents-model-input",
                )
            with Horizontal(classes="settings-input-row"):
                yield Static(
                    "Tools (comma-separated; empty = inherit all; names "
                    "only narrow, never grant)",
                    classes="settings-input-label",
                )
                yield Input(id="agents-tools-input")
            with Horizontal(classes="settings-input-row"):
                yield Static("Enabled", classes="settings-input-label")
                yield Switch(value=True, id="agents-enabled-switch")
            with Horizontal(classes="settings-input-row"):
                yield Button("New", id="agents-new-button")
                yield Button("Save", variant="primary", id="agents-save-button")
                yield Button("Delete", variant="error", id="agents-delete-button")
        yield Static("", id="agents-status", classes="settings-detail-row")

    def on_mount(self) -> None:
        self._reload_list()

    # -- list / selection -------------------------------------------------
    def _reload_list(self) -> None:
        if self._runs_db is None:
            return
        lv = self.query_one("#agents-definition-list", ListView)
        lv.clear()
        self._rows = self._runs_db.list_agent_definitions()
        for row in self._rows:
            marker = "" if row["enabled"] else " (disabled)"
            lv.append(
                ListItem(Static(f"{row['name']}{marker}"), name=row["id"])
            )
        enabled_count = sum(1 for r in self._rows if r["enabled"])
        if enabled_count > ENABLED_DEFINITIONS_SOFT_CAP:
            self._set_status(
                f"{enabled_count} enabled definitions — every one rides the "
                "spawn schema each turn; consider disabling some."
            )

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        definition_id = event.item.name
        row = next((r for r in self._rows if r["id"] == definition_id), None)
        if row is None:
            return
        self._selected_id = definition_id
        self.query_one("#agents-name-input", Input).value = row["name"]
        self.query_one("#agents-description-input", Input).value = row["description"]
        self.query_one("#agents-instructions-area", TextArea).text = row["instructions"]
        self.query_one("#agents-model-input", Input).value = row["model"]
        self.query_one("#agents-tools-input", Input).value = ", ".join(
            row["tool_allowlist"]
        )
        self.query_one("#agents-enabled-switch", Switch).value = bool(row["enabled"])

    # -- buttons ----------------------------------------------------------
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "agents-new-button":
            self._clear_form()
        elif event.button.id == "agents-save-button":
            self._save()
        elif event.button.id == "agents-delete-button":
            self._delete()

    def _clear_form(self) -> None:
        self._selected_id = None
        self.query_one("#agents-name-input", Input).value = ""
        self.query_one("#agents-description-input", Input).value = ""
        self.query_one("#agents-instructions-area", TextArea).text = ""
        self.query_one("#agents-model-input", Input).value = ""
        self.query_one("#agents-tools-input", Input).value = ""
        self.query_one("#agents-enabled-switch", Switch).value = True
        self._set_status("")

    def _form_definition(self) -> AgentDefinition:
        tools = tuple(
            name.strip()
            for name in self.query_one("#agents-tools-input", Input).value.split(",")
            if name.strip() and name.strip() not in RUNTIME_TOOL_NAMES
        )
        return AgentDefinition(
            name=self.query_one("#agents-name-input", Input).value.strip(),
            description=self.query_one(
                "#agents-description-input", Input
            ).value.strip(),
            instructions=self.query_one(
                "#agents-instructions-area", TextArea
            ).text.strip(),
            tool_allowlist=tools,
            model=self.query_one("#agents-model-input", Input).value.strip(),
            enabled=self.query_one("#agents-enabled-switch", Switch).value,
        )

    def _save(self) -> None:
        try:
            defn = self._form_definition()
            if self._selected_id is None:
                self._runs_db.create_agent_definition(defn)
            else:
                self._runs_db.update_agent_definition(self._selected_id, defn)
        except ValueError as exc:
            self._set_status(str(exc))
            return
        self._set_status(f"Saved '{defn.name}'.")
        self._reload_list()

    def _delete(self) -> None:
        if self._selected_id is None:
            self._set_status("Select a definition to delete.")
            return
        self._runs_db.soft_delete_agent_definition(self._selected_id)
        self._clear_form()
        self._set_status("Deleted.")
        self._reload_list()

    def _set_status(self, text: str) -> None:
        self.query_one("#agents-status", Static).update(text)
```

Note for the implementer: `renderable_text` in the tests should read the Static's content the same way existing UI tests do (`Tests/UI/test_destination_shells.py::_visible_text` at :852 is the repo idiom — use that helper if direct attribute access differs).

- [ ] **Step 4: Run panel tests to verify pass**

Run: `pytest Tests/UI/test_settings_agents_category.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Register the category** (all in the two settings modules; mirror the About precedent, commit `f32f21f24`)

1. `settings_config_models.py` — add `AGENTS = "agents"` to `SettingsCategoryId` (StrEnum at :10).
2. `settings_screen.py` `_category_summaries()` (:2374) — add:
   ```python
   SettingsCategorySummary(
       SettingsCategoryId.AGENTS,
       "Agents",
       "Named sub-agent definitions the Console supervisor can spawn.",
       "Local",
   ),
   ```
3. `settings_screen.py` `_category_groups()` (:2550) — add `SettingsCategoryId.AGENTS` to the group tuple that contains the Console/chat-adjacent categories; if no such group reads naturally, add it to the same group as `ABOUT` (:2581). One tuple membership, or the button never renders.
4. `settings_screen.py` `_INSPECTOR_GUIDANCE` (:1080) — add an entry (CI enforces presence for non-domain categories via `Tests/UI/test_settings_configuration_hub.py:195-216`):
   ```python
   SettingsCategoryId.AGENTS.value: (
       ("Affected config", "agent_definitions table in agent_runs.db (DB, not config.toml)"),
       ("Recovery", "definitions are soft-deleted; re-create or re-enable from this screen"),
       ("Boundary", "definitions only narrow a sub-agent's tools — [tools] gates and permission cards still apply"),
   ),
   ```
   (Match the surrounding entries' exact tuple shape — read two neighbors first.)
5. `settings_screen.py` `_category_ownership_records()` (:2694) — add a `SettingsOwnershipRecord` mirroring the ABOUT record's shape (:2898-2917) with `writes_allowed=True` and scope text `"agent_runs.db (SQLite) — immediate CRUD, no draft"`.
6. `settings_screen.py` `_render_detail_pane()` — add before the domain-category `elif` (:12400):
   ```python
   elif category is SettingsCategoryId.AGENTS:
       yield AgentsSettingsPanel(
           self.app_instance, id="settings-agents-panel"
       )
   ```
   with the import `from tldw_chatbook.Widgets.settings_agents_panel import AgentsSettingsPanel` alongside the other panel imports (~:46).
7. `settings_screen.py` `_persistence_badge()` (:4637) and `_category_state_scope_text()` (:4663) — add AGENTS branches returning `"Applies immediately"` / the scope text from point 5 (read the ImageGen branches at :4649-4650 for the exact return shape).
8. `Tests/UI/test_settings_configuration_hub.py:191` — bump the pinned category count `== 24` to `== 25` (deliberate, this is the guard working).

- [ ] **Step 6: Add the category-level test** (append to `Tests/UI/test_settings_agents_category.py`)

```python
@pytest.mark.asyncio
async def test_agents_category_renders_in_settings_screen():
    # The category sweep (test_settings_category_sweep.py) already visits
    # every category; this pins OUR panel specifically: selecting Agents
    # renders either the editor or the no-DB notice (test app runs with a
    # :memory: ChaChaNotes, so the notice is the expected branch).
    import Tests.UI.test_settings_category_sweep as sweep

    app = sweep._build_test_app()
    host = sweep.DestinationHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await sweep._settle_settings(pilot)
        await sweep._click_settings_category(pilot, "agents")
        screen = sweep._active_destination_screen(host)
        assert screen.query("#settings-agents-panel") or screen.query(
            "#agents-no-db-notice"
        )
```

(If those helpers are private to the module in an awkward way, import from where they actually live — the explorer mapped them to `test_settings_category_sweep.py:32-77` and `test_destination_shells.py:825-852`; adjust the import site, not the assertion.)

- [ ] **Step 7: Run the settings battery**

Run: `pytest Tests/UI/test_settings_agents_category.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_category_sweep.py -v`
Expected: ALL PASS — the sweep proves every category (including Agents) renders at 120x35 AND 80x24.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/settings_agents_panel.py tldw_chatbook/UI/Screens/settings_config_models.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/
git commit -m "feat: Settings ▸ Agents definitions editor" && git push
```

---

### Task 7: Docs, spec sync, full battery, live verification

**Files:**
- Modify: `Docs/User_Guide/settings.md` (new Agents section + stamp)
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md` (named-agents subsection + stamp)
- Modify: `Docs/superpowers/specs/2026-08-08-supervisor-agent-fleet-design.md` (precedent correction, if not already applied)
- Modify: backlog task files (AC ticks + Implementation Notes)

- [ ] **Step 1: User Guide — settings.md**

Add an "Agents" category section (mirror the About section's structure): what a definition is, the append-composition rule in user terms ("your instructions are added to the built-in sub-agent prompt"), the narrowing-only tools note, model override = same provider, immediate persistence. Update the trailing stamp per the convention (`*Verified against dev @ <short-sha> — 2026-08-09*` — append a clause if the page carries multi-verification history, cf. `Docs/User_Guide/settings/rag.md:255`).

- [ ] **Step 2: User Guide — console/agent-runs-and-tools.md**

Add a short "Named agents" subsection under the sub-agents material: the supervisor can spawn a named definition; where to create them (Settings ▸ Agents); the run log records which definition ran. Update the stamp line.

- [ ] **Step 3: Targeted full battery + collect-only sweep** (owner ruling: branch-relevant files + a collect-only sweep, never routine full `Tests/UI` runs)

```bash
pytest Tests/Agents/ Tests/DB/test_agent_runs_db.py \
  Tests/UI/test_settings_agents_category.py \
  Tests/UI/test_settings_configuration_hub.py \
  Tests/UI/test_settings_category_sweep.py \
  Tests/Chat/test_console_agent_bridge.py -q
pytest --collect-only -q > /dev/null && echo COLLECT-OK
```
Expected: all PASS with a read, nonzero count; `COLLECT-OK` (no import breakage anywhere).

- [ ] **Step 4: Live TUI verification** (per `backlog/docs/lessons-live-verification.md`; repo-root `*-api-key.txt` files are FOR agent use)

tmux recipe from the dev-environment memory: launch the app against a scratch config (`TLDW_CONFIG_PATH` to a scratch copy — never the live config), with a real provider key. Verify, capturing panes as evidence:
1. Settings ▸ Agents: create `researcher` (instructions "Always cite sources", no tool list).
2. Console: send a message that asks the assistant to delegate to the researcher agent; confirm the reply's sub-agent marker renders.
3. `sqlite3 <profile>/agent_runs.db "SELECT agent_definition, definition_fingerprint FROM agent_runs WHERE agent_kind='subagent' ORDER BY created_at DESC LIMIT 1"` shows `researcher` + a 16-char hash.
4. Disable the definition in Settings; next Console turn's spawn schema no longer offers it (ask the model to list its available agents, or check the run log's rendered protocol).

- [ ] **Step 5: Backlog close-out**

Tick the PR-1 task's ACs, add Implementation Notes (approach, files, deviations), set status per the DoD. Do NOT mark Done if any AC is unticked or any battery step was skipped.

- [ ] **Step 6: Final commit + push**

```bash
git add Docs/ backlog/
git commit -m "docs: user guide + spec sync for agent definitions" && git push
```

PR creation happens via superpowers:finishing-a-development-branch — target branch `dev`, title "feat: named agent definitions (supervisor fleet PR 1)".

---

## Self-review notes (already applied)

- Spec §4 coverage: model+caps (Task 1), table+CRUD+v5 (Task 2), audit columns (Task 3), schema enum+roster+identity (Task 4), resolution/composition/intersection/model-override/load-once/unknown-agent (Task 5), Settings editor incl. runtime-name exclusion + soft cap (Task 6), docs (Task 7). Deferred per spec: no seeding, no per-definition budgets, no export/import.
- The spec's original "tools_settings_screen.py precedent" line is corrected in the spec itself (deprecated TASK-1346, nav-unreachable; live precedent = About category + panel widget).
- Type consistency: `AgentDefinition` field names identical across Tasks 1/2/5/6; `list_agent_definitions` returns dicts (DB) which `definition_from_row` converts (service); the panel consumes dicts directly.
- Unknown-agent refusal deliberately precedes the budget increment (tested by `test_unknown_agent_refused_without_burning_budget`).
