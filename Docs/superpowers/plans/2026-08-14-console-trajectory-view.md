# Console Trajectory View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A trajectory (trace) screen over Console conversations: turn-grouped event ledger with tool-call nesting, per-record token/timing inspector, search, and live tail-follow.

**Architecture:** Schema v38 local-only sidecar table `message_trajectory_metadata` is the sole persisted home for turn identity, per-record timing, and tool records (TOOL messages are deliberately never persisted to `messages`). A pure projection module folds messages + usage + sidecar + variants + compaction into a `TrajectorySnapshot`; a Console-launched screen renders it. Brushable timeline is a follow-up task.

**Tech Stack:** Python 3.11, Textual 8.x DataTable, SQLite (existing migration runner pattern), pytest.

**Spec:** `Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md` — read it first; this plan argues from it.
**ADR:** `backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`

## Global Constraints

- Sidecar table is **local-only**: never added to sync triggers, never serialized to sync payloads (same rule as `usage_json`/`metadata_json`, ADR-010).
- Never violate the TOOL-marker invariant: tool records go **only** in the sidecar.
- Never fabricate timing: NULL timing renders blank.
- Keybindings per ADR-031: single-letter htop-style, no terminal-convention keys; footer hints 1:1 with implemented actions.
- All SQL parameterized; `seq` assigned inside the same transaction as the insert.
- Package root is `tldw_chatbook/tldw_chatbook/`; tests under `tldw_chatbook/Tests/`.

---

### Task 1: Schema v38 migration + sidecar DB accessors

**Files:**
- Create: `tldw_chatbook/DB/migrations/chachanotes_v37_to_v38_message_trajectory_metadata.sql`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py` (bump `_CURRENT_SCHEMA_VERSION` 37→38; add `_migrate_from_v37_to_v38` next to `_migrate_from_v36_to_v37` at ~line 4964; add accessor methods near `update_message_usage_local` at ~line 9778)
- Test: `tldw_chatbook/Tests/DB/test_chachanotes_trajectory_metadata_migration.py`

**Interfaces:**
- Consumes: existing migration-runner pattern (`_get_db_version`, `SchemaError`, `self.transaction()`); `update_message_usage_local` precedent for local-only columns.
- Produces (Task 2/3 consume):
  - `upsert_trajectory_rows(rows: Sequence[TrajectoryRowWrite]) -> None` where `TrajectoryRowWrite` is a dataclass `(message_id: str, conversation_id: str, turn_id: str, seq: int, event_kind: str, step_started_at: float | None, first_token_at: float | None, completed_at: float | None, model: str | None, provider: str | None, payload_json: str | None)`. Seq assignment: rows written with `seq=None` get `max(seq)+1` per conversation inside one transaction; explicit seqs are honored (upsert-by-seq: `INSERT ... ON CONFLICT(message_id, event_kind, seq) DO UPDATE`).
  - `get_trajectory_rows(conversation_id: str) -> list[TrajectoryRowRead]` ordered by `seq` (includes rows whose message was soft-deleted; the projection filters).
  - `get_next_trajectory_seq(conversation_id: str) -> int` (used inside transactions by the store).

- [ ] **Step 1: Write the failing migration test** (NOTE: `db.get_schema_version`/`db.create_conversation`/`db.add_message`/`db.execute` below are illustrative — match the real signatures in `ChaChaNotes_DB.py` and existing `Tests/DB/test_chachanotes_*_migration.py` files)

```python
# tldw_chatbook/Tests/DB/test_chachanotes_trajectory_metadata_migration.py
import pytest
from tldw_chatbook.DB.ChaChaNotes_DB import ChaChaNotesDB, SchemaError

def test_migrates_v37_to_v38_and_creates_table(tmp_path):
    db = ChaChaNotesDB(tmp_path / "test.db", client_id="test")
    assert db.get_schema_version() == 38  # or equivalent version accessor
    cols = {r["name"] for r in db.execute("PRAGMA table_info(message_trajectory_metadata)")}
    assert {"message_id", "conversation_id", "turn_id", "seq", "event_kind",
            "step_started_at", "first_token_at", "completed_at",
            "model", "provider", "payload_json"} <= cols
    idx = {r["name"] for r in db.execute("PRAGMA index_list(message_trajectory_metadata)")}
    assert any("conv_seq" in n for n in idx)

def test_upsert_and_read_roundtrip(tmp_path):
    db = ChaChaNotesDB(tmp_path / "test.db", client_id="test")
    conv = db.create_conversation(name="t")  # match existing factory signature
    msg = db.add_message(conversation_id=conv, sender="user", content="hi")
    db.upsert_trajectory_rows([TrajectoryRowWrite(
        message_id=msg, conversation_id=conv, turn_id=msg, seq=None,
        event_kind="user", step_started_at=1.0, first_token_at=None,
        completed_at=None, model=None, provider=None, payload_json=None)])
    rows = db.get_trajectory_rows(conv)
    assert len(rows) == 1 and rows[0].seq == 1 and rows[0].event_kind == "user"
    # multiple tool_calls under one assistant message: distinct seqs
    db.upsert_trajectory_rows([
        TrajectoryRowWrite(message_id=msg, conversation_id=conv, turn_id=msg, seq=None,
                           event_kind="tool_call", step_started_at=1.0, first_token_at=None,
                           completed_at=2.0, model="m", provider="p", payload_json='{"n":1}'),
        TrajectoryRowWrite(message_id=msg, conversation_id=conv, turn_id=msg, seq=None,
                           event_kind="tool_call", step_started_at=1.0, first_token_at=None,
                           completed_at=3.0, model="m", provider="p", payload_json='{"n":2}'),
    ])
    rows = db.get_trajectory_rows(conv)
    assert [r.seq for r in rows] == [1, 2, 3]
```

- [ ] **Step 2: Run it to verify it fails** — `pytest Tests/DB/test_chachanotes_trajectory_metadata_migration.py -v`; expect failures (version still 37, table missing).
- [ ] **Step 3: Write the migration SQL** — table + indexes exactly as in the spec's schema block, ending with `UPDATE db_schema_version SET version = 38;` (copy the header-comment style of `chachanotes_v36_to_v37_provider_continuation.sql`).
- [ ] **Step 4: Add the runner** — `_migrate_from_v37_to_v38` mirroring `_migrate_from_v36_to_v37` (version guard → PRAGMA table_info idempotence check on `message_trajectory_metadata` → statement-split execution via `sqlite3.complete_statement`), registered in the migration dispatch chain; bump `_CURRENT_SCHEMA_VERSION` to 38. Define `TrajectoryRowWrite`/`TrajectoryRowRead` dataclasses in `ChaChaNotes_DB.py` and implement `upsert_trajectory_rows`/`get_trajectory_rows`/`get_next_trajectory_seq` with parameterized SQL and transactional seq assignment.
- [ ] **Step 5: Run tests** — the new file passes; also run `pytest Tests/DB/ -q` for migration regressions.
- [ ] **Step 6: Commit** — `git commit -m "feat(db): schema v38 message_trajectory_metadata sidecar + accessors"`

### Task 2: Capture timing + tool records in the Console persistence seam

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` (persist-time hook where `turn_id` is assigned, ~lines 4521/4604; tool-marker append path ~line 2112)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (`_stream_assistant_response_inner` ~line 8939: stamp `step_started_at` before the provider call and `first_token_at` on first chunk; `_attach_stream_usage` ~line 9487: stamp `completed_at` and write sidecar rows via `upsert_trajectory_rows`)
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` (thread the writes through the same seam as `update_message_usage`)
- Test: `tldw_chatbook/Tests/Chat/test_trajectory_capture.py`

**Interfaces:**
- Consumes: Task 1's `upsert_trajectory_rows`/`TrajectoryRowWrite`; `ConsoleChatMessage.turn_id`, `usage`, `tool_output_full`.
- Produces: every persisted Console message gets a `user`/`assistant` sidecar row; every tool marker gets `tool_call` and/or `tool_result` rows keyed to its parent assistant message with `payload_json = json.dumps({"name": ..., "args": ..., "result": full_output})`. Timing fields populated for streamed assistant rows.

- [ ] **Step 1: Write failing tests** — unit tests with an in-memory store + temp DB: (a) persisted user message produces a `user` row with `turn_id == message.turn_id`; (b) tool marker append produces `tool_call` + `tool_result` rows with full payload and parent assistant `message_id`; (c) a simulated stream (fake chunk callback) yields `first_token_at` set and `first_token_at - step_started_at > 0`, `completed_at >= first_token_at`; (d) concurrent `upsert_trajectory_rows` calls (two threads, same conversation) produce unique seqs.
- [ ] **Step 2: Verify they fail.**
- [ ] **Step 3: Implement** — timing stamps as module-level mutable capture attached to the turn in the controller (plain dict on the streamSignals-like object passed to `_attach_stream_usage`); sidecar writes batched at finalize (same place usage is attached) so one turn = one upsert call; tool payload written at marker-append time using the existing `tool_output_full` argument. **Cap the stored result** at 256 KiB with a `{"truncated": true}` marker in `payload_json` (full output remains available live in-session via `tool_output_full`). Guard all writes in try/except that logs with context and never fails the turn.
- [ ] **Step 4: Run** — new tests pass; `pytest Tests/Chat/ -q` green.
- [ ] **Step 5: Commit** — `git commit -m "feat(console): capture trajectory timing and tool records to sidecar"`

### Task 3: Pure projection module

**Files:**
- Create: `tldw_chatbook/Chat/trajectory.py`
- Test: `tldw_chatbook/Tests/Chat/test_trajectory_projection.py`

**Interfaces:**
- Consumes: message rows from DB (or `ConsoleChatMessage` models), `ProviderUsage.from_json(usage_json)`, `TrajectoryRowRead`, variant sets, compaction records from `console_context_repository`, and the conversation's **`active_leaf_message_id`** (local-only column from the v23→24 migration). Active path = walk `parent_message_id` from that leaf to the root; siblings off that chain are variants.
- Produces:

```python
@dataclass(frozen=True)
class TrajectoryRecord:
    seq: int
    kind: str                    # user | assistant | tool_call | tool_result | compaction
    turn_id: str
    message_id: str | None
    content_preview: str         # first 120 chars, single line
    usage: ProviderUsage | None
    step_started_at: float | None
    first_token_at: float | None
    completed_at: float | None
    model: str | None
    provider: str | None
    payload: dict | None         # tool records only
    variants: tuple[str, ...]    # superseded variant contents, active-path rendering
    depth: int                   # 0 = top-level, 1 = tool record under assistant step

@dataclass(frozen=True)
class TrajectoryTurn:
    turn_id: str
    records: tuple[TrajectoryRecord, ...]

@dataclass(frozen=True)
class TrajectorySnapshot:
    turns: tuple[TrajectoryTurn, ...]

def derive_trajectory(messages, usage_by_id, traj_rows, variant_sets, compaction_records) -> TrajectorySnapshot: ...
```

- [ ] **Step 1: Write failing tests** — grouping (turn starts at each `user` record), tool nesting (`depth=1`, ordered by seq under the owning assistant record), NULL timing → None fields (never derived/fabricated), seq tie-break ordering, soft-deleted messages excluded, variants surfaced on the owning record not as rows, compaction records rendered as between-turn records with `message_id=None`, empty conversation → empty snapshot.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** `derive_trajectory` — pure stdlib only, no Textual/DB imports; takes plain sequences. Legacy fallback: messages without sidecar rows group by timestamp adjacency (same calendar second as the preceding user message joins its turn).
- [ ] **Step 4: Run tests** — pass; `pytest Tests/Chat/test_trajectory_projection.py -q`.
- [ ] **Step 5: Commit** — `git commit -m "feat(chat): trajectory projection module"`

### Task 4: Trajectory screen (ledger, collapse, inspector, search)

**Files:**
- Create: `tldw_chatbook/UI/Screens/trajectory_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (launch binding only — see Task 5 for the live wiring if split)
- Test: `tldw_chatbook/Tests/UI/test_trajectory_screen.py`

**Interfaces:**
- Consumes: `TrajectorySnapshot` from Task 3; `register_footer_shortcuts`/`clear_footer_shortcuts` pattern from `evals_screen.py`.
- Produces: `TrajectoryScreen(screen_title=..., conversation_id=...)` — App.push_screen-compatible Screen with `BINDINGS` for `escape` (dismiss), `t` (collapse/expand turn), `i` (inspector toggle), `/` (search focus), `enter` (open inspector on cursor row).

- [ ] **Step 1: Write failing tests** (Textual pilot, pattern from existing UI tests): mounting with a snapshot renders one row per record plus turn header rows; `t` toggles collapse of the focused turn (child rows hidden); cursor-on-row + `enter` shows inspector pane with usage breakdown (uncached input / cache read / cache write / output), timing (start → first token → completed, blank when NULL), model/provider, full tool payload for tool records; `/` + query filters rows (turn header survives if any child matches); ADR-031 governance test passes (footer hints 1:1 with bindings).
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** — vertical layout: search Input on top, DataTable (`cursor_type="row"`) middle, inspector LogRich/Static bottom (hidden until toggled). Render newest page first: if `len(records) > 500`, mount the last 500 and put a "load earlier" key (`e`) row at top. No worker requirement at this size; large-conversation loading moves to a worker (`run_worker`) when records > 5000.
- [ ] **Step 4: Run** — new tests pass; run the ADR-031 footer suite (`pytest Tests/UI/ -q -k footer or keybinding`).
- [ ] **Step 5: Commit** — `git commit -m "feat(ui): trajectory screen with ledger, inspector, search"`

### Task 5: Launch from Console + live tail-follow

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` or the snapshot-revision bus the Console already uses (`_bump_payload_revision`) — subscribe-and-refresh
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (single-letter binding, ADR-031-legal, to push `TrajectoryScreen`; footer hint registration)
- Modify: `tldw_chatbook/UI/Screens/trajectory_screen.py` (follow logic)
- Test: `tldw_chatbook/Tests/UI/test_trajectory_live.py`

**Interfaces:**
- Consumes: a **public revision getter on the store** — `get_payload_revision(session_id) -> int` (expose the existing `_payload_revisions` counter; there is no observer bus, and streaming does not bump it, so the screen must ALSO treat `len(session messages)` as part of its revision check, or bump `_bump_payload_revision` in the Task 2 write path — do the latter: make the trajectory write path call `_bump_payload_revision(session_id)` so the counter moves on every trajectory-visible change).
- Produces: trajectory refreshes on revision; follows tail (scroll to bottom) unless the user scrolled up; footer action `f` re-enables follow.

- [ ] **Step 1: Write failing tests** — appending a message to the open conversation updates the open trajectory screen; scrolling up suspends follow (new records do not scroll); `f` resumes follow.
- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** — `TrajectoryScreen` polls `get_payload_revision(conversation_id)` via `set_interval(0.5)`; on change, recompute the snapshot in a worker; DataTable diffed by row key `(seq)` so selection survives refresh; escape pops the screen and clears footer shortcuts.
- [ ] **Step 4: Run** — new tests pass; `pytest Tests/UI/ Tests/Chat/ -q`.
- [ ] **Step 5: Full suite + commit** — `pytest` (whole suite), then `git commit -m "feat(console): trajectory screen launch + live tail-follow"`.

### Task 6: Docs + backlog hygiene

**Files:**
- Modify: `Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md` (status → implemented), ADR-066 if deviations occurred.
- Backlog task files updated (Implementation Notes, ACs checked) via `backlog task edit`.

- [ ] **Step 1:** Verify every AC in the backlog tasks; write Implementation Notes; mark Done via CLI.
- [ ] **Step 2:** Add a lessons entry ONLY if something non-obvious was learned (e.g., TOOL-marker invariant interplay).
- [ ] **Step 3:** Commit docs.

---

**Follow-up (separate backlog task, not in this plan):** brushable/zoomable timeline strip widget using `seq` + timestamps, with brush-to-filter synced to the ledger.
