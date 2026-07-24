# Console Conversation Branching — Phase C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** On resume, anchor each agent run's inline TOOL markers to the assistant reply that run actually produced (by durable persisted id), and **hide** markers whose reply is off the active branch — replacing the ordinal Nth-run→Nth-assistant placement that is wrong once conversations branch (Phases A/B).

**Architecture (durable id-anchoring, per design decision 2026-07-24):** The correct anchor is the assistant reply's **persisted** ChaChaNotes id. That id only exists *after* the run completes and the controller marks the reply complete (the empty assistant node persists lazily). So: `AgentRunsDB` gains an `assistant_message_id` column (v1→v2, via a hand-written idempotent `ALTER` — there is no migration framework); `run_reply` returns the run id; the **controller**, after `mark_message_complete`, writes the reply's *persisted* id onto the run (`set_run_assistant_message_id`); and resume placement matches `message.persisted_message_id == run.assistant_message_id`, hiding off-path runs and falling back to the legacy ordinal placement for runs whose id is `NULL` (pre-Phase-C data).

**Tech Stack:** Python ≥3.11, SQLite (AgentRunsDB, no ORM/migration framework), Textual, pytest.

## Global Constraints

- **Design source:** foundation spec `Docs/superpowers/specs/2026-07-22-console-conversation-branching-foundation-design.md` (Phase C) + the 2026-07-24 decision to use **durable id-anchoring via a post-completion persisted-id write** (NOT threading the native id through `create_run`, which stores an id that never matches on resume).
- **The stored id MUST be the persisted (ChaChaNotes) id**, not the native in-memory session id. Native id at `create_run` time is useless for resume — the load-bearing write is the controller's post-`mark_message_complete` update.
- **`AgentRunsDB` has NO migration framework** — only `CREATE TABLE IF NOT EXISTS`. A new column requires an explicit **PRAGMA-guarded idempotent `ALTER TABLE agent_runs ADD COLUMN assistant_message_id TEXT`** inside `_initialize_schema`, in addition to adding it to the CREATE TABLE DDL (for fresh DBs). Bump `_CURRENT_SCHEMA_VERSION` to 2 (currently cosmetic).
- **`assistant_message_id` is optional everywhere** (`create_run`, `run_turn`, `_run_one` default `None`) so every existing call site + test stays green.
- **Backward-compat:** runs with `assistant_message_id IS NULL` (pre-Phase-C, and sub-agent runs) fall back to the existing ordinal placement; only runs with a set id are id-anchored. Sub-agent runs never carry the id (only the primary run produces a transcript reply).
- **Off-path = hidden:** a run whose set `assistant_message_id` is not among the resumed active-path messages' `persisted_message_id`s → its block is dropped (not appended). Preserve the content-based idempotency guard.
- **Tests:** real SQLite / real store+bridge. Run via `./.venv/bin/python -m pytest`. Baseline pre-existing failures (`test_anthropic_native_tools`, `test_chat_functions`, `test_console_native_chat_flow` continue/regenerate "Text not found: 'hello'") — ignore.
- **No `git stash`** in any subagent (worktrees share a stash stack). Commit after each task; `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- **Modify** `tldw_chatbook/DB/AgentRuns_DB.py` — v1→v2 column + guarded ALTER + `create_run` param + `set_run_assistant_message_id`.
- **Modify** `tldw_chatbook/Agents/agent_service.py` — thread `assistant_message_id` through `run_turn`/`_run_one` into `create_run`; sub-agent recursion passes `None`.
- **Modify** `tldw_chatbook/Chat/console_agent_bridge.py` — `run_reply` passes the id to `run_turn` and **returns the run id**; `resume_marker_messages` returns `(assistant_message_id, block)` pairs; rewrite `inject_resume_agent_markers` (id-anchor + off-path-hide + null-ordinal-fallback + idempotency).
- **Modify** `tldw_chatbook/Chat/console_chat_controller.py` — after `mark_message_complete`, write the reply's persisted id onto the run.
- **Modify** `tldw_chatbook/UI/Screens/chat_screen.py` — `_inject_resume_agent_markers` passes the new block shape through (the `apply_resume_marker_overlay` overlay is unchanged; transcript messages already carry `persisted_message_id`).

Verified facts (research 2026-07-24, worktree base dev 922308e44): `AgentRuns_DB._CURRENT_SCHEMA_VERSION=1`, `_initialize_schema` is `CREATE TABLE IF NOT EXISTS` only; `create_run` INSERT at `AgentRuns_DB.py:117-159`; `list_runs`/`get_run` use `SELECT *` + `_row_to_dict` (new column auto-surfaced); `run_turn` `agent_service.py:585`, `_run_one` `:313`, the single `create_run` call `:325`, sub-agent recursion `:484`; `run_reply` `console_agent_bridge.py:832` (already has `assistant_message_id` arg), `service.run_turn(...)` call `:1008` (id NOT passed, `_run_id` discarded); `resume_marker_messages` `:1118`; `inject_resume_agent_markers` ordinal `zip` `:196`; screen seam `chat_screen.py:4440` + overlay call `:4706`; `ConsoleChatMessage.persisted_message_id` `console_chat_models.py:206`.

---

### Task 1: AgentRunsDB v1→v2 — `assistant_message_id` column + guarded ALTER + accessor

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py`
- Test: `Tests/DB/test_agent_runs_db.py` (extend)

**Interfaces:**
- Produces: `assistant_message_id TEXT` on `agent_runs`; `create_run(..., assistant_message_id: str | None = None)`; `set_run_assistant_message_id(run_id: str, assistant_message_id: str | None) -> None`; the id surfaced in `get_run`/`list_runs` dicts (via `SELECT *`).

- [ ] **Step 1: Write failing tests** (real `AgentRunsDB(tmp_path/"agent_runs.db", client_id="test")`):
  1. `create_run(..., assistant_message_id="m-9")` → `get_run(run_id)["assistant_message_id"] == "m-9"`; and default omitted → `None`.
  2. `set_run_assistant_message_id(run_id, "p-42")` → `get_run` reflects `"p-42"`; `list_runs` includes it.
  3. **Migration on an existing v1 DB:** create the db (schema initialized), then simulate a pre-v2 table by dropping the column is hard — instead: open a second `AgentRunsDB` on the SAME file and confirm `create_run(..., assistant_message_id="x")` works (i.e. the guarded ALTER is idempotent across re-open and the column persists). Additionally, construct a raw v1 table (a temp sqlite file with the old 11-column `agent_runs` DDL, no `assistant_message_id`), then open `AgentRunsDB` on it and assert `create_run(..., assistant_message_id="y")` succeeds and round-trips (proving the ALTER ran on legacy data).
- [ ] **Step 2: Run RED** — `./.venv/bin/python -m pytest Tests/DB/test_agent_runs_db.py -v` (new tests fail: no column / no method).
- [ ] **Step 3: Implement.** In `_initialize_schema`: add `assistant_message_id TEXT` to the `CREATE TABLE agent_runs` column list; AFTER the executescript, run a guarded ALTER: read `PRAGMA table_info(agent_runs)`; if `"assistant_message_id"` absent, `conn.execute("ALTER TABLE agent_runs ADD COLUMN assistant_message_id TEXT")`. Bump `_CURRENT_SCHEMA_VERSION = 2` and change the seed to keep it consistent (harmless; nothing reads it). Add `assistant_message_id: str | None = None` to `create_run` + its INSERT (column, placeholder, param). Add `set_run_assistant_message_id(run_id, assistant_message_id)` — an `UPDATE agent_runs SET assistant_message_id = ?, updated_at = ? WHERE id = ?` in a transaction.
- [ ] **Step 4: Run GREEN** + the full `Tests/DB/test_agent_runs_db.py` (all pre-existing create_run calls that omit the kwarg must still pass).
- [ ] **Step 5: Commit** — `feat(agents): agent_runs.assistant_message_id column (v1->v2) + setter`.

---

### Task 2: Write-path — thread the id + return the run id, then write the PERSISTED id from the controller

**Files:**
- Modify: `tldw_chatbook/Agents/agent_service.py` (`run_turn` :585, `_run_one` :313 + create_run call :325, sub-agent recursion :484)
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`run_reply` :832 — pass id to `run_turn` :1008, return the run id)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` (the agent path in `_stream_assistant_response`/`_complete_agent_message`, after `mark_message_complete`)
- Test: `Tests/Agents/test_agent_service.py` / `Tests/Chat/test_console_agent_bridge.py` + a controller test

**Interfaces:**
- Consumes: Task 1's `create_run(assistant_message_id=…)` + `set_run_assistant_message_id`.
- Produces: `run_turn(..., assistant_message_id=None) -> (run_id, RunOutcome)`; `_run_one(..., assistant_message_id=None)`; `run_reply(...)` returns the primary run id alongside/within its result (see below); controller writes the persisted id post-completion.

- [ ] **Step 1: Write failing tests.**
  - Service: `run_turn(..., assistant_message_id="a1")` → the created primary run's `assistant_message_id == "a1"` (via a real `AgentRunsDB`); a spawned sub-agent run has `assistant_message_id is None`.
  - Bridge/controller: after a full agent reply completes on a persisted store, the primary run's `assistant_message_id` equals the assistant message's **persisted_message_id** (not the native id).
- [ ] **Step 2: Run RED.**
- [ ] **Step 3: Implement.**
  - `agent_service.py`: add `assistant_message_id: str | None = None` to `run_turn` and `_run_one`; pass it into the single `create_run(...)` call; the `spawn` recursion passes `assistant_message_id=None`.
  - `console_agent_bridge.py`: pass `assistant_message_id=assistant_message_id` in the `service.run_turn(...)` call; capture the returned `run_id`. **Expose the primary run id to `run_reply`'s caller** — the least-invasive way (choose one and note it): (a) store it on a bridge field keyed by session/assistant id that the controller reads, or (b) change `run_reply` to return `(run_id, outcome)` and update its call site(s) + the `run_reply` fakes in `test_console_agent_swap.py`. Prefer (b) if the call sites are few; the id must reach the controller.
  - `console_chat_controller.py`: in the agent completion path, AFTER `store.mark_message_complete(assistant_message_id)` (which assigns the persisted id), resolve `persisted = store.get_message(assistant_message_id).persisted_message_id` and, if a run id and persisted id exist, call `agent_runs_db.set_run_assistant_message_id(run_id, persisted)`. Guard for the no-persistence / no-run cases (no-op). Note the native `create_run` value gets corrected to the persisted id here — this is the load-bearing write.
- [ ] **Step 4: Run GREEN** + the agent bridge/service suites + `test_console_agent_swap.py` (fakes still valid).
- [ ] **Step 5: Commit** — `feat(console): record the produced assistant reply's persisted id on the agent run`.

---

### Task 3: Read-path — id-anchored resume placement (off-path hidden, null → ordinal fallback)

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`resume_marker_messages` :1118, `inject_resume_agent_markers` :196)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (`_inject_resume_agent_markers` :4440 — pass the new shape through)
- Test: `Tests/Chat/test_console_agent_bridge.py` (rewrite the placement tests), `Tests/UI/test_console_native_chat_flow.py` (the resume-wiring test)

**Interfaces:**
- Produces: `resume_marker_messages(conversation_id) -> list[tuple[str | None, list[ConsoleChatMessage]]]` (anchor id per block; `None` for legacy/null runs). `inject_resume_agent_markers(messages, anchored_blocks)` — new second-arg shape.

- [ ] **Step 1: Write failing tests.**
  - `resume_marker_messages` returns `(assistant_message_id, block)` pairs (the id read from `record["assistant_message_id"]`).
  - `inject_resume_agent_markers`: (a) a block with an anchor id matching a message's `persisted_message_id` lands **immediately after that message** (even if it's not the last assistant); (b) a block whose anchor id matches NO active-path message is **dropped/hidden** (off-path); (c) a block with `anchor id == None` (legacy) is placed by the **ordinal fallback** (Nth null-block → Nth assistant among the messages not already id-claimed); (d) idempotency (second call adds no dupes) and empty-block skipping preserved.
  - Update the existing ordinal tests (`test_inject_resume_agent_markers_places_block_after_matching_assistant_message` etc.) to set `persisted_message_id` on the test messages + anchor ids on the blocks; keep the leftover/idempotency/empty tests.
- [ ] **Step 2: Run RED.**
- [ ] **Step 3: Implement.**
  - `resume_marker_messages`: attach `record.get("assistant_message_id")` to each block (return `(anchor_id, block)`).
  - `inject_resume_agent_markers`: build `by_persisted = {m.persisted_message_id: index for index, m in enumerate(messages) if m.persisted_message_id}`. Partition anchored blocks into id-set and null. For id-set blocks: if `anchor_id in by_persisted` → insert after that index; else → **drop** (off-path/stale). For null blocks: apply the existing ordinal placement against the assistant indexes **not already used by an id match**. Keep the `_already_present` content-idempotency guard on every insert.
  - `chat_screen.py:_inject_resume_agent_markers`: pass the `(anchor, block)` list straight through (signature of the module fn changed; the call adapts).
- [ ] **Step 4: Run GREEN** + the full `test_console_agent_bridge.py` and the resume-wiring test.
- [ ] **Step 5: Commit** — `feat(console): anchor resumed agent markers to the produced reply; hide off-branch`.

---

### Task 4: End-to-end (agent run on a branch) + regression

**Files:**
- Test: `Tests/integration/test_console_agent_marker_anchoring_e2e.py`

- [ ] **Step 1: Write the E2E** over real DB + store + controller + a fake agent path (reuse the agent-bridge test harness): run an agent reply A1 (produces TOOL steps), regenerate → agent reply A1' (its own steps), so two assistant siblings each with their own run. Persist → drop store → resume on the A1' branch: assert A1's markers are **hidden** and A1''s markers render after A1'; swipe to A1 → its markers now show, A1''s hidden (after a fresh resume/overlay). Include a legacy run (null `assistant_message_id`) asserting the ordinal fallback still places it.
- [ ] **Step 2: Run + regression sweep** — `./.venv/bin/python -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Agents/ Tests/DB/test_agent_runs_db.py Tests/UI/test_console_native_chat_flow.py Tests/integration/ -q`; name the known pre-existing baseline failures and confirm zero new. Fix any integration bug surfaced (document prominently).
- [ ] **Step 3: Commit** — `test(console): end-to-end agent-marker anchoring across branches`.

---

## Self-Review

**Spec coverage:** column+migration (Task 1); id threaded + persisted-id written post-completion (Task 2 — resolves the native-vs-persisted timing the design decision called out); resume placement id-anchored with off-path-hide + null-ordinal-fallback + idempotency (Task 3); e2e across branches (Task 4). Backward-compat (null-id → ordinal) and sub-agent-runs-carry-no-id are explicit constraints.

**Placeholder scan:** Task 2 leaves ONE bounded choice (how `run_reply` exposes the run id — return-tuple vs bridge field) to the implementer with a stated preference and the test as contract; everything else is concrete. No TBDs.

**Type consistency:** `create_run(..., assistant_message_id: str|None=None)`, `set_run_assistant_message_id(run_id, assistant_message_id)`, `run_turn(...) -> (run_id, RunOutcome)`, `resume_marker_messages -> list[tuple[str|None, list[ConsoleChatMessage]]]`, `inject_resume_agent_markers(messages, anchored_blocks)` — consistent across Tasks 1–4. The stored/compared id is always the **persisted** id (`ConsoleChatMessage.persisted_message_id`).

**Risk:** Task 2's `run_reply` return-shape change ripples to the `run_reply` fakes in `test_console_agent_swap.py` — update them. The off-path-hide changes the old "orphan → append" semantics for id-set runs (intended); the null-ordinal path preserves it for legacy data.
