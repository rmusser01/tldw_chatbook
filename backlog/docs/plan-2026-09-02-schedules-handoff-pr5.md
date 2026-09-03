# Schedules Handoff — PR-5: Transfer state machine

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-task ownership transfer in both directions for reminders and `recurring_question` definitions — the spec §6 state machine (`transfer_to_server` / `release_from_server` as SyncEngine mutation actions), with the §3 invariant ("at most one side armed at any instant") proven by a property test, plus the lifecycle mutation replay PR-3 deferred here and the schedule-kind vocabulary translation PR-4's final review flagged.

**Architecture:** `transfer_state ∈ {NULL, to_server_pending, to_server_sent, to_server_failed, from_server_pending}` already exists on both tables (schema v4 — NO new migration). The machine runs inside `SyncEngine`'s push phases at the two existing action-dispatch slots (`_push_definition_mutation`, reminder `_push_mutation`), reusing retry/backoff/error surfaces. Definitions recover ambiguous create timeouts via the server's payload-hash idempotency (safe re-create); reminders recover via `link_type="chatbook_transfer"` + `link_id` list-and-match (fields already ride the client/server contract end-to-end). A pure vocabulary module translates schedule dicts at every push boundary and at server→local copy creation (client `every_seconds`/`time_of_day` vs server `seconds`/`at` — the server validates only `kind`, so an untranslated transfer passes preview and silently never arms). A `SchedulingService` facade computes §6.4 refusals (quoting local health); `task_detail.py`'s disabled-with-reason idiom carries the Move/Cancel buttons. The program ADR (per-task ownership transfer, amending ADR-077 §1's screen-era framing) drafts here.

**Tech Stack:** Python ≥3.11, Textual 8.x, httpx-backed tldw_api client (faked in tests), SQLite, pytest (+hypothesis for the property test if the suite already uses it — check; otherwise a seeded-random interleaving test).

**Spec:** `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` §6 (+§3 invariant, §5.1 lifecycle push, §10 property test). Survey with exact seams: `pr5-survey.md` in this plan's SDD workspace. Planning rulings (binding):
1. **Move/Cancel UI ships in PR-5** (a state machine with no trigger is inert — the program's cardinal defect class); transfer badges/owner-column polish stay PR-6.
2. **Armed-while-pending semantics correction**: `to_server_pending` and `to_server_failed` rows KEEP executing locally (§6.1.1/§6.1.5); only `to_server_sent` and `from_server_pending` are dormant. The existing definitions-side "exclude any non-NULL transfer_state" filters are WRONG per spec and get corrected to the two-state exclusion; reminders gain the same (they have no guards at all today).
3. **to_server_failed error surfacing**: the failed mutation is RETAINED with the server's field errors embedded in its payload (`{"transfer_errors": [...]}`); the detail pane reads them from there. No new column.
4. **Definitions transfer recovery** = hash-idempotent create retry (no list-and-match; that's the reminder leg only, per §6.1.3's wording).
5. **ADR number assigned at merge time** against origin/dev, sweeping all three backlog buckets + open branches (lessons-backlog-hygiene; 7 prior collisions).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-handoff-pr5`, branch `feat/schedules-transfer-machine` off current `origin/dev`. Never the main checkout; NEVER `git stash`; `git --no-pager` for reads; foreground pytest only; tmp_path DBs (never :memory:).
- NO schema migration (columns exist). If any task believes it needs one, STOP and escalate to the controller.
- Server contract references: `Tests/Scheduling/fixtures/server_responses/automation_endpoints.md` (in-repo, byte-accurate) and `/Users/macbook-dev/Documents/GitHub/tldw_server2` @ origin/dev via `git show` (NEVER modify): `scheduled_task_automation_scheduler.py::build_trigger` (server schedule vocab), `reminders.py` (`ReminderTaskCreateRequest` — `link_type`/`link_id` confirmed present).
- Mutation payloads never store preview ids (spec §5.1). Transfer payloads store the full create payload.
- Diagnostics pin: regenerate `--write` + commit the JSON in the SAME commit whenever ANY logger statement is added/moved/reworded (PR-3 and PR-4 both tripped CI on this).
- Boot census (ADR-097): the merge bar is breach-list PARITY with origin/dev tip, not just ≤972 — verify with the diff-vs-snapshot procedure at the final task; any new boot-resident import goes function-level.
- UI change ⇒ update the matching `Docs/User_Guide/` page (or its "Verified against" stamp).
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: Transfer-state arming semantics + reminder-side parity guards

**Files:** Modify `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (armable filters both primitives; `_apply_pulled_reminders` gains the `transfer_state` pop the definitions upsert already has; `run_reminder_now`/`run_automation_now` refusals), `tldw_chatbook/Scheduling/scheduler/queue.py` (the in-memory layer of the same filter — read `PriorityQueue.load` + `is_server_scoped_owner` at queue.py:25 first). Tests: `Tests/Scheduling/test_scheduled_tasks_db.py`, `Tests/Scheduling/test_queue.py` (or wherever the armable filters are pinned — locate first).

**Interfaces (produced):** module constant `DORMANT_TRANSFER_STATES = ("to_server_sent", "from_server_pending")` in `scheduled_tasks_db.py`, used by every armable filter and run-now refusal; DB helpers `set_transfer_state(table_kind, row_id, state, *, expected: tuple[str|None, ...]) -> bool` (compare-and-set inside one transaction — the machine's transitions must be race-safe against concurrent sync/UI calls; returns False when the current state isn't in `expected`) and `clear_transfer_state(...)` sugar.

- [ ] TDD: armed states (`NULL`, `to_server_pending`, `to_server_failed`) arm; dormant states don't — BOTH primitives, both filter layers; pulled reminder payload can't overwrite local `transfer_state`; run-now refuses dormant rows with a reason; compare-and-set rejects a wrong expected-state. FAIL → implement → PASS → commit `feat(scheduling): transfer-state arming semantics + reminder parity guards`.

### Task 2: Definition lifecycle client seams + lifecycle mutation replay

**Files:** Modify `tldw_chatbook/tldw_api/client.py` (+`pause_scheduled_task_definition`/`resume_.../archive_...` via `POST /api/v1/scheduled-tasks/definitions/{id}/{pause|resume|archive}` — mirror the run-now method's construction; responses are `ScheduledTaskDefinitionResponse`), `tldw_chatbook/Notifications/server_notifications_service.py` (three seams under `scheduler.automations.configure`), `tldw_chatbook/Scheduling/services/server_client.py` (retryable wrappers — lifecycle posts are idempotent by nature: pausing a paused definition is a no-op server-side, cite that), `tldw_chatbook/Scheduling/services/sync_engine.py` (`_push_definition_mutation` gains `pause`/`resume`/`archive` actions: direct endpoint calls, `ServerClientNotFoundError` clears the mutation, mirror the echo via the existing upsert). Tests: the client-seam test file from PR-4 + `test_sync_engine.py`.

- [ ] TDD (client-test style per the PR-4 precedent; fixture reuse from `automation_definitions_list.json`). FAIL → implement → PASS → commit `feat(scheduling): definition lifecycle seams + mutation replay`.

### Task 3: Schedule vocabulary translation (pure)

**Files:** Create `tldw_chatbook/Scheduling/schedule_vocabulary.py`. Modify `tldw_chatbook/Scheduling/services/sync_engine.py` (apply `to_server_schedule` in `_push_definition_create`/`_push_definition_update` on the payload's `schedule` before the preview request). Tests: `Tests/Scheduling/test_schedule_vocabulary.py`.

**Interfaces:** `to_server_schedule(schedule: dict) -> dict` and `to_local_schedule(schedule: dict) -> dict` — pure renames per the survey's divergence table (client `every_seconds` ↔ server `seconds`; client `time_of_day` ↔ server `at`; verify the full table against BOTH `schedule_compute.py` and the server's `build_trigger` source and cover every kind `schedule_compute` supports: one_time/interval/daily/weekly/cron; unknown keys pass through untouched; unknown `kind` returns the dict unchanged). Consumed by Task 4 (transfer push), Task 5 (server→local copy), and wired into the EXISTING create/update replay here.

- [ ] TDD: per-kind rename tables both directions; round-trip identity property (`to_local(to_server(s)) == s` for every supported kind, seeded-random field values); existing PR-4 create-replay tests updated to expect server vocab on the wire. FAIL → implement → PASS → commit `feat(scheduling): schedule vocabulary translation at the push boundary`.

### Task 4: Local → server transfer in SyncEngine

**Files:** Modify `tldw_chatbook/Scheduling/services/sync_engine.py` (new `transfer_to_server` action in BOTH dispatch slots — definitions `_push_definition_mutation` sync_engine.py:529-region, reminders `_push_mutation` :825-region), `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (the §6.1.4 convert-or-merge: `convert_row_to_server_mirror(table_kind, local_id, server_item) -> str` — if a pulled mirror already exists for `(owner_id, server_id)` keep it, delete the local row, transplant provenance (created_at, audit linkage); else convert in place: set `server_id`+`owner_id`, clear `transfer_state` — ONE transaction). Tests: `test_sync_engine.py`, `test_scheduled_tasks_db.py`.

**Behavior (read §6.1 verbatim in the spec copy first):**
- Push attempt starts: compare-and-set `to_server_pending` → `to_server_sent` (Task 1 helper — a failed CAS means cancel/concurrent change; skip the mutation). The row is now dormant (Task 1 filters) — disarm precedes the request.
- Definitions: `to_server_schedule` on the payload → preview → create with `initial_lifecycle` matching the source row (`configured`/`paused`). Ambiguous timeout: leave `to_server_sent` + mutation retained; the NEXT replay re-runs preview→create — the payload-hash idempotency dedupes server-side (ruling 4). Reminders: `/tasks` create carrying `link_type="chatbook_transfer"`, `link_id=<local task id>`; ambiguous timeout → recovery is Task 6's list-and-match (leave state + mutation).
- Ack → `convert_row_to_server_mirror` + clear mutation.
- Definitive failure (invalid preview / 4xx): state → `to_server_failed` (re-arms via Task 1 semantics), mutation RETAINED with `{"transfer_errors": [...]}` embedded (ruling 3) and NOT retried automatically (a `transfer_errors`-bearing mutation is skipped by the replay loop until the user retries/cancels — encode the skip).
- Phase discipline: per-mutation containment (the PR-4 Qodo lesson — one poisoned transfer never blocks others).

- [ ] TDD: happy path both primitives (disarm-before-send order pinned by a fake client asserting the row was dormant when the request fired); CAS-skip; timeout-retains; convert-in-place; merge-with-pulled-mirror (seed the mirror first — the §4 UNIQUE race); definitive failure → failed+errors+no-auto-retry. FAIL → implement → PASS → commit `feat(scheduling): local-to-server transfer replay`.

### Task 5: Server → local transfer + release in SyncEngine

**Files:** Modify `tldw_chatbook/Scheduling/services/sync_engine.py` (+`release_from_server` action both slots), `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (+`create_local_copy_from_mirror(table_kind, mirror_id) -> str`: local-owner copy with `from_server_pending` (dormant), schedule through `to_local_schedule`, fresh local id, `server_id=None`; the mirror row stays). Tests: `test_sync_engine.py`, `test_scheduled_tasks_db.py`.

**Behavior (§6.2):** release replay: reminders → the tombstone/delete path the reminder sync already uses (read `_push_tombstone` first — reuse its server call, not a new one); definitions → Task 2's archive seam. Ack → clear the local copy's `transfer_state` (arms on next queue reload — call the same reload-notify the other mutations use) + mirror the archived lifecycle onto the server-mirror row. `ServerClientNotFoundError` on release = the server row is already gone → treat as ack. Offline/unacked: copy stays dormant (no code needed — Task 1 filters — but PIN it with a test).

- [ ] TDD: release both primitives; NotFound-as-ack; dormant-until-ack pinned; copy carries translated schedule + no server_id. FAIL → implement → PASS → commit `feat(scheduling): server-to-local release replay`.

### Task 6: Service facade — transfer/cancel/refusals + startup recovery

**Files:** Modify `tldw_chatbook/Scheduling/services/scheduling_service.py`, `tldw_chatbook/app.py` (startup hook only — mirror the guarded `reconcile_stale_automation_runs` on_mount precedent, one line calling the service). Tests: `Tests/Scheduling/test_scheduling_service.py`, plus `Tests/Scheduling/test_transfer_invariant.py` (the property test).

**Interfaces (consumed by Task 7's UI):**
- `transfer_refusal(row: dict, direction: str) -> str | None` — §6.4 verbatim: no server connection/identity; family (`agent_task`→local always refuses v1; `recurring_question`→local refuses when `compute_local_health` ≠ ready, QUOTING the health reason); transfer already pending; lifecycle not in (`configured`,`paused`) — `archived`/`solved` refuse; returns None when allowed. Separate `transfer_warnings(row, direction) -> list[str]`: imminent one-time `run_at` (<~5 min) warns-not-refuses; non-transferring local fields named (reminders: `timeout_seconds`; definitions: none — `next_run_at` recomputes).
- `async begin_transfer_to_server(table_kind, row_id) -> TransferOutcome`, `async begin_transfer_to_local(table_kind, row_id) -> TransferOutcome`, `async cancel_transfer(table_kind, row_id) -> TransferOutcome` implementing the §6.3 table exactly (unattempted → clear state + drop mutation + keep local; sent/acked → refuse with "offer reverse transfer" copy; release-unpushed → delete dormant copy + drop mutation; release-acked → refuse likewise). All state changes through Task 1's compare-and-set.
- Startup recovery: `async recover_inflight_transfers()` — rows stuck `to_server_sent` at startup: reminders → list-and-match on `link_id` via the existing reminder list wrapper (found ⇒ `convert_row_to_server_mirror`+clear mutation; absent ⇒ CAS back to `to_server_pending` for normal retry); definitions → CAS back to `to_server_pending` (hash-idempotent retry, ruling 4).
- **Property test** (spec §10): drive randomized interleavings of {begin, push-attempt-start, ack, definitive-fail, cancel, release, release-ack} against a real tmp_path DB asserting after EVERY step: not (row armed locally AND its server counterpart live-armed) — model the server side as the fake's created/archived set. Use hypothesis if already a test dep (check `pyproject.toml`); otherwise seeded `random` with ≥200 interleavings and the seed in the failure message.

- [ ] TDD all legs incl. the property test. FAIL → implement → PASS → commit `feat(scheduling): transfer facade, cancel semantics, startup recovery`.

### Task 7: Detail-pane Move/Cancel UI + confirm dialog

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/task_detail.py` (Move to server / Move to local / Cancel transfer buttons using the file's disabled-with-reason idiom (UX-073 tooltip + Static reason) and the UX-059 only-state-changing-action pattern — read both idioms in place first), `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (routing to Task 6's facade; confirm dialog before a transfer listing the warnings from `transfer_warnings` — reuse the smallest existing confirm-modal pattern; honest toasts incl. "still runs here until the server accepts it" for pending, per §6.1.1; reload after mutations via the existing wiring), CSS source if needed (bundle via build flow only), `Docs/User_Guide/` schedules page. Tests: the workbench/detail test files located in Task 1's sweep + extend.

- [ ] TDD: refusal renders disabled-with-reason (health reason quoted); warn dialog shows imminent-run_at + non-transferring fields; cancel routes per state; toasts honest. FAIL → implement → PASS → commit `feat(scheduling): transfer actions in the task detail pane`.

### Task 8: ADR + 18940 note + E2E + gates

**Files:** Create `backlog/decisions/112-per-task-schedule-ownership-transfer.md` (number = next free against origin/dev AT MERGE TIME — draft as `1XX`, controller renumbers; content: per-task ownership transfer amending ADR-077 §1's screen-era framing, local recurring_question execution recap, results sync-down recap — spec §13; status Proposed). Modify `backlog/tasks/task-18940 - Server-offloaded-scheduled-agent-tasks-execution-seam.md` (progress-log pointer: AC#2's client half satisfied for recurring_question by PR #2302, authoring per spec §8). Create `Tests/Scheduling/test_transfer_end_to_end.py`.

- [ ] E2E (real tmp_path DB + fake server client): (a) local reminder → begin transfer → sync (disarm→create with link fields→convert) → row is a server mirror, never both armed at any observed step; (b) local definition same via preview→create; (c) server-owned definition mirror → begin to-local → dormant copy → release ack → copy armed + mirror archived; (d) crash-recovery: to_server_sent seeded → `recover_inflight_transfers` list-and-match completes the reminder leg. Then FULL gates: `Tests/Scheduling/ -q`; the schedules UI files; census with breach-list PARITY vs origin/dev tip (the PR-4 procedure); diagnostics pin; ruff. Commit `test(scheduling): transfer end-to-end + program ADR`.

---

## After the tasks
Final whole-branch review (opus, cross-seam + §3-invariant lens) → one fix wave → PR `feat(scheduling): per-task ownership transfer both directions (handoff PR-5)` → paged bot-comment read (per_page=100, count `in_reply_to_id == null`) → adjudicate → ADR number sweep against origin/dev at merge time → sequential rebase-watch-merge.
