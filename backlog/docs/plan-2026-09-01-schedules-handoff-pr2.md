# Schedules Handoff — PR-2: Local recurring_question Execution

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Local-owner `recurring_question` automation definitions execute on the chatbook scheduler: scope → retrieval → classification → run/result rows → notification, mirroring tldw_server's semantics.

**Architecture:** No new retrieval or generation machinery — the pipeline composes two existing hardened seams: `run_library_rag_search` (retrieval, `Library/library_rag_service.py`) and `generate_library_rag_answer` (contained generation, `Library/library_rag_answer_service.py`). New pure modules port the server's scope normalizer and outcome-classification ladder; a new handler follows `BriefingJobHandler`'s spawn discipline and writes the PR-1 run/result rows. Everything heavy imports lazily inside the spawned run so nothing joins the warm `_ui_ready` census.

**Tech Stack:** Python ≥3.11, asyncio, SQLite (PR-1 accessors), pytest.

**Spec:** `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` §7 (execution), §4 (rows), §6.4 (health-gated refusals). Two deviations from §7.1, ruled at planning: (1) the `_validate_*` authoring validators port in PR-4 where they are consumed; (2) the server's fixed generation-only system prompt is NOT ported — generation runs through `generate_library_rag_answer`, whose own evidence-grounded, citation-validating prompt builder supersedes it (a stricter prompt than the server's; parity is in outcome vocabulary, not prompt bytes).

## Global Constraints

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-handoff-pr2`, branch `feat/schedules-local-rq-execution` off dev `8160301fa`. Never touch the main checkout; never `git stash`.
- Tests: `.venv/bin/python -m pytest <paths> -v` from the worktree root, ALWAYS foreground. tmp_path file DBs, never `:memory:`.
- **Boot-census ratchet (ADR-097, the limit never rises):** the handler module is registered in `app.py`'s handler dict and therefore imports at boot. It must stay light: the execution stack (Tasks 2–3 modules, the two Library seams) is imported INSIDE `handle()`/the spawned coroutine, never at handler-module top level. CI's `_ui_ready` census enforces this.
- **Scheduler loop discipline (locked, briefings phase 4):** `SchedulerLoop.tick` awaits handlers serially inline. A recurring_question run is retrieval + an LLM call — the handler's `handle()` does synchronous claim-checking only and spawns the run as a strong-referenced `asyncio.Task` (`BriefingJobHandler` pattern: `_pending` set + `add_done_callback(discard)`).
- **Single-executor invariant:** only local-owner (`not is_server_scoped_owner`), `lifecycle == "configured"`, `transfer_state IS NULL` definitions ever arm.
- Server vocabulary parity: run `status`/`outcome`, result `answer_mode`/`review_state`, notification kinds `automation_run_{succeeded,failed,timed_out,skipped}` (already in the client's `NotificationKind` literal), bounded summary 1000 chars, `RESULT dedupe recipe {definition_id}:{run_id}:{kind}` via `slot_keys.canonical_hash`-style hashing exactly as PR-1's accessors expect.
- New config keys introduced here: `[scheduling] executor_provider`, `executor_model`, `executor_max_tokens` (all optional; precedence in Task 3).
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: Schedule computation — `next_run_at` for the five kinds

**Files:**
- Create: `tldw_chatbook/Scheduling/schedule_compute.py`
- Test: `Tests/Scheduling/test_schedule_compute.py`

**Interfaces:**
- Produces: `compute_next_run_at(schedule: dict[str, Any], *, now: datetime) -> datetime | None` — timezone-aware UTC result; `None` for a spent one_time or an invalid schedule (never raises for junk — the queue must not die on a bad row). Advance-from-now semantics: elapsed slots are skipped, never replayed (spec §4.3 / TASK-18937 discipline). Also `schedule_slot_for(next_run: datetime) -> str` — the canonical UTC ISO string used as the run's `schedule_slot`.
- Kinds (server `_SUPPORTED_SCHEDULE_KINDS`): `one_time` (`run_at` ISO), `interval` (`every_seconds >= 60`), `daily` (`time_of_day "HH:MM"`, optional IANA `timezone` defaulting to the machine zone), `weekly` (adds `weekday` 0–6, 0=Monday), `cron` (5-field, via `croniter`, optional `timezone`).

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timedelta, timezone

from tldw_chatbook.Scheduling.schedule_compute import (
    compute_next_run_at,
    schedule_slot_for,
)

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def test_one_time_future_passes_through_and_past_returns_none():
    future = "2026-09-02T09:00:00+00:00"
    assert compute_next_run_at({"kind": "one_time", "run_at": future}, now=NOW) \
        == datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    assert compute_next_run_at(
        {"kind": "one_time", "run_at": "2026-08-01T09:00:00+00:00"}, now=NOW
    ) is None


def test_interval_advances_from_now_never_replays():
    nxt = compute_next_run_at({"kind": "interval", "every_seconds": 900}, now=NOW)
    assert nxt == NOW + timedelta(seconds=900)


def test_interval_below_floor_is_invalid():
    assert compute_next_run_at({"kind": "interval", "every_seconds": 30}, now=NOW) is None


def test_daily_respects_timezone():
    nxt = compute_next_run_at(
        {"kind": "daily", "time_of_day": "09:00", "timezone": "America/New_York"},
        now=NOW,  # 12:00 UTC = 08:00 New York -> today's 09:00 NY is next
    )
    assert nxt == datetime(2026, 9, 1, 13, 0, tzinfo=timezone.utc)


def test_weekly_picks_next_weekday_occurrence():
    # 2026-09-01 is a Tuesday (weekday 1). Ask for Monday (0) 09:00 UTC.
    nxt = compute_next_run_at(
        {"kind": "weekly", "weekday": 0, "time_of_day": "09:00", "timezone": "UTC"},
        now=NOW,
    )
    assert nxt == datetime(2026, 9, 7, 9, 0, tzinfo=timezone.utc)


def test_cron_and_junk():
    nxt = compute_next_run_at({"kind": "cron", "cron": "0 9 * * *"}, now=NOW)
    assert nxt == datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)
    assert compute_next_run_at({"kind": "cron", "cron": "not cron"}, now=NOW) is None
    assert compute_next_run_at({"kind": "nope"}, now=NOW) is None
    assert compute_next_run_at("junk", now=NOW) is None


def test_slot_string_is_canonical_utc_iso():
    assert schedule_slot_for(datetime(2026, 9, 2, 9, 0, tzinfo=timezone.utc)) \
        == "2026-09-02T09:00:00+00:00"
```

- [ ] **Step 2: Run to verify FAIL** (`module not found`): `.venv/bin/python -m pytest Tests/Scheduling/test_schedule_compute.py -v`

- [ ] **Step 3: Implement** — pure module; `croniter` for cron only; daily/weekly computed with `zoneinfo` in the schedule's zone then `.astimezone(timezone.utc)`; machine-zone default via `datetime.now().astimezone().tzinfo`; every parse guarded, returning `None` on junk (log at debug). Docstring cites spec §4.3 advance-from-now.

- [ ] **Step 4: PASS + commit** — `feat(scheduling): schedule computation for the five reference kinds`

### Task 2: Scope normalizer port + source mapping

**Files:**
- Create: `tldw_chatbook/Scheduling/recurring_question_scope.py`
- Test: `Tests/Scheduling/test_recurring_question_scope.py`

**Interfaces:**
- Produces: `normalize_recurring_question_scope(scope, *, available_sources=None) -> (normalized: dict, errors: list[dict], warnings: list[dict])` — byte-parity port of the server module of the same name (tldw_server `app/core/Scheduled_Tasks/recurring_question_scope.py`); errors are `{field, code, message}` dicts. `DEFAULT_SEARCHABLE_SOURCES = ("media_db", "notes", "chats")`, `SUPPORTED_SCOPE_FIELDS` identical to the server's.
- Produces: `engine_source_types(normalized_scope: dict) -> tuple[str, ...]` — maps server source names onto the retrieval engine vocabulary: `media_db → "media"`, `notes → "note"`, `chats → "conversation"` (the `rag_service` docstring's own vocabulary). Unknown names are skipped with a warning entry, never raised.

- [ ] **Step 1: Failing tests** — port the semantics as assertions: `mode` default `all_searchable_library` resolves all three sources; unknown scope field → `{"field": "config.scope.<name>", "code": "unsupported"}`; unknown mode returns single unsupported error; `sources` mode drops unavailable sources into `{"code": "source_unavailable"}` warnings; empty resolution → `{"field": "config.scope", "code": "scope_empty"}`; `engine_source_types({"mode": "all_searchable_library", "resolved_sources": ["media_db", "notes", "chats"]}) == ("media", "note", "conversation")`.

- [ ] **Step 2: FAIL run.**

- [ ] **Step 3: Implement** — transcribe the server module verbatim (drop only the server import lines; inline the two constants), then add `engine_source_types`. Module docstring: "Ported from tldw_server `recurring_question_scope.py` @ 5921014aa9 — byte-parity except imports; regenerate the fixture tests when the server module changes (spec §7.1 drift rule)."

- [ ] **Step 4: PASS + commit** — `feat(scheduling): port the recurring_question scope normalizer (server parity)`

### Task 3: Execution core — resolve target, run the two seams, classify

**Files:**
- Create: `tldw_chatbook/Scheduling/automation_execution.py`
- Test: `Tests/Scheduling/test_automation_execution.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True)
class ExecutionOutcome:
    outcome: str            # finding | no_match | degraded  (server RunOutcome vocab)
    title: str
    summary: str            # bounded RESULT_SUMMARY_MAX_CHARS = 1000
    answer: Any | None
    answer_mode: str        # synthesized | evidence_only | none
    confidence: dict
    source_refs: list[dict]
    evidence_summary: dict  # {result_count, answer_present, retrieval_status}
    failure_reason: dict | None

def resolve_execution_target(definition_row: dict) -> dict:
    """{provider, model, max_tokens} — definition input.provider/model/max_tokens
    -> [scheduling] executor_provider/executor_model/executor_max_tokens
    -> resolve_library_rag_answer_provider() (provider/model both), with the
    server's sanitize-each-layer-then-precedence discipline and the
    _MAX_TOKENS_CAP = 4000 / default 1000 bounds."""

async def execute_recurring_question(app: Any, definition_row: dict) -> ExecutionOutcome: ...
```

- Consumes: Task 2's normalizer + `engine_source_types`; `run_library_rag_search(app, LibraryRagSearchRequest(query=..., source_types=..., mode="rag", top_k=..., include_citations=True))` returning `LibraryRagSearchOutcome(status, results, ...)`; `generate_library_rag_answer(query=, results=, coverage_note="", provider=, model=, chat=None)` returning `LibraryRagAnswer(status, text, error, usage, ...)` — read both dataclasses before writing the mapping; their `ANSWER_STATUS_*` constants live in `library_rag_answer_service.py`.
- Classification ladder (the server's `classify_rag_response`, adapted to these seams — implement as a table, test every row):

| retrieval status | results | generation_mode | answer status | → outcome / answer_mode |
|---|---|---|---|---|
| blocked/failed | — | — | — | `degraded`, `none`, failure_reason `{code: "retrieval_" + status}` |
| ok | 0 | — | — | `no_match`, `none`, title "No matching sources found" |
| ok | >0 | disabled | (not called) | `finding`, `evidence_only`, title "Relevant evidence found" |
| ok | >0 | optional/required | ready | `finding`, `synthesized`, summary = bounded answer text, title "Possible answer found" |
| ok | >0 | optional | abstained / no-evidence / failed | `finding`, `evidence_only` (answer text dropped; failure noted in evidence_summary) |
| ok | >0 | required | anything but ready | `degraded`, `none`, failure_reason `{code: "generation_required_unavailable"}` |

- `finding_policy` mapping: preset `balanced_findings` → top_k 10; `high_confidence_only` → top_k 10 + post-filter results by score where the row exposes one (consult `LibraryRagResultRow`; if no usable score field exists, note it in the module docstring and skip the filter — do not invent one). `top_k` override from `finding_policy["top_k"]` bounded 1–100 (server `_coerce_int` semantics).
- `source_refs`: one `{source, id, title}` dict per retrieval row from the fields `LibraryRagResultRow` actually carries — read the class, take what exists, never fabricate keys.
- **Lazy-import rule**: this module may import the Library seams at top level (it is itself only imported inside the spawned run — Task 4 enforces that), but keep `get_cli_setting` usage call-time.

- [ ] **Step 1: Failing tests** — fake `app` carrying a fake `library_rag_search_service`; monkeypatch `run_library_rag_search`/`generate_library_rag_answer` at the `automation_execution` module attributes; one test per ladder row (six), plus `resolve_execution_target` precedence: definition wins, config second (`get_cli_setting("scheduling", "executor_provider", ...)` monkeypatched), library-provider fallback last; blank/junk definition values fall through (server review-#5 discipline); max_tokens capped at 4000.

- [ ] **Step 2: FAIL run.**

- [ ] **Step 3: Implement.** Summary bounding helper `_bounded(text) -> str` at 1000 chars with ellipsis, used for every summary written.

- [ ] **Step 4: PASS + commit** — `feat(scheduling): recurring_question execution core — resolve, retrieve, classify`

### Task 4: The automation handler — spawn shape, rows, timeout, notification

**Files:**
- Create: `tldw_chatbook/Scheduling/scheduler/handlers/automation_handler.py`
- Test: `Tests/Scheduling/test_automation_handler.py`

**Interfaces:**
- Produces: `AutomationDefinitionHandler(db: ScheduledTasksDB, app_getter: Callable[[], Any] | None = None, dispatch_service: NotificationDispatchService | None = None, handler_timeout_seconds: float | None = None, executors: dict[str, Executor] | None = None)` with async `handle(task: dict) -> None` and `__call__ = handle`. `Executor = Callable[[Any, dict], Awaitable[ExecutionOutcome]]`; default registry `{"recurring_question": <lazy wrapper around Task 3>}` — family-keyed so agent_task drops in later.
- The queue row it receives (Task 5 produces): the definition row dict + `{"type": "automation_definition"}`.

Behavior, in order inside `handle` (synchronous parts) then the spawned coroutine:

1. **Family check**: no executor for `row["family"]` → log warning, return (loop-level unhandled-type counting already exists for unknown `type`s; this is the per-family guard).
2. **Claim guard** (spec §7.2, briefing precedent): a per-definition-id in-flight set on the handler. Already claimed → `create_automation_run(..., trigger_reason=row-trigger, status="skipped", outcome="none", schedule_slot=None, run_summary={"skipped": "overlap", "claimed_slot": slot})` and return.
3. **Slot dedupe**: compute `slot = schedule_slot_for(next_run)` from the row's `next_run_at`; `run_id = create_automation_run(owner_id, definition_id, version, "scheduled", status="running", schedule_slot=slot, started_at=now, scope_snapshot=..., finding_policy_snapshot=...)`. `None` → slot already ran → return silently (dedupe is a result).
4. **Advance the schedule AT SPAWN** (spec §4.3): `update_automation_definition(definition_id, next_run_at=compute_next_run_at(schedule, now=now))`.
5. **Spawn** the run coroutine (strong-ref set + `add_done_callback(discard)`), which:
   - lazy-imports `automation_execution` (census rule) and runs `execute_recurring_question` under `asyncio.wait_for(handler_timeout_seconds)` (≤0/None disables, `coerce_positive_float` discipline);
   - on timeout: `update_automation_run(run_id, status="timed_out", outcome="degraded", ended_at=..., failure_reason={"code": "execution_timeout"})`;
   - on exception: `status="failed", outcome="degraded", failure_reason={"code": "execution_error", "error_type": type(exc).__name__}` — never re-raise (spawned task, nothing above catches);
   - on `ExecutionOutcome`: `status="completed"`, `outcome=eo.outcome`, `run_summary`/`evidence_summary`/`failure_reason` from it, `ended_at`; when `eo.outcome == "finding"` also `create_automation_result(owner_id, definition_id, run_id, "finding", eo.title, eo.summary, dedupe_key, answer=eo.answer, answer_mode=eo.answer_mode, confidence=eo.confidence, source_refs=eo.source_refs)` with `dedupe_key = canonical_hash({"definition_id":..., "run_id":..., "kind": "finding"})` (import from `slot_keys`);
   - **notification** (server `NOTIFICATION_KIND_BY_STATUS` parity): map terminal status → kind `automation_run_{succeeded,failed,timed_out}` (completed→succeeded; skipped runs notify only when the definition's `notification_policy` asks — default: no notification for skipped, matching the server's policy-gated default). Dispatch via the injected `dispatch_service` with `app=self.app_getter()` (ReminderHandler pattern), `category="automation"`, title = definition name, message = bounded summary, `source_entity_kind="automation_definition"`, `source_entity_id=definition_id`, payload `{"kind": <kind>, "run_id": run_id, "outcome": ...}`. `notification_policy` gate: port the server's `_notification_enabled(definition, status)` semantics — `{"on_success": bool, "on_failure": bool}` keys with both defaulting True.
   - claim released in a `finally`.

- [ ] **Step 1: Failing tests** — real tmp_path `ScheduledTasksDB` (v4), fake executor injected via `executors=` (no Library imports in this suite): happy path writes running→completed run + unread finding result + notification with kind `automation_run_succeeded` and the app handle; timeout path (executor sleeps, tiny timeout) → `timed_out` + `automation_run_timed_out`; executor raises → `failed` + kind; overlap claim (second `handle` while executor blocked on an `asyncio.Event`) → one `skipped` run row, no double execution; slot dedupe (same slot twice) → second `handle` writes nothing; schedule advanced at spawn (definition's `next_run_at` updated before executor completes); `notification_policy={"on_failure": False}` suppresses the failed notification.

- [ ] **Step 2: FAIL run.**

- [ ] **Step 3: Implement** — module top level imports ONLY stdlib + `slot_keys` + `schedule_compute` + the DB/dispatch types under `TYPE_CHECKING`; the execution stack imports inside the coroutine. Mirror `BriefingJobHandler`'s docstring discipline for the spawn rationale.

- [ ] **Step 4: PASS + commit** — `feat(scheduling): automation definition handler — spawn, claim, rows, notify`

### Task 5: Queue feed + app wiring

**Files:**
- Modify: `tldw_chatbook/Scheduling/scheduler/queue.py` (definitions feed in `load`)
- Modify: `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (one accessor: `list_armable_automation_definitions(owner_id="local") -> list[dict]` — `family='recurring_question'` (v1), `lifecycle='configured'`, `next_run_at IS NOT NULL`, `transfer_state IS NULL`)
- Modify: `tldw_chatbook/app.py` (register the handler + reconcile call)
- Test: `Tests/Scheduling/test_queue.py` or the existing queue suite (locate: `grep -rl "PriorityQueue" Tests/Scheduling/`), `Tests/Scheduling/test_scheduled_tasks_db.py`

**Interfaces:**
- Queue rows for definitions carry `"type": "automation_definition"` (the handler-dict key) — set at load, exactly how projections set theirs.
- Consumes slice-1's `is_server_scoped_owner` seam: the accessor's `owner_id="local"` filter plus the existing queue-level guard both hold (defense in depth, pinned).

- [ ] **Step 1: Failing tests** — DB accessor filters (each of the four conditions excludes); `PriorityQueue.load` arms a qualifying definition sorted by `next_run_at` with `type="automation_definition"`; a `transfer_state='to_server_sent'` row never arms; a server-scoped definition never arms.

- [ ] **Step 2: FAIL run.**

- [ ] **Step 3: Implement.** In `app.py` (anchor: the `handlers: dict[str, Handler]` construction): add
```python
handlers["automation_definition"] = AutomationDefinitionHandler(
    db=self.scheduling_service.db,
    app_getter=lambda: self,
    dispatch_service=self.notification_dispatch_service,
    handler_timeout_seconds=get_cli_setting(
        "scheduling", "handler_timeout_seconds", HANDLER_TIMEOUT_SECONDS
    ),
)
```
and, where the scheduler worker starts (anchor `self.scheduler_loop.run()`), one line before it: `self.scheduling_service.db.reconcile_stale_automation_runs(older_than_seconds=...)` guarded try/except (spec §4.1; cutoff = handler timeout + 2× poll interval). Check the loop's `expected_unhandled_types` and startup `report_configuration` still describe reality.

- [ ] **Step 4: Run the queue + loop + db suites, then the FULL `Tests/Scheduling/` suite.** PASS + commit — `feat(scheduling): arm local automation definitions in the scheduler queue`

### Task 6: Run-now + health for local definitions (service level)

**Files:**
- Modify: `tldw_chatbook/Scheduling/services/scheduling_service.py` (`run_automation_now(definition_id) -> dict | None`), `tldw_chatbook/Scheduling/scheduler/loop.py` only if the service needs a loop seam mirroring `run_reminder_now` — read that path first and mirror it
- Create: `tldw_chatbook/Scheduling/automation_health.py`
- Test: `Tests/Scheduling/test_run_now.py` (extend), `Tests/Scheduling/test_automation_health.py`

**Interfaces:**
- `run_automation_now(definition_id)` → dispatches through the SAME handler seam with `trigger_reason="manual"`, `schedule_slot=None` (manual runs never slot-collide, per PR-1 semantics), refusing (return None + reason string) for: server-scoped owner (slice-1 toast copy precedent), lifecycle not configured/paused, transfer pending, health not ready.
- `compute_local_health(app, definition_row) -> tuple[str, str]` — `(health, reason)`: `"capability_unavailable"` when `app.library_rag_search_service` is absent/None; `"permission_required"` when `resolve_execution_target` yields no provider at any layer; else `"ready"`. Read-time only, never stored (spec §7.4). UI surfacing is PR-6; the transfer refusals (PR-5) consume this function.

- [ ] **Step 1: Failing tests** (fakes per suite's existing style). **Step 2: FAIL. Step 3: implement. Step 4: full `Tests/Scheduling/` PASS + commit** — `feat(scheduling): local automation run-now + read-time health`

### Task 7: Unmocked end-to-end + census guard

**Files:**
- Test: `Tests/Scheduling/test_automation_end_to_end.py`

- [ ] **Step 1: Write the integration test** — real tmp_path `ScheduledTasksDB`; definition row created via `create_automation_definition` with a real schedule dict (`interval`, 900s) and `next_run_at` from Task 1; real `PriorityQueue` + `SchedulerLoop` with a test clock; fake app object exposing `library_rag_search_service` (returns two canned rows) and `notify`; `generate_library_rag_answer` faked via its documented `chat=` seam or module monkeypatch (the ONLY faked seams are the two Library boundaries + provider); REAL `NotificationDispatchService` with a recording store (copy `_FakeNotificationStore` from `test_reminder_handler.py`). Drive one `tick`, await the spawned task (handler exposes its pending set — await `asyncio.gather(*handler._pending)`), assert: run row `completed/finding` with the slot string; result row `unread` with `answer_mode="synthesized"` and source_refs from the canned rows; notification row `kind/automation payload` + toast call on the fake app; definition `next_run_at` advanced by 900s.

- [ ] **Step 2: Run it + the FULL `Tests/Scheduling/` suite** — green.

- [ ] **Step 3: Census self-check** — `.venv/bin/python -m pytest Tests/Performance/test_ui_ready_module_census.py -q` locally; the handler module joins the census (registered at boot) but `automation_execution`, the scope module, `schedule_compute` (imported by the handler at top level — allowed: it is stdlib-light; verify it pulls no heavy deps) and the Library seams must NOT appear. If the census breaches, the lazy-import rule was violated somewhere — fix, don't re-pin.

- [ ] **Step 4: Commit** — `test(scheduling): unmocked local recurring_question end-to-end`

---

## After the tasks

Final whole-branch review (most capable model) → fix wave → PR against dev titled `feat(scheduling): local recurring_question execution (handoff PR-2)`, body linking the spec §7 and noting: reviewer read-only git ban, diagnostics-pin regeneration if any new `logger` statements landed in new modules (`scripts/check_persistent_diagnostic_inventory.py --write`), census expectation (handler resident, execution stack lazy), and the sequential-merge discipline (rebase → watch `mergeStateStatus` → merge on CLEAN).
