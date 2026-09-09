# Schedules Handoff — PR-3: Automation Sync Primitives

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Server automation definitions and results mirror into the local store under the ADR-018 sync discipline, and local review-state changes push back — the "results sync down, review pushes up" half of the handoff contract (spec §5).

**Architecture:** `SyncEngine` (currently reminder-only) gains two pull mirrors: definitions (server-wins upsert keyed `(owner_id, server_id)`, lifecycle mirrored never deleted) and results (bounded newest-pages walk — the server's `/results` endpoint exposes no `updated_at` filter, verified at `origin/dev`), plus one push primitive: `automation_result_review` pending mutations replayed to `POST /results/{id}/review`. New client seams stack the slice-2 pattern (tldw_api schemas → client methods → `ServerNotificationsService` policy-gated seams → `SchedulingServerClient` wrappers). Also closes the PR-2 parked item: the schedule advance stops auto-bumping `version`.

**Tech Stack:** Python ≥3.11, httpx-backed tldw_api client (faked in tests), SQLite, pytest.

**Spec:** `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` §5 (worktree copy). Three consumption-ordered deviations, ruled at planning: (1) definition-PUSH mutation replay (create/update/lifecycle) moves to PR-5, its first real producer (transfer) — PR-3 ships the pull mirrors and the review pushback; (2) the notification-triggered results pull moves to PR-6 (the observer seam is transport-layer plumbing; every `sync_now` pulls results, and freshness-on-toast only matters once the inbox UI exists); (3) incremental results sync is a bounded newest-4-pages walk (200 rows) per sync — review-state drift older than that window waits for a deeper pull, ledgered as the §5.2 limitation the missing server filter forces.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-handoff-pr3`, branch `feat/schedules-automation-sync` off dev `16d3e411d`. Never the main checkout; NEVER `git stash`; `git --no-pager` for reads; foreground pytest only; tmp_path DBs.
- Sync discipline is ADR-018's: server-wins on pull; pending mutations replay on push; conflicts into the existing `sync_conflicts` surface; sync errors via `_record_sync_error`. Read `sync_engine.py` END TO END before touching it — the reminder phases are the template.
- Server response shapes come from the recorded fixtures dir (`Tests/Scheduling/fixtures/server_responses/` — extend it; the slice-2 files are the precedent) and must match tldw_server `origin/dev @ 5921014aa9`'s `ScheduledTaskResultResponse` / `ScheduledTaskResultListResponse` (fields incl. `answer_mode`, `confidence`, `source_refs`, `dedupe_key`, `review_state`, `reviewed_at/by`, `review_note`, pagination `items/total/limit/offset/has_more`).
- Diagnostics pin: if any task adds/moves logger statements, run `scripts/check_persistent_diagnostic_inventory.py --write` in that task's commit.
- Boot census (ADR-097): no new boot-resident modules — sync code lives in already-resident `Scheduling/services/`; verify with the census test in the final task.
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: Non-versioning schedule advance (PR-2 parked item)

**Files:** Modify `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (`update_automation_definition`), `tldw_chatbook/Scheduling/scheduler/handlers/automation_handler.py` (the advance call in `_dispatch`). Tests: `Tests/Scheduling/test_scheduled_tasks_db.py`, `Tests/Scheduling/test_automation_handler.py`.

**Interfaces:** `update_automation_definition(definition_id, *, bump_version: bool = True, **kwargs) -> bool` — default preserves every existing caller's behavior; the handler's advance passes `bump_version=False` (a `next_run_at` advance is not an edit; version churn pollutes conflict detection and inflates ~35k/yr on a 15-min interval — PR-2 final-review parking note).

- [ ] TDD: test that an advance-style update (`next_run_at` only, `bump_version=False`) leaves `version` unchanged while a default update still bumps; handler test asserting `version` stable across a dispatch. FAIL → implement → PASS → commit `fix(scheduling): schedule advance no longer bumps definition version`.

### Task 2: Results client seams (tldw_api → notifications service → scheduling client)

**Files:** Modify `tldw_chatbook/tldw_api/scheduled_tasks_automation_schemas.py` (+`ScheduledTaskResult`, `ScheduledTaskResultList` mirroring the server response models — copy field-for-field from the Global Constraints list; datetimes as `str | None` matching the file's existing style), `tldw_chatbook/tldw_api/client.py` (+`list_scheduled_task_results(limit=50, offset=0, definition_id=None, review_state=None) -> ScheduledTaskResultList` via `GET /api/v1/scheduled-tasks/results`; +`review_scheduled_task_result(result_id, review_state, review_note=None) -> ScheduledTaskResult` via `POST .../results/{id}/review` — place beside the slice-2 automation methods and mirror their construction), `tldw_chatbook/Notifications/server_notifications_service.py` (+`list_scheduled_automation_results(...)` under the same policy action as `list_scheduled_automations`; +`review_scheduled_automation_result(...)` under the configure-class action — READ how `run_scheduled_automation_now` chose its action and mirror that reasoning), `tldw_chatbook/Scheduling/services/server_client.py` (+wrappers: list is `is_read=True` retryable; review is retryable — replaying the same review state is idempotent).

- [ ] TDD against the existing client-test style (locate: `grep -rl "list_scheduled_automations" Tests/`); add fixture `Tests/Scheduling/fixtures/server_responses/automation_results_list.json` with 2 results incl. one `review_state: "read"`. FAIL → implement → PASS → commit `feat(scheduling): results list + review client seams`.

### Task 3: DB upserts for pulled definitions and results

**Files:** Modify `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py`. Test: `Tests/Scheduling/test_scheduled_tasks_db.py`.

**Interfaces (SyncEngine consumes in Task 4):**
- `upsert_automation_definitions_from_server(owner_id: str, items: list[dict]) -> dict` — for each server item: match local row by `(owner_id, server_id)`; absent → insert a mirror row (map server fields onto the v4 columns; `server_id` = server `id`; local `id` = fresh UUID; JSON fields through the existing serialization); present → server-wins update of every server-carried field EXCEPT `transfer_state` (spec §6 parked finding: a server payload must never clear a local transfer marker — write the exclusion + a comment citing the ledger). Archived lifecycle mirrors, never deletes. Returns `{"inserted": n, "updated": n}`.
- `upsert_automation_results_from_server(owner_id: str, items: list[dict]) -> dict` — match by `(owner_id, server_id)`; absent → insert (all fields; local `id` fresh UUID; `run_id` = the server's run id string, plain TEXT per spec §4.2; keep the server `dedupe_key` — on a dedupe-key UNIQUE conflict with a locally-created row, skip + count `skipped_dedupe`); present → update ONLY the review fields (`review_state`, `reviewed_at/by`, `review_note`, `updated_at`) and ONLY when no pending `automation_result_review` mutation exists for that row (`get_pending_mutations` filtered by primitive + payload's result id — a local unpushed review outranks the mirror until it replays; comment the rule). Returns counts.

- [ ] TDD each branch (insert, server-wins update, transfer_state preserved, review-fields-only update, pending-review skip, dedupe skip). FAIL → implement → PASS → commit `feat(scheduling): server-mirror upserts for automation definitions and results`.

### Task 4: SyncEngine primitives — definitions pull, results pull, review pushback

**Files:** Modify `tldw_chatbook/Scheduling/services/sync_engine.py`, `tldw_chatbook/Scheduling/services/scheduling_service.py` (only if `sync_now`'s outcome plumbing needs the new counts surfaced — read first). Test: `Tests/Scheduling/test_sync_engine.py`.

**Behavior:**
- New module constants `_DEFINITION_PRIMITIVE = "automation_definition"`, `_RESULT_REVIEW_PRIMITIVE = "automation_result_review"`, `_RESULTS_PAGE_SIZE = 50`, `_RESULTS_MAX_PAGES = 4`.
- `sync_now` gains, after the reminder phase and inside the same error-containment shape (each new phase in its own try/except → `_record_sync_error`, never aborting the phases after it):
  1. **Review pushback** (before pulls, so a fresh mirror doesn't clobber unpushed reviews): replay each `automation_result_review` pending mutation → `server_client.review_scheduled_task_result(server_result_id, review_state, review_note)`; success clears the mutation; `ServerClientNotFoundError` clears it too (result retired server-side — log info); other errors leave it for retry.
  2. **Definitions pull**: page `list_automation_definitions` (the existing wrapper already follows `has_more` — read it; reuse, don't duplicate) → `upsert_automation_definitions_from_server`.
  3. **Results pull**: up to `_RESULTS_MAX_PAGES` pages of `list_scheduled_task_results` newest-first → `upsert_automation_results_from_server`; stop early when a page returns fewer than `_RESULTS_PAGE_SIZE` or `has_more` is false. Log (info) when the page cap was hit with more remaining (no-silent-caps).
- `pull()` (the read-only variant) gains the two pulls, no pushback.
- Runtime-mode refusals keep the existing "not applicable, never a persisted error" discipline (`ServerClientPolicyError` → info log, return).

- [ ] TDD with the suite's existing fake-server-client style: pushback clears mutation / NotFound clears / error retains; definitions upserted; results pages walked with early stop and cap log; a phase failure doesn't abort later phases; policy refusal stays non-error. FAIL → implement → PASS → commit `feat(scheduling): sync mirrors for automation definitions/results + review pushback`.

### Task 5: Local review entry point

**Files:** Modify `tldw_chatbook/Scheduling/services/scheduling_service.py`. Test: `Tests/Scheduling/test_scheduling_service.py`.

**Interfaces:** `async review_automation_result(result_id: str, review_state: str, review_note: str | None = None) -> bool` — validates `review_state` against `ReviewState` values (reject junk, return False + warning log); `asyncio.to_thread(update_result_review...)`; when the row carries a `server_id`, `record_pending_mutation(local_id, _RESULT_REVIEW_PRIMITIVE, owner, {"server_result_id": ..., "review_state": ..., "review_note": ...})` — payload stores the SERVER id so replay never needs a local join (spec §5.1's payload-not-reference rule). No queue notification (results don't arm anything).

- [ ] TDD: local-only row updates without a mutation; server-mirrored row records one; junk state refused. FAIL → implement → PASS → commit `feat(scheduling): review_automation_result with server pushback enqueue`.

### Task 6: Sync end-to-end + census guard

**Files:** Test `Tests/Scheduling/test_automation_sync_end_to_end.py`.

- [ ] One test, real tmp_path DB + fake server client built from the recorded fixtures: seed a server-mirrored result via a first `sync_now`; locally review it `dismissed` via `review_automation_result`; run `sync_now` again with the fake asserting the review call arrived AND the pull payload still claiming `unread` — assert the local row stays `dismissed` after the pull-phase upsert ran BEFORE the push replay cleared... (order per Task 4: pushback runs FIRST, so after it the pending mutation is gone and the mirror's stale `unread` would overwrite — assert the fake's pull payload is served with the POST-review state as a real server would echo, and separately assert the pending-review-skip branch via a fake whose push fails). Then: full `Tests/Scheduling/ -q` green; `Tests/Performance/test_ui_ready_module_census.py -q` ≤ pin; diagnostics pin check clean (regenerate if this PR added log lines).
- [ ] Commit `test(scheduling): automation sync end-to-end`.

---

## After the tasks
Final whole-branch review (opus) → one fix wave → PR `feat(scheduling): automation definitions/results sync + review pushback (handoff PR-3)` → paged bot-comment read (ALL pages, count `in_reply_to_id == null`) → adjudicate → sequential rebase-watch-merge.
