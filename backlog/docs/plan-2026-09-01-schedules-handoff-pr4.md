# Schedules Handoff — PR-4: Authoring modal + owner-dispatched preview

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user can create and edit `recurring_question` definitions in one modal for either owner ("Runs on: This device | Server"), with a preview seam dispatched by owner returning identical field-addressed errors — the spec §8 authoring slice, which also satisfies TASK-18940 AC#2's client half by construction.

**Architecture:** A new pure validator module ports the server's four authoring validators; a local preview service wraps them plus `compute_next_run_at` to return the server's preview shape. New client seams (tldw_api schemas → client → `ServerNotificationsService` under the existing `scheduler.automations.configure` action → `SchedulingServerClient`) implement the server's strict preview-then-commit contract (`POST /previews` → `POST /definitions {preview_id, initial_lifecycle}`; update = update-mode preview → `PATCH /definitions/{id} {preview_id}`). A `SchedulingService` authoring facade dispatches preview and save by owner; offline server-owned saves queue `automation_definition` create/update pending mutations, and `SyncEngine` gains their replay (the preview→commit two-step at push time from the stored payload — never a stored preview id, spec §5.1). A new `AutomationDefinitionForm` modal (per ADR-099: modal, upgraded in place) reuses TASK-23102's forgiving schedule widgets; `ReminderForm` gains the same "Runs on" selector riding the existing reminder create-mutation path.

**Tech Stack:** Python ≥3.11, Textual 8.x, httpx-backed tldw_api client (faked in tests), SQLite, pytest.

**Spec:** `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` §8 (+§5.1 push, §7.1 validators, §7.4 health). Planning rulings (recorded here, binding for this PR):
1. **Definition create/update push replay moves from PR-5 into this PR** — offline server-owned authoring would otherwise queue mutations nothing replays (an inert feature, the exact class this program keeps catching). PR-5 keeps pause/resume/archive lifecycle mutations + the transfer actions.
2. **Policy action:** preview/create/update reuse `scheduler.automations.configure` — no new registry actions; preview is pointless without configure rights, and PR-3's review seam set the reuse precedent.
3. **Save flow:** Save always runs the preview seam first; validation errors render field-addressed and block the save; a clean preview commits (server: preview_id two-step; local: DB write). An explicit "Preview" button runs the same seam without committing.
4. **New affordances:** the Queue tab's "+ New" button offers a two-item choice (Reminder / Recurring question); the Automations tab header gains its own new-question button.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-handoff-pr4`, branch `feat/schedules-authoring-modal` off dev `fce177449`. Never the main checkout; NEVER `git stash`; `git --no-pager` for reads; foreground pytest only; tmp_path DBs (never :memory:).
- The server contract reference is IN-REPO and byte-accurate: `Tests/Scheduling/fixtures/server_responses/automation_endpoints.md` — read it before writing any schema or client method. Server source of truth for validator porting: `/Users/macbook-dev/Documents/GitHub/tldw_server2` at `origin/dev` (`tldw_Server_API/app/services/scheduled_task_automation_service.py`) — read via `git -C /Users/macbook-dev/Documents/GitHub/tldw_server2 show origin/dev:<path>`; NEVER modify that repo.
- Pending-mutation payloads store full definition payloads, never preview ids (24h-TTL, single-consume — spec §5.1).
- Diagnostics pin: any wave that adds/moves/rewords logger statements regenerates `scripts/check_persistent_diagnostic_inventory.py --write` in the same commit (a mere rewording drifts the digest — this failed CI on PR-3 twice).
- Boot census (ADR-097, limit 972 NEVER rises): the modal and validator modules must not become boot-resident — modal imported lazily where `ReminderForm` is (check how the workbench imports it), validators live under `Scheduling/` (already resident package is fine, new module import happens via already-lazy paths); verify with `Tests/Performance/test_ui_ready_module_census.py -q` in the final task.
- UI change ⇒ update the matching `Docs/User_Guide/` page (or its "Verified against" stamp) — CLAUDE.md rule.
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: Validator port + local preview service (pure)

**Files:** Create `tldw_chatbook/Scheduling/automation_validation.py`, `tldw_chatbook/Scheduling/automation_preview.py`. Tests: `Tests/Scheduling/test_automation_validation.py`, `Tests/Scheduling/test_automation_preview.py`. Fixture: `Tests/Scheduling/fixtures/server_responses/automation_preview_response.json` (create — see below).

**Interfaces (consumed by Tasks 4/5):**
- `automation_validation.py`: port `_validate_schedule`, `_validate_recurring_question_config`, `_normalize_finding_policy`, `_normalize_retention_policy` from the server file named in Global Constraints, as-is (public names, drop the leading underscore): each returns/accumulates field-addressed errors shaped `{"field": str, "code": str, "message": str}` exactly as the server emits them. Reuse the already-ported `normalize_recurring_question_scope` (`Scheduling/recurring_question_scope.py`) — do not duplicate it.
- `automation_preview.py`: `preview_automation_definition(payload: dict) -> AutomationPreview` — pure, no I/O: runs the validators, computes `normalized_config`, `validation_errors`, `warnings`, and `schedule_preview` (next 3 occurrences via `compute_next_run_at`, matching the server's `schedule_preview` dict shape — read the server's preview assembly to copy the keys), `status` = `"valid"`/`"invalid"`. Fills the existing `Scheduling/models.py::AutomationPreview` model (shell already shaped like the server response — verify field names against it and the endpoints doc).

**Fixture-parity contract (spec §7.1 drift rule):** create `automation_preview_response.json` by hand-assembling a realistic `ScheduledTaskPreviewResponse` for a valid `recurring_question` create-preview AND one invalid case (e.g. bad schedule kind), with field values derived from reading the server's service code — record in the fixture file's neighbor `automation_endpoints.md` style. Parity test: feed the fixture's request payload through `preview_automation_definition` and assert `validation_errors` (field+code), `status`, and `schedule_preview` keys match the fixture.

- [ ] TDD: parity tests + direct validator unit tests (each validator: one accept, one reject per distinct code path the server has). FAIL → implement → PASS → commit `feat(scheduling): port authoring validators + local preview service`.

### Task 2: Server preview/create/update client seams

**Files:** Modify `tldw_chatbook/tldw_api/scheduled_tasks_automation_schemas.py` (+`ScheduledTaskPreviewCreateRequest`, `ScheduledTaskPreview` response model, `ScheduledTaskDefinitionCreateRequest`, `ScheduledTaskDefinitionUpdateRequest` — field-for-field from the endpoints doc, datetimes as `str | None` per file style), `tldw_chatbook/tldw_api/client.py` (+`preview_scheduled_task_definition(request) -> ScheduledTaskPreview` via `POST /api/v1/scheduled-tasks/previews`; +`create_scheduled_task_definition(preview_id, initial_lifecycle="configured") -> ScheduledTaskDefinition` via `POST .../definitions`; +`update_scheduled_task_definition(definition_id, preview_id) -> ScheduledTaskDefinition` via `PATCH .../definitions/{id}` — beside the slice-2 automation methods, mirroring their construction), `tldw_chatbook/Notifications/server_notifications_service.py` (+three seams under the `scheduler.automations.configure` policy action — ruling 2), `tldw_chatbook/Scheduling/services/server_client.py` (+wrappers; preview and create are retryable — the server's payload-hash idempotency makes replay retry-safe, spec §5.1; cite that in the wrapper comment).

- [ ] TDD in the existing client-test style (locate via `grep -rl "review_scheduled_task_result" Tests/`), using the Task 1 fixture for the preview response. FAIL → implement → PASS → commit `feat(scheduling): preview/create/update definition client seams`.

### Task 3: Definition create/update push replay in SyncEngine

**Files:** Modify `tldw_chatbook/Scheduling/services/sync_engine.py`, `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (one new method, below). Test: `Tests/Scheduling/test_sync_engine.py`, `Tests/Scheduling/test_scheduled_tasks_db.py`.

**Behavior:**
- `_DEFINITION_PRIMITIVE` pending mutations (payload `{"action": "create"|"update", "definition_payload": {...}, "server_definition_id": str|None}`) replay in the push phase that runs BEFORE the pulls (same ordering rationale as review pushback): `create` → preview(mode=create) → on `status="valid"` → create-with-preview_id → new DB method `adopt_server_definition_identity(local_id, server_item) -> bool` sets `server_id` + applies the server echo's server-wins fields in ONE transaction; on `status="invalid"` → the mutation is cleared and the validation errors recorded via the existing `sync_conflicts`/`_record_sync_error` surface (a payload the server rejects will never succeed by retrying — log warning with the field codes). `update` → preview(mode=update, definition_id=server id) → PATCH; missing `server_definition_id` (authored offline, never synced) → convert to create, mirroring the reminder `_push_mutation` precedent at `sync_engine.py` (read it); `ServerClientNotFoundError` on update → convert to create likewise. Network/retryable errors leave the mutation for retry.
- Pull-ordering note for the implementer: replay runs before the definitions pull, so a successful create sets `server_id` before the pull upsert sees the server's copy — the upsert then matches `(owner_id, server_id)` instead of inserting a duplicate mirror. Assert this in a test (create replay + same-cycle pull containing the new definition → exactly one local row).

- [ ] TDD: create replay happy path (mutation cleared, server_id adopted, one row after same-cycle pull); invalid-preview clears with recorded error; update replay; offline-create conversion; NotFound conversion; retryable error retains. FAIL → implement → PASS → commit `feat(scheduling): definition create/update push replay`.

### Task 4: SchedulingService authoring facade

**Files:** Modify `tldw_chatbook/Scheduling/services/scheduling_service.py`. Test: `Tests/Scheduling/test_scheduling_service.py`.

**Interfaces (consumed by Task 5's modal):**
- `async preview_definition(payload: dict, owner_id: str) -> AutomationPreview` — local owner → Task 1's pure service (via `asyncio.to_thread` only if profiling says so; it is pure compute, direct call is fine); server owner → Task 2 seam; a server seam network failure returns a local preview annotated with a warning `{"field": "_owner", "code": "server_unreachable", "message": ...}` so the modal still renders schedule feedback offline.
- `async save_definition(payload: dict, owner_id: str, definition_id: str | None = None) -> SaveDefinitionOutcome` (small dataclass: `status ∈ {"saved", "queued", "invalid", "error"}`, `errors: list`, `definition_id: str | None`): always previews first (ruling 3); invalid → return errors, write nothing. Local owner → compute `next_run_at` (`compute_next_run_at`) and call the existing `create_automation_definition`/`update_automation_definition` DB methods (they exist — read their signatures first). Server owner online → the preview→commit two-step, then mirror the echo locally via the existing `upsert_automation_definitions_from_server`. Server owner with the seam unreachable → write the local row (owner=server owner, `server_id=None`, `lifecycle="configured"`) + record the Task 3 pending mutation in the same transaction pattern `update_result_review(..., pending_mutation=...)` established (extend the DB create method with the same optional `pending_mutation` kwarg if needed — read that precedent first).
- v1 scope guard: `family` is hard-pinned to `recurring_question` here — reject anything else with `status="invalid"` (agent_task authoring rides the follow-up program).

- [ ] TDD: both owners × create/edit; invalid blocks; offline server save queues exactly one mutation atomically with the row; family guard. FAIL → implement → PASS → commit `feat(scheduling): owner-dispatched authoring facade`.

### Task 5: AutomationDefinitionForm modal + workbench wiring + reminder "Runs on"

**Files:** Create `tldw_chatbook/UI/Screens/scheduling/forms/automation_definition_form.py`. Modify `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (New-button choice + Automations-tab button + form-result handling), `tldw_chatbook/UI/Screens/scheduling/forms/reminder_form.py` ("Runs on" selector), the schedules CSS source the workbench uses (find it — NEVER hand-edit the bundle; rebuild via `build_css.py` if the repo's flow requires), `Docs/User_Guide/` schedules page stamp. Tests: `Tests/UI/test_automation_definition_form.py` (new, modeled on `Tests/UI/test_reminder_form.py` — read it first for the harness pattern), extend `Tests/UI/test_reminder_form.py`.

**Behavior:**
- `AutomationDefinitionForm(ModalScreen)` mirroring `ReminderForm`'s structure (scrolling container, key hints, discard guard — ADR-099 idiom parity): fields = name, question (multiline), scope mode + sources multi-select (three checkboxes: Media / Notes / Chats — v1; collections/tags deferred per spec §8), schedule kind + per-kind fields REUSING the 23102 preset widgets from `reminder_form.py` (extract-to-share only if reuse forces it; prefer importing the existing helpers), generation mode, finding-policy preset select, notification toggle, optional provider/model pin, "Runs on: This device | Server (<id>)" select (default = current screen owner; server option present only when a server owner exists — read how the workbench knows its owners). Preview button + save-runs-preview (ruling 3); `validation_errors` map onto widgets by `field` (error CSS class + message line under the field; unmatched fields render in a form-level error area).
- Workbench: "+ New" (`#scheduling-new-task`) offers Reminder / Recurring question (smallest house-pattern chooser — check for an existing tiny-choice modal/menu before building one); Automations tab header gets a new-question button; form results route to Task 4's facade; Automations list refreshes after save.
- `ReminderForm`: add the same "Runs on" selector; a server-owned reminder create writes the local row under the server owner and rides the EXISTING reminder create-mutation path (verify by reading how `create_reminder` records mutations — pin with one test asserting a server-owned create queues a reminder mutation).

- [ ] TDD: form validation display (invalid preview highlights the named field), save wiring both owners (facade faked), chooser routing, reminder owner selector. FAIL → implement → PASS → commit `feat(scheduling): recurring-question authoring modal + owner selector`.

### Task 6: Sources-readable health check + E2E + gates

**Files:** Modify `tldw_chatbook/Scheduling/automation_health.py` (the PR-2 parked "scoped sources readable" check: the definition's normalized scope sources map through the SAME source mapping `automation_execution.py` uses — import it, don't re-declare; unreadable/missing DB ⇒ `capability_unavailable` with a reason naming the source). Test: extend `Tests/Scheduling/test_automation_health.py`; create `Tests/Scheduling/test_authoring_end_to_end.py`.

- [ ] TDD health check (readable, one-source-unreadable, deps-missing precedence — read the existing check order first). Commit `feat(scheduling): sources-readable local health check`.
- [ ] E2E (real tmp_path DB, fake server client from fixtures): (a) local authoring → `save_definition` → row exists with computed `next_run_at` → a `SchedulerLoop` tick picks it up (reuse the PR-2 E2E harness); (b) server-owned authoring with the seam DOWN → queued mutation → `sync_now` with the seam UP replays preview→create → `server_id` adopted, mutation cleared, one row. Then full `Tests/Scheduling/ -q` + `Tests/UI/test_reminder_form.py Tests/UI/test_automation_definition_form.py -q` green; census ≤ pin; diagnostics pin clean. Commit `test(scheduling): authoring end-to-end`.

---

## After the tasks
Final whole-branch review (opus) → one fix wave → PR `feat(scheduling): recurring-question authoring modal + owner-dispatched preview (handoff PR-4)` → paged bot-comment read (`per_page=100`, ALL pages, count `in_reply_to_id == null`) → adjudicate → sequential rebase-watch-merge. Post-merge: pointer note in TASK-18940's progress log (AC#2 client half satisfied for recurring_question by this PR).
