# Schedules Handoff — PR-6: Results inbox, notification-triggered pull, live verification

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The program's final phase: results become visible and reviewable in a minimal-honest Results tab (per the approved schedules-screen-redesign trim ruling — the redesign later folds this into row-dots + a History group), fresh without polling via the first wiring of the dormant SSE notification observer, with the queue showing ownership, the parked results race-hardening migration, and the spec §10 live verification against a real server.

**Architecture:** A new Results tab consumes the PR-3-built `list_automation_results`/`count_unread_results` (extended for all-owners listing and non-UTC-safe ordering), pushes read/dismiss through the existing `review_automation_result` seam, and adds the resolution slice (DB writes for the so-far-unused `resolution_state` columns; client seams for the server's `POST /definitions/{id}/mark-solved`/`/reopen`). `ServerNotificationEventObserver.observe(handler=...)` — dormant since 18940 slice 3 — gets its first caller: workbench-mounted, filtering `automation_run_*` kinds into a debounced single-flight results pull. Schema v6→v7 adds `UNIQUE(owner_id, server_id)` on `automation_results` (the parked upsert race-hardening; migration follows the warm-boot fast-path pattern). Live E2E runs both transfer directions against a locally-launched tldw_server.

**Tech Stack:** Python ≥3.11, Textual 8.x, httpx-backed tldw_api client (faked in unit/integration tests; REAL in the live task), SQLite, pytest, tmux (live task).

**Spec:** `backlog/docs/spec-2026-08-31-schedules-handoff-parity.md` §5.2, §9, §10 (worktree copy). Planning rulings (binding):
1. **Minimal-honest Results tab** (approved redesign trim): table (both owners, owner shown, `kind="failure"` rows styled distinctly — deciding the PR-2 parked question: failures ARE shown, they're diagnostic) + read/dismiss + unread tab badge + detail render (answer / evidence / source_refs) + Mark solved. No pagination UI v1 (newest-window is what sync mirrors anyway).
2. **Resolution is definition-level** (matches the server's routes): solved/reopen writes `resolution_state`/`resolved_at`/`resolved_by`/`resolved_result_id` on the DEFINITION, with the triggering result id recorded. **Server-owned solved requires connectivity in v1** — offline mark-solved on a server definition returns an honest error (a new offline mutation primitive is deliberately deferred; ledgered).
3. **Observer lifecycle = workbench-scoped v1**: start on schedules-workbench mount, stop on unmount. App-wide residency (rail badges everywhere) rides the redesign program. Debounce: a notification burst coalesces into ONE pull (single-flight worker; a pull already in flight absorbs later triggers).
4. **Owner rendering = row suffix** via the existing `_transfer_row_suffix` append pattern (task_detail.py:434 precedent), hidden at compact width — no new column machinery the redesign would discard.
5. **Census merge bar = COUNT parity with dev's CI run** (sharpened PR-5 lesson — local headroom masks small regressions; breach-list parity is not enough).

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-handoff-pr6`, branch `feat/schedules-results-inbox` off current `origin/dev`. Never the main checkout; NEVER `git stash` (two violations occurred this program; the ban is absolute); `git --no-pager` for reads; foreground pytest only; tmp_path DBs.
- ONE schema migration: v6→v7 (Task 1) — `_CURRENT_SCHEMA_VERSION` in `scheduled_tasks_db.py` goes 6→7 and the warm-boot fast path's threshold moves with it; the new migration module must stay off the warm census exactly like `v5_to_v6.py` (read it as the template).
- Server contract references: `Tests/Scheduling/fixtures/server_responses/automation_endpoints.md` + tldw_server2 @ origin/dev via `git show` (NEVER modify that repo). `/results` has NO `updated_at` filter (reconfirmed) — the bounded page walk stays.
- Diagnostics pin: regenerate `--write` + commit JSON in the SAME commit whenever any logger statement is added/moved/reworded.
- UI change ⇒ update `Docs/User_Guide/` schedules page.
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: v6→v7 migration + results-listing hardening

**Files:** Create `tldw_chatbook/Scheduling/db/migrations/v6_to_v7.py` (template: `v5_to_v6.py`). Modify `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (`_CURRENT_SCHEMA_VERSION` 6→7 + fast-path threshold; `list_automation_results` + `count_unread_results` extensions). Tests: `Tests/Scheduling/test_scheduled_tasks_db.py`, the migrations test file (locate the v5_to_v6 tests as the pattern).

**Behavior:**
- Migration: dedupe any existing `(owner_id, server_id)` duplicates on `automation_results` (keep newest by `updated_at`, log count) then `CREATE UNIQUE INDEX ... ON automation_results(owner_id, server_id) WHERE server_id IS NOT NULL` (partial — local-only rows have NULL server_id and must not collide). The upsert's insert path gains the ON CONFLICT handling this index now enforces (read `upsert_automation_results_from_server` and align — the race it hardens: two pulls inserting the same server row).
- `list_automation_results`: `owner_id=None` lists across owners; ORDER BY `datetime(created_at)` DESC (the parked F7 fix — mixed-offset timestamps sort wrong as strings; apply the same to any other results ORDER BY you find). `count_unread_results(owner_id=None)` likewise spans owners.

- [ ] TDD: migration on a v6 DB with seeded duplicates (newest kept); UNIQUE enforced post-migration; fresh-DB v7 direct create; fast path skips migration imports at v7 (pin like the v5_to_v6 test does); all-owners listing; mixed-offset ordering pinned. FAIL → implement → PASS → commit `feat(scheduling): v7 results unique index + all-owners listing`.

### Task 2: Resolution slice (definition-level solved/reopen)

**Files:** Modify `tldw_chatbook/tldw_api/client.py` (+`mark_scheduled_task_definition_solved(definition_id, result_id=None)` via `POST /api/v1/scheduled-tasks/definitions/{id}/mark-solved`, +`reopen_...` via `.../reopen` — read the server's request/response models in tldw_server2 first and mirror field-for-field), `tldw_chatbook/Notifications/server_notifications_service.py` (two seams under `scheduler.automations.configure`), `tldw_chatbook/Scheduling/services/server_client.py` (retryable wrappers — solving a solved definition no-ops server-side, cite it), `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (+`set_definition_resolution(definition_id, *, state, result_id, resolved_by) -> bool` writing the four columns in one transaction), `tldw_chatbook/Scheduling/services/scheduling_service.py` (+`async resolve_definition(definition_id, solved: bool, result_id: str | None = None) -> ResolveOutcome`: local rows → DB write; server rows online → endpoint + mirror echo; server rows offline → honest `status="error"` "requires a server connection" (ruling 2 — NO mutation queued); the server-wins pull must not clobber a pending nothing — but DO check `upsert_automation_definitions_from_server` passes resolution fields through server-wins like other fields).

- [ ] TDD: client-seam tests (fixture from the server models), DB write, facade both owners + offline honesty, pull carries resolution fields. FAIL → implement → PASS → commit `feat(scheduling): definition resolution slice (mark solved / reopen)`.

### Task 3: Results tab UI

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py` (fourth tab "Results"; label badge via the conflicts precedent `f"Results ({n})" if n else "Results"`; table + detail + actions; refresh after sync and after Task 4's triggered pulls), possibly a small `tldw_chatbook/UI/Screens/scheduling/results_tab.py` widget module if the workbench file's size makes inline composition unreasonable (read the file first; follow whatever the Conflicts tab did). CSS via build flow. `Docs/User_Guide/` schedules page. Tests: new `Tests/UI/test_schedules_results_tab.py` modeled on the conflicts/automations tab tests.

**Behavior:** table columns: kind glyph (finding/failure — failure styled with the error token), title/definition name, owner (suffix style per ruling 4), created (relative), review_state (unread bold + dot). Row actions (tab's keybinding grammar): read, dismiss (both via `service.review_automation_result` — server rows queue the PR-3 pushback mutation automatically), mark-solved (Task 2 facade; enabled only for `kind="finding"` rows whose definition is unresolved; refusal reason shown per UX-073 when offline+server-owned). Detail area renders answer, evidence list, source_refs (the row's stored JSON — read the result-row shape from the fixtures), review metadata. Unread badge updates on every refresh; `Mark all read` action (per-row mutations, the documented fan-out).

- [ ] TDD: badge counts; read/dismiss round-trip incl. server-row mutation queued; failure styling present; solved gating; detail renders the fixture row. FAIL → implement → PASS → commit `feat(scheduling): results inbox tab`.

### Task 4: Owner suffix + notification-triggered pull (first observer wiring)

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/task_detail.py`/`schedules_workbench.py` (owner suffix on queue rows via the `_transfer_row_suffix` append pattern; hidden under the existing compact-width mechanism `SCHEDULES_COMPAT/COMPACT_WORKBENCH_MAX_WIDTH` — read it), `schedules_workbench.py` (observer lifecycle: start `ServerNotificationEventObserver.observe(handler=...)` on mount when a server connection is configured, stop on unmount — read the observer's API end to end first, it has never had a caller; handler filters `automation_run_*` kinds → schedules a debounced single-flight results-pull worker (`run_worker(group=..., exclusive=True)` — the worker-collision lesson; a trigger while a pull runs sets a rerun flag, no pile-up) → refreshes the Results tab + badge). Census: the observer import must not add boot-resident modules (workbench isn't boot-resident, but verify).

- [ ] TDD: suffix rendering both widths; observer started/stopped with mount lifecycle (fake observer); an `automation_run_completed` event triggers exactly one pull under a burst of three; non-automation kinds ignored; pull failure surfaces via the existing sync-error path without killing the observer. FAIL → implement → PASS → commit `feat(scheduling): queue owner suffix + notification-triggered results pull`.

### Task 5: Integration E2E + gates

**Files:** Create `Tests/Scheduling/test_results_inbox_end_to_end.py`.

- [ ] E2E (real tmp_path DB, real service/engine, fake schema-validating server client): (a) sync seeds server results → tab data source lists them, badge counts unread; (b) read pushes back (fake sees the review POST after replay); (c) notification event → debounced pull → new result appears; (d) mark-solved server round-trip + local-only variant; (e) v7 UNIQUE prevents duplicate mirror insert under a simulated double-pull. Then FULL gates: `Tests/Scheduling/ -q`; the schedules UI files incl. the new tab tests; census with **COUNT parity vs dev's CI number** (ruling 5 — compare against the latest dev Perf Guard run's count, not just local pass); diagnostics pin; ruff. Commit `test(scheduling): results inbox end-to-end`.

### Task 6: Live verification (spec §10 — the program gate)

**Files:** `Docs/User_Guide/` stamps; ledger + task-18940 progress note updates; NO code changes expected (defects found go back through a fix round, honestly recorded).

**Procedure** (lessons-live-verification governs; record what was and was NOT verified):
1. Launch tldw_server2 locally: `make install-local && make setup-local-single && make start-local-single` → `127.0.0.1:8000`; key via `make show-api-key`.
2. Scratch chatbook profile: `TLDW_CONFIG_PATH=<scratch>/config.toml` with `[general] users_name="verify_pr6"`, `[tldw_api] base_url="http://127.0.0.1:8000" api_key="<key>"` (delete `~/.local/share/tldw_cli/verify_pr6` after).
3. Drive the real TUI in tmux (the repo's verify skill recipe: `tmux -L verify new-session -d -x 235 -y 52 '.venv/bin/python -m tldw_chatbook.app'`, capture-pane, SGR clicks; cold start ~12s).
4. Verify: (a) local recurring question authored → transferred to server (badge sequence, server accepts) → run-now on server → result syncs down unread → read pushes back (server shows read); (b) server-owned definition → transfer to local → dormant→armed after release → local run-now → result in inbox; (c) reminder round-trip with link-and-match fields visible server-side; (d) the flagged unknown: one-time reminder with past `run_at` transferred → observe server behavior, record it; (e) notification-triggered pull observed live (run fires server-side while the workbench is open → inbox refreshes without pressing "s").
5. Update the ledger + task-18940 (AC#8-adjacent evidence) + `backlog/docs/lessons-live-verification.md` if a trap surfaced. Honest recording of every leg that could not be verified and why.

- [ ] Execute + record. Commit `docs(scheduling): live verification record (handoff program gate)`.

---

## After the tasks
Final whole-branch review (opus) → one fix wave → PR `feat(scheduling): results inbox + notification-triggered pull + live verification (handoff PR-6)` → paged bot-comment read (per_page=100, count `in_reply_to_id == null`) → adjudicate → sequential rebase-watch-merge (peer's gate recipe: up-to-date AND all check-suites on the EXACT head sha completed AND required runs succeeded). Post-merge: program close-out — memory update, TASK-18940 status review, redesign program becomes next.
