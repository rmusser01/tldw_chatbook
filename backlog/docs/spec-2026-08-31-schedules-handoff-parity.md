# Spec: Scheduled-task handoff parity between chatbook and tldw_server

Date: 2026-08-31
Status: Draft — awaiting owner approval
Baselines: tldw_server `origin/dev` @ `5921014aa9`; tldw_chatbook `origin/dev` @ `30a70ef5c`
Related: ADR-018 (hybrid storage/sync), ADR-077 (server-offloaded execution; this spec's ADR will amend its §1), ADR-099 (editor shape: modal, upgraded in place), TASK-18940 (this program absorbs its remaining authoring slice), tasks 23100–23111 (UX burn-down this spec builds on)
Companion analysis: "Schedules Against the Reference" rev 2 (artifact 6283950d)

## 1. Purpose

A scheduled task authored on either side can be executed by the other:
created in chatbook and handed to tldw_server, created on the server and
handed to chatbook, or authored on one while offline and picked up by the
other on the next sync. Execution follows ownership (ADR-077 decision 1,
unchanged); this program makes ownership a per-task property that can be
transferred, and gives chatbook the local execution and storage parity the
transfer target requires.

## 2. Locked decisions

| # | Decision | Chosen by |
|---|---|---|
| 1 | V1 families: **reminders + recurring_question**; agent_task is the immediate follow-up — every seam is family-generic so it drops in without rework | owner |
| 2 | **Per-task owner + transfer action** (not screen-scoped, not mirror-everywhere) | owner |
| 3 | This program **absorbs TASK-18940's remaining authoring slice** (create/preview/submit) | owner |
| 4 | Server-run **results sync down** into the local results store; `review_state` changes push back | owner |
| 5 | Transfer mechanism: **create-on-target + tombstone-on-source**, composed from existing primitives, eventually-atomic, zero server-side changes in v1 (Approach 1) | owner |
| 6 | Editor shape: **modal, upgraded in place** (ADR-099 — decided before this program; followed, not revisited) | ADR-099 |

## 3. Invariant

**At most one side is armed for a given task at any instant.** Transfers
may be slow (offline-queued, retried, failed); they may never produce
double execution. Every state and failure leg in §6 exists to serve this.

## 4. Data model (scheduled-tasks DB, schema v3 → v4)

The schema version is re-verified against `origin/dev` at merge time
(known collision class; see lessons-backlog-hygiene and the v19/v20
ChaChaNotes incidents).

### 4.1 New table `automation_runs` — local execution only in v1

Mirrors the server's `RunRow`:

`id` (uuid PK), `server_id` TEXT NULL (**reserved; nothing writes it in
v1** — avoids a v5 migration when run-sync arrives), `owner_id`,
`definition_id`, `definition_version`, `trigger_reason`
(`scheduled|manual`), `status`
(`queued|running|completed|failed|skipped|cancelled|timed_out`), `outcome`
(`finding|no_match|partial|degraded|none`), `schedule_slot` TEXT NULL,
`scope_snapshot` / `finding_policy_snapshot` / `rag_request_snapshot` /
`run_summary` / `evidence_summary` JSON, `failure_reason` JSON NULL,
`created_at`, `updated_at`, `started_at`, `ended_at`.

- `UNIQUE(definition_id, definition_version, schedule_slot)` is the slot
  dedupe. SQLite treats NULLs as distinct, so manual runs
  (`schedule_slot` NULL) never collide — verified semantics, no
  special-casing.
- `timed_out` is first-class locally (matches `TaskStatus.TIMED_OUT` /
  TASK-18939). When *displaying* server runs (fetched on demand, §5.3),
  the adapter maps server `status=failed` +
  `run_summary.legacy_status="timed_out"` → `timed_out`. Display-layer
  translation only; nothing is persisted.
- **Retention**: runs are pruned on write to the newest 200 per
  definition (an every-15-minutes task would otherwise write ~35k
  rows/year). Results are never pruned by this rule.
- **Stale-run reconciliation** (server precedent:
  `reconcile_stale_runs`): on scheduler start, rows still `running`
  older than the execution timeout + grace are marked `failed` with
  `failure_reason={"code": "interrupted"}` — an app killed mid-run
  must not leave a phantom in-flight run in the UI.

### 4.2 New table `automation_results` — both owners

Mirrors the server's `ResultRow`:

`id` (uuid PK), `server_id` TEXT NULL (server result id when synced
down), `owner_id`, `definition_id`, `run_id` TEXT (**plain TEXT, not a
foreign key** — a synced server result carries the server's run id, which
has no local row), `kind` (`finding|failure`), `title`, `summary`,
`answer` JSON NULL, `answer_mode` (`synthesized|evidence_only|none`),
`confidence` JSON, `source_refs` JSON, `dedupe_key`
(`UNIQUE(owner_id, dedupe_key)`), `visibility_destination` JSON,
`review_state` (`unread|read|dismissed`, default `unread`),
`reviewed_at`, `reviewed_by`, `review_note`, `created_at`, `updated_at`.

Local results use the server's dedupe recipe
(`{definition_id}:{run_id}:{kind}` hashed the same way) so a future
result push-up cannot collide. Synced-down results upsert keyed on
`(owner_id, server_id)`; the `dedupe_key` uniqueness is the backstop,
not the upsert key.

### 4.3 `automation_definitions` — eight reference fields + one local

Add: `disabled_lock_kind` (`none|admin|security|system`; locally only
`none|system` are ever written), `disabled_reason`, `resolution_state`
(`open|solved`, default `open`), `resolved_at`, `resolved_by`,
`resolved_result_id`, `finding_policy` JSON (default
`{"preset":"balanced_findings"}`), `retention_policy` JSON (default
`{"mode":"default"}`).

Add local: **`next_run_at` TEXT NULL** — definitions become real queue
rows (§7.2) and the UI's Next Run column needs it. Computed on write and
recomputed **at spawn time** (not run completion — a slow run must not
stretch the schedule), advancing **from now** (elapsed slots are
skipped, the late slot runs once, intermediates are not replayed —
matching both the server's misfire discipline and TASK-18937).

Local definition updates **bump `version`** on each accepted edit,
matching the server — audit before/after entries and slot keys
reference it.

Add local: `transfer_state` TEXT NULL (§6). Also added to
`reminder_tasks`.

### 4.4 Model changes (`Scheduling/models.py`)

- `AutomationDefinition`: + the eight fields above (+ `next_run_at`,
  `transfer_state`).
- `AutomationPreview`: `validation_errors` and `warnings` retype
  `list[str]` → `list[dict[str, Any]]` (the server returns
  `{field, code, message}` objects — the current type fails on the first
  real response); + `risk_class`.
- New: `AutomationRun`, `AutomationResult` Pydantic models; `RunStatus`,
  `RunOutcome`, `ReviewState` enums matching the server literals plus the
  local `timed_out`.
- `ScheduleKind` (reminders) is **untouched**. Definitions carry their
  schedule as the server's `schedule` dict
  (`kind: one_time|interval|daily|weekly|cron` + per-kind fields),
  validated by the ported validator (§8) — so definitions get all five
  kinds without touching the reminder model. C-01's model half stays out
  of this program. The server's `_validate_schedule` checks only `kind`;
  per-kind field validation is ours to define: `daily`/`weekly` carry an
  IANA `timezone` defaulting to the machine's zone (slots computed in
  it, mirroring reminder cron behavior), `interval` bounds
  `every_seconds ≥ 60`, `weekly` requires a weekday.

## 5. Sync

`SyncEngine` grows two primitives beside `reminder_task`, under the same
ADR-018 discipline (server-wins, pending mutations, tombstones, conflicts
into the existing `sync_conflicts` surface).

### 5.1 `automation_definition`

- **Pull**: `GET /scheduled-tasks/definitions` (paged, `has_more`);
  upsert keyed on `(owner_id, server_id)`. Archived definitions mirror
  their lifecycle — never deleted locally. Archived mirrors are hidden
  from the queue and the Automations list by default.
- **Push** (pending mutations): `create` replays the **preview →
  definition two-step at push time from a stored payload — a pending
  mutation never stores a preview id** (previews are 24h-TTL,
  single-consume; an offline weekend would strand one). The server's
  payload-hash idempotency makes replay retry-safe. `update` likewise
  rides an update-mode preview. `pause`/`resume`/`archive` are direct
  endpoint calls.

### 5.2 `automation_result`

- **Pull**: `GET /scheduled-tasks/results`, incremental. Implementation
  verifies whether the endpoint supports an `updated_at`-style filter
  (the server has the filter helper; exposure on this route is
  unverified) — if not, a bounded page-walk with early-stop on known
  `(server_id, updated_at)` pairs. `updated_at`-based, not
  `created_at`-based: review-state changes made in the server UI must
  reach the local mirror.
- An incoming `automation_run_*` notification (client parsing landed in
  18940 slice 3) triggers an immediate results pull — fresh inbox without
  polling.
- **Push**: a local `review_state` change on a server-owned result queues
  a pending mutation → `POST /results/{id}/review`. Replay over a
  server-side change is last-write-wins under the standing server-wins +
  replay discipline; accepted.
- **Retention**: server retention may delete results we mirror; v1
  accepts local growth (results are bounded text). A local prune setting
  is a named follow-up, not v1.

### 5.3 Runs

Local runs persist in `automation_runs`. Server runs are **fetched on
demand** (the slice-4 Run history pattern) and never persisted in v1.
Results are the durable, offline-readable half (locked decision 4).

### 5.4 Reminder surface — stays on `/api/v1/tasks` (correction)

The control plane's reminder responses are the *normalized*
`ScheduledTask` row, which omits `cron`, `run_at`, and `body` — a
management projection, not a full-fidelity sync surface. Sync,
transfer, and the §6.1 list-and-match recovery therefore stay on the
full-fidelity `/api/v1/tasks` surface. Repointing to
`/scheduled-tasks/reminders` becomes a follow-up contingent on the
server enriching those responses (server-side work, §12).

## 6. Transfer state machine

`transfer_state ∈ {NULL, to_server_pending, to_server_sent,
to_server_failed, from_server_pending}` on both `reminder_tasks` and
`automation_definitions`. The machine runs inside `SyncEngine`'s push
phase as two new mutation actions (`transfer_to_server`,
`release_from_server`) — reusing its retry, backoff, and error surfaces.

### 6.1 Local → server

1. User action → `to_server_pending`; pending mutation stores the full
   create payload. **While merely queued (offline, no attempt yet) the
   task keeps executing locally** — this is the "create for the other to
   eventually pick up" path, and nothing goes dark.
2. When a push attempt starts: state → `to_server_sent` and the row
   **disarms first**, then the request goes out. A paused beat on failure
   is honest (18937 reports it); a double fire is not.
3. Request: definitions → preview → create (hash-idempotent). Reminders →
   full-fidelity `/tasks` create carrying `link_type="chatbook_transfer"`,
   `link_id=<local task id>` so an **ambiguous timeout** is resolved on
   the next pull by list-and-match on `link_id` — found ⇒ complete the
   transfer; absent ⇒ retry. A crash between send and ack recovers
   through this same leg (`to_server_sent` at startup ⇒ list-and-match);
   no second recovery path exists.
   Definitions transfer with `initial_lifecycle` matching the source
   (`configured` or `paused` — both accepted by the server's create).
4. Ack → the local row converts into the server-owned mirror **via
   upsert-or-merge**: if a background pull already created the mirror
   (the §4 `UNIQUE(owner_id, server_id)` race), keep the pulled mirror,
   delete the local row, transplant provenance. Otherwise convert in
   place (set `server_id`, `owner_id`, clear `transfer_state`).
5. Definitive failure (validation reject) → `to_server_failed`, re-arm
   locally, surface the server's field errors on the row with
   retry/cancel.

### 6.2 Server → local

1. User action on a server-owned mirror → local copy created immediately
   with `owner_id="local"`, `from_server_pending`, **dormant** (the queue
   filter excludes transfer-pending rows; extends the slice-1
   `is_server_scoped_owner` seam). Pending mutation queues the release.
2. Release: reminders → control-plane delete; definitions →
   `POST /definitions/{id}/archive`.
3. Ack → `transfer_state` cleared → the copy arms on the next queue
   reload (`request_reload` is already wired to mutations). No
   double-execution window on the server side: `agent_task_jobs`
   re-checks lifecycle at execution time (stated in its docstring), so an
   acked archive cannot fire afterward.
4. Offline / unacked: the server keeps executing; the dormant copy shows
   "waiting for server release".

### 6.3 Cancel, per state

| State | Cancel means |
|---|---|
| `to_server_pending` (unattempted) | Clear state, keep local. |
| `to_server_sent` / after ack | Too late — offer the reverse transfer. |
| `from_server_pending`, release unpushed | Delete the dormant copy; server unaffected. |
| `from_server_pending`, release acked | Reverse transfer. |

Dormant and in-flight rows are **read-only except cancel**.

### 6.4 Refusals (per-task, with the reason in copy)

- No server connection / no server identity.
- Target cannot execute the family: `agent_task` → local always refuses
  in v1; `recurring_question` → local refuses when local health (§7.4) is
  not `ready` (missing RAG extras, unreadable sources, unresolvable
  provider), quoting the health reason.
- A transfer already pending on the row.
- Lifecycle not transferable: only `configured` and `paused` definitions
  transfer; `archived` and `solved` refuse — there is nothing to
  execute.
- One-time task whose `run_at` is imminent (within ~5 minutes): **warn,
  not refuse** — the transfer can outlive the moment, and server behavior
  on a past `run_at` is unverified (flagged for live verification, §10).
- Local-only fields (`timeout_seconds`) do not transfer — stated in the
  confirm dialog, not silently dropped.

An in-flight local run at transfer time completes and writes its result
locally — disarming stops *new* dispatches only. Harmless and stated.

## 7. Local `recurring_question` execution

### 7.1 Ported pure (with a fixture-parity contract)

From the server, as-is: `normalize_recurring_question_scope`, the
`classify_rag_response` ladder (`finding/no_match/degraded` ×
`synthesized/evidence_only/none`), finding-policy →
`top_k/min_score/profile` mapping, the generation-only system prompt and
token caps, `_validate_schedule` / `_validate_recurring_question_config`
/ `_normalize_finding_policy` / `_normalize_retention_policy`.

**Drift rule**: the slice-2 server-response fixture file
(`Tests/Scheduling/fixtures/server_responses/`) is the parity contract —
when server validators change, fixtures are regenerated from the server
repo and the parity tests catch divergence.

### 7.2 Dispatch

Local-owner, `lifecycle=configured` definitions with a `next_run_at` feed
`PriorityQueue.load` as real rows (not projections). One
`automation_definition` handler registers in `SchedulerLoop.handlers`
with a **family-keyed executor registry** inside, so `agent_task` later
adds an executor, not a handler.

**The handler follows the `BriefingJobHandler` spawn shape** — this is a
correction from design review, not an option: `tick` awaits handlers
serially inline (locked decision, briefings phase 4), and a
recurring_question run is RAG + an LLM call. `handle` does synchronous
claim-checking only and spawns the run as an independent `asyncio.Task`
held by strong reference; the execution timeout (18939 semantics, task
`timeout_seconds` override honored) applies **inside** the spawned task
and records `timed_out`.

**Overlap guard** (briefing precedent, `skipped_claimed`): a
per-definition in-flight claim is checked synchronously in `handle`;
an overlapping slot records a `skipped` run instead of racing the
in-flight one. An interval shorter than the run's duration therefore
degrades to back-to-back runs, never concurrent ones.

Per dispatch: run row (`running` → terminal), slot key =
SHA-256(canonical `{definition_id, definition_version, schedule_slot}`),
`INSERT`-or-detect on the §4.1 UNIQUE → a deduped manual run returns the
server's `deduped` semantics ("already ran for this slot" toast).
`next_run_at` recomputed from now.

### 7.3 Execution adapters

- RAG: `RAG_Search/simplified` over the scope's sources; the adapter maps
  the server's source names (`media_db|notes|chats`) onto chatbook's
  corpora, and the health check (§7.4) uses the same mapping.
- Generation: `chat_api_call`, precedence identical to the server's
  `resolve_execution_target` — definition `input.provider/model/max_tokens`
  → `[scheduling]` executor defaults → chat default. This is what makes a
  pinned model survive handoff in both directions.
- Outcome → result row (`finding`/`failure`, `review_state="unread"`) and
  a notification through `NotificationDispatchService` using the same
  four `automation_run_*` kinds the server emits (client parsing already
  landed), **with the app handle** (phase-0 fix is a prerequisite in the
  PR chain).

### 7.4 Local health (read-time, minimal v1)

Three checks: RAG deps importable (`optional_deps`), scoped sources
readable, provider resolvable → `ready` / `capability_unavailable` /
`permission_required`. Never stored; computed where displayed, and
consulted by transfer refusals (§6.4).

## 8. Authoring (absorbs 18940's remaining slice)

Modal, upgraded in place, per ADR-099. One create/edit modal for
`recurring_question` definitions with a **"Runs on: This device |
Server (<id>)"** selector (default = current screen owner), and one
preview seam dispatched by owner:

- **Server-owned** → `POST /previews` → render
  `validation_errors`/`warnings`/`schedule_preview` →
  `POST /definitions {preview_id}` on save. Offline: the same stored
  payload becomes a pending `create` mutation (§5.1) — authored now,
  picked up by the server on next sync.
- **Local-owned** → `AutomationPreviewService` (the §7.1 ported
  validators) returns the identical `{field, code, message}` shape;
  save writes the local definition.

Same form renders both paths; field-addressed errors highlight the
offending input. v1 fields: name, question, scope (mode + sources
multi-select; collections/tags/saved-searches deferred), schedule kind +
per-kind fields (reusing 23102's preset widgets), generation mode,
finding-policy preset, notification toggle, optional provider/model pin.
Reminders keep their existing modal and gain the same "Runs on" selector.
**v1 authoring is `recurring_question`-only** — agent_task authoring
(even server-targeted) rides the follow-up program.

This satisfies the client half of TASK-18940 AC#2 by construction; the
task's progress log gets a pointer to this spec.

## 9. UI

- **Queue tab**: owner column (hidden in the compact-width mode); mixed
  listing (local reminders, local definitions, server mirrors,
  projections — definitions render through a row adapter beside
  `ReminderTask|ScheduledTask`). Per-task "Move to server / Move to
  local" in the detail pane + a key binding, disabled-with-reason per
  §6.4. Transfer states render as row badges ("Moving to server…",
  "Waiting for server release", "Transfer failed — retry/cancel").
- **Automations tab** (slices 2/4/5): definitions list shows both owners;
  Run history shows local runs from `automation_runs` and server runs
  fetched on demand (with the §4.1 `timed_out` display mapping).
- **Results tab** (new): unified inbox, unread-count badge on the tab
  label (existing conflicts-count pattern); detail renders answer /
  evidence / source_refs; read / dismiss (pushing for server rows);
  "Mark solved" on a finding wires `resolution_state` — server endpoint
  for server definitions, local field for local ones.
- **Phase-0 riders**: reminder toast fix (app handle via zero-arg getter,
  the `BriefingJobHandler` pattern) and a visible New button in the
  queue pane header.

## 10. Testing and verification

- **Unit**: validator-port parity against regenerated server fixtures;
  slot-dedupe UNIQUE race; transfer machine — every failure leg (offline
  queue, validation reject, ambiguous timeout + list-and-match recovery,
  crash-at-`to_server_sent`, cancel per state) plus a property test
  asserting **never both armed** across transition interleavings; sync
  upsert/merge including the §6.1 pull-vs-convert race; review-state
  push replay; overlap claim (`skipped` recorded); stale-`running`
  reconciliation; runs prune-on-write.
- **Integration**: real in-memory SQLite; end-to-end local run
  (definition → tick → spawned executor → run row → result row → real
  `NotificationDispatchService`, unmocked — accessor-mock suites have
  hidden inert features here before); the local-owner reminder path
  pinned unchanged (extends `test_owner_filter`).
- **Live** (lessons-live-verification): against a real server with both
  env gates enabled — one full round-trip each direction (create local →
  transfer → server executes → result syncs down `unread`; create on
  server → transfer to local → local executes → result in the same
  inbox). Also resolves the two flagged unknowns: server behavior on
  past `run_at`, and whether `/results` exposes an `updated_at` filter.
  What was and was not verified live is recorded honestly in the task.

## 11. Phasing

| PR | Contents | Depends on |
|---|---|---|
| 0 | Reminder toast fix + visible New button | — |
| 1 | Schema v4 + models + slot-key module | — |
| 2 | Local recurring_question execution (§7) | 1 |
| 3 | Sync primitives (§5) | 1 |
| 4 | Authoring modal + owner-dispatched preview (§8) | 1 (parallel to 2/3) |
| 5 | Transfer machine (§6), both primitives, both directions | 3 |
| 6 | Results inbox + queue owner column + transfer badges; live E2E | 2, 3, 5 |

PR-5 does not depend on PR-4; order swaps if authoring drags.
Implementation runs in a worktree off `origin/dev` (concurrent sessions
mutate this checkout), and coordinates with the session landing 18940
slices before PR-4 starts.

## 12. Out of scope (named follow-ups)

- `agent_task` local execution and authoring (immediate follow-up
  program; seams left ready: family-keyed executors, policy fields
  carried, transfer refusal already family-aware).
- Run-history sync-down (server_id column reserved).
- Local-run result push-up (needs a server ingest endpoint — server-side
  work).
- Reminder `ScheduleKind` model expansion (C-01 model half).
- Local results prune / retention mirroring.
- Reminder repoint to `/scheduled-tasks/reminders` — contingent on the
  server enriching the normalized responses with full reminder fidelity
  (`cron`, `run_at`, `body`); server-side work (§5.4).
- First-class server transfer endpoint (Approach 2; slots behind the same
  service seam if wanted later).

## 13. ADR

One new ADR: per-task ownership transfer (amending ADR-077 §1's
screen-era framing), local recurring_question execution, and results
sync-down. Number assigned against `origin/dev` at merge time (dev is at
099; collisions are a known failure mode — lessons-backlog-hygiene).
