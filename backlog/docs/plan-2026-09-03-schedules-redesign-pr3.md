# Schedules Redesign — PR-3: In-pane editing + owner-row transfer + lifecycle pull-guard + ADRs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The dormant PR-1 surfaces come alive: single-value rows on both detail panes edit in place through the existing facade seams with field-addressed errors inline; "Runs on" becomes the transfer dropdown (refusal → inline error; confirm → the PR-5 machine); lifecycle pause/resume gets its first UI caller with the pull-guard that keeps optimistic writes from flickering back; the program's ADR-116 (renumbered from 115 at commit-time sweep) and the ADR-099-schedule-editor-shape amendment land.

**Architecture:** `DetailValueRow` gains an activation message and a value↔editor swap API (`begin_edit(widget)`/`end_edit()` — the MediaDetailsWidget edit-mode toggle scaled to one row); rows stay dumb, panes provide the editor (plain `Select`/`Input` reuse — no repo inline idiom exists, modal pickers are too heavy for closed enums) and commit through the seams: definitions via `save_definition(definition_id=...)` (merge-on-edit; **schedule dicts resend whole — one-level merge only**), reminders via a NEW service-side validation bridge (`update_reminder` today surfaces raw Pydantic errors — the bridge returns the same `{field,code,message}` shape definitions get). Lifecycle rows route through `set_definition_lifecycle` (its first UI caller); `upsert_automation_definitions_from_server` gains the two-layer pending-mutation guard ported from the results upsert, scoped to lifecycle-action mutations. The owner row's dropdown feeds from `_runs_on_options()` and drives `transfer_refusal` → `ConfirmationDialog` + `transfer_warnings` → `begin_transfer_*`, with refusals rendering through `show_error` (their first row-level surface). `DefinitionDetail` gains the transfer-lock UI wiring `TaskDetail` already has.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite, pytest.

**Spec:** `backlog/docs/spec-2026-09-02-schedules-screen-redesign.md` §6/§7 (tracked on dev). Planning rulings (binding):
1. **Editing scope**: exactly the spec-§5 single-value rows — reminder Frequency (Repeat preset/At/Timezone/Notifications); definition Details (Model, Generation, Finding policy, Sources) + Frequency (Repeat/At/Timezone, Notifications) + header pause/resume via lifecycle. Question text, custom cron, scope rework stay in the modals ("Edit in full…" unchanged). NO debounce machinery (per-record mutation coalescing already exists).
2. **Commit semantics**: commit-on-close (Select change or Input submit → immediate seam call); success repaints the row from the authoritative re-read; failure renders the field-addressed error via `show_error` and restores the display value. Locked rows (transfer in-flight/dormant): the affordance disables and activation shows `transfer_lock_reason` via `show_error` — the same guard surface, never a silent no-op.
3. **Owner row = the spec-§7 flow verbatim**: dropdown always renders; a refused target's selection → `transfer_refusal` reason inline under the row (Textual Selects cannot disable per-option — documented); allowed → `ConfirmationDialog` listing `transfer_warnings` → `begin_transfer_to_server/to_local`; in-flight → the existing badge + Cancel affordance (route release-leg cancels to the DORMANT COPY's row id — the PR-5 lesson).
4. **Lifecycle pull-guard**: port the results-upsert two-layer guard (in-transaction pending-mutation check + same-cycle skip set) onto the definitions upsert, scoped to mutations whose payload `action ∈ {pause,resume,archive}` — lifecycle fields only; every other field stays server-wins. No migration.
5. **ADRs by filename**: new `backlog/decisions/116-schedules-inspector-editing.md` (the unified-workbench IA + hybrid editing decision, spec §10); amend `099-schedule-editor-shape.md` in place (Status → Amended-by-115 note + the hybrid carve-out) — cite by FILENAME everywhere (the 099 number collision is renumbered elsewhere).
6. **Reminder validation bridge**: a `ReminderEditOutcome`-shaped service wrapper (status + `{field,code,message}` errors) — NOT a raw exception surface; reuses the schedule validators the create form uses.

## Global Constraints

- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-redesign-pr3`, branch `feat/schedules-inline-editing` off current `origin/dev`. Never the main checkout; NEVER `git stash`; no pkill beyond own PIDs; `git --no-pager`; FOREGROUND pytest only; tmp_path DBs.
- NO schema migration. Editing routes through EXISTING seams only — any new write path is a plan violation; escalate instead.
- Survey with exact seams: `redesign-pr3-survey.md` in the SDD workspace.
- Diagnostics pin is a SCRIPT (`--write` + commit JSON) on any logger change. Census merge bar = COUNT parity vs dev CI. CSS via build flow, `$ds-*` tokens, source+bundle together. Geometry/paint tests need `CSS_PATH = BUNDLED_STYLESHEET`. Painted assertions only. `DetailGroup(title=...)` keyword-only. Escape/Text discipline on all values.
- Preservation: read-only behavior of non-edited rows, all existing actions/badges/tests — zero unrelated assertion changes.
- UI change ⇒ `Docs/User_Guide/` schedules page.
- Commit trailer on every commit:

```
Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01WocisXw6SEEG6nb1aKFHtv
```

---

### Task 1: DetailValueRow activation + edit-swap API

**Files:** Modify `tldw_chatbook/Widgets/detail_value_row.py` (+CSS via the features file). Test: `Tests/UI/test_detail_value_row.py`.

**Interfaces (produced; Tasks 3-5 consume):**
- Message `DetailValueRow.Activated(row)` posted on click/Enter WHEN `affordance` is truthy AND no editor is open (dormant rows stay inert — preservation).
- `begin_edit(editor: Widget) -> None`: hides the value, mounts the editor in its place, focuses it; `end_edit(*, restore_focus: bool = True) -> None`: unmounts, re-shows the value. One editor at a time; `begin_edit` while editing is a guarded no-op. The error line (`show_error`) coexists with an open editor.
- `affordance` gains a visual live state (the `▾` un-dims) — read the PR-1 dimmed styling and add the active variant.

- [ ] TDD: activation fires only with affordance + not-editing (painted + message assertions); swap round-trip (value hidden, editor mounted/focused, end_edit restores); guarded double-begin; error line + editor coexistence; dormant rows inert. FAIL → implement → PASS → commit `feat(ui): DetailValueRow activation + edit swap`.

### Task 2: Reminder validation bridge + lifecycle pull-guard (service/DB)

**Files:** Modify `tldw_chatbook/Scheduling/services/scheduling_service.py` (+`ReminderEditOutcome` dataclass + `async edit_reminder_fields(task_id, payload) -> ReminderEditOutcome` wrapping `update_reminder` — validates via the SAME schedule validators the create form uses (survey: the forgiving-datetime/preset helpers), catches the Pydantic surface, returns `{field,code,message}` errors; respects `transfer_lock_reason` — a locked row returns the reason as a field error on `_row`), `tldw_chatbook/Scheduling/db/scheduled_tasks_db.py` (the lifecycle pull-guard per ruling 4 — port the results-upsert two-layer mechanism: in-transaction check for a pending `automation_definition` mutation with `action ∈ {pause,resume,archive}` → skip `lifecycle` field on that row; plus the same-cycle skip-set parameter threaded from the sync engine's push phase like `skip_review_server_ids`), `tldw_chatbook/Scheduling/services/sync_engine.py` (thread the pushed-lifecycle skip set — mirror the review-pushback precedent exactly). Tests: `test_scheduling_service.py`, `test_scheduled_tasks_db.py`, `test_sync_engine.py`.

- [ ] TDD: bridge returns field errors for junk schedule/timezone (no raw exception escapes); locked-row refusal; valid edit persists + records the mutation under the ROW owner (the PR-2 threading); pull-guard both layers (pending lifecycle mutation → lifecycle field skipped, other fields still server-wins; same-cycle pushed set skips; no mutation → server-wins unchanged); fault-ordering pinned like the results precedent. FAIL → implement → PASS → commit `feat(scheduling): reminder edit bridge + lifecycle pull-guard`.

### Task 3: Reminder pane in-pane editing

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/task_detail.py` (Frequency rows editable: Repeat preset Select, At Input (forgiving datetime/time — reuse the create-form parsing), Timezone Select (the create form's option source incl. the stored-zone preservation lesson), Notifications Select; commits via Task 2's bridge; success repaints from the re-read, failure → `show_error` + restore; locked rows per ruling 2), workbench wiring for the re-read/refresh after a successful edit (the unified list's row must update — the existing refresh seam). Tests: task_detail/workbench files + unified-list file.

- [ ] TDD: each row's editor opens with the CURRENT value preselected; commit persists (authoritative repaint pinned); a server-owned row's edit records the mutation (row-owner threading pinned); junk At value → inline error + display restored; locked row → reason shown, no editor; non-edited behaviors preserved (zero unrelated assertion changes). FAIL → implement → PASS → commit `feat(scheduling): reminder frequency rows edit in place`.

### Task 4: Definition pane in-pane editing + lifecycle UI

**Files:** Modify `tldw_chatbook/UI/Screens/scheduling/definition_detail.py` (Details rows: Model Input (blank = provider default — the PR-1 "Not set" honesty preserved), Generation Select, Finding-policy Select, Sources multi-toggle (the smallest honest editor — a Select of the 7 combinations is ugly; prefer a 3-checkbox mini-editor mounted via begin_edit, or sequential toggles — pick and document), Frequency rows (schedule edits RESEND THE WHOLE schedule dict — the one-level-merge constraint; build it from the row's current schedule + the edited field), Notifications Select; commits via `save_definition(definition_id=...)`, errors field-addressed; locked rows per ruling 2 + `DefinitionDetail` gains the transfer-lock wiring TaskDetail has (survey point 10)), header pause/resume affordance → `set_definition_lifecycle` (FIRST UI caller: optimistic repaint + the Task 2 pull-guard keeps it from flickering; server rows record the mutation, local rows write direct — read the producer). Tests: definition_detail/automations-tab files.

- [ ] TDD: each row edits + persists (server echo mirrored for online server rows; offline queues); schedule-field edit resends the whole dict (pinned — an edited At must not drop the kind/timezone); lifecycle toggle round-trip with a pull racing it (guard pinned at the UI-visible level: no flicker-back); sources editor honest; locked + "Not set" states preserved. FAIL → implement → PASS → commit `feat(scheduling): definition rows edit in place + lifecycle toggle`.

### Task 5: Owner-row transfer dropdown

**Files:** Modify both panes (the "Runs on" row: affordance on; activation mounts a Select fed by `SchedulesWorkbench._runs_on_options()` — hoist/share if it's workbench-private; selection == current owner → close; other owner → `transfer_refusal` first (reason via `show_error`, editor closes, value restored), allowed → `ConfirmationDialog` with `transfer_warnings` list → confirmed → `begin_transfer_to_server`/`to_local` by direction; outcome drives the existing badge rendering; in-flight rows: affordance disabled + Cancel affordance (release-leg cancel targets the dormant copy's id — assert it); failed: Retry via re-begin), workbench routing. Tests: both pane files + transfer-actions file.

- [ ] TDD: dropdown opens with current owner; same-owner close is a no-op; refused target → inline reason (health-quoting pinned); allowed → dialog lists warnings → confirm fires the right facade call with the right row id per direction (dormant-copy case pinned); cancel/retry affordances; existing transfer buttons/badges unchanged (they remain until PR-4 — coexistence pinned). FAIL → implement → PASS → commit `feat(scheduling): owner-row transfer dropdown`.

### Task 6: ADRs + docs + gates

**Files:** Create `backlog/decisions/116-schedules-inspector-editing.md` (Status Proposed; the unified-workbench IA recap + the hybrid editing decision + the owner-row-as-transfer-surface + the lifecycle pull-guard rationale; cites the spec + `099-schedule-editor-shape.md` BY FILENAME). Modify `backlog/decisions/099-schedule-editor-shape.md` (append an Amendment section: the modal remains for create/full-edit/narrow fallback; single-value rows edit in the pane per ADR-116 (renumbered from 115 at commit-time sweep) — do NOT renumber, do NOT touch the colliding 099 file). Update `Docs/User_Guide/` schedules page (editing, owner dropdown, pause/resume). Gates.

- [ ] Full `Tests/Scheduling/ -q`; the assembled schedules UI set; census (count parity vs dev CI); pin script; ruff; bundle byte-identical. Verify ADR-116 (renumbered from 115 at commit-time sweep) is still unclaimed on origin/dev AND across remote branches at commit time (the collision lesson — sweep). Commit `docs(scheduling): ADR-116 (renumbered from 115 at commit-time sweep) + 099 amendment + editing docs`.

---

## After the tasks
Final whole-branch review (opus; editing-correctness + guard-coherence + preservation lenses) → one fix wave → PR `feat(scheduling): in-pane editing + owner-row transfer (redesign PR-3)` → paged bot read → adjudicate → the full cycle (rebase→artifacts→gate→push→watch→merge-in-loop) as ONE background pipeline; coordinate a merge window with the peer session if the treadmill exceeds two laps.
