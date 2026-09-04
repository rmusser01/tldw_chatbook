# Spec: Schedules screen redesign — unified list + inspector-style detail pane

Status: **Delivered** 2026-09-04 — PRs #2343 (PR-1, detail-pane
regrammar), #2354 (PR-2, unified list + filter chips + rail), #2371 (PR-3,
in-pane editing + owner-row transfer) and #2389 (PR-4, single-surface
workbench), all merged to dev; dev tip `b7f8efde73` is the final merge.
Delivery note: shipped as specified — the tab IA is retired and the
workbench is one surface (list + inspector detail + rail), with results,
conflicts and run history reached as pushed views rather than tabs, and an
80x24 responsive floor that pushes the same detail pane full-screen. The
body below is the approved design and is left intact as the record of what
was decided; ADR-116 records the inspector-editing decision and ADR-112
the per-task ownership transfer.
Reference visuals: two ChatGPT scheduled-tasks screenshots (user-provided
2026-09-02) — filter-chip rail + grouped key-value detail pane with
in-place dropdown editing.

## 1. Purpose

Rebuild the Schedules workbench's layout to the reference design: one
unified task list with filter chips on the left, and an inspector-style
detail pane on the right whose grouped key-value rows are directly
editable. The redesign absorbs the handoff program's surfaces (owner,
transfer, results) instead of bolting them onto the old tab IA.

## 2. Locked decisions (user-approved)

1. **Editing = hybrid.** In-pane quick edits for single-value rows; the
   modal survives for create, for multi-field surgery ("Edit in full…"),
   and as the narrow-width fallback. Ships as an ADR-099 amendment (§10).
2. **List = unified schedulables.** Reminders + `recurring_question`
   definitions (+ `agent_task` later), both owners mixed. Briefing and
   watchlist projections stay out. Conflicts, run history, and the
   results inbox become secondary surfaces (§8).
3. **Chips = clone + map.** `All · Active · Paused · Completed` exactly
   as the reference; transfer states are row badges, never chips.
4. **Owner row = dropdown + confirm.** "Runs on" renders like every
   other row; picking the other owner runs the transfer refusal check,
   then the PR-5 confirm dialog, then the in-flight badge.
5. **Create ▾ = by task type.** "Reminder" / "Recurring question";
   `agent_task` joins later disabled-with-reason.

## 3. Chrome and information architecture

The Queue / Automations / Conflicts tab bar is retired. The screen is a
two-pane workbench:

- **Rail (left, ~40%)**: chip row (`All Active Paused Completed`) +
  `Create ▾`; search input; `✓ Mark all as read` (rendered only while
  unread results exist); the unified list; a slim bottom status strip
  carrying the sync-status widget (compacted), the owner-scope
  indicator, and a conflicts badge chip ("2 conflicts") that opens the
  existing conflicts view as an overlay.
- **Detail pane (right)**: §5.

Chip → row-set mapping:

| Chip | Rows |
|---|---|
| All | Active + Paused (Completed rows excluded -- archived stays out of the default view) |
| Active | armed rows: enabled reminders + `configured` definitions, **including** `to_server_pending`/`to_server_failed` (they still execute locally) |
| Paused | disabled reminders + `paused` definitions |
| Completed | fired one-time reminders + `archived` definitions |

Finding resolution (`solved`) is a **results** property and never a chip
— it lives in the History group and the results surface. Archived
definitions become visible under Completed (a deliberate change from
"archived mirrors hidden by default" — reachable now, but only via the
Completed chip, so the default view stays uncluttered).

**Verification item (spec-time unknown):** whether fired one-time
reminders persist in a queryable state. If they are deleted/disabled
without a distinguishable marker, the Completed chip covers definitions
only and the copy says so; do not fabricate a fired-state column without
a program decision.

## 4. The unified list

One row type serves both primitives:

- Status glyph: `○` recurring · `▶` active one-shot/monitor · `⏸`
  paused · `✓` completed.
- Title; subtitle = schedule summary + `· Next run in 2h` (relative,
  from `next_run_at`, both primitives carry it).
- Owner suffix only when non-default: dim `⇅ server` on server-owned
  rows.
- Transfer badges exactly as PR-5 defines them ("Moving to server…",
  "Waiting for server release", "Transfer failed — retry/cancel").
- Unread-results dot, right-aligned (blue, the reference's affordance).
- Sort: next-run ascending within Active; most-recent-first elsewhere.
- Relative-time ticker: ONE 60-second timer refreshing visible rows
  only, suspended while the screen is inactive (no per-second ticks —
  performance-audit rule).
- Search: in-memory filter over title + question/body of loaded rows
  (bounded, human-authored counts; no pagination v1 — same ruling as
  PR #2302's declined finding).

**Data-layer implication (named task):** the list spans owners. The
Automations side already merges local + server rows (PR-4); reminders'
`list_tasks` is active-owner-scoped and needs a spans-owners listing
seam. This also dissolves the "created for another owner, invisible
after save" wart from PR #2302's Qodo round.

## 5. Detail pane

Header: status word in accent color ("Active"), title, top-right icon
actions — pause/resume toggle, kebab, close. Kebab: Run now · Duplicate
· View runs · View results · Edit in full… · Delete/Archive (§9 copy
rule).

Body: the prompt/question (reminders: body text) in a rounded card, then
grouped key-value rows. Every row is the same widget —
**`DetailValueRow`**: label left, value right, `▾` affordance;
Enter/click opens an inline Select or the appropriate mini-editor;
field-addressed errors render inline under the row. One new reusable
widget + a group container is most of the reference look.

Per-family row table (implementers do not improvise rows):

| Group | Recurring question | Reminder |
|---|---|---|
| Details | Runs on · Model (pinned/"Provider default") · Generation (required/optional) · Finding policy · Sources (Media/Notes/Chats) | Runs on |
| Frequency | Repeat · At · Timezone · Notifications | Repeat · At · Timezone · Notifications |
| History (collapsed) | last run outcome · run count · link to run history · unread results inline w/ read/dismiss | last fire · link to history |

Server-owned rows render the same table; rows whose value the server
owns exclusively and we cannot push (none today — lifecycle, model,
schedule, notifications all have push paths post-PR-5) would render
read-only with a reason.

> **Errata (PR-3, T6):** the "notifications … push paths post-PR-5" claim
> above does not hold for reminders — a reminder's Notifications row has no
> backing schema field at all (verified during PR-3 task 3; a dispatch
> always writes the same fixed inbox+toast), so it stays read-only until a
> field exists to push. `recurring_question` definitions do carry a real
> notifications toggle and edit in place per §6.

## 6. Editing model (ADR-099 amendment)

- In-pane quick edits cover single-value rows only. Each row commit goes
  through the existing seams immediately — `save_definition(definition_id=…)`
  (merge-on-edit choke point preserves unexposed fields) /
  `update_reminder` — with the row flashing the saved value or showing
  the field-addressed error inline. No debounce machinery: successive
  offline edits already coalesce per-record in the mutation table.
- Multi-field surgery (question text, scope rework, custom cron) and all
  creation stay in the modals ("Edit in full…" from the kebab).
- Narrow widths: Enter on a list row opens the modal as today. This IS
  the two-code-path cost ADR-099 warned about, accepted deliberately:
  the pane rows and the modal share the same facade seams and
  validators, so the second path is thin. The amendment (§10) names
  this.
- **Server-owned pause/lifecycle needs a pull guard (named task):**
  `upsert_automation_definitions_from_server` is server-wins on
  everything except `transfer_state`, so an optimistic local pause with
  a queued lifecycle mutation would flicker back on a pull that lands
  before the replay. Extend the upsert with the PR-3 results pattern:
  skip lifecycle fields when a pending lifecycle mutation exists for the
  row. Small DB change, no migration.

## 7. Owner row = the transfer machine

`Runs on` is a dropdown. Choosing the other owner:

1. `transfer_refusal` check — a refused target renders the refusal
   reason as an inline error line under the row (Textual Selects cannot
   disable individual options; do not try).
2. Allowed → the PR-5 confirm dialog (warnings: imminent `run_at`,
   non-transferring fields).
3. Confirmed → `begin_transfer_*`; the row shows the in-flight badge
   until conversion lands; Cancel lives on the badge per the §6.3 cancel
   table. Badge state refreshes off the existing sync/mutation
   reload-notify — no new event plumbing.

## 8. Secondary surfaces

- **Conflicts**: badge chip on the rail status strip → existing
  conflicts view as overlay.
- **Run history**: kebab + History-group link → existing run-history
  surface.
- **Results**: unread dots + Mark-all-read in the rail; per-result
  read/dismiss inline in the History group; the PR-6 Results *tab* is
  trimmed to its minimal honest version (§11) and later retired.
- **Mark all as read** fans out one pending review mutation per
  server-owned unread result (per-id replay; no bulk endpoint; bounded
  by the newest-200 window). Stated so nobody "optimizes" it into a
  local-only lie.

## 9. Copy and honesty rules

- Server-owned definitions get **Archive**, never Delete (the server
  exposes no definition delete). Delete appears only where a true
  delete path exists (local rows, reminder tombstones).
- A `to_server_pending` row's detail says it still runs on this device
  until the server accepts it (§6.1.1 copy, carried over from PR-5).
- Every disabled control carries its reason (UX-073 idiom).

## 10. ADRs

- **New ADR (number ≥113, swept against origin/dev at merge time —
  110/111/112 claimed as of 2026-09-02):** "Schedules unified workbench" — the
  one-list IA, chip vocabulary, inspector-pane grammar, and the
  retirement of the tab bar.
- **ADR-099 amendment** (same PR): hybrid editing — in-pane single-value
  rows + modal for create/full-edit/narrow fallback; names the accepted
  two-path cost and the shared-seam rationale.
- **Errata:** dev carries ADR filename collisions on `099`
  (`099-schedule-editor-shape.md` vs
  `099-persistent-terminal-session-runtime-boundary.md`), and the same on
  `098`, `102` and `104`; this program cites by FILENAME, and renumbering
  is deliberately left to the in-flight `docs/lesson-adr-number-collisions`
  branch. The "≥113" above was written as "≥112" until the final review
  caught that `112-per-task-schedule-ownership-transfer.md` already
  existed at this branch's own base — re-sweep the range at merge time
  rather than trusting the number in this file.

## 11. Sequencing

After the handoff program:

- PR-5 (transfer machine) — in flight, unchanged.
- PR-6 — results sync/notification halves unchanged; the Results tab
  ships minimal-honest (table + read/dismiss + unread count). PR-6's
  acceptance criteria are preserved under the trim: results visible and
  reviewable, unread count surfaced, notification-triggered pull, live
  E2E.
- Then the redesign program, ~4 PRs:
  1. `DetailValueRow` + detail-pane regrammar (read-only rows first,
     per-family table, History group).
  2. Unified list + chips + rail chrome + cross-owner reminder listing
     seam + relative-time ticker.
  3. In-pane editing + owner-row transfer + lifecycle pull-guard +
     ADR-099 amendment + new ADR.
  4. Responsive floor (pushed detail Screen — same widget class,
     fresh instance, NOT reparenting), keyboard map, retire old tabs +
     the trimmed Results tab, polish + live verification.

## 12. Keyboard map (v1)

`1-4` or `f` cycle chips · `/` search · `n` create · Enter open/edit ·
`p` pause/resume · `m` move owner · `r` mark read · Esc back/close ·
Up/Down traverse detail rows when the pane has focus. Footer-visible key
hints extend ADR-099's parity requirement to the pane.

**PR-4 amendment (2026-09-04).** `Enter` is honored only below the
responsive floor's 84-column threshold, where it PUSHES the row's detail
full-screen (plan ruling 6); at or above it the detail pane is already
docked beside the list showing that row, so `Enter` stays the no-op it
has always been. `m` follows the same split: below the threshold it
pushes the detail and opens the Runs-on dropdown inside it, because the
docked pane is hidden there (final review F1).

## 13. Out of scope

- `agent_task` rows beyond the disabled Create entry.
- Briefing/watchlist projections in the list.
- Bulk operations beyond Mark-all-read.
- Any schema migration (none required; the lifecycle pull-guard is
  code-only).

## 14. Verification

- Widget tests per component (`DetailValueRow` grammar incl. error
  rendering; chip mapping; badge states).
- The 80×24 floor: every operation reachable via pushed view or modal —
  pinned by tests at both widths.
- Live verification per lessons-live-verification at program end: drive
  the real TUI (tmux recipe), including one real transfer via the owner
  row against a real server.
- UI PRs update `Docs/User_Guide/` schedules page per CLAUDE.md.
