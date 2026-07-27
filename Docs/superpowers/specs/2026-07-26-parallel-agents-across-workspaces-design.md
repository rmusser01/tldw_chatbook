# Parallel agents across workspaces — per-session runs from the sidebar

Date: 2026-07-26
Status: approved design, pending implementation
Companion to: `Docs/superpowers/specs/2026-07-26-settings-workspaces-category-design.md`
(merged as PRs #943/#944) — that train's **run-bound folder roots** are this
program's enabling primitive and need no changes here (ADR-028).
Builds on: workspace UX program (PRs #928/#932/#943/#944, ADR-027/028).

## 1. Goal and decisions (locked with the user)

Users run **multiple agents in parallel, each in its own workspace**, and
swap between them from the Console sidebar. Today runs are globally
serialized: one `ConsoleRunState` on the controller, one exclusive
`"console-run"` worker group, and an is-send-allowed gate that refuses a
second send anywhere (confirmed live during the folder-roots smoke).

- **Concurrency rule: one in-flight run per session, capped globally.**
  Any number of sessions (same or different workspaces) run in parallel
  under the cap. "Each tab is an agent."
- **Background approvals: badge + park.** A background session's run pauses
  at its approval card; the tab and sidebar row badge; one toast fires;
  the user jumps to the tab to decide. Never auto-approve.
- **Background completion: toast + markers.** One toast naming session,
  workspace, and outcome; sidebar row and tab carry a done/failed marker
  until visited.
- **The cap is user-adjustable in Settings** (§4): default 3, raiseable to
  whatever the user wants, with guidance copy owning the consequences.

## 2. Per-session run model (PR1)

- `ConsoleRunState` moves from a single controller attribute to a
  **per-session map** owned alongside the session store's other per-session
  state (settings, drafts): `run_state_for(session_id)`,
  `set_run_state(session_id, ...)`, terminal-state clears scoped to their
  session. `run_state_history` becomes per-session likewise.
- Send gate: allowed iff (a) the TARGET session has no in-flight run and
  (b) `count(in-flight runs) < max_parallel_runs`. (a) failing keeps
  today's per-session copy ("a run is already running in this tab"); (b)
  failing produces the cap toast (§4).
- Workers: `group=f"console-run-{session_id}"`, still `exclusive=True`
  within the group — one run per session by construction, N groups in
  parallel. The TASK-228 lesson (UI-sync kicks must never share a group
  with runs) carries over: sync workers keep their own groups.
- Run completion/error/interrupt paths (`_set_run_state`,
  `_clear_terminal_run_state`, retry/regenerate entries) all become
  session-scoped; the send-stash and follow-intent state that assume "the"
  run are audited and keyed the same way.
- Workspace binding: unchanged. Each run already resolves folder roots
  from its session's workspace (`session_workspace_id` →
  `BuiltinToolProvider` → `run_workspace`); parallel runs in different
  workspaces enforce different roots with no cross-talk — this is the
  merged #943 behavior and a required regression pin here (§7).

## 3. Background streaming discipline (PR1)

Message and tool writes already land in per-session store trees; the risk
is view-seam bleed. Rule: **store-first writes; view application gated on
"is the writing session the viewed session."** PR1 includes an audit of the
streaming/tool-marker/generation-card paths for any write that targets the
visible view directly without a session check; each such site is gated.
Switching tabs rebuilds the view from the store (existing switch path),
so a background session's accumulated output appears on visit. The
`CONSOLE_RUN_ALREADY_RUNNING_COPY` gate copy is retired in favor of the
per-session/cap messages.

## 4. The cap — config + Settings exposure

- Config: `[console] max_parallel_runs`, integer `>= 1`, **default 3**.
  Values above the default are allowed without an upper bound — the user
  owns the trade-off.
- **Settings exposure (user-requested): the cap is editable in
  Settings ▸ Console Behavior** (the guided category that already owns
  `[console]` values — draft + Save/Revert semantics, not immediate).
  The row ships with everything that comes with a guided setting:
  - Input with validation (integer, minimum 1; non-numeric and `< 1`
    rejected inline with the category's standard validation copy).
  - Focused-field guidance in the Scope Inspector: purpose ("How many
    agent runs may be in flight at once, across all tabs"), saved-as
    (`console.max_parallel_runs`), and an explicit consequences note:
    "Each concurrent run holds a provider generation, its own tool
    activity, and memory for its transcript. Local providers (llama.cpp)
    typically serialize or slow under concurrent generations; high values
    can exhaust provider slots, rate limits, or RAM. Raise it as far as
    you like — the app enforces no ceiling."
  - Applies to NEW sends on save (in-flight runs are never killed by
    lowering the cap; the count simply drains below the new value).
- Cap-exceeded send: honest toast naming the busy sessions —
  `"N agents already running (<session titles, comma-separated, truncated
  to 3 + 'and K more'>). Wait for one to finish or interrupt it."` No
  hidden queue: queued sends that fire later without the user watching
  are worse than a clear refusal.

## 5. Background approvals — badge + park (PR2)

- The per-run review hook already awaits its approval decision; a parked
  run is just that await with nobody looking. New behavior when the owning
  session is not viewed: the approval card does NOT mount over the current
  tab; instead
  - the owning TAB and the sidebar conversation row get a
    `needs-approval` badge,
  - ONE toast fires per card: `"Agent in <session title> (<workspace
    name>) needs approval."`,
  - visiting the tab mounts the card exactly as today.
- Approvals never auto-resolve; other sessions' runs are unaffected while
  one is parked. Card state survives tab switches (it derives from the
  run's pending-approval state, not from mounted-widget lifetime).

## 6. Sidebar/tab fleet indicators (PR2)

- Three-state marker on Console tabs AND sidebar conversation-browser
  rows: `running` / `needs-approval` / `finished-unvisited` (distinct
  glyphs; cleared when the session is viewed). `finished-unvisited`
  renders success and failure with distinct glyphs (e.g. ✓ vs ✗) but
  shares the same clear-on-visit lifecycle. Display-state additions
  flow through the existing browser-row builders (workspace groups
  included, so "which workspace's agents are busy" is visible at a
  glance).
- Completion/failure toast: `"Agent in <session title> (<workspace name>)
  <finished|failed>."` — once per run.
- The Agent rail section shows the VIEWED session's run state (as today)
  plus a one-line fleet summary when other runs exist: `"2 other agents
  running, 1 waiting for approval."`

## 7. Testing

- **PR1:** per-session state unit tests (two sessions run concurrently;
  same-session second send refused with per-session copy; cap refusal at
  N; lowering cap never kills in-flight runs); worker-group isolation
  (interrupting session A's run leaves B's running); background-write
  audit regression (a background run's stream never mutates the viewed
  transcript — store-only until visit); **workspace-isolation pin**: two
  concurrent runs in different workspaces resolve different folder roots
  (extends the #943 provider tests to overlap in time).
- **PR2:** badge/marker state tests (running/needs-approval/
  finished-unvisited transitions incl. clear-on-visit); parked-approval
  flow (card absent while unviewed, badge present, mounts on visit,
  decision releases the run); toast-once contracts; Settings cap row
  (validation, guidance copy, save→config write, cap honored on next
  send).
- **Live smoke (non-negotiable before merge):** two workspaces with
  distinct bound folders; start agent runs in both tabs; verify both run
  concurrently (Agent rail fleet line), each run's file tool confined to
  its own workspace's folder; park an approval on the background session,
  see badge + toast, visit, approve, both complete with correct markers.
  Reuse the folder-roots smoke recipe (scratch profile, llama.cpp :9099,
  `wssmoke` socket; Approve-all THEN Submit).

## 8. Phasing

Two stacked PRs, merged as one train (no released state where the engine
allows parallel runs but the UI can't show them, or vice versa):
- **PR1 `feat/console-per-session-runs`**: §2 + §3 + §4's config/gate
  (Settings row included — it is a plain guided-category addition) +
  PR1 tests.
- **PR2 `feat/console-agent-fleet-ux`** (stacked): §5 + §6 + PR2 tests +
  live smoke.

## 9. Out of scope

Cross-run coordination or shared context between agents; run queues;
per-workspace caps; changes to approval semantics; ACP/external runtimes
(the `runtimeWorkspaceRoots`-shaped export surface from ADR-028 stands
ready); multi-window UI.
