# Home "What you were working on last" + "What's next" — Design

Date: 2026-08-29
Status: Draft — awaiting review
Branch context: authored on `docs/lesson-adr-number-collisions` (unrelated in-flight branch; spec file only, no commit made).

## Context

The Home screen already has a single-item resume affordance (task-190): an idle-canvas
"Resume note/conversation: \<title\>" button. Two problems:

1. **It is a list of one, and the conversation case doesn't resume.** The control carries
   `target_id`, but `_activate_home_resume_latest` (`UI/Screens/home_screen.py:806-832`)
   routes conversations to Console with the id dropped — there is no app-level
   "open conversation N in Console" seam. The capability exists internally
   (`UI/Console_Modules/workspace.py:2121`, `_resume_console_workspace_conversation`,
   with defined TASK-717 failure semantics) but nothing exposes it.
2. **Home's "Recent" section omits the user's actual content work.**
   `_local_recent_work_items` (`Home/active_work_adapter.py:696-736`) only mirrors
   watchlist runs, chatbook artifacts, and finished ingest jobs — not conversations,
   notes, or media.

"What's next" exists as a fixed-priority ladder (`choose_next_best_action`,
`Home/dashboard_state.py:314-403`). It is state-driven but two documented open-task
feeds never reach it (eval runs, read-it-later), one input is dead
(`failed_schedule_count` is never produced — `active_work_adapter.py:168-186`), and its
resume branch routes bare to `chat`.

Freshness note, verified against the code: Home revisit-refresh already works —
screens are never reused, and every Home mount runs
`_refresh_home_active_work_cache` (`home_screen.py:277` → adapter
`refresh_active_work_cache_async`). Task-2763's actual defect is the *while-mounted*
freeze (no timer/subscription; statuses only move on click or remount). Recents
inherit that freeze until task-2763 lands — acceptable here, since a recents list is
far less time-sensitive than running-work statuses, and fixing live-refresh is that
task's own scoped AC (poll/subscription + TTL + test), not this spec's. Nothing
tracks item opens/usage (confirmed by task-18921); that stays out of scope.

## Goals

- A "What you were working on last:" affordance on Home listing the user's most recent
  work across conversations, notes, and media, each resumable in one action.
- Resume deep-links that actually land on the item: conversations in Console, notes in
  the Library notes editor, media in the Library item view.
- "What's next" suggestions informed by existing open-task queues (pending/failed eval
  runs, read-it-later) in addition to the current ladder inputs.
- Recents are current on every Home visit via the existing on-mount refresh path,
  extended to the new providers (no new refresh machinery).

## Non-goals (v1)

- **Opens/reading tracking.** Recency = `last_modified` (edits, messages, reading
  progress). Reading a note without editing it will not surface it. Phase 2: a
  lightweight opens journal (IDs + timestamps only, `model_catalog_cache.json` pattern)
  that slots into the same recents seam and also feeds task-18921.
- **Pattern-learning suggestions** ("you usually follow X with Y"). Out.
- **Eval-run resume deep-links.** The Evals screen has no `apply_navigation_context`
  support; suggestions route to the screen only. Follow-up if wanted.
- **Auto-restore of the last session at boot.** Screen snapshots are memory-only by
  documented design (`UI/Screens/chat_screen_state.py:80-88`); not reversing that here.
- **New DB schema, migrations, or config keys.**

## Approaches considered

### A. Unified recent-work stream + Console deep-link seam (chosen)

Merge conversations/notes/media recency into the existing recents pipeline via the
tables' existing `last_modified` columns; expose conversation-by-id Console opening as
a navigation-context contract; feed the ladder the two missing queues; recompute on
Home visit. No schema changes, no new persistence, no write-path instrumentation.

### B. Opens journal ("working on" = touched, not just edited)

Everything in A plus a persistent opens journal written whenever an item is opened.
Truer to "working on" (captures read-only sessions), but requires write-path
instrumentation across Console/Library/Media and a new persistent store. Deferred to
phase 2; the v1 seam is shaped so it can slot in.

### C. Last-session snapshot restore (rejected)

Persist last foreground screen + item on graceful shutdown, offer "Resume last
session". Only ever one item, breaks on crash, and conflicts with the deliberate
memory-only screen-snapshot decision. Strictly weaker than A.

## Design

### 1. Recents stream (Home adapter)

Extend `LocalNotificationHomeActiveWorkAdapter` with three content-item providers:
`_local_conversation_recent_items()`, `_local_note_recent_items()`,
`_local_media_recent_items()`. Each is a limit-N `last_modified DESC` read:

- Conversations: `conversations.last_modified` (list ordering already exists at
  `DB/ChaChaNotes_DB.py:8233`). Filter to the same visibility the Console/Library
  conversation lists apply (soft-deleted excluded, roleplay conversations included —
  they are rows in the same table).
- Notes: `notes.last_modified` (ordering at `Notes/Notes_Library.py:920+`), same
  filters as the Library notes list.
- Media: coalesce `Media.last_modified` with `ReadingProgress.last_modified` so
  reading progress bumps recency (join key verified during implementation).

Providers map rows to `HomeActiveWorkItem` with source labels
`Conversations` / `Notes` / `Media`, neutral status (`ready`, matching the chatbook
recents pattern), and appropriate `detail_route`s. `_local_recent_work_items` merge-
sorts all six sources by `updated_at` and keeps the existing cap of 8
(`_HOME_RECENT_WORK_LIMIT`). The single newest **content** item
(conversations/notes/media only) is reserved for the canvas banner instead of
repeating as the first Recent row — the same dedupe shape as the existing chatbook
rule (`_local_chatbook_artifacts()[1:]`), whose own behavior is unchanged.

All DB reads go through the adapter's existing off-loop compute path
(`asyncio.to_thread`, `active_work_adapter.py:640`). Titles are user text and are
markup-escaped at item-build time, following the `_watchlist_run_title` /
`build_home_resume_control` hazard pattern (escape exactly once).

### 2. "What you were working on last" canvas banner

The idle-canvas resume control is promoted from "newest note-or-conversation"
(`_home_resume_fields`, `home_screen.py:201-230`) to **newest item across the three
content providers** (conversations/notes/media — watchlist runs, chatbooks, and
ingest jobs keep their existing active/recent routing and never feed the banner),
which adds media as a resume kind:

- conversation → Console (seam below); if a live Console session already holds the
  conversation, switch to that session instead of rehydrating
  (`_console_session_id_for_workspace_conversation`, `workspace.py:2100`).
- note → Library notes editor via `LIBRARY_NAV_CONTEXT_NOTE_ID` (existing contract).
- media → Library item view via `LIBRARY_NAV_CONTEXT_OPEN_SOURCE_TYPE/ID`
  (`media`).

Banner copy: `What you were working on last:` + `Resume conversation/note/reading:
<title> · <relative time>`.

Source-of-truth consolidation: today `resume_kind/id/title` come from limit-1 seam
queries inside `_build_home_content_snapshot` (`home_screen.py:367-422`) via
`_home_resume_fields`. With the adapter now computing the full content-recents
stream, those limit-1 queries and `_home_resume_fields` are retired; the banner
fields derive from the top content item on `HomeDashboardInput`, so there is exactly
one definition of "newest work". Media as a resume kind is new plumbing: a
`HOME_RESUME_KIND_MEDIA` constant, a `build_home_resume_control` branch, and a
dispatch branch in `_activate_home_resume_latest`. The remaining items are the rail's Recent section below
(newest content item not duplicated, per §1). Rail Recent rows get open controls on
the task-153/task-2238 row pattern.

### 3. Console conversation deep-link seam (ADR required)

New navigation-context key `CONSOLE_NAV_CONTEXT_CONVERSATION_ID` in the
`Constants.py` nav-context block (alongside the `LIBRARY_*` keys), consumed by a new
`apply_navigation_context` implementation on `chat_screen.py` (Console becomes the
sixth screen implementing the existing contract — Library, Watchlists, Personas,
Settings, STTs precede it).

Timing, pinned to the actual call site: `handle_screen_navigation` invokes
`apply_navigation_context` on the freshly constructed screen **before**
`switch_screen` (`app.py:8937-8948`; exceptions are logged-and-swallowed, and an
awaitable return would be awaited — but Console must not do async work there, since
the screen is not yet mounted or on the stack). Console's implementation is
store-and-defer: record the conversation id synchronously, consume it in the
existing mount-complete flow.

Precedence rule: if a pending handoff (e.g. `CONSOLE_LIVE_WORK`) and a conversation
nav context are both present at mount, the nav context wins and the handoff is
dropped with a debug log — the nav context is the user's explicit, most-recent
intent.

Behavior, in order:

1. Context present and a live session already maps to that conversation → switch
   session (`controller.switch_session`), no rehydration.
2. No live session → call the existing `_resume_console_workspace_conversation`
   path (hydration via `Chat/console_conversation_hydration.py`).
3. Missing/invalid id → no-op with the existing TASK-717 toast semantics (the resume
   path already distinguishes transient-unavailable vs missing-record and owns the
   UX).

Callers: the Home banner + Recent rows, and the ladder's `resume_active_work`
branch, which stops routing bare to `chat`. Alternative considered and rejected:
the pending-handoff single-slot store (`CONSOLE_LIVE_WORK` pattern) — handoffs are
ephemeral, single-slot, and not a durable deep-link contract; nav context is.

### 4. "What's next" ladder feeds

Spine unchanged (fixed-priority triage ladder). New `HomeDashboardInput` fields and
branches:

- `pending_eval_run_count` / first failed eval run (Evals DB, `eval_runs.status`) →
  suggestion "Review pending/failed eval runs", route `evals`. Count only `pending`
  and `failed` statuses, never `running` — a crashed app orphans `running` rows
  forever, which would permanently pin this suggestion.
- `read_later_count` (`MediaReadItLaterState`) → suggestion "N items in read-it-later",
  route to the read-it-later view.
- Produce `failed_schedule_count` from the schedules service, closing a documented
  dead input — decision rule: include if it is a single query against existing
  schedule state; if it needs new aggregation machinery, skip it here and file a
  follow-up task instead.

New branches slot low — below notifications, above `import_sources` — exact placement
tuned during implementation against the existing branch table.

Scope correction from review: the existing `resume_active_work` branch is left
untouched — it fires only when live work *is* running (`_active_run_count`) and its
bare `chat` route is already correct; deep-linking it to a specific conversation
would change its meaning. The prior-work-informed part of v1 is the banner/Recent-row
deep links (§2), plus one ladder change (decision flagged, default on): when the
ladder falls through to the terminal `start_console` suggestion and a recent
conversation exists, the terminal suggestion becomes "Resume last conversation:
\<title\>" via the Console seam, falling back to "Start a conversation" on fresh
profiles. This is the direct answer to "suggestions based on prior work" and reuses
the seam. Nothing pattern-learning.

### 5. Freshness

No new refresh machinery. Revisit freshness already exists — screens are never
reused and every Home mount runs `_refresh_home_active_work_cache`
(`home_screen.py:277`); the new providers execute inside that same adapter compute,
so recents are current on every visit by construction. The while-mounted freeze is
task-2763's own scoped AC (poll or subscription, TTL-respecting, with a driving
test) and is deliberately not pulled in here; until it lands, Home keeps its
documented "snapshot with buttons" quirk, tolerable for a recents list. One
regression guard is required: a test asserting the on-mount refresh path runs (and
therefore includes the content providers).

### 6. Error handling and edge cases

- Row deleted since snapshot: dispatch validates existence before navigating;
  missing conversation follows TASK-717 `False` semantics (toast, caller-owned).
- Empty state: no recents → Recent section stays hidden (existing empty-section
  behavior); banner hidden, ladder falls through to existing branches.
- First run: unchanged — `import_sources` branch wins.
- Titles: raw at data layer, escaped exactly once at control/row build.
- All new DB reads off the event loop via the adapter's existing to-thread compute.

## Testing

- **Unit — dashboard_state:** merged recents ordering/cap/dedupe (newest content item
  feeds banner, not Recent), markup escaping, banner target selection per kind,
  ladder with the new fields (eval/read-later branches, placement, exclusions).
- **Unit — chat_screen nav context:** conversation id → live-session switch vs
  hydrate vs missing-id no-op.
- **Integration — adapter:** in-memory SQLite seeded with conversations/notes/media
  (incl. soft-deleted), eval runs, read-it-later rows; asserts recents content and
  counts.
- **Unit — Home mount:** on-mount refresh triggers the adapter refresh (the §5
  regression guard).
- Targeted runs only per repo testing guidelines (`lessons-testing-evidence.md`); no
  full sweep unless requested.

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/NNN-console-conversation-deep-link-nav-context.md
Reason: cross-module interface contract (Home/ladder → Console via nav context,
        making Console the 6th apply_navigation_context screen).
```

Number assigned at implementation time (repo is mid ADR renumbering — see current
branch). Created before implementation begins; linked from the backlog tasks.

## Implementation sequencing (proposed tasks, created after approval)

1. **Console deep-link seam** — ADR + `CONSOLE_NAV_CONTEXT_CONVERSATION_ID` +
   `chat_screen.apply_navigation_context`. Foundational; nothing resumes
   conversations without it.
2. **Recents stream + banner** — adapter providers, merged Recent section, banner
   promotion (media resume kind), retirement of the limit-1 resume seam queries,
   row open-controls. Depends on 1 for conversations.
3. **Ladder feeds** — eval/read-later inputs and branches, `failed_schedule_count`
   producer, terminal resume-last-conversation suggestion. Depends on 1 only for
   the terminal suggestion; otherwise independent of 2.

## Open questions (resolve during implementation, none blocking design)

- Media recency join: confirm `ReadingProgress` keyed by media id and cheap to
  coalesce; if the join is heavy, v1 uses `Media.last_modified` alone.
- Whether the rail "Recent" section header relabels to "Recent work" (cosmetic).
- Relative-time formatting helper: reuse an existing formatter if one exists in the
  codebase rather than adding a new one.
- Route id for the read-it-later suggestion target — verify the Media/read-it-later
  view's registered route before wiring the `HomeAction`.
- `console_available` flags on content recents rows (conversations true, notes/media
  false) — pinned during task 2.
