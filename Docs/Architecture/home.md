# Home screen

This document describes the primary destination: the dashboard/triage surface — its pure state module, the active-work adapter seam, per-visit dispatch, the next-best-action engine, deep links, and the honest-unavailable control model.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Screen | `UI/Screens/home_screen.py` | `HomeScreen(BaseAppScreen)` — workers, button dispatch, deep links; `HOME_CONTROL_METHODS` maps control ids → app methods |
| State (pure) | `Home/dashboard_state.py` | dataclasses (`HomeDashboardInput` ~40 readiness/content fields, `HomeActiveWorkItem`, `HomeDashboard`), the triage builder, `choose_next_best_action` |
| Adapter seam | `Home/active_work_adapter.py` | `HomeActiveWorkAdapter` Protocol, `LocalNotificationHomeActiveWorkAdapter`, `UnavailableHomeActiveWorkAdapter`; seam queries + a 3 s TTL cache |
| Rail | `Home/home_rail_state.py`, `Widgets/Home/home_rail.py`, `Widgets/Home/home_canvas.py` | rail sections **Needs Attention / Running / Recent** (+ details), preferences coercion, the right-hand detail canvas |
| App-side controls | `app_destinations.py` | `open_active_home_item_details`, `open_active_home_item_in_console`, `retry_active_home_item`, `open_home_flashcards_review` |
| Wiring | `app_service_wiring.py` | adapter construction with all providers |

## Per-visit dispatch

Home is a reusable route, and its only dispatch seam is **`on_screen_resume`** — deliberately not `on_mount`, which would double-run the first visit (worker exclusivity cannot undo synchronous provider calls already executing). Each visit fires three exclusive workers: a chatbook-artifact/open-tasks snapshot refresh, a content snapshot build, and an adapter cache warm. Landed workers patch the rail/canvas/header in place — no recompose.

Completion behind another screen records pending presentation instead of rebuilding
the hidden dashboard (TASK-35612). The next successful cache warm replays it even
when the snapshot/cache contents are unchanged; failed or cancelled warming leaves
it pending. The adapter TTL and native checks remain unchanged. Content builds
stop issuing further queries once the mounted Home is hidden (TASK-38267). A
read already issued remains awaited through its existing cleanup; a hidden
partial build never replaces the last complete snapshot. The next visit starts
the ordinary refresh. This suspension rule does not change the preexisting
exclusive-worker cancellation behavior on a rapid subsequent resume.

## Dashboard build (dataflow)

1. Visit → the three workers start.
2. `_build_dashboard_input`: the adapter composes `active_work_items` from three sources — **watchlist runs** (ids `local:watchlist_run:<id>`), **chatbook artifacts** (only the newest active one), and **library ingest jobs** (QUEUED/PARSING/WRITING/FAILED; DONE rows move to recent) — plus `recent_work_items` (terminal runs, older artifacts, completed ingests; capped at 8).
3. Unread notifications count comes from the notifications inbox (`ClientNotificationsService.list_queue`, counting `not is_read`); server-event fields from the server-event feed observation.
4. The screen then **overrides** two inputs with real seams: `model_ready` uses the same send-path readiness the Console uses, and `acp_ready` reflects the ACP runtime process status.
5. The content snapshot merges in (console readiness, content counts, a resume candidate, recent content).
6. `build_home_triage_state` buckets items by run status (approval/failed/running/paused) into the rail sections, adds a synthetic flashcards-due row, and builds the canvas for the selection or the next-best action; details lines carry the readiness chips (Model/RAG/MCP/ACP/Mode/Server/Active/Approvals).

## Next-best-action engine

`choose_next_best_action` priority: fix model setup → review approvals → recover schedules → review failed work → resume active work → review notifications → review eval runs → review read-later → import sources (when the library is empty) → search library → resume last conversation → start console. An `exclude` parameter lets the canvas fall through when the top action duplicates the selected item's own recovery control.

## Deep links

- Primary actions carry contexts: settings providers/models category; resume-last-conversation; read-it-later subview; watchlists notifications/runs sections.
- Content items: `local:conversation:` → Console resume; `local:note:` / `local:media:` → Library note/open-source contexts; unknown prefixes are silent no-ops.
- Active-work details: watchlist-run targets → Watchlists runs section (backend parsed from the id prefix); ingest targets → Library ingest; chatbook targets → the artifacts route (which resolves to Library — see [artifacts.md](./artifacts.md)).
- "Open in Console" builds a `HomeConsoleLaunch` through `app.open_console_for_live_work`; flashcards review opens Study with a Home breadcrumb.

## Honest-unavailable controls

Approve/reject/pause/resume are **not wired** on this tree: the local adapter implements only open-details and open-in-console for its id kinds; everything else falls through to the unavailable adapter's toast ("… is not connected to an active run service yet."). Retry is real for ingest jobs (registry requeue or research-source retry); permanent validation-class ingest failures withhold the Retry button. First-run guidance is data-driven (a `model_ready=False` dashboard yields "Set up Console model"), not a wizard. Focus mode is app-level and Console-only — Home's "focus canvas" is just the detail pane.

## Boundaries and failure behavior

- Every content seam call degrades to `None` with a debug log; totals never fall back to page-sample lengths (Home fetches limit-1…limit-8 pages and trusts `total` fields).
- In-memory `:memory:` DBs force seam calls back onto the UI thread (thread-locality guards); unknown service shapes stay inline.
- The adapter TTL cache is invalidated after every control action.
- Server-event feed states are surfaced honestly (`available/empty/requery_required/reconnect_required/unavailable`); a policy denial in local mode is normal and debug-logged.
- Markup discipline: rail button labels escape titles exactly once (they parse Rich markup); canvas lines render `markup=False`; resume titles are truncated **before** escaping.
- Rail section collapse preferences persist to `home.rail_state` off-thread; persistence failures are swallowed.

## Governing docs

No dedicated Home ADR exists. Specs: `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md`, `2026-08-29-home-recents-resume-and-next-up-design.md`. QA evidence: `Docs/superpowers/qa/home-triage-h1-2026-07/`. ADR-015 seats Home as the first destination.

## Verified gotchas

1. `on_screen_resume` is the only dispatch seam — adding refresh to `on_mount` double-runs the first visit.
2. Selection-scoped canvas controls dispatch before the unscoped dashboard controls; the "not connected yet" warning fires only when the app method is missing entirely.
3. Counts that are `None` (seam missing) render nothing — only known positive counts appear.
4. The test-input override skips cache warming entirely, so tests must not assert cache behavior through it.

## Related docs

- [artifacts.md](./artifacts.md), [subscriptions-watchlists.md](./subscriptions-watchlists.md) — where active-work items come from
- [notifications.md](./notifications.md) — the unread-count seam
- [app-shell.md](./app-shell.md) — reusable routes and `on_screen_resume`
