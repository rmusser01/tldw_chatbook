# Artifacts

This document describes the Artifacts surface: a top-level destination that was **retired into Library** (ADR-172), the composite catalog that reads existing owners without owning storage, kept briefings, and Daily Reports. Four surfaces show artifact content today.

## Route retirement (the current shape)

- `screen_registry._SCREEN_ALIASES` maps `"artifacts"` → `"library"`; no `ScreenRoute` exists for artifacts. The shell route map resolves `"artifacts"` to Library while preserving the public route name.
- `ARTIFACTS_COMPATIBILITY_SHORTCUT = "ctrl+6"` is a retained hotkey for the retired route, outside permanent navigation (there is no ctrl+6 slot in the destination shortcut walk); the palette still offers "Open Artifacts".
- `UI/Screens/artifacts_screen.py` (`ArtifactsScreen`) still exists but is **unrouted** — zero instantiations in app code. It survives as a test fixture and as a static-helper donor (its chatbook-console-launch builder is imported by the Library artifacts controller). Any doc claiming the screen is reachable by navigation is wrong: `NavigateToScreen("artifacts")` lands on `LibraryScreen`.
- The `chatbooks` route is a **separate** ZIP-pack manager route under Library, not part of this retirement.

## Where Artifacts surface today

1. **Library Artifacts browse (canonical)** — `UI/Library_Modules/library_artifacts_controller.py` + `Widgets/Library/library_artifacts_widgets.py` / reader shell; rail rows `artifacts-all/-reports/-chatbooks/-dreams` under an Artifacts section, with an `artifacts_open` preference.
2. **The catalog composite read** — `Library/library_artifacts_catalog.py::LibraryArtifactsCatalog` composes existing owners without owning storage: the chatbook registry's artifact snapshot, the live reports snapshot from SubscriptionsDB, kept reports from ChaChaNotes (`kept_briefings`), and the dreams artifact snapshot. Views `all|reports|chatbooks|dreams` plus a Kept filter; `ArtifactReadError` ensures an owner failure never renders as "empty".
3. **Watchlists artifacts pane** — per-watchlist briefings with Generate/Export/Keep actions; deep-linkable via the watchlists nav context (`section="artifacts"` + optional briefing id, local backend only).
4. **Home** — the newest chatbook artifact appears as an active row (older ones as recent), with open-details routing to the artifacts alias and an "Open Chatbook in Console" handoff staged through the pending-handoff store (`local:chatbook:` prefix).

## Daily Reports (the read model, ADR-079)

A "Daily Report" **is** a briefing: `Subscriptions/daily_reports_view.py` is a thin read-only derivation over the briefings/scripts/audio tables (`list_recent_reports`, timestamp normalization). **Kept briefings** (`Subscriptions/briefing_keep.py::keep_briefing`) copy a completed briefing plus its scripts into denormalized ChaChaNotes tables so they survive watchlist deletion; refusal (`KeepRefused`) happens before any write; idempotent on `source_briefing_id`, race-safe via conflict folding. The kept badge requires a per-row cross-DB lookup — the reason it was deferred in ADR-079. Live and kept reports have separate namespaced identities; "all reports" totals count copies. Demo seeding (`daily_report_demo.py`) installs a real Daily Brief watchlist (RSS sources, preset, 24 h cadence) with a dismissible banner.

## Dataflow

1. Scheduled or manual briefing completes (see [subscriptions-watchlists.md](./subscriptions-watchlists.md)); scheduled completions auto-keep and dispatch a `category="briefing"` notification.
2. Library Artifacts browse queries the catalog; the catalog fans out to its owners (chatbook registry, SubscriptionsDB, ChaChaNotes kept tables, dreams snapshot) and merges bounded pages.
3. Watchlists artifacts pane lists the same briefings per watchlist; Keep writes the denormalized copies.
4. Home surfaces the newest chatbook artifact as an active-work row with console/open-details handoffs.

## Governing decisions

ADR-172 (`172-library-artifacts-browse-and-navigation.md` — note: its written status line says "implementation pending", but the route alias, Library sections, and catalog are demonstrably implemented; the ADR lags the code), ADR-079 (`079-daily-reports-surface-and-demo-seeding.md`), ADR-123 (artifact share web export), ADR-015 (Artifacts was the fifth destination pre-172), ADR-078 (no second universal artifact DB). Specs: `2026-08-01-kept-briefings-design.md`, `2026-08-29-daily-reports-demo-design.md` (under `Docs/superpowers/specs/`).

## Verified gotchas

1. `NavigateToScreen("artifacts")` resolves to Library — the old screen class is unreachable.
2. Cross-DB kept lookups are per-row, not joins.
3. The retired screen's dreams modals are reused by the Library artifacts controller — the file is not dead code, just dead UI.

## Related docs

- [subscriptions-watchlists.md](./subscriptions-watchlists.md) — briefing production
- [library-shell.md](./library-shell.md) — the host destination
- [home.md](./home.md) — artifact rows in active work
