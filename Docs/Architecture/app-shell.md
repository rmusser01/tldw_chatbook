# App shell and navigation architecture

This document describes the `TldwCli` Textual application: startup phases, the screen-based navigation system, destination routing, the event/worker/reactive conventions, the generated CSS stack, and the splash and settings surfaces. It is the composition root that every other subsystem hangs off of.

## Authoritative files

| File | Role |
| --- | --- |
| `tldw_chatbook/app.py` | `TldwCli` — composition root (ADR-036): config load, service graphs, navigation dispatch, global bindings, boot worker fleet, quit flow |
| `tldw_chatbook/Constants.py` | `TAB_*` route ids, `TAB_DISPLAY_LABELS`, navigation-context contract keys, subscription/splash constants |
| `tldw_chatbook/UI/Navigation/shell_destinations.py` | `ShellDestination` dataclass, `SHELL_DESTINATION_ORDER`, `SHELL_DESTINATION_SHORTCUTS`, `resolve_shell_route()` |
| `tldw_chatbook/UI/Navigation/screen_registry.py` | `ScreenRoute`, lazy `_SCREEN_ROUTES` map (~30 routes), `_SCREEN_ALIASES`, `resolve_screen_target()` |
| `tldw_chatbook/UI/Navigation/main_navigation.py` | `NavigateToScreen` message, `NavigationButton`, `MainNavigationBar` |
| `tldw_chatbook/UI/Navigation/base_app_screen.py` | `BaseAppScreen` — per-screen chrome (nav bar + content + `AppFooterStatus`), snapshot seam, veto hooks |
| `tldw_chatbook/UI/Navigation/screen_state_store.py` | `ScreenStateStore` — per-route snapshots keyed by runtime identity (ADR-033) |
| `tldw_chatbook/Utils/boot_worker_policy.py` | `BOOT_WORKER_POLICY`, `StaggeredBootWorkerGate` — two-tier boot worker fleet |
| `tldw_chatbook/Logging_Config.py` | redacting rotating file logs, loguru→stdlib bridge, in-app log screen feed, crash forensics |
| `tldw_chatbook/css/build_css.py` | CSS module build: 65 ordered `.tcss` sources → 5 generated sheets |

## The app class

`TldwCli(TextSelectionCrashGuard, LibraryIngestQueueMixin, App[None])` mixes in a crash guard for Textual 8.x text-selection `MouseDown` crashes; the mixin must sit before `App` so its `on_event` wrapper is the last line of defense.

Construction is phased and timed (`self._startup_phases`, histogram `app_startup_phase_duration_seconds`):

1. **basic_init** — `TieAwareStylesheet` (reparses when a stored tie-breaker changes), `load_settings()` config, `ConsoleRuntime` (app-owned, outlives every `ChatScreen`), `ScreenStateStore`, `PendingHandoffStore`, config-failure snapshot, advisory per-profile instance lock (detection only, never blocks boot), `DBStatusManager`, UI responsiveness monitor.
2. **attribute_init** — `self._use_screen_navigation = True` (always), local-server `subprocess.Popen` handles (llamacpp, llamafile, vllm, ollama, mlx, onnx).
3. **parallel init** — independent services start on worker threads.

App-level state includes the DB handles (`media_db`, `prompts_db`, `subscriptions_db`) and the local model-catalog service.

`compose()` yields only a `SplashScreen` widget when `[splash_screen] enabled` (default true); otherwise it mounts exactly one `Container(id="screen-container")`. Screen-based navigation is exclusive — each `BaseAppScreen` composes its own chrome; there is no multi-window tab UI anymore (ADR-014 retired it; `hide_inactive_windows()` survives as a vestigial no-op).

`action_quit` runs as `run_worker(group="application-quit", exclusive=True, exit_on_error=False)`, consults the current screen's `confirm_quit`/`prepare_for_quit` and the console-runtime quit confirmation; any pre-quit failure keeps the app alive with a notify.

## Navigation system

Three layers resolve a route string to a screen:

1. **Destinations** (`shell_destinations.py`) — 15 shell destinations in strip order (home, console, library, personas, watchlists, artifacts, schedules, workflows, mcp, acp, lab, logs, settings, research, meetings). `_ROUTE_MAP` folds legacy route names at import time; unknown routes resolve to themselves.
2. **Screen registry** (`screen_registry.py`) — lazy `module_path`/`class_name` per route so `app.py` never imports screen classes directly (circular-import guard). `resolve_screen_target()` resolution order: aliases → routes → shell destinations (so `"lab"` → `"llm"`, `"console"` → `"chat"`). `dependency_check` gates optional screens via `Utils/optional_deps.py`.
3. **App dispatch** (`app.py`) — `@on(NavigateToScreen)` runs `handle_screen_navigation` as a worker (an inline await of a confirm dialog would starve the App's single message task — the "zombie-modal soft-lock"), under an `asyncio.Lock` that gives overlapping navigations FIFO ordering.

### Screen lifecycle rules

- Every navigation builds a **fresh** screen instance. Re-mounting an unmounted instance raced teardown and caused a total silent UI freeze (root-caused 2026-07-11).
- Exception: routes with `ScreenRoute.reusable=True` — home, chat (Console), library — are `install_screen`-cached per runtime identity and re-switched to; Textual suspends installed screens instead of unmounting them.
- Console message history lives in the app-owned `ConsoleRuntime`, not in snapshots.

### Navigation flow

1. Any widget posts `NavigateToScreen(route, context)` (or presses a `NavigationButton`).
2. The app handler acquires the FIFO navigation lock; refuses navigation before the initial screen push or while a screen with `blocks_stray_navigation` is on the stack (first-run setup wizard gate).
3. Outgoing screen (stack index 1, **not** `self.screen`, which may be an overlay) passes fail-closed veto seams: `flush_pending_work` (shielded, 5 s timeout; a timed-out flush keeps running and is retained against GC), `confirm_navigation`, `acquire_navigation_transition`.
4. The store saves the outgoing snapshot (Console snapshots are always discarded and re-published with a prompt-target projection); the app ensures screen-owned CSS; the incoming screen is reused or constructed; the snapshot is restored; `navigation_context` is applied (legacy routes map through a context table).
5. Pushed navigation overlays are dismissed (bounded at 16 dismissals, resolving `push_screen_wait` futures so awaiters don't hang), `switch_screen` runs, and the target commits ownership.
6. Any failure is fail-closed: stay on the current screen, notify, roll the nav-bar highlight back to the actual current route.

## Event system

The shell's routing message is `NavigateToScreen`. Beyond that, the codebase's event story has three tiers:

- **Widget-local messages handled by `@on(...)` on the owning screen** — this is how the live Console works (e.g. `@on(WorkspaceTreeConversationSelected)` in `chat_screen.py`).
- **Live shared event modules** — `Event_Handlers/TTS_Events/`, `STTS_Events/`, `media_events.py`, plus function-style handler modules (`ingest_events.py`, `notes_events.py`, `LLM_Management_Events/*` per backend, `eval_db_operations.py`, `Chat_Events/chat_rag_events.py`, `chat_image_events.py`).
- **Dormant catalog** — `Event_Handlers/Chat_Events/chat_messages.py` defines ~45 message classes (`LLMResponseChunk`, `RAGSearchRequested`, …) that **nothing imports**; they are the aspirational target of an unfinished migration described by that package's `MIGRATION_GUIDE.md`. Do not wire new code to them.

Worker completions funnel through one hook: `on_worker_state_changed` releases staggered-boot slots, persists `worker_failed` diagnostics (ADR-029), then delegates to `WorkerHandlerRegistry` (`Event_Handlers/worker_handlers/`). Only `Worker.name` may be persisted — `Worker.description` embeds worker arguments verbatim (prompts, API keys).

## Workers and boot fleet

Conventions: `run_worker(coro, group="<kebab-group>", exclusive=True|False, exit_on_error=False)`; `exit_on_error=False` on anything that must not kill the app; `@work(thread=True)` for blocking work with `call_from_thread` for UI updates.

Boot background work is governed by `Utils/boot_worker_policy.py`: an IMMEDIATE tier (scheduler loop, ingest restore) and a STAGGERED tier (actor-pack recovery/staging, FTS backfills for ChaChaNotes and subscriptions) capped at one concurrent worker, strictly serial, released through `StaggeredBootWorkerGate` after `_ui_ready`, with a backstop reconcile timer. A separate daemon-thread screen-module pre-importer (deliberately not a Textual worker) warms the priority routes (`chat`, `library`, `settings`) 0.2 s after mount.

The model-catalog refresh worker group is named by `Constants.MODEL_CATALOG_REFRESH_WORKER_GROUP` so dispatch sites and the worker-handler acknowledgement set cannot drift.

## Reactive conventions

`recompose=True` is used sparingly and deliberately. The canonical current use is pane-level data reactives (Watchlists panes rebuild just that pane's children). The documented anti-pattern is a screen-level `recompose=True` reactive; Settings removed three such reactives because watchers raised `NoActiveAppError` during early compose. Any recompose passes through `BaseAppScreen.refresh()`, which releases stale mouse capture and restores keyboard focus.

## CSS and design tokens

Sources live in `tldw_chatbook/css/{core,layout,components,features,utilities}/*.tcss`; `build_css.py` compiles 65 ordered modules (starting `core/_variables.tcss`, `_reset.tcss`, `_base.tcss`, `_typography.tcss`) into five generated sheets: the app bundle, two consolidated widget-defaults sheets (class-level `BUNDLED_CSS` consolidation dodges Textual's `LRUCache(64)` parse cache — a full destination tour used to hit 94 sources and 125–380 ms cold parses), and two screen-CSS sheets bracketing the bundle. Never edit `tldw_cli_modular.tcss` directly; rebuild with `python tldw_chatbook/css/build_css.py`. `check_bundle_sync.py` plus a build manifest guard the generated files.

The Console sheet is deliberately boot-parsed (the Console is the initial tab; lazy loading cost a ~100 ms full-app restyle), while the library/settings sheets load lazily via each screen's `CSS_PATH` and the evals/scheduling/watchlists feature sheets load app-side on first navigation.

All visual values are `$ds-*` tokens in `core/_variables.tcss` (ADR-150): spacing, control sizing, motion durations, opacity, typography emphasis, colors/status, and component states. A governance test (`Tests/UI/test_design_token_governance.py`) fails on undefined `$ds-*` references and new hex literals. Themes (`css/Themes/themes.py`, `ALL_THEMES`, dozens including user-saved themes) hold the palettes.

## Keybinding conventions (ADR-031)

- `ctrl+q` (quit), `ctrl+p` (stable command palette), `f1` (help), `f6` (next pane) are app-global; screens must not bind them.
- Never bind terminal-convention keys (ctrl+c/v/x/s/d/z/a/r/w) for app actions.
- Screen actions use single-letter htop-style bindings; destructive actions require a confirm dialog.
- Footer hints must be truthful: advertised ⊆ bound, and advertised == working in the active context.
- The destination hotkey layer is one left-to-right keyboard walk: ctrl+1..ctrl+9, ctrl+0, then f2/f3/f4/f5/f7 — defined in `SHELL_DESTINATION_SHORTCUTS`, owned by the destination contract, and moving in lockstep with `SHELL_DESTINATION_ORDER`; destination-local bindings may not shadow these keys (ADR-152). The embedded Console terminal viewport encodes F-keys to the pty when focused.

## Settings surface

`UI/Screens/settings_screen.py` is the canonical settings destination (~30 categories via `SettingsCategoryId`), with companion modules `settings_config_models.py`, `settings_config_adapter.py`, and per-category default modules. The legacy `UI/Tools_Settings_Window.py` is deprecated and unreachable through navigation; `Widgets/enhanced_settings_sidebar.py` no longer exists. Do not add new settings anywhere but the Settings screen.

## Splash subsystem

`Widgets/splash_screen.py` (`SplashScreen`) with effects in `Utils/Splash_Screens/` by category (classic/environmental/tech/gaming/psychedelic/custom) plus custom cards from `examples/custom_splash_cards/`. Config under `[splash_screen]`: `enabled`, `duration` (default 7.0 s), `skip_on_keypress`, `show_progress`, `card_selection`. On close the app removes the splash widget, mounts the main UI, pushes the initial screen (first-run → Home; focus-mode request or default → Console/chat), and defers logging setup until then.

## Failure and fallback behaviors

| Case | Behavior |
| --- | --- |
| Optional dependency missing for a screen | `ScreenRoute.dependencies_available()` gate degrades the route; navigation logs and stays put |
| Screen construction or `compose_content` failure | Guarded; `BaseAppScreen` renders an error panel instead of exiting the app |
| Navigation failure at any veto/seam | Fail-closed: stay on current screen, notify, roll back nav highlight |
| `flush_pending_work` timeout | Warning notify; the shielded save keeps running and is retained against GC |
| Config load failure/schema conflict | Loader falls back to in-memory defaults; the failure is snapshotted and notified once mounted |
| Screen-owned CSS fails to load | Never raises; degrades to unstyled |
| FTS5 unavailable at a search boundary | LIKE fallback (see [database-layer.md](./database-layer.md)) — "the search box must never raise into the reader" |
| Model-catalog cache unreadable | Treated as absent; startup continues without persistence |

## Governing decisions

ADR-014 (retire legacy navigation chrome), ADR-015 (shell destination IA), ADR-016 (palette liveness and hotkey layer), ADR-031 (keybinding and footer conventions), ADR-033 (application session state ownership), ADR-036 (composition root), ADR-150 (design token system) — all in `backlog/decisions/`. Style constitution: `backlog/docs/design-language.md`.

## Verified gotchas

1. `Event_Handlers/Chat_Events/chat_messages.py` message classes have zero importers — live chat events are widget-local messages in `chat_screen.py`.
2. Screens are never cached except the three `reusable=True` routes; reusing unmounted instances caused a reproducible total silent UI freeze.
3. `self.screen` during navigation is an overlay, not the tab being left — the code uses `_navigation_outgoing_screen()` (stack index 1) everywhere it matters.
4. Navigation dispatch must run as a worker, or an open confirm dialog deadlocks all input.
5. Textual `set_timer(0.0)` raises `ZeroDivisionError` inside `Timer._run` on Textual 8 and silently never fires.
6. `Constants.py` still contains a large embedded legacy `css_content` string from the pre-modular-CSS era — it is not part of the live `css/` system.
7. `Worker.description` embeds worker arguments verbatim (prompts, API keys) — only `Worker.name` may be persisted.

## Related docs

- [console.md](./console.md) — the Console screen this shell hosts
- [database-layer.md](./database-layer.md), [llm-providers.md](./llm-providers.md) — layers the shell composes
- `Docs/Development/navigation-architecture-analysis.md` — historical analysis (screen-set paragraph is stale; rules still match)
