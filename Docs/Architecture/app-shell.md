# App shell and navigation architecture

This document describes the `TldwCli` Textual application: startup phases, the screen-based navigation system, destination routing, the event/worker/reactive conventions, the generated CSS stack, and the splash and settings surfaces. It is the composition root that every other subsystem hangs off of.

## Authoritative files

| File | Role |
| --- | --- |
| `tldw_chatbook/app.py` | `TldwCli` — composition root (ADR-036): config load, service graphs, navigation dispatch, global bindings, boot worker fleet, quit flow |
| `tldw_chatbook/app_startup.py` | Delegated startup timers and footer wiring; app methods retain named dependency and callback routes |
| `tldw_chatbook/app_served.py` | Deferred served control/approval relay wiring; App keeps task, publisher and client ownership plus live dependency lookups (ADR-236) |
| `tldw_chatbook/Constants.py` | `TAB_*` route ids, `TAB_DISPLAY_LABELS`, navigation-context contract keys, subscription/splash constants |
| `tldw_chatbook/UI/Navigation/shell_destinations.py` | `ShellDestination` dataclass, `SHELL_DESTINATION_ORDER`, `SHELL_DESTINATION_SHORTCUTS`, `resolve_shell_route()` |
| `tldw_chatbook/UI/Navigation/screen_registry.py` | `ScreenRoute`, lazy `_SCREEN_ROUTES` map (~30 routes), `_SCREEN_ALIASES`, `resolve_screen_target()` |
| `tldw_chatbook/UI/Navigation/main_navigation.py` | `NavigateToScreen` message, `NavigationButton`, `MainNavigationBar` |
| `tldw_chatbook/UI/Navigation/base_app_screen.py` | `BaseAppScreen` — per-screen chrome (nav bar + content + `AppFooterStatus`), snapshot seam, veto hooks |
| `tldw_chatbook/UI/Navigation/screen_state_store.py` | `ScreenStateStore` — per-route snapshots keyed by runtime identity (ADR-033) |
| `tldw_chatbook/Utils/boot_worker_policy.py` | `BOOT_WORKER_POLICY`, `StaggeredBootWorkerGate` — two-tier boot worker fleet |
| `tldw_chatbook/Logging_Config.py` | redacting rotating file logs, loguru→stdlib bridge, in-app log screen feed, crash forensics |
| `tldw_chatbook/css/build_css.py` | CSS module build: 64 ordered `.tcss` sources → 5 generated sheets |

## The app class

`TldwCli` composes `TextualAppGuards` before `App`. The aggregate in `Utils/text_selection_crash_guard.py` includes the Textual 8.x text-selection crash guard, detached-focus recovery, input activity tracking, and thread-worker context isolation.

On native Windows, its `get_driver_class()` adapts Textual's default `WindowsDriver` to enable `ENABLE_WINDOW_INPUT` after every application-mode start, including resume (TASK-37173). Textual 8.2.8 enables VT input but omits this resize-reporting flag, leaving screens at their old dimensions after a terminal resize. The adapter preserves the remaining input flags and upstream console restoration; constructor-supplied and environment-selected drivers, including served transports, are unchanged. Unix does not import the Windows adapter path. Native before/after evidence and a reproducible Logs probe live in `qa/windows-terminal-resize-2026-10-28/`.

Construction is phased and timed (`self._startup_phases`, histogram `app_startup_phase_duration_seconds`):

1. **basic_init** — `TieAwareStylesheet` (reparses when a stored tie-breaker changes), `load_settings()` config, `ConsoleRuntime` (app-owned, outlives every `ChatScreen`), `ScreenStateStore`, `PendingHandoffStore`, config-failure snapshot, advisory per-profile instance lock (detection only, never blocks boot), `DBStatusManager`, UI responsiveness monitor.
2. **attribute_init** — `self._use_screen_navigation = True` (always), local-server `subprocess.Popen` handles (llamacpp, llamafile, vllm, ollama, mlx, onnx).
3. **parallel init** — independent services start on worker threads.

App-level state includes the DB handles (`media_db`, `prompts_db`, `subscriptions_db`) and the local model-catalog service.

`compose()` yields only a `SplashScreen` widget when `[splash_screen] enabled` (default true); otherwise it mounts exactly one `Container(id="screen-container")`. Screen-based navigation is exclusive — each `BaseAppScreen` composes its own chrome; there is no multi-window tab UI anymore (ADR-014 retired it; `hide_inactive_windows()` survives as a vestigial no-op).

`action_quit` runs as `run_worker(group="application-quit", exclusive=True, exit_on_error=False)`, consults the current screen's `confirm_quit`/`prepare_for_quit` and the console-runtime quit confirmation; any pre-quit failure keeps the app alive with a notify.

## Navigation system

Three layers resolve a route string to a screen:

1. **Destinations** (`shell_destinations.py`) — 13 shell destinations in strip order (home, console, library, personas, watchlists_collections, schedules, workflows, acp, lab, logs, settings, research, meetings). Artifacts (ADR-172) and MCP (ADR-240) are retired from the strip: their routes alias Library and Settings respectively, and ctrl+6/ctrl+9 survive as compatibility deep-links. `_ROUTE_MAP` folds legacy route names at import time; unknown routes resolve to themselves.
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
- **Live shared event modules** — `Event_Handlers/TTS_Events/`, `STTS_Events/`, `media_events.py`, plus function-style handler modules (`notes_events.py`, `LLM_Management_Events/*` per backend, `Chat_Events/chat_rag_events.py`, `chat_image_events.py`).
- **Deleted catalog** — the aspirational `Event_Handlers/Chat_Events/chat_messages.py` catalog (~45 message classes) and its `MIGRATION_GUIDE.md` were deleted (task-34890); the migration they described never landed, and live chat events are widget-local messages on the owning screen.

Worker completions funnel through one hook: `on_worker_state_changed` releases staggered-boot slots, persists `worker_failed` diagnostics (ADR-029), then delegates to `WorkerHandlerRegistry` (`Event_Handlers/worker_handlers/`). Only `Worker.name` may be persisted — `Worker.description` embeds worker arguments verbatim (prompts, API keys).

## Workers and boot fleet

Conventions: `run_worker(coro, group="<kebab-group>", exclusive=True|False, exit_on_error=False)`; `exit_on_error=False` on anything that must not kill the app; `@work(thread=True)` for blocking work with `call_from_thread` for UI updates.

Boot background work is governed by `Utils/boot_worker_policy.py`: an IMMEDIATE tier (scheduler loop, ingest restore) and a STAGGERED tier (actor-pack recovery/staging, FTS backfills for ChaChaNotes and subscriptions) capped at one concurrent worker, strictly serial, released through `StaggeredBootWorkerGate` after `_ui_ready`, with a backstop reconcile timer. A separate daemon-thread screen-module pre-importer (deliberately not a Textual worker) warms the priority routes (`chat`, `library`, `settings`) 0.2 s after mount. Between routes it yields to the existing app-owned Console custody projection, including received intent before controller setup; it resumes once turns settle and stops promptly on shutdown. A long response may delay speculative warming; direct navigation and the single-route initial splash warmup retain their normal loader. See ADR-230, TASK-34406.4. After the last route the same thread imports the Console's first-send modules (`Chat/console_first_send_modules.py`), paced and parked the same way, parses the built-in MCP manifest once and freezes the heap again (TASK-33620.15.5); it runs after the routes so a first navigation soon after boot finds its screen as warm as before (TASK-33620.15.14). Each third-party package the send path loads (aiohttp and its dependencies, html5lib, tinycss2, textual_serve) is its own step after its dependencies, so a navigation that arrives mid-warm-up waits behind one package's import at most; no step starts within 0.15 s of any input, whether a key, click, paste, scroll or mouse move (a navigation takes its lock only when the input reaches its handler; input that never pauses that long holds the warm-up back, TASK-33620.15.49), each step that loads modules freezes them at once, and these freezes count what they move instead of walking the frozen set (TASK-33620.15.18). Its module count is pinned as its own ADR-097 row (TASK-33620.15.13).

The model-catalog refresh worker group is named by `Constants.MODEL_CATALOG_REFRESH_WORKER_GROUP` so dispatch sites and the worker-handler acknowledgement set cannot drift.

## Reactive conventions

`recompose=True` is used sparingly and deliberately. The canonical current use is pane-level data reactives (Watchlists panes rebuild just that pane's children). The documented anti-pattern is a screen-level `recompose=True` reactive; Settings removed three such reactives because watchers raised `NoActiveAppError` during early compose. For screen content, `BaseAppScreen.refresh(recompose=True)` releases mouse capture when recomposition is scheduled. Keyboard focus capture and restoration live in `BaseAppScreen.recompose()`, covering both scheduled refreshes and direct awaits; child widgets such as `ConsoleTranscript` own their recomposition separately.

Console dialogs used only by explicit actions can use the existing lazy-export
facade in `Widgets/Console/__init__.py`. `ConsoleRenameSessionModal` keeps
its public export and resolves on first access instead of package import.
The session controller imports it inside its Rename handler (ADR-097).

## CSS and design tokens

Sources live in `tldw_chatbook/css/{core,layout,components,features,utilities}/*.tcss`; `build_css.py` compiles 64 ordered modules (starting `core/_variables.tcss`, `_reset.tcss`, `_base.tcss`, `_typography.tcss`) into five generated sheets: the app bundle, two consolidated widget-defaults sheets (class-level `BUNDLED_CSS` consolidation dodges Textual's `LRUCache(64)` parse cache — a full destination tour used to hit 94 sources and 125–380 ms cold parses), and two screen-CSS sheets bracketing the bundle. Never edit `tldw_cli_modular.tcss` directly; rebuild with `python tldw_chatbook/css/build_css.py`. `check_bundle_sync.py` plus a build manifest guard the generated files.

The Console sheet is deliberately boot-parsed (the Console is the initial tab; lazy loading cost a ~100 ms full-app restyle), while the library/settings sheets load lazily via each screen's `CSS_PATH` and the evals/scheduling/watchlists feature sheets load app-side on first navigation.

All visual values are `$ds-*` tokens in `core/_variables.tcss` (ADR-150): spacing, control sizing, motion durations, opacity, typography emphasis, colors/status, and component states. A governance test (`Tests/UI/test_design_token_governance.py`) fails on undefined `$ds-*` references and new hex literals. Themes (`css/Themes/themes.py`, `ALL_THEMES`, dozens including user-saved themes) hold the palettes.

## Keybinding conventions (ADR-031)

- `ctrl+q` (quit), `ctrl+p` (stable command palette), `f1` (help), `f6` (next pane) are app-global; screens must not bind them.
- Never bind terminal-convention keys (ctrl+c/v/x/s/d/z/a/r/w) for app actions.
- Screen actions use single-letter htop-style bindings; destructive actions require a confirm dialog.
- Footer hints must be truthful: advertised ⊆ bound, and advertised == working in the active context.
- The destination hotkey layer is one left-to-right keyboard walk: ctrl+1..ctrl+5, ctrl+7, ctrl+8, ctrl+0, then f2/f3/f4/f5/f7 — defined in `SHELL_DESTINATION_SHORTCUTS`, owned by the destination contract, and moving in lockstep with `SHELL_DESTINATION_ORDER`; destination-local bindings may not shadow these keys (ADR-152). ctrl+6 (Artifacts → Library, ADR-172) and ctrl+9 (Tools & MCP → Settings, ADR-240) sit outside the walk as retained compatibility deep-links for retired destinations. The embedded Console terminal viewport encodes F-keys to the pty when focused.

## Settings surface

`UI/Screens/settings_screen.py` is the canonical settings destination (~30 categories via `SettingsCategoryId`), with companion modules `settings_config_models.py`, `settings_config_adapter.py`, and per-category default modules. One category is a hosted workbench, not a form: **Tools & MCP** (`SettingsCategoryId.TOOLS`, pane module `settings_tools_hub.py`) re-homes the four-mode MCP workbench — Servers / Tools / Permissions / Audit — that used to be its own shell destination; the `mcp`/`tools_settings` routes alias Settings with a `{"category": "tools"}` default context (ADR-240). The legacy `UI/Tools_Settings_Window.py` was removed (TASK-33244); its surviving test-driven backup helpers live in `Backup_Recovery/settings_backup_helpers.py`. `Widgets/enhanced_settings_sidebar.py` no longer exists. Do not add new settings anywhere but the Settings screen.

Composer paste handling is a bounded child region owned by
`Widgets/Settings_Widgets/composer_paste_panel.py::ComposerPasteSettingsPanel`
(TASK-35676/35677). It keeps the existing field IDs and token-backed layout,
suppresses programmatic `Changed` messages, and posts `Edited` messages for
local changes. Settings retains the category draft, validation and Save/Revert
writer. `sync_values()` updates controls in place; equal threshold values do
not disturb cursor/selection. The panel imports only when its category renders.

The remaining eight Console inputs (five agent budgets, tool-result display cap,
side-chat model and prompt template) stay screen-owned (TASK-35867). Refreshes
use each Input's native `prevent(Input.Changed)` context. A synchronous public
value watcher remembers pending user edits in weak per-Input value queues
until their queued events arrive, so refresh cannot overwrite typing, even
when it returns to a previous display. Initialization, stale and duplicate events do not stage
edits; ordinary refresh retains Input identity, focus and selection. Existing
Settings draft, validation and Save/confirmed Revert writers retain authority
under ADR-033/161; this bookkeeping introduces no persistence boundary.

`SettingsScreen._category_action_supported()` supplies category capabilities
to action dispatch, guided-button composition and the custom footer/F1 help
(TASK-35678). Focus, dirty state, worker fences, validation and destructive
confirmation remain live checks with their existing owners. Guidance-only
shortcuts stay bound; Theme and Personal Context widgets project their own
focus/selection-dependent hints. These are the existing ADR-031/033 contracts.

Image/Video Settings completion acknowledges the submitted category draft, retaining later raw values through configuration reload. `UI/Settings_Modules/media_gen_save.py` owns the narrow completion/recomposition helpers; the screen retains staging and checked-write ownership. Current focus/selection and expanded backend sections survive replacement, and a deliberate focus move wins (TASK-36098).

The Settings Internal Prompts count is a display snapshot prepared after the
first refresh by an exclusive thread worker. Rendering and category filtering
never read prompt storage. The worker retains the original checked config
lifetime; config identity, request revision, cancellation and attachment checks
fence publication. Pending/failed reads display Loading/Unavailable, and prompt
edits update the existing rows without rebuilding the editor.

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

1. The `Chat_Events/chat_messages.py` catalog was deleted (task-34890) — live chat events are widget-local messages in `chat_screen.py`.
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
