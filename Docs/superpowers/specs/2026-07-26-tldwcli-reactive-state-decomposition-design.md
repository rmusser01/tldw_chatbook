# TldwCli Reactive State Decomposition Design

Date: 2026-07-26
Status: Implemented
Final task:
[TASK-906](../../../backlog/tasks/task-906%20-%20Close-TldwCli-reactive-ownership-with-installed-distribution-sentinels.md)
Final plan:
[TldwCli Reactive Ownership Closeout](../plans/2026-07-26-task-906-reactive-ownership-closeout.md)
ADR:
[ADR-033](../../../backlog/decisions/033-application-session-state-ownership.md),
[ADR-006](../../../backlog/decisions/006-provider-aware-generation-settings.md),
[ADR-011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md),
[ADR-029](../../../backlog/decisions/029-local-private-data-boundary.md),
[ADR-032](../../../backlog/decisions/032-immutable-installed-distribution-assets.md)

## Summary

This is the next application-state decomposition tranche after TASK-643
through TASK-646. It removes legacy domain and view state from `TldwCli`
without replacing it with another root store.

The verified root-reactive inventory has exactly 61 descriptors:

- retain 2 application-lifecycle reactives on `TldwCli`;
- move 1 provider selection contract to the existing Settings and Console
  owners, then delete its root descriptor;
- delete 58 dead, duplicate, or legacy-route descriptors.

Every migrated descriptor, initializer, writer, watcher, dynamic string
reference, callback, and snapshot dependency moves or disappears in one
atomic domain slice. There are no compatibility properties, mirrored writes,
or temporary root aliases.

The tranche also repairs four contracts exposed by the inventory:

1. the unused root button-dispatch map is deleted only after supported LLM
   action buttons are registered at their production destination and the
   unsupported custom Transformers server-launch block is removed while the
   retained model list/download output is rehomed with those controls;
2. Media messages stop at `MediaWindow`, preventing the destination and the
   legacy app handler from both applying one mutation;
3. command-palette provider changes become an explicit Console intent instead
   of a root reactive cache;
4. the orphaned TLDW API worker completion pipeline and its shared request
   context are deleted, leaving Library as the sole production ingest owner.

The design follows ADR-033 unchanged. It does not amend that accepted ADR and
does not create a duplicate decision record.

## Verified Baseline

The findings below were re-checked against the current production routes and
source before this specification was written.

### Root descriptors and routing

- `TldwCli` declares 61 class-level Textual reactive descriptors.
- `TldwCli.__init__` sets `_use_screen_navigation = True` unconditionally.
- `TldwCli.on_button_pressed()` logs a bubbled button and returns. It does not
  dispatch `button_handler_map`.
- `button_handler_map` is assigned once from `_build_handler_map()` and has no
  production read.
- `_build_handler_map()` is therefore dead infrastructure. Its dynamic
  `reactive_attr="..."` strings do not establish live ownership.
- `watch_current_tab()` immediately returns under the mandatory screen
  navigation mode. Its legacy tab-window show/hide implementation is not the
  production navigation path.
- The route registry maps the legacy CCP route to `PersonasScreen` with
  canonical tab `personas`; app handlers guarded by `current_tab == TAB_CCP`
  cannot run after successful canonical navigation.
- The legacy `notes`, `prompts`, `skills`, and retired `ingest` routes resolve
  to Library.
- The legacy Tools & Settings route resolves to MCP.
- Search, Media, Evals, LLM, Personas, and Chat use registered production
  screens.

### Destination ownership

- `ChatScreen` composes the native `ConsoleSessionSurface`.
  `_ensure_chat_window()` is defined but never called, so the legacy
  `ChatWindowEnhanced` is not mounted by the production Console route.
- Native Console provider/model choices already live in
  `ConsoleSessionSettings` inside `ConsoleChatStore`. Console rail state and
  native snapshots are already screen-owned.
- `PersonasScreen` owns character/persona selection, editors, workers, and
  center-pane state. It does not read the root CCP reactives.
- `SearchScreen` saves and restores the actual query, mode, and active
  `TabbedContent` tab from `SearchRAGWindow`.
- The standalone Ingest screen was retired by TASK-684.4. The `ingest` route
  resolves to Library, whose Import media canvas owns the active ingest view.
- `EvalsScreen` owns the workbench and has no collapsible app sidebar.
- The production Lab `LLMScreen` owns the lifted Models rail and drives the
  deferred real `LLMManagementWindow`, which owns `active_view` and supported
  provider/model actions.
- `MediaWindow` owns media type, list page, selection, search controls, and
  restored view state. It also declares its own `media_active_view` reactive.
- Library owns Notes editing and autosave. The root Notes timer is never
  scheduled; it is only toggled or stopped by legacy app code.

### Verified hidden failures

#### LLM action buttons

`LLMManagementWindow` initially has no live action handler. Start, stop,
browse, download, and related buttons exist in its production composition,
while their handler maps are reachable only from the unused app dispatcher.
Because `TldwCli.on_button_pressed()` returns, those production LLM actions
currently have no live destination dispatcher.

The LLM slice must repair action routing before deleting the root handler map.
Navigation remains owned by the production Lab `LLMScreen` rail and the
mounted body's `active_view`; the root `llm_active_view` and `llm_nav_events`
path are deleted.

#### Media mutation bubbling

`MediaWindow` handles `MediaMetadataUpdateEvent` through its scoped service.
`TldwCli` also handles the same bubbling message through the legacy
`media_events.handle_media_metadata_update()` path. Neither handler stops the
message.

Textual dispatches a message on the current node and then bubbles it to the
parent while `message.bubble` is true and propagation has not been stopped.
One production metadata event can therefore invoke both paths. The
destination handler must call `event.stop()` before awaited work, and the
legacy app registration must be removed.

#### Provider authority

`chat_api_provider_value` is currently:

- initialized from boot-time `chat_defaults`;
- written by Settings after a durable save;
- written by the command-palette LLM provider command;
- read by Console/provider resolution as an override;
- watched by legacy Chat UI code.

This mixes three lifetimes. Under ADR-006:

- Settings owns persisted global defaults;
- an active Console session owns its explicit provider/model snapshot;
- a command invoked away from Console is an in-memory destination intent.

The boot-time root reactive is not an authority and must not survive as a
compatibility cache.

#### Shared TLDW API request context

The rebuilt MediaIngestScreen and its `tldw_api_events.py` producer were
deleted by TASK-684.4 when the `ingest` route moved to Library. Latest `dev`
has no producer for the old `api_calls` worker group and no matching
`#tldw-api-*` widgets. It still retains
`app._last_tldw_api_request_context`, payload-bearing success/failure
consumers, a worker-registry branch, and compatibility exports.

Those remnants cannot complete a request and do not protect any compatibility
contract. Recreating a producer or adding an envelope would revive an obsolete
second ingest implementation. The correct boundary is deletion: Library's
ingest queue, server request mapping, and public server batch cancellation
remain the live production path.

## Goals

- Make the production owner of every root reactive unambiguous.
- Remove all destination domain/view reactives from `TldwCli`.
- Preserve only genuine app-lifecycle coordination state at the root.
- Preserve fresh-screen navigation, current snapshot precedence, and
  memory-only handoff behavior from ADR-033.
- Preserve Settings as the durable global provider-default owner and Console
  as the active session owner.
- Restore live LLM action routing at `LLMManagementWindow`.
- Remove visible LLM actions that have no implemented runtime contract rather
  than leaving inert controls or inventing a new server lifecycle.
- Ensure one Media event produces one destination mutation.
- Eliminate cross-request TLDW API context bleed.
- Remove dead watchers, initializers, dynamic references, and dispatcher
  entries with their descriptors.
- Add source and installed-wheel verification using only the production app
  or direct app-independent functions.

## Non-Goals

- Creating `AppState`, `AppSessionState`, a Redux-like store, a controller
  layer, a dependency-injection framework, or an event bus.
- Reopening the runtime-policy, screen-snapshot, or existing handoff decisions
  in ADR-033.
- Persisting screen state or provider-command handoffs to disk.
- Queueing provider commands or changing the handoff store's single-slot,
  last-write-wins behavior.
- Caching or remounting screen instances.
- Reworking service construction, shutdown, worker registry, timers, or
  process ownership.
- Moving LLM server process handles out of `TldwCli` in this tranche.
- Implementing a new custom Transformers server process lifecycle.
- Removing importable compatibility state dataclasses.
- Preserving dormant legacy tab behavior that has no production route.
- Replacing the native Console with `ChatWindowEnhanced`.
- A line-count, attribute-count, or method-count target.
- A repo-wide raw `pytest` claim over legacy surrogate-app suites.

## Ownership Rules

### Root retention test

A reactive may remain on `TldwCli` only when `TldwCli` itself coordinates the
value's application lifecycle. Being read by more than one destination is not
enough.

The only retained descriptors are:

| Reactive | Why it remains app-owned |
| --- | --- |
| `current_tab` | The navigation coordinator publishes the canonical active route for app-level routing, status, and lifecycle observers. |
| `splash_screen_active` | The app owns splash creation, dismissal, and the transition into the main UI. |

`current_tab` keeps its descriptor, but the unreachable legacy
`watch_current_tab()` tab-window toggling path is removed. Navigation remains
the only production writer of canonical route identity.

### Atomic vertical slices

For a migrated or deleted reactive, one implementation slice must account for:

1. the class descriptor and default;
2. `__init__` assignments;
3. direct readers and writers;
4. `watch_<name>` methods;
5. dynamic `getattr()`/`setattr()` and `reactive_attr` references;
6. handler-map registration;
7. async callbacks and worker completion paths;
8. snapshot save/restore fields;
9. tests and removed-name guards.

The root name is deleted in the same slice. A read-only compatibility property
or mirrored transition period is prohibited.

Destination widgets may use the same ordinary name when it is genuinely
destination-owned. Static checks therefore prohibit the removed name only as
a `TldwCli` descriptor or app-root access; they do not ban the token
repository-wide.

### Async completion

Async destination work captures immutable IDs, immutable value objects, or
narrow copied mappings. It does not capture arbitrary widget trees or use a
broad `deepcopy()` as a concurrency strategy.

Completion follows this order:

1. marshal to the app thread when the worker originated elsewhere;
2. settle durable work, claim state, and resources regardless of whether the
   original view is still visible;
3. apply presentation only when the exact owner is mounted and any
   flow-specific generation or claim is still current;
4. make stale presentation a no-op;
5. never skip cleanup or claim settlement merely because presentation became
   stale.

There is no global generation framework. A destination adds a local
generation token only when its own async flow needs one.

Textual watchers remain synchronous presentation functions. Work expected to
exceed 100 ms is scheduled through the owning destination's worker boundary.

## Exhaustive Reactive Disposition

The table below is the complete 61-descriptor inventory. “Delete” means remove
the root descriptor and all root access; it does not mean delete an already
correct destination-owned field with a similar name.

| # | `TldwCli` reactive | Disposition | Production authority or reason |
| ---: | --- | --- | --- |
| 1 | `current_tab` | Retain | Canonical app navigation lifecycle. Delete only the dead legacy tab-window watcher path. |
| 2 | `ccp_active_view` | Delete | Legacy CCP route is canonical `personas`; `PersonasScreen` owns its mode and center view. |
| 3 | `splash_screen_active` | Retain | App splash lifecycle. |
| 4 | `chat_api_provider_value` | Move, then delete root | Settings owns durable default; active `ConsoleSessionSettings` owns explicit session value; a pending provider command is a typed memory-only handoff. |
| 5 | `ccp_api_provider_value` | Delete | Only legacy CCP provider UI reads/writes it; no production Personas owner needs it. |
| 6 | `rag_expansion_provider_value` | Delete | Belongs to the unmounted legacy Chat sidebar; native Console RAG uses Console-owned settings/context. |
| 7 | `current_editing_character_id` | Delete | `PersonasScreen`/its character handler owns editor identity. |
| 8 | `current_editing_character_data` | Delete | `PersonasScreen`/its character handler owns the loaded record. |
| 9 | `chat_sidebar_collapsed` | Delete | Legacy Chat sidebar is not mounted; native Console rail openness is already screen-owned and snapshot-backed. |
| 10 | `chat_right_sidebar_collapsed` | Delete | Same; native Inspector rail is screen-owned. |
| 11 | `chat_right_sidebar_width` | Delete | Legacy sidebar geometry; not part of the native Console. |
| 12 | `conv_char_sidebar_left_collapsed` | Delete | Legacy CCP shell; Personas owns its rails. |
| 13 | `conv_char_sidebar_right_collapsed` | Delete | Legacy CCP shell; Personas owns its rails. |
| 14 | `evals_sidebar_collapsed` | Delete | The watcher is a no-op and `EvalsScreen` has no such sidebar. |
| 15 | `media_active_view` | Delete | Duplicate of live `MediaWindow` navigation state; app watcher is a no-op. |
| 16 | `current_selected_note_id` | Delete | No production references; Library owns Notes selection. |
| 17 | `current_selected_note_version` | Delete | No production references; Library owns version state. |
| 18 | `current_selected_note_title` | Delete | No production references; Library owns editor content. |
| 19 | `current_selected_note_content` | Delete | No production references; Library owns editor content. |
| 20 | `notes_sort_by` | Delete | No production references; Library owns list ordering. |
| 21 | `notes_sort_ascending` | Delete | No production references; Library owns list ordering. |
| 22 | `notes_preview_mode` | Delete | No production references; Library owns editor/preview presentation. |
| 23 | `notes_auto_save_enabled` | Delete | Only a legacy switch path writes it; Library owns autosave policy. |
| 24 | `notes_auto_save_timer` | Delete | Never scheduled; only stopped by legacy toggle/quit cleanup. |
| 25 | `notes_last_save_time` | Delete | No production references. |
| 26 | `chat_sidebar_selected_prompt_id` | Delete | Legacy Chat sidebar display state. |
| 27 | `chat_sidebar_selected_prompt_system` | Delete | Legacy Chat sidebar prompt body state. |
| 28 | `chat_sidebar_selected_prompt_user` | Delete | Legacy Chat sidebar prompt body state. |
| 29 | `current_chat_is_ephemeral` | Delete | Legacy Chat session flag; native `ConsoleChatSession` owns session persistence state. |
| 30 | `current_chat_conversation_id` | Delete | Legacy Chat route; native Console store/session owns persisted conversation identity. |
| 31 | `current_conv_char_tab_conversation_id` | Delete | Legacy CCP route; Personas owns its current selection/conversation context. |
| 32 | `current_chat_active_character_data` | Delete | Legacy Chat sidebar record; native Console session/context owns explicit identity settings. |
| 33 | `current_ccp_character_details` | Delete | Legacy CCP cache; Personas handler owns loaded character details. |
| 34 | `active_chat_tab_id` | Delete | No production references; native Console store owns active session ID. |
| 35 | `chat_sessions` | Delete | No production references; native `ConsoleChatStore` owns sessions. |
| 36 | `chat_sidebar_loaded_prompt_id` | Delete | No production references. |
| 37 | `chat_sidebar_loaded_prompt_title_text` | Delete | No production references. |
| 38 | `chat_sidebar_loaded_prompt_system_text` | Delete | No production references and a prompt-body privacy liability. |
| 39 | `chat_sidebar_loaded_prompt_user_text` | Delete | No production references and a prompt-body privacy liability. |
| 40 | `chat_sidebar_loaded_prompt_keywords_text` | Delete | No production references. |
| 41 | `chat_sidebar_prompt_display_visible` | Delete | No production references. |
| 42 | `current_prompt_id` | Delete | Legacy CCP prompt editor; prompt management routes to Library. |
| 43 | `current_prompt_uuid` | Delete | Legacy CCP prompt editor; prompt management routes to Library. |
| 44 | `current_prompt_name` | Delete | Legacy CCP prompt editor; prompt management routes to Library. |
| 45 | `current_prompt_author` | Delete | Legacy CCP prompt editor; prompt management routes to Library. |
| 46 | `current_prompt_details` | Delete | Legacy CCP prompt editor; prompt management routes to Library. |
| 47 | `current_prompt_system` | Delete | Legacy CCP prompt editor and root prompt-body cache. |
| 48 | `current_prompt_user` | Delete | Legacy CCP prompt editor and root prompt-body cache. |
| 49 | `current_prompt_keywords_str` | Delete | Legacy CCP prompt editor; prompt management routes to Library. |
| 50 | `current_prompt_version` | Delete | Legacy CCP prompt editor; prompt management routes to Library. |
| 51 | `_initial_media_view_slug` | Delete | Unread root default; `MediaWindow` restores/owns active media type. |
| 52 | `current_media_type_filter_slug` | Delete | Legacy Media UI; live `MediaWindow.active_media_type` owns the filter. |
| 53 | `current_media_type_filter_display_name` | Delete | Legacy Media UI; live navigation panel owns the label. |
| 54 | `media_current_page` | Delete | Legacy Media UI; live list panel owns pagination. |
| 55 | `current_loaded_media_item` | Delete | Legacy Media detail cache; live window/runtime state owns exact selected detail. |
| 56 | `chat_settings_mode` | Delete | Toggle belongs to the unmounted legacy sidebar; native Console has no root basic/advanced mode. |
| 57 | `chat_settings_search_query` | Delete | No production references. |
| 58 | `search_active_sub_tab` | Delete | Old pane switcher; production `SearchScreen` owns actual `TabbedContent.active`. |
| 59 | `ingest_active_view` | Delete | The standalone Ingest route is retired; Library's Import media canvas owns current ingest view state. |
| 60 | `tools_settings_active_view` | Delete | Legacy Tools route now resolves to MCP. |
| 61 | `llm_active_view` | Delete | `LLMManagementWindow.active_view` is the live owner. |

## Domain Contracts

### Provider selection

The root provider reactive is replaced by three explicit paths.

#### Durable global default

Settings continues to persist `chat_defaults.provider` and
`chat_defaults.model`. After a successful save it invalidates/reloads the
configuration through the existing configuration boundary. It does not write
a root runtime default.

A Console session whose settings are still `source="derived"` may refresh
from the new durable default under the existing stale-default rules. A
session with `source="user"` is never overwritten by a Settings save.

#### Active Console selection

`ChatScreen` exposes a narrow production operation that replaces only the
active session's `ConsoleSessionSettings`. It:

- validates the provider identifier against current provider configuration;
- uses the selected provider's configured default model when available;
- clears an incompatible prior-provider model instead of carrying it across;
- preserves unrelated explicit session fields, including a session system
  prompt;
- marks the new settings `source="user"`;
- refreshes Console-owned controls and readiness from that exact settings
  snapshot.

The operation does not write `app.app_config` and does not create an app-level
provider cache.

#### Command invoked away from Console

The command palette's provider switch is a typed, memory-only
`PendingHandoffStore` channel. It preserves ADR-033 semantics:

- one slot;
- last write wins;
- monotonic revision;
- claim, acknowledge, and release;
- no disk persistence;
- no background retry loop.

When Console is active, the command applies through the `ChatScreen`
operation. When another destination is active, it stages the provider intent
for the next Console entry and reports that bounded behavior honestly.

Chat claims the intent after its session store is ready. A valid selection is
applied and acknowledged. An unknown/unsupported provider is terminally
rejected with recovery copy and acknowledged. A transient readiness failure
releases the exact claim for an existing mount or user-triggered retry.
A stale claimant cannot settle a newer provider intent.

“Show current provider” resolves the active Console session when Console is
active; otherwise it reports the persisted Settings default. It never reports
a boot-time cache as current authority.

The handoff payload contains only a normalized provider identifier and
revision metadata. It contains no credential, endpoint, prompt, response, or
model-catalog body.

`resolve_effective_provider_model()` is refactored to accept explicit
configuration/default/session inputs. Production code no longer constructs a
`SimpleNamespace` app surrogate to call it.

### LLM destination actions

The production Lab Models destination splits ownership cleanly:

- the slice first enumerates every visible actionable button ID in the
  `LLMManagementWindow` composition and assigns each one exactly one outcome: a
  destination-local handler or removal;
- `LLMScreen` owns the lifted Lab rail and updates the mounted body's
  `active_view`;
- the action handler calls `event.stop()` before work;
- action IDs are resolved against an allowlisted destination-local map;
- the existing provider-specific functions may remain separate modules, but
  they are invoked from the destination rather than a root app map;
- destination UI queries resolve against `LLMManagementWindow`;
- app-owned server process handles and worker registry remain lifecycle
  dependencies passed explicitly where needed.

Unknown button IDs are ignored. A handler exception produces bounded recovery
copy and leaves the corresponding start/stop controls in a truthful state.
Long-running process work remains in workers; the button handler itself does
not block the Textual loop.

The existing Transformers model-directory, list-local-models, and
download-model controls remain and receive destination routing through their
existing handler functions. The separate custom server-launch block has no
implemented handlers for:

- `transformers-browse-script-button`;
- `transformers-start-server-button`;
- `transformers-stop-server-button`.

That block, including its orphan server-script, interpreter, host, port,
additional-arguments, and operations-log controls, is removed. This tranche
does not silently disable those buttons and does not invent the excluded
custom Transformers server process lifecycle. A future feature would require
its own accepted design before reintroducing the controls.

After live destination routing exists, delete:

- the LLM entries in `_build_handler_map()`;
- `llm_nav_events` root-reactive navigation;
- `TldwCli.llm_active_view`;
- `watch_llm_active_view()`;
- the legacy LLM initializer path that depends on root view state.

### Media destination events

`MediaWindow` remains the sole owner of Media view and selection state.

- `MediaTypeSelectedEvent`, `MediaSearchEvent`,
  `MediaItemSelectedEvent`, and mutation events stop at the destination
  handler when handled.
- `MediaMetadataUpdateEvent` calls `event.stop()` before its first await.
- The app-level `MediaMetadataUpdateEvent` registration is removed.
- Legacy root Media list/search handlers and their app state are removed when
  no registered production widget emits their old IDs.
- `MediaScreen.save_state()` continues reading the actual mounted
  `MediaWindow`; no duplicate screen or app fields are introduced.
- A stale async detail/search completion may update durable/runtime cache
  cleanup as required, but may update visible selection/list state only for
  the same local generation, active media type, search tuple, and selected
  record.

One real event must produce exactly one scoped-service mutation and one
destination refresh.

### Native Console and legacy Chat state

The production Chat route remains the native Console. Root legacy Chat
reactives, prompt-body caches, sidebar state, and worker/widget fields are
removed with their old app handlers.

`ChatScreen.save_state()` stops reading root
`chat_sidebar_collapsed`/`chat_right_sidebar_collapsed`. Native Console rail
state is serialized only from the Console-owned rail fields. Restoring a
native snapshot never writes those root names.

The dormant legacy Chat composition is retired, not rehabilitated:

- delete `ChatScreen._ensure_chat_window()`, its `chat_window` field, and every
  unreachable save/restore, diagnostics, sidebar, provider-control, and button
  delegation branch that depends on it;
- remove the production imports of `ChatWindow` and `ChatWindowEnhanced`;
- delete `Chat_Window.py`, `Chat_Window_Enhanced.py`, and helper modules,
  handlers, widgets, and CSS rules proved by an import/reachability manifest
  to be exclusive to those dormant compositions;
- retain a shared helper/widget module only when the native Console or another
  registered production destination imports it, and remove only its
  root-legacy branch;
- remove tests whose sole subject is the retired composition, while preserving
  direct tests for shared app-independent functions that still have a live
  consumer.

Direct construction or import of the retired Chat compositions is not a
supported compatibility contract. No `LegacyChatState`, adapter, compatibility
property, or second worker/session owner is introduced to keep dead UI alive.
The import/reachability manifest is part of the implementation notes and must
show that no registered production route imports a deleted module before the
deletion lands.

Native Console worker, cancellation, transcript, and session contracts are
not replaced by the legacy streaming bridge.

### Personas and retired prompt/CCP state

The canonical Personas route retains its current screen/handler ownership.
The legacy CCP descriptor set, app event branches gated by `TAB_CCP`, and old
center-pane watchers are removed.

Import completion must not schedule refresh functions that query the retired
CCP widget IDs. If an import is performed by a production destination:

- refresh the mounted production owner through its narrow destination hook;
- otherwise rely on the next fresh screen load;
- do not create a root character/prompt cache merely to refresh another
  destination.

Prompt management remains Library-owned. No prompt body is copied into root
application state.

### Notes, Search, Ingest, Tools, and Evals

These root reactives are pure deletion slices:

- Notes state and timer cleanup defer to Library's existing owner;
- Search state is read from and restored to actual production controls;
- Library owns the Import media canvas reached through the retired `ingest`
  alias;
- legacy Tools state is removed because the route resolves to MCP;
- Evals removes the no-op sidebar state and watcher.

No replacement store or compatibility property is introduced.

### Retired TLDW API worker pipeline

No result-envelope contract is introduced for a worker group that production
cannot schedule. The `api_calls` registry branch, its media-ingest completion
module, the two compatibility exports, and
`app._last_tldw_api_request_context` are deleted together. Structural coverage
rejects their selectors, group name, imports, exports, root access, and dynamic
access.

This does not remove server-backed ingestion. Library owns the production
ingest form, queue, local/server request mapping, result rows, and the public
server batch-cancellation seam. Direct tests retain the pure server request
contract; a normal mounted `TldwCli` proves the legacy `ingest` alias reaches
Library without importing or querying the retired pipeline.

## Non-Reactive Companion Ledger

This ledger covers non-reactive fields and seams adjacent to the root
reactives or exposed by the same ownership trace.

| Field or seam | Disposition in this tranche | Reason |
| --- | --- | --- |
| `button_handler_map` and `_build_handler_map()` | Delete after LLM action registration | Assigned but never read; retains false dynamic ownership references. |
| `_chat_state_lock`, `current_ai_message_widget`, `current_chat_worker`, `current_chat_is_streaming` and accessors | Delete with legacy Chat slice | App-level singleton state belongs to the unmounted streaming path; native Console already owns run state. |
| `current_chat_note_id`, `current_chat_note_version` | Delete with legacy Chat/Notes slice | Legacy sidebar-only state. |
| `current_ccp_character_image` | Delete with Personas/CCP slice | Legacy center-pane cache. |
| `_conv_char_search_timer`, `_ccp_conversation_search_generation` | Delete with Personas/CCP slice when their old handler path is removed | Legacy CCP search lifecycle. |
| `_conversation_search_timer`, `_chat_sidebar_prompt_search_timer`, `_media_sidebar_search_timer` | Delete with legacy Chat slice | Legacy sidebar debounce state. |
| `_notes_search_timer` | Delete with Notes slice if no non-legacy reader remains | Library owns current Notes search. |
| `media_search_current_page`, `media_search_total_pages`, `current_sidebar_media_item` | Delete in TASK-650 with the legacy Chat slice | These fields are exclusive to the dormant Chat media sidebar and its handlers. |
| `_media_search_timers`, `_media_search_generation`, `_initial_media_view` | Delete in TASK-652 with the Media slice | These fields back the duplicate app-root Media search/navigation path; live `MediaWindow` owns current searches and view state. |
| `_initial_search_sub_tab_view`, `_initial_ingest_view`, `_initial_tools_settings_view`, `_initial_llm_view` | Delete with their root navigation slices | Legacy initializer defaults. |
| `_last_tldw_api_request_context`, `api_calls` routing, and `media_ingest_workers.py` | Delete together | Their MediaIngestScreen/producer was retired; retaining or rebuilding consumers creates a false second ingest owner and preserves payload-bearing diagnostics. |
| `TldwCli.query_one()` fallback into the active screen | Explicitly excluded; requires a separate handler/lifecycle decomposition | It is a broad coupling seam, but process managers and legacy non-state handlers still depend on it. Removing it here would expand beyond the verified state slices. New code in this tranche must not add callers. |
| LLM server process handles and server-worker registry | Explicitly retained for this tranche | Service/process lifecycle is excluded by ADR-033; destination event registration may depend on them explicitly without moving their ownership. |
| Duplicate `_wire_writing_services()` and `_wire_chat_conversation_services()` calls | Explicitly excluded under ADR-033 | Verified construction/lifecycle defect, but not a destination reactive or worker-context handoff. It requires its own service-lifecycle design. |
| Database handles, scope services, runtime policy, snapshot store, handoff store, diagnostics monitor | Retain | Existing application composition/lifecycle owners established by prior tasks and ADRs. |

An excluded row is not evidence that the seam is healthy. It records that the
problem was reviewed and that changing it would require a different
architectural contract.

## Snapshot and Privacy Rules

`ScreenStateStore` remains process-memory only.

Only allowlisted stable view primitives and identifiers may enter a screen
snapshot. The following are prohibited:

- domain records or ORM/database rows;
- Textual widgets, workers, timers, services, or database handles;
- credentials, API keys, tokens, headers, or endpoints containing secrets;
- system/user prompt bodies;
- response bodies or generated content not already covered by an approved
  screen-state contract.

Explicit navigation context continues to override restored state.

Removed prompt/character/media root fields are not copied into snapshots as a
substitute. Memory-only values must not be written to disk or persistent logs,
and their `repr`, exception messages, and diagnostics must remain
payload-free.

## Testing Contract

The user's test rule is binding:

> Do not use test apps or simplified versions of the application. Test the
> full production app or test an app-independent function directly.

### Allowed

- normal production `TldwCli()`;
- `app.run_test()` with the actual registered production screen and widgets;
- direct tests of app-independent immutable models, stores, parsers, and pure
  functions;
- narrow service collaborators, callback recorders, and fault injection
  installed on actual production objects;
- static AST checks targeted at production ownership boundaries.

### Prohibited

- `App`, `TldwCli`, or `Screen` test subclasses;
- `SimpleNamespace`, `MagicMock`, or ad hoc objects standing in for the app;
- `object.__new__(TldwCli)`;
- unbound `TldwCli` method calls;
- destination shell substitutes or locally simplified application
  compositions;
- requesting or importing legacy surrogate-app fixtures;
- treating a raw repo-wide `pytest` collection of prohibited legacy tests as
  the authorized gate.

The dedicated production-app suite establishes private `HOME`, XDG, data, and
config roots before importing or constructing `TldwCli`. Async tests carry
explicit markers.

### Behavioral coverage

Each slice adds the narrowest relevant checks:

- the production app navigates to every changed destination without a removed
  root name, watcher, or initialization failure;
- Models navigation routes through the real Lab `LLMScreen` rail, while one
  safe/fault-injected action routes through its real deferred
  `LLMManagementWindow` body exactly once;
- a real Media event on the real Media screen reaches one scoped-service
  mutation and does not reach the legacy app handler;
- Settings save updates durable defaults without overwriting an explicit
  active Console session;
- command-palette provider selection applies to the active real Console or is
  claimed once after a real fresh-screen navigation;
- stale provider claims cannot acknowledge or release a newer selection;
- native Console snapshot restore preserves actual rail/session settings
  without root sidebar fields;
- Personas, Library, Search, MCP, and Evals continue through registered
  production routes, and the retired Ingest alias continues through Library;
- the retired `api_calls` worker group, shared TLDW API request context,
  payload-bearing completion handlers, and compatibility exports are absent
  while Library remains the live ingest owner;
- invalid or stale async completion settles resources/claims without applying
  stale UI.

### Structural coverage

AST checks:

- enumerate the expected `TldwCli` reactive set after each slice;
- prohibit removed descriptors and root `app.<removed_name>` access;
- inspect string `getattr()`/`setattr()` and handler `reactive_attr` values;
- allow the same field name on a destination owner;
- verify `button_handler_map` is absent after its slice;
- verify no TLDW API completion path, `api_calls` group, selector, or
  `_last_tldw_api_request_context` root access remains;
- verify new production-app tests contain none of the prohibited surrogate
  patterns.

Structural checks are not a substitute for production behavior tests.

### Verification gates

Per task:

- focused direct and production-app tests for the changed slice;
- relevant integrated production-app tests;
- Ruff on changed Python files;
- formatter check on changed Python files;
- `compileall` for changed/imported modules;
- `git diff --check`.

At tranche close:

- the authorized integrated suite;
- the existing production-app maturity sentinel;
- build wheel and sdist from committed source;
- install the wheel into a clean environment outside the source checkout;
- run installed-distribution import/resource tests and the applicable
  production-app/direct-function sentinels against the installed wheel;
- confirm no test imports from the source checkout.

Performance assertions are scoped to the changed destination and operation.
The tranche makes no broad startup or recompose performance claim.

## Failure and Recovery

- Removing a root name is atomic with removing its watcher and writers; an
  `AttributeError` after navigation is a failing test, not a compatibility
  fallback trigger.
- Destination UI completion that loses its owner/generation is ignored, while
  durable work and cleanup still settle.
- LLM action failures restore truthful control state and report bounded,
  non-sensitive recovery copy.
- Media events stop propagation before awaiting; failures do not fall through
  to the legacy root handler.
- Provider validation failure is terminal for the exact claimed intent.
  Transient Console readiness releases only that exact claim.
- no retired TLDW API completion can be routed or guess using a global “last
  request”; Library's live queue owns ingestion and cancellation;
- Logging names operations and reason categories, not private payloads.

## Implementation Slices and Dependency Order

Backlog tasks are created only after this written specification passes
independent review and user review. Each slice is one atomic, testable task
with the existing ADRs linked in its plan.

1. **Restore LLM destination action ownership, remove unsupported Transformers
   launch controls, and remove the dead root button dispatcher.**
   Census every production button, register supported LLM actions locally,
   remove the unimplemented custom Transformers server-launch block, preserve
   explicit existing process lifecycle dependencies, remove root/dynamic
   dispatcher infrastructure, and delete `llm_active_view`.

2. **Move provider selection to Settings, Console session state, and one typed
   handoff.**
   Remove the root provider descriptor/watcher, refactor explicit resolution
   inputs, and preserve active/away-from-Console command behavior.

3. **Retire the unreachable legacy Chat composition.**
   Prove its import graph is absent from registered production routes, delete
   the dormant `ChatWindow`/`ChatWindowEnhanced` composition and exclusive
   helpers, and prune legacy-only branches from shared modules without adding
   replacement state.

4. **Remove legacy native-Console-adjacent Chat root state.**
   Remove sidebars, legacy session/prompt/character reactives, app worker/widget
   singleton state, dead restore branches, and root snapshot dependencies
   after the dormant composition no longer consumes them.

5. **Remove legacy CCP/prompt root state and stale import refresh callbacks.**
   Keep Personas and Library as the only production owners.

6. **Remove duplicate Media root state and stop mutation bubbling.**
   Keep `MediaWindow` as the view/selection owner and prove one mutation per
   event.

7. **Remove retired Notes, Search, Ingest, Tools, and Evals root
   state.**
   Delete no-op watchers, defaults, timers, and dead initializer paths while
   preserving their current production owners and route aliases.

8. **Retire the unreachable TLDW API worker context and handlers.**
   Delete the orphaned `api_calls` graph instead of rebuilding a producer,
   while preserving Library's live ingest request/cancellation behavior.

9. **Run the final ownership and installed-distribution sentinels.**
   Freeze the exact retained root-reactive set, run the authorized integrated
   suite, and verify the clean installed wheel.

The implementation plans may split a slice only if the resulting tasks are
independently valuable and do not reference future task IDs. They may not
combine service lifecycle or broad query-boundary removal into these slices.

### Approved task and plan mapping

| Slice | Backlog task | Implementation plan | Dependencies |
| ---: | --- | --- | --- |
| 1 | TASK-647 | `Docs/superpowers/plans/2026-07-26-task-647-llm-destination-actions.md` | None |
| 2 | TASK-648 | `Docs/superpowers/plans/2026-07-26-task-648-provider-selection-ownership.md` | TASK-647 |
| 3 | TASK-649 | `Docs/superpowers/plans/2026-07-26-task-649-retire-legacy-chat-composition.md` | TASK-648 |
| 4 | TASK-650 | `Docs/superpowers/plans/2026-07-26-task-650-remove-legacy-chat-root-state.md` | TASK-648, TASK-649 |
| 5 | TASK-651 | `Docs/superpowers/plans/2026-07-26-task-651-remove-legacy-ccp-prompt-root-state.md` | TASK-647 |
| 6 | TASK-652 | `Docs/superpowers/plans/2026-07-26-task-652-media-destination-state.md` | TASK-647 |
| 7 | TASK-904 | `Docs/superpowers/plans/2026-07-26-task-904-retired-destination-root-state.md` | TASK-647 |
| 8 | TASK-905 | `Docs/superpowers/plans/2026-07-26-task-905-retire-tldw-api-worker-pipeline.md` | TASK-647 |
| 9 | TASK-906 | `Docs/superpowers/plans/2026-07-26-task-906-reactive-ownership-closeout.md` | TASK-647–652, TASK-904, TASK-905 |

## ADR Check

ADR required: no

ADR path:
`backlog/decisions/033-application-session-state-ownership.md`,
`backlog/decisions/006-provider-aware-generation-settings.md`,
`backlog/decisions/011-chatbook-workbench-ui-system.md`,
`backlog/decisions/029-local-private-data-boundary.md`, and
`backlog/decisions/032-immutable-installed-distribution-assets.md`

Reason: This specification directly implements the accepted decisions that
destination screens own domain/view state, Settings owns persisted generation
defaults, Console owns active session settings, and `TldwCli` coordinates
narrow owners without a root state object. ADR-029 supplies the payload-safe
diagnostic boundary and ADR-032 supplies the installed-artifact gate. This
specification does not change those decisions. Accepted ADRs remain immutable.

## Acceptance Boundary

The tranche is complete only when:

- `TldwCli` has exactly the approved app-owned reactive set plus any unrelated
  descriptors introduced by separately approved work; none of the 59
  remove/move names remain root descriptors or root accesses;
- Settings, Console, Media, Personas, Library, Search, MCP, Evals, and LLM pass
  their production-route checks, including the retired Ingest alias to Library;
- LLM production actions are live from their destination;
- one Media mutation event cannot apply twice;
- provider commands have one explicit durable/session/handoff interpretation;
- the retired TLDW API worker graph and shared request context are absent,
  while Library remains the only production ingest owner;
- snapshots and diagnostics remain payload-safe;
- source, authorized integration, formatting/static, and installed-wheel gates
  pass without surrogate applications.

## Implementation Closeout

TASK-647–652 and TASK-904–905 are Done with checked acceptance criteria and
implementation evidence. TASK-906 completes slice 9 on latest `dev` commit
`6784c4ba3`:

- the source and installed-artifact AST sentinels enforce the exact retained
  set `current_tab` and `splash_screen_active`, including transitive local
  root mixins, all 59 retired names, and their root watchers;
- the normal production `TldwCli` exercises every reviewed route twice with
  fresh registered screens and payload-safe memory-only snapshots;
- the installed-wheel probe runs outside both the checkout and copied build
  input, audits every loaded package module, preserves installed hashes, and
  exercises the registered Home and Chat screens;
- the focused ownership/production-app gate passed 57 tests with 2 warnings
  in 281.97 seconds, the installed-distribution gate passed 6 tests in 21.83
  seconds, and the authorized integrated gate passed 196 tests with 5
  warnings in 567.81 seconds;
- compileall, scoped Ruff lint, the zero-F841 Settings baseline, the 37-file
  format gate, and `git diff --check` passed.

`Tests/UI` remains intentionally excluded because its conftest imports legacy
surrogate app/widget harnesses. No repository-wide pytest result is claimed.
