# ADR-250: Application services bundle

Status: Proposed
Date: 2026-10-27
Related tasks: [TASK-36947.1](<../tasks/task-36947.1 - Record-the-application-services-bundle-decision-as-an-ADR-ADR-036-follow-up.md>) (this record), [TASK-36947.2](<../tasks/task-36947.2 - Application-services-are-built-once-into-a-typed-bundle.md>) (implements it), TASK-36947.3 to TASK-36947.6 (first consumers)
Amends: [ADR-036](036-application-service-composition-lifecycle.md), its deferral sentence only ("A future extraction may use a typed immutable `ApplicationServices` bundle only after ...").
Source: owner decision 2026-10-25; architecture review 2026-10-25, [report](../../Docs/QA/architecture-review-2026-10-25/report.md) §F1 and §F9, and its [verification ledger](../../Docs/QA/architecture-review-2026-10-25/verification.md).

This is a successor rather than an edit. The rest of ADR-036 is unchanged: Writing and Chat are composed once, `_wire_*` helpers are construction helpers, and there is no container. That decision keeps its accepted record.

## Context

ADR-036 deferred the bundle until there was more than one real composition root. There are now five, and each opens the same profile databases in its own way. Line numbers are at `dev` 2b296e60b3.

| Root | Where it builds services |
| --- | --- |
| TUI | `TldwCli.__init__` (`app.py:1054`) and `app_service_wiring.py`. |
| Standalone MCP server | `TldwMCPServer._init_databases` opens ChaChaNotes, Media and a `NotesInteropService` itself (`MCP/server.py:590-657`). |
| Mobile web service | `MobileChatService._build_engine` opens its own isolated chat, workspace and runs databases (`Web_Server/mobile_service.py:413-468`, ADR-234). It also builds `HeadlessAppContext` (`:204-247`), a stand-in for the app, so that the Console controller can read services from it. `_ensure_reading_dbs` opens the profile databases (`:799-841`). |
| CLI tools | The RAG backfill opens its own databases (`RAG_Search/backfill.py:115-121`). `doctor` runs inside the TUI (`/doctor`, `UI/Screens/chat_screen.py:18437`) but gets its database from config (`Utils/doctor.py:141-146`). |
| config's lazy singletons | Defined at `config.py:11795-11797`, with getters at `:11873`, `:11908` and `:11937`. The TUI reads them (`app.py:750`, `:2085`), and so do the in-app MCP delegate (`MCP/local_runtime_delegate.py:671-681`) and six more modules (§F9). |

The roots do not open a database the same way:
- **Migration seed.** Config, the MCP server and the backfill pass the Console Library migration seed. The mobile reading path does not (`mobile_service.py:832`). If a profile database that has not been upgraded is first opened there, it migrates with automatic retrieval off (`DB/ChaChaNotes_DB.py:6742-6752`).
- **Built-in content.** Only config's getter and `app.py:750` seed it (`config.py:11800`). The MCP server, mobile and backfill paths never do.

Inside the TUI, consumers get services from the app. `TldwCli` holds 397 attributes, about 199 of them services. Code reads them through 222 string lookups outside `UI/`, `Widgets/` and `css/`, and through 1,176 inside `UI/` (§F1).

## Decision

### 1. Shape

A new module, `tldw_chatbook/app_services.py`, holds:
- the bundle types;
- `assemble_application_services(app)`, which builds the bundle from the app;
- the shared factory.

At module level it imports only the standard library, with every service type under `TYPE_CHECKING`. Database classes are imported inside the openers.

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class ApplicationServices:
    core: CoreServices
    storage: ProfileStorage
    chat: ChatServices
    notes: NotesServices
    media: ScopedServices[LocalMediaReadingService, ServerMediaReadingService, MediaReadingScopeService]
    workspace: WorkspaceServices
    mcp: McpServices
    sync: SyncServices
    watchlists: WatchlistsServices
    # ...one field per remaining feature (section 2)

@dataclass(frozen=True, slots=True, kw_only=True)
class ScopedServices[L, S, C]:
    """A feature with a server service and a scope service, and optionally a local one."""
    local: L | None = None
    server: S
    scope: C
```

- **Slices.** Every slice is a frozen, slotted, keyword-only dataclass with typed fields. A feature with more members than `ScopedServices` holds gets its own class.
- **Optional fields.** A field is `T | None` with default `None` if it can be unavailable at runtime today, for example a database that failed to open or a local backend that failed to build. Other fields have no default. A test builds only the slice its subject receives.
- **What the bundle holds.** Only objects that exist when composition ends and are never reassigned after it. Everything else is excluded by name (section 2, rules 4 and 5).
- **No app references (end state; pending Q1).** A new member never receives the app or a closure over it. It receives a slice or a narrow callable. Members that capture the app today stay as they are in TASK-36947.2, and each consumer migration removes its own. Those members are:
  - Study, Quiz and Research, through `notification_app=self` (`app_service_wiring.py:2404`, `:2413`, `:2463`), and `media.local` and `watchlists.local`, through `notification_app = self` (`:3294`, `:3298`);
  - `LibraryLocalRagSearchService(self)` (`:2448`);
  - the `lambda: self` getters at `:2521`, `:2682`, `:2724` and `:2742`;
  - the provider's credential-store lambda (`:1390`).

### 2. Which attributes go where

Here, a service is an app attribute whose name ends in `_service`, `_db`, `_provider`, `_repository`, `_store`, `_owner`, `_coordinator`, `_controller`, `_manager`, `_runtime`, `_registry` or `_enforcer`. There are 168 at base. A method defined on the class, such as `get_dreams_db`, is not a service. A lazy property is a service, and rule 5 excludes it.

| Slice | Current app attributes, and the field each becomes |
| --- | --- |
| `core` | `server_context_provider`, `service_policy_enforcer` and `unified_mcp_target_store`. These keep their names. |
| `storage` | `chachanotes_db` → `chachanotes`, `media_db` → `media`, `prompts_db` → `prompts` and `local_workspace_db` → `workspaces`. `agent_runs` is filled only by the mobile engine; in the TUI it is `None` (rule 5). |
| `chat` | The chat-conversation trio → `conversation`; the chat-dictionary and chat-grammars trios; `conversation_local_marks_service`; and `citation_trace_repository`, `citation_artifact_ownership_coordinator` and `citation_legacy_migration_service`. |
| `notes` | `notes_service` → `local`, `server_notes_workspace_service` → `server`, `notes_scope_service` → `scope`, and `file_notes_session_owner`. |
| `quiz` | `local_quiz_service` → `local`, `server_quiz_service` → `server`, `study_quiz_scope_service` → `scope`. |
| `chatbook` | `local_chatbook_service` → `local`, `server_chatbook_service` → `server`, `prompt_chatbook_scope_service` → `scope`. |
| `watchlists` | `local_watchlists_service` → `local`, `server_watchlists_service` → `server`, `watchlist_scope_service` → `scope`; `subscriptions_db` and `watchlist_bundle_service`. |
| `runtime` | `server_runtime_service` → `server`, `server_runtime_scope_service` → `scope`, and `active_server_capability_service`. |
| `workspace` | `workspace_registry_service` and `change_review_consent_service`. |
| `mcp` | `unified_mcp_service`, `server_unified_mcp_service`, `local_mcp_control_service`, `local_mcp_store` and `unified_mcp_context_store`. The MCP-governance pair becomes `governance`. |
| `sync` | `server_sync_service`, `sync_scope_service`, `local_first_sync_service`, `manual_sync_control_service`, `sync_state_repository` and `sync_restore_service`. |

Rules for the rest. Rules 4 and 5 take precedence over rules 1 to 3.

1. **Scoped features.** Where both `server_<f>_service` and `<f>_scope_service` exist, with or without `local_<f>_service`, the feature is `services.<f>`, a `ScopedServices`. At base there are 15 trios and 20 pairs. For example, `server_claims_service` becomes `services.claims.server`, and `local` is `None`. Names that do not line up are mapped in the table.
2. **Other members.** Any other member of feature `<f>` goes into that feature's slice, with the feature prefix dropped from its name.
3. **Storage.** `storage` holds the three databases that config keeps as singletons (ChaChaNotes, Media and Prompts) and the two that the mobile engine also opens (workspaces and agent runs). Every other database stays in its feature's slice. A database that a second root starts opening moves into `storage` when that root migrates. The one case today is the MCP server's read-only `SubscriptionsDB` (`MCP/local_server_tools.py:116`).
4. **Never in the bundle:**
   - the app's own runtime objects, which are built around the app: `ConsoleRuntime(self)` (`app.py:1196`), `DBStatusManager(self)` (`:1296`), `WorkerHandlerRegistry(self)` (`:2215`) and `raw_cli_runtime` (`:1154`);
   - configuration that is reloaded at runtime: `app_config` (`:1153`, replaced at `:2038`) and the runtime policy;
   - UI and session state: reactives, flags, timers, worker and process handles, and the ADR-033 session owners such as `screen_state_store`;
   - objects that feature modules attach to the app after boot, such as the one at `RAG_Search/semantic_availability.py:242`.
5. **Not in the bundle in TASK-36947.2**, because they do not exist yet when it is published, are replaced afterwards, or are pinned in place:
   - **Built on first access,** which covers every lazy property and every first-use getter method:
     - `server_credential_store` and its unavailable reason;
     - `notes_sync_runtime_owner`, `terminal_session_manager`, `persona_buddy_controller`, and the Kanban and RAG-admin trios;
     - `recovery_service` (`app_lifecycle.py:123`), `meeting_session_owner` (`:609`), `daily_report_demo_service` (`app_service_wiring.py:2508`) and `llamacpp_snapshot_service` (`:3638`);
     - `artifact_share_controller`, which `_get_artifact_share_controller()` sets on first use and shutdown clears (`app_lifecycle.py:975`, `:995`).
   - **Composed in `on_mount` or later:**
     - `watchlists_operation_coordinator` and `watchlists_command_service` (`app.py:3307-3312`);
     - tool packs (`app_service_wiring.py:1868-1999`);
     - workspace agent provisioning (`:1800`).
   - **Pinned in place:** the lazily built skills stack (`local_skill_trust_service`, `local_skills_service` and `skills_scope_service`) and the eagerly built `server_skills_service`. The Console's off-loop trust proof checks those exact app slots, the mixin properties and the trust factory's closure over the app (`app_skill_wiring.py:144-496`). Bundling them, or narrowing that closure, would silently send trust preparation back onto the event loop.
   - **Opened on first use:** `dreams_db` (`app_service_wiring.py:3322`), `guardian_db` (`:3353`) and the TUI's agent-runs database (`Chat/console_runtime.py:4629`).
   - **Reset and rebuilt:**
     - Collections capture: `collections_capture_scope_service`, `server_collections_capture_service`, `local_collections_capture_service`, `collections_capture_repository`, `collections_offline_store` and `collections_legacy_recovery_service` (built at `app_collections_wiring.py:411-534`, reset at `:795-804`, activated at `:916`);
     - the Notes-sync wiring, with `notes_organization_repository` and `notes_organization_sync_service`. It is wired at `app_service_wiring.py:476` and `:645`, and `app.py:2049` re-runs it on every source switch.

An app-like receiver is any expression whose last name segment is `app`, `app_instance`, `tldw_app`, `_app` or `_app_instance`, the definition TASK-36942.4 uses. That covers `self._app` in ConsoleRuntime, for example.

TASK-36947.2 adds a coverage test. It collects every service name that production code reads from an app-like receiver, by attribute or by literal `getattr`/`hasattr`. Each such name must be either a bundle field, reached through its alias, or on the rule 4 and rule 5 exclusion list. The exclusion list shrinks as consumers migrate. It may grow only by a new service that is built after composition (pending Q3).

### 3. Construction in the TUI

- **Composition does not change.** Every helper constructs and assigns exactly as it does today, and no database open moves.
- **One publish point.** `__init__` calls `ServiceWiringMixin._publish_application_services()` in `app_service_wiring.py` exactly once. The call comes after `_wire_character_persona_services()` (`app.py:1637`) and before `_init_worker_handlers()` (`:1650`).
- **What publishing does.** The method sets `self.services` from `app_services.assemble_application_services(self)`, which reads the attributes that composition has just assigned. By then every bundled object exists. That includes Prompts (`:1519`) and ChaChaNotes (`:1541`), which are assigned after the first `_wire_*` calls, and the citation repository, which is assigned twice during composition (`app_service_wiring.py:2127`, `:2337`).
- **Size ratchets.** The assembly code lives in `app_services.py`, so the two modules pinned by `Tests/Architecture/test_module_size_ratchet.py` grow only by the call and the method. Their rows are raised to the measured value under the owner's 2026-10-04 ruling.
- **Import cost.** The method imports `app_services` inside its body, so the `import tldw_chatbook.app` closure does not change. The UI-ready census gains one module. Under ADR-097 that must fit the existing headroom, or TASK-36947.2 removes one module elsewhere.

### 4. Aliases, and when they end

- **The app attributes are the aliases.** They stay where composition assigns them, so readers, tests that assign an attribute, `object.__new__(TldwCli)` fakes and teardown all keep working unchanged.
- **One identity test.** TASK-36947.2 pins the alias list in one test. After construction, the test asserts `app.<alias> is <bundle field>` for every entry.
- **Shrink only.** The list only shrinks. A service added later goes into the bundle with no alias.
- **Removal.** An alias is deleted by the PR that migrates its last reader.
  - That PR proves no reader is left: a repo-wide scan finds no attribute read, and no literal `getattr` or `hasattr`, of that name on an app-like receiver.
  - Composition then passes the object to `_publish_application_services()` instead of assigning it to the app.
  - The owner's teardown moves to the bundle in the same PR.
- **End.** The alias period ends when the list is empty. There is no date.

### 5. Teardown

- **Who.** Whoever builds a bundle or a `ProfileStorage` tears it down. `ApplicationServices` has no `close()`.
- **The TUI** keeps its unmount sequence in `on_unmount` (`app_lifecycle.py:997`). That sequence stops workers and pools before it closes stores. It then closes the provider's cached client (`:1297`), the ingest job store (`:1480`), and the writing and research services (`:1483`, `:1486`). The order is deliberate, and it is not the reverse of construction order. During the alias period it keeps reading today's attributes and slots, and it never builds a lazy service just to close it (`app_lifecycle.py:375-376`).
- **`ProfileStorage.close()`** closes the calling thread's connection on each database that was opened. It logs a failure by type name and never raises. The TUI does not call it and leaves storage to process exit, as it does today. The other roots call it (section 6).

### 6. Shared factory for the other entry points

```python
def open_profile_storage(
    *,
    client_id: str,
    chachanotes: Path | None = None,
    media: Path | None = None,
    prompts: Path | None = None,
    workspaces: Path | None = None,
    agent_runs: Path | None = None,
    seed_builtin_content: bool = False,
    check_integrity: bool = False,
) -> ProfileStorage: ...
```

- **What it opens.** Each database whose path is given; the others stay `None`. If one fails, it closes the ones it already opened and raises.
- **Same options for every root.** The Console Library migration seed is always passed, and `check_integrity` is used as given. Built-in content is seeded only when `seed_builtin_content` is set, and only the TUI sets it.
- **Openers.** One opener per database does the construction: `open_chachanotes_db(path, *, client_id, ...)` and the rest.
  - The factory and the openers are added by the first PR that calls them.
  - The TUI's own opens (`app.py:2085`, `:2144`) switch to the openers in TASK-36947.6.
  - Paths still come from config's path helpers.
- **Scope.** For the roots in the table below, the openers are the only code that constructs these databases. About 30 other ad hoc constructions inside the app stay as they are (pending Q2). Examples are the Chatbooks importer and creator, `Tools/workspace_file_roots.py:621` and `Workspaces/change_retention.py:224`.

Who moves each root is pending Q5.

| Entry point | How it gets its services |
| --- | --- |
| Standalone MCP server | `open_profile_storage(client_id=CLI_APP_CLIENT_ID, chachanotes=get_chachanotes_db_path(), media=get_media_db_path())`. It builds `NotesInteropService` and `MCPTools` from that storage, and `main()` (`MCP/server.py:1167`) closes it. |
| Mobile engine | `open_profile_storage(client_id="mobile-service", chachanotes=data_dir / "chat.db", workspaces=data_dir / "workspaces.db", agent_runs=data_dir / "runs.db")`. These are the same isolated files (ADR-234), opened with the shared options. Shutdown keeps today's order (`mobile_service.py:1895-1916`): stop the controller, run `quiesce_connections(timeout_seconds=5)` on the chat database, then call `storage.close()`. Once TASK-36947.4 gives the controller slices, `HeadlessAppContext` is no longer needed. |
| Mobile reading | `open_profile_storage(client_id="mobile-service", chachanotes=..., media=...)` on first read, as today. `shutdown()` (`mobile_service.py:1848`) calls `close()`. |
| RAG backfill CLI | `open_profile_storage` with only the paths its item types need, closed in a `finally`. |
| `doctor` | `run_doctor` gains an `integrity_fn` parameter and passes it on to `check_database_integrity` (`Utils/doctor.py:88`). `/doctor` supplies a callable over `services.storage.chachanotes` that raises when that database is `None`, as `_default_integrity_fn` does (`:141-146`). This is part of TASK-36947.6. |
| In-app MCP delegate and the other modules that use config's getters | They receive their slice through their constructors, for example `LocalMCPControlService` (`app_service_wiring.py:3058`). This is part of TASK-36947.6. |

### 7. Consumers

- **Non-UI code** receives its slices as constructor or function parameters and never receives the app.
- **UI capabilities** are not services. These include `notify`, `call_from_thread`, `query_one` and the screen stack. A consumer that needs one receives that one callable.
- **Screens** read `self.app_instance.services.<slice>`, which is typed through `BaseAppScreen` (`UI/Navigation/base_app_screen.py:101`), and pass slices to whatever they build (pending Q4).
- **First migrations:**
  - RAG (TASK-36947.3) receives `storage`, `workspace` and `chat.citation_trace_repository`.
  - The Console controller (TASK-36947.4) receives its slices, plus `notify` and `call_from_thread`.
  - The modules that use config's getters move in TASK-36947.6.

### 8. Out of scope

- **No lookup by key.** There is no DI container, no service locator, no registry keyed by name or type, and no `get_service(...)`. The bundle is never mutated after it is published.
- **No lifecycle framework.** There is no generic lifecycle manager or teardown protocol, and services are not replaced while the app runs. Storage handles stay fixed until restart (ADR-004).
- **Backup/Recovery.** Its participants are unchanged. F10 owns them (TASK-36948.x).
- **The `prompts_interop` singleton** (`Prompt_Management/Prompts_Interop.py:127`) stays. `LocalPromptService` takes that module itself (`app_service_wiring.py:2289`).

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep ADR-036's deferral | The condition it waited for has been met. `HeadlessAppContext` shows the cost: a second root fakes the app in order to reuse the controller. |
| DI container or mutable registry | The owner rejected it. It turns attribute access back into lookup by key. |
| One flat bundle with ~199 fields | Every consumer would still receive everything, which is the app again. |
| Helpers return their slices, or construction moves into slice builders | Either composition is reordered (Prompts and ChaChaNotes are assigned after the first `_wire_*` calls), or the earlier slices get no storage. Assembling the bundle once from today's attributes gives the same identities without reordering. |
| One `Lazy` cell type for members built on first access | Today's lazy members behave differently from each other. Some setters replace the value after the first read (`app_service_wiring.py:982`, `:1364`, `:1460`). The Buddy getter is conditional, the trust builder is async, and the skills proof checks the exact slots. Each member gets its own cell when its consumer migrates. |
| A `close()` on the bundle that runs in reverse construction order | Today's unmount order is deliberate (workers before stores) and is not the reverse of construction order. |

## Consequences

- **Typed dependencies.** Dependencies become typed and visible. Tests build one slice instead of faking the app; today 162 test files fake an `app_instance` (§F1).
- **Shared options.** Every root that uses the factory opens a profile database with the same options. The mobile reading path gets the migration seed it is missing.
- **Two ways to reach a service.** During the alias period a service is reachable through the app and through the bundle, and the identity test keeps both pointing at the same object. However, a test that assigns an alias does not reach a consumer that has already migrated.

## Open questions for the owner

1. **Q1, app references.** Should TASK-36947.2 leave today's app-capturing members unchanged, so that "no member receives the app" binds new members and each consumer migration (section 1)? Answer: yes or no. Recommended: yes.
2. **Q2, in-app ad hoc databases.** Should the roughly 30 ad hoc profile-database constructions inside the app stay out of this stream, with a census holding their count instead (section 6)? Answer: yes or no. Recommended: yes.
3. **Q3, new late services.** May the exclusion list grow when a new service is built after composition (section 2)? Answer: yes or no. Recommended: yes.
4. **Q4, screens.** Should screens read `app_instance.services`, rather than receive slices through constructors that the navigation registry would have to supply per screen (section 7)? Answer: yes or no. Recommended: yes.
5. **Q5, roots outside the TUI.** Should one follow-up task, rather than TASK-36947.2, move the MCP server, the mobile service and the backfill CLI onto the factory (section 6)? Answer: yes or no. Recommended: yes.

## Links

- [ADR-036: Application Service Composition Lifecycle](036-application-service-composition-lifecycle.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
- [ADR-097: Boot budget ratchets](097-boot-budget-ratchets.md)
- [ADR-234: Served mobile companion surface](234-served-mobile-companion-surface.md)
- [ADR-004: Settings storage defaults restart boundary](004-settings-storage-defaults-restart-boundary.md)
- [Architecture review 2026-10-25](../../Docs/QA/architecture-review-2026-10-25/report.md)
