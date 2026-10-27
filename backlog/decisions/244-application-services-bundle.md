# ADR-244: Application services bundle

Status: Proposed
Date: 2026-10-27
Related tasks: [TASK-36947.1](<../tasks/task-36947.1 - Record-the-application-services-bundle-decision-as-an-ADR-ADR-036-follow-up.md>) (this record), [TASK-36947.2](<../tasks/task-36947.2 - Application-services-are-built-once-into-a-typed-bundle.md>) (implements it), TASK-36947.3 to TASK-36947.6 (first consumers)
Amends: [ADR-036](036-application-service-composition-lifecycle.md), and only its deferral sentence ("A future extraction may use a typed immutable `ApplicationServices` bundle only after ..."). The rest of ADR-036 still holds: each graph is composed once, `_wire_*` helpers are construction helpers, and there is no container.
Decision owner: the owner chose to adopt the bundle on 2026-10-25. This record of that decision is waiting for the owner to accept it.
Source: architecture review 2026-10-25, [report](../../Docs/QA/architecture-review-2026-10-25/report.md) §F1 and §F9, and its [verification ledger](../../Docs/QA/architecture-review-2026-10-25/verification.md).

This is a successor ADR, not an edit of ADR-036: ADR-036's accepted single-pass and Sync decisions are unchanged, and only its deferral clause is replaced. The new design gets its own reviewable record, and the accepted one is not rewritten.

## Context

ADR-036 deferred the bundle until there was more than one real composition root. There are now five, and each opens the same profile databases in its own way. Line numbers are at `dev` 2b296e60b3.

| Root | Where it builds services |
| --- | --- |
| TUI | `TldwCli.__init__` (`app.py:1054`) and `app_service_wiring.py`. |
| Standalone MCP server | `TldwMCPServer._init_databases` opens ChaChaNotes, Media and a `NotesInteropService` itself (`MCP/server.py:590-657`). |
| Mobile web service | `MobileChatService._build_engine` opens its own isolated chat, workspace and runs databases (`Web_Server/mobile_service.py:413-468`, kept separate by ADR-234). It also builds a `HeadlessAppContext` (`:204-247`), a stand-in for the app, so that the Console controller can read services from it. `_ensure_reading_dbs` opens the profile databases (`:799-841`). |
| CLI tools | The RAG backfill opens its own databases (`RAG_Search/backfill.py:115-121`). `doctor` runs inside the TUI (`/doctor`, `UI/Screens/chat_screen.py:18437`) but gets its database from config (`Utils/doctor.py:141-146`). |
| config's lazy singletons | Defined at `config.py:11795-11797`, with getters at `:11873`, `:11908` and `:11937`. The TUI reads them itself (`app.py:750`, `:2085`), and so do the in-app MCP delegate (`MCP/local_runtime_delegate.py:671-681`) and six more modules (§F9). Prompts has a second module singleton (`Prompt_Management/Prompts_Interop.py:127`). |

The roots do not open a database the same way:
- Config, the MCP server and the backfill pass the Console Library migration seed. The mobile reading path does not (`mobile_service.py:832`). If a profile database that has not been upgraded is first opened there, it migrates with automatic retrieval off (`DB/ChaChaNotes_DB.py:6742-6752`).
- Built-in content is seeded only by config's getter and by `app.py:750` (`config.py:11800`). The MCP server, mobile and backfill paths never seed it.

Inside the TUI, consumers get services from the app. `TldwCli` holds 397 attributes, about 199 of them services. Code reads them through 222 string lookups outside `UI/` and 1,176 inside it (§F1).

## Decision

### 1. Shape

A new module, `tldw_chatbook/app_services.py`, holds the bundle types, `Lazy` and the shared factory.
- At module level it imports only the standard library, with every service type under `TYPE_CHECKING`.
- The database classes are imported inside the opener functions, so the boot import budget (ADR-097) does not change.

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
    skills: SkillsServices
    sync: SyncServices
    watchlists: WatchlistsServices
    # ...one field per remaining feature, by the rules in section 2

@dataclass(frozen=True, slots=True, kw_only=True)
class ScopedServices[L, S, C]:
    """A feature that is exactly a local/server/scope trio."""
    local: L | None
    server: S
    scope: C

class Lazy[T]:
    """A member built on first use, as its app property is today."""
    def get(self) -> T: ...                   # builds at most once, under a lock
    def peek(self) -> T | None: ...           # returns the value if built; never builds
    def provide(self, value: T) -> None: ...  # allowed only before the first get()
```

- **Slices:** every slice is a frozen, slotted, keyword-only dataclass with typed fields. A feature that is exactly a trio uses `ScopedServices`. A feature with more members gets its own class.
- **Optional fields:** a field that can be unavailable at runtime today is `T | None` with default `None`. Examples are a database that failed to open and a local backend that failed to build. Other fields have no default. A test builds only the slice its subject receives.
- **Lazy members:** `Lazy` keeps the boot-time savings of today's lazy properties. These are the credential store, the skills stack, the Notes-sync runtime owner, the terminal manager, Persona Buddy, and the Kanban and RAG-admin trios. As today, a failed build is not cached, and neither is a build that returns `None`. `Lazy` is a typed cell that holds one value, not a registry.
- **No app references:** no member receives the app object. A member may receive a narrow callable that reads one live value. For example, the provider's credential-store factory is `lambda: self.server_credential_store` today (`app_service_wiring.py:1390`); it becomes `credential_store.get`.

### 2. Which attributes go where

| Slice | Current app attributes, and the field each becomes |
| --- | --- |
| `core` | `server_context_provider`, `service_policy_enforcer` and `unified_mcp_target_store`; `server_credential_store` as a `Lazy`. These keep their names. |
| `storage` | `chachanotes_db` → `chachanotes`, `media_db` → `media`, `prompts_db` → `prompts`, `local_workspace_db` → `workspaces`. `agent_runs` is filled by the mobile engine; the TUI fills it when ConsoleRuntime stops opening its own (`Chat/console_runtime.py:4629`). |
| `chat` | The chat-conversation trio → `conversation` (a `ScopedServices`); the chat-dictionary and chat-grammars trios; `conversation_local_marks_service`; `citation_trace_repository`, `citation_artifact_ownership_coordinator` and `citation_legacy_migration_service`. |
| `notes` | `notes_service` → `local`, `server_notes_workspace_service` → `server`, `notes_scope_service` → `scope`; `file_notes_session_owner`; `notes_sync_runtime_owner` as a `Lazy`. |
| `media` | The media-reading trio. |
| `workspace` | `workspace_registry_service` and `change_review_consent_service`. |
| `mcp` | `unified_mcp_service`, `local_mcp_control_service`, `local_mcp_store`, `unified_mcp_context_store` and the MCP-governance trio. |
| `skills` | `local_skill_trust_service`, `local_skills_service` and `skills_scope_service`, each as a `Lazy`; `server_skills_service`. |
| `sync` | `server_sync_service`, `sync_scope_service`, `local_first_sync_service`, `manual_sync_control_service`, `sync_state_repository` and `sync_restore_service`. |
| `watchlists` | The watchlists trio, `subscriptions_db` and `watchlist_bundle_service`. |

Rules for the rest:

1. `local_<f>_service`, `server_<f>_service` and `<f>_scope_service` become `services.<f>.local`, `.server` and `.scope`. For example, `server_claims_service` becomes `services.claims.server`.
2. Any other member of feature `<f>` goes into that feature's slice, with the feature prefix dropped from its name. This covers repositories, stores, coordinators and owners.
3. `storage` holds the five databases that more than one root opens today. Every other database stays in its feature's slice. When a second root starts opening a feature's database, that database moves into `storage` as that root migrates. The one case today is the MCP server's read-only `SubscriptionsDB` (`MCP/local_server_tools.py:116`).
4. These stay on the app, outside the bundle:
   - Objects constructed with the app: `ConsoleRuntime(self)` (`app.py:1196`), `DBStatusManager(self)` (`:1296`) and `WorkerHandlerRegistry(self)` (`:2215`).
   - Configuration that is reloaded at runtime: `app_config` (`:1153`, replaced at `:2038`) and the runtime policy.
   - UI and session state: reactives, flags, timers, worker and process handles, and the ADR-033 session owners such as `screen_state_store`.
   - Objects that feature modules attach to the app after boot, for example `RAG_Search/semantic_availability.py:242`. TASK-36942.4's census tracks these.
5. Services that are composed after first paint, or reset and rebuilt, also stay on the app for now:
   - tool packs (`app_service_wiring.py:1868-1999`);
   - Collections capture (`app_collections_wiring.py:411-534`);
   - the deferred Notes-sync wiring and `notes_organization_repository` (`app_service_wiring.py:476`, `:645`, `:2001`);
   - workspace agent provisioning (`:1800`).

   See open question 3.

TASK-36947.2 adds a coverage test. It fails when a service that an existing consumer looks up is neither in the bundle nor on the rule 4 and rule 5 exclusion list. The exclusion list can only shrink.

### 3. Construction order in the TUI

`TldwCli.__init__` keeps today's order. Each `_wire_*` or `_init_*` helper returns the slice or slices it builds, and `__init__` assigns the aliases from them. The order is:

1. `core`, from `_wire_server_context_provider()` (`app.py:1305`).
2. `storage`, once the parallel init phase finishes (`app.py:1359-1400`).
   - Media is opened by the section 6 opener instead of the app's own `MediaDatabase(...)` (`:2144`).
   - ChaChaNotes stays the handle that config's getter creates (`:2085`), and Prompts stays the `prompts_interop` instance (`:2119`, `Prompt_Management/Prompts_Interop.py:127`). Other modules still reach both through those globals; the note tools, for example, read the bare global (`Tools/note_management_tools.py:69`). Opening a second handle would split them. TASK-36947.6 switches ChaChaNotes to its opener when it deletes config's getters.
   - Each task keeps its own try/except, so a database that fails to open is still `None` and the others still open.
   - The TUI leaves `agent_runs` as `None` until ConsoleRuntime stops opening its own.
3. The feature slices, in today's `_wire_*` order (`app.py:1447-1466`, `:1561`, `:1634-1637`).
4. `self.services = ApplicationServices(...)`, exactly once. It runs after `_wire_character_persona_services()` (`app.py:1637`) and before `_init_worker_handlers()` (`:1650`).

Nothing reads `self.services` during composition; the helpers use the slices already returned. ADR-036 already forbids re-running a `_wire_*` helper after construction.

### 4. Aliases, and when they end

- **Same name, same object:** every app attribute whose object moves into the bundle stays as an alias, under the same name, for the same object.
- **Two forms:** for an eagerly built member, the alias is a plain instance attribute assigned from the slice, so reading the attribute and assigning it in a test both keep working. For a lazy member, the alias is its existing property: the getter calls `get()`, and the setter calls `provide()`.
- **One identity test:** TASK-36947.2 keeps the list of aliases in one test. After construction, it asserts that `app.<alias> is <slice field>` for every entry.
- **Shrink only:** the alias list can only shrink. A service added after TASK-36947.2 goes into the bundle with no alias.
- **Removal:** an alias is deleted by the PR that migrates its last reader. That PR shows that no reader is left: no string lookup in TASK-36942.4's census, and no attribute read found by grep.
- **No reassignment:** production code never assigns an alias after construction.
- **End:** the alias period ends when the list is empty. There is no date (open question 2).

### 5. Teardown

- **Owner:** whoever builds a bundle or a `ProfileStorage` tears it down.
- **No generic close:** `ApplicationServices` has no generic `close()`. The TUI keeps the explicit shutdown sequence in `on_unmount` (`app_lifecycle.py:997`), which now reads each owner from the bundle instead of from an app attribute. The sequence stops workers and pools before stores, then closes the provider's cached client (`:1297`), the ingest job store (`:1480`), and the writing and research services (`:1483`, `:1486`). The sequence is deliberate and is not the reverse of construction order.
- **Lazy members at shutdown:** shutdown reads lazy members with `peek()`, so it never builds a service just to close it (the rule at `app_lifecycle.py:375-376`).
- **`ProfileStorage.close()`:** it closes the calling thread's connection on each database that was opened. It logs a failure by type name and never raises. The TUI does not call it and leaves storage to process exit, as it does today. The other roots call it when they stop (section 6).

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

- **What it opens:** each database whose path is given; the others stay `None`. If one fails, it closes the ones it already opened and raises.
- **Same options for every root:**
  - the Console Library migration seed is always passed;
  - `check_integrity` is used as given;
  - built-in content is seeded only when `seed_builtin_content` is set, and only the TUI sets it.
- **Openers:** the factory is built from one opener per database (`open_chachanotes_db(path, *, client_id, ...)` and the rest). The TUI calls the openers directly (section 3). Once the migrations below land, the openers are the only production code outside `DB/` that constructs these database classes for a profile.
- **Paths:** paths still come from config's path helpers (`get_chachanotes_db_path()` and the rest). Only the database instances leave config (TASK-36947.6).

| Entry point | How it gets its services |
| --- | --- |
| Standalone MCP server | `open_profile_storage(client_id=CLI_APP_CLIENT_ID, chachanotes=get_chachanotes_db_path(), media=get_media_db_path())`. It builds `NotesInteropService` and `MCPTools` from that storage, and `main()` (`MCP/server.py:1167`) closes the storage. |
| Mobile engine | `open_profile_storage(client_id="mobile-service", chachanotes=data_dir / "chat.db", workspaces=data_dir / "workspaces.db", agent_runs=data_dir / "runs.db")`. These are the same isolated files (ADR-234), opened with the shared options. After TASK-36947.4 the controller receives slices, so `HeadlessAppContext` is no longer needed. |
| Mobile reading | `open_profile_storage(client_id="mobile-service", chachanotes=..., media=...)` on first read, as today. `shutdown()` (`mobile_service.py:1848`) calls `close()`. |
| RAG backfill CLI | `open_profile_storage` with only the paths its item types need, closed in a `finally`. |
| `doctor` | `run_doctor` gains a `storage` parameter that `/doctor` fills from the bundle, and it passes `storage.chachanotes` to the existing `integrity_fn` parameter of `check_database_integrity` (`Utils/doctor.py:88`). A standalone doctor command, if one is added, calls the factory. |
| In-app MCP delegate and the other modules that use config's getters | They receive their slice through their constructors, for example `LocalMCPControlService` (`app_service_wiring.py:3058`). TASK-36947.6 does this. |

### 7. Consumers

- **Non-UI code** receives its slices as constructor or function parameters and never receives the app.
- **UI capabilities** are not services. These include `notify`, `call_from_thread`, `query_one` and the screen stack. A consumer that needs one receives that one callable.
- **Screens** are the edge between the UI and the bundle. `BaseAppScreen` already types `app_instance` as `TldwCli` (`UI/Navigation/base_app_screen.py:101`). A screen reads `self.app_instance.services.<slice>` as a typed attribute and passes slices to whatever it builds (open question 4).
- **First migrations:**
  - RAG (TASK-36947.3) receives `storage`, `workspace` and `chat.citation_trace_repository`, which are what `chat_rag_events` reads from the app today.
  - The Console controller (TASK-36947.4) receives its slices plus `notify` and `call_from_thread`.
  - The modules that use config's getters move in TASK-36947.6.

### 8. Out of scope

- **No lookup by key:** there is no DI container, no service locator, no registry keyed by name or type, and no `get_service(...)`. The bundle cannot be mutated after it is published.
- **No lifecycle framework:** there is no generic lifecycle manager or teardown protocol, and services are not replaced while the app runs. Storage handles stay fixed until restart (ADR-004).
- **Not in the bundle:** config, the runtime policy and UI state (rule 4).
- **Backup/Recovery:** participants are unchanged. F10 owns them (TASK-36948.x).
- **The `prompts_interop` module singleton** (`Prompt_Management/Prompts_Interop.py:127`) stays. `LocalPromptService` takes that module itself (`app_service_wiring.py:2289`). Retiring the singleton is separate work, not part of this stream.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep ADR-036's deferral | The condition it waited for has been met. `HeadlessAppContext` shows the cost: a second root fakes the app in order to reuse the controller. |
| DI container or mutable registry | The owner rejected it. It turns attribute access back into lookup by key. |
| One flat bundle with ~199 fields | Every consumer would still receive everything, which is the app again. The slice is the unit that a consumer receives. |
| Read-only alias properties | They would break every test that assigns an app attribute, and they protect nothing until readers migrate. |
| A `close()` on the bundle that runs in reverse construction order | Today's unmount order is deliberate (workers before stores) and is not the reverse of construction order. |
| Moving all construction into new slice builders in TASK-36947.2 | It would reorder 3,690 lines of wiring in one change. Helpers that return slices give the same object identities without reordering. |

## Consequences

- **Typed dependencies:** dependencies become typed and visible. Tests build one slice instead of faking the app. Today 162 test files build a fake `app_instance` (§F1).
- **One way to open a database:** every root opens a profile database the same way. As a side effect, the mobile reading path gets the migration seed it is missing.
- **Two ways to reach a service:** during the alias period every service is reachable through the app and through the bundle. The identity test and the shrink-only alias list keep the two in step. However, a test that assigns an alias does not affect a consumer that has already migrated.
- **Rule 5 services:** they stay app attributes until their own decision.
- **Import cost:** the bundle module must not gain eager imports. The boot import ratchets (ADR-097) check this.

## Open questions for the owner

1. **Alias form.** Recommendation: plain writable attributes for eagerly built members, not read-only properties (section 4).
2. **When aliases end.** Recommendation: no date. The shrink-only list, together with TASK-36942.4's ratchet, means the count only goes down.
3. **Services composed after first paint (rule 5).** Recommendation: keep them on the app in TASK-36947.2. When each feature's consumers migrate, decide whether it gets a set-once cell or its own owner.
4. **Screens.** Recommendation: screens read `app_instance.services`. The alternative is to pass slices through screen constructors, which the navigation registry would then have to supply per screen.
5. **Who moves the roots outside the TUI.** Recommendation: TASK-36947.2 adds the factory and moves the TUI onto it. A follow-up task moves the MCP server, the mobile service and the backfill CLI, which no current subtask covers.

On acceptance:
- change this record's status to Accepted;
- change ADR-036's status to "Accepted; amended by ADR-244 (deferral clause)";
- update both index rows.

## Links

- [ADR-036: Application Service Composition Lifecycle](036-application-service-composition-lifecycle.md)
- [ADR-033: Application Session State Ownership](033-application-session-state-ownership.md)
- [ADR-097: Boot budget ratchets](097-boot-budget-ratchets.md)
- [ADR-234: Served mobile companion surface](234-served-mobile-companion-surface.md)
- [ADR-004: Settings storage defaults restart boundary](004-settings-storage-defaults-restart-boundary.md)
- [Architecture review 2026-10-25](../../Docs/QA/architecture-review-2026-10-25/report.md)
