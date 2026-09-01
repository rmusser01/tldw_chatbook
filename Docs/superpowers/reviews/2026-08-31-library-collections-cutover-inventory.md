# Library Collections Capture Cutover Inventory

**Task:** TASK-18919

**ADR:** `backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md`

**Checkpoint:** Before capture production code

This inventory names every current generic-container Collections seam that must be retired, made
read-only, or retained only for recovery. No old operation is redirected to a capture operation:
captures use new capture-specific identities, contracts, services, and UI ownership.

| Surface | Current symbol / selector | Cutover result | Verification |
| --- | --- | --- | --- |
| Library destination route | `LIBRARY_ROW_BROWSE_COLLECTIONS`, `LIBRARY_NAV_MODE_TO_ROW_ID["collections"]`, canvas kind `collections` in `library_screen.py` | **retire** the generic canvas ownership; the existing destination row mounts the new capture reader under a capture-specific controller | Destination shell, route receipt, compose-once, and capture app-wiring tests |
| Generic Collections panel | `LibraryCollectionsPanel`, `#library-collections-panel`, form/pager/action selectors in `Widgets/Library/library_collections_panel.py` | **retire** and delete after the capture reader is mounted | Absence assertions for the old owner/selectors plus mounted capture-reader tests |
| Generic browse state | `CollectionBrowseScope`, `CollectionBrowseResult`, `LibraryCollectionsPanelState`, create/rename/delete action models in `Library/library_collections_state.py` | **retire** and delete; do not reuse unqualified collection identity for captures | Import-closure check and capture model/authority tests |
| Generic browse controller | `LibraryCollectionsBrowseController` in `UI/Library_Modules/library_collections_browse_controller.py` | **retire** and delete; capture paging/mutations use the new generation-fenced controller | Old controller import/selector absence plus capture controller tests |
| Screen-owned generic state and handlers | `_library_collections_*`, `_request_library_collections_browse`, create/rename/delete/undo handlers, `library_collections_page` receipt state in `library_screen.py` | **retire**; the screen only composes/drives the destination-owned capture reader and explicit legacy recovery | Mounted navigation, mutation fencing, receipt, stale-page, and lifecycle tests |
| App composition | `_wire_library_collections_services`, `local_library_collections_db`, `local_library_collections_service`, `library_collections_service` in `app.py` | **legacy_read_only** for the v1 compatibility service; add separately named capture scope/repository/recovery services | App wiring test proves distinct attributes and no capture alias through `library_collections_service` |
| v1 database objects | `library_collections`, `library_collection_items`, schema version 1 in `DB/Library_Collections_DB.py` | **recovery-only** durable data; rows/names remain untouched beside additive v2 capture tables | Real v1 migration fixture, value equality, foreign-key check, and recovery export tests |
| Generic service protocol | `LibraryCollectionsService` and `LocalLibraryCollectionsService` list/get/create/rename/delete/restore/add-member methods | **legacy_read_only**; reads support bounded recovery, while every mutation returns `legacy_read_only` | Legacy compatibility tests cover bounded reads and reject every mutation |
| Generic create action | `create_collection()` and `#library-create-collection` | **retire** from current UI; **legacy_read_only** at the compatibility seam | No old selector; compatibility mutation rejection |
| Generic rename action | `rename_collection()` and `#library-rename-collection` | **retire** from current UI; **legacy_read_only** at the compatibility seam | No old selector; compatibility mutation rejection |
| Generic membership action | `add_item_to_collection()` and direct `library_collection_items` membership | **retire** as a current operation; **recovery-only** in export/inspection | No capture redirect; complete legacy export contains stored memberships |
| Generic delete/restore actions | `delete_collection()`, `restore_collection()`, `#library-delete-collection`, `#library-collections-delete-undo` | **retire** from current UI; **legacy_read_only** at the compatibility seam | No old selectors; compatibility mutation rejection; legacy data unchanged |
| Agent tool descriptors | `library_list_collections`, `library_get_collection`, `library_search_collections` in `Library/library_tool_contract.py` | **retire**; do not rename or redirect these operations to captures | Descriptor and cross-runtime parity tests assert absence while unrelated Library tools remain |
| Local tool dispatch | item type `collection`, `_LIST_METHODS`, `_SEARCH_METHODS`, `_get_collection`, constructor `collections_service` in `local_library_tool_service.py` | **retire** the item type/backend; captures are not a generic Library item backend | Local tool tests assert unknown/absent old operations and unchanged peer types |
| MCP composition | `_build_collections()` and `collections_service=backends["collection"]` in `MCP/server.py` | **retire** the generic backend and its exposed operations | MCP catalog/control tests assert absence and stable count/contracts for retained tools |
| Console activity composition | `collections_service=getattr(app, "local_library_collections_service", None)` in `UI/Console_Modules/library_activity.py` | **retire** the generic service injection | Console Library activity tests prove no Collections item backend is advertised |
| Runtime policy | capability `library_collections`, resource `library.collections` with LIST/DETAIL in `runtime_policy/registry.py` | **retire** only `library.collections`; retain the capability's unrelated templates/media/notes agent-tool policy resources | Runtime-policy tests assert resource absence and retained sibling actions |
| Rail and Home-facing count seam | `LibraryShellState.collections_count/collections_known`, `LIBRARY_ROW_BROWSE_COLLECTIONS`, subtitle `item sets`, and `_build_library_shell_state()` generic total | **retire** the generic-container count/copy; capture authority owns the exact current total. No separate generic Collections Home action was found. | Rail count/tooltip tests use capture semantics and Local/Server replacement, including unknown totals |
| Search/RAG descriptions | `library_search_rag_panel.py` says workspaces/collections have no retrieval seam; `library_screen.py` supplies `counts_by_source["collections"] = 0` | **retire** the generic Collections pseudo-scope/copy; capture search stays inside its authority-specific reader | RAG panel tests keep only supported source toggles and contain no generic Collections source row |
| User guidance and help copy | `LIBRARY_COLLECTIONS_STATUS_LINE`, `No Collections yet`, `item sets`, and descriptor help text for generic collection tools | **retire** and replace only within new capture-owned UI/tool contracts; no global help command was found | Text/selector assertions prohibit generic-container and “adding items is coming” language |
| Generic Collections tests | `test_library_collections_service/state/browse_controller`, `test_library_collections_panel`, phase-39 Collections, tool/MCP/rail/content-hub cases | **retire** tests for deleted current surfaces; preserve/rewrite only recovery, schema-v1 compatibility, and unrelated shared-shell coverage | Test inventory diff plus capture migration/service/controller/mounted/cross-reader suites |
| Legacy inspector/export | No dedicated bounded v1 recovery owner exists today | **recovery-only** through new `collections_legacy_recovery.py`; reachable whenever compatible v1 data exists, including rollback posture | Bounded inspection, coherent-snapshot atomic JSON export, interruption, and recovery-only route tests |

## Explicitly unrelated uses of “collection”

The following remain outside this cutover and must not be renamed or removed by string matching:

- Prompt collections and prompt membership (`prompt_collections`, `prompts.collections`).
- Watchlists Collections screen and feed-subscription bundles.
- Server `Collections_Interop` feeds, Reading List, outputs/templates/artifacts capabilities.
- RAG/vector-store implementation “collections” (Chroma/in-memory index partitions and stats).
- Python `collections` / `collections.abc` imports.

## Checkpoint evidence

- One implementation branch exists: `codex/task-18919-collections-reader`.
- The open-PR search for `18919 Collections capture` returned no match.
- One canonical `TASK-18919` task and one canonical accepted `ADR-107` exist; the other matches are
  their intentional references in the spec, plan, ADR index, and pagination design.
- The Backlog task already contains the required ADR implementation block verbatim.
