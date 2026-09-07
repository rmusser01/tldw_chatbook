# Library Collections Capture Cutover Inventory

**Task:** TASK-18919

**ADR:** `backlog/decisions/113-collections-capture-authority-and-legacy-boundary.md`

**Checkpoint:** Task 14 cutover implemented; Task 15 Local verification passed and enabled-Server
verification is blocked by the unreachable configured deployment

This inventory records the completed generic-container Collections cutover. No old operation is
redirected to a capture operation: captures use capture-specific identities, contracts, services,
and UI ownership. The Local half of Task 15 is complete. The enabled-Server half remains required
and was stopped at the docs-info gate without a bypass or mutation.

| Surface | Current symbol / selector | Cutover result | Verification |
| --- | --- | --- | --- |
| Library destination route | `LIBRARY_ROW_BROWSE_COLLECTIONS`, `LIBRARY_NAV_MODE_TO_ROW_ID["collections"]`, canvas kind `collections` in `library_screen.py` | **cut over**: the existing destination row mounts the capture reader under a capture-specific controller | Destination shell, route receipt, compose-once, capture app-wiring, and six-reader route-cycle tests |
| Generic Collections panel | `LibraryCollectionsPanel`, `#library-collections-panel`, form/pager/action selectors in `Widgets/Library/library_collections_panel.py` | **retired and deleted** after the capture reader mounted | Absence assertions for the old owner/selectors plus mounted capture-reader tests |
| Generic browse state | `CollectionBrowseScope`, `CollectionBrowseResult`, `LibraryCollectionsPanelState`, create/rename/delete action models in `Library/library_collections_state.py` | **retired and deleted**; no unqualified collection identity is reused for captures | Import-closure check and capture model/authority tests |
| Generic browse controller | `LibraryCollectionsBrowseController` in `UI/Library_Modules/library_collections_browse_controller.py` | **retired and deleted**; capture paging/mutations use the generation-fenced controller | Old controller import/selector absence plus capture controller tests |
| Screen-owned generic state and handlers | `_library_collections_*`, `_request_library_collections_browse`, create/rename/delete/undo handlers, `library_collections_page` receipt state in `library_screen.py` | **retired**; the screen composes/drives only the destination-owned capture reader and explicit legacy recovery | Mounted navigation, mutation fencing, receipt, stale-page, lifecycle, focus, and route-retention tests |
| App composition | `_wire_library_collections_services`, `local_library_collections_db`, `local_library_collections_service`, `library_collections_service` in `app.py` | **legacy_read_only** for the v1 compatibility service; add separately named capture scope/repository/recovery services | App wiring test proves distinct attributes and no capture alias through `library_collections_service` |
| v1 database objects | `library_collections`, `library_collection_items`, schema version 1 in `DB/Library_Collections_DB.py` | **recovery-only** durable data; rows/names remain untouched beside additive v2 capture tables | Real v1 migration fixture, value equality, foreign-key check, and recovery export tests |
| Generic service protocol | `LibraryCollectionsService` and `LocalLibraryCollectionsService` list/get/create/rename/delete/restore/add-member methods | **legacy_read_only**; reads support bounded recovery, while every mutation returns `legacy_read_only` | Legacy compatibility tests cover bounded reads and reject every mutation |
| Generic create action | `create_collection()` and `#library-create-collection` | **retired** from current UI; **legacy_read_only** at the compatibility seam | No old selector; compatibility mutation rejection |
| Generic rename action | `rename_collection()` and `#library-rename-collection` | **retired** from current UI; **legacy_read_only** at the compatibility seam | No old selector; compatibility mutation rejection |
| Generic membership action | `add_item_to_collection()` and direct `library_collection_items` membership | **retired** as a current operation; **recovery-only** in export/inspection | No capture redirect; complete legacy export contains stored memberships |
| Generic delete/restore actions | `delete_collection()`, `restore_collection()`, `#library-delete-collection`, `#library-collections-delete-undo` | **retired** from current UI; **legacy_read_only** at the compatibility seam | No old selectors; compatibility mutation rejection; legacy data unchanged |
| Agent tool descriptors | `library_list_collections`, `library_get_collection`, `library_search_collections` in `Library/library_tool_contract.py` | **retired** without renaming or redirecting these operations to captures | Descriptor and cross-runtime parity tests assert absence while unrelated Library tools remain |
| Local tool dispatch | item type `collection`, `_LIST_METHODS`, `_SEARCH_METHODS`, `_get_collection`, constructor `collections_service` in `local_library_tool_service.py` | **retired**; captures are not a generic Library item backend | Local tool tests assert unknown/absent old operations and unchanged peer types |
| MCP composition | `_build_collections()` and `collections_service=backends["collection"]` in `MCP/server.py` | **retired** with its exposed generic operations | MCP catalog/control tests assert absence and stable count/contracts for retained tools |
| Console activity composition | `collections_service=getattr(app, "local_library_collections_service", None)` in `UI/Console_Modules/library_activity.py` | **retired** from generic service injection | Console Library activity no longer advertises a Collections item backend |
| Runtime policy | capability `library_collections`, resource `library.collections` with LIST/DETAIL in `runtime_policy/registry.py` | **retired** only `library.collections`; the stable capability and unrelated templates/media/notes policy resources remain | Runtime-policy tests assert resource absence and retained sibling actions |
| Rail and Home-facing count seam | `LibraryShellState.collections_count/collections_known`, `LIBRARY_ROW_BROWSE_COLLECTIONS`, subtitle `item sets`, and `_build_library_shell_state()` generic total | **cut over** to the active capture authority's exact total and Captures copy; no generic Collections Home action exists | Rail count/tooltip tests use capture semantics and Local/Server replacement, including unknown totals |
| Search/RAG descriptions | `library_search_rag_panel.py` says workspaces/collections have no retrieval seam; `library_screen.py` supplies `counts_by_source["collections"] = 0` | **retired**: global Search/RAG exposes no generic Collections pseudo-scope; capture search remains inside its authority-specific reader | RAG panel/state tests keep only supported source toggles and contain no generic Collections source row |
| User guidance and help copy | `LIBRARY_COLLECTIONS_STATUS_LINE`, `No Collections yet`, `item sets`, and descriptor help text for generic collection tools | **retired** and replaced with capture-owned reading-list copy; no global help command was found | Text/selector assertions prohibit generic-container and “adding items is coming” language |
| Generic Collections tests | `test_library_collections_service/state/browse_controller`, `test_library_collections_panel`, phase-39 Collections, tool/MCP/rail/content-hub cases | **retired or rewritten**; only recovery, schema-v1 compatibility, capture behavior, and unrelated shared-shell coverage remain | Test inventory diff plus capture migration/service/controller/mounted/cross-reader suites |
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
- One canonical `TASK-18919` task and one canonical accepted `ADR-113` exist; the other matches are
  their intentional references in the spec, plan, ADR index, and pagination design.
- The Backlog task already contains the required ADR implementation block verbatim.

## Task 14 cutover evidence

- The generic state, controller, and panel modules and their direct tests are deleted.
- The public local/MCP tool catalog contains 21 retained Library tools and no generic Collections
  list/get/search descriptor or `collection` item backend.
- All legacy mutation methods remain callable and fail closed with `legacy_read_only`; v1 data stays
  reachable only through bounded inspection and coherent recovery export.
- `library.collections.*` is absent from runtime policy while the stable capability and unrelated
  template/media/notes resources remain unchanged.
- The rail says Captures, onboarding evidence comes from the active capture authority, and global
  Search/RAG no longer presents a generic Collections source.
- The capture reader owns route activation, exact paging, independent Library/Items preferences,
  F6 Work focus, and dirty-reader retention across all six Library readers.
- Production-shaped cross-reader verification passed 115 tests, including Captures at
  160×50, 120×35, 100×30, and 80×24. Isolated Local/Server live evidence is intentionally deferred
  to Task 15.
- The combined focused capture, cutover, policy, MCP, and live-closeout gate passed 945 tests; the
  pre-import payload ratchet also passed at 487/500 modules and 376,166/380,000 LOC.

## Task 15 live-verification checkpoint

- The isolated Local production-shaped walkthrough passed with 45 captures at 160×50, 120×35,
  100×30, and 80×24. It covers exact 20/20/5 paging, every optional-pane posture, reclaimed Items
  width, resize restoration, F6 traversal, Quick Capture commit-before-extract, controlled failure
  and Retry, all four Work modes, archive/Undo, offline hard-delete cleanup, and complete 45-row /
  45-membership legacy export.
- The focused controller/reader/live gate passes 29 tests, the complete capture feature/live gate
  passes 206, and the production-shaped cross-reader gate passes 490. Details are recorded in
  `2026-08-31-library-collections-live-verification.md`.
- Production-shaped containment checks found and corrected compact Items and Work toolbar overflow;
  they now verify every visible descendant remains inside its owning pane.
- Unknown Server-save behavior is mounted-test verified without contacting Server: the draft is
  retained, refresh is offered first, retry is not automatic, and the explicit retry warning
  describes possible Saved/Favorite default reapplication.
- The configured Server profile could not reach docs-info (`APIConnectionError`), so exact
  `hasReadingSnapshotPagesV1: true` was not attested. The Server walkthrough did not run and no
  Server data was mutated. TASK-18919 remains In Progress.
