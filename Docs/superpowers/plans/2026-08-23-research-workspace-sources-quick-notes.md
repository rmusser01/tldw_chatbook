# Research Workspace Sources and Quick Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`, apply `superpowers:test-driven-development`
> before each production edit, and apply
> `superpowers:verification-before-completion` before each commit. Do not
> delegate unless the user explicitly requests subagents. Apply `impeccable`
> immediately before UI implementation tasks.

**Goal:** Make source intake, attachment, readiness, selection, organization,
and Quick Notes durable in the explicitly selected Local or Server workspace.

**Architecture:** Treat catalog ingest and workspace association as two
ordered, independently retryable stages. Extend existing ingest jobs with an
opaque operation link; store the qualified association intent and receipt in
WorkspaceDB before intake starts. A completion coordinator resumes from the
canonical item ID and performs an idempotent Local membership or Server
workspace-source write. Canonical Notes remain in the existing Notes owners;
folders/annotations remain in the device overlay.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite migrations, existing Library
ingest registry, Media/Notes scope services, workspace registry, server
workspace APIs, pytest.

**Spec:**
`Docs/superpowers/specs/2026-08-23-research-workspace-design.md`

**Backlog:** `TASK-21508` (depends on `TASK-21507`)

## Global constraints

- Persist the qualified target before starting intake. Never derive it from
  the screen visible when completion arrives.
- Local creates/reuses only local Library; Server creates/reuses only server
  Media. No compensating cross-authority write and no fallback.
- Association is Local `WorkspaceMembership(role="source")` or the Server
  workspace-source row. `workspace:<slug>` is optional projection metadata,
  never the relationship.
- A successful catalog write is never rolled back because association or
  readiness failed.
- Desired selection and effective retrieval readiness are separate states.
- Removing association cannot delete the canonical item.
- Keep annotations/folders in the private overlay. Keep Quick Note content in
  canonical Notes owners.
- Do not synchronously mutate databases from ingest-registry listeners; invoke
  the coordinator from app-owned completion paths/workers.
- No full-suite run or claim without explicit user approval.

## ADR check

ADR required: no new ADR

ADR path:
`backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already defines the two-stage source contract, authority
owners, durable qualified intent, overlay ownership, unlink semantics, and
partial-failure policy. The schema changes below implement that accepted
contract.

## Task 1: Add the durable source-operation and job-link migrations

**Files:**

- Modify: `tldw_chatbook/DB/Workspace_DB.py`
- Create: `tldw_chatbook/DB/migrations/workspaces_v2_to_v3_research_source_operations.sql`
- Modify: `tldw_chatbook/DB/Library_Ingest_Jobs_DB.py`
- Modify: `tldw_chatbook/Library/library_ingest_jobs.py`
- Create: `tldw_chatbook/Research_Workspace/source_operations.py`
- Create: `tldw_chatbook/Research_Workspace/source_operation_store.py`
- Test: `Tests/DB/test_workspace_db.py`
- Test: `Tests/DB/test_library_ingest_jobs_db.py`
- Test: `Tests/Research_Workspace/test_source_operation_store.py`

1. Add RED migration tests from complete historical v2/v5 fixtures, fresh DB
   tests, invalid-state tests, and a restart load test. Assert unrelated rows
   and schema versions remain intact.
2. Add `research_source_operations` in WorkspaceDB v3. It has no workspace
   foreign key because Server workspace IDs are not Local workspace rows. Use
   explicit columns for:

   ```text
   operation_id, idempotency_key, data_source, server_profile_id,
   principal_id, workspace_id, ingest_job_id, canonical_item_type,
   canonical_item_id, workspace_source_id, desired_selected,
   catalog_status, association_status, readiness_status,
   error_stage, error_code, error_message, revision, created_at, updated_at
   ```

   Keep the inline WorkspaceDB migration runner and
   `workspaces_v2_to_v3_research_source_operations.sql` byte-for-byte aligned
   for the shared migration guard.

   Constrain data source/status values, bound reads/pages, and make
   `idempotency_key` unique. Do not persist source bodies, secrets, URLs with
   credentials, or local private paths in the receipt.
3. Add the inline v5-to-v6 migration owned by `LibraryIngestJobsDB`, adding
   nullable `research_source_operation_id`, and extend `LibraryIngestJob`.
   Thread it through submit, copy, persist, reload, retry, server attach, Local
   `mark_done`, and Server `mark_remote_done` without weakening Local
   `media_id` versus Server `remote_media_id` semantics.
4. Implement frozen operation/status models and a small store with
   `create`, `get`, `list_incomplete`, and compare-version `advance_stage`.
   Allow only forward stage transitions except an explicit retry that clears
   the named failed stage.
5. Prove inverses: a Server operation with a Local DB FK must fail the fixture;
   a retry that loses operation ID must fail; and raw secret/path metadata must
   fail validation.
6. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/DB/test_workspace_db.py Tests/DB/test_library_ingest_jobs_db.py Tests/Research_Workspace/test_source_operation_store.py
   ```

7. Commit:

   ```bash
   git commit -m "feat: persist Research source association operations"
   ```

## Task 2: Implement idempotent catalog-to-workspace association

**Files:**

- Create: `tldw_chatbook/Research_Workspace/source_association.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Modify: `tldw_chatbook/app.py`
- Test: `Tests/Workspaces/test_workspace_registry_service.py`
- Test: `Tests/Research_Workspace/test_source_association.py`
- Test: `Tests/Library/test_library_ingest_jobs.py`
- Test: `Tests/Library/test_library_ingest_runner.py`

1. Add RED tests for Local done, Server done, duplicate canonical item,
   association failure, retry after restart, navigation to a different
   workspace before completion, and no-cross-adapter calls.
2. Add the missing exact unlink seam:

   ```python
   def unlink_membership(
       self,
       workspace_id: str,
       *,
       item_type: str,
       item_id: str,
       role: str = "source",
   ) -> bool:
       """Remove one association without deleting the item."""
   ```

   Make repeated unlink a safe no-op and clear only matching `RagScope` items.
3. Implement `ResearchSourceAssociationCoordinator.resume(operation_id)`.
   Resolve the linked ingest job; store Local `media_id` or Server
   `remote_media_id`; then:

   - Local: call `link_membership` and optionally write a sanitized keyword
     projection.
   - Server: call `save_workspace_source` with a deterministic source ID
     derived from operation idempotency key + server media ID.

   Complete the association stage and leave readiness `pending`; Task 3 owns
   the real readiness and desired-selection projection.

4. App completion paths schedule coordinator work after `mark_done` or
   `mark_remote_done`. Association failure updates the receipt and global
   operation status; it does not change the ingest job back to failed or delete
   its media.
5. At startup, resume bounded incomplete operations. Serialize work per
   operation ID; parallel unrelated operations may proceed. Context switches
   never change the stored qualified target.
6. Add explicit retry from catalog or association stage. Catalog retries reuse
   the ingest job/canonical duplicate rules; association retries never
   reingest.
7. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Workspaces/test_workspace_registry_service.py Tests/Research_Workspace/test_source_association.py Tests/Library/test_library_ingest_jobs.py Tests/Library/test_library_ingest_runner.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: associate Research ingest results with workspaces"
   ```

## Task 3: Add authority-specific source browsing, readiness, and selection

**Files:**

- Extend: `tldw_chatbook/Research_Workspace/contracts.py`
- Extend: `tldw_chatbook/Research_Workspace/local_adapter.py`
- Extend: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Create: `tldw_chatbook/Research_Workspace/source_readiness.py`
- Modify: `tldw_chatbook/Research_Workspace/controller.py`
- Modify: `tldw_chatbook/tldw_api/notes_workspace_schemas.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/Notes/server_notes_workspace_service.py`
- Test: `Tests/Research_Workspace/test_source_adapters.py`
- Test: `Tests/Research_Workspace/test_source_readiness.py`
- Test: `Tests/Research_Workspace/test_source_selection.py`
- Test: `Tests/tldw_api/test_workspace_source_client.py`
- Test: `Tests/Notes/test_server_notes_workspace_service.py`

1. Add RED contract tests for bounded search/filter/sort/page requests,
   attachment of existing catalog rows, source preview, ordering, desired
   selection, readiness failure/retry after restart, missing embeddings, stale
   sources, and unknown capability.
2. Extend `ResearchWorkspacePort` with explicit `list_sources`,
   `search_catalog`, `attach_existing`, `remove_source`, `update_source`,
   `preview_source`, `get_readiness`, `set_selected_scope`, and
   `reorder_sources` methods. Keep common result fields normalized while
   retaining owner IDs/versions.
3. Extend `notes_workspace_schemas.py` and the TLDW client for the audited
   server endpoints:

   - `GET /api/v1/workspaces/{id}/sources/{source_id}/preview`,
   - `PUT /api/v1/workspaces/{id}/sources/selection`,
   - `PUT /api/v1/workspaces/{id}/sources/reorder`,
   - `GET /api/v1/workspaces/{id}/sources/status`, and
   - `GET /api/v1/workspaces/{id}/capabilities`.

   Validate path segments, bounded preview/status rows and summaries,
   selection/reorder request bounds, per-output capability fields, and response
   revision through the existing TLDW client/auth/error path.
4. Add thin preview/selection/reorder/status/capability methods to
   `ServerNotesWorkspaceService`. Local catalog browsing uses
   `MediaReadingScopeService(mode="local")` and Local membership/scope
   services; Server uses the same scope seam in Server mode plus that server
   service. Each adapter asserts its qualified ref before dispatch.
5. Map readiness to exactly attached, parsing, indexing, FTS-ready,
   vector-ready, failed, unavailable, or stale. Hybrid is effective only when
   both required paths are ready; missing embeddings yields FTS-only.
6. Persist desired Local IDs in the workspace `RagScope` and Server IDs through
   the audited selection endpoint even while parsing/indexing; the device
   overlay does not own grounding scope. Derive effective retrieval IDs as the
   intersection of desired selection and current mode's readiness. Never
   silently deselect a temporarily unavailable source.
7. Resume readiness-pending source operations at startup and after association;
   advance their readiness receipt through optimistic operation transitions.
   Readiness retry rechecks the canonical source without reingest or
   reassociation.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_source_adapters.py Tests/Research_Workspace/test_source_readiness.py Tests/Research_Workspace/test_source_selection.py Tests/tldw_api/test_workspace_source_client.py Tests/Notes/test_server_notes_workspace_service.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: add Research source readiness and selection"
   ```

## Task 4: Build the Sources pane and intake/receipt UI

**Files:**

- Modify: `tldw_chatbook/UI/Research_Workspace_Modules/sources_region.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/add_source_modal.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/source_list.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/source_receipt.py`
- Modify: `tldw_chatbook/UI/Screens/research_workspace_screen.py`
- Modify: `tldw_chatbook/css/features/_research_workspace.tcss`
- Test: `Tests/UI/test_research_sources_region.py`
- Test: `Tests/UI/test_research_add_source_modal.py`
- Test: `Tests/UI/test_research_source_receipt.py`
- Test: `Tests/UI/test_research_workspace_geometry.py`

1. Add mounted RED tests for authority-specific vocabulary, Import/Upload,
   URL/Paste batch intake, existing Library/My Media attachment, search,
   filters, pagination, selection, readiness, preview, retry, remove, and late
   completion receipts.
2. The modal creates the operation first. Only after successful durable create
   may it submit Library/Server ingest or attach an existing canonical ID.
   Close/navigation leaves the operation visible in global status.
3. Render a paged source list with selected intent and readiness as separate
   columns/text. Include accessible Select all/none, reorder up/down, preview,
   remove-association, and stage retry. Withhold destructive catalog deletion
   from the normal remove control.
4. Receipts show separate `Library/Media`, `Workspace association`, and
   `Index/readiness` results with exact owner and retry stage.
5. Implement source folders/annotations in overlay schema v2, migrating v1
   records without inventing folders. Use each source's qualified stable ID;
   label the section exactly as device-only. Folder selection requires an
   explicit `Select folder sources` action.
6. Rebuild CSS and run mounted geometry at 160x40, 120x30, 100x30, 84x24,
   80x24, 60x20. Assert the essential Add/selection/status/recovery controls
   are painted and reachable in the active Sources pane.
7. Run:

   ```bash
   .venv/bin/python tldw_chatbook/css/build_css.py
   .venv/bin/python -m pytest -q Tests/UI/test_research_sources_region.py Tests/UI/test_research_add_source_modal.py Tests/UI/test_research_source_receipt.py Tests/UI/test_research_workspace_geometry.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: add Research Sources workbench"
   ```

## Task 5: Add canonical Quick Notes

**Files:**

- Create: `tldw_chatbook/Research_Workspace/quick_notes.py`
- Extend: `tldw_chatbook/Research_Workspace/contracts.py`
- Extend: `tldw_chatbook/Research_Workspace/local_adapter.py`
- Extend: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Modify: `tldw_chatbook/UI/Research_Workspace_Modules/studio_region.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/quick_notes_section.py`
- Test: `Tests/Research_Workspace/test_quick_notes.py`
- Test: `Tests/UI/test_research_quick_notes.py`

1. Add RED Local/Server tests for create/list/load/search/update/delete,
   optimistic conflict, message/source provenance, no-cross-call, and Local
   membership cleanup.
2. Extend `ResearchWorkspacePort` with `list_notes(ref, page)`,
   `get_note(ref, note_id)`, `save_note(ref, request)`, and
   `delete_note(ref, note_id, expected_version)`. The returned note ref always
   carries the qualified workspace and canonical owner/version.
3. Implement a thin `ResearchQuickNotesService` through that selected port:

   - Local adapter create/update calls `NotesScopeService.save_note(scope="local_note",
     user_id=...)`; first create then calls `link_membership(item_type="note",
     role="note")`.
   - Server adapter calls `save_note(scope="workspace_note", workspace_id=...)` and
     the canonical server workspace-note list/search/delete methods.

   Do not copy Server notes into Local Notes or store note bodies in overlay.
4. Quick Notes mounts under Studio and supports title, Markdown edit/preview,
   tags, list/search, create/update, download, clear, undo, and conflict
   recovery. Capture provenance fields in owner-supported metadata/keywords,
   not in an alternate note row.
5. Use expected versions for update/delete. A stale write offers Reload or
   Copy as new where the owner supports it; never overwrite silently.
6. Before workspace or authority switching, flush a non-empty Quick Notes
   editor through the exact canonical note owner. Save failure or conflict
   blocks switching with Retry, Discard editor changes, and Cancel; note draft
   bodies never enter the device overlay.
7. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_quick_notes.py Tests/UI/test_research_quick_notes.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: add Research workspace Quick Notes"
   ```

## Task 6: Round-trip verification and closeout

**Files:**

- Modify: `Docs/User_Guide/research_workspace.md`
- Modify: `backlog/tasks/task-21508 - Add-Research-Sources-ingest-association-and-Quick-Notes.md`
- Test: `Tests/integration/test_research_source_round_trip.py`

1. Add integration tests with real temporary SQLite proving Local intake is
   visible in general Library and its captured workspace, duplicate content is
   reused, unlink leaves Library intact, and app restart resumes association.
2. Add Server fake/live-contract tests proving returned server `media_id` is
   never written as Local `media_id`, My Media contains the canonical result,
   and the server workspace-source row references it.
3. Run all tests listed by this plan, Ruff on the changed Python inventory, CSS
   build/parity when CSS changed, and `git diff --check`. Live verification
   must use an isolated `TLDW_CONFIG_PATH`; record server unavailability as a
   limitation, not success.
4. Update the guide and TASK-21508 only from fresh evidence. State that full
   pytest was not run.
5. Commit:

   ```bash
   git commit -m "docs: complete Research Sources and Quick Notes"
   ```

## Required inverse checks

1. Complete ingest after switching workspaces and attach to the visible one;
   captured-target test must fail.
2. Store Server `remote_media_id` in Local `media_id`; authority-ID test must
   fail.
3. Treat a workspace keyword as membership; rename/unlink test must fail.
4. Roll back/delete Library after association failure; partial-receipt test
   must fail.
5. Remove a membership and delete canonical media; unlink test must fail.
6. Mark Hybrid ready with missing embeddings; readiness test must fail.
7. Put Quick Note content into overlay; overlay payload test must fail.

## Focused verification boundary

Run the named DB, Library, Workspace, Notes, Research_Workspace, UI, and single
integration files only; run CSS build/parity if CSS changes, Ruff on the final
changed Python inventory, and `git diff --check`. No full-suite claim.
