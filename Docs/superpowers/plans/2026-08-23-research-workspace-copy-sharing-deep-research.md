# Research Workspace Copy, Sharing, and Deep Research Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`; apply `superpowers:test-driven-development`
> before production changes and `superpowers:verification-before-completion`
> before commits. Do not delegate unless the user explicitly requests
> subagents. Apply `impeccable` immediately before UI implementation tasks.

**Goal:** Add explicit resumable cross-authority Copy, real server workspace
sharing, and a durable launch/preview-return bridge to the separately owned
Deep Research Runs screen.

**Architecture:** A manifest-first `ResearchCopyService` freezes qualified
source/destination identities and per-item treatments, writes payload-free
local receipts after each stage, and uses canonical item adapters for actual
content. Local-to-Server also records the audited server migration session and
chunk/finalize receipts; Server-to-Local resolves server owners into Local
canonical owners. Sharing delegates to the existing server sharing scope, and
Deep Research stores origin identity in the Research run plus bounded overlay
context while leaving the run state machine in Research Interop.

**Tech Stack:** Python 3.11+, Textual 8.x, WorkspaceDB SQLite migrations,
TLDW workspace migration API, existing Library/Notes/Chatbook/Study/Quiz
owners, SharingScopeService, ResearchScopeService, private overlay, pytest.

**Spec:**
`Docs/superpowers/specs/2026-08-23-research-workspace-design.md`

**Backlog:** `TASK-21511` (depends on `TASK-21507` through `TASK-21510`)

## Global constraints

- Authority toggles never transfer data. Copy begins only from an explicit
  confirmed action.
- V1 is Copy only: no Move, automatic merge, continuous sync, or silent
  fallback.
- One frozen manifest contains exact source/destination authority, server
  profile, principal, workspace, item IDs, versions/hashes, treatment,
  conflict choice, redaction class, and estimated bytes.
- Item treatments are Copy content, Reference, Metadata only, Omit, or
  Blocked. Conflict choices are Keep destination, Replace destination, Copy as
  new, Reference existing, Omit, or Cancel.
- Replace requires a second confirmation and exact destination version checks.
- Stable transfer/item idempotency keys and per-stage durable receipts prevent
  duplicate acknowledged items on retry.
- Device-only overlays are always excluded and named in preflight/export/share
  disclosure.
- Local mode has Export bundle, Copy to Server, and Copy to Server and Share;
  only Server mode has Share.
- `Copy to Server and Share` reaches Share confirmation only after Copy is
  fully Completed.
- Deep Research Runs remains the only run/checkpoint/event/artifact/budget
  owner. Return creates nothing until preview confirmation.
- Unknown capability fails closed; no silent Local/Server fallback or blend.
- No full-suite run/claim without explicit user approval.

## ADR check

ADR required: no new ADR

ADR path:
`backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already defines Copy-only transfer, receipt ownership,
server-only sharing, overlay exclusion, and Deep Research ownership/return.

## Task 1: Define Copy manifest, item treatment, conflict, and receipt contracts

**Files:**

- Create: `tldw_chatbook/Research_Workspace/copy_models.py`
- Create: `tldw_chatbook/Research_Workspace/copy_manifest.py`
- Extend: `tldw_chatbook/Research_Workspace/contracts.py`
- Test: `Tests/Research_Workspace/test_copy_models.py`
- Test: `Tests/Research_Workspace/test_copy_manifest.py`

1. Add RED tests for same-authority rejection, unqualified server identity,
   mutable source versions, duplicate items, unsupported item type, overlay
   inclusion, secret/private-path leakage, replacement without version/second
   confirmation, deterministic manifest hash, and stable idempotency keys.
2. Define the exact contracts:

   ```python
   class CopyTreatment(StrEnum):
       CONTENT = "copy_content"
       REFERENCE = "reference"
       METADATA_ONLY = "metadata_only"
       OMIT = "omit"
       BLOCKED = "blocked"

   class CopyConflictChoice(StrEnum):
       KEEP_DESTINATION = "keep_destination"
       REPLACE_DESTINATION = "replace_destination"
       COPY_AS_NEW = "copy_as_new"
       REFERENCE_EXISTING = "reference_existing"
       OMIT = "omit"
       CANCEL = "cancel"

   @dataclass(frozen=True, slots=True)
   class ResearchCopyManifest:
       transfer_id: str
       source: QualifiedWorkspaceRef
       destination: QualifiedWorkspaceRef
       items: tuple[ResearchCopyItem, ...]
       manifest_hash: str
       created_at: str
   ```

3. Derive `transfer_id` and each `item_idempotency_key` from canonical stable
   JSON plus SHA-256. Never hash an API secret, private path, or source body
   into a user-visible receipt field.
4. Implement `ResearchCopyPreflightBuilder.build(...)` using explicit adapter
   capabilities. It returns per-item treatment, conflict, version, size,
   redaction, unsupported reason, and recovery; it does not mutate owners.
5. Extend `ResearchWorkspacePort` with forward-typed
   `preflight_copy(source, destination, item_refs)`, `execute_copy(manifest,
   confirmation)`, `get_copy_receipt(transfer_id)`, and
   `resume_copy(transfer_id)` methods. The selected source/destination adapters
   are captured in the manifest; neither is re-resolved from current UI state.
6. Define item types initially supported by real owner adapters: media/source,
   note, textual output/Chatbook, Study deck/cards, and Quiz/questions.
   Conversations and any owner without a complete import contract are
   `Blocked`, not serialized into an ad hoc substitute.
7. Model operation terminal states exactly as `completed`,
   `partially_completed`, `rolled_back`, `failed_retryable`, and
   `failed_terminal`, with `preflight`, `confirmed`, and `in_progress` as
   non-terminal states.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_copy_models.py Tests/Research_Workspace/test_copy_manifest.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: define Research Copy protocol"
   ```

## Task 2: Add payload-free local transfer receipts and resume ownership

**Files:**

- Create: `tldw_chatbook/DB/migrations/workspaces_v4_to_v5_copy_receipts.sql`
- Modify: `tldw_chatbook/DB/Workspace_DB.py`
- Create: `tldw_chatbook/Research_Workspace/copy_receipts.py`
- Test: `Tests/DB/test_workspace_db_v5_migration.py`
- Test: `Tests/Research_Workspace/test_copy_receipts.py`

1. Add RED tests for v4 migration/fresh schema, exact qualified keys,
   operation/item uniqueness, optimistic revision, state transition graph,
   per-stage crash recovery, partial completion, bounded errors, and content/
   secret rejection.
2. Add WorkspaceDB v5 tables without a foreign key to local workspace rows:
   keep the inline migration runner and
   `workspaces_v4_to_v5_copy_receipts.sql` aligned.

   ```text
   research_copy_operations(
     transfer_id PK, manifest_hash, source_data_source, source_server_profile_id,
     source_principal_id, source_workspace_id, destination_data_source,
     destination_server_profile_id, destination_principal_id,
     destination_workspace_id, server_migration_id, status, summary_json,
     revision, created_at, updated_at
   )
   research_copy_items(
     transfer_id, item_key, item_type, source_owner_id, source_owner_version,
     source_hash, treatment, conflict_choice, destination_owner_kind,
     destination_owner_id, destination_owner_version, stage, status,
     retryable, error_code, error_message, receipt_json, updated_at,
     PK(transfer_id, item_key)
   )
   ```

3. `summary_json` and `receipt_json` are bounded, secret/body-free diagnostics.
   They may contain hashes, byte counts, owner IDs/versions, omitted-field
   names, and timestamps; never source/generated content.
4. Implement `ResearchCopyReceiptStore.create_from_manifest`,
   `record_item_stage`, `record_server_migration`, `finish`, `get`, and
   `list_recoverable`. Enforce monotonic stages and compare expected revision.
5. Restart reconstruction comes only from this store plus destination owner and
   server migration status. It never infers success from a visible UI row.
6. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/DB/test_workspace_db_v5_migration.py Tests/Research_Workspace/test_copy_receipts.py
   ```

7. Commit:

   ```bash
   git commit -m "feat: persist Research Copy receipts"
   ```

## Task 3: Expose audited server workspace migration receipts in Chatbook

**Files:**

- Create: `tldw_chatbook/tldw_api/workspace_migration_schemas.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Create: `tldw_chatbook/Research_Workspace/server_migration_receipts.py`
- Test: `Tests/tldw_api/test_workspace_migration_client.py`
- Test: `Tests/Research_Workspace/test_server_migration_receipts.py`

1. Add RED tests against the audited server routes:

   - `POST /api/v1/workspaces/migrations`,
   - `GET /api/v1/workspaces/migrations` and `/{migration_id}`,
   - `PUT /{migration_id}/chunks/{chunk_id}`,
   - `POST /{migration_id}/finalize`, and
   - `POST /{migration_id}/client-delete-ack` (never used by Copy v1).

2. Port a strict bounded Pydantic subset matching server
   `WorkspaceMigrationCreateRequest`, chunk declaration/upload/receipt,
   finalize request, and response fields. Use `source_product="tldw-chatbook"`.
3. Add the six client methods using `_request`, segment validation, existing
   auth/error normalization, and exact response validation. Do not create a
   second HTTP client.
4. Implement `ServerMigrationReceiptService.begin`, `ack_item`, `status`, and
   `finalize`. The server migration session is an audit/recovery owner; actual
   content is written through canonical destination APIs in Task 4.
5. Map one acknowledged Copy item to one declared/accepted chunk whose SHA,
   byte count, and metadata identify the manifest item but contain no payload.
   Finalize only after every non-omitted declared item has a server receipt.
6. Copy v1 never calls client-delete-ack because Copy retains the source.
7. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/tldw_api/test_workspace_migration_client.py Tests/Research_Workspace/test_server_migration_receipts.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: add server migration receipt client"
   ```

## Task 4: Implement canonical cross-authority item adapters and resumable Copy

**Files:**

- Create: `tldw_chatbook/Research_Workspace/copy_adapters.py`
- Create: `tldw_chatbook/Research_Workspace/copy_service.py`
- Extend: `tldw_chatbook/Research_Workspace/local_adapter.py`
- Extend: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Reuse: `tldw_chatbook/Library/library_ingest_jobs.py`
- Reuse: `tldw_chatbook/Media/media_reading_scope_service.py`
- Reuse: `tldw_chatbook/Notes/notes_scope_service.py`
- Reuse: `tldw_chatbook/Chatbooks/local_chatbook_service.py`
- Reuse: `tldw_chatbook/Study_Interop/study_scope_service.py`
- Reuse: `tldw_chatbook/Study_Interop/quiz_scope_service.py`
- Test: `Tests/Research_Workspace/test_copy_adapters.py`
- Test: `Tests/Research_Workspace/test_copy_service.py`

1. Add RED tests for each supported owner/direction, duplicate destination,
   keep/replace/new/reference/omit/cancel, version conflict, canonical write
   before association, failure between stages, retry without duplicate, stale
   visible workspace, server receipt ordering, and no overlay payload.
2. Define adapter protocol methods `inspect_source`, `detect_conflict`,
   `create_destination`, `replace_destination`, `associate_destination`, and
   `verify_destination`. Register only real owner implementations.
3. Local-to-Server mappings:

   - media/source -> existing server media ingest, poll durable ingest job,
     then server workspace-source association;
   - note -> server workspace note;
   - textual output -> server workspace artifact;
   - Study deck/cards -> server workspace Study owner;
   - Quiz/questions -> server workspace Quiz owner.

4. Server-to-Local mappings:

   - downloadable media/source -> download to a private temporary file, submit
     Local Library ingest, then Local `role="source"` membership;
   - workspace note -> Local Notes owner then membership;
   - textual workspace artifact -> Local Chatbook then `role="output"`
     membership;
   - Study/Quiz -> Local Study/Quiz owners then output membership.

5. Use the ingestion coordinator from TASK-21508 so canonical media creation
   and workspace association keep their independent receipts. Securely remove
   any temporary download after Local canonical ingest accepts it; failure
   receipts contain no temp path.
6. Execute one item at a time in manifest order: destination canonical write,
   association, destination verification, local item receipt, then server
   migration chunk receipt when destination is Server. On restart, verify and
   skip acknowledged items before resuming.
7. `Reference existing` is available only when the destination owner proves an
   exact canonical match. `Replace` requires destination version plus the
   second confirmation token. No operation deletes the source.
8. Finish fully acknowledged Local-to-Server Copy by finalizing and re-reading
   the server migration, then finishing the local receipt. Server-to-Local
   finishes from verified Local owners plus the local receipt.
9. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_copy_adapters.py Tests/Research_Workspace/test_copy_service.py
   ```

10. Commit:

   ```bash
   git commit -m "feat: add resumable Research Copy execution"
   ```

## Task 5: Build preflight, progress, receipt, export, and sharing UI

**Files:**

- Create: `tldw_chatbook/UI/Research_Workspace_Modules/copy_dialog.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/copy_receipt_view.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/sharing_dialog.py`
- Modify: `tldw_chatbook/UI/Research_Workspace_Modules/workspace_menu.py`
- Modify: `tldw_chatbook/UI/Screens/research_workspace_screen.py`
- Modify: `tldw_chatbook/css/features/_research_workspace.tcss`
- Test: `Tests/UI/test_research_copy_dialog.py`
- Test: `Tests/UI/test_research_sharing_dialog.py`

1. Add mounted RED tests for explicit menu ownership, destination selector,
   item preflight rows, conflicts/treatments, estimated bytes, redaction and
   overlay disclosure, confirm/second replace confirm, progress, cancel,
   resumable receipt, export, terminal states, and stale context fencing.
2. Local menu shows exactly Export bundle, Copy to Server, and Copy to Server
   and Share. Server menu shows Copy to Local and Share. Authority switching
   never opens this dialog or creates a receipt.
3. Export bundle uses the same frozen manifest/redaction policy but writes a
   user-selected Chatbook/export artifact; it excludes device overlay and
   includes a human-readable manifest. Export does not count as cross-authority
   Copy completion.
4. Wire Server sharing only through `SharingScopeService(mode="server")`:
   team/organization target, the server-defined permission enum, allow clone,
   private link password/expiry/use limit, active shares/revoke,
   shared-with-me, preview/verify/import/clone. Render unavailable server
   capabilities with exact reason/recovery.
5. For Copy to Server and Share, preserve the requested share draft, execute
   Copy, verify `status == completed`, then open the separate Share confirmation.
   Partial/failed/rolled-back Copy never opens Share.
6. Keep server profile/principal/workspace captured for the operation. A stale
   completion updates its receipt but not the newly visible workspace.
7. Rebuild CSS and prove dialog actions/receipts are usable at the foundation's
   six terminal sizes.
8. Run:

   ```bash
   .venv/bin/python tldw_chatbook/css/build_css.py
   .venv/bin/python -m pytest -q Tests/UI/test_research_copy_dialog.py Tests/UI/test_research_sharing_dialog.py Tests/UI/test_research_workspace_geometry.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: add Research Copy and sharing UI"
   ```

## Task 6: Add durable Deep Research launch and validated return

**Files:**

- Create: `tldw_chatbook/Research_Workspace/deep_research_bridge.py`
- Extend: `tldw_chatbook/Research_Workspace/overlay_store.py`
- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Modify: `tldw_chatbook/UI/Screens/research_workspace_screen.py`
- Modify: `tldw_chatbook/UI/Screens/research_screen.py`
- Test: `Tests/Research_Workspace/test_deep_research_bridge.py`
- Test: `Tests/Research_Workspace/test_overlay_store.py`
- Test: `Tests/UI/test_research_workspace_runs_bridge.py`

1. Add RED tests for Local/Server launch identity, source versions, initiating
   output/message, normalized query, authority-specific conversation field,
   route/timestamp, overlay bound, restart, run mismatch, origin mismatch,
   preview-only return, idempotent import, explicit new version, and stale UI.
2. Define `ResearchLaunchContext` with `launch_id`, qualified origin,
   source/version snapshot, initiating owner ref, normalized query,
   `local_conversation_id` XOR `server_chat_id`, return route, and timestamp.
3. Pass the payload-free context through `ResearchScopeService.create_run(
   mode=..., query=..., chat_handoff=...)`. The returned run ID and backend
   become the authoritative run identity; do not copy run status/events into
   WorkspaceDB.
4. Extend private overlay schema v4 with a bounded
   `deep_research_launches` map keyed by qualified workspace and run identity.
   It stores return/navigation context only and receives the device-only
   disclosure. Migrate v1-v3 records without inventing launch state.
5. Add typed `RESEARCH_RUN_TARGET` and `RESEARCH_WORKSPACE_RETURN` handoffs.
   Launch stages the exact backend/run ID and opens the existing real Runs
   screen. Runs' `Return to Workspace` stages run/origin identity and returns
   to `research_workspace` without replacing the Runs screen.
6. Workspace calls `ResearchScopeService.get_bundle(mode=..., run_id=...)`,
   verifies backend, run ID, launch ID, and origin, and presents a bounded
   bundle summary/Report preview. No owner write occurs at this stage.
7. Confirmation creates a draft Report through TASK-21510's Studio adapter
   with `research_run_ref` provenance. Re-import returns the existing output;
   only explicit `Create new version` creates another owner.
8. Preserve meaningful Runs states (plan/source review, lease elsewhere,
   resume, partial evidence) by rendering the existing normalized Research
   records, not translating them into Workspace statuses.
9. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_deep_research_bridge.py Tests/Research_Workspace/test_overlay_store.py Tests/UI/test_research_workspace_runs_bridge.py Tests/Research_Interop
   ```

10. Commit:

   ```bash
   git commit -m "feat: bridge Research Workspace and Runs"
   ```

## Task 7: Prove restart/recovery contracts and close TASK-21511

**Files:**

- Create: `Tests/integration/test_research_copy_restart_round_trip.py`
- Create: `Tests/integration/test_research_deep_return_round_trip.py`
- Create: `Tests/integration/test_research_server_sharing_contract.py`
- Modify: `backlog/tasks/task-21511 - Add-Research-Copy-sharing-and-Deep-Research-return.md`

1. Add real temporary-DB restart tests that interrupt Copy after canonical
   write, association, local receipt, and server chunk receipt; every restart
   must resume without duplicate destination items.
2. Add server recorded-contract tests for migration create/status/chunk/
   finalize, sharing permissions/link/clone, and no client-delete-ack.
3. Add Local and Server Deep Research launch -> Runs -> bundle -> preview ->
   confirmed draft Report round trips, including mismatch and re-import.
4. Run targeted Task 1-6 tests, these integration files, and
   `git diff --check`. Do not claim the full suite.
5. Review every spec Copy/sharing/Deep Research requirement, scan for
   placeholders, and compare all type/method names to the implementation.
6. Check TASK-21511 acceptance criteria only with captured evidence, add brief
   Implementation Notes, and set Done only after the repository Definition of
   Done is satisfied.
7. Commit exact files only:

   ```bash
   git commit -m "test: prove Research Copy and return recovery"
   ```
