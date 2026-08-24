# Research Workspace Primary Studio Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans`; apply `superpowers:test-driven-development`
> before production changes and `superpowers:verification-before-completion`
> before commits. Do not delegate unless the user explicitly requests
> subagents. Apply `impeccable` immediately before UI implementation tasks.

**Goal:** Add Summary, Flashcards, Quiz, Report, and Compare Sources as
traceable Studio outputs that persist in and reopen from their existing
canonical owners.

**Architecture:** `ResearchStudioService` captures one immutable qualified
workspace/source/version/generation snapshot, generates without tools, and
routes persistence through owner-specific adapters. Content, versions, and
deletion remain owned by Local Chatbooks/Study/Quiz or the corresponding
server workspace services; WorkspaceDB stores only the membership and bounded
generation provenance needed to project those owners back into Studio.

**Tech Stack:** Python 3.11+, Textual 8.x, WorkspaceDB SQLite migrations,
LocalChatbookService, StudyScopeService, QuizScopeService,
NotesScopeService/ServerNotesWorkspaceService, existing provider/RAG seams,
pytest.

**Spec:**
`Docs/superpowers/specs/2026-08-23-research-workspace-design.md`

**Backlog:** `TASK-21510` (depends on `TASK-21507`, `TASK-21508`,
`TASK-21509`)

## Global constraints

- The five primary output cards are Summary, Flashcards, Quiz, Report, and
  Compare Sources; Compare requires at least two selected ready sources.
- No parallel Research output table may own canonical content, version,
  listing, or deletion.
- Local textual outputs are Local Chatbooks; Local Flashcards are Study
  deck/cards; Local Quiz is Quiz record/questions.
- Server textual outputs are workspace artifacts; Server Flashcards and Quiz
  use workspace-scoped Study/Quiz contracts.
- `OutputsScopeService(mode="local")` is explicitly unavailable and must not be
  used or made to pretend otherwise.
- Generation uses no ToolCatalog, MCP/ACP, approvals, or autonomous loop.
- Every generation records selected source IDs/versions, configuration,
  processing route, provider/model, owner ID/version, and terminal status.
- Unsupported lifecycle actions are disabled with an exact reason. Never
  simulate replace/version/delete locally for a server owner that lacks it.
- Unknown capability fails closed; no silent Local/Server fallback or blend.
- No full-suite run/claim without explicit user approval.

## ADR check

ADR required: no new ADR

ADR path:
`backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-078 already assigns each canonical owner and forbids a duplicate
output store. Workspace membership provenance is relationship metadata, not a
new content owner.

## Owner field mapping (implementation contract)

| Output | Authority | Canonical owner and ID | Owner version | Provenance/config | Workspace projection | Reopen | Replace/new version/delete |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Summary, Report, Compare | Local | `LocalChatbookService`, `chatbook_id` | `artifact_revision` | bounded `WorkspaceMembership.metadata["research_generation"]`; citation ownership stays in Chatbook | `item_type="chatbook"`, `role="output"` | Artifacts handoff targeting `local:chatbook:<id>` | update Chatbook for replace; create Chatbook for new version; delete through Chatbook then unlink membership |
| Flashcards | Local | `LocalStudyService`, deck `id` plus card IDs | deck/card `version` | bounded output membership metadata; card citations use card `metadata` where supported | `item_type="study_deck"`, `role="output"` | Study handoff to deck section and exact deck ID | update/rebuild only through Study owner; new deck for new version; delete through Study then unlink |
| Quiz | Local | `LocalQuizService`, quiz `id` plus question IDs | quiz/question `version` | bounded output membership metadata; question citations use `source_citations` | `item_type="study_quiz"`, `role="output"` | Study handoff to Quiz section and exact quiz ID | update only through Quiz owner; new quiz for new version; delete through Quiz then unlink |
| Summary, Report, Compare | Server | `NotesScopeService.save_workspace_artifact`, `artifact_id`; wire `artifact_type` is respectively `summary`, `report`, or `compare_sources` | workspace artifact `version` | server artifact content envelope plus payload-free local receipt fields | server workspace artifact row is the association | server artifact detail from Research/Artifacts | update with exact version; create a new artifact ID for new version; server delete |
| Flashcards | Server | `StudyScopeService`, deck/card IDs with `scope_type="workspace"` | returned deck/card versions | server owner fields plus bounded local receipt | server deck `workspace_id` | Study server/workspace handoff | only methods/capabilities exposed by Study owner |
| Quiz | Server | `QuizScopeService`, quiz/question IDs with `scope_type="workspace"` | returned quiz/question versions | server owner fields plus bounded local receipt | server quiz `workspace_id` | Study Quiz server/workspace handoff | only methods/capabilities exposed by Quiz owner |

The implementation must encode this table as data and test it; prose alone is
not sufficient. Membership metadata is capped and secret/body-free except for
the canonical owner reference: it may contain source refs, generation option
hashes, route/provider/model labels, terminal state, and owner revision, but no
generated content or source excerpts.

## Task 1: Define output mapping, capability, and immutable generation contracts

**Files:**

- Create: `tldw_chatbook/Research_Workspace/studio_models.py`
- Create: `tldw_chatbook/Research_Workspace/studio_mapping.py`
- Extend: `tldw_chatbook/Research_Workspace/contracts.py`
- Test: `Tests/Research_Workspace/test_studio_models.py`
- Test: `Tests/Research_Workspace/test_studio_mapping.py`

1. Add RED tests for all ten authority/output mappings, Compare's two-ready-
   source gate, unknown capability, mismatched workspace/source authority,
   duplicate owner IDs, stale controller revision, metadata bounds, and
   content/secret rejection.
2. Define the fixed contracts:

   ```python
   class ResearchOutputKind(StrEnum):
       SUMMARY = "summary"
       FLASHCARDS = "flashcards"
       QUIZ = "quiz"
       REPORT = "report"
       COMPARE_SOURCES = "compare_sources"

   @dataclass(frozen=True, slots=True)
   class ResearchStudioRequest:
       operation_id: str
       workspace: QualifiedWorkspaceRef
       output_kind: ResearchOutputKind
       source_snapshot: tuple[ResearchSourceVersionRef, ...]
       generation: ResearchGenerationConfig
       processing_route: ProcessingRoute
       context_revision: int

   @dataclass(frozen=True, slots=True)
   class WorkspaceOutputRef:
       workspace: QualifiedWorkspaceRef
       output_kind: ResearchOutputKind
       owner_kind: str
       owner_id: str
       owner_version: int | None
       source_snapshot: tuple[ResearchSourceVersionRef, ...]
       generation_fingerprint: str
       status: Literal["complete", "cancelled", "failed"]
   ```

3. Encode `PRIMARY_OUTPUT_MAPPINGS` with owner kind, membership item type,
   create/update/delete capability IDs, reopen destination/channel, version
   field, and supported actions for every row in the mapping table.
4. Extend `ResearchWorkspacePort` with forward-typed `list_outputs(ref, page)`,
   `resolve_output(output_ref)`, `generate_output(request, on_chunk,
   cancel_event)`, `cancel_output(ref, operation_id)`, `retry_output(ref,
   operation_id)`, and `export_output(output_ref, format)` methods. The Local
   and Server adapter extensions in Tasks 3-4 own their respective dispatch.
5. Implement `project_studio_availability(...)` so every card returns
   `available`, `owner_label`, `processing_route`, `reason_code`, `recovery`,
   and `capability_revision`. Never infer availability solely from mode.
6. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_studio_models.py Tests/Research_Workspace/test_studio_mapping.py
   ```

7. Commit:

   ```bash
   git commit -m "feat: define Research Studio owner mappings"
   ```

## Task 2: Persist bounded output provenance on workspace relationships

**Files:**

- Create: `tldw_chatbook/DB/migrations/workspaces_v3_to_v4_output_metadata.sql`
- Modify: `tldw_chatbook/DB/Workspace_DB.py`
- Modify: `tldw_chatbook/Workspaces/models.py`
- Modify: `tldw_chatbook/Workspaces/registry_service.py`
- Test: `Tests/DB/test_workspace_db_v4_migration.py`
- Test: `Tests/Workspaces/test_workspace_registry_output_metadata.py`

1. Add RED migration tests from WorkspaceDB v3 and fresh schema, plus registry
   tests for idempotent link/update, secret/body rejection, 32 KiB JSON cap,
   output-role-only metadata, and unlink preserving the canonical owner.
2. Add nullable `metadata_json TEXT` and `updated_at TEXT` to
   `workspace_memberships` in WorkspaceDB v4. Do not add content, source text,
   or a second output row. Keep the inline runner and
   `workspaces_v3_to_v4_output_metadata.sql` aligned.
3. Extend `WorkspaceMembership` with a scrubbed bounded `metadata` mapping and
   `updated_at`. Preserve compatibility for existing source memberships with
   empty metadata.
4. Add `WorkspaceRegistryService.upsert_membership_metadata(membership,
   expected_updated_at=None)` with optimistic conflict reporting. Keep
   `link_membership` idempotent and use the explicit unlink method delivered by
   TASK-21508.
5. Store `research_generation` with operation ID, output kind, source/version
   refs, config fingerprint, route/provider/model labels, terminal status, and
   current owner revision only. Generated text remains in its owner.
6. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/DB/test_workspace_db_v4_migration.py Tests/Workspaces/test_workspace_registry_output_metadata.py
   ```

7. Commit:

   ```bash
   git commit -m "feat: add workspace output provenance metadata"
   ```

## Task 3: Implement generation orchestration and Local owner adapters

**Files:**

- Create: `tldw_chatbook/Research_Workspace/studio_generation.py`
- Create: `tldw_chatbook/Research_Workspace/local_studio.py`
- Extend: `tldw_chatbook/Research_Workspace/local_adapter.py`
- Create: `tldw_chatbook/Research_Workspace/chatbook_artifact_payload.py`
- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py`
- Test: `Tests/Research_Workspace/test_studio_generation.py`
- Test: `Tests/Research_Workspace/test_local_studio.py`
- Test: `Tests/UI/Navigation/test_research_output_handoffs.py`

1. Add RED tests for prompt construction from a captured snapshot, no tools,
   Local egress consent, cancellation/failure, Summary/Report/Compare Chatbook
   creation, deck/cards, quiz/questions, partial owner rollback, idempotent
   retry, replace, new version, delete, and exact reopen handoffs.
2. Implement `ResearchStudioGenerator.generate(request, *, on_chunk,
   cancel_event)` on the same route/retrieval/provider primitives delivered by
   TASK-21509, with output-specific bounded JSON schemas for Flashcards and
   Quiz. Reject schema-invalid model output without persisting a half owner.
3. Implement `build_research_chatbook_payload(...)` analogous to
   `console_chatbook_artifact_payload`, including full generated text only in
   the Chatbook's canonical metadata/body field and membership provenance only
   in WorkspaceDB.
4. Implement `LocalStudioAdapter.save(...)` mappings exactly:

   - textual output -> `LocalChatbookService.create_chatbook` then
     `link_membership(item_type="chatbook", role="output")`;
   - Flashcards -> `LocalStudyService.create_deck` then
     `create_flashcards_bulk`, then `item_type="study_deck"` membership;
   - Quiz -> `LocalQuizService.create_quiz` then `create_question` for every
     validated question, then `item_type="study_quiz"` membership.

5. When a child write fails, delete the newly created empty/partial canonical
   owner where the owner supports safe deletion and record the sanitized
   failure. If cleanup fails, preserve and surface the owner ID for recovery;
   never hide an orphan behind a success receipt.
6. Add typed handoffs `RESEARCH_CHATBOOK_TARGET`,
   `RESEARCH_STUDY_DECK_TARGET`, and `RESEARCH_QUIZ_TARGET`; destination screens
   claim/acknowledge the exact target instead of relying on list ordering.
7. Replace updates the exact owner with optimistic version where supported;
   new version creates a new owner and sets `supersedes_owner_ref` in bounded
   membership metadata. Deletion calls the owner first, then unlinks.
8. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_studio_generation.py Tests/Research_Workspace/test_local_studio.py Tests/UI/Navigation/test_research_output_handoffs.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: save Research Studio outputs locally"
   ```

## Task 4: Implement Server owner adapters without client-side masquerading

**Files:**

- Create: `tldw_chatbook/Research_Workspace/server_studio.py`
- Extend: `tldw_chatbook/Research_Workspace/server_adapter.py`
- Reuse: `tldw_chatbook/Study_Interop/study_scope_service.py`
- Reuse: `tldw_chatbook/Study_Interop/quiz_scope_service.py`
- Test: `Tests/Research_Workspace/test_server_studio.py`

1. Add RED tests for exact server profile/principal/workspace qualification,
   workspace artifact persistence, workspace Study/Quiz scope, capability
   block, owner version conflict, stale UI completion, retry idempotency, and
   no Local service calls.
2. Persist textual outputs through
   `NotesScopeService.save_workspace_artifact(workspace_id=..., artifact_id=...,
   artifact_type=..., title=..., status="complete", content=...)`. Generate
   artifact IDs client-side as UUIDs so retries reuse the same owner ID.
3. Persist Flashcards through `StudyScopeService.create_deck(...,
   workspace_id=...)` and owner card methods; persist Quiz through
   `QuizScopeService.create_quiz(..., workspace_id=...)` and question methods.
4. Treat the server response as the only proof of persistence. A downloaded
   file, local preview, or local receipt never flips `saved=True`.
5. Use exact owner versions for replace/delete. If the server contract lacks a
   requested lifecycle method, return the mapped unsupported capability and
   leave the owner untouched.
6. Server generation uses the server processing path established in
   TASK-21509; it never sends server source bodies to a client-selected Local
   or third-party provider.
7. Run:

   ```bash
   .venv/bin/python -m pytest -q Tests/Research_Workspace/test_server_studio.py Tests/Study_Interop/test_study_scope_service.py Tests/Study_Interop/test_quiz_scope_service.py
   ```

8. Commit:

   ```bash
   git commit -m "feat: save Research Studio outputs on server"
   ```

## Task 5: Build Studio cards, canonical history, and lifecycle actions

**Files:**

- Modify: `tldw_chatbook/UI/Research_Workspace_Modules/studio_region.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/studio_output_card.py`
- Create: `tldw_chatbook/UI/Research_Workspace_Modules/studio_history.py`
- Modify: `tldw_chatbook/UI/Screens/research_workspace_screen.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
- Modify: `tldw_chatbook/css/features/_research_workspace.tcss`
- Test: `Tests/UI/test_research_studio_pane.py`
- Test: `Tests/UI/test_research_studio_history.py`
- Test: `Tests/UI/test_research_workspace_geometry.py`

1. Add mounted RED tests for five card labels/buttons, explicit disabled
   reasons, Compare count gate, progress/cancel/retry, saved receipt, source-
   changed warning, owner-derived history, and action visibility by owner
   capability.
2. Mount card controls once and patch availability/progress/result in place.
   Card activation opens a small configuration form with selected sources,
   route, format/detail controls, and exact destination owner.
3. Build Studio history by resolving `role="output"` memberships against the
   Local owner, or listing real server workspace artifacts/decks/quizzes. A
   missing owner renders `Missing canonical item` with unlink recovery; it is
   never reconstructed from membership metadata.
4. Wire View/edit/export, replace, new version, Discuss in Grounded Chat,
   save/append to Quick Notes, delete, undo where the owner supports it, and
   reopen through typed handoffs. Disable unsupported actions with reason.
5. Compare current source versions to the stored source snapshot and show
   `Sources changed since generation`; inspection remains available and
   regeneration captures a fresh snapshot.
6. Preserve pane preference/focus and active generation state across responsive
   reflow; a non-durable active generation blocks authority/workspace switch
   with cancel recovery.
7. Rebuild CSS and prove all five outputs, progress, receipts, and canonical
   history are reachable at 160x40, 120x30, 100x30, 84x24, 80x24, and 60x20.
8. Run:

   ```bash
   .venv/bin/python tldw_chatbook/css/build_css.py
   .venv/bin/python -m pytest -q Tests/UI/test_research_studio_pane.py Tests/UI/test_research_studio_history.py Tests/UI/test_research_workspace_geometry.py
   ```

9. Commit:

   ```bash
   git commit -m "feat: add Research Studio primary output UI"
   ```

## Task 6: Prove canonical owner round trips and close TASK-21510

**Files:**

- Create: `Tests/integration/test_research_studio_local_round_trip.py`
- Create: `Tests/integration/test_research_studio_server_contract.py`
- Modify: `backlog/tasks/task-21510 - Add-primary-Research-Studio-outputs.md`

1. Add a real temporary-DB Local round trip for each output kind: generate,
   canonical owner read, workspace projection, reopen target, replace/new
   version, source-change warning, and canonical delete/unlink order.
2. Add recorded-contract Server tests proving workspace artifact/Study/Quiz
   request paths, workspace IDs, owner versions, no Local calls, and no false
   persistence receipt.
3. Run the targeted suite from Tasks 1-5 plus both integration files and
   `git diff --check`. Do not claim the full suite.
4. Review the spec section-by-section, scan this plan for placeholders, and
   compare all method/type names against the implementation.
5. Check every TASK-21510 acceptance criterion only when supported by the
   captured test output; add concise Implementation Notes and set the task to
   Done only after the repository Definition of Done is satisfied.
6. Commit exact task/code/test files only:

   ```bash
   git commit -m "test: prove Research Studio owner round trips"
   ```
