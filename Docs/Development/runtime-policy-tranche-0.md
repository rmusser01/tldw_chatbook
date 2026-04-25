# Runtime Policy Tranche 0
## tldw_chatbook, April 21, 2026

## Scope

This tranche establishes the first enforceable runtime-policy surface for `tldw_chatbook`.

- authoritative runtime source state now lives in `tldw_chatbook/runtime_policy/`
- `active_source` and `active_server_id` are app-authoritative; restored screen/tab state is contextual only
- raw shared-client and chatbook-service construction is intentionally confined to `tldw_chatbook/runtime_policy/bootstrap.py`
- phase-one UI callers now prefer runtime-policy authority over saved screen-local backend state

## Landed Hard-Stop Seams

- `NotesScopeService` for local notes, server notes, and workspace note routing
- `ServerNotesWorkspaceService` as a policy-enforced server boundary for notes/workspaces
- `MediaReadingScopeService` for local/server media and ingestion-source actions
- `StudyScopeService` for deck/flashcard/review routing with deck-level flashcard policy proxying
- `QuizScopeService` for quiz/question/attempt routing with question-mutation proxying
- `EvaluationScopeService` for local/server evaluation routing
- `RAGAdminScopeService` for local/server retrieval admin routing
- `CharacterPersonaScopeService` for local/server character, persona, greeting, and preset routing

## Approved Raw-Client Boundary

The only approved raw construction boundary in this tranche is `tldw_chatbook/runtime_policy/bootstrap.py`.

It now owns:

- `build_runtime_api_client(...)`
- `build_runtime_api_client_from_config(...)`
- `build_server_chatbook_service(...)`

Representative former bypasses that now route through that boundary:

- `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- `tldw_chatbook/Event_Handlers/tldw_api_events.py`
- `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`
- `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`
- server-facing wrappers that previously imported chatbook-local client helpers directly

## Shared Unsupported-Capability Contract

Later parity work added `tldw_chatbook/runtime_policy/unsupported_capabilities.py` as the shared validation and collection seam for source-scoped unsupported capability reports.

It now owns:

- `validate_unsupported_capability_report(...)`
- `collect_unsupported_capability_reports(...)`
- `UnsupportedCapabilityReportError`

The contract requires each report item to provide `operation_id`, `source`, `supported=False`, `reason_code`, `user_message`, and `affected_action_ids`. Affected action IDs are validated against the authoritative runtime-policy registry, so future UI panes can render local/server/workspace gaps without each screen redefining report shape or accepting stale action IDs.

## Representative UI Callers

The tranche intentionally does not migrate every UI caller. It lands representative restore-precedence and preflight behavior in the most important screen-level seams:

- `StudyScreen` now prefers the app-authoritative runtime source over stale screen/app backend fields
- `CCPScreen` now resolves launched chat runtime from the authoritative runtime-policy accessor first
- `ChatScreen` restore keeps per-tab runtime metadata for identity and reuse without mutating app runtime authority
- `NotesScreen` now routes representative server-note/workspace load paths through `NotesScopeService` instead of bypassing policy with raw server service calls
- `StudyFlashcardsController` and `StudyQuizzesController` now perform representative UI preflight before dispatching constrained scope-service actions
- `TldwCli.handle_runtime_backend_changed(...)` now avoids surprise persistence for lightweight app-like objects that do not yet have runtime policy initialized, while preserving authoritative persistence in the real app

## Verification Matrix

### Phase-One Focused Matrix

Focused phase-one runtime-policy matrix run on April 21, 2026:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/RuntimePolicy/test_runtime_policy_core.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/RuntimePolicy/test_boundary_guards.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Notes/test_notes_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Notes/test_server_notes_workspace_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Media/test_media_reading_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Study_Interop/test_study_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Study_Interop/test_quiz_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Evaluations_Interop/test_evaluation_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/RAG_Admin/test_rag_admin_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Character_Chat/test_character_persona_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_evaluation_browser_screen.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_chat_screen_state.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_notes_screen.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_study_screen.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_ccp_screen.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/tldw_api/test_client_error_classification.py
```

Result:

- `216 passed`
- `1 warning`
- runtime: `17.72s`

### Broader Cross-Domain Smoke Slice

Broader smoke slice covering the touched source-aware domains:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/pytest -q \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Notes/test_notes_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Notes/test_server_notes_workspace_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Media/test_media_reading_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Study_Interop/test_study_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Study_Interop/test_quiz_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/Evaluations_Interop/test_evaluation_scope_service.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_notes_screen.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_study_screen.py \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/Tests/UI/test_evaluation_browser_screen.py
```

Result:

- `152 passed`
- `1 warning`
- runtime: `11.09s`

### Earlier Combined Tranche Sweep

An earlier broader tranche verification run also completed successfully during this work:

- `238 passed`
- `1 warning`
- runtime: `21.68s`

Known warning across these runs:

- `requests` dependency compatibility warning from the project virtual environment:
  `urllib3 (2.6.3) or chardet (6.0.0dev0)/charset_normalizer (3.4.4) doesn't match a supported version`

## Still Deferred

This tranche intentionally does not claim full runtime-policy migration.

Explicitly deferred:

- full-screen migration of every remaining UI caller to runtime-policy-first helpers
- broad chat/full-screen migration beyond representative restore-precedence behavior
- full local/remote sync semantics and dual-write replication
- broader server/workflow/governance UI surfaces beyond the registry and representative preflight seams
- commit/merge handoff work for this tranche

## Handoff Summary

What landed:

- hard-stop policy seams for notes, media, study, quiz, evaluations, rag admin, and character/persona
- authoritative runtime-source ownership in `runtime_policy/`
- representative UI restore-precedence behavior in Chat, Notes, Study, and CCP
- representative UI preflight in Study flashcards/quizzes
- strict boundary guard for raw client/service construction

What remains:

- migrate more UI callers off legacy direct runtime/backend reads
- extend representative preflight into additional screens and handlers
- finish the deferred chat/full-screen runtime-policy normalization work
- continue the broader server-parity roadmap on top of this authority layer
