# Local Skills And Kanban Parity Design

Date: 2026-05-01

## Purpose

Add full local/offline equivalents for Skills and Kanban so both domains can be used without a configured or reachable server. The implementation must preserve the current source-aware scope-service pattern: UI and callers keep using `SkillsScopeService` and `KanbanScopeService`, while `mode="local"` routes to local services and `mode="server"` continues to route to active-server services.

This is full operation-surface parity, not a partial pilot. The work should still be sequenced in safe implementation slices so registry, policy, storage, and app wiring do not drift.

## Current State

Skills and Kanban are currently classified as remote-only surfaces:

- `SkillsScopeService` rejects local mode and only dispatches to `ServerSkillsService`.
- `KanbanScopeService` rejects local mode and only dispatches to `ServerKanbanService`.
- Runtime policy declares the Skills and Kanban capabilities with `REMOTE_ONLY_SOURCES`.
- Domain edge contracts include `server_skills` and `kanban` in the remote-only set.
- App wiring only constructs server-backed services for these domains.

The server-side wrappers and generated schemas already define the operation contracts that local services must match:

- Skills: list, context, detail, create, update, delete, import, import file, export, execute, seed built-ins.
- Kanban: boards, lists, cards, activities, import/export, search, bulk card operations, labels, card labels, checklists, checklist items, comments, and card links.

## Non-Goals

- No UI redesign.
- No background sync, mutation outbox, or local-to-server replay.
- No mutation of Codex runtime skills under `~/.codex/skills`.
- No hidden model invocation during local skill execution.
- No in-app local HTTP server just to emulate server routes.
- No second active-server authority or server capability registry changes beyond source availability.

## Design Issues To Guard Against

### Source-Aware Policy IDs

The current operation specs are server-biased. Local parity must not enforce `.server` policy actions when a local backend handles the request.

The spec requires source-aware action IDs:

- Skills local operations enforce `skills.*.local` action IDs.
- Kanban local operations enforce `kanban.*.local` action IDs.
- Server operations keep enforcing existing `.server` action IDs.
- Runtime policy moves Skills and Kanban from `REMOTE_ONLY_SOURCES` to `SEPARATED_SOURCES`.
- Tests must prove local calls do not require server source state and server calls still enforce server source state.

For Kanban, either add local operation specs or derive action IDs from the existing `KANBAN_OPERATION_SPECS` by replacing the terminal `.server` suffix with `.local`. The implementation must not mutate the server spec in place.

### Canonical Domain Names

Runtime policy uses domain `skills`; older domain edge/tracker language uses `server_skills`. The canonical domain for new code is `skills`.

Compatibility rules:

- Runtime policy capability id may remain `server_skills` if changing it would churn unrelated tests.
- Domain edge contract should expose canonical `skills`.
- Remove `server_skills` from new remote-only and remote-utility rows once canonical `skills` is added.
- If existing callers still request `get_domain_edge_contract("server_skills")`, the compatibility path should return the canonical `skills` contract rather than creating a second domain row.
- New local service modules should be named `local_skills_service.py` and `local_kanban_service.py`.

### Honest Full Parity

Full parity means every public method in the existing scope services has a local implementation or a source-specific unsupported report for an objectively impossible local behavior. It does not mean local behavior must be identical to a multi-user server for fields that are inherently server-owned.

Local services must document deterministic local semantics for:

- Local principal/user id.
- Local timestamps and version increments.
- Soft delete, hard delete, archive, restore, and retention.
- Optimistic version conflicts.
- Search mode degradation.
- Import/export format compatibility.
- Activity generation.
- Skill seeding and execution.

## Local Skills Design

### Storage

Add `LocalSkillsService`, backed by a Chatbook-owned local skill library under `get_user_data_dir()`.

Recommended structure:

- Metadata index: `tldw_chatbook_skills.json`.
- Skill directories: `skills/<skill-name>/SKILL.md` plus supporting files.
- Atomic writes for metadata and file contents.
- A service-level single-writer lock around metadata and supporting-file mutations.
- Path safety validation for skill names and supporting files.
- Version increments on every mutation.
- `expected_version` conflicts reported as deterministic `ValueError` codes, matching existing local service patterns.

The local library must not read from or write to `~/.codex/skills`. A future import command can explicitly import from Codex skills, but this tranche only owns Chatbook's local skill library.

All local skill mutations must run through the service instance so metadata and file contents cannot diverge. The implementation may use an `asyncio.Lock`, thread lock, or narrow synchronous write section, but tests must prove concurrent updates do not lose a version increment or overwrite a newer supporting-file change.

### Contract

`LocalSkillsService` implements the same public methods as `ServerSkillsService`:

- `list_skills(include_hidden=False, limit=100, offset=0)`
- `get_context()`
- `get_skill(skill_name)`
- `create_skill(name, content, supporting_files=None)`
- `update_skill(skill_name, content=None, supporting_files=None, expected_version=None)`
- `delete_skill(skill_name, expected_version=None)`
- `import_skill(content, name=None, supporting_files=None, overwrite=False)`
- `import_skill_file(file_content, filename="SKILL.md", content_type="text/markdown", overwrite=False)`
- `export_skill(skill_name)`
- `execute_skill(skill_name, args=None)`
- `seed_builtin_skills(overwrite=False)`

Use the generated Skills schema models for validation wherever possible.

### Metadata Extraction

Local Skills must expose the same metadata fields as `SkillResponse`, `SkillSummary`, `SkillContextPayload`, and `SkillExecutionResult`. Metadata comes from a normalized Chatbook-owned envelope, not from ad hoc UI defaults.

Extraction rules:

- Prefer explicit metadata stored in the local metadata index.
- On create/import, parse optional YAML front matter at the top of `SKILL.md` for `description`, `argument_hint`, `allowed_tools`, `model`, `context`, `user_invocable`, and `disable_model_invocation`.
- If front matter is absent, preserve caller-provided metadata when the import/create API is extended to accept it; otherwise use deterministic defaults matching `SkillBase`.
- `description` may fall back to the first non-heading paragraph in `SKILL.md`, truncated to schema limits.
- `argument_hint` has no heuristic fallback; absent means `None`.
- `allowed_tools` defaults to `None`.
- `model` defaults to `None`.
- `context` defaults to `inline`.
- `user_invocable` defaults to `True`.
- `disable_model_invocation` defaults to `False`.

The local metadata index is authoritative after import. Updating `SKILL.md` content must not silently drop metadata fields that were already stored. If front matter changes during an update, the implementation must either explicitly re-parse and persist the changed fields or document that metadata is updated only through a dedicated future API. The first implementation should re-parse front matter on content update because that keeps local file edits and service responses consistent.

`get_context()` must build `available_skills` from this normalized metadata and must generate `context_text` deterministically from skill name, description, and argument hint. `execute_skill()` must return `allowed_tools`, `model_override`, and `execution_mode` from the same metadata source.

### Import And Export

Local import/export must be format-compatible with server expectations at the service boundary:

- Plain `SKILL.md` import creates or overwrites one skill.
- Archive import must be supported when the input matches the server export format; path traversal entries must be rejected.
- Export returns a payload usable by the existing caller contract. If server export returns bytes plus filename, local export should do the same.
- Supporting files keep the same file-count, filename, per-file size, and total-size limits as `skills_schemas.py`.

### Execution

Local `execute_skill` renders the prompt contract only:

- Return `skill_name`, `rendered_prompt`, `allowed_tools`, `model_override`, and `execution_mode`.
- Substitute or append `args` consistently with the server's rendered prompt behavior.
- Do not invoke a model.
- Do not run tools.
- Do not fork a chat session.

### Built-In Seeds

`seed_builtin_skills` should seed from a Chatbook-owned packaged directory if one exists. If no packaged local seeds exist, it should return a successful empty result with `seeded=[]` and `count=0`, not call the server or Codex runtime skill directories.

## Local Kanban Design

### Storage

Add `LocalKanbanService`, backed by a local SQLite database such as `get_user_data_dir() / "tldw_chatbook_kanban.db"`.

The database must include migrations and schema versioning. Required tables:

- `kanban_boards`
- `kanban_lists`
- `kanban_cards`
- `kanban_labels`
- `kanban_card_labels`
- `kanban_checklists`
- `kanban_checklist_items`
- `kanban_comments`
- `kanban_activities`
- `kanban_card_links`
- optional FTS tables for card search

Storage rules:

- Enable foreign keys.
- Use indexes for board/list/card lookup, active records, linked content, labels, and search.
- Use transactions for every mutation that affects more than one table.
- Bulk operations are all-or-nothing unless an operation explicitly returns skipped counts.
- Activity rows are written in the same transaction as the mutation that created them.
- Soft delete and archive must preserve enough data for restore operations.

### Local Identity Semantics

Server responses include `user_id`, integer ids, UUIDs, and client ids. Local responses should preserve the same shape:

- Use monotonically increasing integer ids per table.
- Generate stable UUIDs for local records.
- Use `user_id="local"` unless a configured local profile id exists.
- Preserve caller-provided `client_id`.
- Increment `version` on every update, archive, restore, move, reorder, delete, and bulk mutation.

### Operation Coverage

`LocalKanbanService` implements every operation in `KANBAN_OPERATION_SPECS`:

- Board CRUD, archive, unarchive, delete, restore, import, export.
- List CRUD, archive, unarchive, delete, restore, reorder.
- Card CRUD, move, copy, archive, unarchive, delete, restore, reorder.
- Board and card activity listing.
- Basic card search, full search, and search status.
- Bulk move, archive, unarchive, delete, and label mutations.
- Board card filtering.
- Copy card with checklists and optional labels.
- Label CRUD and card-label assignment/removal/listing.
- Checklist CRUD, reorder, item CRUD, item reorder, check/uncheck, toggle all.
- Comment CRUD.
- Card link add/list/count/remove/bulk add/bulk remove/list-by-linked-content.

The local service should use the generated Kanban schema models for request validation and response-compatible dictionaries.

Operation coverage is a hard gate. Add a generated parity test that iterates every key in `KANBAN_OPERATION_SPECS` and proves:

- `LocalKanbanService` exposes a callable with the same operation name.
- `KanbanScopeService(local_service=...)` dispatches each operation in `mode="local"`.
- Local dispatch does not call `server_service`.
- The local action id used for policy enforcement ends in `.local`.

The generated coverage test is not a replacement for behavior tests. It prevents dynamic-dispatch drift; behavior tests still prove each operation's semantics.

### Activity Semantics

Every mutating local Kanban operation creates a local activity record:

- `action_type` should use stable verbs such as `create`, `update`, `archive`, `restore`, `delete`, `move`, `copy`, `reorder`, `label`, `comment`, `link`, `import`, and `export`.
- `entity_type` should identify the mutated entity.
- `entity_id`, `board_id`, `list_id`, and `card_id` should be populated when known.
- `details` should include before/after identifiers for moves, reorder ids, bulk counts, and import/export metadata.

Activity retention follows the board's `activity_retention_days` field. Listing activities prunes expired local activities before returning results.

### Import And Export

Local board export should produce the existing `KanbanBoardExportResponse` shape:

- `format`
- `exported_at`
- `board`
- `labels`
- `lists`

Export must include nested cards, checklists/items, comments, labels, and card links when they belong to the board. Archive/deleted inclusion follows `KanbanBoardExportRequest`.

Local board import should accept the same exported shape and create a new local board unless an explicit future overwrite option is added. Import stats must count imported lists, cards, labels, checklists, checklist items, and comments. Imported local ids must be remapped so card labels, checklists, comments, and links point at new local rows.

### Search Semantics

Local search should be honest about capabilities:

- `search_mode="fts"` uses SQLite FTS when available, falling back to deterministic `LIKE` search if FTS is unavailable.
- `search_mode="hybrid"` and `search_mode="vector"` may degrade to FTS/LIKE in local mode, but responses must include metadata such as `local_search_degraded=True` and `effective_search_mode="fts"` or `"like"`.
- `get_search_status` returns local index state, FTS availability, last index rebuild timestamp if tracked, and any degradation behavior.

If the generated response schema does not currently expose metadata, local responses may include extra fields because the schemas use `extra="allow"` where needed. Tests should assert these fields are present for degraded modes.

### Card Links

Local card links are scoped to linked content types already accepted by the schema: `media` and `note`.

Rules:

- `(card_id, linked_type, linked_id)` is unique.
- Duplicate add returns the existing link or increments skipped count for bulk add.
- Removing by card plus linked content, by card plus link id, and by link id must all work.
- Listing cards by linked content returns card records with board/list names and link metadata.

## Scope Services And App Wiring

### SkillsScopeService

Update constructor:

```python
SkillsScopeService(local_service=None, server_service=None, policy_enforcer=None)
```

Routing:

- `mode=None` keeps the existing default server behavior for backwards compatibility. Callers must pass `mode="local"` to use local Skills until a separate active-source-default tranche changes this.
- `mode="local"` requires `local_service`.
- `mode="server"` requires `server_service`.
- Local responses are normalized with `backend="local"` and `record_id="local:..."`.
- Server responses keep existing normalization behavior.

Local unsupported reports should be empty when `local_service` is configured. If no local service is configured, report `skills.local_backend_unavailable` with reason code `capability_missing` only when the selected local backend is genuinely absent.

### KanbanScopeService

Update constructor:

```python
KanbanScopeService(local_service=None, server_service=None, policy_enforcer=None)
```

Routing:

- `mode=None` keeps the existing default server behavior for backwards compatibility. Callers must pass `mode="local"` to use local Kanban until a separate active-source-default tranche changes this.
- `mode="local"` requires `local_service`.
- `mode="server"` requires `server_service`.
- Local responses reuse `ServerKanbanService._normalize_response(...)` and then rewrite `backend` to `local`.
- Unsupported reports should keep the existing server workflow-control report, but remove the broad local remote-only report once local service is configured.

### App Wiring

App startup should instantiate:

- `self.local_skills_service`
- `self.local_kanban_service`
- existing `self.server_skills_service`
- existing `self.server_kanban_service`
- scope services with both local and server services

Startup failure of a local store should be logged and should result in an unsupported local report, not an app crash.

## Domain And Tracker Contracts

Update domain edge contracts:

- Remove canonical `skills` and `kanban` entries from remote-only contracts.
- Add `skills` with `authority="local_and_server"` and source selector states `("local", "server")`.
- Add `kanban` with `authority="local_and_server"` and source selector states `("local", "server")`.
- Keep `server_skills` only as a compatibility alias if existing tracker rows require it.

Update remote utility local parity:

- `skills`: `state="planned"` until implementation lands, then `state="pilot"` using the current `LocalParityState` vocabulary.
- `kanban`: `state="planned"` until implementation lands, then `state="pilot"` using the current `LocalParityState` vocabulary.

`pilot` means "implemented local adapter exists" in the current vocabulary. It does not reduce the operation-coverage target for this work.

Update tracker rows so UX can see both domains have local/server source selectors and stable backend contracts.

## Error Handling

Use deterministic local error codes in `ValueError` messages, following existing local services:

- `local_skill_not_found:<name>`
- `local_skill_version_conflict:<name>`
- `local_skill_exists:<name>`
- `local_kanban_board_not_found:<id>`
- `local_kanban_list_not_found:<id>`
- `local_kanban_card_not_found:<id>`
- `local_kanban_version_conflict:<entity>:<id>`
- `local_kanban_integrity_error:<reason>`

Policy denial should continue to raise `PolicyDeniedError` from the policy layer.

## Test Strategy

Use TDD for behavior changes.

Skills tests:

- Local service CRUD persists across service reload.
- Version conflicts block stale updates/deletes.
- Concurrent mutations are serialized and cannot lose version or supporting-file updates.
- Metadata extraction covers YAML front matter, deterministic defaults, preserved metadata on content update, `get_context()`, and `execute_skill()`.
- Supporting file validation and path traversal rejection.
- Import/export round trip.
- Execute renders a prompt and does not invoke models or tools.
- Seed built-ins returns deterministic results.
- Scope service routes local and server modes with source-specific policy IDs.

Kanban tests:

- Generated operation-coverage test iterates every `KANBAN_OPERATION_SPECS` key and proves local service plus local scope dispatch exists.
- SQLite schema migration creates required tables and indexes.
- Board/list/card CRUD persists across service reload.
- Archive/delete/restore semantics match response contracts.
- Reorder, move, copy, and copy-with-checklists preserve positions and relationships.
- Bulk operations are transactional.
- Labels, checklists, comments, and links cover create/list/detail/update/delete paths.
- Activity records are written in the same transaction as mutations.
- Import/export round trip preserves nested board structure with remapped ids.
- Search supports FTS or deterministic fallback and reports local degradation for vector/hybrid modes.
- Scope service routes local and server modes with source-specific policy IDs.
- Domain edge contracts no longer classify canonical Skills or Kanban as remote-only.

Focused verification:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest \
  Tests/Skills/test_local_skills_service.py \
  Tests/Skills/test_skills_scope_service.py \
  Tests/Kanban/test_local_kanban_service.py \
  Tests/Kanban/test_kanban_scope_service.py \
  Tests/RuntimePolicy/test_domain_edge_contracts.py \
  Tests/RuntimePolicy/test_unsupported_capabilities.py -q
```

## Implementation Sequence

1. Registry and domain contract seam:
   - Add separated local/server policy actions for Skills and Kanban.
   - Canonicalize the domain contract around `skills` and `kanban`.
   - Add failing tests that prove local calls require local policy actions.

2. Local Skills:
   - Add `LocalSkillsService`.
   - Update `SkillsScopeService` routing.
   - Wire local Skills in app startup.
   - Add persistence, import/export, execution, and seed tests.

3. Kanban schema and core local CRUD:
   - Add `LocalKanbanService` with migrations.
   - Implement boards, lists, cards, labels, checklists, comments, and activities.
   - Update `KanbanScopeService` routing.

4. Advanced Kanban operations:
   - Implement import/export, search, bulk operations, copy-with-checklists, and card links.
   - Add transactional and round-trip tests.

5. App wiring and tracker closeout:
   - Instantiate local services in app startup.
   - Update tests that currently assert server-only wiring.
   - Update the backend parity tracker and UX handoff rows.

## Acceptance Criteria

- `mode="local"` works for every Skills method without a server.
- `mode="local"` works for every Kanban operation in `KANBAN_OPERATION_SPECS` without a server.
- `mode="server"` behavior and policy enforcement are unchanged.
- Local and server responses preserve stable `backend` and `record_id` contracts.
- Runtime policy and domain edge contracts no longer describe canonical Skills or Kanban as remote-only.
- Local stores are Chatbook-owned, persistent, path-safe, transactional where needed, and covered by focused tests.
