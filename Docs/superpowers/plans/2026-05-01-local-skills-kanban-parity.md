# Local Skills And Kanban Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete local/offline Skills and Kanban backends behind the existing source-aware scope services.

**Architecture:** Preserve the existing server services and scope-service API while adding `LocalSkillsService` and `LocalKanbanService` as first-class local backends. Move Skills and Kanban from remote-only runtime policy contracts to local/server separated contracts, then implement local persistence and operation parity in small tested slices.

**Tech Stack:** Python 3.11+, pytest, pydantic schemas from `tldw_chatbook/tldw_api`, SQLite, JSON metadata files, existing runtime-policy registry and scope-service patterns.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-05-01-local-skills-kanban-parity-design.md`
- Runtime policy registry: `tldw_chatbook/runtime_policy/registry.py`
- Domain edge contracts: `tldw_chatbook/runtime_policy/domain_edge_contracts.py`
- Skills scope/service: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Kanban operation map: `tldw_chatbook/Kanban_Interop/server_kanban_service.py`
- Kanban scope service: `tldw_chatbook/Kanban_Interop/kanban_scope_service.py`

## Repo Conventions

- Use the repo venv: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.
- Test paths use `Tests/`, not `tests/`.
- Keep UI redesign out of this tranche.
- Preserve `mode=None` defaulting to server for backwards compatibility.
- Do not read or write Codex runtime skills under `~/.codex/skills`.
- Use deterministic local `ValueError` codes for local domain errors.
- Commit after each task when tests pass.

## File Structure

- Create `tldw_chatbook/Skills_Interop/local_skills_service.py`: Chatbook-owned local skill library with metadata extraction, atomic writes, single-writer mutation serialization, import/export, execution rendering, and seed behavior.
- Modify `tldw_chatbook/Skills_Interop/skills_scope_service.py`: add `local_service`, source-aware action IDs, local routing, and local unsupported reports.
- Modify `tldw_chatbook/Skills_Interop/__init__.py`: export `LocalSkillsService`.
- Create `Tests/Skills/test_local_skills_service.py`: persistence, metadata, concurrency, import/export, execution, seed, and validation tests.
- Modify `Tests/Skills/test_skills_scope_service.py`: local routing, local policy IDs, local unsupported reports, server compatibility.
- Create `tldw_chatbook/Kanban_Interop/local_kanban_db.py`: SQLite connection, schema migration, foreign keys, indexes, FTS setup/fallback, and transaction helpers.
- Create `tldw_chatbook/Kanban_Interop/local_kanban_service.py`: complete local implementation of `KANBAN_OPERATION_SPECS`.
- Modify `tldw_chatbook/Kanban_Interop/kanban_scope_service.py`: add `local_service`, source-aware dispatch, local unsupported reports, and backend normalization.
- Modify `tldw_chatbook/Kanban_Interop/__init__.py`: export `LocalKanbanService`.
- Create `Tests/Kanban/test_local_kanban_service.py`: generated operation coverage plus behavior tests for every operation family.
- Modify `Tests/Kanban/test_kanban_scope_service.py`: local routing and local policy IDs.
- Modify `tldw_chatbook/runtime_policy/registry.py`: move Skills and Kanban capabilities to `SEPARATED_SOURCES`.
- Modify `tldw_chatbook/runtime_policy/domain_edge_contracts.py`: expose canonical `skills` and `kanban` as local/server domains and make `server_skills` a compatibility alias without duplicate matrix rows.
- Modify `Tests/RuntimePolicy/test_domain_edge_contracts.py`: local/server domain contract tests and alias tests.
- Modify `Tests/RuntimePolicy/test_unsupported_capabilities.py`: local unsupported report expectations.
- Modify `tldw_chatbook/app.py`: instantiate local services and pass them to scope services.
- Modify `Tests/UI/test_screen_navigation.py`: update app-wiring assertions.
- Modify `Docs/superpowers/trackers/backend-parity-phase-tracker.md`: mark Skills and Kanban local parity implementation status after code lands.

## Task 1: Runtime Policy And Domain Contract Seam

**Files:**
- Modify: `tldw_chatbook/runtime_policy/registry.py`
- Modify: `tldw_chatbook/runtime_policy/domain_edge_contracts.py`
- Test: `Tests/RuntimePolicy/test_domain_edge_contracts.py`
- Test: `Tests/RuntimePolicy/test_unsupported_capabilities.py`

- [ ] **Step 1: Write failing domain contract tests**

Add tests that prove canonical `skills` and `kanban` expose local/server selector states and are not remote-only.

```python
def test_skills_and_kanban_are_local_server_domains():
    from tldw_chatbook.runtime_policy.domain_edge_contracts import (
        build_domain_capability_matrix,
        get_domain_edge_contract,
        get_remote_utility_local_parity,
    )

    skills = get_domain_edge_contract("skills")
    kanban = get_domain_edge_contract("kanban")

    assert skills.authority == "local_and_server"
    assert skills.source_selector_states == ("local", "server")
    assert kanban.authority == "local_and_server"
    assert kanban.source_selector_states == ("local", "server")

    matrix = build_domain_capability_matrix()
    assert "skills" in matrix
    assert "kanban" in matrix
    assert "server_skills" not in matrix
    assert get_remote_utility_local_parity("skills").state == "planned"
    assert get_remote_utility_local_parity("kanban").state == "planned"
```

Add an alias test:

```python
def test_server_skills_alias_returns_canonical_skills_contract():
    from tldw_chatbook.runtime_policy.domain_edge_contracts import get_domain_edge_contract

    assert get_domain_edge_contract("server_skills") is get_domain_edge_contract("skills")
```

- [ ] **Step 2: Write failing policy registry tests**

Add tests proving local action IDs exist for representative Skills and Kanban operations.

```python
def test_skills_and_kanban_have_local_policy_actions():
    from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY

    for action_id in [
        "skills.list.local",
        "skills.execute.launch.local",
        "kanban.boards.list.local",
        "kanban.cards.create.local",
        "kanban.card_links.delete.local",
    ]:
        entry = CAPABILITY_REGISTRY[action_id]
        assert entry.required_source == "local"
        assert entry.authority_owner == "local"
```

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/RuntimePolicy/test_domain_edge_contracts.py \
  Tests/RuntimePolicy/test_unsupported_capabilities.py -q
```

Expected: fail because Skills and Kanban are still remote-only.

- [ ] **Step 4: Implement registry and contract changes**

In `registry.py`, change the `kanban_boards_tasks` and `server_skills` capabilities from `REMOTE_ONLY_SOURCES` to `SEPARATED_SOURCES`. Preserve the `server_skills` capability id unless changing it is required by tests.

In `domain_edge_contracts.py`:

- Remove `server_skills` and `kanban` from `REMOTE_ONLY_DOMAIN_IDS`.
- Add `skills` and `kanban` explicit `DomainEdgeContract` entries with `authority="local_and_server"`.
- Replace `server_skills` with canonical `skills` in `REMOTE_UTILITY_DOMAIN_IDS`; keep `kanban` in the utility parity list but not in the remote-only domain list.
- Make `get_domain_edge_contract("server_skills")` return the canonical `skills` contract without listing a duplicate row in `build_domain_capability_matrix()`.
- Set remote utility parity for `skills` and `kanban` to `planned` until implementation tasks land.

- [ ] **Step 5: Verify and commit**

Run the command from Step 3 again.

Expected: PASS.

Commit:

```bash
git add tldw_chatbook/runtime_policy/registry.py \
  tldw_chatbook/runtime_policy/domain_edge_contracts.py \
  Tests/RuntimePolicy/test_domain_edge_contracts.py \
  Tests/RuntimePolicy/test_unsupported_capabilities.py
git commit -m "feat: add local policy contracts for skills and kanban"
```

## Task 2: Local Skills Service

**Files:**
- Create: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `tldw_chatbook/Skills_Interop/__init__.py`
- Test: `Tests/Skills/test_local_skills_service.py`

- [ ] **Step 1: Write failing persistence and metadata tests**

Create `Tests/Skills/test_local_skills_service.py` with tests for create/list/get/reload and metadata extraction.

```python
@pytest.mark.asyncio
async def test_local_skills_service_persists_skill_metadata(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    content = """---
description: Summarize notes
argument_hint: note id
allowed_tools:
  - notes.read
model: local-model
context: fork
user_invocable: true
disable_model_invocation: false
---
# Summarize
Summarize {{args}}.
"""

    created = await service.create_skill(name="summarize-notes", content=content)
    reloaded = LocalSkillsService(store_dir=tmp_path)
    loaded = await reloaded.get_skill("summarize-notes")

    assert created["version"] == 1
    assert loaded["description"] == "Summarize notes"
    assert loaded["argument_hint"] == "note id"
    assert loaded["allowed_tools"] == ["notes.read"]
    assert loaded["model"] == "local-model"
    assert loaded["context"] == "fork"
```

- [ ] **Step 2: Write failing concurrency and version tests**

Add tests proving `expected_version` conflicts and concurrent updates serialize.

```python
@pytest.mark.asyncio
async def test_local_skills_service_serializes_concurrent_updates(tmp_path):
    service = LocalSkillsService(store_dir=tmp_path)
    await service.create_skill(name="demo-skill", content="# Demo\nInitial")

    await asyncio.gather(
        service.update_skill("demo-skill", content="# Demo\nA"),
        service.update_skill("demo-skill", supporting_files={"a.md": "A"}),
    )

    loaded = await service.get_skill("demo-skill")
    assert loaded["version"] == 3
    assert loaded["supporting_files"]["a.md"] == "A"
```

- [ ] **Step 3: Write failing import/export/execute/seed tests**

Cover plain import, file import, zip export or equivalent server-compatible payload, execution rendering, and empty seed behavior.

- [ ] **Step 4: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_local_skills_service.py -q
```

Expected: fail because `LocalSkillsService` does not exist.

- [ ] **Step 5: Implement `LocalSkillsService`**

Implement:

- `__init__(store_dir: str | Path, policy_enforcer=None)`
- metadata index loading and atomic persistence.
- YAML front matter parser using only existing dependencies or a small safe parser.
- single-writer mutation lock.
- skill directory path validation.
- generated response dictionaries compatible with `skills_schemas.py`.
- import/export methods.
- `execute_skill` that returns rendered prompt only.
- deterministic empty seed behavior if no packaged local seeds exist.

Use this public shape:

```python
class LocalSkillsService:
    def __init__(self, *, store_dir: str | Path, policy_enforcer: Any | None = None) -> None: ...
    async def list_skills(self, *, include_hidden: bool = False, limit: int = 100, offset: int = 0) -> dict[str, Any]: ...
    async def get_context(self) -> dict[str, Any]: ...
    async def get_skill(self, skill_name: str) -> dict[str, Any]: ...
    async def create_skill(self, *, name: str, content: str, supporting_files: dict[str, str] | None = None) -> dict[str, Any]: ...
    async def update_skill(self, skill_name: str, *, content: str | None = None, supporting_files: dict[str, str | None] | None = None, expected_version: int | None = None) -> dict[str, Any]: ...
    async def delete_skill(self, skill_name: str, *, expected_version: int | None = None) -> bool: ...
    async def import_skill(self, *, content: str, name: str | None = None, supporting_files: dict[str, str] | None = None, overwrite: bool = False) -> dict[str, Any]: ...
    async def import_skill_file(self, file_content: bytes, *, filename: str = "SKILL.md", content_type: str = "text/markdown", overwrite: bool = False) -> dict[str, Any]: ...
    async def export_skill(self, skill_name: str) -> Any: ...
    async def execute_skill(self, skill_name: str, *, args: str | None = None) -> dict[str, Any]: ...
    async def seed_builtin_skills(self, *, overwrite: bool = False) -> dict[str, Any]: ...
```

- [ ] **Step 6: Verify and commit**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_local_skills_service.py -q
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook diff --check
```

Expected: PASS and clean diff check.

Commit:

```bash
git add tldw_chatbook/Skills_Interop/local_skills_service.py \
  tldw_chatbook/Skills_Interop/__init__.py \
  Tests/Skills/test_local_skills_service.py
git commit -m "feat: add local skills service"
```

## Task 3: Skills Scope Routing And App Wiring

**Files:**
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Skills/test_skills_scope_service.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write failing scope routing tests**

Add tests proving local mode routes to local service, rewrites records to `backend="local"`, enforces `.local` action IDs, and does not dispatch to server.

```python
@pytest.mark.asyncio
async def test_skills_scope_service_routes_local_operations():
    local = FakeSkillsService()
    server = FakeSkillsService()
    policy = FakePolicyEnforcer()
    scope = SkillsScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    result = await scope.list_skills(mode="local")

    assert result["backend"] == "local"
    assert local.calls == ["list_skills"]
    assert server.calls == []
    assert policy.calls == ["skills.list.local"]
```

Update unsupported-report tests:

- local mode with a local service returns no broad remote-only report.
- local mode without a local service returns `skills.local_backend_unavailable`.
- server mode remains unchanged.

- [ ] **Step 2: Write failing app wiring test**

Update `Tests/UI/test_screen_navigation.py` to assert `app.local_skills_service` exists and `app.skills_scope_service` has both local and server services.

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_skills_scope_service.py \
  Tests/UI/test_screen_navigation.py -q
```

Expected: fail because scope/app wiring is still server-only.

- [ ] **Step 4: Implement scope routing**

Update `SkillsScopeService`:

- accept `local_service`.
- keep `mode=None` defaulting to server.
- route local mode through local service.
- derive local action ids by replacing terminal `.server` with `.local` or use explicit local action constants.
- normalize local records using existing `_normalize_response(..., SkillsBackend.LOCAL)`.
- return honest local unsupported capability reports when local service is missing.

- [ ] **Step 5: Wire app startup**

In `app.py`, instantiate:

```python
self.local_skills_service = LocalSkillsService(
    store_dir=get_user_data_dir() / "skills",
    policy_enforcer=self.service_policy_enforcer,
)
```

Pass `local_service=self.local_skills_service` into `SkillsScopeService`.

- [ ] **Step 6: Verify and commit**

Run the Step 3 command plus `Tests/Skills/test_local_skills_service.py`.

Expected: PASS.

Commit:

```bash
git add tldw_chatbook/Skills_Interop/skills_scope_service.py \
  tldw_chatbook/app.py \
  Tests/Skills/test_skills_scope_service.py \
  Tests/UI/test_screen_navigation.py
git commit -m "feat: route skills local backend"
```

## Task 4: Local Kanban Database Foundation

**Files:**
- Create: `tldw_chatbook/Kanban_Interop/local_kanban_db.py`
- Create: `tldw_chatbook/Kanban_Interop/local_kanban_service.py`
- Modify: `tldw_chatbook/Kanban_Interop/__init__.py`
- Test: `Tests/Kanban/test_local_kanban_service.py`

- [ ] **Step 1: Write failing schema migration tests**

Create tests that instantiate `LocalKanbanService(db_path=tmp_path / "kanban.db")`, inspect SQLite tables, and verify foreign keys are enabled.

- [ ] **Step 2: Write failing service construction tests**

Add tests proving `LocalKanbanService(db_path=...)` opens the database, exposes helper methods for transactions, and does not require a server client. Do not add operation coverage tests in this task.

- [ ] **Step 3: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Kanban/test_local_kanban_service.py -q
```

Expected: fail because local Kanban storage files do not exist.

- [ ] **Step 4: Implement database foundation**

In `local_kanban_db.py`, implement:

- `open_connection(db_path)`.
- schema version table.
- migration to create required Kanban tables.
- foreign keys and indexes.
- transaction context manager.
- FTS availability detection and fallback flag.

In `local_kanban_service.py`, implement:

- constructor.
- `_enforce(action_id)`.
- `_now()`, UUID generation, pagination helpers.
- database lifecycle helpers.

Do not commit public no-op operation methods. Operation methods are introduced in the tasks that implement their behavior.

- [ ] **Step 5: Verify and commit**

Run the Step 3 command.

Expected: schema and service construction tests pass.

Commit:

```bash
git add tldw_chatbook/Kanban_Interop/local_kanban_db.py \
  tldw_chatbook/Kanban_Interop/local_kanban_service.py \
  tldw_chatbook/Kanban_Interop/__init__.py \
  Tests/Kanban/test_local_kanban_service.py
git commit -m "feat: add local kanban storage foundation"
```

## Task 5: Local Kanban Core CRUD And Activities

**Files:**
- Modify: `tldw_chatbook/Kanban_Interop/local_kanban_service.py`
- Modify: `tldw_chatbook/Kanban_Interop/local_kanban_db.py`
- Test: `Tests/Kanban/test_local_kanban_service.py`

- [ ] **Step 1: Write failing board/list/card tests**

Cover:

- create/list/get/update/archive/unarchive/delete/restore boards.
- create/list/get/update/archive/unarchive/delete/restore lists.
- create/list/get/update/archive/unarchive/delete/restore cards.
- reload persistence.
- version increment behavior.

- [ ] **Step 2: Write failing reorder/move/copy tests**

Cover:

- `reorder_lists`.
- `reorder_cards`.
- `move_card`.
- `copy_card`.
- position normalization after moves and deletes.

- [ ] **Step 3: Write failing activity tests**

Assert mutating operations create activity rows in the same transaction and activity listing respects board/card filters.

- [ ] **Step 4: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Kanban/test_local_kanban_service.py -q
```

Expected: fail because core local Kanban operations are not implemented.

- [ ] **Step 5: Implement core CRUD**

Implement board/list/card operations with generated schema validation from `tldw_chatbook/tldw_api/kanban_schemas.py`. Return dictionaries shaped like the generated response models and compatible with `ServerKanbanService._normalize_response`.

Write activities inside the same transaction as the mutation.

- [ ] **Step 6: Verify and commit**

Run the Step 4 command.

Expected: PASS for core CRUD/activity tests.

Commit:

```bash
git add tldw_chatbook/Kanban_Interop/local_kanban_service.py \
  tldw_chatbook/Kanban_Interop/local_kanban_db.py \
  Tests/Kanban/test_local_kanban_service.py
git commit -m "feat: add local kanban core records"
```

## Task 6: Local Kanban Subresources, Bulk Ops, Search, Import, Export, Links

**Files:**
- Modify: `tldw_chatbook/Kanban_Interop/local_kanban_service.py`
- Modify: `tldw_chatbook/Kanban_Interop/local_kanban_db.py`
- Test: `Tests/Kanban/test_local_kanban_service.py`

- [ ] **Step 1: Write failing label/checklist/comment tests**

Cover label CRUD, card label assignment/removal/listing, checklist CRUD, item CRUD, item reorder, check/uncheck, toggle all, comment CRUD.

- [ ] **Step 2: Write failing bulk operation tests**

Cover bulk move/archive/unarchive/delete/label with transaction rollback on invalid card ids.

- [ ] **Step 3: Write failing import/export tests**

Export a nested board with labels, lists, cards, checklists/items, comments, and links. Import it into a fresh DB and assert ids are remapped and stats are correct.

- [ ] **Step 4: Write failing search tests**

Cover:

- `search_cards_basic`.
- `search_cards_basic_get`.
- `search_cards`.
- `search_cards_get`.
- `filter_board_cards`.
- `get_search_status`.
- vector/hybrid local degradation metadata.

- [ ] **Step 5: Write failing card link tests**

Cover add/list/count/remove by all variants, bulk add/remove, duplicate handling, and list cards by linked content.

- [ ] **Step 6: Write failing generated operation coverage test**

Add a generated coverage test after all operation families have behavior tests. This test must fail until every `KANBAN_OPERATION_SPECS` operation has a real implementation.

```python
def test_local_kanban_service_exposes_every_operation():
    from tldw_chatbook.Kanban_Interop.local_kanban_service import LocalKanbanService
    from tldw_chatbook.Kanban_Interop.server_kanban_service import KANBAN_OPERATION_SPECS

    service = LocalKanbanService(db_path=":memory:")

    missing = [name for name in KANBAN_OPERATION_SPECS if not callable(getattr(service, name, None))]

    assert missing == []
```

- [ ] **Step 7: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Kanban/test_local_kanban_service.py -q
```

Expected: fail for unimplemented advanced operations or missing operation methods.

- [ ] **Step 8: Implement advanced operations**

Implement all remaining `KANBAN_OPERATION_SPECS` operations. Keep generated operation coverage green throughout. Use transactions for bulk and import operations.

- [ ] **Step 9: Verify and commit**

Run the Step 6 command plus diff check.

Expected: PASS.

Commit:

```bash
git add tldw_chatbook/Kanban_Interop/local_kanban_service.py \
  tldw_chatbook/Kanban_Interop/local_kanban_db.py \
  Tests/Kanban/test_local_kanban_service.py
git commit -m "feat: complete local kanban operation parity"
```

## Task 7: Kanban Scope Routing And App Wiring

**Files:**
- Modify: `tldw_chatbook/Kanban_Interop/kanban_scope_service.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/Kanban/test_kanban_scope_service.py`
- Modify: `Tests/UI/test_screen_navigation.py`

- [ ] **Step 1: Write failing scope dispatch tests**

Add a generated scope dispatch test.

```python
@pytest.mark.asyncio
async def test_kanban_scope_service_dispatches_every_operation_locally():
    local = FakeLocalKanbanService()
    server = ExplodingServerKanbanService()
    policy = FakePolicyEnforcer()
    scope = KanbanScopeService(local_service=local, server_service=server, policy_enforcer=policy)

    for operation_name in KANBAN_OPERATION_SPECS:
        await scope.invoke(operation_name, *minimal_args_for(operation_name), mode="local", **minimal_kwargs_for(operation_name))

    assert all(action_id.endswith(".local") for action_id in policy.calls)
```

Use a minimal fake local service that returns valid generic dictionaries. The goal is dispatch coverage; behavior is covered by `test_local_kanban_service.py`.

- [ ] **Step 2: Write failing unsupported report tests**

Assert local mode with `local_service` has no broad `kanban.remote_only.local` report, while server mode keeps the workflow-control unsupported report.

- [ ] **Step 3: Write failing app wiring test**

Update `Tests/UI/test_screen_navigation.py` to assert `app.local_kanban_service` exists and `app.kanban_scope_service` has both local and server services.

- [ ] **Step 4: Run tests and verify failure**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Kanban/test_kanban_scope_service.py \
  Tests/UI/test_screen_navigation.py -q
```

Expected: fail because scope/app wiring is still server-only.

- [ ] **Step 5: Implement scope routing and app wiring**

Update `KanbanScopeService`:

- accept `local_service`.
- keep `mode=None` defaulting to server.
- route local mode through local service.
- enforce source-aware action ids.
- normalize local responses with `backend="local"` and local record ids.
- preserve server workflow-control unsupported report.

In `app.py`, instantiate:

```python
self.local_kanban_service = LocalKanbanService(
    db_path=get_user_data_dir() / "tldw_chatbook_kanban.db",
    policy_enforcer=self.service_policy_enforcer,
)
```

Pass `local_service=self.local_kanban_service` into `KanbanScopeService`.

- [ ] **Step 6: Verify and commit**

Run the Step 4 command plus `Tests/Kanban/test_local_kanban_service.py`.

Expected: PASS.

Commit:

```bash
git add tldw_chatbook/Kanban_Interop/kanban_scope_service.py \
  tldw_chatbook/app.py \
  Tests/Kanban/test_kanban_scope_service.py \
  Tests/UI/test_screen_navigation.py
git commit -m "feat: route kanban local backend"
```

## Task 8: Tracker Closeout And Focused Verification

**Files:**
- Modify: `Docs/superpowers/trackers/backend-parity-phase-tracker.md`
- Optional modify: `Docs/superpowers/handoffs/2026-04-30-backend-parity-ux-handoff.md`

- [ ] **Step 1: Update tracker rows**

Update Skills and Kanban rows to reflect local/server source selectors and implementation evidence commits. Keep sync/outbox work explicitly out of scope.

- [ ] **Step 2: Run focused verification**

Run:

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Skills/test_local_skills_service.py \
  Tests/Skills/test_skills_scope_service.py \
  Tests/Kanban/test_local_kanban_service.py \
  Tests/Kanban/test_kanban_scope_service.py \
  Tests/RuntimePolicy/test_domain_edge_contracts.py \
  Tests/RuntimePolicy/test_unsupported_capabilities.py \
  Tests/UI/test_screen_navigation.py -q
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook diff --check
```

Expected: all selected tests pass and diff check is clean.

- [ ] **Step 3: Commit docs and verification evidence**

Commit:

```bash
git add Docs/superpowers/trackers/backend-parity-phase-tracker.md \
  Docs/superpowers/handoffs/2026-04-30-backend-parity-ux-handoff.md
git commit -m "docs: mark local skills kanban parity ready"
```

Only include the handoff file if it exists and was updated.

## Final Verification

Run the full focused suite from Task 8 and then:

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook status --short --branch
```

Expected:

- Focused tests pass.
- `git diff --check` passes.
- Working tree is clean.
- Branch contains one commit per task or a clear equivalent sequence.
