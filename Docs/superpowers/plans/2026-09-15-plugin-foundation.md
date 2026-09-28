# Managed plugin foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install, trust, activate, use, update and remove a native local plugin with authenticated recovery and scoped runtime ownership.

**Architecture:** Keep immutable package inspection separate from the private registry and protected authority store. One coordinator publishes authority and existing Console/Skills seams consume leased run snapshots; scoped live fences stop work independently of durable writes.

**Tech Stack:** Python 3.12+, Textual 8.2.8, Pydantic 2, private SQLite, httpx, portalocker and existing trust/credential primitives.

**Spec:** [Plugin spec](../specs/2026-09-15-managed-plugins-design.md) and [hook spec](../specs/2026-09-15-expanded-hook-runtime-design.md). Read both; this plan covers its assigned subsystem within the complete [delivery plan](2026-09-15-managed-plugins-delivery.md).

## Global Constraints

- Python >=3.12; current checkout pins Textual 8.2.8, Pydantic >=2.4,<3 and portalocker 3.2.0. Preserve these pins; use the existing SQLite/httpx/crypto/keyring seams.
- One installation and selected revision exist per user-data directory.
- Global default activation starts disabled. Importing a catalog does not install or enable its entries.
- All approved hook additions are in scope. Delivery stages are ordering, not deferral.
- No parallel agent/permission runtime, package build/install execution, vendor grants or per-workspace package versions.
- Required constraints never disappear because parsing, configuration, hooks or persistence fail. Native/foreign instructions remain attributed untrusted context.
- Package activation grants no filesystem binding, tool permission, network credential or trusted project status. Console local tools retain scratch/explicit-binding authority.
- Apply current authority before injection/launch/dispatch/result acceptance. Workspace disable preserves other authorized scopes; namespace marker changes alone do not cancel all work.
- Full-suite runs require explicit user opt-in. Every task runs its exact feature/regression files and a successful control on the same production entry.
- Implementation uses an isolated execution worktree and profile; do not repoint a shared editable environment. Verify child interpreter package provenance as well as pytest cwd.
- Every To Do task must move In Progress and receive its Implementation Plan via Backlog CLI before code changes. Add Implementation Notes and mark Done only after its acceptance criteria, review, targeted tests and static checks pass.

ADR required: yes
ADR paths: [ADR-162](../../../backlog/decisions/162-managed-agent-plugins.md); [ADR-163](../../../backlog/decisions/163-expanded-console-hook-runtime.md)
Reason: Implements the accepted storage, trust, runtime and UI contracts. No additional ADR is needed unless implementation changes one of those decisions.

---

## Execution and evidence

This is an implementation plan, not implemented code or passing runtime evidence.
The code blocks below are small invariant/RED-test sketches, not a complete
implementation to paste blindly. Each task must also exercise its named production
entry and failure/control matrix. Preserve the stated interfaces across tasks;
when current library behavior contradicts a sketch, establish the real RED failure
and correct the sketch/test before implementation, as the repository's
[testing-evidence lesson](../../../backlog/docs/lessons-testing-evidence.md) requires.

Read [live verification](../../../backlog/docs/lessons-live-verification.md) before
running the app. Isolate config, data, credential and child-process roots before
importing runtime code, and verify the isolation. Use sys.executable for controlled
children. A disabled path failing from an unrelated event-loop error is not evidence.

Each task is one independently reviewable deliverable. Its checklist is the
sequence of small test/implementation increments; repeat the RED/GREEN cycle for
each listed failure/control case. Do not implement the entire subsystem before
running its first integration test. File roles and public contracts below define
the decomposition; no unrelated broad refactoring is part of these plans.

## File ownership map

| Task | New implementation units | Existing integration boundaries |
| --- | --- | --- |
| F1 | `tldw_chatbook/Plugins/__init__.py`, `tldw_chatbook/Plugins/models.py`, `tldw_chatbook/Plugins/inspection.py`, `tldw_chatbook/Plugins/package_files.py`, `tldw_chatbook/Plugins/adapters/__init__.py`, `tldw_chatbook/Plugins/adapters/portable.py` | No existing runtime entry changed |
| F2 | `tldw_chatbook/Plugins/registry.py`, `tldw_chatbook/Plugins/runtime_owner.py`, `tldw_chatbook/Plugins/migrations/001_initial.sql` | `tldw_chatbook/DB/private_sqlite.py` |
| F3 | `tldw_chatbook/Plugins/authority.py`, `tldw_chatbook/Plugins/authority_store.py` | `tldw_chatbook/Skills_Interop/skill_trust_service.py`, `tldw_chatbook/Skills_Interop/skill_trust_crypto.py`, `tldw_chatbook/Skills_Interop/skill_trust_store.py` |
| F4 | `tldw_chatbook/Plugins/coordinator.py`, `tldw_chatbook/Plugins/recovery.py`, `tldw_chatbook/Plugins/review.py` | No existing runtime entry changed |
| F5 | `tldw_chatbook/Plugins/admission.py`, `tldw_chatbook/Plugins/skill_provider.py`, `tldw_chatbook/Plugins/context.py` | `tldw_chatbook/Skills_Interop/local_skills_service.py`, `tldw_chatbook/Skills_Interop/skills_scope_service.py`, `tldw_chatbook/Agents/tool_catalog.py`, `tldw_chatbook/Chat/console_skill_resolver.py`, `tldw_chatbook/Chat/console_runtime.py` |
| F6 | `tldw_chatbook/Plugins/revocation.py` | `tldw_chatbook/Plugins/coordinator.py`, `tldw_chatbook/Plugins/admission.py`, `tldw_chatbook/Plugins/runtime_owner.py`, `tldw_chatbook/Chat/console_runtime.py` |
| F7 | `tldw_chatbook/Plugins/revisions.py`, `tldw_chatbook/Plugins/retention.py` | `tldw_chatbook/Plugins/coordinator.py`, `tldw_chatbook/Plugins/runtime_owner.py` |
| F8 | `tldw_chatbook/Plugins/data_cleanup.py` | `tldw_chatbook/Plugins/authority.py`, `tldw_chatbook/Plugins/coordinator.py`, `tldw_chatbook/Plugins/runtime_owner.py` |

## F1: Inspect immutable native plugin packages

**Backlog:** [TASK-32668](../../../backlog/tasks/task-32668%20-%20Inspect-immutable-native-plugin-packages.md). **Requires:** [TASK-32645](../../../backlog/tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md).

**Deliverable:** Give users a bounded, inspectable native package inventory with stable identity and explicit compatibility blockers before any code can execute.

**Files:**

- Create: `tldw_chatbook/Plugins/__init__.py`
- Create: `tldw_chatbook/Plugins/models.py`
- Create: `tldw_chatbook/Plugins/inspection.py`
- Create: `tldw_chatbook/Plugins/package_files.py`
- Create: `tldw_chatbook/Plugins/adapters/__init__.py`
- Create: `tldw_chatbook/Plugins/adapters/portable.py`
- Create: `Tests/Plugins/__init__.py`
- Create: `Tests/Plugins/conftest.py`
- Test: `Tests/Plugins/test_native_inspection.py`
- Test: `Tests/Plugins/test_package_files.py`

**Interfaces**

- Consumes: Utils/path_validation.py and Utils/input_validation.py; the portable/core and extension schemas in plugin spec sections 3 and 10.
- Produces: inspect_package(root: Path, *, dialect: str | None = None) -> PackageInspection. PackageInspection and ComponentRecord are frozen Pydantic models in Plugins/models.py: candidate dialect/version, source/overlay identities, content/effective digests, inventory keyed by typed component ID, dependency edges, activation blockers and diagnostics. ComponentRecord exposes support, selection, availability and evidence separately. materialize_package(source: Path, destination: Path) -> PackageInspection returns only after bounded validation. Unknown constraints remain blockers, never empty edges.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_missing_required_hook_stays_a_blocker(native_package):
    from tldw_chatbook.Plugins.inspection import inspect_package
    package = native_package(requires={"skill:review": ["hook:missing"]})
    result = inspect_package(package)
    assert "skill:review" in result.inventory
    assert result.inventory["skill:review"].activation_blockers
    control = inspect_package(native_package())
    assert not control.inventory["skill:review"].activation_blockers
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_native_inspection.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
from pathlib import PurePosixPath

def validate_relative_member(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("package_path_invalid")
    return path
```

  - [ ] 3.1. Create native_package in Tests/Plugins/conftest.py: each call writes a fresh tmp_path child with the spec section 3.1 manifest, skills/review/SKILL.md and optional requires/extension overrides; it returns Path. Keep expected inventories authored independently of the parser.
  - [ ] 3.2. Implement closed native extension/frontmatter validation and deterministic candidate selection. Add fields for every declared component now; publish only skills in the initial integration. Enforce variable declarations, unknown dependency/cycle blocking and content versus effective identity.
  - [ ] 3.3. Implement streamed file-count/byte/depth accounting, executable-mode hashing and secure materialization. The lexical kernel is only an early rejection; resolved containment, link target type, reparse handling, case/Unicode collisions and destination identity remain required before read/write.
  - [ ] 3.4. Add a packaged test fixture manifest with provenance/license and run mutation cases for missing required hooks, malformed recognized extensions and valid unrelated unknown namespaces.

**Failure and successful-control matrix:** Malformed extension/version, unknown requires scope, two vendor candidates, unknown top-level portable fields, duplicate IDs, cyclic edges, escaping/internal links, executable bit drift, special files and every inspection limit. A parser-only result never establishes runtime compatibility.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_native_inspection.py Tests/Plugins/test_package_files.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32668 --plain
git diff --check
```

## F2: Store plugin registry state under one runtime owner

**Backlog:** [TASK-32669](../../../backlog/tasks/task-32669%20-%20Store-plugin-registry-state-under-one-runtime-owner.md). **Requires:** [TASK-32668](../../../backlog/tasks/task-32668%20-%20Inspect-immutable-native-plugin-packages.md).

**Deliverable:** Keep plugin metadata and runtime ownership isolated from conversation storage so concurrent app instances cannot mutate or execute the same installation.

**Files:**

- Create: `tldw_chatbook/Plugins/registry.py`
- Create: `tldw_chatbook/Plugins/runtime_owner.py`
- Create: `tldw_chatbook/Plugins/migrations/001_initial.sql`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`
- Modify: `pyproject.toml`
- Test: `Tests/Plugins/test_registry.py`
- Test: `Tests/Plugins/test_runtime_owner.py`

**Interfaces**

- Consumes: F1 immutable identities; connect_private_sqlite(owner_id, database, **kwargs); portalocker 3.2.0.
- Produces: PluginRegistry(path: Path) with schema_version: int, transaction(), list_installations(*, limit: int, offset: int) -> tuple[dict, ...], close(). PluginRuntimeOwner(root: Path).try_acquire() -> bool and close(). reserve_launch(operation_id: str, installation_id: str, workspace_id: str | None, revision_digest: str) -> str durably creates a launch token before execution; publish_process(token: str, provenance: dict) and settle_process(token: str, confirmed: bool) retain unresolved ownership. No execute path is exposed by this task.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_secondary_owner_cannot_execute(tmp_path):
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner
    first, second = PluginRuntimeOwner(tmp_path), PluginRuntimeOwner(tmp_path)
    try:
        assert first.try_acquire()
        assert not second.try_acquire()
    finally:
        first.close()
        second.close()
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_registry.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
SCHEMA_VERSION = 1
PAGE_SIZE = 50

SELECT_INSTALLATIONS = """
SELECT installation_id, revision_digest, activation_default
FROM installations ORDER BY installation_id LIMIT ? OFFSET ?
"""
```

  - [ ] 3.1. Register the private-SQLite owner plugins.registry and package the migration through pyproject.toml package data. Tables cover immutable revisions/components, activation, mappings, authority generations, operations, root ownership, process provenance and receipts; registry rows alone never confer trust.
  - [ ] 3.2. Implement schema creation/user_version and reopen validation in one transaction; test real in-memory SQLite for data behavior and isolated disk files for private path/WAL ownership.
  - [ ] 3.3. Acquire the profile-derived OS lock before mutation or execution; secondary owners receive read-only registry views. Release locks on close without clearing surviving-process evidence.
  - [ ] 3.4. Record pending launch identity before subprocess creation, publish exact process provenance afterward and leave unclean records across owner death. Add separate process tests for competing owners and PID reuse; do not infer cleanup from an acquired lock.

**Failure and successful-control matrix:** Independent subprocess contention, read-only secondary browse, owner death between reserve/spawn/publish, schema reopening/corruption, invalid network/synchronized roots and private DB owner inventory. Assert unrelated standalone functionality remains available.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_registry.py Tests/Plugins/test_runtime_owner.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32669 --plain
git diff --check
```

## F3: Authenticate complete plugin authority snapshots

**Backlog:** [TASK-32670](../../../backlog/tasks/task-32670%20-%20Authenticate-complete-plugin-authority-snapshots.md). **Requires:** [TASK-32669](../../../backlog/tasks/task-32669%20-%20Store-plugin-registry-state-under-one-runtime-owner.md).

**Deliverable:** Prevent package or registry tampering from changing activation, mappings or revocation without reviewed authority.

**Files:**

- Create: `tldw_chatbook/Plugins/authority.py`
- Create: `tldw_chatbook/Plugins/authority_store.py`
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_crypto.py`
- Modify: `tldw_chatbook/Skills_Interop/skill_trust_store.py`
- Test: `Tests/Plugins/test_authority.py`
- Test: `Tests/Plugins/test_authority_store.py`
- Test: `Tests/Skills/test_skill_trust_store.py`
- Test: `Tests/Skills/test_skill_trust_service.py`

**Interfaces**

- Consumes: F2 registry projections; existing passphrase-derived skill trust crypto, protected snapshots and keyring posture. The existing skill marker pair is not the new plugin marker tuple.
- Produces: PluginMarker is a frozen model with generation: int, operation_id: str, recovery_snapshot_digest: str. authority_message(purpose: str, payload: dict) -> bytes uses a closed purpose set. PluginAuthorityStore(store_dir: Path, marker_store: object) provides unlock(passphrase: str), prepare(snapshot: dict, old: PluginMarker, new: PluginMarker), certify_commit(old: PluginMarker, new: PluginMarker), advance_marker(old: PluginMarker, new: PluginMarker), verify_current() -> dict. Protected payload schema includes complete authority, root fences and references; methods validate and authenticate before returning.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_prepared_intent_is_not_commit_proof():
    from tldw_chatbook.Plugins.authority import authority_message
    payload = {"generation": 1, "operation_id": "op", "digest": "abc"}
    assert authority_message("prepared", payload) != authority_message("committed", payload)
    assert authority_message("prepared", payload) == authority_message("prepared", payload)
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_authority.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
from tldw_chatbook.Skills_Interop.skill_trust_crypto import canonical_json

PURPOSES = {"snapshot", "prepared", "committed"}

def authority_message(purpose: str, payload: dict) -> bytes:
    if purpose not in PURPOSES:
        raise ValueError("plugin_authority_purpose_invalid")
    return b"chatbook.plugins.v1\0" + purpose.encode("ascii") + b"\0" + canonical_json(payload)
```

  - [ ] 3.1. Add purpose-separated plugin trust keys/store identity without changing standalone keyring accounts, manifests or trust defaults. Reuse established authenticated encryption/MAC helpers; do not build another cryptographic primitive.
  - [ ] 3.2. Encode complete logical authority deterministically, distinguishing absence, Inherit and Disabled. Include root generations/fences and stable credential-binding references; exclude token bytes and volatile connection liveness.
  - [ ] 3.3. Persist encrypted snapshots, exact old/new marker intent and separate post-commit certificate with qualified fsync/replace helpers. Limit certificate creation to the coordinator call after a durable database commit.
  - [ ] 3.4. Exercise locked keyring, explicit reduced posture, reset, corrupt snapshot/MAC, same bytes with changed workspace override and replayed marker. Add successful authenticated load and standalone-skill controls.

**Failure and successful-control matrix:** Real crypto with an injected isolated marker backend; every authenticated field is tampered independently. Hash-only same-directory authority must never pass. Existing standalone trust reset/reopen behavior must remain unchanged.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_authority.py Tests/Plugins/test_authority_store.py Tests/Skills/test_skill_trust_store.py Tests/Skills/test_skill_trust_service.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32670 --plain
git diff --check
```

## F4: Commit and recover reviewed plugin installation changes

**Backlog:** [TASK-32671](../../../backlog/tasks/task-32671%20-%20Commit-and-recover-reviewed-plugin-installation-changes.md). **Requires:** [TASK-32670](../../../backlog/tasks/task-32670%20-%20Authenticate-complete-plugin-authority-snapshots.md).

**Deliverable:** Make reviewed local installation and authority changes recoverable without accidentally publishing uncommitted capabilities.

**Files:**

- Create: `tldw_chatbook/Plugins/coordinator.py`
- Create: `tldw_chatbook/Plugins/recovery.py`
- Create: `tldw_chatbook/Plugins/review.py`
- Test: `Tests/Plugins/test_coordinator.py`
- Test: `Tests/Plugins/test_recovery.py`
- Test: `Tests/Plugins/recovery_worker.py`

**Interfaces**

- Consumes: F1 inspections, F2 registry/ownership and F3 protected authority.
- Produces: PluginCoordinator(registry: PluginRegistry, authority: PluginAuthorityStore, owner: PluginRuntimeOwner).review(inspection: PackageInspection, *, selection: tuple[str, ...], workspace_id: str | None) -> PluginReview; async commit(review: PluginReview, operation_id: str) -> OperationReceipt; async recover() -> tuple[OperationReceipt, ...]. PluginReview binds all spec section 8.1 inputs and an expiry/invalidation token. OperationReceipt exposes operation_id, phase, committed and recovery_reason without source bodies or secrets. recovery_action(*, registry_new: bool, marker_new: bool, certificate_valid: bool, snapshot_valid: bool) -> str implements the evidence table.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_registry_flag_cannot_replace_commit_certificate():
    from tldw_chatbook.Plugins.recovery import recovery_action
    assert recovery_action(registry_new=True, marker_new=False, certificate_valid=False, snapshot_valid=True) == "review_required"
    assert recovery_action(registry_new=True, marker_new=False, certificate_valid=True, snapshot_valid=True) == "complete_marker"
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_coordinator.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def recovery_action(*, registry_new: bool, marker_new: bool,
                    certificate_valid: bool, snapshot_valid: bool) -> str:
    if not snapshot_valid:
        return "review_required"
    if marker_new:
        return "reconcile_committed"
    if certificate_valid:
        return "complete_marker"
    return "review_required" if registry_new else "abort_prepared"
```

  - [ ] 3.1. Create plugin_stack(tmp_path) in Tests/Plugins/conftest.py returning the real registry, authority and coordinator with isolated marker storage; add named fault barriers after each durable step, never a replacement fake coordinator.
  - [ ] 3.2. Implement the six-step commit order from the spec. Keep certificate creation after SQLite returns durable success; fencing remains until marker/projections agree. Revalidate exact review inputs immediately before mutation.
  - [ ] 3.3. Implement same-operation receipt lookup and recovery from the protected complete snapshot. The small decision kernel assumes identity/MAC validation already succeeded; unrelated markers, mismatched operations and missing provenance quarantine before that classifier.
  - [ ] 3.4. Use recovery_worker.py to kill the actual owner after each durable boundary and recover in a fresh process. Preserve other installations, do not synthesize external grants and protect the current marker snapshot from retention.

**Failure and successful-control matrix:** Prepared-only abort; valid post-commit proof with old marker; missing proof after DB commit; new marker with rolled-back/missing DB; invalid journal; stale review; full disk; lost response; repeated operation ID. Verify positive install remains disabled and projections cannot self-activate.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_coordinator.py Tests/Plugins/test_recovery.py Tests/Plugins/recovery_worker.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32671 --plain
git diff --check
```

## F5: Admit native plugin skills through existing Console authority

**Backlog:** [TASK-32672](../../../backlog/tasks/task-32672%20-%20Admit-native-plugin-skills-through-existing-Console-authority.md). **Requires:** [TASK-32671](../../../backlog/tasks/task-32671%20-%20Commit-and-recover-reviewed-plugin-installation-changes.md).

**Deliverable:** Deliver the first usable local plugin path through existing skill, tool and context services with workspace-specific eligibility.

**Files:**

- Create: `tldw_chatbook/Plugins/admission.py`
- Create: `tldw_chatbook/Plugins/skill_provider.py`
- Create: `tldw_chatbook/Plugins/context.py`
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `tldw_chatbook/Skills_Interop/skills_scope_service.py`
- Modify: `tldw_chatbook/Agents/tool_catalog.py`
- Modify: `tldw_chatbook/Chat/console_skill_resolver.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Test: `Tests/Plugins/test_native_skill_flow.py`
- Test: `Tests/Plugins/test_admission.py`
- Test: `Tests/Chat/test_console_skill_resolver.py`
- Test: `Tests/Agents/test_skill_tool_provider.py`

**Interfaces**

- Consumes: F4 committed authority and existing ToolProvider/list_catalog/load_schema/invoke and Console skill resolution seams.
- Produces: PluginAdmission.capture(installation_id: str, workspace_id: str | None, run_id: str) -> RunPluginSnapshot; check(snapshot: RunPluginSnapshot, component_id: str) -> None raises PluginUnavailable. RunPluginSnapshot is frozen and carries revision, selection, workspace, mappings, installation/scope generations and dependency requirements. PluginSkillProvider implements the existing ToolProvider protocol. effective_activation(default: bool, override: str) -> bool supports only inherit/enabled/disabled.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_explicit_workspace_disable_beats_global_default():
    from tldw_chatbook.Plugins.admission import effective_activation
    assert not effective_activation(True, "disabled")
    assert effective_activation(True, "inherit")
    assert effective_activation(False, "enabled")
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_native_skill_flow.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def effective_activation(default: bool, override: str) -> bool:
    if override == "inherit":
        return default
    if override in {"enabled", "disabled"}:
        return override == "enabled"
    raise ValueError("plugin_activation_invalid")
```

  - [ ] 3.1. Extend plugin_stack with real Console/provider composition. Install a native skill, explicitly trust/enable it, invoke through the production skill resolver and assert the final provider input includes attributed untrusted context.
  - [ ] 3.2. Capture actual workspace identity and immutable selected revision at run admission. Check current authenticated authority before every new injection/invoke/launch and approval acceptance; an unrelated namespace marker advance triggers revalidation rather than cancelling all scopes.
  - [ ] 3.3. Implement dependency-aware tool eligibility, inline/fork metadata and whole-block 8 KiB/32 KiB context limits. Empty agent/skill tool restrictions never become inherit; new enablements wait for new run admission.
  - [ ] 3.4. Add package-owned projections to Skills listing/detail and reject standalone update/delete/import-overwrite against owned IDs at service entry, not only in the UI.

**Failure and successful-control matrix:** Workspace A versus B/default, invalid workspace IDs, missing dependencies, manual-only use, forked instruction authority, tamper after review, new capability mid-run, alias collisions and standalone skill successful controls. A pure activation test is only the first RED; the Console flow is mandatory.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_native_skill_flow.py Tests/Plugins/test_admission.py Tests/Chat/test_console_skill_resolver.py Tests/Agents/test_skill_tool_provider.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32672 --plain
git diff --check
```

## F6: Stop and revoke plugin work within the requested scope

**Backlog:** [TASK-32673](../../../backlog/tasks/task-32673%20-%20Stop-and-revoke-plugin-work-within-the-requested-scope.md). **Requires:** [TASK-32672](../../../backlog/tasks/task-32672%20-%20Admit-native-plugin-skills-through-existing-Console-authority.md).

**Deliverable:** Let users immediately stop plugin activity while preserving unrelated authorized work and reporting persistence honestly.

**Files:**

- Create: `tldw_chatbook/Plugins/revocation.py`
- Modify: `tldw_chatbook/Plugins/coordinator.py`
- Modify: `tldw_chatbook/Plugins/admission.py`
- Modify: `tldw_chatbook/Plugins/runtime_owner.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Test: `Tests/Plugins/test_revocation.py`
- Test: `Tests/Plugins/test_revocation_persistence.py`

**Interfaces**

- Consumes: F5 scoped snapshots and F2 process/lease ownership.
- Produces: RevocationTarget(installation_id: str, workspace_id: str | None, everywhere: bool) is frozen. async PluginCoordinator.disable(target: RevocationTarget, operation_id: str) -> OperationReceipt and uninstall(installation_id: str, operation_id: str) -> OperationReceipt. PluginAdmission.seal(target: RevocationTarget) -> tuple[str, ...] synchronously fences affected owners and returns cancellation tokens; the persistence path is a separate await.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
import pytest

@pytest.mark.asyncio
async def test_disable_starts_cleanup_before_blocked_persistence(revocation_case):
    case = revocation_case
    case.block_persistence()
    operation = case.start_disable_here()
    await case.cleanup_started.wait()
    assert not case.admission_allows_a()
    assert case.admission_allows_b()
    case.release_persistence()
    await operation
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_revocation.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
async def disable_scoped(admission, target, cancel_owned, persist):
    tokens = admission.seal(target)
    cancel_owned(tokens)  # Transfers cleanup to a retained host task before awaiting.
    return await persist()
```

  - [ ] 3.1. Build revocation_case in test_revocation_persistence.py around plugin_stack: A/B admitted skills, an owned controlled subprocess, real admission checks, an asyncio.Event for cleanup-start and a storage fault gate. Define block_persistence/start_disable_here/admission_allows_a/admission_allows_b/release_persistence locally on that fixture; start_disable_here schedules the real coordinator method.
  - [ ] 3.2. Implement the ordering kernel with cancellation owned by a retained host task; cancellation of the caller must not abandon cleanup. Return/report persistence and runtime outcomes independently, preserving the original storage/cancellation error if cleanup also fails.
  - [ ] 3.3. Invalidate pending approvals, checkpoints, context callbacks and continuations only for the target scope. Suppress affected plugin cleanup hooks immediately; independently configured user handlers retain their authority.
  - [ ] 3.4. Make uninstall tombstones and owned-grant/registration removal durable before file removal. Preserve data and independent credentials; expose session-only failure without a restart guarantee.

**Failure and successful-control matrix:** Stalled storage before every durable boundary, locked trust, dead child and surviving child, re-enable after stale approval, global-default inheritors versus explicit overrides, uninstall idempotency and no new plugin-owned callback after the live fence.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_revocation.py Tests/Plugins/test_revocation_persistence.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32673 --plain
git diff --check
```

## F7: Drain plugin work before applying updates and rollback

**Backlog:** [TASK-32674](../../../backlog/tasks/task-32674%20-%20Drain-plugin-work-before-applying-updates-and-rollback.md). **Requires:** [TASK-32673](../../../backlog/tasks/task-32673%20-%20Stop-and-revoke-plugin-work-within-the-requested-scope.md).

**Deliverable:** Apply one reviewed package revision without overlapping incompatible active work or silently restoring historical permissions.

**Files:**

- Create: `tldw_chatbook/Plugins/revisions.py`
- Create: `tldw_chatbook/Plugins/retention.py`
- Modify: `tldw_chatbook/Plugins/coordinator.py`
- Modify: `tldw_chatbook/Plugins/runtime_owner.py`
- Test: `Tests/Plugins/test_revision_drain.py`
- Test: `Tests/Plugins/test_retention.py`

**Interfaces**

- Consumes: F4 review/commit and F6 cancellation; real leases from F2/F5.
- Produces: async PluginCoordinator.apply_revision(review: PluginReview, operation_id: str) -> OperationReceipt. RevisionDrain.begin(installation_id: str, revision_digest: str) -> str; blockers(token: str) -> tuple[dict, ...]; cancel(token: str) -> None; async wait(token: str) -> None. retention_candidates(revisions: tuple[dict, ...], protected: frozenset[str], now: datetime) -> tuple[str, ...] never returns current/leased/recovery IDs.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
def test_retention_never_removes_recovery_material(retention_case):
    from tldw_chatbook.Plugins.retention import retention_candidates
    case = retention_case
    candidates = retention_candidates(case.revisions, case.protected, case.now)
    assert not (set(candidates) & case.protected)
    assert case.expired_unleased in candidates
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_revision_drain.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def preserve_selection(old_selected: frozenset[str], available: frozenset[str]) -> frozenset[str]:
    return old_selected & available
```

  - [ ] 3.1. Create retention_case in test_retention.py with dated current, old, leased, recovery and expired unleased records; define revisions/protected/now/expired_unleased directly in the fixture. Test real coordinator update drain with held approvals and running tools.
  - [ ] 3.2. Fence admission under the lifecycle lock and await existing work without holding that lock. Prevent Stop chains extending the drain; close idle MCP connections only after requests drain once the MCP integration is present.
  - [ ] 3.3. Implement wait, pre-commit cancel and explicit work cancellation as different requests. Publish a reviewed revision only after confirmed cleanup via F4; cancelled review never retargets another installation.
  - [ ] 3.4. Preserve exclusions, report unknown mutable-data compatibility, implement rollback as fresh current-policy review, and enforce exact package/cache/staging/receipt retention limits. Add archive-resume refusal when a pinned old revision no longer matches data generation.

**Failure and successful-control matrix:** Pending approval, stale callback, foreground work, cancelled waiter, failed startup with no automatic rollback, shared data already mutated, lost completion response and quota with only protected material remaining.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_revision_drain.py Tests/Plugins/test_retention.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32674 --plain
git diff --check
```

## F8: Delete plugin data only after exact-root users drain

**Backlog:** [TASK-32675](../../../backlog/tasks/task-32675%20-%20Delete-plugin-data-only-after-exact-root-users-drain.md). **Requires:** [TASK-32673](../../../backlog/tasks/task-32673%20-%20Stop-and-revoke-plugin-work-within-the-requested-scope.md), [TASK-32674](../../../backlog/tasks/task-32674%20-%20Drain-plugin-work-before-applying-updates-and-rollback.md).

**Deliverable:** Remove only the reviewed saved data after every owned user has stopped, including processes that can write while idle.

**Files:**

- Create: `tldw_chatbook/Plugins/data_cleanup.py`
- Modify: `tldw_chatbook/Plugins/authority.py`
- Modify: `tldw_chatbook/Plugins/coordinator.py`
- Modify: `tldw_chatbook/Plugins/runtime_owner.py`
- Test: `Tests/Plugins/test_data_cleanup.py`
- Test: `Tests/Plugins/test_surviving_process_recovery.py`

**Interfaces**

- Consumes: F2 provenance, F3 authenticated root generations, F6 cancellation and F7 drain ownership.
- Produces: DataRootRef(installation_id: str, root_id: str, generation: int, path: Path) is frozen. RootUsage.acquire(root: DataRootRef, owner_token: str) -> str; release(token: str, confirmed: bool) -> None; fence(roots: tuple[DataRootRef, ...]) -> str. async PluginCoordinator.delete_data(roots: tuple[DataRootRef, ...], operation_id: str) -> OperationReceipt validates reviewed identities, persists the fence and waits for actual root users.

- [ ] **1. Create the first behavioral test and its local fixtures.** Use the fixture contract in the implementation increments below; initial import/behavior must fail for the missing feature.

```python
import pytest

@pytest.mark.asyncio
async def test_idle_writer_blocks_data_deletion(data_cleanup_case):
    case = data_cleanup_case
    deletion = case.start_delete()
    await case.fenced.wait()
    assert case.root.exists()
    assert not deletion.done()
    await case.stop_owned_writer()
    await deletion
    assert not case.root.exists()
```

- [ ] **2. Establish RED through the intended entry.** Run `python -m pytest Tests/Plugins/test_data_cleanup.py -q`. Expected: the named new behavior fails, while any positive precondition/control succeeds. Resolve test harness/API errors before changing production code.

- [ ] **3. Implement the smallest invariant, then integrate the real owner.** This kernel states the ordering/data rule; the following increments supply the complete behavior and limits.

```python
def deletion_ready(*, pending_launches: int, readers: int, writers: int,
                   unresolved_owners: int, fence_committed: bool) -> bool:
    return fence_committed and not any((pending_launches, readers, writers, unresolved_owners))
```

  - [ ] 3.1. Create data_cleanup_case using plugin_stack, a real root and a controlled child that writes after an event while otherwise idle. start_delete schedules the real coordinator method; fenced observes its access gate; stop_owned_writer kills/reaps the exact owned child.
  - [ ] 3.2. Track roots for the entire lifetime of processes and operations given access, including pending launch and idle MCP. Fence new root users and show blockers from every workspace sharing PLUGIN_DATA.
  - [ ] 3.3. Persist exact root identities/fence through F4 before the first unlink. Revalidate resolved containment and generation immediately before each destructive operation; reject stale review, replacement links and retained-data reattachment.
  - [ ] 3.4. Retain fences and ownership after partial deletion or unknown writer survival. Advance generation through the coordinator only after confirmed cleanup; fresh authorized admissions may create a new root, while cancelled old work stays cancelled.

**Failure and successful-control matrix:** Shared A/B writers, idle server, active readers/Windows handles, late launch publication, root rename/link replacement, kill failure, owner restart, stale PID and pre/post-deletion cancellation. Never claim containment of arbitrary external writers.

- [ ] **4. Establish GREEN and preserve the neighboring path.** Run the exact files below. Expected: all execute and pass, with no silent coroutine/platform skips used as qualification. Inspect real resources/output, not source-string matches.

```bash
python -m pytest Tests/Plugins/test_data_cleanup.py Tests/Plugins/test_surviving_process_recovery.py -q
```

- [ ] **5. Review, record evidence and commit the task.** Update its ACs/notes and the relevant authoring/operation documentation; record platform limits. Run `git diff --check`, Python syntax checks and the verification-environment formatter/linter on the changed Python files as specified in the delivery plan. Stage the exact task-owned files, including any test fixtures and generated CSS, and commit; never stage unrelated work.

```bash
backlog task task-32675 --plain
git diff --check
```

## Shared package limits

These exact approved limits apply wherever this plan handles the corresponding resource.

| Resource | V1 limit | Exhaustion behavior |
| --- | --- | --- |
| Manifest or hooks definition JSON | 256 KiB each; depth 32 | Reject that document with a bounded diagnostic. |
| Catalog snapshot | 5 MiB; 10,000 entries; 50 sources | Reject oversized refresh; retain previous catalog. |
| Package snapshot | 100 MiB expanded; 10,000 files; 10 MiB/file; path depth 32 | Abort materialization; preserve current installation. |
| Normalized component inventory | 512/package | Reject inspection; do not drop arbitrary tail components. |
| Concurrent acquisitions | 2; 120 s overall per acquisition | Queue visibly or cancel with timeout; no plugin execution. |
| Git staging including object data | 500 MiB/operation | Terminate fetch at quota check; discard staging. This is host acquisition control, not an OS disk quota. |
| Managed package/cache storage | 2 GiB total; require estimated new bytes plus 100 MiB free reserve | Prune eligible cache or refuse before commit. |
| Inactive revisions | 2 most recent per installation; 30-day age target | Prune only unleased/unreferenced revisions; current and recovery records are protected. |
| Abandoned staging | 24 hours | Remove only after journal reconciliation and ownership checks. |
| Plugin instruction blocks | 8 KiB/block, 32 KiB combined per send, also bounded by remaining model context | Reject oversized selected material whole; explain affected components. |
| Listing page | 50 rows | Paginate; search remains over cached metadata. |
| Display metadata | 256 characters/name; 2,000/summary; 64 KiB README preview | Sanitize and mark display truncation; preserve immutable source for explicit file review. |
| Operation receipts | 1,000 terminal receipts or 30 days | Drop oldest eligible terminal receipts; never delete recovery authority. |
