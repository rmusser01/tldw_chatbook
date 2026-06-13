# Runtime Policy And Capability Registry Tranche 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Tranche 0 runtime-policy foundation so Chatbook has one authoritative local/server source state, one action-level capability registry, central policy enforcement, and phase-one protection against source-authority bypasses.

**Architecture:** Add a dedicated `runtime_policy` package that owns typed source state, capability registry rows, policy decisions, persistence, and shared enforcement helpers. Wire it into `AppState` and `app.py`, then use that one policy seam in two places: hard-stop checks at phase-one service/client boundaries and early UI preflight in a small set of representative mode-sensitive screens. Replace or fence the known raw server-client construction paths so Tranche 0 can honestly claim fail-closed source authority.

**Tech Stack:** Python 3.11+, Textual, existing scope services, existing `tldw_api` client/service wrappers, pytest

---

## Scope Check

This plan intentionally implements only the approved Tranche 0 slice:

- one authoritative runtime source state
- one action-level capability registry seeded from the audited matrix
- fixed reason-code policy decisions
- hard-stop enforcement for the approved phase-one seams
- representative UI preflight and restore-precedence behavior
- direct-call boundary cleanup and anti-regression guard coverage

This plan explicitly does **not** implement:

- end-user capability-map UX
- sync, mirroring, dual-write, or mixed local/server views
- multi-server switching UX
- full chat action hard-stop migration across every chat code path
- broad screen-by-screen preflight migration outside the representative callers below

Implementation note for agentic workers:

- Use `@superpowers:test-driven-development` before each code change.
- Use `@superpowers:verification-before-completion` before claiming any task complete.

## File Map

- Create: `tldw_chatbook/runtime_policy/__init__.py`
  Responsibility: Export the public runtime-policy surface used by the app and tests.
- Create: `tldw_chatbook/runtime_policy/types.py`
  Responsibility: Define typed source/auth/reason literals, `RuntimeSourceState`, `CapabilityEntry`, `PolicyDecision`, and `PolicyDeniedError`.
- Create: `tldw_chatbook/runtime_policy/registry.py`
  Responsibility: Hold the audited action-level capability registry and completeness assertions for phase-one coverage.
- Create: `tldw_chatbook/runtime_policy/source_state.py`
  Responsibility: Normalize, persist, load, and freshness-downgrade runtime source/auth/reachability state.
- Create: `tldw_chatbook/runtime_policy/engine.py`
  Responsibility: Resolve allow/deny decisions from runtime state plus registry entry and optional scope/auth context.
- Create: `tldw_chatbook/runtime_policy/enforcement.py`
  Responsibility: Shared helpers for service hard stops, UI preflight checks, backend-exception classification, and direct-call factory guard helpers.
- Create: `tldw_chatbook/runtime_policy/bootstrap.py`
  Responsibility: Startup wiring, approved server-client/service factory functions, and app-facing runtime-policy access helpers.

- Modify: `tldw_chatbook/state/app_state.py`
  Responsibility: Store the authoritative runtime-policy snapshot in app state and keep round-trip serialization stable.
- Modify: `tldw_chatbook/state/__init__.py`
  Responsibility: Re-export updated state surfaces cleanly if needed by app startup/tests.
- Modify: `tldw_chatbook/app.py`
  Responsibility: Load runtime policy before screen restore, expose app-level accessors, persist source changes, and route representative screen consumers to the authoritative source state.

- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
  Responsibility: Add phase-one hard-stop policy checks for local/server/workspace note actions before routing to backing services.
- Modify: `tldw_chatbook/Notes/server_notes_workspace_service.py`
  Responsibility: Add the required hard-stop policy checks at the server notes/workspace service boundary itself so direct and future callers cannot bypass the seam.
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
  Responsibility: Add hard-stop policy checks for media, reading-progress, and server-only ingestion-source actions.
- Modify: `tldw_chatbook/Study_Interop/study_scope_service.py`
  Responsibility: Add hard-stop policy checks for flashcard/deck actions and workspace/local incompatibility handling through one policy seam.
- Modify: `tldw_chatbook/Study_Interop/quiz_scope_service.py`
  Responsibility: Add hard-stop policy checks for quiz/question/attempt actions.
- Modify: `tldw_chatbook/Evaluations_Interop/evaluation_scope_service.py`
  Responsibility: Add hard-stop policy checks for local/server evaluation, dataset, target, and run actions.
- Modify: `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py`
  Responsibility: Add hard-stop policy checks for local/server retrieval-admin actions.
- Modify: `tldw_chatbook/Character_Chat/character_persona_scope_service.py`
  Responsibility: Add hard-stop policy checks for character/persona reads and remote-only persona/preset flows.

- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
  Responsibility: Route approved server-note/workspace actions through enforced seams and add representative preflight before dispatch.
- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
  Responsibility: Use the authoritative runtime source accessor and preserve restore precedence for scope-sensitive study activation.
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
  Responsibility: Use the authoritative runtime source accessor and preserve restore precedence for CCP launch mode decisions.
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
  Responsibility: Preserve per-tab source metadata as identity only; never let restored tab state overwrite app runtime source.
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
  Responsibility: Add small helpers or comments/tests only if needed to clarify non-authoritative tab source metadata during restore.
- Modify: `tldw_chatbook/UI/Evals/screens/evaluation_browser.py`
  Responsibility: Prefer the app-owned `evaluation_scope_service` wired with runtime policy and stop constructing an unmanaged fallback seam that would bypass enforcement.
- Modify: `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`
  Responsibility: Add representative UI preflight for flashcard actions before service dispatch.
- Modify: `tldw_chatbook/UI/Study_Modules/quizzes_handler.py`
  Responsibility: Add representative UI preflight for quiz actions before service dispatch.

- Modify: `tldw_chatbook/tldw_api/exceptions.py`
  Responsibility: Preserve enough auth failure context to distinguish `server_auth_required` from `server_session_invalid`.
- Modify: `tldw_chatbook/tldw_api/client.py`
  Responsibility: Preserve structured `401` response detail on auth failures so runtime-policy classification has executable evidence.

- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
  Responsibility: Remove raw `TLDWAPIClient(...)` construction from the screen path and use an approved factory/helper seam.
- Modify: `tldw_chatbook/Event_Handlers/tldw_api_events.py`
  Responsibility: Remove raw `TLDWAPIClient(...)` construction from event-handling paths and use the approved factory/helper seam.
- Modify: `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`
  Responsibility: Remove raw `ServerChatbookService(...)` construction from wizard code and use the approved factory/helper seam.
- Modify: `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`
  Responsibility: Remove raw `ServerChatbookService(...)` construction from wizard code and use the approved factory/helper seam.

- Create: `tests/RuntimePolicy/test_runtime_policy_core.py`
  Responsibility: Cover typed state normalization, registry completeness for phase-one rows, reason-code behavior, freshness downgrade logic, and backend-exception classification.
- Create: `tests/RuntimePolicy/test_runtime_policy_bootstrap.py`
  Responsibility: Cover app bootstrap, persistence, restore precedence, and approved client/service factory behavior.
- Create: `tests/RuntimePolicy/test_boundary_guards.py`
  Responsibility: Fail if new raw `TLDWAPIClient` or `ServerChatbookService` constructions appear outside the approved allowlist.

- Modify: `tests/Notes/test_notes_scope_service.py`
  Responsibility: Prove note actions deny with fixed reasons when runtime/source policy forbids them.
- Modify: `tests/Notes/test_server_notes_workspace_service.py`
  Responsibility: Prove the server notes/workspace service boundary itself denies with fixed reasons when runtime/source policy forbids it.
- Modify: `tests/Media/test_media_reading_scope_service.py`
  Responsibility: Prove media and ingestion-source actions deny with fixed reasons when runtime/source policy forbids them.
- Modify: `tests/Study_Interop/test_study_scope_service.py`
  Responsibility: Prove study deck/card actions deny with fixed reasons when runtime/source policy forbids them.
- Modify: `tests/Study_Interop/test_quiz_scope_service.py`
  Responsibility: Prove quiz/question/attempt actions deny with fixed reasons when runtime/source policy forbids them.
- Modify: `tests/Evaluations_Interop/test_evaluation_scope_service.py`
  Responsibility: Prove evaluation, dataset, target, and run actions deny with fixed reasons when runtime/source policy forbids them.
- Modify: `tests/RAG_Admin/test_rag_admin_scope_service.py`
  Responsibility: Prove retrieval-admin actions deny with fixed reasons when runtime/source policy forbids them.
- Modify: `tests/Character_Chat/test_character_persona_scope_service.py`
  Responsibility: Prove character/persona reads and server-only flows deny with fixed reasons when runtime/source policy forbids them.
- Modify: `tests/UI/test_notes_screen.py`
  Responsibility: Prove representative notes preflight and routing no longer bypass policy through direct server service calls.
- Modify: `tests/UI/test_evaluation_browser_screen.py`
  Responsibility: Prove the evaluation browser consumes the app-owned enforced scope service instead of constructing an unmanaged fallback.
- Modify: `tests/UI/test_study_screen.py`
  Responsibility: Prove study screen uses authoritative runtime source and preserves restore precedence.
- Modify: `tests/UI/test_ccp_screen.py`
  Responsibility: Prove CCP launch uses authoritative runtime source and does not resurrect stale saved mode.
- Modify: `tests/UI/test_chat_screen_state.py`
  Responsibility: Prove restored chat tabs keep source metadata without silently switching app runtime source.
- Create: `tests/tldw_api/test_client_error_classification.py`
  Responsibility: Prove `401` auth failures preserve structured detail so runtime-policy classification can produce `server_session_invalid` when evidence exists.

## Task 1: Build The Core Runtime Policy Package

**Files:**
- Create: `tldw_chatbook/runtime_policy/__init__.py`
- Create: `tldw_chatbook/runtime_policy/types.py`
- Create: `tldw_chatbook/runtime_policy/registry.py`
- Create: `tldw_chatbook/runtime_policy/source_state.py`
- Create: `tldw_chatbook/runtime_policy/engine.py`
- Create: `tldw_chatbook/runtime_policy/enforcement.py`
- Create: `tests/RuntimePolicy/test_runtime_policy_core.py`

- [ ] **Step 1: Write the failing core runtime-policy tests**

```python
from datetime import datetime, timedelta, timezone

from tldw_chatbook.runtime_policy.engine import PolicyEngine
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import RuntimeSourceState


def test_runtime_source_state_downgrades_stale_server_signals_to_unknown():
    state = RuntimeSourceState(
        active_source="server",
        active_server_id="primary",
        server_configured=True,
        server_reachability="reachable",
        server_reachability_checked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        server_auth_state="authenticated",
        server_auth_checked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )

    normalized = state.normalized_for_policy(
        now=datetime.now(timezone.utc),
        freshness_window=timedelta(minutes=5),
    )

    assert normalized.server_reachability == "unknown"
    assert normalized.server_auth_state == "unknown"


def test_policy_engine_denies_remote_only_action_in_local_mode():
    engine = PolicyEngine(CAPABILITY_REGISTRY)
    decision = engine.evaluate(
        action_id="workflows.launch.server",
        state=RuntimeSourceState(active_source="local"),
    )

    assert decision.allowed is False
    assert decision.reason_code == "wrong_source"


def test_phase_one_registry_contains_required_rows():
    required = {
        "notes.create.local",
        "notes.create.server",
        "notes.create.workspace",
        "media.ingestion_sources.list.server",
        "study.deck.create.server",
        "quiz.create.server",
        "rag.template.create.local",
        "character.persona.list.server",
        "workflows.launch.server",
    }
    assert required.issubset(CAPABILITY_REGISTRY)
```

- [ ] **Step 2: Run the focused core tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/RuntimePolicy/test_runtime_policy_core.py -q`
Expected: FAIL because the `runtime_policy` package and its core types/engine do not exist yet.

- [ ] **Step 3: Implement the minimal typed runtime-policy package**

```python
@dataclass(frozen=True)
class RuntimeSourceState:
    active_source: Literal["local", "server"] = "local"
    active_server_id: str | None = None
    server_configured: bool = False
    server_reachability: Literal["unknown", "reachable", "unreachable"] = "unknown"
    server_reachability_checked_at: datetime | None = None
    server_auth_state: Literal["unknown", "authenticated", "auth_required", "session_invalid"] = "unknown"
    server_auth_checked_at: datetime | None = None
    last_known_server_label: str | None = None

    def normalized_for_policy(self, *, now: datetime, freshness_window: timedelta) -> "RuntimeSourceState":
        ...


@dataclass(frozen=True)
class CapabilityEntry:
    action_id: str
    domain_id: str
    action_kind: Literal["browse", "detail", "create", "update", "delete", "launch", "observe"]
    required_source: Literal["local", "server"]
    authority_owner: Literal["chatbook_local", "server", "dual_backend"]
    default_reason_code: str
    workspace_aware: bool = False


class PolicyEngine:
    def __init__(self, registry: Mapping[str, CapabilityEntry]):
        self._registry = registry

    def evaluate(self, *, action_id: str, state: RuntimeSourceState, scope_type: str | None = None) -> PolicyDecision:
        ...
```

Implementation requirements:

- The registry must be a code-defined literal seeded from `Docs/Parity/2026-04-21-capability-matrix.md`, not an ad hoc partial stub.
- Missing `action_id` lookups must fail closed.
- `enforcement.py` must define one shared backend-exception classification helper now, even if service integration lands in later tasks.

- [ ] **Step 4: Re-run the focused core tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/RuntimePolicy/test_runtime_policy_core.py -q`
Expected: PASS, with fixed reason codes and stale-state downgrade behavior covered.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/runtime_policy tests/RuntimePolicy/test_runtime_policy_core.py
git commit -m "feat: add runtime policy core package"
```

## Task 2: Wire Runtime Policy Persistence And App Bootstrap

**Files:**
- Modify: `tldw_chatbook/state/app_state.py`
- Modify: `tldw_chatbook/state/__init__.py`
- Modify: `tldw_chatbook/runtime_policy/source_state.py`
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `tldw_chatbook/app.py`
- Create: `tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Write the failing bootstrap and persistence tests**

```python
from pathlib import Path

from tldw_chatbook.runtime_policy.bootstrap import load_runtime_policy_for_app
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.state.app_state import AppState


def test_app_state_round_trip_preserves_runtime_policy_snapshot():
    state = AppState()
    state.runtime_source = RuntimeSourceState(active_source="server", active_server_id="primary")

    restored = AppState.from_dict(state.to_dict())

    assert restored.runtime_source.active_source == "server"
    assert restored.runtime_source.active_server_id == "primary"


def test_runtime_policy_store_persists_source_state(tmp_path: Path):
    store = RuntimeSourceStateStore(tmp_path / "runtime_policy.json")
    original = RuntimeSourceState(active_source="server", active_server_id="primary", server_configured=True)

    store.save(original)
    restored = store.load()

    assert restored == original


def test_restore_precedence_drops_saved_screen_snapshot_for_wrong_server():
    runtime_state = RuntimeSourceState(active_source="server", active_server_id="primary")
    saved_screen_state = {
        "runtime_policy_snapshot": {
            "active_source": "server",
            "active_server_id": "secondary",
        },
        "chat_state": {
            "tabs": [],
            "active_tab_id": None,
            "tab_order": [],
        },
    }

    resolved = reconcile_saved_screen_state(saved_screen_state, runtime_state=runtime_state)

    assert resolved.get("runtime_policy_snapshot", {}).get("active_server_id") != "secondary"


@pytest.mark.asyncio
async def test_handle_runtime_backend_changed_updates_authoritative_runtime_state(mock_app):
    mock_app.runtime_policy = load_runtime_policy_for_app(mock_app, state_path=None)

    await mock_app.handle_runtime_backend_changed("server")

    assert mock_app.runtime_policy.state.active_source == "server"
```

- [ ] **Step 2: Run the focused bootstrap tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q`
Expected: FAIL because `AppState` does not yet serialize runtime policy and `app.py` does not bootstrap or persist it.

- [ ] **Step 3: Implement authoritative runtime-policy bootstrap and persistence**

```python
class RuntimeSourceStateStore:
    def __init__(self, path: Path | None = None):
        self.path = path or (Path.home() / ".config" / "tldw_cli" / "runtime_policy.json")

    def load(self) -> RuntimeSourceState:
        ...

    def save(self, state: RuntimeSourceState) -> None:
        ...


def load_runtime_policy_for_app(app, *, state_path: Path | None = None) -> RuntimePolicyContext:
    store = RuntimeSourceStateStore(state_path)
    state = store.load()
    context = RuntimePolicyContext(store=store, state=state, engine=PolicyEngine(CAPABILITY_REGISTRY))
    app.current_runtime_backend = state.active_source
    app.runtime_backend = state.active_source
    return context
```

Implementation requirements:

- Load runtime policy before any screen restore path runs.
- Persist immediately on authoritative source change instead of relying on best-effort screen save hooks.
- Store a small `runtime_policy_snapshot` with saved screen state so restore reconciliation can detect wrong-source and wrong-server snapshots without making screen-local state authoritative.
- Wrong-server saved screen snapshots must be dropped or re-bound during restore instead of silently replacing `active_server_id`.
- Keep `chat_screen_state.py` and saved tab/session metadata non-authoritative: bootstrap owns `active_source`.

- [ ] **Step 4: Re-run the focused bootstrap tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q`
Expected: PASS, with stable disk persistence and authoritative app bootstrap behavior.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/state/app_state.py tldw_chatbook/state/__init__.py tldw_chatbook/runtime_policy/source_state.py tldw_chatbook/runtime_policy/bootstrap.py tldw_chatbook/app.py tests/RuntimePolicy/test_runtime_policy_bootstrap.py
git commit -m "feat: bootstrap authoritative runtime policy state"
```

## Task 3: Add Hard-Stop Enforcement To Notes, Media, Study, And Quiz Scope Services

**Files:**
- Modify: `tldw_chatbook/Notes/notes_scope_service.py`
- Modify: `tldw_chatbook/Notes/server_notes_workspace_service.py`
- Modify: `tldw_chatbook/Media/media_reading_scope_service.py`
- Modify: `tldw_chatbook/Study_Interop/study_scope_service.py`
- Modify: `tldw_chatbook/Study_Interop/quiz_scope_service.py`
- Modify: `tests/Notes/test_notes_scope_service.py`
- Modify: `tests/Notes/test_server_notes_workspace_service.py`
- Modify: `tests/Media/test_media_reading_scope_service.py`
- Modify: `tests/Study_Interop/test_study_scope_service.py`
- Modify: `tests/Study_Interop/test_quiz_scope_service.py`

- [ ] **Step 1: Extend the existing scope-service tests with failing policy-denial coverage**

```python
@pytest.mark.asyncio
async def test_notes_scope_service_denies_server_create_when_active_source_is_local():
    service = NotesScopeService(
        local_notes_service=FakeLocalNotes(),
        server_service=FakeServerNotes(),
        policy_enforcer=FakePolicyEnforcer.deny("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await service.save_note(
            scope=ScopeType.SERVER_NOTE,
            title="Remote",
            content="Body",
        )

    assert exc.value.decision.reason_code == "wrong_source"


@pytest.mark.asyncio
async def test_media_scope_service_denies_server_ingestion_sources_when_server_is_unreachable():
    scope_service = MediaReadingScopeService(
        local_service=FakeLocalMediaService(),
        server_service=FakeServerMediaService(),
        policy_enforcer=FakePolicyEnforcer.deny("server_unreachable"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope_service.list_ingestion_sources(mode="server")


@pytest.mark.asyncio
async def test_server_notes_workspace_service_denies_workspace_mutation_when_source_is_wrong():
    service = ServerNotesWorkspaceService(
        client=FakeClient(),
        policy_enforcer=FakePolicyEnforcer.deny("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError) as exc:
        await service.save_workspace_note(
            workspace_id="ws-1",
            title="Draft",
            content="Body",
        )

    assert exc.value.decision.reason_code == "wrong_source"
```

- [ ] **Step 2: Run the affected service tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/Notes/test_notes_scope_service.py tests/Notes/test_server_notes_workspace_service.py tests/Media/test_media_reading_scope_service.py tests/Study_Interop/test_study_scope_service.py tests/Study_Interop/test_quiz_scope_service.py -q`
Expected: FAIL because the scope services do not yet call a shared policy hard-stop seam, and the server notes/workspace service does not yet enforce its own boundary.

- [ ] **Step 3: Implement service-level hard-stop enforcement using one shared helper**

```python
class ServicePolicyEnforcer:
    def __init__(self, runtime_policy_context: RuntimePolicyContext):
        self.context = runtime_policy_context

    def require_allowed(self, *, action_id: str, scope_type: str | None = None, workspace_id: str | None = None) -> None:
        decision = self.context.engine.evaluate(
            action_id=action_id,
            state=self.context.current_state(),
            scope_type=scope_type,
        )
        if not decision.allowed:
            raise PolicyDeniedError(decision)
```

Implementation requirements:

- Notes must distinguish `notes.create.local`, `notes.create.server`, and `notes.create.workspace`.
- `ServerNotesWorkspaceService` must enforce policy at its own boundary and not rely on callers reaching it through `NotesScopeService`.
- Media must distinguish local CRUD/read actions from server-only ingestion-source actions.
- Study and quiz services must continue to normalize local/server payloads after the hard stop passes.
- Existing workspace-local incompatibility errors can remain, but source authority checks must happen through the runtime-policy seam first.

- [ ] **Step 4: Re-run the affected service tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/Notes/test_notes_scope_service.py tests/Notes/test_server_notes_workspace_service.py tests/Media/test_media_reading_scope_service.py tests/Study_Interop/test_study_scope_service.py tests/Study_Interop/test_quiz_scope_service.py -q`
Expected: PASS, with denials surfacing as fixed reason-code policy failures instead of ad hoc route errors at both the scope seam and the server notes/workspace boundary.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Notes/notes_scope_service.py tldw_chatbook/Notes/server_notes_workspace_service.py tldw_chatbook/Media/media_reading_scope_service.py tldw_chatbook/Study_Interop/study_scope_service.py tldw_chatbook/Study_Interop/quiz_scope_service.py tests/Notes/test_notes_scope_service.py tests/Notes/test_server_notes_workspace_service.py tests/Media/test_media_reading_scope_service.py tests/Study_Interop/test_study_scope_service.py tests/Study_Interop/test_quiz_scope_service.py
git commit -m "feat: enforce runtime policy in core scope services"
```

## Task 4: Add Hard-Stop Enforcement To Evaluations, RAG Admin, And Character/Persona Seams, Including Executable Auth Classification

**Files:**
- Modify: `tldw_chatbook/Evaluations_Interop/evaluation_scope_service.py`
- Modify: `tldw_chatbook/RAG_Admin/rag_admin_scope_service.py`
- Modify: `tldw_chatbook/Character_Chat/character_persona_scope_service.py`
- Modify: `tldw_chatbook/UI/Evals/screens/evaluation_browser.py`
- Modify: `tldw_chatbook/runtime_policy/enforcement.py`
- Modify: `tldw_chatbook/tldw_api/exceptions.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tests/Evaluations_Interop/test_evaluation_scope_service.py`
- Modify: `tests/RAG_Admin/test_rag_admin_scope_service.py`
- Modify: `tests/Character_Chat/test_character_persona_scope_service.py`
- Modify: `tests/UI/test_evaluation_browser_screen.py`
- Modify: `tests/RuntimePolicy/test_runtime_policy_core.py`
- Create: `tests/tldw_api/test_client_error_classification.py`

- [ ] **Step 1: Add failing tests for auth/session and authority classification**

```python
from tldw_chatbook.tldw_api.exceptions import APIConnectionError, APIResponseError, AuthenticationError


def test_classify_backend_exception_maps_auth_and_connectivity_to_fixed_reason_codes():
    assert classify_backend_exception(AuthenticationError("missing")) == "server_auth_required"
    assert classify_backend_exception(APIConnectionError("down")) == "server_unreachable"
    assert classify_backend_exception(APIResponseError(403, "forbidden")) == "authority_denied"


def test_classify_backend_exception_maps_session_invalid_marker_to_fixed_code():
    exc = AuthenticationError(
        "session expired",
        response_data={"code": "session_invalid", "detail": "Session expired"},
    )
    assert classify_backend_exception(exc) == "server_session_invalid"


@pytest.mark.asyncio
async def test_character_persona_scope_service_denies_server_persona_listing_in_local_mode():
    scope = CharacterPersonaScopeService(
        local_service=Mock(),
        server_service=Mock(),
        policy_enforcer=FakePolicyEnforcer.deny("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope.list_persona_profiles(mode="server")


@pytest.mark.asyncio
async def test_evaluation_scope_service_denies_server_run_creation_in_local_mode():
    scope = EvaluationScopeService(
        local_service=FakeLocalEvaluationService(),
        server_service=FakeServerEvaluationService(),
        policy_enforcer=FakePolicyEnforcer.deny("wrong_source"),
    )

    with pytest.raises(PolicyDeniedError):
        await scope.create_run(mode="server", eval_id="eval_123", target_model="openai:gpt-4.1")
```

- [ ] **Step 2: Run the affected auth/classification tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/Evaluations_Interop/test_evaluation_scope_service.py tests/RAG_Admin/test_rag_admin_scope_service.py tests/Character_Chat/test_character_persona_scope_service.py tests/UI/test_evaluation_browser_screen.py tests/RuntimePolicy/test_runtime_policy_core.py tests/tldw_api/test_client_error_classification.py -q`
Expected: FAIL because evaluations are not yet covered by the hard-stop seam, the evaluation browser can still construct an unmanaged fallback service, and auth/session failures do not yet preserve enough evidence to distinguish `server_session_invalid`.

- [ ] **Step 3: Implement shared exception classification and apply it to the remaining approved seams**

```python
def classify_backend_exception(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError) and getattr(exc, "response_data", {}).get("code") == "session_invalid":
        return "server_session_invalid"
    if isinstance(exc, AuthenticationError):
        return "server_auth_required"
    if isinstance(exc, APIConnectionError):
        return "server_unreachable"
    if isinstance(exc, APIResponseError) and exc.status_code == 403:
        return "authority_denied"
    raise
```

Implementation requirements:

- `EvaluationScopeService` is part of the approved phase-one hard-stop seam and must be wired through the shared enforcer.
- `evaluation_browser.py` must prefer the app-owned `evaluation_scope_service` and must not construct an unmanaged fallback service that bypasses policy.
- Preserve enough structured `401` response detail in auth exceptions so `server_session_invalid` is actually reachable when the server says the session is invalid or expired.
- Treat `401`/authentication failures as `server_auth_required` for Tranche 0 unless the calling seam has enough evidence to classify `server_session_invalid`.
- Keep `403` and equivalent policy/tenant/feature denials mapped to `authority_denied`.
- Do not duplicate exception mapping logic across services; keep one helper in `runtime_policy/enforcement.py`.

- [ ] **Step 4: Re-run the affected auth/classification tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/Evaluations_Interop/test_evaluation_scope_service.py tests/RAG_Admin/test_rag_admin_scope_service.py tests/Character_Chat/test_character_persona_scope_service.py tests/UI/test_evaluation_browser_screen.py tests/RuntimePolicy/test_runtime_policy_core.py tests/tldw_api/test_client_error_classification.py -q`
Expected: PASS, with evaluation, RAG admin, and character/persona seams failing through the same fixed policy contract and `server_session_invalid` covered by a real classification path.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evaluations_Interop/evaluation_scope_service.py tldw_chatbook/RAG_Admin/rag_admin_scope_service.py tldw_chatbook/Character_Chat/character_persona_scope_service.py tldw_chatbook/UI/Evals/screens/evaluation_browser.py tldw_chatbook/runtime_policy/enforcement.py tldw_chatbook/tldw_api/exceptions.py tldw_chatbook/tldw_api/client.py tests/Evaluations_Interop/test_evaluation_scope_service.py tests/RAG_Admin/test_rag_admin_scope_service.py tests/Character_Chat/test_character_persona_scope_service.py tests/UI/test_evaluation_browser_screen.py tests/RuntimePolicy/test_runtime_policy_core.py tests/tldw_api/test_client_error_classification.py
git commit -m "feat: classify runtime policy auth and authority failures"
```

## Task 5: Remove Known Direct-Call Bypasses And Add The Boundary Guard

**Files:**
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Modify: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- Modify: `tldw_chatbook/Event_Handlers/tldw_api_events.py`
- Modify: `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`
- Modify: `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`
- Create: `tests/RuntimePolicy/test_boundary_guards.py`

- [ ] **Step 1: Write the failing boundary-guard test**

```python
from pathlib import Path


ALLOWLIST = {
    "tldw_chatbook/runtime_policy/bootstrap.py",
    "tldw_chatbook/Notes/server_notes_workspace_service.py",
    "tldw_chatbook/Study_Interop/server_study_service.py",
    "tldw_chatbook/Study_Interop/server_quiz_service.py",
}


def test_raw_server_client_construction_is_confined_to_approved_boundaries():
    repo = Path(__file__).resolve().parents[2]
    offenders = []
    for path in repo.joinpath("tldw_chatbook").rglob("*.py"):
        rel = path.relative_to(repo).as_posix()
        if rel in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if "TLDWAPIClient(" in text or "ServerChatbookService(" in text:
            offenders.append(rel)

    assert offenders == []
```

- [ ] **Step 2: Run the boundary-guard test to verify it fails**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/RuntimePolicy/test_boundary_guards.py -q`
Expected: FAIL because the known UI/event/wizard call sites still construct raw server clients/services directly.

- [ ] **Step 3: Introduce approved factory helpers and route the known bypasses through them**

```python
def build_runtime_api_client(*, app_config: dict[str, Any], endpoint_url: str | None = None, auth_token: str | None = None, auth_method: str | None = None) -> TLDWAPIClient:
    ...


def build_server_chatbook_service(*, app_config: dict[str, Any]) -> ServerChatbookService:
    client = build_runtime_api_client(app_config=app_config)
    return ServerChatbookService(client)
```

Implementation requirements:

- `MediaIngestWindowRebuilt.py` and `tldw_api_events.py` must stop constructing raw clients inline.
- `ChatbookImportWizard.py` and `ChatbookCreationWizard.py` must stop constructing `ServerChatbookService(...)` inline.
- Approved factory construction must live in one place, not be re-invented per caller.

- [ ] **Step 4: Re-run the boundary-guard test**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/RuntimePolicy/test_boundary_guards.py -q`
Expected: PASS, with only the explicit allowlist still constructing raw server clients/services.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/runtime_policy/bootstrap.py tldw_chatbook/UI/MediaIngestWindowRebuilt.py tldw_chatbook/Event_Handlers/tldw_api_events.py tldw_chatbook/UI/Wizards/ChatbookImportWizard.py tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py tests/RuntimePolicy/test_boundary_guards.py
git commit -m "refactor: route server clients through runtime policy factories"
```

## Task 6: Apply Representative UI Preflight And Restore Precedence

**Files:**
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen_state.py`
- Modify: `tldw_chatbook/UI/Screens/notes_screen.py`
- Modify: `tldw_chatbook/UI/Screens/study_screen.py`
- Modify: `tldw_chatbook/UI/Screens/ccp_screen.py`
- Modify: `tldw_chatbook/UI/Study_Modules/flashcards_handler.py`
- Modify: `tldw_chatbook/UI/Study_Modules/quizzes_handler.py`
- Modify: `tests/UI/test_chat_screen_state.py`
- Modify: `tests/UI/test_notes_screen.py`
- Modify: `tests/UI/test_study_screen.py`
- Modify: `tests/UI/test_ccp_screen.py`

- [ ] **Step 1: Add failing UI tests for restore precedence and representative preflight**

```python
@pytest.mark.asyncio
async def test_restored_chat_tab_state_does_not_overwrite_authoritative_runtime_source():
    screen = ChatScreen(mock_app)
    mock_app.runtime_policy.state = RuntimeSourceState(active_source="local")
    screen.chat_state = ChatScreenState(
        tabs=[TabState(tab_id="tab-1", title="Remote", conversation_id="conv-1", runtime_backend="server")],
        active_tab_id="tab-1",
        tab_order=["tab-1"],
    )

    await screen._restore_tab_sessions(tab_container)

    assert mock_app.runtime_policy.state.active_source == "local"


@pytest.mark.asyncio
async def test_restored_screen_state_does_not_replace_active_server_id_with_wrong_server_snapshot():
    mock_app.runtime_policy.state = RuntimeSourceState(active_source="server", active_server_id="primary")
    screen = ChatScreen(mock_app)

    screen.restore_state(
        {
            "runtime_policy_snapshot": {
                "active_source": "server",
                "active_server_id": "secondary",
            },
            "chat_state": {"tabs": [], "active_tab_id": None, "tab_order": []},
        }
    )

    assert mock_app.runtime_policy.state.active_server_id == "primary"


@pytest.mark.asyncio
async def test_study_screen_prefers_authoritative_runtime_source_over_saved_screen_mode():
    app_instance.runtime_policy.state = RuntimeSourceState(active_source="server")
    app_instance.current_runtime_backend = "local"

    screen = StudyScreen(app_instance=app_instance)

    assert screen._runtime_backend() == "server"
```

- [ ] **Step 2: Run the representative UI tests to verify they fail**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/UI/test_chat_screen_state.py tests/UI/test_notes_screen.py tests/UI/test_study_screen.py tests/UI/test_ccp_screen.py -q`
Expected: FAIL because the screens/controllers still read ad hoc runtime fields and restore order is not yet runtime-policy-first.

- [ ] **Step 3: Implement representative preflight and restore-precedence behavior**

```python
def get_authoritative_runtime_source(app_instance) -> str:
    runtime_policy = getattr(app_instance, "runtime_policy", None)
    if runtime_policy is not None:
        return runtime_policy.state.active_source
    ...


def require_ui_action_allowed(app_instance, *, action_id: str, scope_type: str | None = None) -> PolicyDecision:
    decision = app_instance.runtime_policy.engine.evaluate(
        action_id=action_id,
        state=app_instance.runtime_policy.current_state(),
        scope_type=scope_type,
    )
    if not decision.allowed:
        app_instance.notify(decision.user_message, severity="warning")
    return decision
```

Implementation requirements:

- `StudyScreen` and `CCPScreen` must prefer the authoritative runtime-policy accessor over saved screen-local runtime values.
- `ChatScreen` may preserve per-tab source metadata for identity and reuse, but restoring a tab must never switch `active_source`.
- Restore helpers must reject or drop saved `runtime_policy_snapshot` values whose `active_server_id` no longer matches the authoritative runtime server.
- `NotesScreen` must stop bypassing policy through direct `server_notes_workspace_service` usage in the covered create/save/load paths.
- `flashcards_handler.py` and `quizzes_handler.py` must do representative preflight before calling their scope services.

- [ ] **Step 4: Re-run the representative UI tests**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/UI/test_chat_screen_state.py tests/UI/test_notes_screen.py tests/UI/test_study_screen.py tests/UI/test_ccp_screen.py -q`
Expected: PASS, with authoritative runtime source preserved and representative UI callers warning early instead of dispatching invalid actions.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/chat_screen_state.py tldw_chatbook/UI/Screens/notes_screen.py tldw_chatbook/UI/Screens/study_screen.py tldw_chatbook/UI/Screens/ccp_screen.py tldw_chatbook/UI/Study_Modules/flashcards_handler.py tldw_chatbook/UI/Study_Modules/quizzes_handler.py tests/UI/test_chat_screen_state.py tests/UI/test_notes_screen.py tests/UI/test_study_screen.py tests/UI/test_ccp_screen.py
git commit -m "feat: apply runtime policy preflight and restore precedence"
```

## Task 7: Run The Phase-One Verification Matrix And Record The New Runtime-Policy Surface

**Files:**
- Modify: `Docs/Development/navigation-architecture-analysis.md`
- Create: `Docs/Development/runtime-policy-tranche-0.md`

- [ ] **Step 1: Write the documentation delta and verification checklist**

```markdown
# Runtime Policy Tranche 0

- authoritative runtime source state now lives in `tldw_chatbook/runtime_policy/`
- `active_source` is app-authoritative; restored screen/tab state is contextual only
- phase-one hard-stop seams: notes scope, server notes/workspace service, media, study, quiz, evaluations, rag admin, character/persona
- approved raw-client factory boundary: `runtime_policy/bootstrap.py`
```

- [ ] **Step 2: Run the phase-one focused verification matrix**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/RuntimePolicy/test_runtime_policy_core.py tests/RuntimePolicy/test_runtime_policy_bootstrap.py tests/RuntimePolicy/test_boundary_guards.py tests/Notes/test_notes_scope_service.py tests/Notes/test_server_notes_workspace_service.py tests/Media/test_media_reading_scope_service.py tests/Study_Interop/test_study_scope_service.py tests/Study_Interop/test_quiz_scope_service.py tests/Evaluations_Interop/test_evaluation_scope_service.py tests/RAG_Admin/test_rag_admin_scope_service.py tests/Character_Chat/test_character_persona_scope_service.py tests/UI/test_evaluation_browser_screen.py tests/UI/test_chat_screen_state.py tests/UI/test_notes_screen.py tests/UI/test_study_screen.py tests/UI/test_ccp_screen.py tests/tldw_api/test_client_error_classification.py -q`
Expected: PASS for the entire phase-one runtime-policy matrix.

- [ ] **Step 3: Run one broader smoke slice that includes the touched domains together**

Run: `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook .venv/bin/pytest tests/Notes/test_notes_scope_service.py tests/Notes/test_server_notes_workspace_service.py tests/Media/test_media_reading_scope_service.py tests/Study_Interop/test_study_scope_service.py tests/Study_Interop/test_quiz_scope_service.py tests/Evaluations_Interop/test_evaluation_scope_service.py tests/UI/test_notes_screen.py tests/UI/test_study_screen.py tests/UI/test_evaluation_browser_screen.py -q`
Expected: PASS, confirming the hard-stop layer did not regress the existing source-aware vertical seams.

- [ ] **Step 4: Commit the docs and final verification state**

```bash
git add Docs/Development/navigation-architecture-analysis.md Docs/Development/runtime-policy-tranche-0.md
git commit -m "docs: record runtime policy tranche 0 surface"
```

- [ ] **Step 5: Prepare the execution handoff**

```text
Summarize:
- which hard-stop seams landed
- which representative UI callers landed
- which raw-client bypasses were removed
- which chat/full-screen migrations remain explicitly deferred
```
