"""Atomic cutover guards for lasting Notes folder sync."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore


pytestmark = pytest.mark.unit
_PRODUCTION = Path("tldw_chatbook")
_LEGACY_MODULES = (
    _PRODUCTION / "Notes" / "sync_engine.py",
    _PRODUCTION / "Notes" / "sync_service.py",
    _PRODUCTION / "Library" / "library_notes_sync_state.py",
)
_LEGACY_WRITER_NAMES = {"NotesSyncEngine", "NotesSyncService"}
_LEGACY_CONFIG_KEYS = {
    "auto_sync_enabled",
    "sync_conflict_resolution",
    "sync_directory",
    "sync_direction",
    "sync_on_close",
}


def _python_trees() -> Iterator[tuple[Path, ast.AST]]:
    for path in _PRODUCTION.rglob("*.py"):
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_cutover_release_deletes_every_legacy_writer_module() -> None:
    assert [str(path) for path in _LEGACY_MODULES if path.exists()] == []


def test_production_never_imports_or_constructs_a_legacy_writer() -> None:
    violations: list[str] = []
    for path, tree in _python_trees():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imported = {alias.name.rsplit(".", 1)[-1] for alias in node.names}
                if imported & _LEGACY_WRITER_NAMES:
                    violations.append(f"{path}:{node.lineno}:import")
            if isinstance(node, ast.Call):
                called = _dotted_name(node.func).rsplit(".", 1)[-1]
                if called in _LEGACY_WRITER_NAMES:
                    violations.append(f"{path}:{node.lineno}:construct")
    assert violations == []


def test_library_screen_has_no_legacy_timer_worker_or_mutating_handler() -> None:
    path = _PRODUCTION / "UI" / "Screens" / "library_screen.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.name.startswith("handle_library_notes_sync_")
            or node.name.startswith("_library_notes_sync_")
            or node.name
            in {
                "_arm_library_notes_auto_sync_timer",
                "_cancel_library_notes_auto_sync_timer",
                "_run_library_notes_sync",
            }
        ):
            violations.append(f"{node.name}:{node.lineno}")
        if isinstance(node, ast.Attribute) and node.attr.startswith(
            "_library_notes_auto_sync_timer"
        ):
            violations.append(f"{node.attr}:{node.lineno}")
        if isinstance(node, ast.keyword) and node.arg == "group":
            if isinstance(node.value, ast.Constant) and node.value.value == (
                "library_notes_sync"
            ):
                violations.append(f"worker-group:{node.lineno}")
    assert violations == []


def test_legacy_sync_config_is_read_only_and_only_the_migrator_reads_it() -> None:
    violations: list[str] = []
    for path, tree in _python_trees():
        if path.name == "notes_sync_legacy.py":
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = _dotted_name(node.func).rsplit(".", 1)[-1]
            if called not in {
                "get_cli_setting",
                "save_setting_to_cli_config",
                "save_settings_to_cli_config",
            }:
                continue
            literals = {
                child.value
                for child in ast.walk(node)
                if isinstance(child, ast.Constant) and isinstance(child.value, str)
            }
            if literals & _LEGACY_CONFIG_KEYS:
                violations.append(f"{path}:{node.lineno}:{called}")
    assert violations == []


def test_note_library_exposes_no_callable_legacy_sync_metadata_writer() -> None:
    path = _PRODUCTION / "Notes" / "Notes_Library.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    retired = {
        "get_notes_for_sync",
        "get_unsynced_notes",
        "get_sync_status",
        "update_note_sync_metadata",
        "link_note_to_file",
        "unlink_note_from_file",
        "set_note_sync_enabled",
    }
    exposed = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in retired
    }

    assert exposed == set()


def test_production_routes_share_the_canonical_local_note_authority() -> None:
    from tldw_chatbook.Library.library_notes_lasting_sync_state import LastingSyncSetup

    app_source = Path("tldw_chatbook/app.py").read_text(encoding="utf-8")

    assert LastingSyncSetup().note_scope_id == "local_note"
    assert "note_scope_id=ScopeType.LOCAL_NOTE.value" in app_source
    assert 'note_scope_id="local"' not in app_source
    assert 'note_scope_id="local-notes"' not in app_source


@pytest.mark.parametrize(
    "relative_path",
    (
        "UI/Screens/library_screen.py",
        "UI/Tools_Settings_Window.py",
        "UI/Wizards/FirstRunSetupWizard.py",
        "UI/Wizards/first_run_setup_state.py",
    ),
)
def test_retired_product_surfaces_do_not_read_or_write_legacy_sync_config(
    relative_path: str,
) -> None:
    path = _PRODUCTION / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert sorted(values & _LEGACY_CONFIG_KEYS) == []


class _Adapter:
    pass


class _Coordinator:
    def __init__(self) -> None:
        self.calls = 0

    def try_acquire(self, *_args: object, **_kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("cutover fence admitted a root")


class _Watcher:
    async def run(self) -> None:
        raise AssertionError("cutover fence started the watcher")

    async def stop(self) -> None:
        return None


class _Admission:
    state = SimpleNamespace(name="OWNER")
    authoritative = True

    def require_authority(self, _operation: str) -> object:
        return self


class _SetupCoordinator:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.close_calls = 0

    def try_acquire(self, *_args: object, **_kwargs: object) -> _Admission:
        from tldw_chatbook.Notes.notes_sync_coordinator import RootAdmissionState

        self.acquire_calls += 1
        admission = _Admission()
        admission.state = RootAdmissionState.OWNER
        return admission

    def close_admission(self, _lease: object, settle: object) -> None:
        self.close_calls += 1
        if callable(settle):
            settle()


class _SetupAdapter:
    def __init__(self, *, with_file: bool = False) -> None:
        self.observed_states: list[str] = []
        self.created_folders: list[str] = []
        self.rolled_back_folders: list[object] = []
        self.with_file = with_file
        self.built_requests: list[object] = []
        self.executed_requests: list[object] = []

    async def observe_root(self, root: object) -> object:
        from tldw_chatbook.Notes.notes_sync_reconciler import (
            BindingObservation,
            ReconciliationInput,
        )

        bindings = ()
        if self.with_file:
            bindings = (
                BindingObservation(
                    binding_id="binding-new",
                    baseline_file_digest="a" * 64,
                    baseline_note_digest="a" * 64,
                    baseline_identity_digest="b" * 64,
                    baseline_relative_path="new.md",
                    file_digest="a" * 64,
                    note_digest=None,
                    file_identity_digest="b" * 64,
                    relative_path="new.md",
                    note_scope_id="local_note",
                    note_id="note-new",
                    note_version=0,
                    bound=False,
                ),
            )

        self.observed_states.append(getattr(root, "state").value)
        return ReconciliationInput(
            root_id=getattr(root, "root_id"),
            direction=getattr(root, "direction"),
            bindings=bindings,
            observation_generation=1,
            expected_generation=1,
            root_available=True,
            root_overlap=False,
            write_capable=True,
        )

    async def build_execution_request(self, *_args: object) -> object:
        request = _args[-1]
        self.built_requests.append(request)
        return request

    def executor_for(self, *_args: object, **_kwargs: object) -> object:
        adapter = self

        class _Executor:
            async def execute(self, request: object) -> object:
                from tldw_chatbook.Notes.notes_sync_models import (
                    NotesSyncOperationState,
                )

                adapter.executed_requests.append(request)
                return SimpleNamespace(
                    operation_id="operation-new",
                    state=NotesSyncOperationState.COMPLETED,
                    reason_code=None,
                )

        return _Executor()

    async def create_root_folder(self, display_name: str) -> tuple[str, object]:
        self.created_folders.append(display_name)
        return "folder-real", "folder-receipt"

    async def rollback_root_folder(self, receipt: object) -> None:
        self.rolled_back_folders.append(receipt)


@pytest.mark.asyncio
async def test_successful_migration_records_marker_before_runtime_admission(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import (
        CUTOVER_MARKER,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    events: list[str] = []
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: events.append("migrated"),
        coordinator=_Coordinator(),
        adapter=_Adapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )

    await owner.start()

    marker = store.get_setting("cutover_marker")
    assert events == ["migrated"]
    assert marker is not None and marker.value == CUTOVER_MARKER
    assert owner.snapshot().status == "active"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_migration_failure_never_records_the_cutover_marker(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: (_ for _ in ()).throw(RuntimeError("private")),
        coordinator=_Coordinator(),
        adapter=_Adapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )

    await owner.start()

    assert store.get_setting("cutover_marker") is None
    assert (owner.snapshot().status, owner.snapshot().next_action) == (
        "failed",
        "review_settings",
    )
    await owner.shutdown()


@pytest.mark.asyncio
async def test_other_profile_process_blocks_activation_until_restart(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    coordinator = _Coordinator()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=coordinator,
        adapter=_Adapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=False,
    )

    await owner.start()

    assert (owner.snapshot().status, owner.snapshot().next_action) == (
        "awaiting_cutover",
        "close_other_process_and_restart",
    )
    assert coordinator.calls == 0
    with pytest.raises(RuntimeError, match="cutover"):
        await owner.activate_root("root-1", authorization=None)
    await owner.shutdown()


@pytest.mark.asyncio
async def test_clean_install_requires_mutation_free_setup_review_before_activation(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    adapter = _SetupAdapter()
    coordinator = _SetupCoordinator()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=coordinator,
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    setup = NotesSyncRootSetup(
        display_name="Research",
        canonical_path=str(tmp_path / "research"),
        note_scope_id="local_note",
        direction=NotesSyncDirection.BIDIRECTIONAL,
    )

    review = await owner.review_setup(setup)

    assert store.list_root_summaries() == ()
    with pytest.raises(ValueError, match="current.*review|stale_review"):
        await owner.activate_root(review.root_id, "0" * 64)
    result = await owner.activate_root(review.root_id, review.observation_token)
    root = store.get_root(review.root_id)
    assert result.accepted is True
    assert root.state.value == "active"
    assert root.logical_folder_id == "folder-real"
    assert adapter.created_folders == ["Research"]
    assert adapter.observed_states == ["pending", "pending", "pending"]
    await owner.shutdown()


@pytest.mark.asyncio
async def test_activation_executes_the_exact_reviewed_safe_actions_before_success(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    adapter = _SetupAdapter(with_file=True)
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(tmp_path / "research"),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )

    result = await owner.activate_root(review.root_id, review.observation_token)

    assert result.accepted is True
    assert [request.action_id for request in adapter.executed_requests] == [
        action.action_id for action in review.safe_actions
    ]
    assert owner.snapshot().roots[0].status == "up_to_date"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_setup_activation_persists_authority_before_managed_folder_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    adapter = _SetupAdapter()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(tmp_path / "research"),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )
    monkeypatch.setattr(
        store,
        "create_root",
        lambda _record: (_ for _ in ()).throw(RuntimeError("private")),
    )

    with pytest.raises(RuntimeError, match="root_activation_persistence_failed"):
        await owner.activate_root(review.root_id, review.observation_token)

    assert adapter.created_folders == []
    assert adapter.rolled_back_folders == []
    assert store.list_root_summaries() == ()
    await owner.shutdown()


@pytest.mark.asyncio
async def test_failed_folder_rollback_retains_restart_visible_recovery_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    adapter = _SetupAdapter()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(tmp_path / "research"),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )
    monkeypatch.setattr(
        store,
        "assign_root_folder",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("private")),
    )

    async def fail_rollback(_receipt: object) -> None:
        raise RuntimeError("private rollback")

    monkeypatch.setattr(adapter, "rollback_root_folder", fail_rollback)
    result = await owner.activate_root(review.root_id, review.observation_token)

    assert (result.accepted, result.status, result.next_action) == (
        False,
        "needs_attention",
        "review_settings",
    )
    await owner.shutdown()
    reopened = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    retained = reopened.get_root(review.root_id)
    assert retained.logical_folder_id == "folder-real"
    assert retained.state.value == "paused"
    assert retained.last_status_code == "activation_recovery_required"


@pytest.mark.asyncio
async def test_folder_creation_failure_retires_setup_owner_and_allows_retry(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_models import (
        NotesSyncDirection,
        NotesSyncRootState,
    )
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    class _FailOnceFolderAdapter(_SetupAdapter):
        attempts = 0

        async def create_root_folder(self, display_name: str) -> tuple[str, object]:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("private folder creation failure")
            return await super().create_root_folder(display_name)

    database_path = tmp_path / "sync.sqlite3"
    store = NotesDeviceStateStore(database_path)
    adapter = _FailOnceFolderAdapter()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    setup = NotesSyncRootSetup(
        display_name="Research",
        canonical_path=str(tmp_path / "research"),
        note_scope_id="local_note",
        direction=NotesSyncDirection.BIDIRECTIONAL,
    )
    review = await owner.review_setup(setup)

    result = await owner.activate_root(review.root_id, review.observation_token)

    assert (result.accepted, result.status, result.next_action) == (
        False,
        "failed",
        "review_settings",
    )
    assert store.get_root(review.root_id).state is NotesSyncRootState.DISCONNECTED
    assert review.root_id not in owner._setup_reviews
    assert review.root_id not in owner._leases
    assert review.root_id not in owner._root_paths
    await owner.shutdown()

    reopened_store = NotesDeviceStateStore(database_path)
    reopened = NotesSyncRuntimeOwner(
        store=reopened_store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await reopened.start()
    assert reopened.snapshot().roots == ()
    assert reopened._root_paths == {}
    assert all(
        summary.state is NotesSyncRootState.DISCONNECTED
        for summary in reopened_store.list_root_summaries()
    )

    retry = await reopened.review_setup(setup)
    accepted = await reopened.activate_root(retry.root_id, retry.observation_token)

    assert retry.root_id != review.root_id
    assert accepted.accepted is True
    assert reopened_store.get_root(retry.root_id).state is NotesSyncRootState.ACTIVE
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_successful_folder_rollback_retires_setup_owner_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Notes.notes_sync_models import (
        NotesSyncDirection,
        NotesSyncRootState,
    )
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    database_path = tmp_path / "sync.sqlite3"
    store = NotesDeviceStateStore(database_path)
    adapter = _SetupAdapter()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    setup = NotesSyncRootSetup(
        display_name="Research",
        canonical_path=str(tmp_path / "research"),
        note_scope_id="local_note",
        direction=NotesSyncDirection.BIDIRECTIONAL,
    )
    review = await owner.review_setup(setup)
    monkeypatch.setattr(
        store,
        "assign_root_folder",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("private assignment")),
    )

    result = await owner.activate_root(review.root_id, review.observation_token)

    assert (result.accepted, result.status, result.next_action) == (
        False,
        "failed",
        "review_settings",
    )
    assert adapter.rolled_back_folders == ["folder-receipt"]
    assert store.get_root(review.root_id).state is NotesSyncRootState.DISCONNECTED
    assert review.root_id not in owner._setup_reviews
    assert review.root_id not in owner._leases
    assert review.root_id not in owner._root_paths
    await owner.shutdown()

    reopened_store = NotesDeviceStateStore(database_path)
    reopened = NotesSyncRuntimeOwner(
        store=reopened_store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await reopened.start()
    assert reopened.snapshot().roots == ()
    assert reopened._root_paths == {}

    retry = await reopened.review_setup(setup)
    accepted = await reopened.activate_root(retry.root_id, retry.observation_token)

    assert retry.root_id != review.root_id
    assert accepted.accepted is True
    assert reopened_store.get_root(retry.root_id).state is NotesSyncRootState.ACTIVE
    await reopened.shutdown()


@pytest.mark.asyncio
async def test_post_folder_stale_review_becomes_durable_visible_recovery(
    tmp_path: Path,
) -> None:
    from dataclasses import replace

    from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    class _DriftingSetupAdapter(_SetupAdapter):
        calls = 0

        async def observe_root(self, root: object) -> object:
            self.calls += 1
            observed = await super().observe_root(root)
            if self.calls >= 3:
                return replace(
                    observed,
                    observation_generation=2,
                    expected_generation=2,
                )
            return observed

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    adapter = _DriftingSetupAdapter()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(tmp_path / "research"),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )

    result = await owner.activate_root(review.root_id, review.observation_token)

    assert (result.accepted, result.status, result.next_action) == (
        False,
        "needs_attention",
        "review_settings",
    )
    retained = store.get_root(review.root_id)
    assert retained.logical_folder_id == "folder-real"
    assert retained.state.value == "paused"
    assert retained.last_status_code == "activation_recovery_required"
    assert review.root_id not in owner._setup_reviews
    assert owner.snapshot().roots[0].status == "needs_attention"
    await owner.shutdown()

    reopened = NotesSyncRuntimeOwner(
        store=NotesDeviceStateStore(tmp_path / "sync.sqlite3"),
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=_SetupAdapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await reopened.start()
    projected = reopened.snapshot().roots[0]
    assert (projected.status, projected.next_action) == (
        "needs_attention",
        "review_settings",
    )
    await reopened.shutdown()


def test_migration_activation_atomically_claims_exact_candidate_bindings(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_device_state_store import (
        NotesSyncBindingRecord,
        NotesSyncRootRecord,
    )
    from tldw_chatbook.Notes.notes_sync_models import (
        NotesSyncBindingState,
        NotesSyncDirection,
        NotesSyncRootState,
        NotesSyncSerializationProfile,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    store.initialize()
    root_id = "legacy-root-" + "a" * 40
    binding_id = "legacy-binding-" + "b" * 40
    store.create_root(
        NotesSyncRootRecord(
            root_id=root_id,
            note_scope_id="local_note",
            logical_folder_id=None,
            canonical_path=str(tmp_path / "legacy"),
            direction=NotesSyncDirection.BIDIRECTIONAL,
            state=NotesSyncRootState.PAUSED,
            last_status_code="migration_review_required",
        )
    )
    store.create_binding(
        NotesSyncBindingRecord(
            binding_id=binding_id,
            root_id=root_id,
            note_scope_id="local_note",
            note_id="note-1",
            normalized_relative_path="note.md",
            stable_identity_digest="c" * 64,
            state=NotesSyncBindingState.CANDIDATE,
            serialization=NotesSyncSerializationProfile(
                utf8_bom=False, newline="lf", final_newline=False, mode=0o644
            ),
            content_digest="d" * 64,
            note_version=1,
        )
    )

    activated = store.activate_migration_candidate(
        root_id,
        "folder-real",
        (binding_id,),
    )

    assert activated.state is NotesSyncRootState.ACTIVE
    assert activated.logical_folder_id == "folder-real"
    assert store.get_binding(binding_id).state is NotesSyncBindingState.ACTIVE


@pytest.mark.asyncio
async def test_rechecking_and_abandoning_setup_does_not_leak_a_root_lease(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    coordinator = _SetupCoordinator()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=coordinator,
        adapter=_SetupAdapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
    )
    await owner.start()
    setup = NotesSyncRootSetup(
        display_name="Research",
        canonical_path=str(tmp_path / "research"),
        note_scope_id="local_note",
        direction=NotesSyncDirection.BIDIRECTIONAL,
    )

    first = await owner.review_setup(setup)
    second = await owner.review_setup(setup)
    await owner.abandon_setup(first.root_id)

    assert second.root_id == first.root_id
    assert coordinator.acquire_calls == 1
    assert coordinator.close_calls == 1
    assert owner._leases == {}
    assert store.list_root_summaries() == ()
    await owner.shutdown()


def test_production_app_opens_both_restart_cutover_fences() -> None:
    path = _PRODUCTION / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    builds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _dotted_name(node.func).endswith("build_notes_sync_runtime_owner")
    ]

    assert len(builds) == 1
    keywords = {keyword.arg: keyword.value for keyword in builds[0].keywords}
    assert isinstance(keywords["cutover_admitted"], ast.Constant)
    assert keywords["cutover_admitted"].value is True
    process_gate = keywords["profile_process_is_sole"]
    assert isinstance(process_gate, ast.Attribute)
    assert _dotted_name(process_gate) == "self._instance_lock_status.acquired"


def test_production_app_builds_the_lasting_runtime_off_the_import_path() -> None:
    """TASK-21108: the runtime is built lazily, never in ``__init__``.

    Constructing the owner is what imported ``Notes/notes_sync_runtime`` and
    ``Notes/notes_sync_legacy`` (15 modules) at ``import tldw_chatbook.app``.
    Both imports now live inside ``_construct_notes_sync_runtime_owner``,
    which the ``notes_sync_runtime_owner`` property calls on first access. The
    residency half of this guard is
    ``Tests/Packaging/test_app_import_diet_closure.py``; this half pins the
    structure that produces it.
    """
    path = _PRODUCTION / "app.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Scoped to TldwCli's OWN method bodies: app.py defines 13 `__init__`
    # methods across its command providers, helper apps and classes nested
    # inside TldwCli methods, so neither a module-wide lookup nor an
    # `ast.walk` of the class picks the right one (both land on a nested
    # class's `__init__` and assert nothing).
    app_class = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
    )
    functions = {
        node.name: node
        for node in app_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    # `_construct_`, not `_build_`: see the method's own docstring -- a
    # `_build_...` wrapper would match the endswith() fence above.
    builder = functions["_construct_notes_sync_runtime_owner"]
    builds = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Call)
        and _dotted_name(node.func).endswith("build_notes_sync_runtime_owner")
    ]
    assert len(builds) == 1

    deferred = {
        (node.module, alias.name)
        for node in ast.walk(builder)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert ("Notes.notes_sync_runtime", "build_notes_sync_runtime_owner") in deferred
    assert ("Notes.notes_sync_legacy", "legacy_sync_directory_configured") in deferred

    # Neither module may be imported at app module scope any more.
    module_scope_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "Notes.notes_sync_runtime" not in module_scope_imports
    assert "Notes.notes_sync_legacy" not in module_scope_imports
    assert "tldw_chatbook.Notes.notes_sync_runtime" not in module_scope_imports
    assert "tldw_chatbook.Notes.notes_sync_legacy" not in module_scope_imports

    init = functions["__init__"]
    assert not [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and _dotted_name(node.func).endswith(
            ("build_notes_sync_runtime_owner", "build_notes_sync_legacy_migrator")
        )
    ], "TldwCli.__init__ must not build the lasting-sync runtime"


def test_lasting_runtime_property_builds_once_and_accepts_a_double() -> None:
    """TASK-21108: the lazy property memoizes, and assignment still works.

    Tests install runtime doubles with ``app.notes_sync_runtime_owner = ...``;
    replacing the eager attribute with a property would silently break that
    if the setter were dropped.
    """
    import threading

    from tldw_chatbook.app import TldwCli

    class _Host:
        notes_sync_runtime_owner = TldwCli.notes_sync_runtime_owner

        def __init__(self) -> None:
            self._notes_sync_runtime_owner = None
            self._notes_sync_runtime_owner_lock = threading.Lock()
            self.builds = 0

        def _construct_notes_sync_runtime_owner(self) -> object:
            self.builds += 1
            return SimpleNamespace(tag="built")

    host = _Host()
    assert host.builds == 0, "the property must not build before first access"
    first = host.notes_sync_runtime_owner
    second = host.notes_sync_runtime_owner
    assert first is second
    assert host.builds == 1

    double = SimpleNamespace(tag="double")
    host.notes_sync_runtime_owner = double
    assert host.notes_sync_runtime_owner is double
    assert host.builds == 1


def test_missing_lasting_runtime_has_no_legacy_fallback() -> None:
    screen = (_PRODUCTION / "UI" / "Screens" / "library_screen.py").read_text(
        encoding="utf-8"
    )

    assert "InertLastingSyncRuntime" in screen
    assert "NotesSyncEngine" not in screen
    assert "NotesSyncService" not in screen
    assert "sync_service" not in screen


@pytest.mark.asyncio
async def test_unconfigured_start_defers_without_creating_the_state_db(
    tmp_path: Path,
) -> None:
    """TASK-21112: a zero-profile boot must not create notes-sync state."""

    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    migrations: list[str] = []
    coordinator = _Coordinator()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: migrations.append("migrated"),
        coordinator=coordinator,
        adapter=_Adapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
        start_evidence=lambda: False,
    )

    await owner.start()

    assert not (tmp_path / "sync.sqlite3").exists()
    assert (owner.snapshot().status, owner.snapshot().next_action) == (
        "not_configured",
        "none",
    )
    assert migrations == []
    assert coordinator.calls == 0
    with pytest.raises(RuntimeError, match="cutover"):
        await owner.activate_root("root-1", authorization=None)

    await owner.shutdown()

    assert not (tmp_path / "sync.sqlite3").exists()
    assert owner.snapshot().status == "stopped"


@pytest.mark.asyncio
async def test_forced_start_brings_up_a_previously_deferred_runtime(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    migrations: list[str] = []
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: migrations.append("migrated"),
        coordinator=_SetupCoordinator(),
        adapter=_SetupAdapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
        start_evidence=lambda: False,
    )

    await owner.start()
    assert owner.snapshot().status == "not_configured"
    assert not (tmp_path / "sync.sqlite3").exists()

    await owner.start(force=True)

    assert (tmp_path / "sync.sqlite3").exists()
    assert migrations == ["migrated"]
    assert owner.snapshot().status == "active"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_review_setup_live_starts_a_deferred_runtime(tmp_path: Path) -> None:
    """Activating the first root at runtime brings the machinery up on demand."""

    from tldw_chatbook.Notes.notes_sync_models import NotesSyncDirection
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootSetup,
        NotesSyncRuntimeOwner,
    )

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    adapter = _SetupAdapter()
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_SetupCoordinator(),
        adapter=adapter,
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
        start_evidence=lambda: False,
    )
    await owner.start()
    assert owner.snapshot().status == "not_configured"

    review = await owner.review_setup(
        NotesSyncRootSetup(
            display_name="Research",
            canonical_path=str(tmp_path / "research"),
            note_scope_id="local_note",
            direction=NotesSyncDirection.BIDIRECTIONAL,
        )
    )

    assert owner.snapshot().status == "active"
    assert (tmp_path / "sync.sqlite3").exists()
    result = await owner.activate_root(review.root_id, review.observation_token)
    assert result.accepted is True
    assert store.get_root(review.root_id).state.value == "active"
    await owner.shutdown()


@pytest.mark.asyncio
async def test_start_evidence_probe_failure_fails_open_and_starts(
    tmp_path: Path,
) -> None:
    """A broken probe must never silently disable a configured user's sync."""

    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

    def broken_probe() -> bool:
        raise RuntimeError("private probe failure")

    store = NotesDeviceStateStore(tmp_path / "sync.sqlite3")
    owner = NotesSyncRuntimeOwner(
        store=store,
        migrate_legacy=lambda: None,
        coordinator=_Coordinator(),
        adapter=_Adapter(),
        watcher_factory=lambda _schedule: _Watcher(),
        cutover_admitted=True,
        profile_process_is_sole=True,
        start_evidence=broken_probe,
    )

    await owner.start()

    assert owner.snapshot().status == "active"
    assert (tmp_path / "sync.sqlite3").exists()
    await owner.shutdown()
