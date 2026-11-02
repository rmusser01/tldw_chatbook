"""Actual consent workers retain native ownership through readiness and close."""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.execution_capacity import WorkOrigin, current_execution_owner
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import capture_change_review_admission
from tldw_chatbook.Workspaces.change_review_consent import RootReadinessState
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

pytestmark = pytest.mark.bootstrap_profile


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "native initializer did not settle"
        time.sleep(0.005)


@pytest.fixture
def native_initializer(tmp_path, monkeypatch):
    from tldw_chatbook import app_service_wiring
    from tldw_chatbook.config import load_settings

    app = SimpleNamespace(app_config=load_settings())
    runtime = app.console_runtime = ConsoleRuntime(app)
    monkeypatch.setattr(
        app_service_wiring, "get_workspaces_db_path", lambda: tmp_path / "workspaces.db"
    )
    app_service_wiring.ServiceWiringMixin._wire_workspace_registry_services(app)
    registry = app.workspace_registry_service
    assert registry is not None
    service = app.change_review_consent_service
    registry.create_workspace(workspace_id="init-workspace", name="Initialize")
    root = tmp_path / "root"
    root.mkdir()
    (root / "example.txt").write_text("original\n")
    binding = registry.add_folder_binding("init-workspace", root)
    host = SimpleNamespace(**locals())
    try:
        yield host
    finally:
        service.shutdown(timeout=5)
        asyncio.run(runtime.dispose())


@pytest.mark.parametrize("cause", ["lazy", "settings"])
def test_actual_initializer_and_handoff_retain_one_native_owner(
    native_initializer, monkeypatch, cause
):
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.Workspaces import change_turn_tracker

    h = native_initializer
    with storage_admission._lock:
        original_leases = set(storage_admission._live_leases)
    entered, release = threading.Event(), threading.Event()
    handed, finish = threading.Event(), threading.Event()
    seen = []
    initialize = change_turn_tracker.initialize_shadow_root
    complete = h.service._complete

    def held_initialize(root, **kwargs):
        seen.append(current_execution_owner())
        entered.set()
        assert release.wait(5)
        return initialize(root, **kwargs)

    def held_complete(work, error):
        handed.set()
        assert finish.wait(5)
        return complete(work, error)

    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", held_initialize)
    monkeypatch.setattr(h.service, "_complete", held_complete)
    try:
        if cause == "lazy":
            h.registry.set_change_review_enabled("init-workspace", True)
            roots, _, skipped = capture_change_review_admission(h.app, "init-workspace")
            assert roots == () and skipped
        else:
            h.service.toggle(
                "init-workspace",
                expected=h.registry.read_change_review_consent("init-workspace"),
                enabled=True,
            )
        assert entered.wait(3)
        with storage_admission._lock:
            worker_leases = storage_admission._live_leases - original_leases
        assert len(worker_leases) >= 3, "original feature storage was not retained"
        owner = seen[0]
        assert owner is not None, "actual initializer entered without native ownership"
        assert owner.origin is (
            WorkOrigin.AUTOMATIC if cause == "lazy" else WorkOrigin.MANUAL
        )
        assert owner.conversation_id == "" and not owner.child and owner.run_id is None
        assert h.runtime._chat_store is None and h.runtime._agent_bridge is None
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        release.set()
        assert handed.wait(3)
        with storage_admission._lock:
            assert worker_leases <= storage_admission._live_leases
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None, (
                "initializer return retired unfinished readiness handoff"
            )
        finish.set()
        wait_until(
            lambda: (
                h.service.status("init-workspace").roots[0].state
                is RootReadinessState.READY
            )
        )
        wait_until(lambda: h.runtime.execution_capacity.snapshot().executions == ())
        with storage_admission._lock:
            assert worker_leases.isdisjoint(storage_admission._live_leases)
    finally:
        release.set()
        finish.set()


def test_disable_rejects_stale_admission_without_holding_bookkeeping_lock(
    native_initializer, monkeypatch
):
    h = native_initializer
    h.registry.set_change_review_enabled("init-workspace", True)
    expected = h.registry.read_change_review_consent("init-workspace")
    entered, release, disabled = (threading.Event() for _ in range(3))
    original = h.registry.read_change_review_consent
    reads, errors, admissions = [], [], []

    def held_read(workspace_id):
        value = original(workspace_id)
        reads.append(value)
        if threading.current_thread().name == "stale-consent-reader":
            entered.set()
            assert release.wait(5)
        return value

    monkeypatch.setattr(h.registry, "read_change_review_consent", held_read)

    def read():
        try:
            admissions.append(h.service.admit_turn("init-workspace"))
        except BaseException as error:  # noqa: BLE001 -- surface thread failures on the test thread
            errors.append(error)

    def disable():
        try:
            h.service.toggle("init-workspace", expected=expected, enabled=False)
            disabled.set()
        except BaseException as error:  # noqa: BLE001 -- surface thread failures on the test thread
            errors.append(error)

    reader = threading.Thread(target=read, name="stale-consent-reader")
    toggle = threading.Thread(target=disable)
    reader.start()
    try:
        assert entered.wait(3)
        toggle.start()
        assert disabled.wait(1), "registry callback held the consent bookkeeping lock"
    finally:
        release.set()
        reader.join(5)
        if toggle.ident is not None:
            toggle.join(5)
    assert not errors
    assert admissions[0].ready_roots == () and admissions[0].skipped_roots == ()
    assert h.service._queue.unfinished_tasks == 0
    assert not h.service._workers_started


def test_automatic_initializer_preserves_manual_slot_and_retry_uses_it(
    native_initializer,
):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

    h = native_initializer
    h.runtime._execution_capacity = RuntimeCapacity(
        max_tool_workers=1, reserved_manual_tool_workers=1
    )
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    wait_until(
        lambda: (
            h.service.status("init-workspace").roots[0].state
            is RootReadinessState.FAILED
        )
    )
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert "busy" in h.service.status("init-workspace").roots[0].reason
    assert h.service.retry_failed_roots("init-workspace") == 1
    wait_until(
        lambda: (
            h.service.status("init-workspace").roots[0].state
            is RootReadinessState.READY
        )
    )
    wait_until(lambda: h.runtime.execution_capacity.snapshot().executions == ())


def test_activity_busy_never_enters_initializer(native_initializer, monkeypatch):
    from tldw_chatbook.Workspaces import change_turn_tracker

    h = native_initializer
    calls = []
    original = change_turn_tracker.initialize_shadow_root

    def observed(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", observed)
    owner = h.runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="competitor"
    )
    claim = h.runtime.worktree_activity.try_acquire(
        owner=owner, footprint=WorktreeFootprint(opaque=True)
    )
    assert claim is not None
    try:
        h.registry.set_change_review_enabled("init-workspace", True)
        capture_change_review_admission(h.app, "init-workspace")
        wait_until(
            lambda: (
                h.service.status("init-workspace").roots[0].state
                is RootReadinessState.FAILED
            )
        )
        assert calls == []
        assert "busy" in h.service.status("init-workspace").roots[0].reason
    finally:
        claim.settle(durable=True)
        owner.finish_root()


@pytest.mark.parametrize(
    "replacement", ["capacity", "activity", "registry", "callback", "binding"]
)
def test_original_source_replacement_before_first_dispatch_refuses(
    native_initializer, monkeypatch, replacement
):
    from dataclasses import replace

    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Workspaces import change_turn_tracker
    from tldw_chatbook.Workspaces.change_review_ownership import _InitializationSource
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivity

    h = native_initializer
    # These are the original resources captured before the first queued callback.
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    entered, release = threading.Event(), threading.Event()
    calls = []
    original_admit = _InitializationSource.admit
    original_init = change_turn_tracker.initialize_shadow_root

    def held(source):
        entered.set()
        assert release.wait(5)
        return original_admit(source)

    def observed(*args, **kwargs):
        calls.append(args)
        return original_init(*args, **kwargs)

    monkeypatch.setattr(_InitializationSource, "admit", held)
    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", observed)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    try:
        assert entered.wait(3)
        if replacement == "capacity":
            monkeypatch.setattr(h.runtime, "_execution_capacity", RuntimeCapacity())
        elif replacement == "activity":
            monkeypatch.setattr(h.runtime, "_worktree_activity", WorktreeActivity())
        elif replacement == "registry":
            monkeypatch.setattr(h.app, "workspace_registry_service", object())
        elif replacement == "callback":
            monkeypatch.setattr(
                h.service, "_initialize_root", lambda root: calls.append(root)
            )
        else:
            h.registry.save_runtime_binding(
                replace(h.binding, metadata={"access": "rw"})
            )
        release.set()
        wait_until(
            lambda: (
                h.service.status("init-workspace").roots[0].state
                is RootReadinessState.FAILED
            )
        )
        assert calls == []
        assert capacity.snapshot().executions == ()
        with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
            assert claim is not None
    finally:
        release.set()


def test_runtime_close_retains_actual_worker_after_bounded_join_and_cancellation(
    native_initializer, monkeypatch
):
    from tldw_chatbook.Workspaces import change_turn_tracker

    h = native_initializer
    entered, release = threading.Event(), threading.Event()
    original = change_turn_tracker.initialize_shadow_root

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", held)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")

    async def close():
        assert await asyncio.to_thread(entered.wait, 3)
        h.service.shutdown(timeout=0)
        operation = asyncio.create_task(h.runtime.dispose(timeout_seconds=0))
        try:
            for _ in range(100):
                if h.runtime.execution_capacity.snapshot().closed:
                    break
                await asyncio.sleep(0.005)
            assert h.runtime.execution_capacity.snapshot().closed
            operation.cancel()
            await asyncio.sleep(0.03)
            operation.cancel()
            await asyncio.sleep(0.03)
            assert not operation.done(), (
                "cancelled waiter retired the actual initializer"
            )
            assert h.runtime.execution_capacity.snapshot().tool_workers == 1
            with h.runtime.worktree_activity.prelaunch(
                footprint=WorktreeFootprint(opaque=True)
            ) as claim:
                assert claim is None
        finally:
            release.set()
            try:
                await asyncio.wait_for(asyncio.shield(operation), 5)
            except asyncio.CancelledError:
                pass
        assert h.runtime.execution_capacity.snapshot().executions == ()
        assert all(not worker.is_alive() for worker in h.service._workers)
        assert h.service._readiness == {}

    try:
        asyncio.run(close())
    finally:
        release.set()


def test_real_sqlite_completion_failure_is_not_ready_and_does_not_kill_worker(
    native_initializer, monkeypatch
):
    from tldw_chatbook.Workspaces import change_turn_tracker

    h = native_initializer
    original = change_turn_tracker.initialize_shadow_root

    def lose_consent_storage(*args, **kwargs):
        result = original(*args, **kwargs)
        with h.registry.db.connection() as conn:
            conn.execute(
                "ALTER TABLE workspace_change_review RENAME TO unavailable_consent"
            )
            conn.commit()
        return result

    monkeypatch.setattr(
        change_turn_tracker, "initialize_shadow_root", lose_consent_storage
    )
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    try:
        wait_until(lambda: h.service._queue.unfinished_tasks == 0)
        assert (
            h.service._readiness[("init-workspace", str(h.root))].state
            is RootReadinessState.FAILED
        )
        assert h.runtime.execution_capacity.snapshot().executions == ()
        assert all(worker.is_alive() for worker in h.service._workers)
    finally:
        with h.registry.db.connection() as conn:
            conn.execute(
                "ALTER TABLE unavailable_consent RENAME TO workspace_change_review"
            )
            conn.commit()


def test_wiring_is_inert_and_busy_capture_does_not_secure_new_data_directory(
    native_initializer, monkeypatch
):
    from tldw_chatbook import config

    h = native_initializer
    assert h.runtime._execution_capacity is None
    assert h.runtime._worktree_activity is None
    assert not h.service._workers_started
    secured = []
    original = config.get_user_data_dir

    def observed():
        secured.append(threading.current_thread())
        return original()

    monkeypatch.setattr(config, "get_user_data_dir", observed)
    owner = h.runtime.execution_capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="busy"
    )
    claim = h.runtime.worktree_activity.try_acquire(
        owner=owner, footprint=WorktreeFootprint(opaque=True)
    )
    try:
        h.registry.set_change_review_enabled("init-workspace", True)
        capture_change_review_admission(h.app, "init-workspace")
        wait_until(
            lambda: (
                h.service.status("init-workspace").roots[0].state
                is RootReadinessState.FAILED
            )
        )
        assert secured == [], (
            "metadata capture secured storage before activity admission"
        )
    finally:
        claim.settle(durable=True)
        owner.finish_root()


@pytest.mark.parametrize("change", ["close", "profile"])
def test_dequeued_unadmitted_work_cannot_launch_after_close_or_profile_drift(
    native_initializer, monkeypatch, change
):
    from tldw_chatbook.Workspaces import change_turn_tracker
    from tldw_chatbook.Workspaces.change_review_ownership import _InitializationSource

    h = native_initializer
    entered, release = threading.Event(), threading.Event()
    original = _InitializationSource.admit
    calls = []
    initialize = change_turn_tracker.initialize_shadow_root

    def held(source):
        entered.set()
        assert release.wait(5)
        return original(source)

    def observed(*args, **kwargs):
        calls.append(args)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(_InitializationSource, "admit", held)
    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", observed)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    try:
        assert entered.wait(3)
        if change == "close":
            h.service.begin_close()
            h.runtime.execution_capacity.close()
        else:
            monkeypatch.setenv(
                "TLDW_CONFIG_PATH", str(h.tmp_path / "not-selected.toml")
            )
        release.set()
        wait_until(lambda: h.service._queue.unfinished_tasks == 0)
        assert calls == []
        assert not (h.tmp_path / "not-selected.toml").exists()
        if h.runtime._execution_capacity is not None:
            assert h.runtime._execution_capacity.snapshot().executions == ()
    finally:
        release.set()
        monkeypatch.undo()


@pytest.mark.parametrize(
    "config_line",
    [
        '[filter "blocked"]\nclean = forbidden-helper',
        '[remote "partial"]\npromisor = true',
    ],
)
def test_actual_native_initializer_uses_helper_refusing_git(
    native_initializer, monkeypatch, config_line
):
    import subprocess

    from tldw_chatbook.Workspaces.change_tracking import ShadowRepo

    h = native_initializer
    git_home = h.tmp_path / "git-home"
    git_home.mkdir()
    (git_home / ".gitconfig").write_text(config_line + "\n")
    (h.root / ".gitattributes").write_text("*.txt filter=blocked\n")
    environment = ShadowRepo._env
    monkeypatch.setattr(
        ShadowRepo,
        "_env",
        lambda repo: {
            **environment(repo),
            "HOME": str(git_home),
            "XDG_CONFIG_HOME": str(git_home),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    run = subprocess.run
    commands = []

    def guarded(command, **kwargs):
        commands.append(command)
        if "add" in command or "update-index" in command or "cat-file" in command:
            raise AssertionError("helper-capable command entered before native refusal")
        return run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    wait_until(lambda: h.service._queue.unfinished_tasks == 0)
    assert any("--name-only" in command for command in commands), (
        "native qualification never ran"
    )
    assert not any(
        "add" in command or "update-index" in command or "cat-file" in command
        for command in commands
    )
    assert (
        h.service.status("init-workspace").roots[0].state is RootReadinessState.FAILED
    )
    assert h.runtime.execution_capacity.snapshot().executions == ()


@pytest.mark.parametrize("change", ["disable_reenable", "registry"])
def test_admitted_callback_cannot_publish_ready_after_original_source_changes(
    native_initializer, monkeypatch, change
):
    from tldw_chatbook.Workspaces import change_turn_tracker

    h = native_initializer
    entered, release = threading.Event(), threading.Event()
    initialize = change_turn_tracker.initialize_shadow_root

    def held(*args, **kwargs):
        result = initialize(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", held)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    try:
        assert entered.wait(3)
        if change == "disable_reenable":
            disabled = h.service.toggle(
                "init-workspace",
                expected=h.registry.read_change_review_consent("init-workspace"),
                enabled=False,
            )
            h.service.toggle("init-workspace", expected=disabled, enabled=True)
            wait_until(
                lambda: (
                    h.service.status("init-workspace").roots[0].state
                    is RootReadinessState.FAILED
                )
            )
        else:
            monkeypatch.setattr(h.app, "workspace_registry_service", object())
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        release.set()
        wait_until(lambda: h.service._queue.unfinished_tasks == 0)
        assert (
            h.service.status("init-workspace").roots[0].state
            is RootReadinessState.FAILED
        )
        assert h.runtime.execution_capacity.snapshot().executions == ()
        if change == "disable_reenable":
            assert h.service.retry_failed_roots("init-workspace") == 1
            wait_until(
                lambda: (
                    h.service.status("init-workspace").roots[0].state
                    is RootReadinessState.READY
                )
            )
    finally:
        release.set()


def test_raised_completion_failure_retires_once_and_existing_worker_can_retry(
    native_initializer, monkeypatch
):
    h = native_initializer
    complete = h.service._complete

    def failed(work, error):
        with h.registry.db.connection() as conn:
            conn.execute("SELECT no_such_column FROM workspaces")
        return complete(work, error)

    monkeypatch.setattr(h.service, "_complete", failed)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    wait_until(lambda: h.service._queue.unfinished_tasks == 0)
    assert (
        h.service.status("init-workspace").roots[0].state is RootReadinessState.FAILED
    )
    assert h.runtime.execution_capacity.snapshot().executions == ()
    monkeypatch.setattr(h.service, "_complete", complete)
    assert h.service.retry_failed_roots("init-workspace") == 1
    wait_until(
        lambda: (
            h.service.status("init-workspace").roots[0].state
            is RootReadinessState.READY
        )
    )
