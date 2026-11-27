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
    run = ShadowRepo._run_process
    commands = []

    def guarded(repo, command, env, **kwargs):
        commands.append(command)
        if "add" in command or "update-index" in command or "cat-file" in command:
            raise AssertionError("helper-capable command entered before native refusal")
        return run(repo, command, env, **kwargs)

    monkeypatch.setattr(ShadowRepo, "_run_process", guarded)
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
    root = h.service.status("init-workspace").roots[0]
    assert root.state is RootReadinessState.FAILED
    # TASK-42101: the turn's "change review skipped" row and Settings show this
    # reason, so it names the actionable cause in plain words.
    expected = (
        "Files here use configured Git filters (such as Git LFS), which Change"
        " Review never runs."
        if config_line.startswith("[filter")
        else "uses partial-clone remotes"
    )
    assert expected in root.reason, root.reason
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


@pytest.mark.parametrize("boundary", ["handoff", "consent_read", "root_identity_read"])
def test_original_source_replacement_during_readiness_handoff_is_not_ready(
    native_initializer, monkeypatch, boundary
):
    from tldw_chatbook.Workspaces import change_review_ownership

    h = native_initializer
    entered, release = threading.Event(), threading.Event()
    local = threading.local()
    complete = h.service._complete
    read_consent = h.registry.read_change_review_consent
    capture_chain = change_review_ownership.capture_directory_chain

    def hold():
        entered.set()
        assert release.wait(5)

    def held_complete(work, error):
        assert error is None
        local.completing = True
        try:
            if boundary == "handoff":
                hold()
            return complete(work, error)
        finally:
            local.completing = False

    def held_read(workspace_id):
        observed = read_consent(workspace_id)
        if boundary == "consent_read" and getattr(local, "completing", False):
            hold()
        return observed

    def held_chain(root):
        observed = capture_chain(root)
        if boundary == "root_identity_read" and getattr(local, "completing", False):
            hold()
        return observed

    monkeypatch.setattr(h.service, "_complete", held_complete)
    monkeypatch.setattr(h.registry, "read_change_review_consent", held_read)
    monkeypatch.setattr(change_review_ownership, "capture_directory_chain", held_chain)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    try:
        assert entered.wait(3), "readiness consumer did not reach its held boundary"
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        monkeypatch.setattr(h.app, "workspace_registry_service", object())
        release.set()
        wait_until(lambda: h.service._queue.unfinished_tasks == 0)
        assert h.runtime.execution_capacity.snapshot().executions == ()
        assert (
            h.service._readiness[("init-workspace", str(h.root))].state
            is RootReadinessState.FAILED
        )
        assert h.service.admit_turn("init-workspace").ready_roots == ()
    finally:
        release.set()


def test_source_replacement_while_readiness_waits_for_publication_lock(
    native_initializer, monkeypatch
):
    import inspect
    import sys

    h = native_initializer
    checked, proceed = threading.Event(), threading.Event()
    local = threading.local()
    complete = h.service._complete
    issuer_check = h.service._native.check
    worker_ids = []
    lines, first_line = inspect.getsourcelines(complete)
    publication_line = first_line + next(
        index for index, line in enumerate(lines) if line.strip() == "with self._lock:"
    )

    def completing(work, error):
        local.completing = True
        local.checks = 0
        try:
            return complete(work, error)
        finally:
            local.completing = False

    def observed_check():
        result = issuer_check()
        if getattr(local, "completing", False):
            local.checks += 1
            if local.checks == 2:
                worker_ids.append(threading.get_ident())
                checked.set()
                assert proceed.wait(5)
        return result

    def at_publication_lock():
        frame = sys._current_frames().get(worker_ids[0])
        return (
            frame is not None
            and frame.f_code is complete.__func__.__code__
            and frame.f_lineno == publication_line
        )

    monkeypatch.setattr(h.service, "_complete", completing)
    monkeypatch.setattr(h.service._native, "check", observed_check)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    try:
        assert checked.wait(3), "consumer did not finish its genuine source check"
        with h.service._lock:
            proceed.set()
            wait_until(at_publication_lock)
            assert h.runtime.execution_capacity.snapshot().tool_workers == 1
            monkeypatch.setattr(h.app, "workspace_registry_service", object())
        wait_until(lambda: h.service._queue.unfinished_tasks == 0)
        assert h.runtime.execution_capacity.snapshot().executions == ()
        assert (
            h.service._readiness[("init-workspace", str(h.root))].state
            is RootReadinessState.FAILED
        )
        assert h.service.admit_turn("init-workspace").ready_roots == ()
    finally:
        proceed.set()


_RESTART_COPY = (
    "Change history waits for an app restart: an earlier cleanup could not "
    "be proven, so its workspace stays held. This turn continues without it."
)


def test_retained_cleanups_hold_later_roots_until_restart_and_say_so(
    native_initializer, monkeypatch, tmp_path
):
    """Two unproven cleanups on an opaque turn's owner stop later roots (TASK-41635).

    The turn's owner and its opaque claim stay held until restart (TASK-40716),
    so a later root is refused inside the turn (its tool reservation) and
    outside it (the opaque claim). Every such root says it waits for a restart
    and is not offered a retry. The cleanups hold daemon custody workers, not
    the pool's threads, so the retained roots do not read "preparing" forever.
    """
    from tldw_chatbook.Agents.execution_capacity import (
        bind_ordinary_tree,
        holds_unproven_custody,
    )
    from tldw_chatbook.Workspaces import change_turn_tracker
    from tldw_chatbook.Workspaces.change_review_ownership import (
        native_snapshot_ticket,
    )

    h = native_initializer
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    owner = capacity.begin_execution(origin=WorkOrigin.AUTOMATIC, conversation_id="t")
    footprint = WorktreeFootprint(opaque=True)
    claim = activity.try_acquire(owner=owner, footprint=footprint)
    assert claim is not None

    class OpaqueTurn:
        """The borrow of an opaque ordinary turn (Agents/ordinary_tree.py)."""

        controller = None

        def __init__(self):
            self.owner = owner

        def borrow(self, runtime, *, opaque=True):
            assert runtime is h.runtime and opaque
            activity.validate_claim(claim, owner=owner, footprint=footprint)
            return owner

    turn = OpaqueTurn()
    initialize = change_turn_tracker.initialize_shadow_root
    held_roots: set[str] = set()
    both_reserved = threading.Barrier(2, timeout=10)
    holders: list[threading.Thread] = []

    def unproven(root, **kwargs):
        if root not in held_roots:
            return initialize(root, **kwargs)
        both_reserved.wait()
        holders.append(threading.current_thread())
        # As run_native_git does when no cleanup proof remains.
        native_snapshot_ticket().operation.retain_unproven("native Git").hold()
        raise OSError("native Git cleanup proof unavailable")

    def bind(name):
        path = tmp_path / name
        path.mkdir()
        return h.registry.add_folder_binding("init-workspace", path)

    def readiness():
        return {
            root.alias: (root.state.value, root.reason)
            for root in h.service.status("init-workspace").roots
        }

    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", unproven)
    second = bind("second")
    held_roots.update({h.binding.locator, second.locator})
    h.registry.set_change_review_enabled("init-workspace", True)
    try:
        with bind_ordinary_tree(turn):
            h.service.admit_turn("init-workspace")
        wait_until(lambda: len(capacity.retained()) == 2, timeout=15)
        # Binding a folder admits it at once (binding_added).
        outside = bind("outside")  # its own owner: the turn's claim refuses it
        with bind_ordinary_tree(turn):
            inside = bind("inside")  # the turn's owner: its reservation refuses

        expected = {
            binding.binding_id: ("restart_required", _RESTART_COPY)
            for binding in (h.binding, second, outside, inside)
        }
        wait_until(
            lambda: len(readiness()) == 4
            and all(state != "preparing" for state, _ in readiness().values()),
            timeout=15,
        )
        assert readiness() == expected
        assert h.service.retry_failed_roots("init-workspace") == 0
        assert readiness() == expected
        custody = [t for t in threading.enumerate() if holds_unproven_custody(t)]
        assert len(custody) == 2
        assert not set(custody) & set(h.service._workers)
        assert h.service._queue.unfinished_tasks == 0
    finally:
        for retention in capacity.retained():
            retention.prove()
        wait_until(
            lambda: not any(holds_unproven_custody(t) for t in threading.enumerate()),
            timeout=15,
        )
        # Leaving hold() is not the end: the worker still completes and
        # returns its ticket, which must finish before the fixture tears down.
        for holder in holders:
            holder.join(15)
            assert not holder.is_alive(), "custody worker did not return"
        owner.finish_root()
        claim.settle(durable=True)


_BUSY_COPY = "Workspace activity is busy; retry after it finishes."
_HELD_COPY = (
    "Workspace activity is held by a cleanup that could not be proven; "
    "restart Chatbook to clear it."
)


def _member_of_claim(activity, owner, *, opaque=False):
    """Whether ``owner`` is a member of a held claim (an opaque one, if asked)."""
    with activity._lock:
        for member in activity._members.values():
            state = activity._claims.get(member.claim_id)
            if member.owner is owner and state is not None:
                if not opaque or state.footprint.opaque:
                    return True
    return False


def _console_rig(h, monkeypatch, tmp_path):
    """A real Console controller over the fixture's runtime, with one chat in
    the init workspace and one in another workspace."""
    from Tests.Chat.test_console_skill_substitution import _RecordingGateway
    from Tests.console_provider_doubles import persisted_console_store
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
    from tldw_chatbook.Tools import workspace_file_roots as roots_api

    monkeypatch.setattr(roots_api, "_registry_factory", lambda: h.registry)
    monkeypatch.setattr(roots_api, "_LAUNCH_CWD", str(h.root))
    store = persisted_console_store(
        db_path=tmp_path / "chat.db", workspace_registry=h.registry
    )
    gateway = _RecordingGateway()
    controller = ConsoleChatController(
        store=store, provider_gateway=gateway, provider="llama_cpp", model="m"
    )
    h.app.chachanotes_db = store.persistence.db
    h.app.local_chat_conversation_service = ChatConversationService(
        store.persistence.db
    )
    h.runtime.set_chat_store(store)
    h.runtime.set_chat_controller(controller)
    h.registry.create_workspace(workspace_id="elsewhere", name="Elsewhere")

    async def send(chat, text):
        result = await controller.submit_draft(text, session_id=chat.id)
        return result.accepted, result.visible_copy

    async def admitted(chat, text):
        accepted, copy = await send(chat, text)
        assert accepted, copy
        stream = controller._active_stream_tasks.get(chat.id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
        await asyncio.to_thread(wait_until, lambda: not h.runtime._ordinary_turns, 15)

    return SimpleNamespace(
        controller=controller,
        gateway=gateway,
        session=store.create_session(workspace_id="init-workspace"),
        elsewhere=store.create_session(workspace_id="elsewhere"),
        send=send,
        admitted=admitted,
        sandbox=lambda: roots_api.run_file_sandbox(h.root),
    )


@pytest.mark.parametrize(
    "holder", ["initialization", "tool_operation", "child_member", "marked_unproven"]
)
def test_a_retained_cleanups_claim_refuses_later_console_sends_until_restart(
    native_initializer, monkeypatch, tmp_path, holder
):
    """Owner decisions 2026-11-25 (TASK-41950, ADR-134): a claim with a member
    whose cleanup is unproven stays held until restart (TASK-40716): an
    initialization job's own opaque claim, an agent turn's claim whose tool
    operation (or a sub-agent's) was retained, or one whose owner was marked
    unproven with no retained record. A later Console Send is refused at
    ordinary-turn admission, in the job's workspace and in any other, and
    says that a restart clears it; Release frees only the tool slot, and not
    even a proof changes that. An ordinary busy conflict keeps the "retry"
    copy, and the same rig admits a Send once that conflict is gone."""
    from tldw_chatbook.Agents.execution_capacity import holds_unproven_custody
    from tldw_chatbook.Chat.console_unproven_cleanup import release_held_tool_slot
    from tldw_chatbook.Workspaces import change_turn_tracker
    from tldw_chatbook.Workspaces.change_review_ownership import (
        native_snapshot_ticket,
    )

    h = native_initializer
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    owners = []  # finished in teardown
    workers: list[threading.Thread] = []  # custody workers, joined

    def unproven(root, **kwargs):
        workers.append(threading.current_thread())
        # As run_native_git does when no cleanup proof remains.
        native_snapshot_ticket().operation.retain_unproven("native Git").hold()
        raise OSError("native Git cleanup proof unavailable")

    monkeypatch.setattr(change_turn_tracker, "initialize_shadow_root", unproven)
    rig = _console_rig(h, monkeypatch, tmp_path)
    session, elsewhere, send = rig.session, rig.elsewhere, rig.send

    def no_custody():
        return not any(holds_unproven_custody(t) for t in threading.enumerate())

    def holds_claim(owner):
        return _member_of_claim(activity, owner)

    def begin(**kwargs):
        owners.append(capacity.begin_execution(origin=WorkOrigin.MANUAL, **kwargs))
        return owners[-1]

    async def scenario():
        # Control: a live claim is ordinary busy work, even beside a retained
        # cleanup whose owner holds no claim. Its copy still says "retry".
        bystander = capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id=""
        )
        unrelated = bystander.reserve_tool()
        unrelated_retention = unrelated.retain_unproven("Raw CLI")
        live = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="t")
        claim = activity.try_acquire(
            owner=live, footprint=WorktreeFootprint(opaque=True)
        )
        assert claim is not None
        assert await send(session, "while busy") == (False, _BUSY_COPY)
        claim.settle(durable=True)
        live.finish_root()
        unrelated_retention.prove()
        unrelated.finish()
        bystander.finish_root()
        # Control: once that conflict is gone, this rig admits a Send in
        # either workspace.
        await rig.admitted(session, "before the hold")
        await rig.admitted(elsewhere, "elsewhere before the hold")
        assert len(rig.gateway.payloads) == 2

        retention = None
        if holder == "initialization":
            h.registry.set_change_review_enabled("init-workspace", True)
            h.service.admit_turn("init-workspace")  # no turn bound: its own owner
            await asyncio.to_thread(wait_until, lambda: capacity.retained(), 15)
            (retention,) = capacity.retained()
            owner = retention.operation.owner
            assert owner.conversation_id == "" and owner.run_id is None
        else:
            # An agent turn's opaque claim.
            turn = begin(conversation_id="agent-turn")
            turn_claim = activity.try_acquire(
                owner=turn, footprint=WorktreeFootprint(opaque=True)
            )
            assert turn_claim is not None
            owner = turn
            if holder == "child_member":  # a sub-agent joined the turn's claim
                owner = begin(conversation_id="agent-turn", child=True)
                turn_claim.attach_child(owner)
            if holder == "marked_unproven":  # no retained record at all
                owner.mark_cleanup_unproven()
                assert not capacity.retained()
            else:
                retention = owner.reserve_tool().retain_unproven("native Git")
        assert _member_of_claim(activity, owner, opaque=True)

        assert await send(session, "a later send") == (False, _HELD_COPY)
        # The claim is opaque: a chat in another workspace is refused as well.
        assert await send(elsewhere, "elsewhere") == (False, _HELD_COPY)
        if retention is not None:
            # Release (the Held tool slots action) returns the tool slot only;
            # the claim, and the refusal, stay.
            assert capacity.snapshot().tool_workers == 1
            assert release_held_tool_slot(rig.controller, owner.execution_id)
            assert retention.abandoned
            assert capacity.snapshot().tool_workers == 0
            assert holds_claim(owner)
            assert await send(session, "after release") == (False, _HELD_COPY)
            assert await send(elsewhere, "after release") == (False, _HELD_COPY)
            # Not even a proof (no production code proves) retires the claim:
            # the owner finishes, but its drain stays unproven.
            retention.prove()
            for worker in workers:  # the job's ticket returns, finishing its root
                await asyncio.to_thread(worker.join, 15)
                assert not worker.is_alive(), "custody worker did not return"
            assert not capacity.retained()
        if owner in owners:
            owners.remove(owner)
            owner.finish_root()
        assert owner.drain_state() == "unproven"
        assert holds_claim(owner)
        # The claim is still held, so the copy still names the restart.
        assert await send(session, "after the proof") == (False, _HELD_COPY)
        assert len(rig.gateway.payloads) == 2

    try:
        with rig.sandbox():
            asyncio.run(scenario())
    finally:
        for retention in capacity.retained():
            retention.prove()
        wait_until(no_custody, timeout=15)
        for worker in workers:
            worker.join(15)
            assert not worker.is_alive(), "custody worker did not return"
        for owner in reversed(owners):
            owner.finish_root()


def test_a_retained_claim_that_a_send_does_not_need_keeps_the_busy_copy(
    native_initializer, monkeypatch, tmp_path
):
    """The restart copy needs a held claim that conflicts with the Send's own
    footprint (TASK-41950). A retained cleanup's bounded claim on one folder
    beside a live claim on the folder that holds each turn's scratch space:
    the Send conflicts only with the live claim, so it still says "retry"."""
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeScope

    h = native_initializer
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    rig = _console_rig(h, monkeypatch, tmp_path)
    folder = tmp_path / "held-folder"
    folder.mkdir()
    retained_owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="retained"
    )
    bounded = WorktreeFootprint(
        writes=(WorktreeScope(folder, True, capture_directory_chain(folder)),)
    )
    assert activity.try_acquire(owner=retained_owner, footprint=bounded)
    operation = retained_owner.reserve_tool()
    retention = operation.retain_unproven("native Git")
    # A Send's own footprint is its turn's scratch space (light surface).
    scratch = rig.controller._scratch_spaces._temp_parent
    live = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="live")
    claim = activity.try_acquire(
        owner=live,
        footprint=WorktreeFootprint(
            writes=(WorktreeScope(scratch, True, capture_directory_chain(scratch)),)
        ),
    )
    assert claim is not None

    async def scenario():
        assert await rig.send(rig.session, "while busy") == (False, _BUSY_COPY)
        claim.settle(durable=True)
        live.finish_root()
        # With the live claim gone, the bounded held claim does not stand in
        # the way of this Send.
        await rig.admitted(rig.session, "once the live claim is gone")

    try:
        with rig.sandbox():
            asyncio.run(scenario())
    finally:
        retention.prove()
        operation.finish()
        retained_owner.finish_root()
        claim.settle(durable=True)
        if live.is_admitted:
            live.finish_root()


def test_a_claim_whose_owner_is_marked_unproven_holds_roots_until_restart(
    native_initializer,
):
    """A claim whose owner's cleanup is unproven never retires, even with no
    retained record to Release (TASK-41950 review): the refused root waits
    for a restart and is not offered a retry."""
    h = native_initializer
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="t")
    claim = activity.try_acquire(owner=owner, footprint=WorktreeFootprint(opaque=True))
    assert claim is not None
    owner.mark_cleanup_unproven()
    assert not capacity.retained()
    try:
        h.registry.set_change_review_enabled("init-workspace", True)
        h.service.admit_turn("init-workspace")
        wait_until(
            lambda: (
                h.service.status("init-workspace").roots[0].state
                is not RootReadinessState.PREPARING
            )
        )
        root = h.service.status("init-workspace").roots[0]
        assert (root.state, root.reason) == (
            RootReadinessState.RESTART_REQUIRED,
            h.service._RESTART_REASON,
        )
        assert root.reason == _RESTART_COPY
        assert h.service.retry_failed_roots("init-workspace") == 0
    finally:
        owner.finish_root()
        claim.settle(durable=True)


def test_a_claim_held_without_a_retained_cleanup_stays_retryable(native_initializer):
    """Only a retained cleanup's own claim means "restart" (TASK-41635 control).

    A retained cleanup whose owner holds no claim sits beside an opaque claim
    held by a live owner that retained nothing. The refused root reads "busy"
    with a retry, and once that claim is released the retry reaches READY.
    """
    h = native_initializer
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    bystander = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="")
    retained = bystander.reserve_tool()
    retention = retained.retain_unproven("Raw CLI")
    holder = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="t")
    claim = activity.try_acquire(owner=holder, footprint=WorktreeFootprint(opaque=True))
    assert claim is not None
    try:
        h.registry.set_change_review_enabled("init-workspace", True)
        h.service.admit_turn("init-workspace")
        wait_until(
            lambda: h.service.status("init-workspace").roots[0].state
            is not RootReadinessState.PREPARING
        )
        root = h.service.status("init-workspace").roots[0]
        assert (root.state, root.reason) == (
            RootReadinessState.FAILED,
            "Change history preparation is busy; retry from Settings.",
        )

        claim.settle(durable=True)
        holder.finish_root()
        assert h.service.retry_failed_roots("init-workspace") == 1
        wait_until(
            lambda: h.service.status("init-workspace").roots[0].state
            is RootReadinessState.READY
        )
    finally:
        retention.prove()
        retained.finish()
        bystander.finish_root()
        claim.settle(durable=True)
        holder.finish_root()


def test_source_change_after_admission_still_returns_the_slot(
    native_initializer, monkeypatch
):
    """TASK-41957: the worker's own activation refuses a changed source.

    The slot is reserved at admission; the source then changes before the
    worker enters ``ticket.activate()``. Returning must not depend on that
    activation, or the tool slot stays counted until restart.
    """
    from tldw_chatbook.Workspaces.change_review_ownership import _InitializationSource

    h = native_initializer
    original_admit = _InitializationSource.admit
    tickets = []

    def admit_then_replace(source):
        ticket = original_admit(source)
        tickets.append(ticket)
        monkeypatch.setattr(h.app, "workspace_registry_service", object())
        return ticket

    monkeypatch.setattr(_InitializationSource, "admit", admit_then_replace)
    h.registry.set_change_review_enabled("init-workspace", True)
    capture_change_review_admission(h.app, "init-workspace")
    wait_until(lambda: tickets)
    wait_until(lambda: h.service._queue.unfinished_tasks == 0)
    assert (
        h.service.status("init-workspace").roots[0].state is RootReadinessState.FAILED
    )
    assert tickets[0].retired.is_set(), "ticket never retired"
    assert h.runtime.execution_capacity.snapshot().executions == ()
