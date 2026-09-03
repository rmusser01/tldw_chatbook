"""Startup-init hygiene guards (TASK-21111).

Four separate defects in `TldwCli.__init__` / `on_mount`, each pinned here by
the property that would go quietly wrong again:

(a) the phase-3 parallel-init log measured its clock AFTER `future.result()`
    returned, so every task reported 0.000s;
(b) four OS-keyring touches ran during construction (the loudest, and the one
    the review missed, was a real `keyring.get_password` from the generated-
    video store's retention pass);
(c) the persisted ingest-job restore did its DB open, read and reconcile
    writes synchronously on the UI thread inside `on_mount`;
(d) -- covered in `Tests/Character_Chat/test_samira_preflight_query.py`.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from tldw_chatbook.app import TldwCli

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# (a) the parallel-init timing log
# --------------------------------------------------------------------------


class _Recorder:
    """The minimum of `TldwCli` that `_timed_init_task` touches."""

    def __init__(self) -> None:
        self._startup_parallel_tasks: dict[str, float] = {}


def test_timed_init_task_records_the_work_not_the_wait() -> None:
    """The duration must come from around the CALL, not from `result()`.

    The shipped defect stamped `perf_counter()` immediately before
    `future.result()` on a future `as_completed` had already yielded, so the
    measured interval contained none of the work.
    """
    recorder = _Recorder()

    def slow(marker: str) -> str:
        time.sleep(0.05)
        return marker

    result = TldwCli._timed_init_task(recorder, "slow_task", slow, "done")

    assert result == "done"
    assert recorder._startup_parallel_tasks["slow_task"] >= 0.045


def test_timed_init_task_times_a_failing_task_too() -> None:
    """A slow FAILING initializer is exactly the one worth timing."""
    recorder = _Recorder()

    def boom() -> None:
        time.sleep(0.03)
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        TldwCli._timed_init_task(recorder, "boom", boom)

    assert recorder._startup_parallel_tasks["boom"] >= 0.025


def test_parallel_phase_submits_every_initializer_through_the_timer() -> None:
    """No initializer may be submitted raw -- a raw one logs 0.000s forever."""
    source = (REPO_ROOT / "tldw_chatbook/app.py").read_text(encoding="utf-8")
    app_class = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef) and node.name == "TldwCli"
    )
    init = next(
        node
        for node in app_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    submits = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "submit"
    ]
    assert len(submits) == 4, "expected the four phase-3 parallel initializers"
    for call in submits:
        first = call.args[0]
        assert isinstance(first, ast.Attribute) and first.attr == "_timed_init_task", (
            "a parallel initializer is submitted without `_timed_init_task`; its "
            "logged duration would be the wait, not the work"
        )


# --------------------------------------------------------------------------
# (b) no OS keyring at boot
# --------------------------------------------------------------------------


@pytest.fixture()
def keyring_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every keyring entry point the app can reach."""
    import keyring
    import keyring.core

    calls: list[str] = []
    real_get = keyring.core.get_keyring
    real_pw = keyring.core.get_password

    def spy_get():
        calls.append("get_keyring")
        return real_get()

    def spy_pw(service, username):
        calls.append(f"get_password:{service}")
        return real_pw(service, username)

    for module in (keyring, keyring.core):
        monkeypatch.setattr(module, "get_keyring", spy_get)
        monkeypatch.setattr(module, "get_password", spy_pw)
    return calls


def test_app_construction_touches_no_os_keyring(keyring_spy: list[str]) -> None:
    """Constructing the app must not discover or query an OS credential store.

    Four sites did: the generated-video store's retention pass (a real
    `get_password` for the MiniMax key -- 18 ms on macOS, and a Keychain
    round trip that can block or prompt), the server credential store, and
    the skill-trust marker store + key cache. On a locked keychain any of
    them can stall startup.
    """
    from Tests.UI.app_factory import _build_test_app

    _build_test_app()

    assert keyring_spy == [], f"app construction reached the OS keyring: {keyring_spy}"


def test_the_credential_store_still_resolves_when_asked(
    keyring_spy: list[str],
) -> None:
    """Deferred, not deleted: the first read still builds the real store."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    assert keyring_spy == []

    store = app.server_credential_store

    assert store is not None
    assert "get_keyring" in keyring_spy
    # Idempotent: a second read must not re-discover the backend.
    before = len(keyring_spy)
    assert app.server_credential_store is store
    assert len(keyring_spy) == before


def test_the_skills_stack_defers_only_the_trust_service(
    keyring_spy: list[str],
) -> None:
    """The Console takes the scope facade at mount; that must stay keyring-free.

    Deferring the whole stack was not enough on its own -- `ChatScreen`'s
    agent bridge reads `skills_scope_service` during mount, which merely
    relocated the keyring work. Only the trust service itself is deferred
    behind a factory.
    """
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()

    scope = app.skills_scope_service
    local = app.local_skills_service

    assert scope is not None and local is not None
    assert keyring_spy == [], (
        f"building the skills facade hit the keyring: {keyring_spy}"
    )

    assert local.trust_service is not None
    assert "get_keyring" in keyring_spy, (
        "asking for a trust decision must still build the real trust service"
    )
    assert local.trust_service is app.local_skill_trust_service


def test_an_injected_skills_service_is_not_clobbered_by_a_sibling_read() -> None:
    """Filling the stack lazily must never overwrite a test's double."""
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    double = SimpleNamespace(marker="injected")
    app.local_skills_service = double

    scope = app.skills_scope_service

    assert app.local_skills_service is double
    assert scope is not None


# --------------------------------------------------------------------------
# (c) the ingest-job restore leaves the UI thread
# --------------------------------------------------------------------------


class _FakeApp:
    """Just enough of `TldwCli` for the restore methods under test."""

    _ingest_shutdown = False
    # The real methods under test, so `self._restore_ingest_jobs_off_thread`
    # resolves to the production body rather than a stand-in.
    _restore_ingest_jobs_off_thread = TldwCli._restore_ingest_jobs_off_thread
    _apply_ingest_job_restore = TldwCli._apply_ingest_job_restore

    def __init__(self) -> None:
        self.worker_calls: list[dict[str, Any]] = []
        self.marshalled: list[tuple[Any, ...]] = []
        self.library_ingest_jobs = SimpleNamespace(
            merged=None,
            attached=None,
            merge_restored=lambda jobs, next_id: setattr(
                self.library_ingest_jobs, "merged", (jobs, next_id)
            ),
            attach_store=lambda store: setattr(
                self.library_ingest_jobs, "attached", store
            ),
        )

    def run_worker(self, work, **kwargs):
        self.worker_calls.append({"work": work, **kwargs})

    def call_from_thread(self, callback, *args):
        self.marshalled.append((callback, *args))
        return callback(*args)


def test_restore_starts_a_thread_worker_and_does_no_io_inline() -> None:
    app = _FakeApp()

    TldwCli._restore_ingest_jobs(app)

    assert len(app.worker_calls) == 1
    call = app.worker_calls[0]
    assert call["thread"] is True
    assert call["exit_on_error"] is False, (
        "a history-restore failure must never be able to exit the app"
    )
    assert call["work"].__func__ is TldwCli._restore_ingest_jobs_off_thread


def test_restore_is_skipped_once_shutdown_has_started() -> None:
    app = _FakeApp()
    app._ingest_shutdown = True

    TldwCli._restore_ingest_jobs(app)

    assert app.worker_calls == []


def test_apply_seeds_the_registry_and_attaches_the_store() -> None:
    app = _FakeApp()
    store = SimpleNamespace(closed=False, close=lambda: None)
    plan = SimpleNamespace(jobs=["j1"], next_id=7)

    TldwCli._apply_ingest_job_restore(app, store, plan)

    assert app.library_ingest_jobs.merged == (["j1"], 7)
    assert app.library_ingest_jobs.attached is store
    assert app._library_ingest_jobs_store is store


# --------------------------------------------------------------------------
# Portable Tool Pack service stays post-ready and all-or-nothing
# --------------------------------------------------------------------------


def _guarded_tool_pack_registry():
    from tldw_chatbook.Workspaces.registry_service import (
        DeferredWorkspaceToolProfileGuard,
    )

    bootstrap = DeferredWorkspaceToolProfileGuard()
    registry = SimpleNamespace(
        tool_profile_guard=bootstrap,
        attachments=[bootstrap],
    )
    registry.attach_tool_profile_guard = lambda guard: registry.attachments.append(
        guard
    )
    return registry, bootstrap


def test_deferred_startup_does_not_schedule_tool_pack_composition() -> None:
    """Tool Packs stay out of boot until the user opens Tool Profiles."""
    fake = Mock(spec=TldwCli)
    fake.citation_artifact_ownership_coordinator = None
    fake.citation_legacy_migration_service = None

    def close_deferred(work, *, name: str):
        del name
        work.close()

    fake._create_deferred_startup_task = close_deferred

    TldwCli._schedule_deferred_startup_work(fake)

    callbacks = [call.args[1] for call in fake.set_timer.call_args_list]
    assert fake._deferred_wire_tool_pack_service not in callbacks


def test_tool_pack_wiring_schedules_one_post_ready_thread_worker() -> None:
    calls: list[dict[str, Any]] = []
    scheduled_worker = object()
    registry, bootstrap = _guarded_tool_pack_registry()

    def schedule(work, **kwargs):
        calls.append({"work": work, **kwargs})
        return scheduled_worker

    fake = SimpleNamespace(
        _ui_ready=False,
        _tool_pack_wiring_started=False,
        _tool_pack_guard_bootstrap=bootstrap,
        workspace_registry_service=registry,
        tool_pack_service=None,
        tool_pack_service_unavailable_reason="not_ready",
        _compose_tool_pack_service_off_thread=lambda: None,
        run_worker=schedule,
    )

    assert TldwCli._deferred_wire_tool_pack_service(fake) is None
    assert calls == []

    fake._ui_ready = True
    first = TldwCli._deferred_wire_tool_pack_service(fake)
    second = TldwCli._deferred_wire_tool_pack_service(fake)

    assert len(calls) == 1
    assert first is scheduled_worker
    assert second is scheduled_worker
    assert calls[0]["thread"] is True
    assert calls[0]["exit_on_error"] is False
    assert fake.tool_pack_service_unavailable_reason == "starting"
    assert bootstrap.active_guard is None


def test_tool_pack_worker_start_failure_keeps_the_bootstrap_fail_closed() -> None:
    attempts = 0

    def fail_start(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("cancelled before start")

    registry, bootstrap = _guarded_tool_pack_registry()
    fake = SimpleNamespace(
        _ui_ready=True,
        _tool_pack_wiring_started=False,
        _tool_pack_guard_bootstrap=bootstrap,
        workspace_registry_service=registry,
        tool_pack_service=None,
        tool_pack_service_unavailable_reason="not_ready",
        _compose_tool_pack_service_off_thread=lambda: None,
        run_worker=fail_start,
    )

    TldwCli._deferred_wire_tool_pack_service(fake)

    assert fake.tool_pack_service is None
    assert fake.tool_pack_service_unavailable_reason == "composition_unavailable"
    assert fake._tool_pack_wiring_started is False
    assert bootstrap.active_guard is None

    TldwCli._deferred_wire_tool_pack_service(fake)
    assert attempts == 2


def test_tool_pack_composition_failure_attaches_no_partial_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Tool_Packs.service import ToolPackService

    registry, bootstrap = _guarded_tool_pack_registry()
    fake = SimpleNamespace(
        unified_mcp_service=SimpleNamespace(permission_store=object()),
        local_mcp_control_service=object(),
        workspace_registry_service=registry,
        _tool_pack_guard_bootstrap=bootstrap,
        tool_pack_service=None,
        tool_pack_service_unavailable_reason="starting",
        call_from_thread=lambda callback, *args: callback(*args),
    )
    fake._mark_tool_pack_service_unavailable = lambda category: (
        TldwCli._mark_tool_pack_service_unavailable(fake, category)
    )
    monkeypatch.setattr(
        ToolPackService,
        "compose",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("/private secret")),
    )

    TldwCli._compose_tool_pack_service_off_thread(fake)

    assert fake.tool_pack_service is None
    assert fake.tool_pack_service_unavailable_reason == "composition_unavailable"
    assert fake._tool_pack_wiring_started is False
    assert registry.attachments == [bootstrap]
    assert bootstrap.active_guard is None


def test_tool_pack_prerequisite_failure_attaches_no_guard() -> None:
    registry, bootstrap = _guarded_tool_pack_registry()
    fake = SimpleNamespace(
        unified_mcp_service=None,
        local_mcp_control_service=object(),
        workspace_registry_service=registry,
        _tool_pack_guard_bootstrap=bootstrap,
        tool_pack_service=None,
        tool_pack_service_unavailable_reason="starting",
        call_from_thread=lambda callback, *args: callback(*args),
    )
    fake._mark_tool_pack_service_unavailable = lambda category: (
        TldwCli._mark_tool_pack_service_unavailable(fake, category)
    )

    TldwCli._compose_tool_pack_service_off_thread(fake)

    assert fake.tool_pack_service is None
    assert fake.tool_pack_service_unavailable_reason == "prerequisites_unavailable"
    assert registry.attachments == [bootstrap]
    assert bootstrap.active_guard is None


def test_complete_tool_pack_composition_attaches_exactly_once_at_user_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tldw_chatbook.app as app_module
    from tldw_chatbook.Tool_Packs.catalog_snapshot import PermissionInventoryRegistry
    from tldw_chatbook.Tool_Packs.service import ToolPackService

    calls: list[dict[str, object]] = []

    class Guard:
        @contextmanager
        def mutation_scope(self, **_kwargs):
            yield

    guard = Guard()
    composed = SimpleNamespace(
        binding_guard=guard,
        reconcile_receipts=lambda: SimpleNamespace(unavailable_category=None),
    )
    registry, bootstrap = _guarded_tool_pack_registry()
    fake = SimpleNamespace(
        unified_mcp_service=SimpleNamespace(permission_store=object()),
        local_mcp_control_service=object(),
        workspace_registry_service=registry,
        _tool_pack_guard_bootstrap=bootstrap,
        tool_pack_service=None,
        tool_pack_service_unavailable_reason="starting",
        tool_pack_receipt_reconciliation_unavailable_reason="not_run",
        call_from_thread=lambda callback, *args: callback(*args),
    )
    fake._mark_tool_pack_service_unavailable = lambda category: (
        TldwCli._mark_tool_pack_service_unavailable(fake, category)
    )
    fake._attach_tool_pack_service = lambda service, owner, guard_bootstrap: (
        TldwCli._attach_tool_pack_service(fake, service, owner, guard_bootstrap)
    )
    fake._record_tool_pack_receipt_reconciliation = lambda service, category: (
        TldwCli._record_tool_pack_receipt_reconciliation(fake, service, category)
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        PermissionInventoryRegistry, "v1", lambda *_args, **_kwargs: "sealed"
    )
    monkeypatch.setattr(
        ToolPackService,
        "compose",
        lambda **kwargs: calls.append(kwargs) or composed,
    )

    TldwCli._compose_tool_pack_service_off_thread(fake)
    TldwCli._attach_tool_pack_service(fake, composed, registry, bootstrap)

    assert calls[0]["inventory"] == "sealed"
    assert calls[0]["receipt_root"] == tmp_path / "tool_pack_receipts"
    assert fake.tool_pack_service is composed
    assert fake.tool_pack_service_unavailable_reason is None
    assert fake.tool_pack_receipt_reconciliation_unavailable_reason is None
    assert registry.attachments == [bootstrap]
    assert bootstrap.active_guard is guard


def test_tool_pack_attachment_rejects_registry_replacement() -> None:
    registry_a, bootstrap_a = _guarded_tool_pack_registry()
    registry_b, bootstrap_b = _guarded_tool_pack_registry()
    composed = SimpleNamespace(
        binding_guard=SimpleNamespace(mutation_scope=lambda **_: None)
    )
    fake = SimpleNamespace(
        workspace_registry_service=registry_b,
        _tool_pack_guard_bootstrap=bootstrap_b,
        tool_pack_service=None,
        tool_pack_service_unavailable_reason="starting",
    )
    fake._mark_tool_pack_service_unavailable = lambda category: (
        TldwCli._mark_tool_pack_service_unavailable(fake, category)
    )

    attached = TldwCli._attach_tool_pack_service(
        fake, composed, registry_a, bootstrap_a
    )

    assert attached is False
    assert fake.tool_pack_service is None
    assert fake.tool_pack_service_unavailable_reason == "prerequisites_unavailable"
    assert bootstrap_a.active_guard is None and bootstrap_b.active_guard is None


def test_completed_attachment_never_reinvokes_a_mutating_registry_setter() -> None:
    class Guard:
        @contextmanager
        def mutation_scope(self, **_kwargs):
            yield

    registry, bootstrap = _guarded_tool_pack_registry()
    setter_calls: list[object] = []

    def mutate_then_raise(value: object) -> None:
        setter_calls.append(value)
        registry.tool_profile_guard = value
        raise RuntimeError("partial")

    registry.attach_tool_profile_guard = mutate_then_raise
    service = SimpleNamespace(binding_guard=Guard())
    fake = SimpleNamespace(
        workspace_registry_service=registry,
        _tool_pack_guard_bootstrap=bootstrap,
        tool_pack_service=None,
        tool_pack_service_unavailable_reason="starting",
        tool_pack_receipt_reconciliation_unavailable_reason="not_run",
    )
    fake._mark_tool_pack_service_unavailable = lambda category: (
        TldwCli._mark_tool_pack_service_unavailable(fake, category)
    )

    assert TldwCli._attach_tool_pack_service(fake, service, registry, bootstrap) is True
    assert setter_calls == []
    assert registry.tool_profile_guard is bootstrap
    assert bootstrap.active_guard is service.binding_guard


def test_apply_closes_the_store_instead_of_attaching_it_during_shutdown() -> None:
    """A restore that lands after quit began must not leak its connection."""
    closed: list[bool] = []
    app = _FakeApp()
    app._ingest_shutdown = True
    store = SimpleNamespace(close=lambda: closed.append(True))

    TldwCli._apply_ingest_job_restore(app, store, SimpleNamespace(jobs=[], next_id=1))

    assert closed == [True]
    assert app.library_ingest_jobs.attached is None
    assert getattr(app, "_library_ingest_jobs_store", None) is None


def test_worker_swallows_a_store_failure_and_leaves_the_registry_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt store still means "start empty", never a WorkerFailed."""
    import tldw_chatbook.DB.Library_Ingest_Jobs_DB as jobs_db

    def boom(*_a, **_kw):
        raise RuntimeError("corrupt store")

    monkeypatch.setattr(jobs_db, "LibraryIngestJobsDB", boom)
    app = _FakeApp()

    TldwCli._restore_ingest_jobs_off_thread(app)

    assert app.marshalled == []
    assert app.library_ingest_jobs.attached is None


def test_alternate_startup_metrics_failures_are_type_only() -> None:
    source = (REPO_ROOT / "tldw_chatbook/app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main_blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    ]
    metrics_tries = [
        node
        for main_block in main_blocks
        for node in main_block.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id in {"init_metrics_server", "init_otel_metrics"}
            for call in ast.walk(node)
        )
    ]
    assert len(metrics_tries) == 2
    initializer_calls = [
        call.func.id
        for node in metrics_tries
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in {"init_metrics_server", "init_otel_metrics"}
    ]
    assert initializer_calls == ["init_metrics_server", "init_otel_metrics"]

    class FakeLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []
            self.warnings: list[str] = []

        def info(self, message: str, *args: object) -> None:
            self.infos.append(message.format(*args))

        def warning(self, message: str, *args: object) -> None:
            self.warnings.append(message.format(*args))

    exception_sentinel = "METRICS-EXCEPTION-SENTINEL-7d31"

    def fail_metrics(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(exception_sentinel)

    executable = ast.fix_missing_locations(
        ast.Module(body=metrics_tries, type_ignores=[])
    )
    failure_logger = FakeLogger()
    failure_namespace = {
        "init_metrics_server": fail_metrics,
        "init_otel_metrics": fail_metrics,
        "loguru_logger": failure_logger,
    }
    code = compile(executable, "<startup-metrics>", "exec")
    exec(code, failure_namespace)

    assert failure_logger.infos == []
    assert failure_logger.warnings == [
        "Prometheus metrics initialization failed (exception_type=RuntimeError).",
        "OpenTelemetry metrics initialization failed (exception_type=RuntimeError).",
    ]
    assert exception_sentinel not in "\n".join(failure_logger.warnings)

    success_logger = FakeLogger()
    success_namespace = {
        "init_metrics_server": lambda **_kwargs: True,
        "init_otel_metrics": lambda: True,
        "loguru_logger": success_logger,
    }
    exec(code, success_namespace)

    assert success_logger.infos == []
    assert success_logger.warnings == []
