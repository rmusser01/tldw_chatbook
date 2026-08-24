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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    assert keyring_spy == [], f"building the skills facade hit the keyring: {keyring_spy}"

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
