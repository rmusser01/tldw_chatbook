"""App-owned Change Review consent and root-readiness lifecycle tests."""

from __future__ import annotations

import threading
import time

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
import tldw_chatbook.Workspaces.change_review_consent as consent_module


def _registry(tmp_path) -> LocalWorkspaceRegistryService:
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "workspaces.sqlite", client_id="consent-test")
    )
    registry.create_workspace(workspace_id="ws-review", name="Review")
    return registry


def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not settle before timeout")
        time.sleep(0.005)


def _start_thread(target) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def checked_target() -> None:
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - surface thread failures
            errors.append(exc)

    thread = threading.Thread(target=checked_target, daemon=True)
    thread.start()
    return thread, errors


def _join_thread(
    thread: threading.Thread,
    errors: list[BaseException],
    *,
    timeout: float = 2.0,
) -> None:
    thread.join(timeout)
    assert not thread.is_alive(), "test worker did not finish"
    if errors:
        raise errors[0]


def test_disabled_admission_never_schedules_initializer(tmp_path) -> None:
    """Missing consent is a silent opt-out, not background filesystem work."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    initialize_calls: list[str] = []
    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_calls.append,
        worker_count=1,
    )
    try:
        admission = service.admit_turn("ws-review")

        assert admission.ready_roots == ()
        assert admission.skipped_roots == ()
        assert initialize_calls == []
    finally:
        service.shutdown(timeout=0.2)


@pytest.mark.parametrize(
    "capability_state",
    [
        consent_module.ChangeReviewState.DISABLED,
        consent_module.ChangeReviewState.UNAVAILABLE,
    ],
)
def test_unavailable_global_capability_schedules_no_work(
    tmp_path, capability_state
) -> None:
    """Disabled or unreadable global capability fails tracking off."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    registry.set_change_review_enabled("ws-review", True)
    calls: list[str] = []
    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=calls.append,
        capability_reader=lambda: consent_module.ChangeReviewCapability(
            capability_state
        ),
        worker_count=1,
    )
    try:
        assert service.admit_turn("ws-review") == consent_module.ChangeReviewAdmission()
        assert calls == []
    finally:
        service.shutdown(timeout=0.2)


def test_enabled_admission_prepares_once_then_returns_ready_root(tmp_path) -> None:
    """A cold enabled root is skipped until one bounded initializer settles."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    binding = registry.add_folder_binding("ws-review", root)
    registry.set_change_review_enabled("ws-review", True)
    entered = threading.Event()
    release = threading.Event()
    initialize_calls: list[str] = []

    def initialize_root(path: str) -> None:
        initialize_calls.append(path)
        entered.set()
        assert release.wait(2.0), "test never released initializer"

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_root,
        worker_count=1,
    )
    try:
        first = service.admit_turn("ws-review")
        assert entered.wait(1.0), "initializer never started"
        second = service.admit_turn("ws-review")

        assert first.ready_roots == second.ready_roots == ()
        assert first.skipped_roots == second.skipped_roots
        assert first.skipped_roots[0].alias == binding.binding_id
        assert "Preparing change history" in first.skipped_roots[0].reason
        assert initialize_calls == [str(root.resolve())]

        release.set()
        _wait_until(
            lambda: service.admit_turn("ws-review").ready_roots
            == (str(root.resolve()),)
        )
        assert initialize_calls == [str(root.resolve())]
    finally:
        release.set()
        service.shutdown(timeout=0.2)


def test_enabling_consent_starts_background_preparation(tmp_path) -> None:
    """Opting in prepares existing bindings without waiting for a chat turn."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    binding = registry.add_folder_binding("ws-review", root)
    expected = registry.read_change_review_consent("ws-review")
    entered = threading.Event()
    release = threading.Event()

    def initialize_root(_root: str) -> None:
        entered.set()
        assert release.wait(2.0), "test never released initializer"

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_root,
        worker_count=1,
    )
    try:
        service.toggle("ws-review", expected=expected, enabled=True)

        assert entered.wait(1.0), "enabling did not start preparation"
        assert service.status("ws-review").roots == (
            consent_module.RootReadiness(
                alias=binding.binding_id,
                state=consent_module.RootReadinessState.PREPARING,
                reason="Preparing change history; this turn continues without it.",
            ),
        )
    finally:
        release.set()
        service.shutdown(timeout=0.2)


def test_admission_and_disable_linearize_on_one_lock(tmp_path) -> None:
    """A toggle waits for an admission already holding the consent lock."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    registry.set_change_review_enabled("ws-review", True)
    expected = registry.read_change_review_consent("ws-review")
    entered = threading.Event()
    release = threading.Event()
    capability_calls = 0

    def capability_reader():
        nonlocal capability_calls
        capability_calls += 1
        if capability_calls == 1:
            entered.set()
            assert release.wait(2.0), "test never released admission"
        return consent_module.ChangeReviewCapability(
            consent_module.ChangeReviewState.ENABLED
        )

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=lambda _root: None,
        capability_reader=capability_reader,
        worker_count=1,
    )
    admissions: list[consent_module.ChangeReviewAdmission] = []
    toggle_done = threading.Event()
    try:
        admission_thread, admission_errors = _start_thread(
            lambda: admissions.append(service.admit_turn("ws-review"))
        )
        assert entered.wait(1.0), "admission did not acquire the service lock"

        def disable() -> None:
            service.toggle("ws-review", expected=expected, enabled=False)
            toggle_done.set()

        toggle_thread, toggle_errors = _start_thread(disable)
        assert not toggle_done.wait(0.05), "disable bypassed admission lock"
        release.set()
        _join_thread(admission_thread, admission_errors)
        _join_thread(toggle_thread, toggle_errors)

        assert admissions[0].skipped_roots
        assert service.admit_turn("ws-review") == consent_module.ChangeReviewAdmission()
    finally:
        release.set()
        service.shutdown(timeout=0.2)


def test_disable_then_reenable_rejects_old_initializer_aba(tmp_path) -> None:
    """A completion captured under an older enabled revision cannot publish."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    registry.set_change_review_enabled("ws-review", True)
    first_revision = registry.read_change_review_consent("ws-review")
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def initialize_root(_root: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(2.0), "test never released old initializer"

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_root,
        worker_count=1,
    )
    try:
        assert service.admit_turn("ws-review").skipped_roots
        assert entered.wait(1.0), "initializer never started"
        disabled = service.toggle(
            "ws-review", expected=first_revision, enabled=False
        )
        service.toggle("ws-review", expected=disabled, enabled=True)

        release.set()
        _wait_until(lambda: calls == 1)
        next_admission = service.admit_turn("ws-review")
        assert next_admission.ready_roots == ()
        assert next_admission.skipped_roots
        _wait_until(
            lambda: service.admit_turn("ws-review").ready_roots
            == (str(root.resolve()),)
        )
        assert calls == 2
    finally:
        release.set()
        service.shutdown(timeout=0.2)


def test_external_revision_change_rejects_old_initializer(tmp_path) -> None:
    """A second registry process can invalidate in-flight initialization."""
    db_path = tmp_path / "workspaces.sqlite"
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(db_path, client_id="consent-test-a")
    )
    registry.create_workspace(workspace_id="ws-review", name="Review")
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    registry.set_change_review_enabled("ws-review", True)
    other_registry = LocalWorkspaceRegistryService(
        WorkspaceDB(db_path, client_id="consent-test-b")
    )
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def initialize_root(_root: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            assert release.wait(2.0), "test never released old initializer"

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_root,
        worker_count=1,
    )
    try:
        service.admit_turn("ws-review")
        assert entered.wait(1.0), "initializer never started"
        other_registry.set_change_review_enabled("ws-review", True)
        release.set()

        _wait_until(lambda: calls == 1)
        assert service.admit_turn("ws-review").ready_roots == ()
        _wait_until(
            lambda: service.admit_turn("ws-review").ready_roots
            == (str(root.resolve()),)
        )
        assert calls == 2
    finally:
        release.set()
        service.shutdown(timeout=0.2)


def test_failed_toggle_cas_preserves_readiness(tmp_path, monkeypatch) -> None:
    """A rejected toggle has no runtime side effects."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    registry.set_change_review_enabled("ws-review", True)
    expected = registry.read_change_review_consent("ws-review")
    entered = threading.Event()
    release = threading.Event()

    def initialize_root(_root: str) -> None:
        entered.set()
        assert release.wait(2.0), "test never released initializer"

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_root,
        worker_count=1,
    )
    try:
        service.admit_turn("ws-review")
        assert entered.wait(1.0), "initializer never started"
        before = service.status("ws-review")

        def fail_cas(*_args, **_kwargs):
            raise consent_module.ChangeReviewStateConflict("stale")

        monkeypatch.setattr(
            registry, "compare_and_set_change_review_consent", fail_cas
        )
        with pytest.raises(consent_module.ChangeReviewStateConflict):
            service.toggle("ws-review", expected=expected, enabled=False)

        assert service.status("ws-review") == before
    finally:
        release.set()
        service.shutdown(timeout=0.2)


def test_full_initializer_queue_fails_fast_and_retry_is_bounded(tmp_path) -> None:
    """Queue pressure never blocks admission or creates duplicate work."""
    registry = _registry(tmp_path)
    first_root = tmp_path / "root-1"
    first_root.mkdir()
    registry.add_folder_binding("ws-review", first_root)
    registry.set_change_review_enabled("ws-review", True)
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def initialize_root(root: str) -> None:
        calls.append(root)
        if root == str(first_root.resolve()):
            entered.set()
            assert release.wait(2.0), "test never released first initializer"

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_root,
        worker_count=1,
        queue_capacity=1,
    )
    try:
        service.admit_turn("ws-review")
        assert entered.wait(1.0), "first initializer never started"
        second_root = tmp_path / "root-2"
        third_root = tmp_path / "root-3"
        second_root.mkdir()
        third_root.mkdir()
        registry.add_folder_binding("ws-review", second_root)
        registry.add_folder_binding("ws-review", third_root)

        started = time.monotonic()
        admission = service.admit_turn("ws-review")
        assert time.monotonic() - started < 0.2
        assert len(admission.skipped_roots) == 3
        failed = [
            root
            for root in service.status("ws-review").roots
            if root.state is consent_module.RootReadinessState.FAILED
        ]
        assert len(failed) == 1
        assert service.retry_failed_roots("ws-review") == 0

        release.set()
        _wait_until(
            lambda: len(
                [
                    root
                    for root in service.status("ws-review").roots
                    if root.state is consent_module.RootReadinessState.READY
                ]
            )
            == 2
        )
        assert service.retry_failed_roots("ws-review") == 1
        _wait_until(
            lambda: len(service.admit_turn("ws-review").ready_roots) == 3
        )
        assert len(calls) == 3
    finally:
        release.set()
        service.shutdown(timeout=0.2)


def test_shutdown_is_bounded_and_late_completion_reads_no_registry(tmp_path) -> None:
    """A blocked initializer becomes generation-inert after shutdown."""
    registry = _registry(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    registry.add_folder_binding("ws-review", root)
    registry.set_change_review_enabled("ws-review", True)
    entered = threading.Event()
    release = threading.Event()
    exited = threading.Event()
    reads = 0
    original_read = registry.read_change_review_consent

    def counted_read(workspace_id: str):
        nonlocal reads
        reads += 1
        return original_read(workspace_id)

    registry.read_change_review_consent = counted_read  # type: ignore[method-assign]

    def initialize_root(_root: str) -> None:
        entered.set()
        assert release.wait(2.0), "test never released initializer"
        exited.set()

    service = consent_module.ChangeReviewConsentService(
        registry,
        initialize_root=initialize_root,
        worker_count=1,
    )
    service.admit_turn("ws-review")
    assert entered.wait(1.0), "initializer never started"
    reads_before_shutdown = reads

    started = time.monotonic()
    service.shutdown(timeout=0.02)
    assert time.monotonic() - started < 0.2
    release.set()
    assert exited.wait(1.0), "initializer did not exit"
    service.shutdown(timeout=0.2)

    assert reads == reads_before_shutdown
