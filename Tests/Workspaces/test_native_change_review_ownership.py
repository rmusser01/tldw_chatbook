"""Real Console snapshot jobs retain original physical ownership through publication."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.execution_capacity import WorkOrigin, current_execution_owner
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def native_review(tmp_path):
    chat = CharactersRAGDB(tmp_path / "chat.db", "snapshot-native")
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=chat, app_config={}))
    store = runtime.ensure_chat_store()
    session = store.create_session(ephemeral=True)
    bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store,
        provider_gateway_factory=lambda: object(),
        skills_service=None,
    )
    coordinator = runtime.change_review_coordinator
    tracker = coordinator._tracker
    tracker.service = ShadowRepoService(tmp_path / "shadow")
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("before\n")
    host = SimpleNamespace(**locals())
    try:
        yield host
    finally:
        assert coordinator.shutdown(5)
        asyncio.run(runtime.dispose())
        chat.close_connection()


def register(
    host, *, origin=WorkOrigin.MANUAL, owner=None, claim=None, on_unadmitted=None
):
    native = getattr(host.coordinator, "_native", None)
    kwargs = {}
    if native is not None:
        kwargs["native_source"] = native.capture(
            host.bridge,
            session_id=host.session.id,
            conversation_id=owner.conversation_id if owner else "conversation",
            origin=origin,
            roots=(host.root,),
            owner=owner,
            claim=claim,
        )
    return host.coordinator.register([host.root], on_unadmitted=on_unadmitted, **kwargs)


def test_actual_discovery_remains_owned_after_timeout_and_shutdown(
    native_review, monkeypatch
):
    h = native_review
    entered, release = threading.Event(), threading.Event()
    observed = []
    discover = h.tracker.discover_baseline

    def held(handle):
        observed.append(current_execution_owner())
        entered.set()
        assert release.wait(5)
        return discover(handle)

    monkeypatch.setattr(h.tracker, "discover_baseline", held)
    reservation = register(h)
    try:
        assert entered.wait(3)
        assert observed[0] is not None, "actual native discover has no owner"
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        assert not h.coordinator.await_baseline(reservation, 0)
        assert not h.coordinator.shutdown(0)
        contender = h.runtime.execution_capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id="contender"
        )
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None, "timeout/shutdown released actual snapshot activity"
        contender.finish_root()
    finally:
        release.set()
        assert h.coordinator.shutdown(5)
    assert h.runtime.execution_capacity.snapshot().executions == ()


def test_capacity_release_notifies_once_before_owner_drains():
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

    capacity = RuntimeCapacity(max_tool_workers=1, reserved_manual_tool_workers=0)
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="busy")
    operation = owner.reserve_tool()
    notices = []
    subscribe = getattr(capacity, "subscribe_tool_availability", None)
    assert callable(subscribe), "tool slot release has no availability seam"
    unsubscribe = subscribe(lambda: notices.append(capacity.snapshot().tool_workers))
    operation.finish()
    operation.finish()
    assert notices == [0]
    assert owner.is_admitted
    capacity.set_tool_limits(2, 0)
    assert notices == [0, 0]
    unsubscribe()
    owner.finish_root()


@pytest.mark.parametrize("filter_key", ["clean", "process", "smudge"])
def test_native_capture_refuses_configured_filters_before_helper_entry(
    native_review, monkeypatch, tmp_path, filter_key
):
    from tldw_chatbook.Workspaces.change_tracking import ShadowRepo

    h = native_review
    git_home = tmp_path / "git-home"
    git_home.mkdir()
    sentinel = tmp_path / "helper-entered"
    helper = tmp_path / "controlled-filter"
    helper.write_text(f"#!/bin/sh\nprintf entered > '{sentinel}'\ncat\n")
    helper.chmod(0o700)
    included = git_home / "filters"
    included.write_text(f'[filter "test"]\n{filter_key} = {helper}\n')
    (git_home / ".gitconfig").write_text(f"[include]\npath = {included}\n")
    (h.root / ".gitattributes").write_text("*.txt filter=test\n")
    old_env = ShadowRepo._env
    monkeypatch.setattr(
        ShadowRepo,
        "_env",
        lambda repo: {
            **old_env(repo),
            "HOME": str(git_home),
            "XDG_CONFIG_HOME": str(git_home),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    )
    # The RED probe intercepts helper-capable entry; never execute an unknown helper.
    import subprocess

    run = subprocess.run
    attempted = []

    def guarded_run(cmd, **kwargs):
        if "add" in cmd or "update-index" in cmd:
            attempted.append(tuple(cmd))
            raise AssertionError("unguarded helper-capable Git entry")
        return run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    reservation = register(h)
    assert reservation.await_baseline(4)
    h.coordinator.cancel(reservation)
    assert h.coordinator.wait_idle(3)
    assert not attempted, "native Git entered before filter qualification"
    assert not sentinel.exists()
    assert "configured filters" in reservation._handle.errors[str(h.root)]


def wait_for(predicate):
    import time

    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "native transition did not complete"
        time.sleep(0.01)


@pytest.mark.parametrize("failures", [0, 1, 2])
def test_original_end_owner_survives_actual_publication(
    native_review, monkeypatch, failures
):
    h = native_review
    reservation = register(h)
    assert reservation.await_baseline(4)
    entered, release = threading.Event(), threading.Event()
    original_publish = h.coordinator._publish
    observed = []

    def held(publication):
        observed.append(current_execution_owner())
        entered.set()
        assert release.wait(5)
        if len(observed) <= failures:
            raise RuntimeError("controlled publication failure")
        original_publish(publication)

    # Empty E still invokes the actual SQLite publisher; no fake run/diff row.
    monkeypatch.setattr(h.coordinator, "_publish", held)
    h.coordinator.finalize(reservation, run_id="", kind="turn")
    try:
        assert entered.wait(4)
        assert observed[0] is not None
        h.runtime.execution_capacity.close()
        assert not h.coordinator.shutdown(0)
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
        assert reservation.completion.value == "pending"
    finally:
        release.set()
        assert h.coordinator.shutdown(5)
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert (
        reservation.completion.value
        == ("published", "disclosed_failure", "unresolved")[failures]
    )
    assert len(observed) == (1, 2, 2)[failures]
    assert all(owner is observed[0] for owner in observed)


def test_capacity_wait_retries_on_real_slot_release_without_self_retry(
    native_review, monkeypatch
):
    h = native_review
    capacity = h.runtime.execution_capacity
    capacity._read_settings = False
    capacity.set_tool_limits(2, 1)
    busy = capacity.begin_execution(origin=WorkOrigin.AUTOMATIC, conversation_id="busy")
    occupied = busy.reserve_tool()
    entered = threading.Event()
    discover = h.tracker.discover_baseline
    attempts = []
    begin = capacity.begin_execution

    def counted(**kwargs):
        attempts.append(kwargs)
        return begin(**kwargs)

    monkeypatch.setattr(capacity, "begin_execution", counted)
    monkeypatch.setattr(
        h.tracker,
        "discover_baseline",
        lambda handle: (entered.set(), discover(handle))[1],
    )
    reservation = register(h, origin=WorkOrigin.AUTOMATIC)
    try:
        wait_for(lambda: bool(attempts))
        assert not entered.wait(0.1)
        assert len(attempts) == 1, "capacity refusal retries itself"
        assert capacity.snapshot().tool_workers == 1
        manual = begin(origin=WorkOrigin.MANUAL, conversation_id="manual")
        manual_operation = manual.reserve_tool()
        manual_operation.finish()
        manual.finish_root()
        wait_for(lambda: len(attempts) == 2)
        assert not entered.wait(0.1)
        occupied.finish()
        assert entered.wait(3)
        assert busy.is_admitted, "retry incorrectly depended on whole owner drain"
        assert reservation.await_baseline(4)
    finally:
        occupied.finish()
        busy.finish_root()
        h.coordinator.cancel(reservation)


def test_discover_exception_preserves_original_roots_and_error(
    native_review, monkeypatch
):
    h = native_review

    def fail(_handle):
        raise RuntimeError("controlled discover failure")

    monkeypatch.setattr(h.tracker, "discover_baseline", fail)
    reservation = register(h)
    assert reservation.await_baseline(4)
    assert reservation._handle.roots == [h.root]
    assert "controlled discover failure" in reservation._handle.errors[str(h.root)]
    h.coordinator.cancel(reservation)


def test_original_opaque_claim_is_borrowed_and_bounded_claim_refuses(native_review):
    h = native_review
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="tree")
    claim = activity.try_acquire(owner=owner, footprint=WorktreeFootprint(opaque=True))
    try:
        reservation = register(h, owner=owner, claim=claim)
        assert reservation.await_baseline(4)
        assert not reservation._handle.errors
        h.coordinator.cancel(reservation)
        assert h.coordinator.wait_idle(3)
        assert owner.is_admitted
        with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as contender:
            assert contender is None
    finally:
        claim.settle(durable=True)
        owner.finish_root()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="narrow")
    claim = activity.try_acquire(owner=owner, footprint=WorktreeFootprint())
    try:
        reservation = register(h, owner=owner, claim=claim)
        assert reservation.await_baseline(3)
        assert reservation._handle.errors
        h.coordinator.cancel(reservation)
    finally:
        claim.settle(durable=True)
        owner.finish_root()


def test_terminal_disposition_waits_for_original_publisher_scope_return(
    native_review, monkeypatch
):
    from contextlib import contextmanager

    from tldw_chatbook.Workspaces.change_review_ownership import _SnapshotSource

    h = native_review
    reservation = register(h)
    assert reservation.await_baseline(4)
    entered, release = threading.Event(), threading.Event()
    wait_for(lambda: h.runtime.execution_capacity.snapshot().tool_workers == 0)
    activate = _SnapshotSource.activate

    @contextmanager
    def held_exit(source, owner):
        with activate(source, owner):
            yield
            if threading.current_thread() is h.coordinator._publisher:
                entered.set()
                assert release.wait(5)

    monkeypatch.setattr(_SnapshotSource, "activate", held_exit)
    h.coordinator.finalize(reservation, run_id="", kind="turn")
    try:
        assert entered.wait(4)
        assert reservation.completion.value == "pending", (
            "published label preceded original scope return"
        )
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        release.set()
        wait_for(lambda: h.runtime.execution_capacity.snapshot().tool_workers == 0)
    assert reservation.completion.value == "published"


def test_discovered_root_escape_never_reaches_baseline(
    native_review, monkeypatch, tmp_path
):
    from tldw_chatbook.Workspaces.change_turn_tracker import BaselineRootPreparation

    h = native_review
    escaped = tmp_path / "outside"
    escaped.mkdir()
    baseline = h.tracker.populate_prepared_baseline
    seen = []
    monkeypatch.setattr(
        h.tracker,
        "discover_baseline",
        lambda handle: (BaselineRootPreparation(root=escaped),),
    )

    def observe(handle, preparations):
        seen.extend(item.root for item in preparations if not item.tracking_error)
        return baseline(handle, preparations)

    monkeypatch.setattr(h.tracker, "populate_prepared_baseline", observe)
    reservation = register(h)
    assert reservation.await_baseline(4)
    h.coordinator.cancel(reservation)
    assert not seen, "outside discovery root reached snapshot callback"
    assert reservation._handle.roots == [h.root]
    assert reservation._handle.errors[str(h.root)]


def test_native_bridge_missing_source_does_not_fall_back_to_legacy(
    native_review, monkeypatch
):
    from Tests.Chat.test_change_turn_tracking import _run, _SideEffectGateway
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    h = native_review
    h.bridge._gateway = _SideEffectGateway([["done."]])
    h.store.append_message(h.session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = h.store.append_message(
        h.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    entered = []
    monkeypatch.setattr(
        h.tracker, "begin_turn", lambda *a, **k: entered.append("legacy")
    )
    monkeypatch.setattr(h.bridge, "_change_review_native", None)
    monkeypatch.setattr(h.bridge, "_change_finalization_coordinator", None)
    _, outcome = _run(h.bridge, h.session, assistant.id, h.root)
    assert outcome.final_text.strip() == "done."
    assert not entered, "native source mismatch silently used standalone legacy capture"
    assert any(
        "native snapshot source unavailable" in m.content
        for m in h.store.messages_for_session(h.session.id)
    )


@pytest.mark.parametrize("phase", ["baseline", "end", "survivor_end"])
def test_actual_baseline_and_end_callbacks_retain_finite_claim(
    native_review, monkeypatch, phase
):
    h = native_review
    entered, release = threading.Event(), threading.Event()
    method = "populate_prepared_baseline" if phase == "baseline" else "finish_turn"
    original = getattr(h.tracker, method)
    calls = []

    def held(*args, **kwargs):
        calls.append(current_execution_owner())
        if len(calls) == (2 if phase == "survivor_end" else 1):
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(h.tracker, method, held)
    source = h.coordinator._native.capture(
        h.bridge,
        session_id=h.session.id,
        conversation_id="capture",
        origin=WorkOrigin.MANUAL,
        roots=(h.root,),
    )
    reservation = h.coordinator.register(
        [h.root], survivor_key="original", native_source=source
    )
    try:
        if phase != "baseline":
            assert reservation.await_baseline(4)
            h.coordinator.finalize(
                reservation,
                run_id="",
                kind="turn",
                has_live_survivors=phase == "survivor_end",
            )
            if phase == "survivor_end":
                wait_for(
                    lambda: h.coordinator._states[reservation.id].phase == "survivor"
                )
                wait_for(
                    lambda: h.runtime.execution_capacity.snapshot().tool_workers == 0
                )
                assert reservation.completion.value == "pending", (
                    "open survivor window is not terminal"
                )
                h.coordinator.settle_survivors("original")
        assert entered.wait(4)
        assert calls[-1] is not None
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        h.coordinator.cancel(reservation)
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as contender:
            assert contender is None
    finally:
        release.set()
        assert h.coordinator.shutdown(5)
    assert h.runtime.execution_capacity.snapshot().executions == ()


def test_worker_scope_entry_failure_delivers_original_error_and_retires(
    native_review, monkeypatch
):
    from contextlib import contextmanager

    from tldw_chatbook.Workspaces.change_review_ownership import _SnapshotSource

    h = native_review
    activate = _SnapshotSource.activate
    invoked = []

    @contextmanager
    def fail_worker(source, owner):
        if threading.current_thread().name.startswith("change-review-fs"):
            raise RuntimeError("controlled native source entry failure")
        with activate(source, owner):
            yield

    monkeypatch.setattr(_SnapshotSource, "activate", fail_worker)
    monkeypatch.setattr(h.tracker, "discover_baseline", lambda _h: invoked.append(True))
    reservation = register(h)
    assert reservation.await_baseline(4)
    wait_for(lambda: reservation.baseline_completion.value != "pending")
    assert reservation._handle.errors[str(h.root)]
    assert not invoked
    assert h.runtime.execution_capacity.snapshot().executions == ()
    h.coordinator.cancel(reservation)


def test_never_admitted_native_source_has_no_worker_or_direct_publication(
    native_review, monkeypatch
):
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewFinalizeResult,
    )

    h = native_review
    entries = []
    monkeypatch.setattr(
        h.tracker, "discover_baseline", lambda _h: entries.append("worker")
    )
    monkeypatch.setattr(
        h.coordinator, "_publish", lambda _p: entries.append("publisher")
    )
    reservation = h.coordinator.register([h.root])
    assert (
        reservation.admission_error
        == "native change-review source or roots unavailable"
    )
    assert reservation.completion.value == "unadmitted"
    assert reservation.baseline_completion.value == "unadmitted"
    assert (
        h.coordinator.finalize(reservation, run_id="run", kind="turn")
        is ChangeReviewFinalizeResult.OVERLOAD_VISIBLE
    )
    assert h.coordinator.wait_idle(1)
    assert not entries
    assert h.runtime.execution_capacity.snapshot().executions == ()


@pytest.mark.parametrize("reason", ["overload", "degraded"])
def test_native_overload_and_degraded_refuse_without_direct_publication(
    native_review, monkeypatch, reason
):
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewFinalizeResult,
    )

    h = native_review
    h.coordinator._capacity = 1
    entered, release = threading.Event(), threading.Event()
    entries, publications = [], []

    def held(handle):
        entries.append(handle)
        entered.set()
        assert release.wait(5)
        return ()

    monkeypatch.setattr(h.tracker, "discover_baseline", held)
    monkeypatch.setattr(h.coordinator, "_publish", publications.append)
    original = register(h)
    try:
        assert entered.wait(3)
        if reason == "degraded":
            assert not h.coordinator.await_baseline(original, 0)
        refused = register(h)
        assert ("resynchronizing" if reason == "degraded" else "at capacity") in (
            refused.admission_error
        )
        assert refused.completion.value == "unadmitted"
        assert refused.baseline_completion.value == "unadmitted"
        assert not refused._publication_reserved
        assert (
            h.coordinator.finalize(refused, run_id="unadmitted", kind="turn")
            is ChangeReviewFinalizeResult.OVERLOAD_VISIBLE
        )
        assert entries == [original._handle]
        assert not publications
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        h.coordinator.cancel(original)
        release.set()
    assert h.coordinator.shutdown(5)
    assert entries == [original._handle]
    assert not publications
    assert h.runtime.execution_capacity.snapshot().executions == ()


def test_root_identity_change_during_capacity_wait_refuses_without_read(
    native_review, monkeypatch
):
    h = native_review
    capacity = h.runtime.execution_capacity
    capacity._read_settings = False
    capacity.set_tool_limits(1, 0)
    busy = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="busy")
    operation = busy.reserve_tool()
    entries = []
    monkeypatch.setattr(h.tracker, "discover_baseline", lambda _h: entries.append(True))
    reservation = register(h)
    try:
        wait_for(
            lambda: h.coordinator._states[reservation.id].pending_reason == "capacity"
        )
        h.root.rename(h.root.with_name("original-root"))
        h.root.mkdir()
        operation.finish()
        assert reservation.await_baseline(4)
        assert not entries
        assert reservation._handle.errors[str(h.root)]
    finally:
        operation.finish()
        busy.finish_root()
        h.coordinator.cancel(reservation)


@pytest.mark.parametrize("source_loss", ["bridge", "capacity"])
def test_late_original_source_loss_never_calls_ticketless_publisher(
    native_review, monkeypatch, source_loss
):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

    h = native_review
    reservation = register(h)
    assert reservation.await_baseline(4)
    wait_for(lambda: reservation.baseline_completion.value != "pending")
    published = []
    entries = []
    monkeypatch.setattr(
        h.tracker, "finish_turn", lambda *args, **kwargs: entries.append(True) or []
    )
    monkeypatch.setattr(h.coordinator, "_publish", published.append)
    with monkeypatch.context() as changed:
        changed.setattr(
            h.runtime,
            "_agent_bridge" if source_loss == "bridge" else "_execution_capacity",
            object() if source_loss == "bridge" else RuntimeCapacity(),
        )
        h.coordinator.finalize(reservation, run_id="", kind="turn")
        wait_for(lambda: reservation.completion.value != "pending")
    assert reservation.completion.value == "unadmitted"
    assert not published
    assert not entries
    assert reservation._handle.errors[str(h.root)]
    assert h.runtime.execution_capacity.snapshot().executions == ()


def test_borrowed_tree_keeps_its_unsettled_claim_after_both_publications_fail(
    native_review, monkeypatch
):
    h = native_review
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    owner = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="original-tree"
    )
    claim = activity.try_acquire(owner=owner, footprint=WorktreeFootprint(opaque=True))
    reservation = register(h, owner=owner, claim=claim)
    try:
        assert reservation.await_baseline(4)

        def fail(_publication):
            raise RuntimeError("controlled durable failure")

        monkeypatch.setattr(h.coordinator, "_publish", fail)
        h.coordinator.finalize(reservation, run_id="", kind="turn")
        wait_for(lambda: reservation.completion.value != "pending")
        assert reservation.completion.value == "unresolved"
        owner.finish_root()
        assert not owner.is_admitted
        with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as contender:
            assert contender is None, (
                "snapshot retirement incorrectly settled parent custody"
            )
    finally:
        owner.finish_root()
        claim.settle(durable=True)


def test_pending_activity_wakes_on_actual_release_and_covers_external_reads(
    native_review, monkeypatch, tmp_path
):
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeScope

    h = native_review
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    blocker = capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="first"
    )
    claim = activity.try_acquire(
        owner=blocker, footprint=WorktreeFootprint(opaque=True)
    )
    entered, release = threading.Event(), threading.Event()
    original = h.tracker.discover_baseline

    def held(handle):
        entered.set()
        assert release.wait(5)
        return original(handle)

    monkeypatch.setattr(h.tracker, "discover_baseline", held)
    reservation = register(h)
    try:
        wait_for(
            lambda: h.coordinator._states[reservation.id].pending_reason == "activity"
        )
        assert not entered.is_set()
        blocker.finish_root()
        assert not entered.is_set()
        claim.settle(durable=True)
        assert entered.wait(3)
        external = tmp_path / "external-config"
        external.mkdir()
        read = WorktreeScope(external, True, capture_directory_chain(external))
        with activity.prelaunch(
            footprint=WorktreeFootprint(reads=(read,))
        ) as contender:
            assert contender is None, (
                "root READ alone missed inherited external configuration"
            )
    finally:
        release.set()
        blocker.finish_root()
        claim.settle(durable=True)
        h.coordinator.cancel(reservation)


def test_cancelled_queued_native_job_unwinds_once_without_tracker_entry(
    native_review, monkeypatch, tmp_path
):
    h = native_review
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="tree")
    claim = activity.try_acquire(owner=owner, footprint=WorktreeFootprint(opaque=True))
    release = threading.Event()
    entered = [threading.Event(), threading.Event()]
    calls = []
    original = h.tracker.discover_baseline

    def held(handle):
        calls.append(str(handle.roots[0]))
        if len(calls) <= 2:
            entered[len(calls) - 1].set()
            assert release.wait(5)
        return original(handle)

    monkeypatch.setattr(h.tracker, "discover_baseline", held)
    reservations = []
    roots = [h.root, tmp_path / "root-two", tmp_path / "queued-root"]
    try:
        for root in roots:
            root.mkdir(exist_ok=True)
            source = h.coordinator._native.capture(
                h.bridge,
                session_id=h.session.id,
                conversation_id="tree",
                origin=WorkOrigin.MANUAL,
                roots=(root,),
                owner=owner,
                claim=claim,
            )
            reservations.append(h.coordinator.register([root], native_source=source))
        assert all(event.wait(3) for event in entered)
        wait_for(lambda: capacity.snapshot().tool_workers == 3)
        assert h.coordinator.cancel(reservations[2])
        assert capacity.snapshot().tool_workers == 3
        for reservation in reservations[:2]:
            h.coordinator.cancel(reservation)
        release.set()
        wait_for(lambda: capacity.snapshot().tool_workers == 0)
        assert str(roots[2]) not in calls
        assert reservations[2].completion.value == "cancelled"
        assert owner.is_admitted
    finally:
        release.set()
        for reservation in reservations:
            h.coordinator.cancel(reservation)
        assert h.coordinator.shutdown(5)
        owner.finish_root()
        claim.settle(durable=True)


def test_real_native_bridge_keeps_ordinary_capture_and_dispatch_behavior(native_review):
    from Tests.Chat.test_change_turn_tracking import (
        _calc_fence,
        _run,
        _SideEffectGateway,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    h = native_review
    h.bridge._gateway = _SideEffectGateway(
        [[_calc_fence()], ["done."]],
        side_effect_on_call=2,
        side_effect=lambda: (h.root / "agent-output.txt").write_text(
            "from native turn\n"
        ),
    )
    h.store.append_message(h.session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = h.store.append_message(
        h.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    run_id, outcome = _run(h.bridge, h.session, assistant.id, h.root)
    assert outcome.final_text.strip() == "done."
    assert h.coordinator.wait_idle(5)
    rows = h.bridge._db.change_snapshots_for_run(run_id)
    assert len(rows) == 1 and rows[0]["files_changed"] == 1, rows
    repo = h.tracker.service.repo_for_root(h.root)
    assert repo.file_bytes(rows[0]["baseline_sha"], "agent-output.txt") is None
    assert (
        repo.file_bytes(rows[0]["end_sha"], "agent-output.txt") == b"from native turn\n"
    )
    wait_for(lambda: not h.runtime.execution_capacity.snapshot().executions)


def test_pending_discovery_source_loss_is_unadmitted(native_review, monkeypatch):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    h = native_review
    assistant = h.store.append_message(
        h.session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )
    notice = h.store.capture_change_review_failure_notice(
        h.session.id,
        assistant.id,
        "⚠ change review skipped workspace: native snapshot unavailable",
    )
    capacity = h.runtime.execution_capacity
    capacity._read_settings = False
    capacity.set_tool_limits(1, 0)
    busy = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="busy")
    occupied = busy.reserve_tool()
    entries = []
    monkeypatch.setattr(h.tracker, "discover_baseline", entries.append)
    reservation = register(h, on_unadmitted=notice)
    try:
        wait_for(
            lambda: h.coordinator._states[reservation.id].pending_reason == "capacity"
        )
        h.runtime._agent_bridge = object()
        occupied.finish()
        wait_for(lambda: reservation.baseline_completion.value != "pending")
        assert not entries
        assert reservation.baseline_completion.value == "unadmitted"
        assert reservation.completion.value == "unadmitted"
        assert (
            reservation.admission_error == "native change-review admission unavailable"
        )
        assert reservation.await_baseline(0)
        wait_for(
            lambda: any(
                "change review skipped" in m.content
                for m in h.store.messages_for_session(h.session.id)
            )
        )
        reservation.notify_unadmitted()
        h.coordinator.finalize(reservation, run_id="no-admission", kind="turn")
        assert (
            len(
                [
                    m
                    for m in h.store.messages_for_session(h.session.id)
                    if "change review skipped" in m.content
                ]
            )
            == 1
        )
    finally:
        h.runtime._agent_bridge = h.bridge
        occupied.finish()
        busy.finish_root()
        h.coordinator.cancel(reservation)


def test_late_native_refusal_reaches_original_visible_marker(
    native_review, monkeypatch
):
    from Tests.Chat.test_change_turn_tracking import (
        _calc_fence,
        _run,
        _SideEffectGateway,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    h = native_review
    h.bridge._gateway = _SideEffectGateway([[_calc_fence()], ["done."]])
    h.store.append_message(h.session.id, role=ConsoleMessageRole.USER, content="hi")
    assistant = h.store.append_message(
        h.session.id, role=ConsoleMessageRole.ASSISTANT, content=""
    )
    original = h.coordinator.finalize
    captured = []

    def lose_source(reservation, **kwargs):
        captured.append(reservation)
        h.runtime._agent_bridge = object()
        return original(reservation, **kwargs)

    monkeypatch.setattr(h.coordinator, "finalize", lose_source)
    try:
        run_id, outcome = _run(h.bridge, h.session, assistant.id, h.root)
        assert outcome.final_text.strip() == "done."
        wait_for(lambda: captured[0].completion.value != "pending")
        assert captured[0].completion.value == "unadmitted"
        assert not h.bridge._db.change_snapshots_for_run(run_id)
        wait_for(
            lambda: any(
                "change review skipped" in m.content
                for m in h.store.messages_for_session(h.session.id)
            )
        )
        markers = [
            m
            for m in h.store.messages_for_session(h.session.id)
            if "change review skipped" in m.content
            or "change tracking failed" in m.content
        ]
        assert len(markers) == 1, (
            "late native refusal must disclose once to its original target"
        )
        original(captured[0], run_id=run_id, kind="turn")
        assert h.coordinator.wait_idle(2)
        assert (
            len(
                [
                    m
                    for m in h.store.messages_for_session(h.session.id)
                    if m.id == markers[0].id
                ]
            )
            == 1
        )
    finally:
        h.runtime._agent_bridge = h.bridge


def test_native_bridge_construction_keeps_capacity_settings_lazy(tmp_path, monkeypatch):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

    calls = []
    original = RuntimeCapacity.from_settings

    def observed():
        calls.append(threading.get_ident())
        return original()

    monkeypatch.setattr(RuntimeCapacity, "from_settings", observed)
    chat = CharactersRAGDB(tmp_path / "chat.db", "lazy-snapshot")
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=chat, app_config={}))
    try:
        store = runtime.ensure_chat_store()
        runtime.ensure_agent_bridge(
            store_factory=lambda: store, provider_gateway_factory=lambda: object()
        )
        assert not calls, (
            "constructing the issuer must not initialize capacity settings"
        )
        assert runtime._execution_capacity is None
        assert runtime.change_review_coordinator._subscriptions == ()
    finally:
        assert runtime.change_review_coordinator.shutdown(5)
        asyncio.run(runtime.dispose())
        chat.close_connection()


@pytest.mark.parametrize("phase", ["baseline", "end", "survivor"])
def test_refusal_notice_stays_on_original_target_once(
    native_review, monkeypatch, phase
):
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore

    h = native_review
    assistant = h.store.append_message(
        h.session.id, role=ConsoleMessageRole.ASSISTANT, content="original"
    )
    notice = h.store.capture_change_review_failure_notice(
        h.session.id,
        assistant.id,
        "⚠ change review skipped workspace: native snapshot unavailable",
    )
    source = h.coordinator._native.capture(
        h.bridge,
        session_id=h.session.id,
        conversation_id="conversation",
        origin=WorkOrigin.MANUAL,
        roots=(h.root,),
    )
    original_store = h.bridge._store
    replacement = ConsoleChatStore()
    replacement_session = replacement.ensure_session()
    bodies, publications, trajectory = [], [], []
    original_b = h.tracker.populate_prepared_baseline
    original_e = h.tracker.finish_turn
    original_publish = h.coordinator._publish
    monkeypatch.setattr(
        h.tracker,
        "populate_prepared_baseline",
        lambda *args: (bodies.append("baseline"), original_b(*args))[1],
    )
    monkeypatch.setattr(
        h.tracker,
        "finish_turn",
        lambda *args, **kwargs: (bodies.append("end"), original_e(*args, **kwargs))[1],
    )
    monkeypatch.setattr(
        h.coordinator,
        "_publish",
        lambda item: (publications.append(item), original_publish(item))[1],
    )
    if phase == "baseline":
        consume = h.coordinator._consume_operation_result

        def lose_after_discover(result):
            consume(result)
            if result.kind == "discover":
                h.bridge._store = replacement

        monkeypatch.setattr(
            h.coordinator, "_consume_operation_result", lose_after_discover
        )
    reservation = h.coordinator.register(
        [h.root], native_source=source, survivor_key=assistant.id, on_unadmitted=notice
    )
    try:
        if phase != "baseline":
            assert reservation.await_baseline(4)
            wait_for(lambda: reservation.baseline_completion.value != "pending")
            if phase == "survivor":
                (h.root / "a.txt").write_text("first end\n")
                run_id = h.bridge._db.create_run(
                    conversation_id="capture", agent_kind="primary"
                )
                h.coordinator.finalize(
                    reservation, run_id=run_id, kind="turn", has_live_survivors=True
                )
                wait_for(
                    lambda: h.coordinator._states[reservation.id].phase == "survivor"
                )
                wait_for(lambda: reservation._completion.ticket.retired.is_set())
            h.bridge._store = replacement
        # Navigate away and append a successor: the notice still belongs to its original assistant.
        later = h.store.append_message(
            h.session.id, role=ConsoleMessageRole.ASSISTANT, content="successor"
        )
        elsewhere = h.store.create_session(ephemeral=True)
        monkeypatch.setattr(
            h.store,
            "_record_trajectory_tool_marker",
            lambda *args: trajectory.append(args),
        )
        before_publications = len(publications)
        if phase == "end":
            h.coordinator.finalize(reservation, run_id="no-admission", kind="turn")
        elif phase == "survivor":
            h.coordinator.settle_survivors(assistant.id)
        wait_for(lambda: reservation.completion.value == "unadmitted")
        wait_for(
            lambda: any(
                "change review skipped" in m.content
                for m in h.store.messages_for_session(h.session.id)
            )
        )
        reservation.notify_unadmitted()
        h.coordinator.finalize(reservation, run_id="duplicate", kind="turn")
        h.coordinator.settle_survivors(assistant.id)
        messages = h.store.messages_for_session(h.session.id)
        markers = [m for m in messages if "change review skipped" in m.content]
        assert len(markers) == 1
        assert (
            messages.index(markers[0])
            == next(i for i, m in enumerate(messages) if m.id == assistant.id) + 1
        )
        assert messages.index(markers[0]) < next(
            i for i, m in enumerate(messages) if m.id == later.id
        )
        assert not h.store.messages_for_session(elsewhere.id)
        assert not replacement.messages_for_session(replacement_session.id)
        assert not trajectory
        assert len(publications) == before_publications
        assert bodies == (
            []
            if phase == "baseline"
            else ["baseline", "end"]
            if phase == "survivor"
            else ["baseline"]
        )
        assert reservation.baseline_completion.value == (
            "unadmitted" if phase == "baseline" else "observed"
        )
    finally:
        h.bridge._store = original_store
        h.coordinator.cancel(reservation)


@pytest.mark.parametrize("cancel", [False, True])
def test_cold_capacity_first_use_keeps_publisher_lifetime(
    tmp_path, monkeypatch, cancel
):
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity

    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    loaded, threads, bodies = [], [], []
    original = RuntimeCapacity.from_settings

    def held_load():
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        capacity = original()
        loaded.append(capacity)
        return capacity

    monkeypatch.setattr(RuntimeCapacity, "from_settings", held_load)
    chat = CharactersRAGDB(tmp_path / "chat.db", "cold-snapshot")
    runtime = ConsoleRuntime(SimpleNamespace(chachanotes_db=chat, app_config={}))
    store = runtime.ensure_chat_store()
    session = store.create_session(ephemeral=True)
    bridge = runtime.ensure_agent_bridge(
        store_factory=lambda: store, provider_gateway_factory=lambda: object()
    )
    coordinator = runtime.change_review_coordinator
    tracker = coordinator._tracker
    tracker.service = ShadowRepoService(tmp_path / "shadow")
    root = tmp_path / "root"
    root.mkdir()
    host = SimpleNamespace(**locals())
    original_close = coordinator._close_publisher
    monkeypatch.setattr(
        coordinator, "_close_publisher", lambda: (closed.set(), original_close())
    )
    discover = tracker.discover_baseline

    def observed(handle):
        bodies.append(current_execution_owner())
        assert coordinator._subscriptions
        return discover(handle)

    monkeypatch.setattr(tracker, "discover_baseline", observed)
    try:
        assert not entered.is_set() and runtime._execution_capacity is None
        reservation = register(host)
        assert entered.wait(3)
        assert threads == [coordinator._publisher.ident]
        if cancel:
            assert coordinator.cancel(reservation)
            assert not coordinator.shutdown(0)
            assert not closed.is_set()
        release.set()
        if cancel:
            assert coordinator.shutdown(5)
            assert not bodies
        else:
            assert reservation.await_baseline(4)
            assert bodies[0].capacity is loaded[0]
            assert (
                coordinator._native.capacity
                is runtime.execution_capacity
                is bridge.runtime_capacity
                is loaded[0]
            )
            coordinator.cancel(reservation)
            assert coordinator.shutdown(5)
        assert closed.is_set()
        assert not loaded[0]._tool_observers
        assert not runtime.worktree_activity._observers
        assert not loaded[0].snapshot().executions
    finally:
        release.set()
        assert coordinator.shutdown(5)
        asyncio.run(runtime.dispose())
        chat.close_connection()


def test_view_detach_and_session_navigation_preserve_original_capture(native_review):
    h = native_review
    source = h.coordinator._native.capture(
        h.bridge,
        session_id=h.session.id,
        conversation_id="conversation",
        origin=WorkOrigin.MANUAL,
        roots=(h.root,),
    )
    elsewhere = h.store.create_session(ephemeral=True)
    h.store.switch_session(elsewhere.id)
    assert h.runtime.detach_view()
    reservation = h.coordinator.register([h.root], native_source=source)
    try:
        assert reservation.await_baseline(4)
        wait_for(lambda: reservation.baseline_completion.value != "pending")
        assert reservation.baseline_completion.value == "observed"
        assert not reservation.admission_error
    finally:
        h.coordinator.cancel(reservation)
