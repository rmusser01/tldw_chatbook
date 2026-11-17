"""Original Windows Agents entries retain one live lease per selected path."""

import dis
import inspect
import json
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.test_windows_metadata_prerequisites import native_scope  # noqa: F401, F811
from tldw_chatbook.Agents import activation as agents
from tldw_chatbook.Backup_Recovery import activation, bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    register_pending,
)
from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Requires native Windows admission"
)


@pytest.fixture(autouse=True)
def sequential_activation_compatibility(monkeypatch):
    """Exercise the original admission sequence and its scalar injection edges."""
    monkeypatch.setattr(storage, "_agents_vector_definitions_current", lambda: False)


@pytest.fixture
def ordinary_agents(native_scope, monkeypatch):  # noqa: F811 -- pytest injects the fixture
    root, selector, data, startup = native_scope
    win = windows_files.WindowsOS()
    paths = [
        data / name
        for name in (
            "runs",
            "history",
            "workspace",
            "sandbox",
            "permissions",
            "scratch",
        )
    ]
    for path in paths:
        win.mkdir(path, 0o700)
    runs, history, workspace, sandbox, permissions, scratch = paths
    config = sys.modules.get("tldw_chatbook.config")
    if config is None:
        from tldw_chatbook import config
    monkeypatch.setattr(config, "_CONFIG_CACHE_SOURCE", selector)
    monkeypatch.setattr(
        config,
        "_CONFIG_CACHE",
        {
            "paths": {"data_dir": str(data)},
            "database": {"workspaces_db_path": str(workspace)},
            "tools": {"file_sandbox_root": str(sandbox)},
        },
    )
    roots_module = sys.modules.get("tldw_chatbook.Tools.workspace_file_roots")
    if roots_module is not None:
        monkeypatch.setattr(roots_module, "_default_registry_instance", None)
    service = SimpleNamespace(
        db=SimpleNamespace(db_path=runs),
        _store=SimpleNamespace(
            persistence=SimpleNamespace(db=SimpleNamespace(db_path=history))
        ),
    )
    rows = agents.selected_sources(service)
    assert tuple(path for _, path in rows) == (
        selector,
        runs,
        history,
        workspace,
        sandbox,
    )
    return SimpleNamespace(
        root=root,
        selector=selector,
        startup=startup,
        service=service,
        rows=rows,
        extra=(("mcp.permissions", permissions), ("agents.history", scratch)),
    )


def _warm(paths):
    for path in paths:
        for _ in range(2):
            with storage.acquire_storage(path):
                pass


@contextmanager
def _observe_guard(
    *, after_first_lease=None, after_full_registry=None, after_control=None
):
    codes = {
        bootstrap._parse_control_records.__code__: "control",
        bootstrap._parse_registry.__code__: "registry",
        windows_files._Native.open_handle.__code__: "opens",
        windows_files._Native.security.__code__: "security",
    }
    counts = dict.fromkeys(codes.values(), 0)
    original_scope = inspect.unwrap(activation.execution_scope).__code__
    guard = inspect.unwrap(RecoveryAdmissionGuard.execution).__code__
    acquire = storage.acquire_storage.__code__
    close = storage.StorageLease.close.__code__
    registry_lock = bootstrap._registry_read_lock.__wrapped__.__code__
    registry_yield = next(
        instruction.offset
        for instruction in dis.get_instructions(registry_lock)
        if instruction.opname == "YIELD_VALUE"
    )
    acquired, retired = [], []
    previous = sys.getprofile()
    full_control = False

    def observe(frame, event, result):
        nonlocal full_control
        if previous is not None:
            previous(frame, event, result)
        if (
            event == "return"
            and frame.f_code is acquire
            and type(result) is storage.StorageLease
        ):
            acquired.append((frame.f_locals["path"], result))
            if len(acquired) == 1 and after_first_lease is not None:
                after_first_lease(result)
        elif event == "return" and frame.f_code is close:
            lease = frame.f_locals["self"]
            if any(lease is token for _, token in acquired):
                assert lease not in storage._live_leases
                retired.append(lease)
        elif event == "call" and frame.f_code in codes:
            parent = frame.f_back
            while parent is not None and parent.f_code is not guard:
                if parent.f_code is acquire:
                    return
                parent = parent.f_back
            # Original execution_scope or the replacement guard's finite proof.
            if parent is not None or frame.f_back.f_code is original_scope:
                counts[codes[frame.f_code]] += 1
                if frame.f_code is bootstrap._parse_control_records.__code__:
                    full_control = frame.f_locals["activation"]
        elif (
            event == "return"
            and frame.f_code is bootstrap._parse_control_records.__code__
            and frame.f_locals["ordinary_only"]
            and after_control is not None
        ):
            parent = frame.f_back
            while parent is not None and parent.f_code is not guard:
                if parent.f_code is acquire:
                    return
                parent = parent.f_back
            if parent is not None:
                after_control(result)
        elif (
            event == "return"
            and frame.f_code is registry_lock
            and frame.f_lasti != registry_yield
            and full_control
            and after_full_registry is not None
        ):
            parent = frame.f_back
            while parent is not None and parent.f_code is not guard:
                if parent.f_code is acquire:
                    return
                parent = parent.f_back
            if parent is not None:
                full_control = False
                after_full_registry(acquired)

    sys.setprofile(observe)
    try:
        yield SimpleNamespace(counts=counts, acquired=acquired, retired=retired)
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("bound", [True, False])
def test_warm_agents_entry_uses_fixed_fresh_native_observations(
    ordinary_agents, request, monkeypatch, bound
):
    case = ordinary_agents
    if not bound:
        case.startup.close()
        case.selector = case.selector.with_name("unbound-config")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(case.selector))
        config = sys.modules["tldw_chatbook.config"]
        monkeypatch.setattr(config, "_CONFIG_CACHE_SOURCE", case.selector)
        unbound_startup = storage.acquire_storage()
        request.addfinalizer(unbound_startup.close)
        case.rows = agents.selected_sources(case.service)
    all_rows = case.rows + case.extra
    _warm(path for _, path in all_rows)

    def check(sources, expected_acquired):
        with _observe_guard() as observed:
            with agents.execution(case.service, sources=sources):
                state = agents._guard.context.get()
                assert state.live
                assert all(
                    lease in storage._live_leases for lease in state.leases.values()
                )
                assert all(
                    lease.execution_context(path)[1]
                    == (("profile",) if bound else (storage.UNBOUND_NAMESPACE,))
                    for path, lease in state.leases.items()
                )
                assert [path for path, _ in observed.acquired] == expected_acquired
        assert [lease for _, lease in reversed(observed.acquired)] == observed.retired
        assert observed.counts["opens"] and observed.counts["security"]
        request.node.user_properties.append(
            (
                f"rows_{len(state.sources)}_acquisitions_{len(expected_acquired)}",
                json.dumps(observed.counts),
            )
        )
        return observed.counts

    with agents.execution(case.service):
        five_nested = check((), [])
        six = check(case.extra[:1], [case.extra[0][1]])
        seven = check(case.extra, [path for _, path in case.extra])
    five = check((), [path for _, path in case.rows])
    assert agents._guard.context.get() is None
    for counts in (five, five_nested, six, seven):
        assert (counts["control"], counts["registry"]) == (2, 2), counts


@pytest.mark.parametrize(
    "change",
    ["activation_intent", "registry", "registry_lock", "selector", "lease", "scope"],
)
def test_nested_warm_entry_refuses_changed_fresh_evidence(
    ordinary_agents, monkeypatch, change
):
    case = ordinary_agents
    _warm(path for _, path in case.rows + case.extra)
    effect = []
    calls = []
    win = windows_files.WindowsOS()
    marker = case.root / "activation-update-task7.json"
    other = case.selector.with_name("other")
    fd = win.open(other, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
    win.write(fd, b"different selector")
    win.close(fd)
    with agents.execution(case.service):
        parent = agents._guard.context.get()
        lost = parent.leases[case.rows[0][1]]

        def mutate(_acquired):
            if calls:
                return
            calls.append(change)
            if change == "activation_intent":
                marker.write_bytes(b"{}")
            elif change == "registry":
                # An actual independent registry publication and epoch change.
                new_root = case.root.parent / "additional-root"
                win.mkdir(new_root, 0o700)
                admission_authority(case.root).register("other", (new_root,))
            elif change == "registry_lock":
                lock = case.root / "admission" / "registry.lock"
                win.rename(lock, lock.with_name("former.lock"))
                win.close(
                    win.open(lock, win.O_CREAT | win.O_EXCL | win.O_WRONLY, 0o600)
                )
            elif change == "selector":
                monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
            elif change == "lease":
                lost.close()
            else:
                hold = storage._holds[lost._key]
                monkeypatch.setitem(
                    hold.authority._observed_groups, hold.names, ("profile", "another")
                )

        try:
            with _observe_guard(after_full_registry=mutate) as observed:
                with pytest.raises(agents.AgentActivationRequired):
                    with agents.execution(case.service):
                        effect.append(True)
            assert agents._guard.context.get() is parent
            assert not effect and calls == [change]
            assert not observed.acquired
        finally:
            marker.unlink(missing_ok=True)
            monkeypatch.setenv("TLDW_CONFIG_PATH", str(case.selector))


def test_first_cold_row_uses_original_order_and_native_checks(ordinary_agents):
    case = ordinary_agents
    with _observe_guard() as observed:
        with agents.execution(case.service):
            assert [path for path, _ in observed.acquired] == [
                path for _, path in case.rows
            ]
    assert [lease for _, lease in reversed(observed.acquired)] == observed.retired
    assert (observed.counts["control"], observed.counts["registry"]) == (25, 20)


def test_later_cold_source_is_independent_and_keeps_original_gate(ordinary_agents):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    with agents.execution(case.service):
        parent = agents._guard.context.get()
        with _observe_guard() as observed:
            with agents.execution(case.service, sources=case.extra[:1]):
                nested = agents._guard.context.get()
                assert nested.parent is parent
                assert all(
                    nested.leases[path] is parent.leases[path] for _, path in case.rows
                )
        assert [path for path, _ in observed.acquired] == [case.extra[0][1]]
        assert observed.retired == [observed.acquired[0][1]]
        assert (observed.counts["control"], observed.counts["registry"]) == (32, 26)
        assert all(lease in storage._live_leases for lease in parent.leases.values())


def test_initial_pending_stops_before_second_row_or_effect(ordinary_agents):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    control = case.root.parent / "control"
    windows_files.WindowsOS().mkdir(control, 0o700)

    def pend(_lease):
        register_pending(case.root, "task7", ("profile",), control, (case.selector,))

    effect = []
    with _observe_guard(after_first_lease=pend) as observed:
        with pytest.raises(agents.AgentActivationRequired):
            with agents.execution(case.service):
                effect.append(True)
    assert not effect
    assert [path for path, _ in observed.acquired] == [case.rows[0][1]]
    assert observed.retired == [observed.acquired[0][1]]


def test_failed_first_native_startup_proof_is_not_retried(ordinary_agents):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    broken = case.root / "unrecognized-record.json"

    def make_unreadable(_lease):
        broken.write_bytes(b"{}")

    def restore(result):
        assert result is None
        broken.unlink()

    try:
        with _observe_guard(
            after_first_lease=make_unreadable, after_control=restore
        ) as observed:
            with pytest.raises(agents.AgentActivationRequired):
                with agents.execution(case.service):
                    pytest.fail("A failed native proof was retried into execution")
        assert [path for path, _ in observed.acquired] == [case.rows[0][1]]
        assert observed.retired == [observed.acquired[0][1]]
    finally:
        broken.unlink(missing_ok=True)


def test_recursive_initialization_preserves_original_acquisition_refusal(
    ordinary_agents,
):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    attempt = storage._Acquisition()
    try:
        with (
            attempt.initializing(case.root, case.rows[0][1]),
            _observe_guard() as observed,
        ):
            with pytest.raises(agents.AgentActivationRequired):
                with agents.execution(case.service):
                    pytest.fail("Recursive initializer entered domain callback")
        assert not observed.acquired and not observed.retired
        assert (observed.counts["control"], observed.counts["registry"]) == (0, 0)
    finally:
        attempt.close()


def test_existing_leases_during_initialization_use_original_gate(ordinary_agents):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    with agents.execution(case.service):
        attempt = storage._Acquisition()
        try:
            with (
                attempt.initializing(case.root, case.rows[0][1]),
                _observe_guard() as observed,
            ):
                with agents.execution(case.service):
                    assert not observed.acquired
            assert (observed.counts["control"], observed.counts["registry"]) == (25, 20)
        finally:
            attempt.close()


def test_substituted_agent_owner_mapping_keeps_original_gate(
    ordinary_agents, monkeypatch
):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    owners = []

    def custom(owner):
        owners.append(owner)
        return "config", owner

    monkeypatch.setattr(agents._guard, "owners", custom)
    with _observe_guard() as observed:
        with agents.execution(case.service):
            assert owners == [owner for owner, _ in case.rows]
    assert (observed.counts["control"], observed.counts["registry"]) == (25, 20)


def test_native_restored_entry_requires_every_observed_owner(ordinary_agents):
    case = ordinary_agents
    case.startup.close()
    control = case.root.parent / "restore-control"
    win = windows_files.WindowsOS()
    win.mkdir(control, 0o700)
    authority = admission_authority(case.root)
    owners = tuple(owner for owner, _ in case.rows)
    register_pending(
        case.root, "restore-test7", ("profile",), control, (case.selector,)
    )
    with authority.maintenance(("profile",), 2) as session:
        activation.bind_activation(
            case.root,
            "restore-test7",
            case.selector,
            "generation",
            owners,
            session=session,
        )
    win.unlink(case.root / ("pending-" + bootstrap._key("restore-test7") + ".json"))
    approvals = activation.ActivationStore(control / "activation")
    approvals.approve("generation", "config")
    effects = []
    with _observe_guard() as denied:
        with pytest.raises(agents.AgentActivationRequired):
            with agents.execution(case.service):
                effects.append("unapproved")
    assert [path for path, _ in denied.acquired] == [path for _, path in case.rows[:2]]
    assert [lease for _, lease in reversed(denied.acquired)] == denied.retired
    for owner in owners[1:]:
        approvals.approve("generation", owner)
    with agents.execution(case.service):
        effects.append("approved")
    assert effects == ["approved"]


def test_native_worker_guard_retains_leases_until_original_body_exits(ordinary_agents):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    entered, release = threading.Event(), threading.Event()
    worker_leases, errors = [], []

    @agents.worker_guard(case.service)
    def effect():
        worker_leases.extend(agents._guard.context.get().leases.values())
        entered.set()
        assert release.wait(5)
        assert all(lease in storage._live_leases for lease in worker_leases)

    def run():
        try:
            effect()
        except BaseException as error:
            errors.append(error)
            entered.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(5) and not errors
        assert len(worker_leases) == len(case.rows)
        assert all(lease in storage._live_leases for lease in worker_leases)
        pause = storage._begin_local_pause()
        try:
            assert not pause.drain(time.monotonic() + 0.02)
        finally:
            pause.resume()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and not errors
    assert all(lease not in storage._live_leases for lease in worker_leases)
