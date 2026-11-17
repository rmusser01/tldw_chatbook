"""One Windows admission proof preserves independent ordered Agents leases."""

import json
import sys
import threading
import time
from collections import Counter

import pytest

from Tests.Backup_Recovery.test_agents_ordinary_activation_batch import (
    _warm,
    ordinary_agents as _ordinary_agents,  # noqa: F401 - pytest fixture
)
from tldw_chatbook.Agents import activation as agents
from tldw_chatbook.Backup_Recovery import activation, bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Utils import windows_files

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows admission"
)


@pytest.fixture
def native_scope(tmp_path, monkeypatch):
    win = windows_files.WindowsOS()
    parent = tmp_path / "explicit-user-private"
    win.mkdir(parent, 0o700)
    root, selector, data = (parent / name for name in ("bootstrap", "config", "data"))
    fd = win.open(selector, win.O_CREAT | win.O_EXCL | win.O_RDWR, 0o600)
    try:
        win.write(fd, b"scope1")
    finally:
        win.close(fd)
    win.mkdir(data, 0o700)
    authority = admission_authority(root)
    authority.register("profile", (selector, data))
    # The implicit 100 ms enrollment deadline can expire on a busy Windows
    # runner. Use the real explicit maintenance entry for test setup.
    with authority.maintenance((storage.UNBOUND_NAMESPACE, "profile"), 5) as session:
        bind_profile(root, selector, ("profile",), root / "admission", session=session)
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0)
    with storage.acquire_storage() as startup:
        yield root, selector, data, startup


@pytest.fixture
def ordinary_agents(request):
    # Import the real config under pytest's bootstrap profile before the bound
    # fixture selects its separate native authority. No fixture creates a store.
    from tldw_chatbook import config  # noqa: F401

    return request.getfixturevalue("_ordinary_agents")


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("bound", [True, False])
def test_one_native_observation_preserves_independent_leases(
    ordinary_agents, request, monkeypatch, extra, bound
):
    case = ordinary_agents
    names = ("profile",)
    if not bound:
        case.startup.close()
        case.selector = case.selector.with_name("unbound-config")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(case.selector))
        monkeypatch.setattr(
            sys.modules["tldw_chatbook.config"], "_CONFIG_CACHE_SOURCE", case.selector
        )
        unbound = storage.acquire_storage()
        request.addfinalizer(unbound.close)
        case.rows = agents.selected_sources(case.service)
        names = (storage.UNBOUND_NAMESPACE,)
    # Separate owner on the same path still gets its own activation row only.
    duplicate = (("mcp.permissions", case.rows[0][1]),)
    sources = duplicate + (case.extra if extra else ())
    paths = tuple(dict.fromkeys(path for _, path in sources + case.rows))
    _warm(paths)
    input_code = storage._windows_inputs.__wrapped__.__code__
    birth = storage.StorageLease.__init__.__code__
    close = storage.StorageLease.close.__code__
    codes = {
        storage._observe_evidence.__code__: "union",
        activation._ordinary_activation_observation.__code__: "activation",
        windows_files._Native.open_handle.__code__: "opens",
        windows_files._Native.security.__code__: "security",
        windows_files._Native.ntfs.__code__: "volume",
    }
    previous = sys.getprofile()
    leases, retired, inputs = [], [], set()
    counts = Counter()
    before = set(storage._live_leases)

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        code = frame.f_code
        if event == "call":
            if code is input_code:
                inputs.add(frame)
            if code in codes:
                counts[codes[code]] += 1
        if event == "return" and code is birth:
            leases.append(frame.f_locals["self"])
        if event == "return" and code is close:
            retired.append(frame.f_locals["self"])

    sys.setprofile(observe)
    try:
        with agents.execution(case.service, sources=sources):
            state = agents._guard.context.get()
            assert tuple(state.leases) == paths
            assert tuple(state.leases.values()) == tuple(leases)
            assert len(set(leases)) == len(paths)
            assert all(
                lease.execution_context(path) == (case.root, names)
                for path, lease in state.leases.items()
            )
            assert state.sources == sources + case.rows
    finally:
        sys.setprofile(previous)

    assert retired == list(reversed(leases))
    assert set(storage._live_leases) == before
    assert agents._guard.context.get() is None
    counts["inputs"] = len(inputs)
    request.node.user_properties.append(
        ("agents_vector_native_counts", json.dumps(counts))
    )
    assert counts["activation"] == 2
    assert counts["opens"] and counts["security"] and counts["volume"]
    assert (counts["inputs"], counts["union"]) == (1, 1), counts


@pytest.mark.parametrize(
    "change", ["pending", "source", "selector", "reader", "pause", "path"]
)
def test_vector_refuses_drift_without_retry(ordinary_agents, monkeypatch, change):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    before = set(storage._live_leases)
    attempts = set(storage._pending_acquisitions)
    previous = sys.getprofile()
    control = activation._ordinary_activation_observation.__code__
    inputs = storage._windows_inputs.__wrapped__.__code__
    acquisitions, changes, pauses = set(), [], []
    original_path = case.service.db.db_path
    marker = case.root / "activation-update-vector.json"

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "call" and frame.f_code is inputs:
            acquisitions.add(frame)
        if changes:
            return
        if change in {"pending", "source", "selector", "reader"}:
            if frame.f_code is not control or event != (
                "call" if change == "pending" else "return"
            ):
                return
            changes.append(change)
            assert set(storage._live_leases) == before
            if change == "pending":
                marker.write_bytes(b"{}")
            elif change == "source":
                monkeypatch.setattr(
                    agents._guard,
                    "sources",
                    lambda _: pytest.fail("substituted source reader used"),
                )
            elif change == "selector":
                monkeypatch.setenv(
                    "TLDW_CONFIG_PATH", str(case.selector.with_name("foreign"))
                )
            else:
                monkeypatch.setattr(
                    storage,
                    "_observe_evidence",
                    lambda _: pytest.fail("substituted reader used"),
                )
        elif event == "call" and frame.f_code is storage._observe_evidence.__code__:
            changes.append(change)
            assert len(set(storage._live_leases) - before) == 5
            if change == "pause":
                pauses.append(storage._begin_local_pause())
            else:
                win = windows_files.WindowsOS()
                win.rename(original_path, original_path.with_name("replacement-target"))
                win.mkdir(original_path, 0o700)

    sys.setprofile(observe)
    try:
        with pytest.raises(agents.AgentActivationRequired):
            with agents.execution(case.service):
                pytest.fail("changed vector entered the Agents body")
    finally:
        sys.setprofile(previous)
        marker.unlink(missing_ok=True)
        for pause in pauses:
            pause.resume()
    assert changes == [change]
    assert len(acquisitions) == (1 if change in {"pause", "path"} else 0)
    assert set(storage._live_leases) == before
    assert set(storage._pending_acquisitions) == attempts


def test_custom_storage_acquisition_body_retains_sequential_lease_ownership(
    ordinary_agents, monkeypatch
):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    original = storage._acquire_ordinary
    acquired = []

    def acquire(path, **kwargs):
        acquired.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(storage, "_acquire_ordinary", acquire)
    with agents.execution(case.service):
        assert len(agents._guard.context.get().leases) == 5
    assert acquired == [path for _, path in case.rows]


@pytest.mark.parametrize("route", ["cold", "nested", "restored"])
def test_ineligible_vector_uses_original_scalar_acquisition(ordinary_agents, route):
    case = ordinary_agents
    if route != "cold":
        _warm(path for _, path in case.rows)
    association = case.root / f"activation-{bootstrap._key(str(case.selector))}.json"
    if route == "restored":
        # Recognized activation-only content is positively nonordinary. An
        # unpaired association must still follow the original refusal route.
        witness = dict(
            operation_id="task14",
            generation="g1",
            owners=["config"],
            namespaces=["profile"],
            store_root=str(case.root / "review"),
        )
        association.write_text(
            json.dumps(dict(version=1, selector=str(case.selector), activation=witness))
        )
    previous = sys.getprofile()
    paths = []

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event == "call" and frame.f_code is storage.acquire_storage.__code__:
            paths.append(frame.f_locals["path"])

    def run(extra=()):
        sys.setprofile(observe)
        try:
            with agents.execution(case.service, sources=extra):
                assert route != "restored"
        finally:
            sys.setprofile(previous)

    if route == "restored":
        with pytest.raises(agents.AgentActivationRequired):
            run()
        assert paths == [case.selector]
    elif route == "nested":
        _warm(path for _, path in case.extra)
        with agents.execution(case.service):
            outer = agents._guard.context.get()
            run(case.extra)
            assert agents._guard.context.get() is outer and outer.live
        assert paths == [path for _, path in case.extra]
    else:
        assert route == "cold"
        _warm(path for _, path in case.rows[:-1])
        run()
        assert paths == [path for _, path in case.rows]


@pytest.mark.parametrize("late", [False, True])
def test_unqualified_lease_class_keeps_original_or_refuses(
    ordinary_agents, monkeypatch, late
):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    original = storage.StorageLease
    spawned = []
    previous = sys.getprofile()

    class CustomLease(original):
        def __init__(self, key):
            super().__init__(key)
            spawned.append(self)

    def substitute(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if (
            event == "return"
            and frame.f_code is activation._ordinary_activation_observation.__code__
        ):
            monkeypatch.setattr(storage, "StorageLease", CustomLease)

    if not late:
        monkeypatch.setattr(storage, "StorageLease", CustomLease)
        with agents.execution(case.service):
            assert tuple(agents._guard.context.get().leases.values()) == tuple(spawned)
        assert len(spawned) == 5
        return
    sys.setprofile(substitute)
    try:
        with pytest.raises(agents.AgentActivationRequired):
            with agents.execution(case.service):
                pytest.fail("substituted lease class entered the Agents body")
    finally:
        sys.setprofile(previous)
    assert storage.StorageLease is CustomLease and not spawned


def test_vector_uncertain_metadata_close_keeps_exact_record(
    ordinary_agents, monkeypatch
):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    native = windows_files._native()
    close = native.kernel.CloseHandle
    target = case.service.db.db_path
    previous = sys.getprofile()
    before = set(storage._live_leases)
    held, closes = [], []

    def failed_close(handle):
        if held and handle == held[0].handle:
            closes.append(handle)
            raise OSError("test close failed before native retirement")
        return close(handle)

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        record = frame.f_locals.get("_metadata_record")
        if (
            event == "return"
            and frame.f_code is windows_files._Native.open_handle.__code__
            and record is not None
            and record.path == target
            and not held
        ):
            assert len(set(storage._live_leases) - before) == 5
            held.append(record)
            monkeypatch.setattr(native.kernel, "CloseHandle", failed_close)

    sys.setprofile(observe)
    try:
        with pytest.raises(agents.AgentActivationRequired) as caught:
            with agents.execution(case.service):
                pytest.fail("uncertain close entered the Agents body")
        uncertainty = caught.value.__context__
        assert isinstance(uncertainty, windows_files._AdmissionMetadataCloseError)
        assert uncertainty.failed_handles == tuple(held)
        assert closes == [held[0].handle]
        assert set(storage._live_leases) == before
        info = native.info(held[0].handle)
        assert (info.volume, (info.index_high << 32) | info.index_low) == held[
            0
        ].identity
    finally:
        sys.setprofile(previous)
        # The injected close demonstrably happened before the native effect.
        if closes:
            native.check(close(closes[0]))


def test_vector_reobserves_registry_after_actual_lock_wait(ordinary_agents):
    case = ordinary_agents
    _warm(path for _, path in case.rows)
    entered, locked, waiting = threading.Event(), threading.Event(), threading.Event()
    authority = admission_authority(case.root)
    registry = case.root / "admission" / "registry.json"
    original = registry.read_bytes()
    errors = []
    previous = sys.getprofile()
    before = set(storage._live_leases)

    def hold_registry():
        try:
            assert entered.wait(10)
            with bootstrap.pinned_directory(authority.control_root) as parent:
                with authority._lock(parent, "registry.lock", bootstrap.fcntl.LOCK_EX):
                    locked.set()
                    assert waiting.wait(10)
                    # Hold the real exclusive lock through the peer's LockFileEx call.
                    time.sleep(0.1)
                    data = json.loads(original)
                    data["entries"]["profile"]["historical"].append("external:test")
                    registry.write_text(json.dumps(data), encoding="utf-8")
        except BaseException as error:
            errors.append(error)
            locked.set()

    def observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "call":
            return
        if (
            frame.f_code is storage._windows_inputs.__wrapped__.__code__
            and not entered.is_set()
        ):
            entered.set()
            assert locked.wait(10)
        if (
            frame.f_code is windows_files.WindowsLocks.flock.__code__
            and entered.is_set()
        ):
            assert len(set(storage._live_leases) - before) == 5
            waiting.set()

    worker = threading.Thread(target=hold_registry)
    worker.start()
    sys.setprofile(observe)
    try:
        with pytest.raises(agents.AgentActivationRequired):
            with agents.execution(case.service):
                pytest.fail("registry changed during real lock wait")
    finally:
        sys.setprofile(previous)
        entered.set()
        waiting.set()
        worker.join(10)
        registry.write_bytes(original)
    assert not worker.is_alive() and not errors and locked.is_set() and waiting.is_set()
    assert set(storage._live_leases) == before
