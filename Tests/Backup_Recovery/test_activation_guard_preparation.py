"""One guard entry shares control reads while retaining native owner checks."""

import sys
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import test_activation_preparation as activation_cases
from Tests.Backup_Recovery.test_generation_witness_observation import (
    _insert_pending,
    native_observation_counts,
)
from tldw_chatbook.Backup_Recovery import activation, admission_runtime, bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission_runtime import RecoveryAdmissionGuard

activation_case = activation_cases.activation_case


class _AdmissionRefused(PermissionError):
    pass


def _guard(selector):
    return RecoveryAdmissionGuard(
        "finite_activation_preparation_test",
        error=_AdmissionRefused,
        sources=lambda _service: (
            ("config", selector),
            ("agents.run", selector),
        ),
    )


def test_guard_prepares_one_original_observation_per_entry(activation_case, request):
    root, _control, selector, _store = activation_case
    guard = _guard(selector)
    bodies, observations = [], []
    with guard.execution():
        parent = guard.context.get()
        lease = parent.leases[selector]
        assert lease.execution_context(selector) == (root, ("profile",))
        for entry in range(2):
            # The real parent owns the unchanged lease. Only this fresh nested
            # pre-yield is counted, not the independent acquire_storage route.
            with native_observation_counts() as counts:
                with guard.execution():
                    current = guard.context.get()
                    assert current is not parent
                    assert current.parent is parent
                    assert current.sources == (
                        ("config", selector),
                        ("agents.run", selector),
                    )
                    assert current.leases[selector] is lease
                    assert lease.execution_context(selector) == (root, ("profile",))
                    bodies.append(entry)
            observations.append(dict(counts))
            assert guard.context.get() is parent
            assert not current.live
            assert lease.execution_context(selector) == (root, ("profile",))

    assert bodies == [0, 1]
    assert guard.context.get() is None
    assert not parent.live
    assert [item.get("records", 0) for item in observations] == [1, 1], observations
    assert [item.get("registry", 0) for item in observations] == [1, 1], observations
    for entry, counts in enumerate(observations, 1):
        request.node.user_properties.extend(
            (f"guard_entry_{entry}_{name}", counts[name])
            for name in ("records", "registry", "native_opens")
            if name in counts
        )


@contextmanager
def _original_events(observe):
    previous = sys.getprofile()

    def chained(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        observe(frame, event, result)

    sys.setprofile(chained)
    try:
        yield
    finally:
        sys.setprofile(previous)


@pytest.mark.parametrize("revoke_second", [False, True])
def test_guard_checks_every_original_owner_live(activation_case, revoke_second):
    _root, _control, selector, store = activation_case
    guard = _guard(selector)
    decisions = []
    startup_code = bootstrap._startup_permission_from_records.__code__
    allowed_code = activation.ActivationStore.allowed.__code__
    approval = store._generation("g") / (
        "approved-" + activation._key("agents.run") + ".json"
    )

    def observe(frame, event, result):
        if frame.f_code is startup_code and event == "call":
            decisions.append(("startup", None))
        elif frame.f_code is allowed_code:
            owner = frame.f_locals["owner"]
            if event == "call":
                decisions.append(("owner", owner))
            elif event == "return" and result is True and owner == "config":
                if revoke_second:
                    approval.unlink()

    bodies = []
    with guard.execution():
        parent = guard.context.get()
        with _original_events(observe):
            if revoke_second:
                with pytest.raises(_AdmissionRefused):
                    with guard.execution():
                        bodies.append("must not execute")
            else:
                with guard.execution():
                    bodies.append("accepted")
        assert guard.context.get() is parent
    assert decisions == [
        ("startup", None),
        ("owner", "config"),
        ("startup", None),
        ("owner", "agents.run"),
    ]
    assert bodies == ([] if revoke_second else ["accepted"])


def test_guard_control_drift_after_last_owner_refuses_body(activation_case):
    root, control, selector, _store = activation_case
    guard = _guard(selector)
    allowed_code = activation.ActivationStore.allowed.__code__
    changed, bodies = [], []

    def observe(frame, event, result):
        if (
            frame.f_code is allowed_code
            and event == "return"
            and result is True
            and frame.f_locals["owner"] == "agents.run"
            and not changed
        ):
            _insert_pending(root, control, selector)
            changed.append(True)

    with guard.execution():
        parent = guard.context.get()
        with _original_events(observe), pytest.raises(_AdmissionRefused):
            with guard.execution():
                bodies.append("must not execute")
        assert guard.context.get() is parent
    assert changed == [True]
    assert bodies == []


@pytest.mark.parametrize("change", ["selector", "lease", "permission_callback"])
def test_guard_rechecks_after_original_observation_completion(
    activation_case, monkeypatch, change
):
    _root, _control, selector, _store = activation_case
    guard = _guard(selector)
    observation_code = bootstrap._control_observation.__wrapped__.__code__
    allowed_code = activation.ActivationStore.allowed.__code__
    other = selector.with_name("different-selector.toml")
    other.write_bytes(selector.read_bytes())
    other.chmod(0o600)
    completed, owners, bodies = [], [], []

    with guard.execution():
        parent = guard.context.get()
        lease = parent.leases[selector]
        with monkeypatch.context() as mutation:

            def observe(frame, event, result):
                if (
                    frame.f_code is allowed_code
                    and event == "return"
                    and result is True
                ):
                    owners.append(frame.f_locals["owner"])
                elif (
                    frame.f_code is observation_code
                    and event == "return"
                    and result is None
                    and owners == ["config", "agents.run"]
                    and not completed
                ):
                    completed.append(change)
                    if change == "selector":
                        mutation.setenv("TLDW_CONFIG_PATH", str(other))
                    elif change == "lease":
                        lease.close()
                    else:
                        mutation.setattr(
                            activation, "activation_permission", lambda *a, **k: False
                        )

            with _original_events(observe), pytest.raises(_AdmissionRefused):
                with guard.execution():
                    bodies.append("must not execute")
        assert guard.context.get() is parent
    assert completed == [change], "original final observation was not reached"
    assert owners == ["config", "agents.run"]
    assert bodies == []


def test_later_guard_reads_new_pending_evidence(activation_case):
    root, control, selector, _store = activation_case
    guard = _guard(selector)
    bodies = []
    with guard.execution():
        with native_observation_counts() as first:
            with guard.execution():
                bodies.append("accepted")
        _insert_pending(root, control, selector)
        with native_observation_counts() as second, pytest.raises(_AdmissionRefused):
            with guard.execution():
                bodies.append("must not execute")
    assert bodies == ["accepted"]
    assert first["records"] == first["registry"] == 1, dict(first)
    assert second["records"] == second["registry"] == 1, dict(second)


def test_first_owner_refusal_never_acquires_later_path(activation_case):
    root, _control, selector, store = activation_case
    later_path = selector.with_name("unreachable.json")
    sources = [("config", selector)]
    guard = RecoveryAdmissionGuard(
        "first_refusal_test",
        error=_AdmissionRefused,
        sources=lambda _service: tuple(sources),
    )
    acquire_code = storage.acquire_storage.__code__
    acquired, checked = [], []
    allowed_code = activation.ActivationStore.allowed.__code__

    def observe(frame, event, _result):
        if event == "call" and frame.f_code is acquire_code:
            acquired.append(frame.f_locals["path"])
        elif event == "call" and frame.f_code is allowed_code:
            checked.append(frame.f_locals["owner"])

    with guard.execution():
        lease = guard.context.get().leases[selector]
        sources.append(("agents.run", later_path))
        (
            store._generation("g") / ("approved-" + activation._key("config") + ".json")
        ).unlink()
        with _original_events(observe), pytest.raises(_AdmissionRefused):
            with guard.execution():
                pytest.fail("first denied owner entered the body")
        assert lease.execution_context(selector) == (root, ("profile",))
    assert checked == ["config"]
    assert later_path not in acquired


def test_custom_execution_scope_keeps_original_shape_and_order(
    activation_case, monkeypatch
):
    _root, _control, selector, _store = activation_case
    guard = _guard(selector)
    original_scope = admission_runtime.execution_scope
    observation_code = bootstrap._control_observation.__wrapped__.__code__
    calls, events = [], []

    def observe(frame, event, _result):
        if frame.f_code is observation_code and event == "call":
            events.append("observation")

    with guard.execution():
        parent = guard.context.get()
        lease = parent.leases[selector]

        @contextmanager
        def custom_scope(owners, path=None, *, retained=None):
            calls.append((owners, path, retained))
            events.append(owners[0])
            if owners == ("config",):
                with original_scope(owners, path, retained=retained) as allowed:
                    yield allowed
            else:
                yield False

        with monkeypatch.context() as mutation:
            mutation.setattr(admission_runtime, "execution_scope", custom_scope)
            with _original_events(observe), pytest.raises(_AdmissionRefused):
                with guard.execution():
                    pytest.fail("custom refusal entered the body")
        assert guard.context.get() is parent
    assert calls == [
        (("config",), selector, lease),
        (("agents.run",), selector, lease),
    ]
    assert events[0] == "config", "shared reads preceded the custom public route"
    assert "observation" in events
    assert events[-1] == "agents.run"


@pytest.mark.parametrize("retire_first", [False, True])
def test_guard_shares_observation_across_distinct_enrolled_files(
    tmp_path, monkeypatch, request, retire_first
):
    import os
    from contextlib import nullcontext

    from Tests.Backup_Recovery.test_activation_binding import bind, finish_pending
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        register_pending,
    )

    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n")
    selector.chmod(0o600)
    run_path = tmp_path / "runs.json"
    run_path.write_text("{}\n")
    run_path.chmod(0o600)
    root, control = tmp_path / "bootstrap", tmp_path / "control"
    control.mkdir(mode=0o700)
    authority = admission_authority(root)
    authority.register("profile", (selector, run_path))
    register_pending(root, "op", ("profile",), control, (selector,))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    sources = (("config", selector), ("agents.run", run_path))
    guard = RecoveryAdmissionGuard(
        "distinct_activation_paths_test",
        error=_AdmissionRefused,
        sources=lambda _service: sources,
    )
    allowed_code = activation.ActivationStore.allowed.__code__
    observation_code = bootstrap._control_observation.__wrapped__.__code__
    bodies, owners, completed = [], [], []
    try:
        store_root = bind(
            (root, control, selector, authority), owners=("config", "agents.run")
        )
        finish_pending(root)
        store = activation.ActivationStore(store_root)
        store.approve("g", "config")
        store.approve("g", "agents.run")

        with guard.execution():
            parent = guard.context.get()
            config_lease = parent.leases[selector]
            run_lease = parent.leases[run_path]
            assert config_lease is not run_lease
            assert config_lease.execution_context(selector) == (root, ("profile",))
            assert run_lease.execution_context(run_path) == (root, ("profile",))

            def observe(frame, event, result):
                if (
                    frame.f_code is allowed_code
                    and event == "return"
                    and result is True
                ):
                    owners.append(frame.f_locals["owner"])
                elif (
                    frame.f_code is observation_code
                    and event == "return"
                    and result is None
                    and owners == ["config", "agents.run"]
                    and not completed
                ):
                    completed.append(True)
                    if retire_first:
                        config_lease.close()

            expected = (
                pytest.raises(_AdmissionRefused) if retire_first else nullcontext()
            )
            with native_observation_counts() as counts, _original_events(observe):
                with expected:
                    with guard.execution():
                        if retire_first:
                            pytest.fail("first retired lease entered the guarded body")
                        current = guard.context.get()
                        assert current.sources == sources
                        assert current.parent is parent
                        assert current.leases[selector] is config_lease
                        assert current.leases[run_path] is run_lease
                        assert config_lease.execution_context(selector) == (
                            root,
                            ("profile",),
                        )
                        assert run_lease.execution_context(run_path) == (
                            root,
                            ("profile",),
                        )
                        bodies.append("accepted")
            assert guard.context.get() is parent
            if retire_first:
                assert config_lease not in storage._live_leases
            else:
                assert not current.live
                assert config_lease.execution_context(selector) == (root, ("profile",))
            assert run_lease.execution_context(run_path) == (root, ("profile",))
        assert config_lease not in storage._live_leases
        assert run_lease not in storage._live_leases
        assert guard.context.get() is None
        assert not parent.live
        assert owners == ["config", "agents.run"]
        assert completed == [True], "original final observation was not reached"
        assert bodies == ([] if retire_first else ["accepted"])
        assert counts["records"] == counts["registry"] == 1, dict(counts)
        request.node.user_properties.extend(
            ("distinct_files_" + name, counts[name])
            for name in ("records", "registry", "native_opens")
            if name in counts
        )
    finally:
        startup = storage._startups.pop((os.getpid(), str(root)), None)
        if startup is not None:
            startup.close()


@pytest.mark.parametrize("callback", ["admit_source", "finalize", "resolve"])
def test_custom_guard_callbacks_keep_ordinary_per_source_route(
    activation_case, callback
):
    root, _control, selector, _store = activation_case
    guard = _guard(selector)
    sources = (("config", selector), ("agents.run", selector))
    allowed_code = activation.ActivationStore.allowed.__code__
    events, bodies = [], []

    def observe(frame, event, _result):
        if frame.f_code is allowed_code and event == "call":
            events.append(("owner", frame.f_locals["owner"]))

    with guard.execution():
        parent = guard.context.get()
        lease = parent.leases[selector]

        def admit_source(stack, owner, path, retained):
            assert stack is not None
            assert path == selector
            assert retained is lease
            assert guard.context.get() is parent
            events.append(("admit_source", owner))
            return False

        def finalize(observed, leases):
            assert observed == sources
            assert leases == {selector: lease}
            assert guard.context.get() is parent
            events.append(("finalize", None))
            return True

        def resolve(service):
            assert service is None
            assert guard.context.get() is parent
            events.append(("resolve", None))
            return sources

        kwargs = {}
        if callback == "admit_source":
            guard.admit_source = admit_source
        elif callback == "finalize":
            guard.finalize = finalize
        else:
            kwargs["resolve"] = resolve

        with native_observation_counts() as counts, _original_events(observe):
            with guard.execution(**kwargs):
                current = guard.context.get()
                assert current.parent is parent
                assert current.sources == sources
                assert current.leases[selector] is lease
                assert lease.execution_context(selector) == (root, ("profile",))
                bodies.append("accepted")
        assert guard.context.get() is parent
        assert not current.live
        assert lease.execution_context(selector) == (root, ("profile",))

    expected = {
        "admit_source": [
            ("admit_source", "config"),
            ("owner", "config"),
            ("admit_source", "agents.run"),
            ("owner", "agents.run"),
        ],
        "finalize": [
            ("owner", "config"),
            ("owner", "agents.run"),
            ("finalize", None),
        ],
        "resolve": [
            ("resolve", None),
            ("owner", "config"),
            ("owner", "agents.run"),
        ],
    }
    assert events == expected[callback]
    assert bodies == ["accepted"]
    assert counts["records"] == counts["registry"] == 2, dict(counts)


def test_replaced_defining_execution_body_keeps_ordinary_route(
    activation_case, monkeypatch
):
    root, _control, selector, _store = activation_case
    guard = _guard(selector)
    original_scope = activation.execution_scope
    original_body = original_scope.__wrapped__
    calls, bodies = [], []

    # The original generator has no closure. Its replacement resolves this
    # temporary observation list in its unchanged defining-module globals.
    def replacement(owners, path=None, *, retained=None):
        globals()["_activation_guard_test_calls"].append((owners, path, retained))
        yield False

    with guard.execution():
        parent = guard.context.get()
        lease = parent.leases[selector]
        assert admission_runtime.execution_scope is original_scope
        with monkeypatch.context() as mutation:
            mutation.setattr(
                activation, "_activation_guard_test_calls", calls, raising=False
            )
            mutation.setattr(original_body, "__code__", replacement.__code__)
            with native_observation_counts() as counts, pytest.raises(
                _AdmissionRefused
            ):
                with guard.execution():
                    bodies.append("must not execute")
            assert activation.execution_scope is original_scope
            assert admission_runtime.execution_scope is original_scope
            assert original_scope.__wrapped__ is original_body
        assert guard.context.get() is parent
        assert lease.execution_context(selector) == (root, ("profile",))
    assert calls == [(("config",), selector, lease)]
    assert bodies == []
    assert counts["records"] == counts["registry"] == 0, dict(counts)


def test_mixed_fallback_lease_rechecked_after_shared_observation(tmp_path, monkeypatch):
    import os

    from Tests.Backup_Recovery.test_activation_binding import bind, finish_pending
    from tldw_chatbook.Backup_Recovery.control_records import (
        admission_authority,
        register_pending,
    )

    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n")
    selector.chmod(0o600)
    run_path = tmp_path / "runs.json"
    run_path.write_text("{}\n")
    run_path.chmod(0o600)
    root, control = tmp_path / "bootstrap", tmp_path / "control"
    control.mkdir(mode=0o700)
    authority = admission_authority(root)
    authority.register("profile", (selector, run_path))
    register_pending(root, "op", ("profile",), control, (selector,))
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    guard = RecoveryAdmissionGuard(
        "mixed_activation_paths_test",
        error=_AdmissionRefused,
        sources=lambda _service: (("config", selector), ("agents.run", run_path)),
        owners=lambda owner: () if owner == "agents.run" else (owner,),
    )
    allowed_code = activation.ActivationStore.allowed.__code__
    scope_code = activation.execution_scope.__wrapped__.__code__
    observation_code = bootstrap._control_observation.__wrapped__.__code__
    approved, fallback, completed, bodies, refused = [], [], [], [], []
    try:
        store_root = bind(
            (root, control, selector, authority), owners=("config", "agents.run")
        )
        finish_pending(root)
        store = activation.ActivationStore(store_root)
        store.approve("g", "config")
        store.approve("g", "agents.run")

        with guard.execution():
            parent = guard.context.get()
            config_lease = parent.leases[selector]
            run_lease = parent.leases[run_path]
            assert config_lease is not run_lease
            assert config_lease.execution_context(selector) == (root, ("profile",))
            assert run_lease.execution_context(run_path) == (root, ("profile",))

            def observe(frame, event, result):
                if (
                    frame.f_code is allowed_code
                    and event == "return"
                    and result is True
                ):
                    approved.append(frame.f_locals["owner"])
                elif (
                    frame.f_code is scope_code
                    and event == "return"
                    and result is True
                    and frame.f_locals["owners"] == ()
                    and frame.f_locals["path"] == run_path
                    and frame.f_locals["retained"] is run_lease
                ):
                    fallback.append(True)
                elif (
                    frame.f_code is observation_code
                    and event == "return"
                    and result is None
                    and approved == ["config"]
                    and fallback == [True]
                    and not completed
                ):
                    completed.append(True)
                    run_lease.close()

            with _original_events(observe):
                try:
                    with guard.execution():
                        bodies.append("must not execute")
                except _AdmissionRefused:
                    refused.append(True)
            assert guard.context.get() is parent
            assert approved == ["config"]
            assert fallback == [True], "original empty-owner route was not admitted"
            assert completed == [True], "shared observation completion was not reached"
            assert run_lease not in storage._live_leases
            assert config_lease.execution_context(selector) == (root, ("profile",))
        assert config_lease not in storage._live_leases
        assert run_lease not in storage._live_leases
        assert guard.context.get() is None
        assert not parent.live
        assert bodies == [], "retired fallback lease entered the guarded body"
        assert refused == [True]
    finally:
        startup = storage._startups.pop((os.getpid(), str(root)), None)
        if startup is not None:
            startup.close()
