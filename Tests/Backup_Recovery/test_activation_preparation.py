"""Finite activation preparation keeps original native owner decisions."""

import json
import os
import sys
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery.test_activation_binding import bind, finish_pending, pending
from Tests.Backup_Recovery.test_generation_witness_observation import (
    _insert_pending,
    native_observation_counts,
)
from tldw_chatbook.Backup_Recovery import activation, bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage


@pytest.fixture
def activation_case(tmp_path, monkeypatch):
    case = pending(tmp_path)
    root, control, selector, _authority = case
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    store_root = bind(case, owners=("config", "agents.run"))
    finish_pending(root)
    store = activation.ActivationStore(store_root)
    store.approve("g", "config")
    store.approve("g", "agents.run")
    try:
        yield root, control, selector, store
    finally:
        startup = storage._startups.pop((os.getpid(), str(root)), None)
        if startup is not None:
            startup.close()


@pytest.mark.parametrize("owners", [("config",), ("config", "agents.run")])
def test_execution_scope_prepares_one_original_observation(activation_case, owners):
    _root, _control, selector, _store = activation_case
    body = []
    with storage.acquire_storage(selector) as lease:
        with native_observation_counts() as counts:
            with activation.execution_scope(
                owners, selector, retained=lease
            ) as allowed:
                if allowed:
                    body.append("accepted")
        assert lease.execution_context(selector)[1] == ("profile",)

    assert body == ["accepted"]
    assert counts["records"] == 1, dict(counts)
    assert counts["registry"] == 1, dict(counts)


@contextmanager
def _original_owner_decisions(mutate=None):
    """Inject only fixture changes at an original owner's successful return."""
    startup_code = bootstrap._startup_permission_from_records.__code__
    allowed_code = activation.ActivationStore.allowed.__code__
    decisions, changed = [], []
    previous = sys.getprofile()

    def observe(frame, event, result):
        if frame.f_code is startup_code and event == "call":
            decisions.append(("startup", None))
        elif frame.f_code is allowed_code:
            owner = frame.f_locals["owner"]
            if event == "call":
                decisions.append(("owner", owner))
            elif (
                event == "return"
                and result is True
                and owner == "config"
                and mutate is not None
                and not changed
            ):
                changed.append(True)
                mutate()

    sys.setprofile(observe)
    try:
        yield decisions, changed
    finally:
        sys.setprofile(previous)


def test_each_owner_keeps_its_startup_projection_and_live_approval(activation_case):
    _root, _control, selector, _store = activation_case
    with storage.acquire_storage(selector) as lease:
        with _original_owner_decisions() as (decisions, _):
            with activation.execution_scope(
                ("config", "agents.run"), selector, retained=lease
            ) as allowed:
                assert allowed

    assert decisions == [
        ("startup", None),
        ("owner", "config"),
        ("startup", None),
        ("owner", "agents.run"),
    ]


@pytest.mark.parametrize("owners", [("config",), ("config", "agents.run")])
@pytest.mark.parametrize("damage", ["pending", "association_generation", "registry"])
def test_control_drift_during_first_owner_refuses_before_body(
    activation_case, owners, damage
):
    root, control, selector, _store = activation_case

    def mutate():
        if damage == "pending":
            _insert_pending(root, control, selector)
        elif damage == "association_generation":
            path = root / ("activation-" + bootstrap._key(str(selector)) + ".json")
            payload = json.loads(path.read_bytes())
            payload["activation"]["generation"] = "foreign"
            path.write_text(json.dumps(payload))
        else:
            path = root / "admission" / "registry.json"
            payload = json.loads(path.read_bytes())
            payload["entries"]["profile"]["proposed"] = [str(selector)]
            path.write_text(json.dumps(payload))

    body = []
    with storage.acquire_storage(selector) as lease:
        with _original_owner_decisions(mutate) as (_decisions, changed):
            with activation.execution_scope(
                owners, selector, retained=lease
            ) as allowed:
                if allowed:
                    body.append("must not execute")
                assert not allowed
        assert changed == [True]
    assert body == []


def test_second_owner_on_same_path_observes_removed_approval(activation_case):
    _root, _control, selector, store = activation_case
    approval = store._generation("g") / (
        "approved-" + activation._key("agents.run") + ".json"
    )
    assert approval.is_file()
    with storage.acquire_storage(selector) as lease:
        with _original_owner_decisions(approval.unlink) as (decisions, changed):
            with activation.execution_scope(
                ("config", "agents.run"), selector, retained=lease
            ) as allowed:
                assert not allowed
        assert lease.execution_context(selector)[1] == ("profile",)
    assert changed == [True]
    assert [owner for kind, owner in decisions if kind == "owner"] == [
        "config",
        "agents.run",
    ]
    assert not approval.exists()


def test_later_scope_observes_new_pending_evidence(activation_case):
    root, control, selector, _store = activation_case
    with storage.acquire_storage(selector) as lease:
        with native_observation_counts() as first:
            with activation.execution_scope(
                ("config",), selector, retained=lease
            ) as allowed:
                assert allowed
        _insert_pending(root, control, selector)
        with native_observation_counts() as second:
            with activation.execution_scope(
                ("config",), selector, retained=lease
            ) as allowed:
                assert not allowed
    assert first["records"] == first["registry"] == 1, dict(first)
    assert second["records"] == second["registry"] == 1, dict(second)


@pytest.mark.parametrize("change", ["path", "selector", "retired"])
def test_retained_native_selection_is_checked_before_preparation(
    activation_case, monkeypatch, change
):
    _root, _control, selector, _store = activation_case
    other = selector.with_name("different.toml")
    other.write_bytes(selector.read_bytes())
    other.chmod(0o600)
    with storage.acquire_storage(selector) as lease:
        path = selector
        if change == "path":
            path = other
        elif change == "selector":
            monkeypatch.setenv("TLDW_CONFIG_PATH", str(other))
        else:
            lease.close()
        with native_observation_counts() as counts:
            with activation.execution_scope(
                ("config",), path, retained=lease
            ) as allowed:
                assert not allowed
    assert counts["records"] == counts["registry"] == 0, dict(counts)


@pytest.mark.parametrize("callback", ["permission", "source", "startup"])
def test_custom_callbacks_keep_original_order_and_refusal(
    activation_case, monkeypatch, callback
):
    root, _control, selector, _store = activation_case
    called = []
    with storage.acquire_storage(selector) as lease:
        if callback == "permission":

            def permission(
                owner, *, config_selector, bootstrap_root, namespaces, ordinary_only
            ):
                called.append(
                    (owner, config_selector, bootstrap_root, namespaces, ordinary_only)
                )
                return owner == "config"

            monkeypatch.setattr(activation, "activation_permission", permission)
        elif callback == "source":

            def source(actual_root, names, path):
                called.append((actual_root, names, path))
                return False

            monkeypatch.setattr(activation, "_source_scope_admitted", source)
        else:

            def startup(actual_selector, actual_root):
                called.append((actual_selector, actual_root))
                return len(called) == 1, "custom startup decision"

            monkeypatch.setattr(bootstrap, "startup_permission", startup)

        # A fallback must not eagerly enter the shared read before these callbacks.
        observed = []
        code = bootstrap._control_observation.__wrapped__.__code__
        previous = sys.getprofile()

        def observe(frame, event, _result):
            if event == "call" and frame.f_code is code:
                observed.append(True)

        sys.setprofile(observe)
        try:
            with activation.execution_scope(
                ("config", "agents.run"), selector, retained=lease
            ) as allowed:
                assert not allowed
        finally:
            sys.setprofile(previous)

    assert observed == []
    if callback == "permission":
        assert called == [
            (owner, selector, root, ("profile",), False)
            for owner in ("config", "agents.run")
        ]
    elif callback == "source":
        assert called == [(root, ("profile",), selector)]
    else:
        assert called == [(selector, root), (selector, root)]


@pytest.mark.parametrize("reader", ["records", "control_records", "registry"])
def test_legacy_reader_signatures_keep_original_route(
    activation_case, monkeypatch, reader
):
    root, _control, selector, _store = activation_case
    calls = []
    with storage.acquire_storage(selector) as lease:
        if reader == "control_records":
            original = bootstrap._control_records

            def read(actual_root, *, activation=True):
                calls.append((actual_root, activation))
                return original(actual_root, activation=activation)
        else:
            original = getattr(bootstrap, "_" + reader)

            def read(actual_root):
                calls.append(actual_root)
                return original(actual_root)

        monkeypatch.setattr(bootstrap, "_" + reader, read)
        with activation.execution_scope(
            ("config", "agents.run"), selector, retained=lease
        ) as allowed:
            assert allowed

    if reader == "control_records":
        assert calls == [(root, flag) for flag in (True, False, True, False, True)]
    else:
        assert calls == [root] * (2 if reader == "records" else 4)


@pytest.mark.parametrize("has_path", [False, True])
def test_empty_owner_scope_keeps_existing_pending_behavior(activation_case, has_path):
    root, control, selector, _store = activation_case
    path = selector if has_path else None
    with storage.acquire_storage(path) as lease:
        _insert_pending(root, control, selector)
        with native_observation_counts() as counts:
            with activation.execution_scope((), path, retained=lease) as allowed:
                assert allowed
    assert counts["records"] == int(has_path), dict(counts)
    assert counts["registry"] == 0, dict(counts)


def test_public_activation_permission_retains_standalone_reads(activation_case):
    root, _control, selector, _store = activation_case
    with native_observation_counts() as counts:
        assert activation.activation_permission(
            "config",
            config_selector=selector,
            bootstrap_root=root,
            namespaces=("profile",),
        )
    assert counts["records"] == counts["registry"] == 2, dict(counts)


@pytest.mark.parametrize("change", ["selector", "lease", "permission_callback"])
def test_observation_completion_rechecks_selector_lease_and_callbacks(
    activation_case, monkeypatch, change
):
    _root, _control, selector, _store = activation_case
    other = selector.with_name("after-observation.toml")
    other.write_bytes(selector.read_bytes())
    other.chmod(0o600)
    completed, decisions = [], []
    code = bootstrap._control_observation.__wrapped__.__code__

    with storage.acquire_storage(selector) as lease, monkeypatch.context() as mutation:
        previous = sys.getprofile()

        def observe(frame, event, result):
            if (
                frame.f_code is code
                and event == "return"
                and result is None
                and not completed
            ):
                completed.append(change)
                if change == "selector":
                    mutation.setenv("TLDW_CONFIG_PATH", str(other))
                elif change == "lease":
                    lease.close()
                else:
                    mutation.setattr(
                        activation,
                        "activation_permission",
                        lambda *args, **kwargs: True,
                    )

        sys.setprofile(observe)
        try:
            with activation.execution_scope(
                ("config",), selector, retained=lease
            ) as allowed:
                decisions.append(allowed)
        finally:
            sys.setprofile(previous)

    assert completed == [change], "original observation completion was not reached"
    assert decisions == [False]


def test_large_owner_tuple_keeps_ordinary_first_denial_short_circuit(activation_case):
    _root, _control, selector, _store = activation_case
    owners = tuple(f"unapproved-{index}" for index in range(1000))
    allowed_code = activation.ActivationStore.allowed.__code__
    checked = []
    with storage.acquire_storage(selector) as lease:
        with native_observation_counts() as counts:
            previous = sys.getprofile()

            def observe(frame, event, result):
                previous(frame, event, result)
                if frame.f_code is allowed_code and event == "call":
                    checked.append(frame.f_locals["owner"])

            sys.setprofile(observe)
            try:
                with activation.execution_scope(
                    owners, selector, retained=lease
                ) as allowed:
                    assert not allowed
            finally:
                sys.setprofile(previous)

    assert checked == ["unapproved-0"]
    assert counts["records"] == 3, dict(counts)
    assert counts["registry"] == 2, dict(counts)
