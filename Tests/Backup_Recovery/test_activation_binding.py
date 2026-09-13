"""Paired local witnesses survive loss independently; no archive grants authority."""

import json
import os
import subprocess
import sys

import pytest

from tldw_chatbook.Backup_Recovery.activation import (
    ActivationStore,
    activation_permission,
    bind_activation,
)
from tldw_chatbook.Backup_Recovery.bootstrap import _key, startup_permission
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
    register_pending,
)


def pending(tmp_path, operation="op"):
    selector = tmp_path / "config.toml"
    selector.write_text("[general]\n")
    selector.chmod(0o600)
    bootstrap, control = tmp_path / "bootstrap", tmp_path / "control"
    control.mkdir(mode=0o700)
    authority = admission_authority(bootstrap)
    authority.register("profile", (selector,))
    register_pending(bootstrap, operation, ("profile",), control, (selector,))
    return bootstrap, control, selector, authority


def bind(fixture, generation="g", owners=("sync", "schedules"), operation="op"):
    bootstrap, _control, selector, authority = fixture
    with authority.maintenance(("profile",), 2) as session:
        return bind_activation(
            bootstrap, operation, selector, generation, owners, session=session
        )


def finish_pending(bootstrap, operation="op"):
    # Test-only simulation of a later qualified executor completing its fence.
    (bootstrap / ("pending-" + _key(operation) + ".json")).unlink()


def allowed(fixture, owner="sync", **kwargs):
    bootstrap, _control, selector, _authority = fixture
    return activation_permission(
        owner, config_selector=selector, bootstrap_root=bootstrap, **kwargs
    )


def test_fixed_pair_requires_each_owner_after_reopen_and_fingerprint_drift(tmp_path):
    fixture = pending(tmp_path)
    bootstrap, control, selector, _ = fixture
    root = bind(fixture)
    assert root == control / "activation"
    assert startup_permission(selector, bootstrap)[0] is False
    finish_pending(bootstrap)
    ActivationStore(root).approve("g", "sync")
    assert allowed(fixture)
    assert not allowed(fixture, "schedules")
    selector.write_text("[general]\nuser_name='Changed preferences'\n")
    assert not allowed(fixture, "schedules")
    assert allowed(fixture)


def test_ordinary_absence_and_intact_disjoint_scope_never_create_records(tmp_path):
    bootstrap = tmp_path / "absent"
    assert activation_permission(
        "sync", config_selector=tmp_path / "ordinary", bootstrap_root=bootstrap
    )
    assert not bootstrap.exists()
    fixture = pending(tmp_path)
    bind(fixture)
    finish_pending(fixture[0])
    before = {p: p.read_bytes() for p in fixture[0].glob("*.json")}
    assert activation_permission(
        "sync", config_selector=tmp_path / "disjoint", bootstrap_root=fixture[0]
    )
    assert before == {p: p.read_bytes() for p in fixture[0].glob("*.json")}


@pytest.mark.parametrize("record", ["profile", "activation"])
@pytest.mark.parametrize(
    "damage", ["missing", "corrupt", "generation", "owners", "link"]
)
def test_either_surviving_record_prevents_missing_counterpart_bypass(
    tmp_path, record, damage
):
    fixture = pending(tmp_path)
    root = bind(fixture)
    ActivationStore(root).approve("g", "sync")
    finish_pending(fixture[0])
    path = fixture[0] / (record + "-" + _key(str(fixture[2])) + ".json")
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"{")
    elif damage == "link":
        os.link(path, tmp_path / "hardlink")
    else:
        data = json.loads(path.read_bytes())
        data["activation"][damage] = "other" if damage == "generation" else ["sync"]
        path.write_text(json.dumps(data))
    assert not allowed(fixture)


@pytest.mark.parametrize("record", ["required.json", "approved"])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "generation"])
def test_requirements_and_approvals_stay_bound_and_unrepaired(tmp_path, record, damage):
    fixture = pending(tmp_path)
    root = bind(fixture)
    ActivationStore(root).approve("g", "sync")
    finish_pending(fixture[0])
    path = next(
        root.rglob("required.json" if record == "required.json" else "approved-*.json")
    )
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"{")
    else:
        data = json.loads(path.read_bytes())
        data["generation"] = "other"
        path.write_text(json.dumps(data))
    assert not allowed(fixture)
    assert (not path.exists()) if damage == "missing" else path.exists()


def test_new_selector_must_check_actual_shared_admitted_namespaces(tmp_path):
    fixture = pending(tmp_path)
    bind(fixture)
    finish_pending(fixture[0])
    assert not activation_permission(
        "sync",
        config_selector=tmp_path / "new-selector",
        bootstrap_root=fixture[0],
        namespaces=("profile",),
    )
    assert not allowed(fixture, namespaces=("unknown",))


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "fake",
        "retired",
        "other-operation",
        "other-selector",
        "other-authority",
    ],
)
def test_binding_requires_actual_matching_native_session_before_any_write(
    tmp_path, case
):
    fixture = pending(tmp_path)
    bootstrap, control, selector, authority = fixture
    with authority.maintenance(("profile",), 2) as retired:
        pass
    with authority.maintenance(("profile",), 2) as session:
        operation, selected = "op", selector
        supplied = session
        if case == "missing":
            supplied = None
        elif case == "fake":
            supplied = object()
        elif case == "retired":
            supplied = retired
        elif case == "other-operation":
            operation = "other"
        elif case == "other-selector":
            selected = tmp_path / "unrelated"
        elif case == "other-authority":
            bootstrap = tmp_path / "wrong-bootstrap"
        with pytest.raises((ValueError, RuntimeError)):
            bind_activation(
                bootstrap, operation, selected, "g", ("sync",), session=supplied
            )
    assert not (control / "activation").exists()
    assert not list(fixture[0].glob("profile-*.json"))


def test_generation_change_never_inherits_approval_and_bind_profile_cannot_erase_witness(
    tmp_path,
):
    fixture = pending(tmp_path)
    root = bind(fixture, "g1")
    ActivationStore(root).approve("g1", "sync")
    finish_pending(fixture[0])
    with pytest.raises(FileExistsError):
        bind_profile(fixture[0], fixture[2], ("profile",), fixture[0] / "admission")
    register_pending(fixture[0], "next", ("profile",), fixture[1], (fixture[2],))
    bind(fixture, "g2", operation="next")
    finish_pending(fixture[0], "next")
    assert not allowed(fixture)
    assert ActivationStore(root).allowed("g1", "sync")


def test_complete_identical_retry_preserves_pair_and_approval(tmp_path):
    fixture = pending(tmp_path)
    root = bind(fixture)
    ActivationStore(root).approve("g", "sync")
    before = {p: p.read_bytes() for p in fixture[0].glob("*.json")}
    assert bind(fixture) == root
    assert before == {p: p.read_bytes() for p in fixture[0].glob("*.json")}
    finish_pending(fixture[0])
    assert allowed(fixture)


def test_child_exit_between_pair_writes_keeps_pending_and_denies(tmp_path):
    script = r"""
import os, sys
from pathlib import Path
from Tests.Backup_Recovery.test_activation_binding import pending, bind
from tldw_chatbook.Backup_Recovery import control_records
fixture = pending(Path(sys.argv[1]))
original = control_records.publish_new
def publish(source, destination, **kwargs):
    result = original(source, destination, **kwargs)
    if destination.name.startswith('activation-'):
        os._exit(73)
    return result
control_records.publish_new = publish
bind(fixture)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert child.returncode == 73, child.stderr.decode()[-4000:]
    bootstrap, selector = tmp_path / "bootstrap", tmp_path / "config.toml"
    assert startup_permission(selector, bootstrap)[0] is False
    assert not activation_permission(
        "sync", config_selector=selector, bootstrap_root=bootstrap
    )
    assert list(bootstrap.glob("pending-*.json"))


def test_existing_profile_mapping_is_preserved_during_binding(tmp_path):
    fixture = pending(tmp_path)
    finish_pending(fixture[0])
    bind_profile(fixture[0], fixture[2], ("profile",), fixture[0] / "admission")
    path = next(fixture[0].glob("profile-*.json"))
    previous = json.loads(path.read_bytes())
    register_pending(fixture[0], "op", ("profile",), fixture[1], (fixture[2],))
    fixture[2].write_text("[changed]\n")
    bind(fixture)
    current = json.loads(path.read_bytes())
    assert {k: current[k] for k in ("selector", "namespaces", "roots")} == {
        k: previous[k] for k in ("selector", "namespaces", "roots")
    }
    assert current["fingerprint"] != previous["fingerprint"]


def test_reused_generation_cannot_inherit_earlier_operation_approval(tmp_path):
    fixture = pending(tmp_path)
    root = bind(fixture)
    ActivationStore(root).approve("g", "sync")
    finish_pending(fixture[0])
    register_pending(fixture[0], "next", ("profile",), fixture[1], (fixture[2],))
    with pytest.raises(ValueError):
        bind(fixture, operation="next")
    assert not allowed(fixture)


def test_unheld_pending_namespace_is_refused_before_requirement_write(tmp_path):
    fixture = pending(tmp_path)
    extra = tmp_path / "extra"
    extra.write_text("extra")
    fixture[3].register("extra", (extra,))
    finish_pending(fixture[0])
    register_pending(fixture[0], "op", ("profile", "extra"), fixture[1], (fixture[2],))
    with pytest.raises((ValueError, RuntimeError)):
        bind(fixture)
    assert not (fixture[1] / "activation").exists()


def test_custom_control_root_must_be_private_and_disjoint_before_write(tmp_path):
    fixture = pending(tmp_path)
    fixture[1].chmod(0o755)
    with pytest.raises((ValueError, RuntimeError)):
        bind(fixture)
    assert not (fixture[1] / "activation").exists()


def test_missing_requirement_does_not_prevent_safe_local_startup(tmp_path):
    fixture = pending(tmp_path)
    root = bind(fixture)
    finish_pending(fixture[0])
    next(root.rglob("required.json")).unlink()
    assert startup_permission(fixture[2], fixture[0]) == (True, "startup_allowed")
    assert not allowed(fixture)


@pytest.mark.parametrize("record", ["activation", "profile"])
def test_activation_only_corruption_preserves_mapping_for_local_inspection(
    tmp_path, record
):
    from tldw_chatbook.Backup_Recovery.bootstrap import _binding, _records, _registry

    fixture = pending(tmp_path)
    bind(fixture)
    finish_pending(fixture[0])
    path = fixture[0] / (record + "-" + _key(str(fixture[2])) + ".json")
    if record == "activation":
        path.write_bytes(b"{")
    else:
        data = json.loads(path.read_bytes())
        data["activation"] = {"generation": None}
        path.write_text(json.dumps(data))
    assert startup_permission(fixture[2], fixture[0]) == (True, "startup_allowed")
    assert (
        _binding(fixture[2], _records(fixture[0])[1], _registry(fixture[0])) is not None
    )
    assert not allowed(fixture)


def test_captured_profile_identity_cannot_change_while_requirements_publish(
    tmp_path, monkeypatch
):
    fixture = pending(tmp_path)
    finish_pending(fixture[0])
    bind_profile(fixture[0], fixture[2], ("profile",), fixture[0] / "admission")
    register_pending(fixture[0], "op", ("profile",), fixture[1], (fixture[2],))
    path = next(fixture[0].glob("profile-*.json"))
    original = ActivationStore.require

    def require(store, generation, owners):
        original(store, generation, owners)
        temporary = tmp_path / "replacement"
        temporary.write_bytes(path.read_bytes())
        temporary.chmod(0o600)
        temporary.replace(path)

    monkeypatch.setattr(ActivationStore, "require", require)
    with pytest.raises(ValueError, match="activation_record_changed"):
        bind(fixture)
    assert not list(fixture[0].glob("activation-*.json"))
    assert startup_permission(fixture[2], fixture[0])[0] is False


def test_held_unbound_guard_is_not_persisted_as_affected_profile_scope(tmp_path):
    fixture = pending(tmp_path)
    with fixture[3].maintenance(("profile", "bootstrap.unbound"), 2) as session:
        bind_activation(fixture[0], "op", fixture[2], "g", ("sync",), session=session)
    finish_pending(fixture[0])
    assert not allowed(fixture)
    assert activation_permission(
        "sync",
        config_selector=tmp_path / "ordinary",
        bootstrap_root=fixture[0],
        namespaces=("bootstrap.unbound",),
    )
