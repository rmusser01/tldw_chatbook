"""Fresh finite witness reads retain native recovery refusals and completion fences."""

import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_activation_binding import bind, finish_pending, pending
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import generation_witnesses as generations
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import (
    ActivationStore,
    _private,
    _source_scope_admitted,
)


@pytest.fixture
def witness_case(tmp_path, monkeypatch):
    case = pending(tmp_path)
    root, control, selector, _authority = case
    monkeypatch.setattr(bootstrap, "default_bootstrap_root", lambda: root)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    bind(case, owners=("mcp.permissions",))
    finish_pending(root)
    yield root, control, selector
    startup = storage._startups.pop((os.getpid(), str(root)), None)
    if startup is not None:
        startup.close()


def legacy_witnesses(path, lease):
    """Frozen pre-fix reader oracle, using the actual public fresh native readers."""
    root, names = lease.execution_context(path)
    if path is not None and not _source_scope_admitted(root, names or (), path):
        raise ValueError("projection_source_scope_unavailable")
    selected = bootstrap.effective_config_path()
    if not bootstrap.startup_permission(selected, root)[0]:
        raise ValueError("projection_generation_unavailable")
    _, profiles, associations = bootstrap._control_records(root)
    registry = bootstrap._registry(root)
    held = set(names or ())
    profile = next((p for p in profiles if p["selector"] == str(selected)), None)
    if profile:
        held.update(profile["namespaces"])
    results, seen = [], {}
    for selector in sorted({p["selector"] for p in profiles + associations}):
        profile = next((p for p in profiles if p["selector"] == selector), None)
        association = next((p for p in associations if p["selector"] == selector), None)
        witness = profile.get("activation") if profile else None
        other = association["activation"] if association else None
        relevant = (
            selector == str(selected)
            or any(
                item and held.intersection(item["namespaces"])
                for item in (witness, other)
            )
            or path is not None
            and any(
                bootstrap._overlap(path, Path(p))
                for p in (profile or {}).get("roots", [])
            )
        )
        if not relevant or witness is None and other is None:
            continue
        if witness is None or witness != other or names is None:
            raise ValueError("projection_generation_unavailable")
        if registry is None or not set(witness["namespaces"]) <= registry.keys():
            raise ValueError("projection_generation_unavailable")
        roots = sorted(
            {p for name in witness["namespaces"] for p in registry[name]["roots"]}
        )
        if roots != profile["roots"]:
            raise ValueError("projection_generation_unavailable")
        for name in witness["namespaces"]:
            if name in seen and not bootstrap._same_activation_generation(
                seen[name], witness
            ):
                raise ValueError("projection_generation_unavailable")
            seen[name] = witness
        store = ActivationStore(Path(witness["store_root"]))
        with _private(store.root), _private(
            store._generation(witness["generation"])
        ) as parent:
            if (
                store._required(parent, witness["generation"]).owners
                != witness["owners"]
            ):
                raise ValueError("projection_generation_unavailable")
        if witness not in results:
            results.append(witness)
    return results


@contextmanager
def native_observation_counts():
    codes = {
        bootstrap._control_records.__code__: "records",
        bootstrap._registry.__code__: "registry",
    }
    if os.name == "nt":
        from tldw_chatbook.Utils.windows_files import _Native

        codes[_Native.open_handle.__code__] = "native_opens"
    counts = Counter()
    previous = sys.getprofile()

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code in codes:
            counts[codes[frame.f_code]] += 1

    sys.setprofile(observe)
    try:
        yield counts
    finally:
        sys.setprofile(previous)


def test_witness_reads_one_fresh_control_observation(witness_case):
    _root, _control, selector = witness_case
    with storage.acquire_storage(selector) as lease:
        with native_observation_counts() as legacy_counts:
            expected = legacy_witnesses(selector, lease)
        with native_observation_counts() as counts:
            actual = generations._witnesses(selector, lease)
    assert actual == expected
    assert actual[0]["generation"] == "g"
    assert counts["records"] == 1, dict(counts)
    assert counts["registry"] == 1, dict(counts)
    if os.name == "nt":
        assert counts["native_opens"] < legacy_counts["native_opens"], (
            dict(counts),
            dict(legacy_counts),
        )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_profile",
        "missing_association",
        "bad_generation",
        "bad_owners",
        "pending",
        "unknown_record",
        "registry_pending",
        "registry_corrupt",
        "registry_proposed",
        "required_owner",
    ],
)
def test_fresh_observation_matches_native_legacy_refusals(witness_case, damage):
    root, control, selector = witness_case
    # Admit first: mutation is a real nonparticipant edit, not an injected guard result.
    with storage.acquire_storage(selector) as lease:
        profile = root / ("profile-" + bootstrap._key(str(selector)) + ".json")
        association = root / ("activation-" + bootstrap._key(str(selector)) + ".json")
        if damage == "missing_profile":
            profile.unlink()
        elif damage == "missing_association":
            association.unlink()
        elif damage in {"bad_generation", "bad_owners"}:
            payload = json.loads(association.read_bytes())
            payload["activation"][
                "generation" if damage == "bad_generation" else "owners"
            ] = "foreign" if damage == "bad_generation" else ["sync"]
            association.write_text(json.dumps(payload))
        elif damage == "pending":
            _insert_pending(root, control, selector)
        elif damage == "unknown_record":
            (root / "unknown.json").write_text('{"version":1}')
        elif damage == "registry_pending":
            (root / "admission" / "registry.pending.json").write_text('{"version":1}')
        elif damage == "registry_corrupt":
            (root / "admission" / "registry.json").write_text("{")
        elif damage == "registry_proposed":
            registry_path = root / "admission" / "registry.json"
            payload = json.loads(registry_path.read_bytes())
            payload["entries"]["profile"]["proposed"] = [str(selector)]
            registry_path.write_text(json.dumps(payload))
        else:
            required = next((control / "activation").rglob("required.json"))
            payload = json.loads(required.read_bytes())
            payload["owners"] = ["sync"]
            required.write_text(json.dumps(payload))
        with pytest.raises((ValueError, OSError, RuntimeError)):
            legacy_witnesses(selector, lease)
        with pytest.raises((ValueError, OSError, RuntimeError)):
            generations._witnesses(selector, lease)


def _insert_pending(root, control, selector):
    payload = dict(
        version=1,
        operation_id="during-read",
        namespaces=["profile"],
        control_root=str(control),
        selectors=[str(selector)],
    )
    target = root / ("pending-" + bootstrap._key("during-read") + ".json")
    target.write_text(json.dumps(payload))
    target.chmod(0o600)


def test_pending_inserted_after_registry_read_never_returns_witness(witness_case):
    root, control, selector = witness_case
    with storage.acquire_storage(selector) as lease:
        previous = sys.getprofile()
        inserted = []
        registry_code = bootstrap._registry.__code__

        def mutation_barrier(frame, event, _argument):
            if event == "return" and frame.f_code is registry_code and not inserted:
                _insert_pending(root, control, selector)
                inserted.append(True)

        sys.setprofile(mutation_barrier)
        try:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                generations._witnesses(selector, lease)
        finally:
            sys.setprofile(previous)
        assert inserted


def test_successive_calls_observe_new_pending_evidence(witness_case):
    root, control, selector = witness_case
    with storage.acquire_storage(selector) as lease:
        assert generations._witnesses(selector, lease)[0]["generation"] == "g"
        _insert_pending(root, control, selector)
        with pytest.raises((ValueError, OSError, RuntimeError)):
            generations._witnesses(selector, lease)


@pytest.mark.parametrize(
    "damage",
    [
        "profile_content",
        "profile_replacement",
        "association_content",
        "registry_content",
        "registry_replacement",
        "marker_removal",
        "registry_pending",
        "unknown_record",
    ],
)
def test_named_control_changes_after_registry_read_refuse(witness_case, damage):
    root, _control, selector = witness_case
    profile = root / ("profile-" + bootstrap._key(str(selector)) + ".json")
    association = root / ("activation-" + bootstrap._key(str(selector)) + ".json")
    registry = root / "admission" / "registry.json"
    with storage.acquire_storage(selector) as lease:
        previous = sys.getprofile()
        changed = []
        completed = []
        registry_code = bootstrap._registry.__code__

        def mutation_barrier(frame, event, _argument):
            if event != "return" or frame.f_code is not registry_code or changed:
                return
            changed.append(True)
            if damage in {"profile_content", "association_content"}:
                target = profile if damage == "profile_content" else association
                payload = json.loads(target.read_bytes())
                payload["activation"]["generation"] = "foreign"
                target.write_text(json.dumps(payload))
            elif damage == "registry_content":
                payload = json.loads(registry.read_bytes())
                payload["entries"]["profile"]["proposed"] = [str(selector)]
                registry.write_text(json.dumps(payload))
            elif damage in {"profile_replacement", "registry_replacement"}:
                target = profile if damage == "profile_replacement" else registry
                replacement = target.with_suffix(".replacement")
                replacement.write_bytes(target.read_bytes())
                replacement.chmod(0o600)
                replacement.replace(target)
            elif damage == "marker_removal":
                (root / "unbound-owner").unlink()
            elif damage == "registry_pending":
                (root / "admission" / "registry.pending.json").write_text(
                    '{"version":1}'
                )
            else:
                (root / "unknown.json").write_text('{"version":1}')
            completed.append(True)

        sys.setprofile(mutation_barrier)
        try:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                generations._witnesses(selector, lease)
        finally:
            sys.setprofile(previous)
        assert changed and completed


def test_empty_witness_results_are_fresh_on_every_call(witness_case):
    root, _control, selector = witness_case
    profile = root / ("profile-" + bootstrap._key(str(selector)) + ".json")
    association = root / ("activation-" + bootstrap._key(str(selector)) + ".json")
    payload = json.loads(profile.read_bytes())
    payload.pop("activation")
    profile.write_text(json.dumps(payload))
    association.unlink()
    with storage.acquire_storage(selector) as lease:
        assert legacy_witnesses(selector, lease) == []
        with native_observation_counts() as counts:
            assert generations._witnesses(selector, lease) == []
            assert generations._witnesses(selector, lease) == []
        assert counts["records"] == 2
        assert counts["registry"] == 2
        (root / "unknown.json").write_text('{"version":1}')
        with pytest.raises((ValueError, OSError, RuntimeError)):
            generations._witnesses(selector, lease)


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX native parent mode policy; Windows uses ACLs"
)
def test_parent_becoming_shared_during_control_observation_refuses(witness_case):
    root, _control, selector = witness_case
    with storage.acquire_storage(selector) as lease:
        previous = sys.getprofile()
        changed = []
        registry_code = bootstrap._registry.__code__
        parent = root.parent
        original_mode = parent.stat().st_mode & 0o777

        def mutation_barrier(frame, event, _argument):
            if event == "return" and frame.f_code is registry_code and not changed:
                parent.chmod(0o777)
                changed.append(True)

        sys.setprofile(mutation_barrier)
        try:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                generations._witnesses(selector, lease)
        finally:
            sys.setprofile(previous)
            parent.chmod(original_mode)
        assert changed


@pytest.mark.skipif(
    os.name == "nt",
    reason="Real POSIX rename with pinned directories; Windows uses its tree fence",
)
def test_named_ancestor_replacement_after_completion_pins_refuses(witness_case):
    root, _control, selector = witness_case
    original_parent = root.parent
    retained_parent = original_parent.with_name(original_parent.name + "-retained")
    authority = root / "admission"
    pin_code = bootstrap.pinned_directory.__wrapped__.__code__
    authority_pins = []
    changed = []
    with storage.acquire_storage(selector) as lease:
        previous = sys.getprofile()

        def mutation_barrier(frame, event, result):
            if (
                event == "return"
                and frame.f_code is pin_code
                and result is not None
                and frame.f_locals.get("root") == authority
            ):
                authority_pins.append(True)
                if len(authority_pins) == 2:
                    # Move only the containing ancestor. The held descendant
                    # inodes and their timestamps stay unchanged.
                    original_parent.rename(retained_parent)
                    original_parent.mkdir(mode=0o700)
                    changed.append(True)

        sys.setprofile(mutation_barrier)
        try:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                generations._witnesses(selector, lease)
        finally:
            sys.setprofile(previous)
            if retained_parent.exists():
                original_parent.rmdir()
                retained_parent.rename(original_parent)
        assert changed


@pytest.mark.skipif(
    os.name != "nt", reason="Actual native Windows DACL policy mutation"
)
def test_windows_ancestor_becoming_shared_at_final_snapshot_refuses(witness_case):
    from Tests.Utils.test_windows_native_admission import _replace_security
    from tldw_chatbook.Utils.windows_files import _native

    root, _control, selector = witness_case
    parent = root.parent
    user = _native().user_sid
    private = f"D:P(A;;FA;;;{user})(A;;FA;;;SY)(A;;FA;;;BA)"
    snapshot_code = bootstrap.os.stat_many_for_admission.__func__.__code__
    changed = []
    with storage.acquire_storage(selector) as lease:
        previous = sys.getprofile()

        def mutation_barrier(frame, event, _argument):
            if event == "call" and frame.f_code is snapshot_code and not changed:
                _replace_security(parent, private + "(A;;FA;;;WD)")
                assert storage.os.stat(parent).st_mode & 0o022
                changed.append(True)

        sys.setprofile(mutation_barrier)
        try:
            with pytest.raises((ValueError, OSError, RuntimeError)):
                generations._witnesses(selector, lease)
        finally:
            sys.setprofile(previous)
            _replace_security(parent, private)
        assert changed
