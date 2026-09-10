"""Terminal publication uses actual native authority and independently read proof."""

import json
import subprocess
import sys
from contextlib import contextmanager
from threading import Event

import pytest

from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery import publication
from tldw_chatbook.Backup_Recovery.activation import (
    ActivationStore,
    activation_permission,
)
from tldw_chatbook.Backup_Recovery.bootstrap import _key, startup_permission
from tldw_chatbook.Backup_Recovery.control_records import (
    UNBOUND_NAMESPACE,
    admission_authority,
    register_pending,
)
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.staging import stage_restore


@contextmanager
def installed(tmp_path, *, unbound_guard=True):
    def config_with_state(doc):
        doc["owners"][0]["owner_id"] = "config"
        doc["owners"].append(
            {"owner_id": "ui.state", "schema_version": 1, "capabilities": []}
        )
        doc["files"][0].update(
            owner_id="config",
            logical_id="profile:profile:config",
            relative_path="config.toml",
        )
        doc["files"].append(
            {
                **doc["files"][0],
                "owner_id": "ui.state",
                "logical_id": "profile:profile:ui.state",
                "relative_path": "ui_state.toml",
                "payload": "payload/2",
            }
        )
        doc["dependency_groups"][0]["members"] = [
            "profile:profile:config",
            "profile:profile:ui.state",
        ]

    archive = sealed(
        tmp_path, mutate=config_with_state, data=b'[general]\nusers_name="original"\n'
    )
    bootstrap = tmp_path / "bootstrap"
    authority = admission_authority(bootstrap)
    parent = tmp_path / "isolated"
    parent.mkdir(mode=0o700)
    authority.register("profile", (parent,))
    destination = parent / "config"
    plan = plan_restore(
        archive,
        mode="isolated",
        destinations={
            "root": destination,
            "profile:profile:paths.data_dir": parent / "data",
        },
        target=None,
        profile_names={"profile": "recovered"},
    )
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    journal = Journal(control, "finalize-op")
    selector = destination / "config.toml"
    register_pending(
        bootstrap,
        journal.operation_id,
        ("profile",),
        control,
        (selector, parent / "data"),
    )
    candidate = stage_restore(
        archive, plan, tmp_path / "stage", Event(), journal=journal
    )
    names = ("profile", UNBOUND_NAMESPACE) if unbound_guard else ("profile",)
    with authority.maintenance(names, 3) as session:
        journal.prepare_publication(
            candidate,
            plan,
            bootstrap_root=bootstrap,
            namespaces=("profile",),
            selectors=(selector, parent / "data"),
            generation="g",
        )
        publication.publish_candidate(candidate, plan, journal, None)
        yield candidate, plan, journal, session, bootstrap, selector


def finalize(case, **kwargs):
    candidate, plan, journal, session, _, _ = case
    return publication.finalize_candidate(
        candidate, plan, journal, session=kwargs.get("session", session)
    )


def events(journal):
    return [
        json.loads(path.read_bytes())
        for path in sorted(journal.root.glob("[0-9]*.json"))
    ]


def test_native_finalization_commits_then_clears_only_matching_fence(tmp_path):
    with installed(tmp_path) as case:
        candidate, _, journal, session, bootstrap, selector = case
        archive_before = (tmp_path / "fixture.zip").read_bytes()
        assert not startup_permission(selector, bootstrap)[0]
        assert finalize(case) == "g"
        session._check()
        assert [row["event"] for row in events(journal)][-2:] == [
            "activation_recorded",
            "committed",
        ]
        assert startup_permission(selector, bootstrap)[0]
        profile = json.loads(
            (bootstrap / ("profile-" + _key(str(selector)) + ".json")).read_bytes()
        )
        assert profile["namespaces"] == ["profile"]
        assert profile["activation"]["namespaces"] == ["profile"]
        assert not activation_permission(
            "config", config_selector=selector, bootstrap_root=bootstrap
        )
        store = ActivationStore(journal.root.parent / "activation")
        store.approve("g", "config")
        assert activation_permission(
            "config", config_selector=selector, bootstrap_root=bootstrap
        )
        assert not activation_permission(
            "ui.state", config_selector=selector, bootstrap_root=bootstrap
        )
        store.approve("g", "models.artifacts")
        assert activation_permission(
            "models.artifacts", config_selector=selector, bootstrap_root=bootstrap
        )
        assert not store.allowed("g", "tts.voices")
        store.approve("g", "tts.voices")
        assert store.allowed("g", "tts.voices")
        assert finalize(case) == "g"
        assert store.allowed("g", "config")
        assert len([row for row in events(journal) if row["event"] == "committed"]) == 1
        assert (tmp_path / "fixture.zip").read_bytes() == archive_before
        assert not list(candidate.glob("installed-check-*"))


@pytest.mark.parametrize("kind", ["missing", "fake", "retired", "foreign"])
def test_invalid_session_cannot_write_terminal_evidence(tmp_path, kind):
    with installed(tmp_path) as case:
        supplied = None if kind == "missing" else object()
        if kind == "retired":
            supplied = case[3]
            supplied._active = False
        if kind == "foreign":
            supplied = case[3]
            supplied._control = tmp_path / "foreign"
        before = events(case[2])
        with pytest.raises((ValueError, RuntimeError)):
            finalize(case, session=supplied)
        assert events(case[2]) == before
        assert not (case[2].root.parent / "activation").exists()


@pytest.mark.parametrize("event", ["activation_recorded", "committed"])
def test_direct_terminal_labels_do_not_clear_pending(tmp_path, event):
    with installed(tmp_path) as case:
        with pytest.raises(ValueError):
            case[2].record(event, {})
        assert not startup_permission(case[5], case[4])[0]


def test_later_installed_drift_cannot_clear_fence_from_old_validation(tmp_path):
    with installed(tmp_path) as case:
        case[2].validate_installed(case[0], case[1])
        case[5].write_text("changed after validation")
        with pytest.raises(ValueError, match="changed"):
            finalize(case)
        assert not startup_permission(case[5], case[4])[0]
        assert "committed" not in [row["event"] for row in events(case[2])]


@pytest.mark.parametrize(
    "boundary",
    ["installed_validated", "paired", "activation_recorded", "committed", "unlinked"],
)
def test_child_exit_at_terminal_boundaries_retains_ordered_evidence(tmp_path, boundary):
    script = r"""
import os, sys
from pathlib import Path
from Tests.Backup_Recovery.test_publication_finalization import installed, finalize
from tldw_chatbook.Backup_Recovery.journal import Journal
from tldw_chatbook.Backup_Recovery import publication, activation
bind = activation.bind_activation
def after_pair(*args, **kwargs):
    result = bind(*args, **kwargs)
    if sys.argv[2] == 'paired':
        os._exit(73)
    return result
activation.bind_activation = after_pair
original = Journal._append
def append(self, parent, event, evidence):
    result = original(self, parent, event, evidence)
    if event == sys.argv[2]:
        os._exit(73)
    return result
Journal._append = append
unlink = publication.os.unlink
def stop_after_unlink(path, **kwargs):
    result = unlink(path, **kwargs)
    if sys.argv[2] == 'unlinked' and str(path).startswith('pending-'):
        os._exit(73)
    return result
publication.os.unlink = stop_after_unlink
with installed(Path(sys.argv[1])) as case:
    finalize(case)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), boundary],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 73, result.stderr[-4000:]
    bootstrap = tmp_path / "bootstrap"
    pending = bootstrap / ("pending-" + _key("finalize-op") + ".json")
    journal = Journal(tmp_path / "control", "finalize-op")
    recorded = [row["event"] for row in events(journal)]
    assert ("committed" in recorded) is (boundary in {"committed", "unlinked"})
    assert pending.exists() is (boundary != "unlinked")
    assert not activation_permission(
        "config",
        config_selector=tmp_path / "isolated/config/config.toml",
        bootstrap_root=bootstrap,
    )


@pytest.mark.parametrize("boundary", ["activation_recorded", "committed", "unlinked"])
def test_real_write_failure_retries_without_duplicate_commit_or_lost_review(
    tmp_path, monkeypatch, boundary
):
    with installed(tmp_path) as case:
        journal = case[2]
        append = Journal._append
        flush = publication.flush_directory
        unlink = publication.os.unlink
        unlinked = False
        fired = False

        def fail_append(self, parent, event, evidence):
            nonlocal fired
            result = append(self, parent, event, evidence)
            if event == boundary and not fired:
                fired = True
                raise OSError("terminal barrier failure")
            return result

        def remember_unlink(path, **kwargs):
            nonlocal unlinked
            result = unlink(path, **kwargs)
            if str(path).startswith("pending-"):
                unlinked = True
            return result

        def fail_flush(parent):
            nonlocal fired
            if boundary == "unlinked" and unlinked and not fired:
                fired = True
                raise OSError("terminal barrier failure")
            return flush(parent)

        monkeypatch.setattr(Journal, "_append", fail_append)
        monkeypatch.setattr(publication.os, "unlink", remember_unlink)
        monkeypatch.setattr(publication, "flush_directory", fail_flush)
        with pytest.raises(OSError, match="terminal barrier"):
            finalize(case)
        pending = case[4] / ("pending-" + _key(journal.operation_id) + ".json")
        assert pending.exists() is (boundary != "unlinked")
        store = ActivationStore(journal.root.parent / "activation")
        store.approve("g", "config")
    authority = admission_authority(case[4])
    with authority.maintenance(("profile", UNBOUND_NAMESPACE), 3) as fresh:
        assert finalize(case, session=fresh) == "g"
        assert store.allowed("g", "config")
        assert [row["event"] for row in events(journal)].count("committed") == 1


def test_pending_identity_replacement_cannot_be_retired(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import activation

    with installed(tmp_path) as case:
        bind = activation.bind_activation
        pending = case[4] / ("pending-" + _key(case[2].operation_id) + ".json")

        def replace_pending(*args, **kwargs):
            result = bind(*args, **kwargs)
            replacement = case[4] / "replacement"
            replacement.write_bytes(pending.read_bytes())
            replacement.chmod(0o600)
            replacement.replace(pending)
            return result

        monkeypatch.setattr(activation, "bind_activation", replace_pending)
        with pytest.raises(ValueError, match="pending_changed"):
            finalize(case)
        assert pending.exists()


@pytest.mark.parametrize("record", ["profile", "activation", "requirements"])
def test_terminal_retry_refuses_lost_independent_activation_evidence(
    tmp_path, monkeypatch, record
):
    with installed(tmp_path) as case:
        original = Journal._append

        def stop(self, parent, event, evidence):
            result = original(self, parent, event, evidence)
            if event == "committed":
                raise OSError("stop after commit")
            return result

        monkeypatch.setattr(Journal, "_append", stop)
        with pytest.raises(OSError, match="stop after commit"):
            finalize(case)
        monkeypatch.setattr(Journal, "_append", original)
        store = ActivationStore(case[2].root.parent / "activation")
        path = (
            store._generation("g") / "required.json"
            if record == "requirements"
            else case[4] / (record + "-" + _key(str(case[5])) + ".json")
        )
        path.unlink()
        with pytest.raises((ValueError, OSError)):
            finalize(case)
        assert not startup_permission(case[5], case[4])[0]
        assert not path.exists()


def test_installed_change_during_activation_cannot_commit(tmp_path, monkeypatch):
    from tldw_chatbook.Backup_Recovery import activation

    with installed(tmp_path) as case:
        bind = activation.bind_activation

        def change_installed(*args, **kwargs):
            result = bind(*args, **kwargs)
            (case[5].parent / "ui_state.toml").write_text("changed during activation")
            return result

        monkeypatch.setattr(activation, "bind_activation", change_installed)
        with pytest.raises(ValueError, match="installed.*changed"):
            finalize(case)
        assert "committed" not in [row["event"] for row in events(case[2])]
        assert not startup_permission(case[5], case[4])[0]


def test_native_writer_stays_blocked_after_fence_clear_until_session_exit(tmp_path):
    import select

    from Tests.Backup_Recovery.test_installed_validation_maintenance import _WRITER

    marker = tmp_path / "writer-finished"
    child = None
    try:
        with installed(tmp_path) as case:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _WRITER,
                    str(case[4] / "admission"),
                    str(marker),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert select.select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline().strip() == "ready"
            assert finalize(case) == "g"
            assert not select.select([child.stdout], [], [], 0.15)[0]
            assert not marker.exists()
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 0, stderr
        assert "finished" in stdout
        assert marker.exists()
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()


def test_well_formed_terminal_labels_do_not_authorize_absent_pointer(tmp_path):
    from tldw_chatbook.Backup_Recovery.journal import _evidence_digest, observe_artifact

    with installed(tmp_path) as case:
        candidate, plan, journal, _, bootstrap, selector = case
        journal.validate_installed(candidate, plan)
        installed_proof = events(journal)[-1]["evidence"]
        proof = {
            "generation": "g",
            "plan_digest": publication._plan_digest(plan),
            "installed_digest": _evidence_digest(installed_proof),
            "selectors": [str(selector)],
            "owners": ["config"],
            "records": [
                observe_artifact(path)
                for path in (
                    selector,
                    selector.parent / "ui_state.toml",
                    candidate / "candidate.json",
                )
            ],
        }
        journal.record("activation_recorded", proof)
        journal.record(
            "committed",
            {"generation": "g", "activation_digest": _evidence_digest(proof)},
        )
        pending = bootstrap / ("pending-" + _key(journal.operation_id) + ".json")
        pending.unlink()
        with pytest.raises(ValueError, match="activation_unverified"):
            finalize(case)
        assert not (journal.root.parent / "activation").exists()


def test_finalization_requires_actual_unbound_guard(tmp_path):
    with installed(tmp_path, unbound_guard=False) as case:
        assert "bootstrap.unbound" not in case[3]._names
        before = events(case[2])
        with pytest.raises(ValueError, match="unbound_guard_required"):
            finalize(case)
        assert events(case[2]) == before
        assert not startup_permission(case[5], case[4])[0]


def test_unbound_storage_writer_stays_blocked_after_actual_fence_unlink(tmp_path):
    import select

    writer = r"""
import sys
from pathlib import Path
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission as storage
root, selector, target = map(Path, sys.argv[1:])
bootstrap.default_bootstrap_root = lambda: root
storage.effective_config_path = lambda: selector
assert storage._scope(root, selector, target) == ('bootstrap.unbound',)
print('ready', flush=True)
with storage.acquire_storage(target):
    target.write_text('unbound writer admitted')
print('finished', flush=True)
"""
    child = None
    try:
        with installed(tmp_path) as case:
            target = case[5].parent / "ui_state.toml"
            original = target.read_bytes()
            assert finalize(case) == "g"
            pending = case[4] / ("pending-" + _key(case[2].operation_id) + ".json")
            assert not pending.exists()
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    writer,
                    str(case[4]),
                    str(tmp_path / "new-unbound.toml"),
                    str(target),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            assert select.select([child.stdout], [], [], 5)[0]
            assert child.stdout.readline().strip() == "ready"
            assert not select.select([child.stdout], [], [], 1)[0]
            assert target.read_bytes() == original
        stdout, stderr = child.communicate(timeout=5)
        assert child.returncode == 0, stderr
        assert "finished" in stdout
        assert target.read_text() == "unbound writer admitted"
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait()
