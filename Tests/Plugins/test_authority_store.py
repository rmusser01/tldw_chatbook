"""Real crypto and private files, with isolated rollback-marker backend."""

import copy
import json

import pytest

from Tests.hooks_v2_process_support import child_argv


class MemoryMarker:
    reduced_protection = False

    def __init__(self):
        self.marker = None

    def load_marker(self):
        return copy.deepcopy(self.marker)

    def save_marker(self, marker):
        self.marker = copy.deepcopy(marker)

    def clear(self):
        self.marker = None


def store_at(path, marker=None, **kwargs):
    from tldw_chatbook.Plugins.authority_store import PluginAuthorityStore

    return PluginAuthorityStore(path, marker or MemoryMarker(), **kwargs)


def transition(store, operation_id="operation-1"):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    snapshot = store.verify_current()
    snapshot["installations"] = [
        {
            "installation_id": "installed",
            "revision_digest": None,
            "activation_default": False,
        }
    ]
    snapshot["operation_result"] = {
        "operation_id": operation_id,
        "installation_id": "installed",
        "kind": "install",
        "revision_digest": None,
        "result": "committed",
    }
    old = store.load_marker()
    new = PluginMarker(
        generation=old.generation + 1,
        operation_id=operation_id,
        recovery_snapshot_digest=snapshot_digest(snapshot),
    )
    return snapshot, old, new


def test_bootstrap_unlock_and_prepared_only_is_not_commit(tmp_path):
    store = store_at(tmp_path / "plugins")
    assert store.posture() == "needs_setup"
    with pytest.raises(ValueError):
        store.unlock("pw")
    store.bootstrap("pw")
    assert store.load_marker().generation == 0
    snapshot, old, new = transition(store)
    store.prepare(snapshot, old, new)
    evidence = store.verify_transition(new.operation_id)
    assert evidence.old == old and evidence.new == new
    assert evidence.committed is False
    with pytest.raises(ValueError):
        store.advance_marker(old, new)
    assert store.load_marker() == old
    store.certify_commit(old, new)
    assert store.verify_transition(new.operation_id).committed
    store.advance_marker(old, new)
    reopened = store_at(tmp_path / "plugins", store.marker_store)
    assert reopened.posture() == "locked"
    with pytest.raises(ValueError):
        reopened.unlock("wrong")
    with pytest.raises(ValueError):
        reopened.prepare(snapshot, old, new)
    reopened.unlock("pw")
    assert reopened.verify_current() == snapshot
    assert reopened.verify_snapshot(new) == snapshot
    assert reopened.posture() == "ready"
    with pytest.raises(ValueError):
        reopened.bootstrap("pw")


def test_immutable_retry_conflicts_and_replay_are_rejected(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, new = transition(store, "../unsafe/operation")
    store.prepare(snapshot, old, new)
    store.prepare(snapshot, old, new)
    store.certify_commit(old, new)
    store.certify_commit(old, new)
    store.advance_marker(old, new)
    store.advance_marker(old, new)  # exact lost-response retry
    assert store.verify_current() == snapshot
    store.marker_store.save_marker(old.model_dump())
    with pytest.raises(ValueError):
        store.verify_snapshot(new.model_copy(update={"operation_id": "another"}))
    store.unlock("pw")  # old marker authenticates only old snapshot
    assert store.verify_current()["installations"] == []
    assert store.verify_transition(new.operation_id).committed
    store.advance_marker(old, new)
    assert store.verify_current() == snapshot


def test_reduced_backend_requires_explicit_acceptance(tmp_path):
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore

    backend = FilePluginMarkerStore(tmp_path / "plugins")
    store = store_at(tmp_path / "plugins", backend)
    assert store.posture() == "reduced_acceptance_required"
    with pytest.raises(ValueError):
        store.bootstrap("pw")
    accepted = store_at(tmp_path / "plugins", backend, accept_reduced_protection=True)
    accepted.bootstrap("pw")
    assert accepted.posture() == "ready_reduced"
    assert accepted.verify_current()["installations"] == []


def test_partial_setup_and_marker_unavailable_fail_closed(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.store_dir.mkdir()
    (store.store_dir / "metadata.json").write_text("{}")
    assert store.posture() == "recovery_required"
    with pytest.raises(ValueError):
        store.bootstrap("pw")

    class LockedMarker(MemoryMarker):
        def load_marker(self):
            raise RuntimeError("locked backend")

    locked = store_at(tmp_path / "other", LockedMarker())
    assert locked.posture() == "unavailable"
    with pytest.raises(RuntimeError):
        locked.bootstrap("pw")


@pytest.mark.parametrize(
    "artifact,field",
    [
        ("snapshot", "ciphertext"),
        ("snapshot", "tag"),
        ("snapshot", "nonce"),
        ("snapshot", "header"),
        ("prepared", "mac"),
        ("prepared", "old"),
        ("prepared", "new"),
        ("committed", "mac"),
        ("committed", "old"),
        ("committed", "new"),
    ],
)
def test_tampered_protected_evidence_never_returns_authority(tmp_path, artifact, field):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    store.prepare(snapshot, old, new)
    store.certify_commit(old, new)
    assert store.verify_transition(new.operation_id).snapshot == snapshot
    if artifact == "snapshot":
        path = next(
            (store.store_dir / "snapshots").glob(f"{new.recovery_snapshot_digest}.json")
        )
    else:
        path = next(
            (
                store.store_dir
                / ("intents" if artifact == "prepared" else "certificates")
            ).glob("*.json")
        )
    value = json.loads(path.read_text())
    if artifact == "snapshot" and field == "header":
        value["header"]["marker"]["operation_id"] = "substituted"
    elif artifact == "snapshot":
        value["blob"][field] = "AAAA"
    elif field == "mac":
        value["mac"] = "0" * 64
    else:
        value["payload"][field]["recovery_snapshot_digest"] = "f" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        store.verify_transition(new.operation_id)
    assert store.load_marker() == old


def test_copied_prepared_intent_cannot_substitute_commit_certificate(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    store.prepare(snapshot, old, new)
    assert not store.verify_transition(new.operation_id).committed
    intent = next((store.store_dir / "intents").glob("*.json"))
    target = store.store_dir / "certificates" / intent.name
    target.parent.mkdir(mode=0o700)
    target.write_bytes(intent.read_bytes())
    with pytest.raises(ValueError, match="authentication"):
        store.advance_marker(old, new)


@pytest.mark.bootstrap_profile
def test_recovery_read_works_in_fresh_process_with_old_marker(tmp_path):
    import subprocess

    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore

    path = tmp_path / "plugins"
    store = store_at(path, FilePluginMarkerStore(path), accept_reduced_protection=True)
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    store.prepare(snapshot, old, new)
    store.certify_commit(old, new)
    script = """
import sys
from pathlib import Path
import tldw_chatbook.Plugins.authority_store as module
p = Path(sys.argv[1])
s = module.PluginAuthorityStore(p, module.FilePluginMarkerStore(p), accept_reduced_protection=True)
s.unlock("pw")
e = s.verify_transition("operation-1")
assert s.load_marker() == e.old and e.committed
s.advance_marker(e.old, e.new)
assert s.verify_current()["operation_result"]["operation_id"] == "operation-1"
print(module.__file__)
"""
    completed = subprocess.run(
        child_argv(script) + [str(path)],
        capture_output=True,
        check=False,
        timeout=30,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert (
        str(
            __import__("pathlib").Path.cwd()
            / "tldw_chatbook/Plugins/authority_store.py"
        )
        in completed.stdout
    )
    assert store.verify_current() == snapshot


@pytest.mark.parametrize(
    "boundary", ["write", "file_fsync", "rename", "parent_fsync", "postcondition"]
)
def test_failed_durable_preparation_never_advances_marker(
    tmp_path, monkeypatch, boundary
):
    import os
    import stat

    import tldw_chatbook.Utils.private_paths as private

    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    # Establish artifact directories before injecting the file-publication faults.
    for name in ("snapshots", "intents"):
        (store.store_dir / name).mkdir(mode=0o700, exist_ok=True)
    if boundary == "postcondition":
        monkeypatch.setattr(
            private, "_private_file_postcondition_holds", lambda *a, **k: False
        )
    elif boundary in ("file_fsync", "parent_fsync"):
        original = os.fsync

        def fsync(fd):
            is_dir = stat.S_ISDIR(os.fstat(fd).st_mode)
            if is_dir == (boundary == "parent_fsync"):
                raise OSError("injected sync failure")
            original(fd)

        monkeypatch.setattr(private.os, "fsync", fsync)
    else:

        def fail(*a, **k):
            raise OSError("injected publication failure")

        monkeypatch.setattr(private.os, boundary, fail)
    with pytest.raises((OSError, ValueError)):
        store.prepare(snapshot, old, new)
    assert store.load_marker() == old
    assert not (store.store_dir / "certificates").exists()


def test_unverified_platform_is_not_claimed_durable(tmp_path, monkeypatch):
    import tldw_chatbook.Plugins.authority_store as module
    from tldw_chatbook.Utils.private_paths import PrivatePathResult, PrivatePathStatus

    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    monkeypatch.setattr(
        module,
        "atomic_private_write_bytes",
        lambda path, *a, **k: PrivatePathResult(
            path, PrivatePathStatus.UNVERIFIED_PLATFORM
        ),
    )
    with pytest.raises(ValueError, match="unqualified"):
        store.prepare(snapshot, old, new)
    assert store.load_marker() == old


def test_reset_is_independent_in_both_directions(tmp_path):
    from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
    from tldw_chatbook.Skills_Interop.skill_trust_store import (
        FileSkillTrustGenerationMarkerStore,
        SkillTrustStore,
    )

    root = tmp_path / "trust"
    root.mkdir(mode=0o700)
    skills = tmp_path / "skills"
    skills.mkdir()
    standalone = SkillTrustService(
        skills_dir=skills,
        trust_store=SkillTrustStore(
            store_dir=root,
            marker_store=FileSkillTrustGenerationMarkerStore(
                root / "generation_marker.json", store_dir=root
            ),
        ),
        key_cache=None,
    )
    standalone.bootstrap_trust("standalone", salt=b"s" * 32)
    plugin = store_at(root / "plugins")
    plugin.bootstrap("plugin")
    plugin_marker = plugin.load_marker()
    standalone.reset_trust()
    assert plugin.load_marker() == plugin_marker
    assert plugin.verify_current()["installations"] == []
    standalone.bootstrap_trust("standalone", salt=b"s" * 32)
    manifest = standalone.trust_store.manifest_path.read_bytes()
    archive = plugin.reset(operation_id="reset-1")
    assert archive.is_dir()
    assert plugin.posture() == "needs_setup"
    assert standalone.trust_posture() == "ready"
    assert standalone.trust_store.manifest_path.read_bytes() == manifest
    plugin.bootstrap("new plugin")
    assert plugin.verify_current()["installations"] == []


def test_reset_failure_retains_evidence(tmp_path):
    class RefusesClear(MemoryMarker):
        def clear(self):
            raise RuntimeError("locked")

    store = store_at(tmp_path / "plugins", RefusesClear())
    store.bootstrap("pw")
    marker = store.load_marker()
    with pytest.raises(RuntimeError):
        store.reset(operation_id="reset-1")
    assert store.load_marker() == marker
    assert (store.store_dir / "metadata.json").exists()
    assert store.posture() == "recovery_required"


def test_snapshot_header_rejects_bool_generation_even_when_equal_to_integer(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    store.prepare(snapshot, old, new)
    path = store.store_dir / "snapshots" / f"{new.recovery_snapshot_digest}.json"
    value = json.loads(path.read_text())
    value["header"]["marker"]["generation"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        store.verify_snapshot(new)


def test_keyring_namespaces_and_profile_scopes_are_independent(tmp_path, monkeypatch):
    import tldw_chatbook.Plugins.authority_store as module
    import tldw_chatbook.Skills_Interop.skill_trust_store as standalone

    class Backend:
        def __init__(self):
            self.values = {}

        def get_password(self, service, account):
            return self.values.get((service, account))

        def set_password(self, service, account, value):
            self.values[service, account] = value

        def delete_password(self, service, account):
            del self.values[service, account]

    backend = Backend()
    monkeypatch.setattr(
        module, "is_secure_keyring_backend", lambda candidate: candidate is backend
    )
    monkeypatch.setattr(
        standalone, "is_secure_keyring_backend", lambda candidate: candidate is backend
    )
    skill_marker = standalone.KeyringSkillTrustGenerationMarkerStore(
        keyring_backend=backend
    )
    skill_marker.save_marker(generation=9, manifest_digest="standalone")
    first = module.KeyringPluginMarkerStore(tmp_path / "one", keyring_backend=backend)
    second = module.KeyringPluginMarkerStore(tmp_path / "two", keyring_backend=backend)
    store = store_at(tmp_path / "one", first)
    store.bootstrap("pw")
    assert second.load_marker() is None
    skill_marker.clear()
    assert store.verify_current()["installations"] == []
    skill_marker.save_marker(generation=9, manifest_digest="standalone")
    first.clear()
    assert skill_marker.load_marker() == {
        "generation": 9,
        "manifest_digest": "standalone",
    }
    backend.set_password(module.MARKER_SERVICE, first.account, "corrupt marker")
    first.clear()
    assert first.load_marker() is None
    assert skill_marker.load_marker()["generation"] == 9


def test_store_rejects_symlink_artifact_and_preserves_target(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    marker = store.load_marker()
    path = store.store_dir / "snapshots" / f"{marker.recovery_snapshot_digest}.json"
    target = tmp_path / "untouched"
    original = path.read_bytes()
    target.write_bytes(original)
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(OSError):
        store.verify_current()
    assert target.read_bytes() == original


def complete_snapshot():
    from tldw_chatbook.Plugins.authority import empty_snapshot

    value = empty_snapshot()
    value.update(
        {
            "installations": [
                {
                    "installation_id": "installed",
                    "revision_digest": "a" * 64,
                    "activation_default": False,
                }
            ],
            "revisions": [
                {
                    "installation_id": "installed",
                    "revision_digest": "a" * 64,
                    "source_identity": "local-source",
                    "dialect": "agent-plugins",
                    "format_version": "1.0.0",
                    "adapter_version": "chatbook-portable/1",
                    "root_manifest": "plugin.json",
                    "overlay_identities": ["overlay-digest-reference"],
                    "content_digest": "b" * 64,
                    "source_digest": "c" * 64,
                    "materialized_identity": "/retained/package",
                    "link_targets": {"asset": "resolved-asset"},
                    "activation_blockers": ["constraints_unknown"],
                    "variables_digest": "d" * 64,
                    "rejected": False,
                }
            ],
            "components": [
                {
                    "installation_id": "installed",
                    "revision_digest": "a" * 64,
                    "component_id": "mcp:server",
                    "kind": "mcp",
                    "local_id": "server",
                    "path": "mcp.json",
                    "definition_digest": "e" * 64,
                    "support": "unsupported",
                    "selection": "excluded",
                    "dependencies": ["hook:missing"],
                    "activation_blockers": ["transport_unsupported"],
                }
            ],
            "selections": [
                {
                    "installation_id": "installed",
                    "revision_digest": "a" * 64,
                    "component_id": "mcp:server",
                    "selected": False,
                }
            ],
            "activation": [
                {
                    "installation_id": "installed",
                    "workspace_id": "work",
                    "intent": "disabled",
                }
            ],
            "mappings": [
                {
                    "installation_id": "installed",
                    "mapping_id": "mapping",
                    "component_id": "mcp:server",
                    "revision_digest": "a" * 64,
                    "kind": "connection",
                    "target_reference": "connection-ref",
                    "definition_digest": "e" * 64,
                    "configuration_digest": "f" * 64,
                    "credential_bindings": [
                        {
                            "reference_id": "credential-ref",
                            "authority_generation": 7,
                            "authentication_method": "oauth",
                            "issuer": "issuer-ref",
                            "audience": "audience-ref",
                            "endpoint_origin": "https://service.example",
                            "principal": "principal-ref",
                            "scopes": ["read", "write"],
                            "identity_state": "verified",
                        }
                    ],
                }
            ],
            "authority_generations": [
                {
                    "installation_id": "installed",
                    "scope_kind": "workspace",
                    "workspace_id": "work",
                    "generation": 4,
                    "revoked": True,
                }
            ],
            "revision_trust": [
                {
                    "installation_id": "installed",
                    "revision_digest": "a" * 64,
                    "reviewed": False,
                }
            ],
            "tombstones": [
                {
                    "installation_id": "removed",
                    "generation": 9,
                    "operation_id": "removal",
                }
            ],
            "data_roots": [
                {
                    "root_id": "root",
                    "installation_id": "installed",
                    "workspace_id": "work",
                    "path": "/owned/data",
                    "generation": 2,
                    "deletion_fenced": True,
                }
            ],
            "operation_result": {
                "operation_id": "complete-op",
                "installation_id": "installed",
                "kind": "configure",
                "revision_digest": "a" * 64,
                "result": "committed",
            },
        }
    )
    return value


def _leaves(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, (*path, key))
    elif isinstance(value, list):
        for key, child in enumerate(value):
            yield from _leaves(child, (*path, key))
    else:
        yield path


@pytest.mark.parametrize(
    "field_path",
    list(_leaves(complete_snapshot())),
    ids=lambda path: ".".join(map(str, path)),
)
def test_every_authoritative_field_is_bound_independently(tmp_path, field_path):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot = complete_snapshot()
    old = store.load_marker()
    new = PluginMarker(
        generation=1,
        operation_id="complete-op",
        recovery_snapshot_digest=snapshot_digest(snapshot),
    )
    store.prepare(snapshot, old, new)
    assert store.verify_transition("complete-op").snapshot == snapshot
    changed = copy.deepcopy(snapshot)
    node = changed
    for key in field_path[:-1]:
        node = node[key]
    key = field_path[-1]
    value = node[key]
    node[key] = (
        (not value)
        if type(value) is bool
        else (value + 1)
        if type(value) is int
        else "tampered"
    )
    with pytest.raises(ValueError):
        store.prepare(changed, old, new)
    assert store.load_marker() == old


def test_same_package_bytes_with_reviewed_workspace_change_need_new_marker(tmp_path):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    first = complete_snapshot()
    old = store.load_marker()
    current = PluginMarker(
        generation=1,
        operation_id="complete-op",
        recovery_snapshot_digest=snapshot_digest(first),
    )
    store.prepare(first, old, current)
    store.certify_commit(old, current)
    store.advance_marker(old, current)
    changed = copy.deepcopy(first)
    changed["activation"][0]["intent"] = "enabled"
    changed["operation_result"]["operation_id"] = "workspace-edit"
    next_marker = PluginMarker(
        generation=2,
        operation_id="workspace-edit",
        recovery_snapshot_digest=snapshot_digest(changed),
    )
    assert next_marker.recovery_snapshot_digest != current.recovery_snapshot_digest
    store.prepare(changed, current, next_marker)
    assert store.verify_current()["activation"][0]["intent"] == "disabled"
    store.certify_commit(current, next_marker)
    store.advance_marker(current, next_marker)
    assert store.verify_current()["activation"][0]["intent"] == "enabled"
    assert store.verify_current()["revision_trust"][0]["reviewed"] is False


def test_snapshot_operation_mismatch_fails_before_persisting(tmp_path):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, _ = transition(store)
    wrong = PluginMarker(
        generation=1,
        operation_id="other-operation",
        recovery_snapshot_digest=snapshot_digest(snapshot),
    )
    before = set((store.store_dir / "snapshots").iterdir())
    with pytest.raises(ValueError):
        store.prepare(snapshot, old, wrong)
    assert set((store.store_dir / "snapshots").iterdir()) == before


def test_retry_requires_durable_existing_evidence(tmp_path, monkeypatch):
    import os

    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    store.prepare(snapshot, old, new)

    def fail(fd):
        raise OSError("durability unavailable")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        store.prepare(snapshot, old, new)


def test_partial_bootstrap_marker_failure_stays_recovery_required(tmp_path):
    class RefusesSave(MemoryMarker):
        def save_marker(self, marker):
            raise RuntimeError("keyring locked during save")

    store = store_at(tmp_path / "plugins", RefusesSave())
    with pytest.raises(RuntimeError):
        store.bootstrap("pw")
    assert store.posture() == "recovery_required"
    with pytest.raises(ValueError):
        store.bootstrap("pw")
    with pytest.raises(ValueError):
        store.verify_current()


def test_complete_snapshot_is_encrypted_and_unknown_secret_fields_rejected(tmp_path):
    from tldw_chatbook.Plugins.authority import PluginMarker, snapshot_digest

    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    snapshot = complete_snapshot()
    marker = PluginMarker(
        generation=1,
        operation_id="complete-op",
        recovery_snapshot_digest=snapshot_digest(snapshot),
    )
    old = store.load_marker()
    store.prepare(snapshot, old, marker)
    persisted = (
        store.store_dir / "snapshots" / f"{marker.recovery_snapshot_digest}.json"
    ).read_text()
    assert "credential-ref" not in persisted
    assert "principal-ref" not in persisted
    assert "configuration_digest" not in persisted
    changed = copy.deepcopy(snapshot)
    changed["mappings"][0]["credential_bindings"][0]["access_token"] = "do-not-store"
    with pytest.raises(ValueError):
        store.prepare(changed, old, marker)


def test_bootstrap_synchronizes_all_new_directory_ancestors(tmp_path, monkeypatch):
    import os
    import stat

    import tldw_chatbook.Plugins.authority_store as module

    synced = set()
    original = os.fsync

    def fsync(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            synced.add((info.st_dev, info.st_ino))
        return original(fd)

    monkeypatch.setattr(module.os, "fsync", fsync)
    path = tmp_path / "skills" / "trust" / "plugins"
    store = store_at(path)
    store.bootstrap("pw")
    for directory in (tmp_path, path.parent.parent, path.parent, path):
        info = directory.stat()
        assert (info.st_dev, info.st_ino) in synced


def test_explicit_reset_can_clear_corrupt_reduced_marker(tmp_path):
    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore

    path = tmp_path / "plugins"
    store = store_at(path, FilePluginMarkerStore(path), accept_reduced_protection=True)
    store.bootstrap("pw")
    (path / "generation_marker.json").write_text("not valid JSON")
    assert store.posture() == "unavailable"
    archive = store.reset(operation_id="reset-1")
    assert archive.is_dir()
    assert store.posture() == "needs_setup"
    store.bootstrap("new passphrase")
    assert store.verify_current()["installations"] == []


def test_marker_retry_finishes_durability_after_marker_rename(tmp_path, monkeypatch):
    import os
    import stat

    from tldw_chatbook.Plugins.authority_store import FilePluginMarkerStore
    from tldw_chatbook.Utils.private_paths import PrivatePathError

    path = tmp_path / "plugins"
    store = store_at(path, FilePluginMarkerStore(path), accept_reduced_protection=True)
    store.bootstrap("pw")
    snapshot, old, new = transition(store)
    store.prepare(snapshot, old, new)
    store.certify_commit(old, new)
    real_fsync = os.fsync
    renamed = False
    fail_parent_sync = True
    syncs = []
    parent_info = path.stat()

    def fsync(fd):
        nonlocal renamed
        info = os.fstat(fd)
        marker_parent = (info.st_dev, info.st_ino) == (
            parent_info.st_dev,
            parent_info.st_ino,
        )
        if marker_parent and store.load_marker() == new:
            renamed = True
        if renamed and marker_parent and fail_parent_sync:
            raise OSError("marker parent sync failed after rename")
        syncs.append((stat.S_ISDIR(info.st_mode), marker_parent, renamed))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(PrivatePathError):
        store.advance_marker(old, new)
    assert renamed and store.load_marker() == new
    fail_parent_sync = False
    renamed = False
    syncs.clear()
    store.advance_marker(old, new)
    assert any(not is_dir for is_dir, _, _ in syncs), (
        "retry must synchronize marker bytes"
    )
    assert any(is_parent and after_rename for _, is_parent, after_rename in syncs), (
        "retry must synchronize marker publication"
    )
    assert store.verify_current() == snapshot


@pytest.mark.parametrize(
    "failure_boundary", ["archive_parent", "completed_receipt_parent"]
)
def test_reset_retry_recovers_exact_archive_after_failed_parent_sync(
    tmp_path, monkeypatch, failure_boundary
):
    import os

    path = tmp_path / "plugins"
    marker = MemoryMarker()
    store = store_at(path, marker)
    store.bootstrap("pw")
    source_identity = path.stat()
    parent_identity = path.parent.stat()
    real_fsync = os.fsync
    fail_sync = True
    synced = set()

    def fsync(fd):
        info = os.fstat(fd)
        parent = (info.st_dev, info.st_ino) == (
            parent_identity.st_dev,
            parent_identity.st_ino,
        )
        receipt = path.with_name("plugins-reset-state.json")
        completed = receipt.exists() and any(
            record["phase"] == "completed"
            for record in json.loads(receipt.read_text())["resets"]
        )
        boundary_reached = not path.exists() and (
            failure_boundary == "archive_parent" or completed
        )
        if fail_sync and parent and boundary_reached:
            raise OSError("reset parent sync failed after rename")
        synced.add((info.st_dev, info.st_ino))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    with pytest.raises(OSError):
        store.reset(operation_id="reset-1")
    archives = list(tmp_path.glob("plugins-reset-*"))
    archives = [candidate for candidate in archives if candidate.is_dir()]
    assert len(archives) == 1 and not path.exists()
    archive = archives[0]
    assert archive.stat().st_ino == source_identity.st_ino
    fresh = store_at(path, marker)
    assert fresh.posture() == "recovery_required"
    with pytest.raises((OSError, ValueError)):
        fresh.bootstrap("must not ignore pending reset")
    fail_sync = False
    synced.clear()
    assert fresh.reset(operation_id="reset-1") == archive
    assert (parent_identity.st_dev, parent_identity.st_ino) in synced
    assert (source_identity.st_dev, source_identity.st_ino) in synced
    assert fresh.posture() == "needs_setup"
    synced.clear()
    assert fresh.reset(operation_id="reset-1") == archive
    assert (parent_identity.st_dev, parent_identity.st_ino) in synced


def test_old_reset_ids_preserve_rebootstrapped_namespace_after_later_reset(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("first")
    first_archive = store.reset(operation_id="reset-first")
    store.bootstrap("second")
    second_archive = store.reset(operation_id="reset-second")
    store.bootstrap("third")
    current = store.load_marker()
    metadata = (store.store_dir / "metadata.json").read_bytes()
    current_snapshot = store.verify_current()
    assert store.reset(operation_id="reset-first") == first_archive
    assert store.reset(operation_id="reset-second") == second_archive
    assert store.load_marker() == current
    assert (store.store_dir / "metadata.json").read_bytes() == metadata
    assert store.verify_current() == current_snapshot
    assert first_archive != second_archive
    assert first_archive.is_dir() and second_archive.is_dir()


def test_new_reset_refuses_receipt_capacity_without_evicting_history(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("first")
    store.reset(operation_id="reset-first")
    store.bootstrap("second")
    current = store.load_marker()
    receipt_path = store.store_dir.with_name("plugins-reset-state.json")
    state = json.loads(receipt_path.read_text())
    original = state["resets"][0]
    for index in range(1, 1000):
        state["resets"].append(
            {
                **original,
                "operation_id": f"old-{index}",
                "archive": f"plugins-reset-{state['store_scope']}-{index:032x}",
            }
        )
    receipt_path.write_text(json.dumps(state))
    before = receipt_path.read_bytes()
    with pytest.raises(ValueError, match="history limit"):
        store.reset(operation_id="new-reset")
    assert receipt_path.read_bytes() == before
    assert store.store_dir.is_dir() and store.load_marker() == current


def test_pending_reset_refuses_other_id_and_can_resume_same_id(tmp_path):
    class FailingMarker(MemoryMarker):
        fail = True

        def clear(self):
            if self.fail:
                raise RuntimeError("marker unavailable")
            super().clear()

    marker = FailingMarker()
    store = store_at(tmp_path / "plugins", marker)
    store.bootstrap("pw")
    old = store.load_marker()
    with pytest.raises(RuntimeError):
        store.reset(operation_id="pending")
    marker.fail = False
    fresh = store_at(store.store_dir, marker)
    with pytest.raises(ValueError, match="different plugin reset pending"):
        fresh.reset(operation_id="other")
    assert fresh.load_marker() == old and fresh.store_dir.exists()
    with pytest.raises(ValueError, match="reset recovery"):
        fresh.unlock("pw")
    archive = fresh.reset(operation_id="pending")
    assert archive.is_dir() and fresh.posture() == "needs_setup"


@pytest.mark.parametrize(
    "change", ["scope", "archive", "inode", "duplicate", "unknown"]
)
def test_invalid_reset_history_blocks_setup_and_retry(tmp_path, change):
    store = store_at(tmp_path / "plugins")
    store.bootstrap("pw")
    archive = store.reset(operation_id="reset-1")
    receipt = store.store_dir.with_name("plugins-reset-state.json")
    state = json.loads(receipt.read_text())
    if change == "scope":
        state["store_scope"] = "other-profile"
    elif change == "archive":
        state["resets"][0]["archive"] = "../unrelated"
    elif change == "inode":
        state["resets"][0]["directory_inode"] += 1
    elif change == "duplicate":
        state["resets"].append(dict(state["resets"][0]))
    else:
        state["authority"] = "allow"
    receipt.write_text(json.dumps(state))
    fresh = store_at(store.store_dir, store.marker_store)
    assert fresh.posture() == "recovery_required"
    with pytest.raises(ValueError):
        fresh.bootstrap("new")
    with pytest.raises(ValueError):
        fresh.reset(operation_id="reset-1")
    assert archive.is_dir() and not store.store_dir.exists()


def test_pristine_reset_id_retry_cannot_reset_later_bootstrap(tmp_path):
    store = store_at(tmp_path / "plugins")
    assert store.reset(operation_id="empty-reset") is None
    store.bootstrap("new namespace")
    current = store.load_marker()
    metadata = (store.store_dir / "metadata.json").read_bytes()
    assert store.reset(operation_id="empty-reset") is None
    assert store.load_marker() == current
    assert (store.store_dir / "metadata.json").read_bytes() == metadata
    assert store.verify_current()["installations"] == []


def test_deeply_nested_reset_receipt_reports_recovery(tmp_path):
    store = store_at(tmp_path / "plugins")
    store.reset(operation_id="empty-reset")
    receipt = store.store_dir.with_name("plugins-reset-state.json")
    receipt.write_text('{"resets":' + "[" * 20000 + "0" + "]" * 20000 + "}")
    assert store.posture() == "recovery_required"
    with pytest.raises(ValueError):
        store.bootstrap("pw")
