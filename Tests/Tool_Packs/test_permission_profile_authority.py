"""Authority and concurrency tests for portable Tool policy profiles."""

from __future__ import annotations

import copy
import errno
import os
import threading

import pytest

from tldw_chatbook.MCP import permission_store as permission_store_module
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.Tool_Packs import binding as binding_module
from tldw_chatbook.Tool_Packs.binding import (
    ProfileMutationError,
    ToolProfileLifecycleCoordinator,
    profile_policy_digest,
)


class _ContestedLockProbe:
    """Expose exact outer lock acquire attempts without timing assumptions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self.first_acquired = threading.Event()
        self.release_first = threading.Event()
        self.second_attempted = threading.Event()
        self.second_acquired = threading.Event()
        self.order: list[str] = []

    def __enter__(self):
        depth = getattr(self._local, "depth", 0)
        thread_name = threading.current_thread().name
        if depth == 0 and thread_name == "second-mutator":
            self.second_attempted.set()
        self._lock.acquire()
        self._local.depth = depth + 1
        if depth == 0 and thread_name == "first-mutator":
            self.order.append("first_acquired")
            self.first_acquired.set()
            assert self.release_first.wait(timeout=2)
        elif depth == 0 and thread_name == "second-mutator":
            self.order.append("second_acquired")
            self.second_acquired.set()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        depth = self._local.depth - 1
        if depth == 0 and threading.current_thread().name == "first-mutator":
            self.order.append("first_released")
        self._local.depth = depth
        self._lock.release()


def _temporary_store_paths(path) -> list:
    private_paths = list(path.parent.glob(f".{path.name}.*.tmp"))
    predictable_path = path.with_suffix(f"{path.suffix}.tmp")
    if predictable_path.exists() or predictable_path.is_symlink():
        private_paths.append(predictable_path)
    return private_paths


def _fail_directory_fsync(monkeypatch, directory, error_number: int) -> None:
    original_open = permission_store_module.os.open
    original_fsync = permission_store_module.os.fsync
    directory_fd: list[int] = []

    def tracked_open(path, flags, mode=0o777):
        fd = original_open(path, flags, mode)
        if os.fspath(path) == os.fspath(directory):
            directory_fd.append(fd)
        return fd

    def controlled_fsync(fd):
        if directory_fd and fd == directory_fd[-1]:
            raise OSError(error_number, os.strerror(error_number))
        return original_fsync(fd)

    monkeypatch.setattr(permission_store_module.os, "open", tracked_open)
    monkeypatch.setattr(permission_store_module.os, "fsync", controlled_fsync)


def _imported_profile(*, first_bind_required: bool = True) -> dict:
    profile = {
        "global_default": "ask",
        "servers": {
            "agent:builtin": {"default": "ask"},
            "local:docs": {
                "default": "deny",
                "tools": {
                    "search": {
                        "state": "allow",
                        "definition_hash": "a" * 64,
                    }
                },
            },
        },
        "profile_kind": "tool_pack_imported",
        "tool_pack_lifecycle": {
            "schema": "tldw.tool-pack-lifecycle/v1",
            "origin": "imported",
            "pack_digest": "b" * 64,
            "imported_at": "2026-08-31T00:00:00Z",
            "first_bind_confirmation_required": first_bind_required,
            "receipt_id": "tp-" + "c" * 32,
            "receipt_digest": "d" * 64,
            "counts": {"matched": 1, "omitted": 0, "pending_deny": 0},
            "policy_digest": "0" * 64,
            "revision": 1,
        },
    }
    profile["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(profile)
    return profile


def _tombstone_profile() -> dict:
    profile = {
        "global_default": "deny",
        "servers": {"agent:builtin": {"default": "deny"}},
        "profile_kind": "tool_pack_tombstone",
        "tool_pack_lifecycle": {
            "schema": "tldw.tool-pack-lifecycle/v1",
            "origin": "tombstone",
            "pack_digest": "b" * 64,
            "imported_at": "2026-08-31T00:00:00Z",
            "removed_at": "2026-08-31T01:00:00Z",
            "first_bind_confirmation_required": False,
            "receipt_id": "tp-" + "e" * 32,
            "receipt_digest": "f" * 64,
            "policy_digest": "0" * 64,
            "revision": 2,
        },
    }
    profile["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(profile)
    return profile


def test_two_store_instances_share_one_path_fence(tmp_path, monkeypatch):
    """A locked reload/save cycle must not lose another instance's write."""
    path = tmp_path / "permissions.json"
    probe = _ContestedLockProbe()
    resolved = path.expanduser().resolve(strict=False)
    monkeypatch.setitem(permission_store_module._PATH_LOCKS, resolved, probe)
    first = MCPPermissionStore(path)
    second = MCPPermissionStore(path)
    assert first._path_lock is second._path_lock is probe

    first_thread = threading.Thread(
        target=first.set_server_default,
        args=("local:first", "deny"),
        name="first-mutator",
    )

    def mutate_second() -> None:
        second.set_server_default("local:second", "ask")

    second_thread = threading.Thread(target=mutate_second, name="second-mutator")
    first_thread.start()
    assert probe.first_acquired.wait(timeout=2)
    second_thread.start()
    assert probe.second_attempted.wait(timeout=2)
    probe.release_first.set()
    assert probe.second_acquired.wait(timeout=2)
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert probe.order == ["first_acquired", "first_released", "second_acquired"]
    servers = MCPPermissionStore(path).load()["profiles"]["default"]["servers"]
    assert servers["local:first"]["default"] == "deny"
    assert servers["local:second"]["default"] == "ask"


def test_install_profile_if_absent_commits_exact_imported_profile(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    snapshot = store.read_snapshot_strict()
    profile = _imported_profile()

    result = store.install_profile_if_absent(
        "research",
        profile,
        expected_generation=snapshot.generation,
        max_profiles=128,
        max_store_bytes=8 * 1024 * 1024,
    )

    assert result.profile_id == "research"
    assert result.revision == 1
    assert result.policy_digest == profile_policy_digest(profile)
    assert result.store_generation == store.read_snapshot_strict().generation
    frozen_profile = store.read_snapshot_strict().payload["profiles"]["research"]
    assert profile_policy_digest(frozen_profile) == result.policy_digest

    with pytest.raises(ProfileMutationError, match="profile_exists"):
        store.install_profile_if_absent(
            "research",
            profile,
            expected_generation=result.store_generation,
            max_profiles=128,
            max_store_bytes=8 * 1024 * 1024,
        )


def test_install_requires_initial_first_bind_confirmation_marker(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    profile = _imported_profile(first_bind_required=False)

    with pytest.raises(ProfileMutationError, match="first_bind_required"):
        store.install_profile_if_absent(
            "research",
            profile,
            expected_generation=store.read_snapshot_strict().generation,
            max_profiles=128,
            max_store_bytes=8 * 1024 * 1024,
        )


def test_save_does_not_reuse_predictable_symlink_temp(tmp_path):
    path = tmp_path / "permissions.json"
    legacy_temp = path.with_suffix(".json.tmp")
    victim = tmp_path / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    try:
        legacy_temp.symlink_to(victim)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    MCPPermissionStore(path).set_kill_switch(True)

    assert victim.read_text(encoding="utf-8") == "untouched"
    assert legacy_temp.is_symlink()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


@pytest.mark.parametrize("failure", ["write", "file_fsync", "replace"])
def test_save_cleans_private_temp_before_replace_failures(
    tmp_path, monkeypatch, failure
):
    path = tmp_path / "permissions.json"
    store = MCPPermissionStore(path)

    def fail(*_args, **_kwargs):
        raise OSError(f"{failure} failed")

    if failure == "write":
        monkeypatch.setattr(permission_store_module.json, "dump", fail)
    elif failure == "file_fsync":
        monkeypatch.setattr(permission_store_module.os, "fsync", fail)
    else:
        monkeypatch.setattr(permission_store_module.os, "replace", fail)

    with pytest.raises(OSError, match=f"{failure} failed"):
        store.set_kill_switch(True)

    assert not path.exists()
    assert _temporary_store_paths(path) == []


def test_save_propagates_real_directory_fsync_failure(tmp_path, monkeypatch):
    path = tmp_path / "permissions.json"
    _fail_directory_fsync(monkeypatch, tmp_path, errno.EIO)

    with pytest.raises(OSError) as caught:
        MCPPermissionStore(path).set_kill_switch(True)

    assert caught.value.errno == errno.EIO
    assert path.exists()
    assert _temporary_store_paths(path) == []


@pytest.mark.parametrize(
    "unsupported_errno",
    sorted(
        {
            errno.EINVAL,
            errno.ENOSYS,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
    ),
)
def test_save_tolerates_unsupported_directory_fsync(
    tmp_path, monkeypatch, unsupported_errno
):
    path = tmp_path / "permissions.json"
    _fail_directory_fsync(monkeypatch, tmp_path, unsupported_errno)

    MCPPermissionStore(path).set_kill_switch(True)

    assert MCPPermissionStore(path).get_kill_switch() is True
    assert _temporary_store_paths(path) == []


def test_install_rejects_casefold_collision_and_stale_generation(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    snapshot = store.read_snapshot_strict()
    first = store.install_profile_if_absent(
        "Research",
        _imported_profile(),
        expected_generation=snapshot.generation,
        max_profiles=128,
        max_store_bytes=8 * 1024 * 1024,
    )

    with pytest.raises(ProfileMutationError, match="profile_id_collision"):
        store.install_profile_if_absent(
            "research",
            _imported_profile(),
            expected_generation=first.store_generation,
            max_profiles=128,
            max_store_bytes=8 * 1024 * 1024,
        )

    stale = first.store_generation
    store.set_kill_switch(True)
    with pytest.raises(ProfileMutationError, match="stale_store"):
        store.install_profile_if_absent(
            "other",
            _imported_profile(),
            expected_generation=stale,
            max_profiles=128,
            max_store_bytes=8 * 1024 * 1024,
        )


def test_install_rejects_invalid_lifecycle_digest_and_projected_caps(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    snapshot = store.read_snapshot_strict()
    mismatched = _imported_profile()
    mismatched["tool_pack_lifecycle"]["policy_digest"] = "9" * 64
    with pytest.raises(ProfileMutationError, match="policy_digest_mismatch"):
        store.install_profile_if_absent(
            "research",
            mismatched,
            expected_generation=snapshot.generation,
            max_profiles=128,
            max_store_bytes=8 * 1024 * 1024,
        )

    structurally_invalid = _imported_profile()
    structurally_invalid["tool_pack_lifecycle"].pop("receipt_id")
    with pytest.raises(ProfileMutationError, match="lifecycle_invalid"):
        store.install_profile_if_absent(
            "research",
            structurally_invalid,
            expected_generation=snapshot.generation,
            max_profiles=128,
            max_store_bytes=8 * 1024 * 1024,
        )

    with pytest.raises(ProfileMutationError, match="profile_limit"):
        store.install_profile_if_absent(
            "research",
            _imported_profile(),
            expected_generation=snapshot.generation,
            max_profiles=1,
            max_store_bytes=8 * 1024 * 1024,
        )

    with pytest.raises(ProfileMutationError, match="store_bytes_limit"):
        store.install_profile_if_absent(
            "research",
            _imported_profile(),
            expected_generation=snapshot.generation,
            max_profiles=128,
            max_store_bytes=1,
        )


def test_imported_field_mutation_updates_digest_revision_and_keeps_marker(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    installed = store.install_profile_if_absent(
        "research",
        _imported_profile(),
        expected_generation=store.read_snapshot_strict().generation,
        max_profiles=128,
        max_store_bytes=8 * 1024 * 1024,
    )

    store.set_server_default(
        "local:docs",
        "ask",
        profile_id="research",
        expected_profile_digest=installed.policy_digest,
        expected_revision=installed.revision,
    )

    profile = store.load()["profiles"]["research"]
    lifecycle = profile["tool_pack_lifecycle"]
    assert lifecycle["revision"] == 2
    assert lifecycle["policy_digest"] == profile_policy_digest(profile)
    assert lifecycle["policy_digest"] != installed.policy_digest
    assert lifecycle["first_bind_confirmation_required"] is True


def test_profile_digest_cas_allows_unrelated_edit_but_rejects_same_profile(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    store.ensure_profile("other")
    default_digest = profile_policy_digest(store.load()["profiles"]["default"])

    store.set_server_default("local:other", "deny", profile_id="other")
    store.set_server_default(
        "local:one",
        "deny",
        expected_profile_digest=default_digest,
    )
    prior_digest = profile_policy_digest(store.load()["profiles"]["default"])
    store.set_server_default("local:two", "ask")

    with pytest.raises(ProfileMutationError, match="stale_profile"):
        store.set_server_default(
            "local:three",
            "deny",
            expected_profile_digest=prior_digest,
        )


def test_field_mutation_refuses_invalid_lifecycle_pair(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    payload = store.load()
    payload["profiles"]["broken"] = {
        "servers": {},
        "profile_kind": "tool_pack_imported",
    }
    store.save(payload)

    with pytest.raises(ProfileMutationError, match="lifecycle_invalid"):
        store.set_server_default(
            "local:docs",
            "deny",
            profile_id="broken",
        )


def test_low_level_save_rejects_stale_expected_generation(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    stale = store.read_snapshot_strict().generation
    store.set_kill_switch(True)
    replacement = store.load()
    replacement["kill_switch"] = False

    with pytest.raises(ProfileMutationError, match="stale_store"):
        store.save(replacement, expected_generation=stale)

    assert store.get_kill_switch() is True


def test_update_imported_profile_enforces_revision_and_preserves_marker(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    installed = store.install_profile_if_absent(
        "research",
        _imported_profile(first_bind_required=True),
        expected_generation=store.read_snapshot_strict().generation,
        max_profiles=128,
        max_store_bytes=8 * 1024 * 1024,
    )
    replacement = copy.deepcopy(store.load()["profiles"]["research"])
    replacement["servers"]["local:docs"]["default"] = "ask"
    replacement["tool_pack_lifecycle"]["first_bind_confirmation_required"] = False
    replacement["tool_pack_lifecycle"]["policy_digest"] = profile_policy_digest(
        replacement
    )

    updated = store.update_imported_profile(
        "research",
        replacement,
        expected_revision=installed.revision,
        max_store_bytes=8 * 1024 * 1024,
    )

    assert updated.revision == 2
    profile = store.load()["profiles"]["research"]
    assert profile["tool_pack_lifecycle"]["first_bind_confirmation_required"] is True
    assert profile["tool_pack_lifecycle"]["policy_digest"] == profile_policy_digest(
        profile
    )
    with pytest.raises(ProfileMutationError, match="stale_revision"):
        store.update_imported_profile(
            "research",
            replacement,
            expected_revision=installed.revision,
            max_store_bytes=8 * 1024 * 1024,
        )


def test_tombstone_replacement_contains_no_allow_or_ask_rows(tmp_path):
    store = MCPPermissionStore(tmp_path / "permissions.json")
    installed = store.install_profile_if_absent(
        "research",
        _imported_profile(),
        expected_generation=store.read_snapshot_strict().generation,
        max_profiles=128,
        max_store_bytes=8 * 1024 * 1024,
    )

    result = store.replace_profile_with_tombstone(
        "research",
        _tombstone_profile(),
        expected_revision=installed.revision,
        max_store_bytes=8 * 1024 * 1024,
    )

    profile = store.load()["profiles"]["research"]
    assert result.revision == 2
    assert profile["global_default"] == "deny"
    assert profile["servers"]["agent:builtin"]["default"] == "deny"
    states = [profile["global_default"]]
    for server in profile["servers"].values():
        if "default" in server:
            states.append(server["default"])
        states.extend(tool["state"] for tool in server.get("tools", {}).values())
    assert set(states) == {"deny"}


def test_lifecycle_coordinator_is_process_wide_and_counts_exact_profile_leases(
    monkeypatch,
):
    first = ToolProfileLifecycleCoordinator()
    second = ToolProfileLifecycleCoordinator()
    probe = _ContestedLockProbe()
    monkeypatch.setattr(binding_module, "_LIFECYCLE_LOCK", probe)

    def hold_first() -> None:
        with first.mutation():
            pass

    def enter_second() -> None:
        with second.mutation():
            pass

    first_thread = threading.Thread(target=hold_first, name="first-mutator")
    second_thread = threading.Thread(target=enter_second, name="second-mutator")
    first_thread.start()
    assert probe.first_acquired.wait(timeout=2)
    second_thread.start()
    assert probe.second_attempted.wait(timeout=2)
    probe.release_first.set()
    assert probe.second_acquired.wait(timeout=2)
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert probe.order == ["first_acquired", "first_released", "second_acquired"]

    with first.lease("research"):
        with second.lease("research"):
            with first.lease("other"):
                assert first.active_lease_count("research") == 2
                assert second.active_lease_count("other") == 1
    assert first.active_lease_count("research") == 0
