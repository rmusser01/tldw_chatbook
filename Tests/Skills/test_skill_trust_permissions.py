from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Skills_Interop import skill_trust_store as trust_store_module
from tldw_chatbook.Skills_Interop.skill_trust_crypto import derive_skill_trust_keys
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX mode-bit assertions")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _manifest(generation: int) -> dict:
    return {
        "version": 1,
        "generation": generation,
        "skills": {},
        "audit": [],
    }


def _store(tmp_path: Path):
    trust_dir = tmp_path / "trust"
    marker_path = trust_dir / "generation_marker.json"
    marker = FileSkillTrustGenerationMarkerStore(
        marker_path=marker_path,
        store_dir=trust_dir,
    )
    return (
        SkillTrustStore(store_dir=trust_dir, marker_store=marker),
        marker_path,
    )


def test_snapshot_first_secures_store_snapshot_manifest_and_marker(
    tmp_path, monkeypatch
):
    store, marker_path = _store(tmp_path)
    keys = derive_skill_trust_keys("passphrase", salt=b"p" * 32)
    store.store_dir.mkdir()
    store.store_dir.chmod(0o755)
    observed_store_modes: list[int] = []
    original_replace = Path.replace

    def inspect_then_replace(self, other):
        observed_store_modes.append(_mode(store.store_dir))
        return original_replace(self, other)

    monkeypatch.setattr(Path, "replace", inspect_then_replace)

    store.save_snapshot(
        "demo-1",
        {"files": {"SKILL.md": "# Demo"}},
        keys,
        generation=1,
    )

    assert observed_store_modes
    assert observed_store_modes[0] == 0o700

    snapshot_path = store.snapshots_dir / "demo-1.json"
    assert _mode(store.store_dir) == 0o700
    assert _mode(store.snapshots_dir) == 0o700
    assert _mode(snapshot_path) == 0o600

    store.save_manifest(_manifest(1), keys, salt=b"p" * 32)

    assert _mode(store.manifest_path) == 0o600
    assert _mode(marker_path) == 0o600


def test_trust_temp_is_owner_only_before_replace(tmp_path, monkeypatch):
    store, _ = _store(tmp_path)
    keys = derive_skill_trust_keys("passphrase", salt=b"q" * 32)
    observed: list[tuple[str, int]] = []
    observed_owner_only: list[bool] = []
    original_replace = Path.replace
    original_replace_atomically = trust_store_module.replace_atomically

    def inspect_atomic_call(temp_path, target_path, write_fn, *, owner_only=False):
        observed_owner_only.append(owner_only)
        return original_replace_atomically(
            temp_path,
            target_path,
            write_fn,
            owner_only=owner_only,
        )

    def inspect_then_replace(self, other):
        observed.append((self.name, _mode(self)))
        return original_replace(self, other)

    monkeypatch.setattr(trust_store_module, "replace_atomically", inspect_atomic_call)
    monkeypatch.setattr(Path, "replace", inspect_then_replace)
    store.save_manifest(_manifest(1), keys, salt=b"q" * 32)

    assert observed
    assert observed_owner_only
    assert all(observed_owner_only)
    assert all(name.startswith(".") for name, _ in observed)
    assert all(mode == 0o600 for _, mode in observed)


def test_trust_bytes_writer_requests_owner_only_creation(tmp_path, monkeypatch):
    target = tmp_path / "trust" / "manifest-rollback.json"
    observed_owner_only: list[bool] = []
    original_replace_atomically = trust_store_module.replace_atomically

    def inspect_atomic_call(temp_path, target_path, write_fn, *, owner_only=False):
        observed_owner_only.append(owner_only)
        return original_replace_atomically(
            temp_path,
            target_path,
            write_fn,
            owner_only=owner_only,
        )

    monkeypatch.setattr(trust_store_module, "replace_atomically", inspect_atomic_call)
    trust_store_module._atomic_write_bytes(
        target,
        b"previous manifest bytes",
        base_dir=target.parent,
    )

    assert observed_owner_only == [True]
    assert _mode(target) == 0o600


def test_next_write_tightens_legacy_files_and_directories(tmp_path):
    store, marker_path = _store(tmp_path)
    keys = derive_skill_trust_keys("passphrase", salt=b"r" * 32)
    snapshot_path = store.snapshots_dir / "demo-1.json"

    store.save_snapshot("demo-1", {"files": {}}, keys, generation=1)
    store.save_manifest(_manifest(1), keys, salt=b"r" * 32)

    for directory in (store.store_dir, store.snapshots_dir):
        directory.chmod(0o755)
    for path in (snapshot_path, store.manifest_path, marker_path):
        path.chmod(0o666)

    store.save_snapshot("demo-1", {"files": {}}, keys, generation=2)
    store.save_manifest(_manifest(2), keys)

    assert _mode(store.store_dir) == 0o700
    assert _mode(store.snapshots_dir) == 0o700
    assert _mode(snapshot_path) == 0o600
    assert _mode(store.manifest_path) == 0o600
    assert _mode(marker_path) == 0o600
