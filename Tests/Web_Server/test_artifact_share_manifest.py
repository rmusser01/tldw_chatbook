"""Manifest/staging/auth tests for the artifact share web export."""

import json
import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Web_Server.artifact_share_manifest import (
    ArtifactShareAuth,
    ArtifactShareStagingError,
    build_share_auth,
    load_manifest,
    pid_alive,
    share_root_dir,
    stage_share,
    sweep_stale_shares,
    verify_share_auth,
)

pytestmark = pytest.mark.unit


def _make_zip(path: Path, payload: bytes = b"chatbook-zip-bytes") -> Path:
    path.write_bytes(payload)
    return path


def _record(tmp_path: Path, name: str, *, file_name: str | None = None, cid: int = 1) -> dict:
    file_path = None if file_name is None else str(_make_zip(tmp_path / file_name))
    return {
        "id": str(cid),
        "chatbook_id": cid,
        "name": name,
        "description": f"desc for {name}",
        "file_path": file_path,
        "created_at": "2026-09-05T00:00:00+00:00",
        "updated_at": "2026-09-05T00:00:00+00:00",
    }


def test_stage_share_copies_and_hashes_and_writes_manifest_0600(tmp_path):
    root = tmp_path / "share-root"
    records = [
        _record(tmp_path, "Weekly Report", file_name="a.zip", cid=1),
        _record(tmp_path, "Notes <b>bold</b>", file_name="b.zip", cid=2),
    ]
    auth = build_share_auth("alice", "secret-pass")

    manifest = stage_share(
        records, share_name="My Library", auth=auth, share_root=root
    )

    share_dir = root / manifest.share_id
    manifest_path = share_dir / "manifest.json"
    assert manifest_path.is_file()
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert len(manifest.artifacts) == 2
    for item in manifest.artifacts:
        staged = share_dir / item.staged_name
        assert staged.is_file()
        assert stat.S_IMODE(staged.stat().st_mode) == 0o600
        assert staged.read_bytes() in (b"chatbook-zip-bytes",)
        assert item.size_bytes == staged.stat().st_size
    # user-controlled names never become raw filenames
    assert all("<" not in item.staged_name for item in manifest.artifacts)
    # bundle.zip exists and contains both staged files
    import zipfile

    with zipfile.ZipFile(share_dir / "bundle.zip") as bundle:
        assert sorted(bundle.namelist()) == sorted(
            item.staged_name for item in manifest.artifacts
        )
    # round-trip through the on-disk JSON
    loaded = load_manifest(manifest_path)
    assert loaded.share_name == "My Library"
    assert loaded.auth is not None and loaded.auth.username == "alice"
    assert {i.key for i in loaded.artifacts} == {i.key for i in manifest.artifacts}
    assert json.loads(manifest_path.read_text())["schema"] == 1


def test_stage_share_rejects_record_without_bundle(tmp_path):
    records = [_record(tmp_path, "Ghost", file_name=None, cid=3)]
    with pytest.raises(ArtifactShareStagingError, match="Ghost"):
        stage_share(records, share_name="x", auth=None, share_root=tmp_path / "r")
    # fail-closed: no share directory left behind
    assert not (tmp_path / "r").exists() or not any((tmp_path / "r").iterdir())


def test_stage_share_rejects_missing_file(tmp_path):
    records = [_record(tmp_path, "Vanished", file_name="gone.zip", cid=4)]
    Path(tmp_path / "gone.zip").unlink(missing_ok=True)
    with pytest.raises(ArtifactShareStagingError, match="Vanished"):
        stage_share(records, share_name="x", auth=None, share_root=tmp_path / "r")


def test_auth_verify_roundtrip_and_reject():
    auth = build_share_auth("alice", "secret-pass")
    assert isinstance(auth, ArtifactShareAuth)
    assert verify_share_auth(auth, "alice", "secret-pass") is True
    assert verify_share_auth(auth, "alice", "wrong") is False
    assert verify_share_auth(auth, "bob", "secret-pass") is False


def test_sweep_removes_dead_pid_and_keeps_live(tmp_path):
    root = tmp_path / "share-root"
    dead = root / "deadbeef"
    live = root / "cafebabe"
    for entry in (dead, live):
        entry.mkdir(parents=True)
        (entry / "manifest.json").write_text("{}")
    (dead / "status.json").write_text(json.dumps({"pid": 999999999}))
    (live / "status.json").write_text(json.dumps({"pid": os.getpid()}))

    removed = sweep_stale_shares(root)

    assert removed == [dead]
    assert not dead.exists()
    assert live.exists()


def test_pid_alive_current_and_bogus():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(999999999) is False


def test_share_root_dir_under_user_data():
    assert share_root_dir().name == "share"
    assert share_root_dir().parent.name != ""  # anchored under the user data dir
