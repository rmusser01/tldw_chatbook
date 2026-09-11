"""Manifest/staging/auth tests for the artifact share web export."""

import json
import os
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Web_Server.artifact_share_manifest import (
    ArtifactShareAuth,
    ArtifactShareError,
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
    # staging tree is owner-only: freshly created root and share dir 0700
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(share_dir.stat().st_mode) == 0o700
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
    # bundle is staged 0600 throughout (never briefly 0644 pre-chmod)
    assert stat.S_IMODE((share_dir / "bundle.zip").stat().st_mode) == 0o600
    # no mkstemp leftovers remain in the staging directory
    assert not [p for p in share_dir.iterdir() if p.name.startswith(".bundle-")]
    # round-trip through the on-disk JSON
    loaded = load_manifest(manifest_path)
    assert loaded.share_name == "My Library"
    assert loaded.auth is not None and loaded.auth.username == "alice"
    assert {i.key for i in loaded.artifacts} == {i.key for i in manifest.artifacts}
    assert json.loads(manifest_path.read_text())["schema"] == 1


def test_load_manifest_enforces_schema_version(tmp_path):
    manifest = stage_share(
        [_record(tmp_path, "S", file_name="s.zip", cid=9)],
        share_name="x",
        auth=None,
        share_root=tmp_path / "r",
    )
    manifest_path = tmp_path / "r" / manifest.share_id / "manifest.json"

    # an explicit future/mismatched schema value is refused (fail closed)
    data = json.loads(manifest_path.read_text())
    data["schema"] = 999
    manifest_path.write_text(json.dumps(data))
    with pytest.raises(ArtifactShareError, match="schema"):
        load_manifest(manifest_path)

    # a missing "schema" key loads as the accepted default (version 1)
    data = json.loads(manifest_path.read_text())
    del data["schema"]
    manifest_path.write_text(json.dumps(data))
    loaded = load_manifest(manifest_path)
    assert loaded.schema_version == 1


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


def test_stage_share_sanitizes_untrusted_chatbook_ids(tmp_path):
    # Qodo #8: the chatbook_id is untrusted service data; traversal and
    # absolute-path ids must stage as plain, contained filenames.
    evil = _record(tmp_path, "Evil", file_name="evil.zip", cid=1)
    evil["chatbook_id"] = "../../evil"
    absolute = _record(tmp_path, "Abs", file_name="abs.zip", cid=2)
    absolute["chatbook_id"] = "/abs/path"
    manifest = stage_share(
        [evil, absolute], share_name="x", auth=None, share_root=tmp_path / "r"
    )
    share_dir = tmp_path / "r" / manifest.share_id
    assert len(manifest.artifacts) == 2
    for item in manifest.artifacts:
        assert "/" not in item.staged_name
        assert "\\" not in item.staged_name
        assert ".." not in item.staged_name
        staged = share_dir / item.staged_name
        assert staged.is_file()
        assert staged.read_bytes() == b"chatbook-zip-bytes"
    # nothing escaped the staging directory
    assert set(share_dir.iterdir()) == {
        *(share_dir / item.staged_name for item in manifest.artifacts),
        share_dir / "bundle.zip",
        share_dir / "manifest.json",
    }


def test_stage_share_rejects_symlinked_source(tmp_path):
    # Qodo #1: a swapped symlink must not launder an arbitrary file into a
    # share, even when it points at a real bundle.
    real = tmp_path / "real.zip"
    real.write_bytes(b"real-bundle")
    link = tmp_path / "link.zip"
    link.symlink_to(real)
    record = _record(tmp_path, "Linked", file_name=None, cid=5)
    record["file_path"] = str(link)
    with pytest.raises(ArtifactShareStagingError, match="symlink"):
        stage_share([record], share_name="x", auth=None, share_root=tmp_path / "r")
    # fail-closed: no share directory left behind
    assert not (tmp_path / "r").exists() or not any((tmp_path / "r").iterdir())


def test_auth_verify_roundtrip_and_reject():
    auth = build_share_auth("alice", "secret-pass")
    assert isinstance(auth, ArtifactShareAuth)
    assert verify_share_auth(auth, "alice", "secret-pass") is True
    assert verify_share_auth(auth, "alice", "wrong") is False
    assert verify_share_auth(auth, "bob", "secret-pass") is False


def test_sweep_removes_dead_pid_and_live_non_share_pid(tmp_path):
    # Qodo #12: liveness is two-step. A dead pid is swept as before, and a
    # LIVE pid that is not the share server no longer pins a stale
    # directory -- the ps command-line check catches PID reuse. The live
    # non-share pid is a real `sleep` child (real ps, deterministic: using
    # pytest's own pid would collide whenever the test argv happens to
    # contain the server module name, e.g. this suite's sibling file).
    import subprocess as _subprocess

    holder = _subprocess.Popen(["sleep", "30"])
    try:
        root = tmp_path / "share-root"
        dead = root / "deadpid"
        recycled = root / "recycledpid"
        for entry in (dead, recycled):
            entry.mkdir(parents=True)
            (entry / "manifest.json").write_text("{}")
        (dead / "status.json").write_text(json.dumps({"pid": 999999999}))
        (recycled / "status.json").write_text(json.dumps({"pid": holder.pid}))

        removed = sweep_stale_shares(root)

        assert removed == [dead, recycled]
        assert not dead.exists()
        assert not recycled.exists()
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_sweep_keeps_live_pid_confirmed_as_share_server(tmp_path, monkeypatch):
    # Positive half of Qodo #12: a live pid whose command line matches the
    # share server module keeps its directory (ps output monkeypatched; the
    # controller integration tests cover the real child-process case).
    import tldw_chatbook.Web_Server.artifact_share_manifest as manifest_module

    root = tmp_path / "share-root"
    active = root / "active"
    active.mkdir(parents=True)
    (active / "manifest.json").write_text("{}")
    (active / "status.json").write_text(json.dumps({"pid": os.getpid()}))
    monkeypatch.setattr(manifest_module, "_pid_is_share_server", lambda pid: True)

    removed = sweep_stale_shares(root)

    assert removed == []
    assert active.exists()


def test_pid_is_share_server_requires_share_server_command(monkeypatch):
    # The ps gate itself: a matching command line passes, an unrelated one
    # fails, and a ps failure fails open (keeps the pid-exists behavior).
    from subprocess import CompletedProcess

    import tldw_chatbook.Web_Server.artifact_share_manifest as manifest_module

    def _ps(stdout: str, returncode: int = 0):
        return lambda *_args, **_kwargs: CompletedProcess(
            args=[], returncode=returncode, stdout=stdout
        )

    monkeypatch.setattr(
        manifest_module.subprocess,
        "run",
        _ps("/usr/bin/python -m tldw_chatbook.Web_Server.artifact_share_server m.json"),
    )
    assert manifest_module._pid_is_share_server(123) is True

    monkeypatch.setattr(
        manifest_module.subprocess, "run", _ps("/usr/local/bin/pytest Tests/")
    )
    assert manifest_module._pid_is_share_server(123) is False

    def _broken(*_args, **_kwargs):
        raise OSError("no ps here")

    monkeypatch.setattr(manifest_module.subprocess, "run", _broken)
    assert manifest_module._pid_is_share_server(123) is True


def test_pid_alive_current_and_bogus():
    assert pid_alive(os.getpid()) is True
    assert pid_alive(999999999) is False


def test_share_root_dir_under_user_data():
    assert share_root_dir().name == "share"
    assert share_root_dir().parent.name != ""  # anchored under the user data dir
