"""Controller lifecycle tests for artifact share sessions."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.Web_Server.artifact_share import (
    ArtifactShareController,
    ShareStatus,
    compute_display_urls,
)
from tldw_chatbook.Web_Server.artifact_share_manifest import (
    ArtifactShareError,
    share_root_dir,
)

pytestmark = [pytest.mark.unit, pytest.mark.loopback_network]


def _record(tmp_path: Path, cid: int = 1) -> dict:
    bundle = tmp_path / f"src-{cid}.zip"
    bundle.write_bytes(f"bundle-{cid}".encode())
    return {
        "id": str(cid),
        "chatbook_id": cid,
        "name": f"Artifact {cid}",
        "description": "",
        "file_path": str(bundle),
    }


@pytest.fixture
def isolated_share_root(tmp_path, monkeypatch):
    root = tmp_path / "share"
    monkeypatch.setattr(
        "tldw_chatbook.Web_Server.artifact_share.share_root_dir", lambda: root
    )
    monkeypatch.setattr(
        "tldw_chatbook.Web_Server.artifact_share_manifest.share_root_dir", lambda: root
    )
    return root


def test_compute_display_urls_loopback_and_lan():
    assert compute_display_urls("127.0.0.1", 8000) == ["http://127.0.0.1:8000"]
    lan = compute_display_urls("0.0.0.0", 8000)
    assert lan[0] == "http://127.0.0.1:8000"
    import re

    assert re.fullmatch(r"http://\d{1,3}(\.\d{1,3}){3}:8000", lan[1])


def test_start_and_stop_share_lifecycle(tmp_path, isolated_share_root):
    controller = ArtifactShareController()
    statuses = []
    controller._status_callback = statuses.append

    status = controller.start_share(
        records=[_record(tmp_path, 1), _record(tmp_path, 2)],
        share_name="Field kit",
    )

    assert isinstance(status, ShareStatus)
    assert status.artifact_count == 2
    assert status.urls and status.urls[0].startswith("http://127.0.0.1:")
    share_dir = status.share_dir
    assert (share_dir / "manifest.json").is_file()
    assert controller.status == status

    controller.stop_share()

    assert controller.status is None
    assert statuses[-1] is None
    assert not share_dir.exists()  # staging cleaned
    import httpx

    with pytest.raises(httpx.ConnectError):
        httpx.get(status.urls[0], timeout=2)


def test_single_active_share_second_start_replaces_first(tmp_path, isolated_share_root):
    controller = ArtifactShareController()
    first = controller.start_share(records=[_record(tmp_path, 1)], share_name="one")
    second = controller.start_share(records=[_record(tmp_path, 2)], share_name="two")
    assert not first.share_dir.exists()
    assert controller.status == second
    controller.stop_share()


def test_start_share_without_web_deps_is_clean_error(tmp_path, isolated_share_root, monkeypatch):
    import tldw_chatbook.Web_Server.artifact_share as controller_module

    monkeypatch.setattr(controller_module, "is_web_server_available", lambda: False)
    controller = ArtifactShareController()
    with pytest.raises(ArtifactShareError, match=r"tldw_chatbook\[web\]"):
        controller.start_share(records=[_record(tmp_path, 1)], share_name="x")


def test_start_share_popen_failure_cleans_staging(tmp_path, isolated_share_root, monkeypatch):
    # Qodo #9: a spawn failure must surface as ArtifactShareError and must
    # not strand the staged directory.
    import subprocess

    def _boom(*_args, **_kwargs):
        raise OSError("fork failed")

    monkeypatch.setattr(subprocess, "Popen", _boom)
    controller = ArtifactShareController()
    with pytest.raises(ArtifactShareError, match="Could not start share server"):
        controller.start_share(records=[_record(tmp_path, 1)], share_name="x")
    assert not any(isolated_share_root.iterdir())


def test_start_share_rejects_colon_username(tmp_path, isolated_share_root):
    # Qodo #14 (defensive twin of the dialog check): Basic auth splits on
    # the first ':', so a colon username can never authenticate.
    controller = ArtifactShareController()
    with pytest.raises(ArtifactShareError, match="':'"):
        controller.start_share(
            records=[_record(tmp_path, 1)],
            share_name="x",
            username="alice:admin",
            password="secret-pass",
        )
    # rejected before staging: nothing laid down
    assert not isolated_share_root.exists() or not any(isolated_share_root.iterdir())


def test_startup_sweep_removes_stale_dirs(tmp_path, isolated_share_root):
    stale = isolated_share_root / "deadbeef"
    stale.mkdir(parents=True)
    (stale / "manifest.json").write_text("{}")
    controller = ArtifactShareController()
    removed = controller.startup_sweep()
    assert removed == [stale]
    assert not stale.exists()
