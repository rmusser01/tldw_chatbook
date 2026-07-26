"""Run-bound allowed file roots (spec 2026-07-26 settings-workspaces §3)."""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces import LocalWorkspaceRegistryService
from tldw_chatbook.Tools import workspace_file_roots as wfr


def _registry(tmp_path: Path) -> LocalWorkspaceRegistryService:
    registry = LocalWorkspaceRegistryService(
        WorkspaceDB(tmp_path / "ws.sqlite", client_id="roots-tests")
    )
    registry.ensure_default_workspace()
    registry.create_workspace(workspace_id="ws-a", name="Client A")
    registry.create_workspace(workspace_id="ws-b", name="Client B")
    return registry


def test_roots_follow_run_workspace_not_active(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    folder_a = tmp_path / "a"
    folder_a.mkdir()
    folder_b = tmp_path / "b"
    folder_b.mkdir()
    registry.add_folder_binding("ws-a", folder_a)
    registry.add_folder_binding("ws-b", folder_b)
    registry.set_active_workspace("ws-b")
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with wfr.run_workspace("ws-a"):
        roots = wfr.allowed_file_roots(write=False, sandbox_root=sandbox)
    assert roots == (sandbox, folder_a.resolve())

    # Outside a run: falls back to the ACTIVE workspace (ws-b).
    roots = wfr.allowed_file_roots(write=False, sandbox_root=sandbox)
    assert roots == (sandbox, folder_b.resolve())


def test_write_roots_require_rw_and_existing_dirs(tmp_path, monkeypatch) -> None:
    registry = _registry(tmp_path)
    ro_folder = tmp_path / "ro"
    ro_folder.mkdir()
    rw_folder = tmp_path / "rw"
    rw_folder.mkdir()
    gone = tmp_path / "gone"
    gone.mkdir()
    registry.add_folder_binding("ws-a", ro_folder)
    registry.add_folder_binding("ws-a", rw_folder, allow_write=True)
    registry.add_folder_binding("ws-a", gone, allow_write=True)
    gone.rmdir()  # deleted after binding: must drop out at call time
    monkeypatch.setattr(wfr, "_registry_factory", lambda: registry)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()

    with wfr.run_workspace("ws-a"):
        read_roots = wfr.allowed_file_roots(write=False, sandbox_root=sandbox)
        write_roots = wfr.allowed_file_roots(write=True, sandbox_root=sandbox)

    assert ro_folder.resolve() in read_roots
    assert write_roots == (sandbox, rw_folder.resolve())


def test_registry_failure_degrades_to_sandbox_only(tmp_path, monkeypatch) -> None:
    def _boom():
        raise RuntimeError("registry down")

    monkeypatch.setattr(wfr, "_registry_factory", _boom)
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    with wfr.run_workspace("ws-a"):
        assert wfr.allowed_file_roots(write=True, sandbox_root=sandbox) == (sandbox,)
