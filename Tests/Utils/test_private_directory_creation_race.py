"""TASK-32039: a competing mkdir must not disable private storage."""

import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_chatbook.Utils import private_paths

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX directory guards")


def synchronize_missing_directory(monkeypatch, component):
    """Let both real opens fail before either caller can attempt mkdir."""
    original = private_paths._open_directory_component
    missing = threading.Barrier(2, timeout=5)

    def open_component(parent_fd, name):
        try:
            return original(parent_fd, name)
        except FileNotFoundError:
            if name == component:
                missing.wait()
            raise

    monkeypatch.setattr(private_paths, "_open_directory_component", open_component)


@pytest.mark.parametrize("nested", [False, True])
def test_concurrent_private_directory_creators_both_succeed(
    tmp_path, monkeypatch, nested
):
    synchronize_missing_directory(monkeypatch, "shared")
    target = tmp_path / "shared"
    if nested:
        target /= "logs"

    def create():
        return private_paths.secure_private_directory(
            target, create=True, application_owned=True
        )

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(create) for _ in range(2)]
        results = [future.result(timeout=5) for future in futures]

    assert all(result.verified_private for result in results)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert target.stat().st_uid == os.geteuid()


@pytest.mark.parametrize("replacement", ["file", "symlink", "unsafe_parent"])
def test_competing_non_directory_is_rejected_without_changing_target(
    tmp_path, monkeypatch, replacement
):
    selected = tmp_path / "selected"
    other = tmp_path / "other"
    other.mkdir(mode=0o755)
    original_mode = stat.S_IMODE(other.stat().st_mode)
    original = private_paths._open_directory_component

    def open_component(parent_fd, name):
        try:
            return original(parent_fd, name)
        except FileNotFoundError:
            if name == "selected":
                if replacement == "file":
                    selected.write_text("keep")
                elif replacement == "symlink":
                    selected.symlink_to(other, target_is_directory=True)
                else:
                    selected.mkdir()
                    selected.chmod(0o777)
            raise

    monkeypatch.setattr(private_paths, "_open_directory_component", open_component)
    with pytest.raises(private_paths.PrivatePathError):
        private_paths.secure_private_directory(
            selected / "nested" if replacement == "unsafe_parent" else selected,
            create=True,
            application_owned=True,
        )

    assert stat.S_IMODE(other.stat().st_mode) == original_mode
    if replacement == "file":
        assert selected.read_text() == "keep"
    elif replacement == "symlink":
        assert selected.is_symlink()
    else:
        assert stat.S_IMODE(selected.stat().st_mode) == 0o777
        assert not (selected / "nested").exists()


def test_concurrent_first_run_writers_both_persist_records(tmp_path, monkeypatch):
    from tldw_chatbook import config
    from tldw_chatbook.Agents.run_log import RunLogWriter
    from tldw_chatbook.Agents.run_log_format import iter_records
    from tldw_chatbook.Tools import workspace_file_roots

    def setting(section, key, default=None):
        return {
            ("paths", "data_dir"): str(tmp_path),
            ("general", "users_name"): "racing-user",
        }.get((section, key), default)

    monkeypatch.setattr(config, "get_cli_setting", setting)
    monkeypatch.setattr(
        workspace_file_roots,
        "allowed_file_roots",
        lambda *, write, sandbox_root: (sandbox_root,),
    )
    synchronize_missing_directory(monkeypatch, "racing-user")

    def write(run_id):
        writer = RunLogWriter()
        writer.bind(run_id)
        writer.append(run_id=run_id, kind="primary", type="model", content=run_id)
        return writer

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(write, run_id) for run_id in ("first", "second")]
        writers = [future.result(timeout=5) for future in futures]

    assert all(writer.is_active for writer in writers)
    for writer, run_id in zip(writers, ("first", "second"), strict=True):
        records = list(iter_records((writer.log_dir / "logs.0001.txt").read_bytes()))
        assert [(record.run_id, record.content) for record in records] == [
            (run_id, run_id)
        ]
