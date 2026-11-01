"""Held observed-read races at the existing writer, including real worker dispatch."""

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Event

import pytest

from tldw_chatbook.Tools import local_tool_impls, patch_tool_impls, remote_worker_bundle
from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor
from tldw_chatbook.Tools.workspace_tool_protocol import WorkspaceToolResponse
from tldw_chatbook.Tools.workspace_tool_worker import run_workspace_worker

pytestmark = pytest.mark.bootstrap_profile


def _diff(path, old="old", new="new"):
    return f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-{old}\n+{new}\n"


def _worker(root, operation, arguments, backend="local"):
    request = WorkspaceToolExecutor(root)._build_request(
        operation, arguments, intent="write"
    )
    output = BytesIO()
    run = (
        run_workspace_worker
        if backend == "local"
        else remote_worker_bundle.run_workspace_worker
    )
    run(BytesIO(request.to_bytes()), output, BytesIO())
    return [
        WorkspaceToolResponse.from_bytes(
            frame, expected_operation_id=request.operation_id
        )
        for frame in output.getvalue().splitlines()
    ]


@pytest.mark.parametrize("backend", ["local", "bundle"])
@pytest.mark.parametrize("operation", ["fs_edit", "fs_patch"])
@pytest.mark.parametrize("peer_action", ["replace", "remove"])
def test_held_read_refuses_peer_change(
    tmp_path, monkeypatch, operation, peer_action, backend
):
    target = tmp_path / "a.txt"
    target.write_bytes(b"old\r\n")
    entered, release = Event(), Event()
    module = (
        (local_tool_impls if operation == "fs_edit" else patch_tool_impls)
        if backend == "local"
        else remote_worker_bundle
    )
    writer = module._atomic_write_target

    def held_writer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return writer(*args, **kwargs)

    monkeypatch.setattr(module, "_atomic_write_target", held_writer)
    arguments = (
        {"path": "a.txt", "old_string": "old", "new_string": "new"}
        if operation == "fs_edit"
        else {"diff": _diff("a.txt")}
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_worker, tmp_path, operation, arguments, backend)
        try:
            assert entered.wait(5)
            target.unlink()
            if peer_action == "replace":
                target.write_bytes(b"peer\r\n")
        finally:
            release.set()
        frames = future.result(5)
    assert [frame.outcome for frame in frames] == ["admitted", "failure"]
    assert frames[-1].code == "tool_failure"
    assert frames[-1].cleanup_proven
    assert "precondition failed" in frames[-1].error
    assert (
        target.read_bytes() == b"peer\r\n"
        if peer_action == "replace"
        else not target.exists()
    )


@pytest.mark.parametrize("backend", ["local", "bundle"])
def test_patch_create_race_preserves_peer(tmp_path, monkeypatch, backend):
    module = patch_tool_impls if backend == "local" else remote_worker_bundle
    writer = module._atomic_write_target

    def create_peer(target, *args, **kwargs):
        target.write_bytes(b"peer\n")
        return writer(target, *args, **kwargs)

    monkeypatch.setattr(module, "_atomic_write_target", create_peer)
    frames = _worker(
        tmp_path,
        "fs_patch",
        {"diff": "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+mine\n"},
        backend,
    )
    assert frames[-1].outcome == "failure"
    assert "target is present" in frames[-1].error
    assert (tmp_path / "new.txt").read_bytes() == b"peer\n"


@pytest.mark.parametrize("backend", ["local", "bundle"])
def test_patch_later_race_reports_prior_effects(tmp_path, monkeypatch, backend):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_bytes(b"old\n")
    module = patch_tool_impls if backend == "local" else remote_worker_bundle
    writer = module._atomic_write_target

    def change_second(target, *args, **kwargs):
        if target.name == "b.txt":
            target.write_bytes(b"peer\n")
        return writer(target, *args, **kwargs)

    monkeypatch.setattr(module, "_atomic_write_target", change_second)
    frames = _worker(
        tmp_path, "fs_patch", {"diff": _diff("a.txt") + _diff("b.txt")}, backend
    )
    assert frames[-1].outcome == "failure"
    assert frames[-1].cleanup_proven
    assert "b.txt" in frames[-1].error
    assert "patched a.txt" in frames[-1].error
    assert (tmp_path / "a.txt").read_bytes() == b"new\n"
    assert (tmp_path / "b.txt").read_bytes() == b"peer\n"


@pytest.mark.parametrize("operation", ["fs_edit", "fs_patch"])
def test_observed_digest_preserves_crlf_and_repeated_writes(tmp_path, operation):
    target = tmp_path / "a.txt"
    target.write_bytes(b"old\r\n")
    for old, new in (("old", "new"), ("new", "final")):
        arguments = (
            {"path": "a.txt", "old_string": old, "new_string": new}
            if operation == "fs_edit"
            else {"diff": _diff("a.txt", old, new)}
        )
        assert _worker(tmp_path, operation, arguments)[-1].outcome == "success"
    assert target.read_bytes() == b"final\r\n"


@pytest.mark.parametrize("operation", ["fs_edit", "fs_patch"])
def test_edit_patch_respects_held_target_lock(tmp_path, operation):
    portalocker = pytest.importorskip("portalocker")
    target = tmp_path / "a.txt"
    target.write_bytes(b"old\n")
    arguments = (
        {"path": "a.txt", "old_string": "old", "new_string": "new"}
        if operation == "fs_edit"
        else {"diff": _diff("a.txt")}
    )
    with target.open("r+b") as handle:
        portalocker.lock(
            handle, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
        )
        try:
            frames = _worker(tmp_path, operation, arguments)
        finally:
            portalocker.unlock(handle)
    assert frames[-1].outcome == "failure"
    assert "target is being modified" in frames[-1].error
    assert "digest changed" not in frames[-1].error
    assert target.read_bytes() == b"old\n"


@pytest.mark.parametrize(
    ("backend", "entrypoint"),
    [("local", "public"), ("local", "pinned"), ("bundle", "pinned")],
)
@pytest.mark.parametrize("failure", ["permission", "missing"])
def test_patch_later_read_io_failure_retains_progress(
    tmp_path, monkeypatch, backend, entrypoint, failure
):
    import builtins
    import errno

    from tldw_chatbook.Tools.workspace_root_pin import pin_workspace_root
    from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain

    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_bytes(b"old\n")
    module = patch_tool_impls if backend == "local" else remote_worker_bundle
    error = (
        PermissionError(errno.EACCES, "second target unreadable", "b.txt")
        if failure == "permission"
        else FileNotFoundError(errno.ENOENT, "second target disappeared", "b.txt")
    )

    def fail_second_read(path, *args, **kwargs):
        if str(path).endswith("b.txt"):
            if failure == "missing":
                (tmp_path / "b.txt").unlink()
            raise error
        return builtins.open(path, *args, **kwargs)

    monkeypatch.setattr(module, "open", fail_second_read, raising=False)
    diff = _diff("a.txt") + _diff("b.txt")
    with pytest.raises(module.LocalToolError) as caught:
        if entrypoint == "public":
            module.patch_files(diff, workspace_root=tmp_path)
        else:
            chain = capture_directory_chain(tmp_path)
            with pin_workspace_root(chain.canonical_root, chain) as root:
                module.patch_validated_files(
                    module.parse_patch_targets(diff), root=root
                )
    assert caught.value.__cause__ is error
    assert caught.value.__cause__.errno == error.errno
    assert "b.txt" in str(caught.value)
    assert "earlier effects retained: patched a.txt" in str(caught.value)
    assert (tmp_path / "a.txt").read_bytes() == b"new\n"
    if failure == "permission":
        assert (tmp_path / "b.txt").read_bytes() == b"old\n"
    else:
        assert not (tmp_path / "b.txt").exists()


@pytest.mark.parametrize("backend", ["local", "bundle"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_patch_worker_read_io_failure_reports_only_real_prior_effects(
    tmp_path, monkeypatch, backend, dry_run
):
    import builtins
    import errno

    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_bytes(b"old\n")
    module = patch_tool_impls if backend == "local" else remote_worker_bundle

    def fail_second_read(path, *args, **kwargs):
        if str(path).endswith("b.txt"):
            raise PermissionError(errno.EACCES, "second target unreadable", "b.txt")
        return builtins.open(path, *args, **kwargs)

    monkeypatch.setattr(module, "open", fail_second_read, raising=False)
    frames = _worker(
        tmp_path,
        "fs_patch",
        {"diff": _diff("a.txt") + _diff("b.txt"), "dry_run": dry_run},
        backend,
    )
    assert [frame.outcome for frame in frames] == ["admitted", "failure"]
    assert frames[-1].code == "tool_failure"
    assert frames[-1].cleanup_proven
    assert "b.txt" in frames[-1].error
    assert "second target unreadable" in frames[-1].error
    if dry_run:
        assert "earlier effects retained" not in frames[-1].error
        assert (tmp_path / "a.txt").read_bytes() == b"old\n"
    else:
        assert "earlier effects retained: patched a.txt" in frames[-1].error
        assert (tmp_path / "a.txt").read_bytes() == b"new\n"
    assert (tmp_path / "b.txt").read_bytes() == b"old\n"
