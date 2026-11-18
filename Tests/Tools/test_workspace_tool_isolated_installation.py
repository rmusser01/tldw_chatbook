"""The real no-site Console interpreter must launch its original file worker."""

import json
import os
from pathlib import Path
import subprocess
import sys

from Tests.private_profile import private_profile_test


@private_profile_test
def test_no_site_backend_executes_fixed_file_worker_from_same_installation(tmp_path, request):
    from Tests.Tools.test_workspace_tool_executor import _isolated_harness_environment

    repository = Path(__file__).resolve().parents[2]
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("original isolated worker\n", encoding="utf-8")
    (tmp_path / "outside.txt").write_text("outside must remain unread", encoding="utf-8")
    hostile = tmp_path / "hostile"
    shadow = hostile / "tldw_chatbook"
    shadow.mkdir(parents=True)
    marker = hostile / "imported"
    (shadow / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('unexpected import')\n",
        encoding="utf-8",
    )
    (hostile / "portalocker.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('unexpected lock import')\n",
        encoding="utf-8",
    )
    roots = [str(repository)]
    core = repository / "packages" / "tldw_profile_core" / "src"
    if (core / "tldw_profile_core" / "__init__.py").is_file():
        roots.append(str(core))
    # Match ConsoleExecutionClient._launch: pywin32 installs actual import
    # roots below site-packages, which the no-site parent must also receive.
    package_roots = tuple(
        Path(path).resolve()
        for path in sys.path
        if path and Path(path).name == "site-packages"
    )
    roots.extend(
        str(Path(path).resolve())
        for path in sys.path
        if path and any(Path(path).resolve().is_relative_to(item) for item in package_roots)
    )
    probe = "import sys\nsys.path[:0]=" + repr(roots) + "\n" + r'''
import hashlib
import json
from pathlib import Path
import portalocker
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Tools import workspace_tool_executor as source

assert sys.flags.isolated and sys.flags.no_site
expected = Path(sys.argv[1]).resolve()
assert Path(source.__file__).resolve() == expected / "tldw_chatbook/Tools/workspace_tool_executor.py"
workspace = Path(sys.argv[2])
original_bytes = (workspace / "note.txt").read_bytes()
assert original_bytes in (b"original isolated worker\n", b"original isolated worker\r\n")
executor = source.WorkspaceToolExecutor(workspace)
argv = source._workspace_worker_argv(workspace, executor._build_request(
    "fs_read", {"path": "note.txt"}, intent="read",
))
assert argv[:5] == [sys.executable, "-I", "-S", "-B", "-c"]
stage = "fs_list"
try:
    listed = executor.execute("fs_list", {"path": "."}, intent="read")
    stage = "fs_read"
    read = executor.execute("fs_read", {"path": "note.txt"}, intent="read")
    stage = "fs_edit"
    edited = executor.execute("fs_edit", {
        "path": "note.txt", "old_string": "original isolated worker",
        "new_string": "updated isolated worker",
    }, intent="write")
    # The original Windows write_text seed has CRLF; fs_edit must preserve it.
    expected_bytes = original_bytes.replace(b"original", b"updated")
    assert "made 1 replacement" in edited
    assert f"\nsha256: {hashlib.sha256(expected_bytes).hexdigest()}\nsize: {len(expected_bytes)}" in edited
    assert (workspace / "note.txt").read_bytes() == expected_bytes
except source.WorkspaceToolExecutionError as error:
    reason = next((code for prefix, code in (
        ("file changed while opening:", "read_open_metadata_changed"),
        ("file changed while reading:", "read_final_metadata_changed"),
        ("file changed or is not regular:", "read_not_regular"),
        ("not a directory:", "list_not_directory"),
    ) if str(error).startswith(prefix)), "other_validated_failure")
    print(json.dumps({"worker_stage": stage, "worker_error_code": error.code,
                      "worker_fixed_reason": reason}))
    raise SystemExit(1) from None
try:
    executor.execute("fs_read", {"path": "../outside.txt"}, intent="read")
except source.WorkspaceToolExecutionError as error:
    assert error.code in {"invalid_request", "tool_failure"}
else:
    raise AssertionError("Original path refusal was lost")
stale_digest = hashlib.sha256(original_bytes).hexdigest()
try:
    executor.execute("fs_write", {"path": "note.txt", "content": "lost",
        "expected_sha256": stale_digest}, intent="write")
except source.WorkspaceToolExecutionError as error:
    assert error.code == "tool_failure" and "target digest changed" in str(error)
else:
    raise AssertionError("Original stale digest refusal was lost")
current_digest = hashlib.sha256(expected_bytes).hexdigest()
with (workspace / "note.txt").open("r+b", buffering=0) as handle:
    # Shared locking permits the child's original read, but its unchanged
    # exclusive nonblocking CAS lock must refuse this real competing owner.
    portalocker.lock(handle, portalocker.LOCK_SH | portalocker.LOCK_NB)
    try:
        executor.execute("fs_write", {"path": "note.txt", "content": "lost",
            "expected_sha256": current_digest}, intent="write")
    except source.WorkspaceToolExecutionError as error:
        assert error.code == "tool_failure" and "target is being modified" in str(error)
    else:
        raise AssertionError("Original competing-lock refusal was lost")
    finally:
        portalocker.unlock(handle)
assert (workspace / "note.txt").read_bytes() == expected_bytes
assert not list(workspace.glob(".chatbook-write-*.tmp"))
print(json.dumps({"listed": "note.txt" in listed,
                  "read": "original isolated worker" in read,
                  "edited": True, "stale_digest_refused": True,
                  "contending_lock_refused": True, "outside_refused": True}))
'''
    environment = _isolated_harness_environment()
    environment.update({name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR") if name in os.environ})
    environment["PYTHONPATH"] = str(hostile)
    completed = subprocess.run(
        [str(Path(getattr(sys, "_base_executable", sys.executable)).resolve()),
         "-I", "-S", "-B", "-c", probe, str(repository), str(workspace)],
        cwd=hostile,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout[-1000:]
    assert json.loads(completed.stdout.splitlines()[-1]) == {
        "listed": True, "read": True, "edited": True,
        "stale_digest_refused": True, "contending_lock_refused": True,
        "outside_refused": True,
    }
    assert not marker.exists()
