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
    roots = [str(repository)]
    core = repository / "packages" / "tldw_profile_core" / "src"
    if (core / "tldw_profile_core" / "__init__.py").is_file():
        roots.append(str(core))
    roots.extend(str(Path(path).resolve()) for path in sys.path if path and Path(path).name == "site-packages")
    probe = "import sys\nsys.path[:0]=" + repr(roots) + "\n" + r'''
import json
from pathlib import Path
from Tests import network_guard
network_guard.install()
from tldw_chatbook.Tools import workspace_tool_executor as source

assert sys.flags.isolated and sys.flags.no_site
expected = Path(sys.argv[1]).resolve()
assert Path(source.__file__).resolve() == expected / "tldw_chatbook/Tools/workspace_tool_executor.py"
executor = source.WorkspaceToolExecutor(Path(sys.argv[2]))
argv = source._workspace_worker_argv()
assert argv[:5] == [sys.executable, "-I", "-S", "-B", "-c"]
try:
    listed = executor.execute("fs_list", {"path": "."}, intent="read")
    read = executor.execute("fs_read", {"path": "note.txt"}, intent="read")
except source.WorkspaceToolExecutionError as error:
    print(json.dumps({"worker_error_code": error.code}))
    raise SystemExit(1) from None
try:
    executor.execute("fs_read", {"path": "../outside.txt"}, intent="read")
except source.WorkspaceToolExecutionError as error:
    assert error.code in {"invalid_request", "tool_failure"}
else:
    raise AssertionError("Original path refusal was lost")
print(json.dumps({"listed": "note.txt" in listed,
                  "read": "original isolated worker" in read,
                  "outside_refused": True}))
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
        "listed": True, "read": True, "outside_refused": True,
    }
    assert not marker.exists()
