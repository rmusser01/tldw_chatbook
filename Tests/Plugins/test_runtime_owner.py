"""Real competing processes and persistent unresolved launch evidence."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from Tests.hooks_v2_process_support import child_argv

pytestmark = pytest.mark.bootstrap_profile

CHILD = """
import json, os, sys
from pathlib import Path
import tldw_chatbook
from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner
assert Path(tldw_chatbook.__file__).resolve().parents[1] == Path(sys.argv[2])
owner = PluginRuntimeOwner(Path(sys.argv[1]))
acquired = owner.try_acquire()
if sys.argv[3] == 'contend':
    print(json.dumps({'acquired': acquired}), flush=True)
    owner.close()
else:
    assert acquired
    token = owner.reserve_launch('operation', 'installed', 'workspace', 'revision')
    if sys.argv[3] == 'published':
        owner.publish_process(token, {'pid': os.getpid(), 'start_identity': 'old-instance', 'group': os.getpgrp()})
    print(token, flush=True)
    os._exit(0)
"""


def child(root, phase):
    result = subprocess.run(
        child_argv(CHILD) + [str(root), str(Path.cwd()), phase],
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_independent_process_contention_and_reacquisition(tmp_path):
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        assert json.loads(child(tmp_path, "contend")) == {"acquired": False}
    finally:
        owner.close()
    assert json.loads(child(tmp_path, "contend")) == {"acquired": True}


@pytest.mark.parametrize("phase", ["pending", "published"])
def test_owner_death_retains_records_and_blocks_same_installation(tmp_path, phase):
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    token = child(tmp_path, phase)
    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        rows = owner.list_processes(limit=50, offset=0)
        assert len(rows) == 1
        assert rows[0]["token"] == token
        assert rows[0]["state"] == phase
        assert rows[0]["installation_id"] == "installed"
        with pytest.raises(PermissionError):
            owner.reserve_launch("again", "installed", None, "revision")
        other = owner.reserve_launch("other", "unrelated", None, "revision")
        owner.settle_process(other, confirmed=True)
        owner.settle_process(token, confirmed=False)
        assert owner.list_processes(limit=50, offset=0)[0]["state"] in (
            "unresolved",
            "settled",
        )
        with pytest.raises(PermissionError):
            owner.reserve_launch("again", "installed", None, "revision")
        owner.settle_process(token, confirmed=True)
        assert owner.reserve_launch("fresh", "installed", None, "revision")
    finally:
        owner.close()


def test_exact_provenance_pid_reuse_and_lease_kinds(tmp_path):
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path)
    assert owner.try_acquire()
    try:
        token = owner.reserve_launch("op", "installed", None, "digest")
        provenance = {
            "pid": os.getpid(),
            "start_identity": "different-from-live-pid",
            "argv_digest": "abc",
        }
        owner.publish_process(token, provenance)
        assert owner.list_processes(limit=1, offset=0)[0]["provenance"] == provenance
        with pytest.raises(ValueError):
            owner.publish_process(token, {"pid": 99999})
        assert owner.active_revision_leases("installed", "digest") == 1
        owner.set_process_kind(token, "idle_connection")
        assert owner.active_revision_leases("installed", "digest") == 0
        owner.set_process_kind(token, "archived_history")
        assert owner.active_revision_leases("installed", "digest") == 0
        owner.set_process_kind(token, "active_run")
        assert owner.active_revision_leases("installed", "digest") == 1
        owner.settle_process(token, confirmed=False)
        assert owner.active_revision_leases("installed", "digest") == 1
        assert owner.list_processes(limit=1, offset=0)[0]["provenance"] == provenance
        # A reused live PID is never signalled or interpreted as proof of ownership.
        assert os.getpid() == provenance["pid"]
    finally:
        owner.close()


@pytest.mark.parametrize(
    "name", ["Dropbox", "OneDrive - Example", "CloudStorage", "Mobile Documents"]
)
def test_synchronized_roots_refuse_ownership(tmp_path, name):
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    owner = PluginRuntimeOwner(tmp_path / name / "plugins")
    with pytest.raises(PermissionError):
        owner.try_acquire()
    assert not (tmp_path / name).exists()


def test_unknown_platform_network_and_probe_errors_refuse(tmp_path, monkeypatch):
    import tldw_chatbook.Plugins.runtime_owner as module

    with monkeypatch.context() as patch:
        patch.setattr(module.sys, "platform", "unknown")
        with pytest.raises(PermissionError):
            module.PluginRuntimeOwner(tmp_path).try_acquire()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_darwin_filesystem", lambda path: (0, "nfs"))
        with pytest.raises(PermissionError):
            module.PluginRuntimeOwner(tmp_path).try_acquire()
    with monkeypatch.context() as patch:

        def fail(path):
            raise OSError("probe failed")

        patch.setattr(module, "_darwin_filesystem", fail)
        with pytest.raises(PermissionError):
            module.PluginRuntimeOwner(tmp_path).try_acquire()


def test_symlink_lock_and_root_replacement_refuse(tmp_path):
    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(PermissionError):
        PluginRuntimeOwner(alias).try_acquire()
    (target / "runtime.lock").symlink_to(tmp_path / "victim")
    with pytest.raises((PermissionError, OSError)):
        PluginRuntimeOwner(target).try_acquire()
    assert not (tmp_path / "victim").exists()


def test_replaced_lock_and_fork_inherited_owner_cannot_mutate(tmp_path):
    # Fork in a fresh process so pytest's background threads are not inherited.
    contend = child_argv(CHILD) + [str(tmp_path), str(Path.cwd()), "contend"]
    code = """
import json, os, subprocess, sys, warnings
from pathlib import Path
from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner
warnings.simplefilter('error', DeprecationWarning)
root = Path(sys.argv[1])
owner = PluginRuntimeOwner(root)
assert owner.try_acquire()
try:
    token = owner.reserve_launch('op', 'installed', None, 'digest')
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            owner.settle_process(token, confirmed=True)
        except PermissionError:
            owner.close()
            os.write(write_fd, b'refused')
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 32) == b'refused'
    finally:
        os.close(read_fd)
        os.waitpid(pid, 0)
    result = subprocess.run(CONTEND, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'acquired': False}
    (root / 'runtime.lock').rename(root / 'old.lock')
    (root / 'runtime.lock').touch()
    try:
        owner.reserve_launch('op2', 'other', None, 'digest')
    except PermissionError:
        pass
    else:
        raise AssertionError('replaced lock accepted')
finally:
    owner.close()
""".replace("CONTEND", repr(contend))
    result = subprocess.run(
        child_argv(code) + [str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_owner_death_after_spawn_before_publication_preserves_pending(tmp_path):
    """A real orphaned child stays alive; lock acquisition does not settle it."""
    import time

    from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner

    finished = tmp_path / "child-finished"
    read_fd, write_fd = os.pipe()
    spawning_owner = """
import json, os, subprocess, sys
from pathlib import Path
from tldw_chatbook.Plugins.runtime_owner import PluginRuntimeOwner
owner = PluginRuntimeOwner(Path(sys.argv[1]))
assert owner.try_acquire()
token = owner.reserve_launch('op', 'installed', None, 'revision')
code = "import os,sys; from pathlib import Path; os.read(int(sys.argv[1]), 1); Path(sys.argv[2]).write_text('stopped')"
process = subprocess.Popen([sys.executable, '-I', '-c', code, sys.argv[2], sys.argv[3]], pass_fds=(int(sys.argv[2]),), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(json.dumps({'token': token, 'pid': process.pid}), flush=True)
os._exit(0)
"""
    owner = PluginRuntimeOwner(tmp_path)
    try:
        result = subprocess.run(
            child_argv(spawning_owner) + [str(tmp_path), str(read_fd), str(finished)],
            pass_fds=(read_fd,),
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        launch = json.loads(result.stdout)
        assert owner.try_acquire()
        row = owner.list_processes(limit=1, offset=0)[0]
        assert row["token"] == launch["token"]
        assert row["state"] == "pending"
        assert row["provenance"] is None
        assert not finished.exists()
        with pytest.raises(PermissionError):
            owner.reserve_launch("op2", "installed", None, "revision")
    finally:
        owner.close()
        os.close(read_fd)
        os.close(write_fd)
    # The test-owned child exits through its pipe, never by signalling a stale PID.
    deadline = time.monotonic() + 10
    while not finished.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert finished.read_text() == "stopped"


def test_unqualified_local_filesystem_refuses_even_with_local_flag(
    tmp_path, monkeypatch
):
    import tldw_chatbook.Plugins.runtime_owner as module

    monkeypatch.setattr(module, "_darwin_filesystem", lambda path: (0x1000, "hfs"))
    with pytest.raises(PermissionError):
        module.PluginRuntimeOwner(tmp_path).try_acquire()
