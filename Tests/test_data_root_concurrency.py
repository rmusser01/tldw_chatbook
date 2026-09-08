"""Default-root choice stays consistent across concurrent first launches."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX namespace and lock contract")
def test_concurrent_first_starts_agree_when_ancestor_permissions_change(tmp_path):
    home = tmp_path / "home"
    shared = home / ".local/share"
    shared.mkdir(parents=True)
    shared.chmod(0o775)
    script = """
import os
from pathlib import Path
import sys
import time
import portalocker
from tldw_chatbook import config

root, role = Path(sys.argv[1]), sys.argv[2]
home = root / 'home'
os.environ['HOME'] = str(home)
config.get_user_folder_name = lambda: 'alice'
config.get_cli_setting = lambda section, key, default=None: default
real_secure = config.secure_private_directory
real_lock = portalocker.lock

def paused_secure(path, **kwargs):
    if role == 'a' and Path(path) == home / '.tldw_cli-data':
        (root / 'a-ready').touch()
        deadline = time.monotonic() + 30
        while not (root / 'release-a').exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('timed out waiting to resume root creation')
            time.sleep(0.01)
    return real_secure(path, **kwargs)

def observed_lock(stream, flags):
    try:
        return real_lock(stream, flags | portalocker.LockFlags.NON_BLOCKING)
    except portalocker.exceptions.LockException:
        # Signal actual OS lock contention, not a guessed sleep interval.
        (root / 'b-blocked').touch()
        return real_lock(stream, flags)

config.secure_private_directory = paused_secure
if role == 'b':
    portalocker.lock = observed_lock
selected = config.get_user_data_dir()
(root / (role + '-result')).write_text(str(selected))
"""
    processes = []

    def start(role):
        bootstrap = tmp_path / f"bootstrap-{role}"
        config_path = bootstrap / "config/config.toml"
        config_path.parent.mkdir(parents=True)
        env = dict(
            os.environ,
            HOME=str(bootstrap),
            TLDW_CONFIG_PATH=str(config_path),
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path), role],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        return process

    def wait_for_signal(*names):
        deadline = time.monotonic() + 30
        while not any((tmp_path / name).exists() for name in names):
            for process in processes:
                if process.poll() is not None and process.returncode != 0:
                    _, stderr = process.communicate(timeout=5)
                    pytest.fail(stderr)
            assert time.monotonic() < deadline, f"Missing process signal: {names}"
            time.sleep(0.01)

    try:
        start("a")
        wait_for_signal("a-ready")
        shared.chmod(0o755)
        start("b")
        # Without coordination, B creates the conventional root and finishes;
        # with coordination, it demonstrably blocks on A's OS-backed lock.
        wait_for_signal("b-blocked", "b-result")
        (tmp_path / "release-a").touch()
        for process in processes:
            _, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
    finally:
        (tmp_path / "release-a").touch()
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.communicate(timeout=5)

    expected = str(home / ".tldw_cli-data/alice")
    assert (tmp_path / "a-result").read_text() == expected
    assert (tmp_path / "b-result").read_text() == expected
    assert not (shared / "tldw_cli").exists()
    assert (tmp_path / "b-blocked").exists()
