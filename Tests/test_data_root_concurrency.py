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
from pathlib import Path
import sys
import time
import portalocker
from tldw_chatbook.Utils import private_paths

root, role = Path(sys.argv[1]), sys.argv[2]
home = root / 'home'
real_secure = private_paths.secure_private_directory
real_lock = portalocker.lock

def wait_for_release(name):
    deadline = time.monotonic() + 30
    while not (root / name).exists():
        if time.monotonic() >= deadline:
            raise RuntimeError(f'timed out waiting for {name}')
        time.sleep(0.01)

def paused_secure(path, **kwargs):
    if role == 'a' and Path(path) == home / '.tldw_cli-data':
        (root / 'a-ready').touch()
        wait_for_release('release-a')
    return real_secure(path, **kwargs)

def observed_lock(stream, flags):
    try:
        return real_lock(stream, flags)
    except portalocker.exceptions.AlreadyLocked:
        # Signal actual OS lock contention, not a guessed sleep interval.
        (root / 'b-blocked').touch()
        raise

# Settings load resolves the data root during the first config import.
if role == 'a':
    private_paths.secure_private_directory = paused_secure
else:
    portalocker.lock = observed_lock
from tldw_chatbook import config

selected = config.get_user_data_dir()
(root / (role + '-result')).write_text(str(selected))
"""
    processes = []

    def start(role):
        bootstrap = tmp_path / f"bootstrap-{role}"
        config_path = bootstrap / "config/config.toml"
        config_path.parent.mkdir(mode=0o700, parents=True)
        config_path.write_text('[general]\nusers_name = "alice"\n', encoding="utf-8")
        config_path.chmod(0o600)
        env = dict(
            os.environ,
            HOME=str(home),
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
        assert not (home / ".tldw_cli-data").exists()
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
