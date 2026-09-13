"""Windows restart releases the old process only after a safely quoted spawn."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from tldw_chatbook.Backup_Recovery import recovery_restart


@pytest.mark.parametrize("spawn_failure", [False, True])
def test_windows_restart_spawns_with_filtered_environment_before_exit(
    tmp_path, monkeypatch, spawn_failure
):
    request = recovery_restart.RecoveryRestart(
        tmp_path / "archive with spaces Ω.zip", tmp_path / "config with spaces.toml"
    )
    events = []
    failure = OSError("synthetic spawn failure")
    monkeypatch.setenv("TEST_PROVIDER_API_KEY", "synthetic-secret")

    def spawn(argv, **kwargs):
        events.append(("spawn", argv, kwargs))
        if spawn_failure:
            raise failure
        return SimpleNamespace(pid=123)

    def exit_parent(code):
        events.append(("exit", code))
        raise SystemExit(code)

    def unsafe_exec(*args):
        raise AssertionError("Windows restart must bypass the UCRT execve path")

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(
        recovery_restart,
        "os",
        SimpleNamespace(name="nt", execve=unsafe_exec, _exit=exit_parent),
    )
    if spawn_failure:
        with pytest.raises(OSError) as caught:
            recovery_restart.restart(request)
        assert caught.value is failure
        assert len(events) == 1
    else:
        with pytest.raises(SystemExit) as caught:
            recovery_restart.restart(request)
        assert caught.value.code == 0
        assert events[-1] == ("exit", 0)
    _, argv, options = events[0]
    assert argv == [
        sys.executable, "-c", recovery_restart._ENTRY,
        str(request.archive), str(request.target_config),
    ]
    assert options.keys() == {"env", "stdin", "stdout", "stderr", "close_fds"}
    assert (options["stdin"], options["stdout"], options["stderr"]) == (0, 1, 2)
    assert options["close_fds"] is True
    assert "TEST_PROVIDER_API_KEY" not in options["env"]
    assert options["env"]["PYTHON_KEYRING_BACKEND"] == "keyring.backends.null.Keyring"
