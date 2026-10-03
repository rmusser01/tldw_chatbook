"""One startup unlock for both entry points (TASK-34100.4).

``python -m tldw_chatbook.app`` used to run a private Textual
``PasswordPromptApp`` that crashed with ``NoActiveWorker`` on every encrypted
launch, printed config frame locals (including the password verifier), and
exited 1 (protect-summary-01, new-protect-summary-02). Both entry points now
run ``Backup_Recovery.launcher.startup_unlock``. These tests drive the real
module entry in a subprocess with only the secret prompt and the reset/quit
choice scripted:

* the right password unlocks with strict decryption, and a wrong one
  re-prompts in place (protect-summary-02);
* giving up offers reset or quit; reset strips every encrypted value and the
  ``[encryption]`` table and still opens the app;
* no failure path prints the verifier or a typed password;
* the recovery host no longer jumps to Backup & Restore on an unlock problem.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

_MODULE_ENTRY = r"""
import builtins, os, runpy, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
condition = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
import toml
from tldw_chatbook.Utils.config_encryption import ConfigEncryption
engine = ConfigEncryption()
verifier = engine.create_password_verifier('unlock-sentinel')
document = {
    'general': {'users_name': 'selected'},
    'api_settings': {'openai': {
        'api_key': engine.encrypt_value('sentinel-key', 'unlock-sentinel'),
        'model': 'gpt-4o',
    }},
    'encryption': {'enabled': True, 'password_verifier': verifier},
}
selector.write_text(toml.dumps(document))
selector.chmod(0o600)
before = selector.read_bytes()
# Published to the parent so it can prove the verifier never reached output.
print('VERIFIER=' + verifier, file=open(os.environ['VERIFIER_FILE'], 'w'))
from tldw_chatbook.Backup_Recovery import launcher
answers = {
    'good': ['unlock-sentinel'],
    'retry': ['typed-secret-sentinel', 'unlock-sentinel'],
    'wrong-then-quit': ['typed-secret-sentinel', ''],
    'reset': [''],
    'forced-failure': ['typed-secret-sentinel'],
    'forced-crash': ['typed-secret-sentinel'],
}[condition]
choices = {'wrong-then-quit': ['q'], 'reset': ['r']}.get(condition, [])
launcher.getpass.getpass = lambda prompt: answers.pop(0)
launcher._choice = lambda prompt: (sys.stderr.write(prompt), choices.pop(0))[1]
observed = []
launcher.minimal_recovery = lambda reason: observed.append(reason) or 17
if condition == 'forced-failure':
    def failing(self, password, verifier):
        raise RuntimeError('verifier read failed')
    ConfigEncryption.verify_password = failing
if condition == 'forced-crash':
    def crashing(self, password, verifier):
        raise TypeError('unexpected verifier type')
    ConfigEncryption.verify_password = crashing
class ReachedApplication(Exception):
    pass
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    # `_run_module_main` imports this right after the unlock and config load.
    if (
        level == 1 and name == 'Utils.terminal_utils'
        and (globals or {}).get('__name__') == 'tldw_chatbook.app_entry'
    ):
        from tldw_chatbook import config
        loaded = config.load_cli_config_and_ensure_existence()
        key = loaded['api_settings']['openai'].get('api_key')
        print('PASSWORD_SET=' + str(config.get_encryption_password() == 'unlock-sentinel'))
        print('PASSWORD_CLEARED=' + str(config.get_encryption_password() is None))
        print('KEY_DECRYPTED=' + str(key == 'sentinel-key'))
        print('KEY_USABLE=' + str(config.resolve_provider_api_key(key) is not None))
        raise ReachedApplication
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
sys.argv = ['tldw_chatbook.app']
try:
    runpy.run_module('tldw_chatbook.app', run_name='__main__', alter_sys=True)
except ReachedApplication:
    print('REACHED_APPLICATION')
except SystemExit as stop:
    print('EXIT=' + str(stop.code))
builtins.__import__ = original
assert not answers and not choices, (answers, choices)
print('RECOVERY=' + ','.join(observed))
text = selector.read_text()
print('FILE_UNCHANGED=' + str(selector.read_bytes() == before))
print('FILE_HAS_CIPHERTEXT=' + str('enc:' in text))
print('FILE_HAS_ENCRYPTION=' + str('[encryption]' in text))
assert not blocked_attempts(), blocked_attempts()
"""


def _run_module_entry(tmp_path: Path, condition: str) -> tuple[subprocess.CompletedProcess, str]:
    root = tmp_path.resolve()
    for name in ("home", "config", "data"):
        (root / name).mkdir(mode=0o700)
    verifier_file = root / "verifier.txt"
    environment = os.environ.copy()
    environment.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        TLDW_CONFIG_PATH=str(root / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        VERIFIER_FILE=str(verifier_file),
    )
    result = subprocess.run(
        [sys.executable, "-c", _MODULE_ENTRY, condition],
        cwd=_REPO,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    verifier = verifier_file.read_text().strip().removeprefix("VERIFIER=")
    return result, verifier


def _secrets_absent(result: subprocess.CompletedProcess, verifier: str) -> None:
    output = result.stdout + result.stderr
    assert verifier and verifier not in output
    # Any 16-character slice of the verifier's base64 body is enough to
    # matter; the old crash printed its first ~40 characters.
    body = verifier.removeprefix("enc:")
    assert all(body[i : i + 16] not in output for i in range(0, len(body) - 16, 8))
    assert "typed-secret-sentinel" not in output
    assert "unlock-sentinel" not in output


@pytest.mark.timeout(240)
def test_module_entry_unlocks_with_the_right_password(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "good")
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "PASSWORD_SET=True" in out
    assert "KEY_DECRYPTED=True" in out
    assert "FILE_UNCHANGED=True" in out
    assert "RECOVERY=\n" in out
    assert "Enter the master password you set during setup." in result.stderr
    assert "NoActiveWorker" not in result.stderr
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_module_entry_wrong_password_reprompts_in_place(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "retry")
    assert "REACHED_APPLICATION" in result.stdout, result.stderr[-4000:]
    assert "PASSWORD_SET=True" in result.stdout
    assert result.stderr.count("That password didn't match. Try again.") == 1
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_module_entry_give_up_and_quit_leaves_everything_untouched(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "wrong-then-quit")
    out = result.stdout
    assert "EXIT=0" in out, result.stderr[-4000:]
    assert "REACHED_APPLICATION" not in out
    assert "RECOVERY=\n" in out  # no Backup & Restore detour
    assert "FILE_UNCHANGED=True" in out
    assert "[R]eset saved keys or [Q]uit" in result.stderr
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_module_entry_reset_strips_encrypted_keys_and_opens_the_app(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "reset")
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "PASSWORD_CLEARED=True" in out
    assert "KEY_USABLE=False" in out
    assert "FILE_HAS_CIPHERTEXT=False" in out
    assert "FILE_HAS_ENCRYPTION=False" in out
    assert "Saved keys were reset" in result.stderr
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_forced_unlock_failure_prints_no_secret(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "forced-failure")
    assert "EXIT=17" in result.stdout, result.stderr[-4000:]
    assert "RECOVERY=configuration_unlock_failed" in result.stdout
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_unexpected_unlock_crash_prints_no_secret(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "forced-crash")
    assert result.returncode != 0
    assert "TypeError" in result.stderr
    _secrets_absent(result, verifier)


@pytest.mark.parametrize(
    "reason", ["configuration_unlock_failed", "configuration_unlock_unavailable"]
)
async def test_recovery_host_explains_unlock_problems_without_backup_restore(reason):
    from textual.widgets import Static

    from tldw_chatbook.Backup_Recovery.launcher import recovery_app
    from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen

    app = recovery_app(reason)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert not isinstance(app.screen, BackupRestoreScreen)
        text = str(app.query_one("#minimal-recovery-reason", Static).render())
        assert "master password" in text
        assert reason not in text
        await pilot.click("#minimal-recovery-open")
        await pilot.pause()
        assert isinstance(app.screen, BackupRestoreScreen)
