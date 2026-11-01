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
import re
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
# 'stranded': the state a second enable used to leave behind -- the verifier
# is for 'unlock-sentinel', but the saved key is encrypted under another one.
key_password = 'stranded-sentinel' if condition.startswith('stranded') else 'unlock-sentinel'
data_root = selector.parent.parent / 'data'
chats_db = data_root / 'chats-sentinel.db'
media_db = data_root / 'media-sentinel.db'
chats_db.write_bytes(b'chats, notes: SQLite sentinel bytes')
media_db.write_bytes(b'documents: SQLite sentinel bytes')
document = {
    'general': {'users_name': 'selected'},
    'database': {'chachanotes_db_path': str(chats_db), 'media_db_path': str(media_db)},
    'api_settings': {'openai': {
        'api_key': engine.encrypt_value('sentinel-key', key_password),
        'model': 'gpt-4o',
    }},
    'encryption': {'enabled': True, 'password_verifier': verifier},
}
if condition.startswith('no-verifier'):
    del document['encryption']['password_verifier']
if condition in ('no-verifier-headless', 'no-verifier-nothing-encrypted'):
    # Encryption on, no verifier, and nothing encrypted: no password matters,
    # so only the (lossless) reset is offered.
    del document['api_settings']['openai']['api_key']
if condition == 'served':
    # textual-serve's child: stdin is the web driver's pipe and getpass would
    # open the SERVER operator's terminal.
    os.environ['CHATBOOK_SERVED_CHILD'] = '1'
selector.write_text(toml.dumps(document))
selector.chmod(0o600)
before = selector.read_bytes()
# TASK-34100.16: a config carried to another machine and launched with
# --config, while TLDW_CONFIG_PATH names another file (the flag must win).
decoy = selector.parent / 'decoy.toml'
launch_flags = []
if os.environ.get('UNLOCK_CONFIG_VIA_FLAG') == '1':
    os.environ['TLDW_CONFIG_PATH'] = str(decoy)
    launch_flags = ['--config', str(selector)]
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
    'stranded': ['unlock-sentinel', ''],
    'stranded-reset': ['unlock-sentinel', ''],
    # Review round 2 (R2-F5): the earlier password reads every key.
    'stranded-recover': ['unlock-sentinel', 'stranded-sentinel'],
    'interrupt-during-check': ['typed-secret-sentinel'],
    'no-verifier': [''],
    'no-verifier-recover': ['typed-secret-sentinel', 'unlock-sentinel'],
    'no-verifier-nothing-encrypted': [],
    'no-verifier-headless': [],
    'served': [],
    'no-terminal': [],
}[condition]
# After a reset or a re-key, '' answers "Press Enter to start chatbook."
# (review round 2).
choices = {
    'wrong-then-quit': ['q'],
    'reset': ['r', ''],
    'stranded': ['q'],
    'stranded-reset': ['r', ''],
    'stranded-recover': [''],
    'no-verifier': ['r', ''],
    'no-verifier-recover': [''],
    'no-verifier-nothing-encrypted': ['r', ''],
}.get(condition, [])
def secret(prompt):
    if 'CONFIG_IMPORTED_BEFORE_PROMPT' not in seen:
        seen.add('CONFIG_IMPORTED_BEFORE_PROMPT')
        print('CONFIG_IMPORTED_BEFORE_PROMPT=' + str('tldw_chatbook.config' in sys.modules))
    if condition == 'no-terminal':
        # getpass's echoed fallback when no private terminal is available.
        raise launcher.getpass.GetPassWarning('Can not control echo on the terminal.')
    return answers.pop(0)
seen = set()
launcher.getpass.getpass = secret
launcher._can_answer = lambda: condition != 'no-verifier-headless'
launcher._choice = lambda prompt: (sys.stderr.write(prompt), choices.pop(0))[1]
observed = []
launcher.minimal_recovery = lambda reason: observed.append(reason) or 17
# Both entries import startup_unlock at call time, so this wrapper is what
# they run: it marks where the pre-TUI unlock ends in stderr.
_real_startup_unlock = launcher.startup_unlock
def _marked_startup_unlock():
    try:
        return _real_startup_unlock()
    finally:
        sys.stderr.write('\nUNLOCK_FINISHED\n')
launcher.startup_unlock = _marked_startup_unlock
if condition == 'forced-failure':
    def failing(self, password, verifier):
        raise RuntimeError('verifier read failed')
    ConfigEncryption.verify_password = failing
if condition == 'forced-crash':
    def crashing(self, password, verifier):
        raise TypeError('unexpected verifier type')
    ConfigEncryption.verify_password = crashing
if condition == 'interrupt-during-check':
    # Ctrl+C lands after Enter, while the scrypt check is still running.
    def interrupted(self, password, verifier):
        raise KeyboardInterrupt
    ConfigEncryption.verify_password = interrupted
class ReachedApplication(Exception):
    pass
# 'module' drives `python -m tldw_chatbook.app`; 'cli' drives the packaged
# `tldw-cli` entry (cli.main_cli_runner). Both run the same startup_unlock.
entry = os.environ.get('UNLOCK_ENTRY', 'module')
def reached_application():
    sys.stderr.write('\nAPPLICATION_STARTS_HERE\n')
    from tldw_chatbook import config
    loaded = config.load_cli_config_and_ensure_existence()
    key = loaded['api_settings']['openai'].get('api_key')
    print('PASSWORD_SET=' + str(config.get_encryption_password() == 'unlock-sentinel'))
    print('PASSWORD_RECOVERED=' + str(config.get_encryption_password() == 'stranded-sentinel'))
    print('PASSWORD_CLEARED=' + str(config.get_encryption_password() is None))
    print('KEY_DECRYPTED=' + str(key == 'sentinel-key'))
    print('KEY_USABLE=' + str(config.resolve_provider_api_key(key) is not None))
    raise ReachedApplication
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    # `_run_module_main` imports this right after the unlock and config load.
    if (
        entry == 'module' and level == 1 and name == 'Utils.terminal_utils'
        and (globals or {}).get('__name__') == 'tldw_chatbook.app_entry'
    ):
        reached_application()
    # `tldw-cli` imports the application right after the unlock.
    if entry == 'cli' and level == 0 and name == 'tldw_chatbook.app':
        reached_application()
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
if entry == 'cli':
    from tldw_chatbook.cli import main_cli_runner
    sys.argv = ['tldw-cli', *launch_flags]
    try:
        print('EXIT=' + str(main_cli_runner()))
    except ReachedApplication:
        print('REACHED_APPLICATION')
else:
    sys.argv = ['tldw_chatbook.app', *launch_flags]
    try:
        runpy.run_module('tldw_chatbook.app', run_name='__main__', alter_sys=True)
    except ReachedApplication:
        print('REACHED_APPLICATION')
    except SystemExit as stop:
        print('EXIT=' + str(stop.code))
builtins.__import__ = original
assert not answers and not choices, (answers, choices)
print('RECOVERY=' + ','.join(observed))
print('DECOY_CREATED=' + str(decoy.exists()))
text = selector.read_text()
print('FILE_UNCHANGED=' + str(selector.read_bytes() == before))
print('FILE_HAS_CIPHERTEXT=' + str('enc:' in text))
print('FILE_HAS_ENCRYPTION=' + str('[encryption]' in text))
import tomllib
final = tomllib.loads(text)
if condition in ('stranded-recover', 'no-verifier-recover'):
    final_verifier = final.get('encryption', {}).get('password_verifier')
    for name, password in (('STRANDED', 'stranded-sentinel'), ('UNLOCK', 'unlock-sentinel')):
        print('VERIFIER_FOR_' + name + '=' + str(
            bool(final_verifier) and engine.verify_password(password, final_verifier)
        ))
database = final.get('database', {})
print('DB_SETTINGS_KEPT=' + str(
    database.get('chachanotes_db_path') == str(chats_db)
    and database.get('media_db_path') == str(media_db)
))
print('DB_FILES_UNCHANGED=' + str(
    chats_db.read_bytes() == b'chats, notes: SQLite sentinel bytes'
    and media_db.read_bytes() == b'documents: SQLite sentinel bytes'
))
assert not blocked_attempts(), blocked_attempts()
"""


def _run_module_entry(
    tmp_path: Path, condition: str, *, entry: str = "module", via_flag: bool = False
) -> tuple[subprocess.CompletedProcess, str]:
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
        UNLOCK_ENTRY=entry,
        UNLOCK_CONFIG_VIA_FLAG="1" if via_flag else "0",
    )
    environment.pop("TLDW_VERBOSE_STARTUP", None)
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
    assert "stranded-sentinel" not in output


_LOG_LINE = re.compile(r"\| (TRACE|DEBUG|INFO|SUCCESS|WARNING|ERROR|CRITICAL) +\|")


def _unlock_output(result: subprocess.CompletedProcess) -> str:
    """Everything printed before the pre-TUI unlock returned."""
    assert "UNLOCK_FINISHED" in result.stderr, result.stderr[-4000:]
    return result.stderr.split("UNLOCK_FINISHED", 1)[0]


def _assert_no_log_lines(region: str) -> None:
    # Review round 2 (R2-F6/F-R2-4): installing the password (or resetting)
    # imported config while the file was still locked, so two "Encryption is
    # enabled but no password is set" WARNINGs printed right after the RIGHT
    # password -- and under `python -m`, ~40 DEBUG/INFO lines too.
    assert "no password is set" not in region, region[-3000:]
    lines = [line for line in region.splitlines() if _LOG_LINE.search(line)]
    assert not lines, lines


@pytest.mark.timeout(240)
@pytest.mark.parametrize("entry", ["module", "cli"])
def test_config_flag_sends_an_encrypted_copied_config_through_the_one_unlock(
    tmp_path, entry
):
    """TASK-34100.16: ``--config`` selects the file the unlock asks about.

    An encrypted config.toml carried to another machine and launched with
    ``--config`` asks for its master password through the same pre-TUI
    unlock as a default profile, on both entry points -- even when
    ``TLDW_CONFIG_PATH`` names a different file, because the flag wins.
    """
    result, verifier = _run_module_entry(tmp_path, "good", entry=entry, via_flag=True)
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "Enter the master password you set during setup." in result.stderr
    assert "PASSWORD_SET=True" in out
    assert "KEY_DECRYPTED=True" in out
    assert "FILE_UNCHANGED=True" in out
    assert "DECOY_CREATED=False" in out
    assert "RECOVERY=\n" in out
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("entry", ["module", "cli"])
def test_module_entry_unlocks_with_the_right_password(tmp_path, entry):
    result, verifier = _run_module_entry(tmp_path, "good", entry=entry)
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "PASSWORD_SET=True" in out
    assert "KEY_DECRYPTED=True" in out
    assert "FILE_UNCHANGED=True" in out
    assert "RECOVERY=\n" in out
    assert "Enter the master password you set during setup." in result.stderr
    assert "NoActiveWorker" not in result.stderr
    # Review round 1 (G4-R1-F3): the module path unlocks before app.py
    # imports config, exactly like tldw-cli -- so config never loads the
    # still-encrypted file first and no "no password is set" warning prints
    # above the prompt.
    above_prompt = result.stderr.split("Your saved API keys are encrypted.", 1)[0]
    assert "no password is set" not in above_prompt, above_prompt[-2000:]
    assert "CONFIG_IMPORTED_BEFORE_PROMPT=False" in out
    _assert_no_log_lines(_unlock_output(result))
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_module_entry_wrong_password_reprompts_in_place(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "retry")
    assert "REACHED_APPLICATION" in result.stdout, result.stderr[-4000:]
    assert "PASSWORD_SET=True" in result.stdout
    assert result.stderr.count("That password didn't match. Try again.") == 1
    # A wrong password is an expected outcome, not an error: no log line may
    # land between the prompts (live run printed "ERROR ... Decryption failed").
    assert "Decryption failed" not in result.stderr
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
@pytest.mark.parametrize("entry", ["module", "cli"])
def test_module_entry_reset_strips_encrypted_keys_and_opens_the_app(tmp_path, entry):
    result, verifier = _run_module_entry(tmp_path, "reset", entry=entry)
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "PASSWORD_CLEARED=True" in out
    assert "KEY_USABLE=False" in out
    assert "FILE_HAS_CIPHERTEXT=False" in out
    assert "FILE_HAS_ENCRYPTION=False" in out
    assert "Saved keys were reset" in result.stderr
    # Review round 2 (F-R2-4): the confirmation is the last thing before a
    # pause, not buried under log lines and painted over by the TUI.
    region = _unlock_output(result)
    _assert_no_log_lines(region)
    done = region.index("Saved keys were reset")
    assert region.index("Press Enter to start chatbook.", done) > done
    # Chats, notes and documents are untouched: the reset writes only the
    # encrypted values and [encryption] out of config.toml.
    assert "DB_SETTINGS_KEPT=True" in out
    assert "DB_FILES_UNCHANGED=True" in out
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_right_password_over_stranded_keys_says_so_and_never_unlocks(tmp_path):
    # Review round 1 (TASK-34100.4): the verifier is for the typed password,
    # but a saved key is encrypted under another one -- the state a second
    # enable used to leave behind. Strict decrypt refuses the unlock; the user
    # is told the password is RIGHT and that reset is the way out, and no
    # decrypt error is logged between the prompts.
    result, verifier = _run_module_entry(tmp_path, "stranded")
    out = result.stdout
    assert "EXIT=0" in out, result.stderr[-4000:]
    assert "REACHED_APPLICATION" not in out
    assert "RECOVERY=\n" in out
    assert "FILE_UNCHANGED=True" in out
    assert "That password is right, but some saved keys" in result.stderr
    assert "That password didn't match" not in result.stderr
    assert "Decryption failed" not in result.stderr
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_stranded_keys_can_be_reset_from_the_prompt(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "stranded-reset")
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "PASSWORD_CLEARED=True" in out
    assert "FILE_HAS_CIPHERTEXT=False" in out
    assert "FILE_HAS_ENCRYPTION=False" in out
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
@pytest.mark.parametrize("entry", ["module", "cli"])
def test_an_earlier_password_that_reads_every_key_unlocks_and_repairs_the_check(
    tmp_path, entry
):
    # Review round 2 (R2-F5): verifier for one password, keys under an
    # earlier one. The earlier password was answered "didn't match" and the
    # only way on was a reset that deleted keys it still reads.
    result, verifier = _run_module_entry(tmp_path, "stranded-recover", entry=entry)
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "PASSWORD_RECOVERED=True" in out
    assert "KEY_DECRYPTED=True" in out
    assert "VERIFIER_FOR_STRANDED=True" in out
    assert "VERIFIER_FOR_UNLOCK=False" in out
    assert "FILE_HAS_CIPHERTEXT=True" in out and "FILE_HAS_ENCRYPTION=True" in out
    assert "RECOVERY=\n" in out
    region = _unlock_output(result)
    # The right-but-stranded password now points at the earlier one.
    assert "If you set an earlier master password, enter that one" in region
    assert "That password didn't match" not in region
    recovered = region.index("it is your master password again")
    assert region.index("Press Enter to start chatbook.", recovered) > recovered
    _assert_no_log_lines(region)
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_missing_verifier_asks_for_the_password_and_repairs_the_check(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "no-verifier-recover")
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "PASSWORD_SET=True" in out
    assert "KEY_DECRYPTED=True" in out
    assert "VERIFIER_FOR_UNLOCK=True" in out
    assert "check for the master password is missing" in result.stderr
    assert result.stderr.count("That password didn't match. Try again.") == 1
    assert "[R]eset saved keys or [Q]uit" not in result.stderr
    _assert_no_log_lines(_unlock_output(result))
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_missing_verifier_with_nothing_encrypted_offers_only_the_reset(tmp_path):
    result, _verifier = _run_module_entry(tmp_path, "no-verifier-nothing-encrypted")
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "Master password" not in result.stderr
    assert "[R]eset saved keys or [Q]uit" in result.stderr
    assert "FILE_HAS_ENCRYPTION=False" in out


@pytest.mark.timeout(240)
def test_ctrl_c_during_the_password_check_quits_without_recovery(tmp_path):
    # Review round 1: Ctrl+C after Enter (while scrypt runs) used to open the
    # recovery host with "no private terminal ... or reading it failed".
    result, verifier = _run_module_entry(tmp_path, "interrupt-during-check")
    out = result.stdout
    assert "EXIT=0" in out, result.stderr[-4000:]
    assert "RECOVERY=\n" in out
    assert "FILE_UNCHANGED=True" in out
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_missing_verifier_offers_the_reset_in_a_terminal(tmp_path):
    # Review round 1: encryption on but no verifier used to send the user to
    # hand-edit config.toml; the same safe reset is offered instead.
    result, verifier = _run_module_entry(tmp_path, "no-verifier")
    out = result.stdout
    assert "REACHED_APPLICATION" in out, result.stderr[-4000:]
    assert "RECOVERY=\n" in out
    assert "check for the master password is missing" in result.stderr
    assert "[R]eset saved keys or [Q]uit" in result.stderr
    assert "FILE_HAS_CIPHERTEXT=False" in out
    assert "FILE_HAS_ENCRYPTION=False" in out
    assert "DB_FILES_UNCHANGED=True" in out


@pytest.mark.timeout(240)
def test_missing_verifier_without_a_terminal_opens_the_recovery_host(tmp_path):
    result, _verifier = _run_module_entry(tmp_path, "no-verifier-headless")
    assert "EXIT=17" in result.stdout, result.stderr[-4000:]
    assert "RECOVERY=configuration_unlock_unavailable" in result.stdout
    assert "FILE_UNCHANGED=True" in result.stdout


@pytest.mark.timeout(240)
def test_served_child_never_prompts_on_the_server_terminal(tmp_path):
    # Review round 1: `tldw-cli --serve` spawns `python -m tldw_chatbook.app`
    # per browser session. getpass would open the server operator's terminal
    # and hang the browser; the child shows the recovery host (rendered in
    # the browser) with a plain sentence instead.
    result, verifier = _run_module_entry(tmp_path, "served")
    out = result.stdout
    assert "EXIT=17" in out, result.stderr[-4000:]
    assert "RECOVERY=configuration_unlock_served" in out
    assert "Master password" not in result.stderr
    assert "FILE_UNCHANGED=True" in out
    _secrets_absent(result, verifier)


@pytest.mark.timeout(240)
def test_no_private_terminal_has_its_own_reason(tmp_path):
    result, verifier = _run_module_entry(tmp_path, "no-terminal")
    assert "EXIT=17" in result.stdout, result.stderr[-4000:]
    assert "RECOVERY=configuration_unlock_no_terminal" in result.stdout
    assert "FILE_UNCHANGED=True" in result.stdout
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


class _FakeTerminal:
    """Stands in for /dev/tty: answers lines, records what was written."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.written: list[str] = []
        self.closed = False

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        pass

    def readline(self) -> str:
        return self._answers.pop(0) if self._answers else ""

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def test_the_reset_choice_reads_the_terminal_when_stdin_is_redirected(monkeypatch):
    # Review round 2 (R2-F7): the password comes from /dev/tty (getpass), but
    # the [R]eset/[Q]uit answer was read from stdin. With stdin redirected
    # (`tldw-cli < /dev/null`, some IDE run configs) the choice hit EOF and
    # quit at once, so a person at the terminal could never reach reset.
    import io

    from tldw_chatbook.Backup_Recovery import launcher

    monkeypatch.setattr(launcher.sys, "stdin", io.StringIO(""))
    terminal = _FakeTerminal(["r\n"])
    monkeypatch.setattr(launcher, "_open_tty", lambda: terminal, raising=False)

    assert launcher._give_up_choice() == launcher.UNLOCK_RESET_REASON
    assert launcher.UNLOCK_CHOICE_PROMPT in "".join(terminal.written)
    assert terminal.closed


_REAL_TERMINAL_CHOICE = r"""
import fcntl, sys, termios
# A new session (start_new_session) with no terminal yet: make the pty on
# stdout its controlling terminal, so /dev/tty is real. stdin is /dev/null.
fcntl.ioctl(1, termios.TIOCSCTTY, 0)
from tldw_chatbook.Backup_Recovery import launcher
print('CHOICE=' + launcher._give_up_choice(), flush=True)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX controlling terminal")
@pytest.mark.timeout(120)
def test_the_reset_choice_reads_a_real_controlling_terminal(tmp_path):
    # Found live in review round 2: the fake terminal above hid that
    # open("/dev/tty", "r+") raises io.UnsupportedOperation (a terminal is
    # not seekable), so `tldw-cli < /dev/null` still quit at the choice.
    import pty
    import select
    import time

    master, slave = pty.openpty()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO)
    process = subprocess.Popen(  # noqa: S603 -- fixed interpreter and script
        [sys.executable, "-c", _REAL_TERMINAL_CHOICE],
        cwd=_REPO,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
        close_fds=True,
    )
    os.close(slave)
    output = b""
    answered = False
    deadline = time.monotonic() + 90
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.1)
            if ready:
                try:
                    chunk = os.read(master, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
            if not answered and b"[Q]uit" in output:
                os.write(master, b"r\n")
                answered = True
            if process.poll() is not None and not ready:
                break
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=30)
        os.close(master)
    text = output.decode("utf-8", "replace")
    assert answered, text
    assert "CHOICE=configuration_unlock_reset" in text, text


def test_without_any_terminal_the_choice_still_reads_stdin(monkeypatch):
    import io

    from tldw_chatbook.Backup_Recovery import launcher

    monkeypatch.setattr(launcher.sys, "stdin", io.StringIO("q\n"))
    monkeypatch.setattr(launcher, "_open_tty", lambda: None, raising=False)

    assert launcher._give_up_choice() == launcher.UNLOCK_QUIT_REASON
    assert launcher._can_answer() is False


async def test_opening_a_profile_where_the_terminal_cannot_suspend_reports_it():
    # Review round 2 (R2-F3): a served (browser) child now reaches the
    # recovery host. Its Backup & Restore "Open" profile button calls
    # open_recovery_profile, which suspends the terminal -- the web driver
    # cannot (SuspendNotSupported), and the uncaught error in a default
    # exit_on_error worker ended the whole browser session. The headless
    # test driver cannot suspend either, so it reproduces that.
    from tldw_chatbook.Backup_Recovery.launcher import (
        PROFILE_OPEN_NEEDS_TERMINAL,
        UNLOCK_SERVED_REASON,
        recovery_app,
    )

    app = recovery_app(UNLOCK_SERVED_REASON)
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        assert not app._driver.can_suspend
        worker = app.open_recovery_profile("profile-sentinel")
        await worker.wait()
        await pilot.pause()
        assert app.return_code is None and app.is_running
        messages = [notice.message for notice in app._notifications]
        assert PROFILE_OPEN_NEEDS_TERMINAL in messages, messages


@pytest.mark.parametrize(
    "reason",
    [
        "configuration_unlock_failed",
        "configuration_unlock_unavailable",
        "configuration_unlock_no_terminal",
        "configuration_unlock_served",
    ],
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
        # Review round 1: no unlock copy sends the user to hand-edit
        # config.toml, and the generic failure no longer blames a missing
        # terminal it never checked for.
        assert "remove its [encryption] table" not in text
        if reason != "configuration_unlock_no_terminal":
            assert "no private terminal" not in text
        await pilot.click("#minimal-recovery-open")
        await pilot.pause()
        assert isinstance(app.screen, BackupRestoreScreen)
