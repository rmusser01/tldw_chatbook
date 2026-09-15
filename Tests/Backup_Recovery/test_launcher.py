"""The recovery launcher remains independent of normal application startup."""

import builtins

import pytest


def test_recovery_help_does_not_import_normal_app(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name not in {"tldw_chatbook.app", "tldw_chatbook.config"}
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    from tldw_chatbook.Backup_Recovery.launcher import recovery_main

    with pytest.raises(SystemExit) as result:
        recovery_main(["--help"])
    assert result.value.code == 0


def test_password_value_is_not_an_argument():
    from tldw_chatbook.Backup_Recovery.launcher import recovery_main

    with pytest.raises(SystemExit) as result:
        recovery_main(["inspect", "archive.zip", "--password", "do-not-accept"])
    assert result.value.code == 2


_MINIMAL = r"""
import builtins, os, runpy, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
route, condition = sys.argv[1:]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
if condition != 'missing':
    selector.write_text('broken = [' if condition == 'malformed' else '[encryption]\nenabled=true\n')
    selector.chmod(0o600)
before = selector.read_bytes() if selector.exists() else None
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name not in {'tldw_chatbook.app', 'tldw_chatbook.config'}, name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from Tests.Backup_Recovery.test_restore_plan import sealed
sealed(Path.home())
archive = Path.home() / 'fixture.zip'
from tldw_chatbook.Backup_Recovery.launcher import recovery_main
arguments = ['--control-root', str(Path.home() / 'control'), 'inspect', str(archive)]
if route == 'function':
    assert recovery_main(arguments) == 0
elif route == 'module':
    sys.argv = ['recovery', *arguments]
    try:
        runpy.run_module('tldw_chatbook.Backup_Recovery', run_name='__main__')
    except SystemExit as result:
        assert result.code == 0
else:
    from tldw_chatbook.cli import main_cli_runner
    sys.argv = ['tldw-chatbook', 'recovery', *arguments]
    assert main_cli_runner() == 0
assert (selector.read_bytes() if selector.exists() else None) == before
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.app' not in sys.modules
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize("route", ["function", "module", "cli"])
@pytest.mark.parametrize("condition", ["missing", "malformed", "locked"])
def test_real_inspection_before_normal_startup(tmp_path, route, condition):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, route, condition, script=_MINIMAL)


_ISOLATED = (
    _MINIMAL.split("from Tests.Backup_Recovery.test_restore_plan import sealed", 1)[0]
    + r"""
import tomllib
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.launcher import recovery_main
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
def config_manifest(doc):
    doc['owners'][0]['owner_id'] = 'config'
    doc['files'][0].update(owner_id='config', logical_id='profile:profile:config', relative_path='config.toml')
    doc['dependency_groups'][0]['members'] = ['profile:profile:config']
sealed(Path.home(), mutate=config_manifest, data=b'[general]\nusers_name="original"\n')
parent = Path.home() / 'selected'
parent.mkdir(mode=0o700)
control = Path.home() / 'control'
arguments = ['--control-root', str(control), 'restore', str(Path.home() / 'fixture.zip'),
    '--isolated', '--destination', 'root=' + str(parent / 'config'),
    '--destination', 'profile:profile:paths.data_dir=' + str(parent / 'data'),
    '--profile-name', 'profile=Restored CLI profile']
builtins.input = lambda prompt: 'restore' if route == 'confirm' else 'cancel'
assert recovery_main(arguments) == 0
service = RecoveryService(control)
try:
    profiles = service.profiles()
    if route == 'confirm':
        assert len(profiles) == 1, profiles
        assert profiles[0]['config'] == str(parent / 'config' / 'config.toml'), profiles
        assert tomllib.loads((parent / 'config' / 'config.toml').read_text())['general']['users_name'] == 'Restored CLI profile'
        assert profiles[0]['status'] == 'restoration_validated'
    else:
        assert not profiles
        assert not (parent / 'config').exists()
        assert not (parent / 'data').exists()
finally:
    service.close()
assert (selector.read_bytes() if selector.exists() else None) == before
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.app' not in sys.modules
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("decision", ["confirm", "cancel"])
def test_real_isolated_restore_requires_review_confirmation(tmp_path, decision):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, decision, "malformed", script=_ISOLATED)


@pytest.mark.parametrize("password", [None, "wrong", "correct"])
def test_actual_encrypted_inspection_prompts_without_printing_password(
    tmp_path, monkeypatch, helper_resource_root, capsys, password
):
    from threading import Event

    from Tests.Backup_Recovery.test_archive_writer import captured
    from tldw_chatbook.Backup_Recovery import crypto, launcher
    from tldw_chatbook.Backup_Recovery.archive_writer import write_archive

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    source = tmp_path / "encrypted.tldw-backup.zip.age"
    write_archive(captured(tmp_path), source, password=b"correct", cancel=Event())
    original = source.read_bytes()
    monkeypatch.setattr(launcher.getpass, "getpass", lambda prompt: password)
    args = ["--control-root", str(tmp_path / "control"), "inspect", str(source)]
    if password is not None:
        args.append("--ask-password")
    assert launcher.recovery_main(args) == (0 if password == "correct" else 1)
    output = capsys.readouterr()
    assert "correct" not in output.out + output.err
    assert "wrong" not in output.out + output.err
    assert source.read_bytes() == original


_BOOTSTRAP = r"""
import builtins, os, sys
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
condition = sys.argv[1]
selector = Path(os.environ['TLDW_CONFIG_PATH'])
if condition.startswith('encrypted'):
    import toml
    from tldw_chatbook.Utils.config_encryption import ConfigEncryption
    encryption = ConfigEncryption()
    document = encryption.encrypt_config({'general': {'users_name': 'selected'},
        'api_settings': {'openai': {'api_key': 'sentinel-key'}}}, 'unlock-sentinel')
    document['encryption'] = {'enabled': True,
        'password_verifier': encryption.create_password_verifier('unlock-sentinel')}
    selector.write_text(toml.dumps(document))
elif condition != 'missing':
    selector.write_text('not = [' if condition == 'malformed' else '[general]\nusers_name="selected"\n')
if selector.exists(): selector.chmod(0o600)
before = selector.read_bytes() if selector.exists() else None
from tldw_chatbook.Backup_Recovery import launcher
if condition == 'inaccessible':
    selector.parent.chmod(0)
if condition == 'uncertain':
    from tldw_chatbook.Backup_Recovery import bootstrap
    bootstrap_root = bootstrap.default_bootstrap_root()
    bootstrap_root.mkdir(parents=True, mode=0o700)
    (bootstrap_root / 'unbound-owner').write_bytes(b'local pending ownership')
observed = []
launcher.minimal_recovery = lambda reason: observed.append(reason) or 17
def password(prompt):
    if condition == 'encrypted-cancel': raise EOFError
    return 'unlock-sentinel' if condition == 'encrypted-good' else 'wrong-sentinel'
launcher.getpass.getpass = password
class ReachedApplication(Exception): pass
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'tldw_chatbook.app':
        if condition == 'encrypted-good':
            from tldw_chatbook import config
            assert config.get_encryption_password() == 'unlock-sentinel'
            assert config.chachanotes_db is None and config.media_db is None and config.prompts_db is None
        raise ReachedApplication
    if name == 'tldw_chatbook.config':
        assert condition == 'encrypted-good', condition
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tldw_chatbook.cli import main_cli_runner
sys.argv = ['tldw-chatbook']
try:
    result = main_cli_runner()
except ReachedApplication:
    assert condition in ('missing', 'valid', 'encrypted-good'), condition
    assert not observed
else:
    assert condition in ('malformed', 'encrypted-wrong', 'encrypted-cancel', 'inaccessible', 'uncertain'), condition
    assert result == 17 and observed
    assert 'tldw_chatbook.config' not in sys.modules
if condition == 'inaccessible': selector.parent.chmod(0o700)
assert (selector.read_bytes() if selector.exists() else None) == before
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""


@pytest.mark.parametrize(
    "condition",
    [
        "missing",
        "valid",
        "malformed",
        "encrypted-good",
        "encrypted-wrong",
        "encrypted-cancel",
        "inaccessible",
        "uncertain",
    ],
)
def test_prestartup_routes_damage_and_preserves_verified_unlock(tmp_path, condition):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, condition, "normal", script=_BOOTSTRAP)


def test_repeated_interrupt_keeps_actual_inspection_worker_until_settled(
    tmp_path, monkeypatch
):
    from threading import Event

    from Tests.Backup_Recovery.test_restore_plan import sealed
    from tldw_chatbook.Backup_Recovery import launcher, recovery_service

    sealed(tmp_path)
    entered, release = Event(), Event()
    native_acquire = recovery_service.archive_reader.acquire

    def held(*args, **kwargs):
        result = native_acquire(*args, **kwargs)
        entered.set()
        assert release.wait(10)
        return result

    monkeypatch.setattr(recovery_service.archive_reader, "acquire", held)
    service = recovery_service.RecoveryService(tmp_path / "control")
    native_wait = service.wait
    calls = 0

    def interrupted(operation):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise KeyboardInterrupt
        release.set()
        return native_wait(operation)

    try:
        operation = service.start_inspection(tmp_path / "fixture.zip", password=None)
        assert entered.wait(10)
        monkeypatch.setattr(service, "wait", interrupted)
        state = launcher._wait(service, operation)
        assert state["state"] != "running" and state["cancellation_requested"]
    finally:
        release.set()
        service.close()


_MINIMAL_UI = (
    _MINIMAL.split("from Tests.Backup_Recovery.test_restore_plan import sealed", 1)[0]
    + r"""
import asyncio
from textual.widgets import Input, Static
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.launcher import recovery_app
from tldw_chatbook.UI.Screens.backup_restore_screen import BackupRestoreScreen
sealed(Path.home())
async def main():
    app = recovery_app('configuration_invalid')
    async with app.run_test(size=(90, 32)) as pilot:
        assert isinstance(app.screen, BackupRestoreScreen)
        service = app.recovery_service
        await pilot.click('#backup-open-inspect')
        app.screen.query_one('#backup-source', Input).value = str(Path.home() / 'fixture.zip')
        await pilot.click('#backup-inspect')
        async with asyncio.timeout(15):
            while not service.current() or service.current()['state'] == 'running':
                await asyncio.sleep(.03)
        assert service.current()['result']['archive_verified']
        await pilot.press('escape')
        await pilot.click('#minimal-recovery-open')
        await pilot.pause()
        assert isinstance(app.screen, BackupRestoreScreen)
        assert app.screen.service is service
        assert 'Archive verified' in str(app.screen.query_one('#backup-status', Static).render())
    assert service._closed
asyncio.run(main())
assert (selector.read_bytes() if selector.exists() else None) == before
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.app' not in sys.modules
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""
)


def test_minimal_host_uses_real_inspection_and_retains_status_without_normal_app(
    tmp_path,
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "minimal", "malformed", script=_MINIMAL_UI)


def test_password_prompt_refuses_echoing_getpass_fallback(monkeypatch):
    import warnings

    from tldw_chatbook.Backup_Recovery import launcher

    fallback_read = []

    def fallback(prompt):
        warnings.warn("Password input may be echoed.", launcher.getpass.GetPassWarning)
        fallback_read.append(True)
        return "echoed-secret"

    monkeypatch.setattr(launcher.getpass, "getpass", fallback)
    with pytest.raises(ValueError, match="private_password_prompt_unavailable"):
        launcher._password("Password: ")
    assert not fallback_read


def test_recover_command_replays_actual_stopped_native_rollback(
    tmp_path, monkeypatch, helper_resource_root, capsys
):
    from dataclasses import replace
    from threading import Event

    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import (
        bootstrap,
        crypto,
        launcher,
        publication,
        replacement,
    )
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        candidate, template, _, _, source, selector = case
        before = {path: path.read_bytes() for path in (source, selector)}
        plan = replace(
            template, acknowledged_credential_issues=("credential_format_unreadable",)
        )
        native_retire = publication._retire

        def interrupted(item):
            native_retire(item)
            raise InterruptedError("stop after actual native retirement")

        monkeypatch.setattr(publication, "_retire", interrupted)
        with pytest.raises(InterruptedError):
            replacement.replace(
                plan,
                candidate,
                control_root=tmp_path / "control",
                rollback_password=b"rollback-sentinel",
                cancel=Event(),
            )
        monkeypatch.setattr(publication, "_retire", native_retire)
        service = RecoveryService(tmp_path / "control")
        try:
            operation = service.pending_operations()[0]["operation_id"]
            assert "rollback" in service.status(operation)["actions"]
        finally:
            service.close()
        monkeypatch.setattr(
            launcher.getpass, "getpass", lambda prompt: "rollback-sentinel"
        )
        monkeypatch.setattr("builtins.input", lambda prompt: "rollback")
        assert (
            launcher.recovery_main(
                [
                    "--control-root",
                    str(tmp_path / "control"),
                    "recover",
                    operation,
                    "--rollback",
                    "--ask-password",
                ]
            )
            == 0
        )
        assert all(path.read_bytes() == content for path, content in before.items())
        assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
        assert "rollback-sentinel" not in capsys.readouterr().out


_REFUSAL = (
    _MINIMAL.split("from Tests.Backup_Recovery.test_restore_plan import sealed", 1)[0]
    + r"""
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.launcher import recovery_main
sealed(Path.home())
control = Path.home() / 'control'
for command in ('profiles', 'copies'):
    assert recovery_main(['--control-root', str(control), command]) == 0
def no_confirmation(prompt):
    raise AssertionError('unverified replacement must not reach confirmation')
builtins.input = no_confirmation
arguments = ['--control-root', str(control), 'restore', str(Path.home() / 'fixture.zip'),
    '--replace', '--destination', 'root=' + str(Path.home() / 'unpublished')]
if route == 'explicit': arguments += ['--target-config', str(selector)]
assert recovery_main(arguments) == 1
assert not (Path.home() / 'unpublished').exists()
assert (selector.read_bytes() if selector.exists() else None) == before
assert 'tldw_chatbook.config' not in sys.modules
assert 'tldw_chatbook.app' not in sys.modules
assert not blocked_attempts(), blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("target", ["absent", "explicit"])
def test_replace_refuses_unverified_current_target_and_lists_remain_available(
    tmp_path, target
):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, target, "malformed", script=_REFUSAL)


@pytest.fixture
def pre_safety_recovery(tmp_path, monkeypatch, helper_resource_root):
    """A real interrupted pre-safety replacement, reopened through the service."""
    from Tests.Backup_Recovery.test_held_sqlite_rollback import replacement_case
    from tldw_chatbook.Backup_Recovery import control_records, crypto
    from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService

    monkeypatch.setattr(crypto, "_package_resource_root", lambda: helper_resource_root)
    with replacement_case(tmp_path, monkeypatch, prepared=False) as case:
        _, original, _, _, source, selector = case
        before = {path: path.read_bytes() for path in (source, selector)}
        register = control_records.register_pending

        def interrupted(*args, **kwargs):
            register(*args, **kwargs)
            raise InterruptedError("after actual pending registration")

        monkeypatch.setattr(control_records, "register_pending", interrupted)
        service = RecoveryService(tmp_path / "control")
        try:
            inspection = service.start_inspection(
                tmp_path / "replacement.zip", password=None
            )
            assert service.wait(inspection, timeout=15)["state"] == "succeeded"
            plan = service.preview_restore(
                inspection,
                mode="replace",
                destinations=dict((*original.destinations, *original.selectors)),
                target=original.target,
                profile_names=dict(original.profile_names),
            )
            operation = service.start_restore(
                inspection, plan, rollback_password=b"unused"
            )
            assert service.wait(operation, timeout=40)["state"] == "recovery_required"
            pending = service.pending_operations()[0]["operation_id"]
        finally:
            service.close()
        monkeypatch.setattr(control_records, "register_pending", register)
        reopened = RecoveryService(tmp_path / "control")
        try:
            assert reopened.status(pending)["actions"] == ("abort",)
            yield reopened, pending, before, selector
        finally:
            reopened.close()


def test_recover_abort_preserves_actual_pre_safety_originals(
    pre_safety_recovery, monkeypatch, tmp_path, capsys
):
    from tldw_chatbook.Backup_Recovery import bootstrap, launcher

    service, pending, before, selector = pre_safety_recovery
    service.close()
    monkeypatch.setattr("builtins.input", lambda prompt: "abort")

    def forbidden_password(prompt):
        pytest.fail("Abort must not ask for an unrelated rollback password")

    monkeypatch.setattr(launcher.getpass, "getpass", forbidden_password)
    assert (
        launcher.recovery_main(
            [
                "--control-root",
                str(tmp_path / "control"),
                "recover",
                pending,
                "--abort",
                "--ask-password",
            ]
        )
        == 0
    )
    assert all(path.read_bytes() == content for path, content in before.items())
    assert bootstrap.startup_permission(selector, tmp_path / "bootstrap")[0]
    assert '"aborted": true' in capsys.readouterr().out
