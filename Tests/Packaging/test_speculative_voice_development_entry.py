"""Source-only entry point for an unqualified speculative-voice checkout."""

from __future__ import annotations

from importlib import import_module, util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from Tests.UI.test_console_dictation import _mounted_console
from Tests.UI.test_console_hands_free_wiring import _ready_host
from Tests.private_profile import is_private_profile_child, private_profile_test
from Tests.UI.hands_free_consent_support import (
    accept_hands_free_consent,
    wait_for_hands_free_consent,
)
from tldw_chatbook.Chat.console_hands_free import HandsFreeController
from tldw_chatbook.Chat.console_voice_controls import ControlKind
from tldw_chatbook.UI.Console_Modules.hands_free import (
    ConsoleSpeculativeHandsFreeSession,
)


def _entry():
    spec = util.find_spec("scripts.run_speculative_voice_dev")
    assert spec is not None, "source-only speculative voice entry is unavailable"
    return import_module("scripts.run_speculative_voice_dev")


def _head(dev_entry) -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=dev_entry.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"),
    ).stdout.strip()


def _prepare_profile(request, dev_entry, *, legacy=False):
    from tldw_chatbook import config

    assert is_private_profile_child(request)
    selected = config.get_cli_config_path()
    assert selected == Path(os.environ["TLDW_CONFIG_PATH"])
    head = _head(dev_entry)
    data_root = selected.parent.parent / "voice-data"
    data_root.mkdir(mode=0o700, exist_ok=True)
    values = {
        "general": {"users_name": f"speculative_voice_dev_{head[:12]}"},
        "Paths" if legacy else "paths": {"data_dir": str(data_root)},
    }
    if legacy:
        values["database"] = {"chachanotes_db_path": str(data_root / "custom.db")}
    assert config.save_settings_to_cli_config(values)
    return head


class _QualifiedSession:
    def __init__(self) -> None:
        self.entered_with: list[bool] = []
        self.close_reasons: list[ControlKind] = []

    def enter(self, *, capture_live: bool) -> None:
        self.entered_with.append(capture_live)

    def fence_and_close(self, reason: ControlKind) -> None:
        self.close_reasons.append(reason)


def test_development_entry_refuses_a_non_checkout_before_enabling_the_gate(
    tmp_path,
) -> None:
    dev_entry = _entry()
    package_root = tmp_path / "tldw_chatbook"
    package_root.mkdir()

    with pytest.raises(dev_entry.DevelopmentVoiceEntryError, match="Git worktree"):
        dev_entry.validate_source_checkout(
            repo_root=tmp_path,
            package_root=package_root,
            expected_head="0" * 40,
        )


@pytest.mark.parametrize(
    "invalid", ["abbreviated", "mismatch", "foreign_package", "detached"]
)
def test_development_entry_rejects_inexact_checkout_authority(
    monkeypatch, tmp_path, invalid
):
    dev_entry = _entry()
    head = _head(dev_entry)
    package = dev_entry.REPO_ROOT / "tldw_chatbook"
    if invalid == "abbreviated":
        head = head[:12]
    elif invalid == "mismatch":
        head = "0" * 40
    elif invalid == "foreign_package":
        package = tmp_path / "tldw_chatbook"
    else:
        git = dev_entry._git
        monkeypatch.setattr(
            dev_entry,
            "_git",
            lambda root, *args: (
                "" if args == ("branch", "--show-current") else git(root, *args)
            ),
        )
    with pytest.raises(dev_entry.DevelopmentVoiceEntryError):
        dev_entry.validate_source_checkout(
            repo_root=dev_entry.REPO_ROOT,
            package_root=package,
            expected_head=head,
        )


def test_check_mode_reports_exact_checkout_without_starting_the_app(capsys) -> None:
    dev_entry = _entry()
    assert dev_entry.main(["--expect-head", _head(dev_entry), "--check"]) == 0

    output = capsys.readouterr().out
    assert f"head={_head(dev_entry)}" in output
    assert "mode=source-checkout-only" in output


@private_profile_test
async def test_development_entry_forces_pipeline_only_inside_its_process(
    monkeypatch, request
) -> None:
    dev_entry = _entry()
    from tldw_chatbook.UI.Console_Modules import hands_free

    original_gate = hands_free.speculative_voice_qualified
    original_resolver = hands_free.resolve_handsfree_engine

    def realtime() -> str:
        return "realtime"

    monkeypatch.setattr(hands_free, "resolve_handsfree_engine", realtime)

    head = _prepare_profile(request, dev_entry)
    with dev_entry.development_voice_enabled(expected_head=head):
        assert hands_free.speculative_voice_qualified() is True
        assert hands_free.resolve_handsfree_engine() == "pipeline"

    assert hands_free.speculative_voice_qualified is original_gate
    assert hands_free.resolve_handsfree_engine is realtime
    monkeypatch.setattr(hands_free, "resolve_handsfree_engine", original_resolver)


@pytest.mark.parametrize("legacy", [False, True])
@private_profile_test
async def test_development_entry_uses_commit_scoped_data_profile(request, legacy) -> None:
    dev_entry = _entry()
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import profile_paths

    original_profile = config.get_user_folder_name
    head = _prepare_profile(request, dev_entry, legacy=legacy)
    selected = config.get_cli_config_path()
    before = selected.read_bytes()
    data_root = selected.parent.parent / "voice-data"

    with dev_entry.development_voice_enabled(expected_head=head) as identity:
        assert identity.profile == f"speculative_voice_dev_{head[:12]}"
        assert config.get_user_folder_name() == identity.profile
        assert config.get_user_folder_name is original_profile
        values = config.get_atomic_config_snapshot().values
        data_dir = config.get_user_data_dir()
        assert values["general"]["users_name"] == identity.profile
        assert profile_paths.user_data_dir(values) == data_dir == data_root / identity.profile
        default = data_dir / profile_paths.database_leaf("chachanotes_db_path")
        assert config.get_chachanotes_db_path(ignore_override=True) == default
        assert config.get_chachanotes_db_path() == (
            data_root / "custom.db" if legacy else default
        )

    assert config.get_user_folder_name is original_profile
    assert selected.read_bytes() == before


@pytest.mark.asyncio
@private_profile_test
async def test_macos_say_tts_override_is_scoped_to_launcher_process(
    monkeypatch, request,
) -> None:
    dev_entry = _entry()
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        _LazyHandsFreeTts,
    )

    original_synthesize = _LazyHandsFreeTts.synthesize_hands_free

    async def render(text: str) -> bytes:
        assert text == "Voice system check."
        return b"RIFF-local-development-wav"

    monkeypatch.setattr(dev_entry, "_render_macos_say_wav", render)

    head = _prepare_profile(request, dev_entry)
    with dev_entry.development_voice_enabled(
        expected_head=head,
        macos_say_tts=True,
    ):
        response = await _LazyHandsFreeTts().synthesize_hands_free(
            text="Voice system check."
        )
        chunks = [chunk async for chunk in response.byte_stream]
        await response.aclose()

        assert response.provider_id == "macos_say_development"
        assert response.audio_format == "wav"
        assert response.sample_rate == 24_000
        assert response.metadata == {"channels": 1}
        assert chunks == [b"RIFF-local-development-wav"]

    assert _LazyHandsFreeTts.synthesize_hands_free is original_synthesize


def test_check_mode_runs_as_a_direct_source_script(tmp_path) -> None:
    dev_entry = _entry()
    result = subprocess.run(
        (
            sys.executable,
            str(dev_entry.__file__),
            "--expect-head",
            _head(dev_entry),
            "--check",
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"root={dev_entry.REPO_ROOT}" in result.stdout


@pytest.mark.parametrize(
    "mode", ["check", "missing-selection", "missing-config", "wrong-profile"]
)
def test_cold_entry_requires_profile_only_for_launch(tmp_path, mode) -> None:
    dev_entry = _entry()
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    env = {"HOME": str(home), "PATH": os.environ["PATH"], "GIT_OPTIONAL_LOCKS": "0"}
    if mode in ("wrong-profile", "missing-config"):
        selected = home / "config.toml"
        if mode == "wrong-profile":
            selected.write_text('[general]\nusers_name = "wrong_profile"\n')
            selected.chmod(0o600)
        env["TLDW_CONFIG_PATH"] = str(selected)
    child = r"""
import builtins, runpy, sys
path, head, mode = sys.argv[1:]
entry = runpy.run_path(path)
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    blocked = ('tldw_chatbook.config', 'tldw_chatbook.app',
               'tldw_chatbook.Backup_Recovery.launcher', 'tldw_chatbook.UI')
    package_import = name == 'tldw_chatbook' and bool(fromlist) and any(
        part in ('config', 'app', 'UI') for part in fromlist)
    if ((mode == 'check' and name.startswith('tldw_chatbook')) or package_import
            or any(name == root or name.startswith(root + '.') for root in blocked)):
        raise AssertionError('unexpected startup import: ' + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
args = ['--expect-head', head]
if mode == 'check':
    assert entry['main'](args + ['--check']) == 0
    assert not any(name.startswith('tldw_chatbook') for name in sys.modules)
else:
    try:
        entry['main'](args)
    except entry['DevelopmentVoiceEntryError'] as error:
        expected = {'missing-selection': 'TLDW_CONFIG_PATH',
                    'missing-config': 'already exist', 'wrong-profile': 'general.users_name'}[mode]
        assert expected in str(error), str(error)
    else:
        raise AssertionError('unprepared profile was accepted')
"""
    result = subprocess.run(
        (sys.executable, "-I", "-B", "-c", child, dev_entry.__file__, _head(dev_entry), mode),
        cwd=tmp_path,
        env=env,
        timeout=15,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    if mode == "missing-config":
        assert not selected.exists()


@pytest.mark.parametrize("result", [0, 7, "ordinary-result", RuntimeError])
@private_profile_test
async def test_app_launch_does_not_forward_development_flags(
    monkeypatch, request, result
) -> None:
    dev_entry = _entry()
    from tldw_chatbook import cli
    from tldw_chatbook.Backup_Recovery import launcher
    from tldw_chatbook.Chat.console_speculative_voice_session import _LazyHandsFreeTts
    from tldw_chatbook.UI.Console_Modules import hands_free

    head = _prepare_profile(request, dev_entry)
    calls = []
    real_unlock = launcher.startup_unlock
    original_path = sys.path[:]
    gate = hands_free.speculative_voice_qualified
    resolver = hands_free.resolve_handsfree_engine
    tts = _LazyHandsFreeTts.synthesize_hands_free

    def unlock():
        calls.append("unlock")
        return real_unlock()

    def run():
        calls.append(list(sys.argv))
        assert hands_free.speculative_voice_qualified() is True
        assert hands_free.resolve_handsfree_engine() == "pipeline"
        assert _LazyHandsFreeTts.synthesize_hands_free is dev_entry._synthesize_macos_say
        if result is RuntimeError:
            raise RuntimeError("app failed")
        return result

    monkeypatch.setattr(sys, "argv", ["launcher", "--outer-argument"])
    original_argv = sys.argv
    monkeypatch.setattr(launcher, "startup_unlock", unlock)
    monkeypatch.setattr(cli, "_run_prepared_app", run)
    args = ["--expect-head", head, "--macos-say-tts"]
    if result is RuntimeError:
        with pytest.raises(RuntimeError, match="app failed"):
            dev_entry.main(args)
    else:
        assert dev_entry.main(args) == (result if isinstance(result, int) else 0)
    assert calls == ["unlock", ["launcher"]]
    assert sys.argv is original_argv and sys.path == original_path
    assert hands_free.speculative_voice_qualified is gate
    assert hands_free.resolve_handsfree_engine is resolver
    assert _LazyHandsFreeTts.synthesize_hands_free is tts


@pytest.mark.asyncio
@private_profile_test
async def test_visible_hands_free_control_creates_speculative_session(
    monkeypatch, request,
) -> None:
    from Tests.UI.test_console_dictation_streaming import _patch_availability
    from Tests.UI.test_console_native_chat_flow import _ReadyResolutionGateway

    dev_entry = _entry()
    head = _prepare_profile(request, dev_entry)
    session = _QualifiedSession()
    _patch_availability(monkeypatch)

    with dev_entry.development_voice_enabled(expected_head=head):
        app, host = _ready_host()
        app.console_speculative_voice_session_factory = lambda **_kwargs: session

        async with host.run_test(size=(140, 42)) as pilot:
            console = await _mounted_console(host, pilot)
            monkeypatch.setattr(
                console._ensure_console_chat_controller().provider_gateway,
                "resolve_for_send",
                _ReadyResolutionGateway().resolve_for_send,
            )
            assert console._hands_free._console_hands_free_entry_refusal() is None
            assert await pilot.click("#console-hands-free-switch")
            await wait_for_hands_free_consent(host, pilot)
            assert session.entered_with == []
            await accept_hands_free_consent(host, pilot)
            await pilot.pause()

            selected: Any = console._console_hands_free
            assert type(selected) is ConsoleSpeculativeHandsFreeSession
            assert not isinstance(selected.controller, HandsFreeController)
            assert session.entered_with == [False]

            await pilot.click("#console-hands-free-switch")
            await pilot.pause()

    assert session.close_reasons == [ControlKind.HANDS_FREE_EXIT]


@private_profile_test
async def test_wrong_loaded_profile_stops_before_resolving_data(monkeypatch, request):
    dev_entry = _entry()
    from tldw_chatbook import config

    monkeypatch.setattr(
        config, "get_user_data_dir", lambda: pytest.fail("resolved a wrong profile")
    )
    with (
        pytest.raises(dev_entry.DevelopmentVoiceEntryError, match="loaded commit profile"),
        dev_entry.development_voice_enabled(expected_head=_head(dev_entry)),
    ):
        pytest.fail("wrong loaded profile admitted voice")


@private_profile_test
async def test_source_context_refuses_config_source_drift(monkeypatch, request, tmp_path):
    dev_entry = _entry()
    head = _prepare_profile(request, dev_entry)
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    original = config.read_cli_config_snapshot()
    foreign = tmp_path / "other.toml"
    with monkeypatch.context() as drift:
        drift.setenv("TLDW_CONFIG_PATH", str(foreign))
        with (
            pytest.raises(RecoveryRequired, match="raw_source_selection_changed"),
            dev_entry.development_voice_enabled(expected_head=head),
        ):
            pytest.fail("config source drift admitted voice")
    assert config.read_cli_config_snapshot() == original
    assert not foreign.exists()


_COLD_RECOVERY = r"""
import builtins, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts import run_speculative_voice_dev as entry
from tldw_chatbook.Backup_Recovery import bootstrap, launcher
selector = Path(os.environ['TLDW_CONFIG_PATH'])
mode, head = sys.argv[2:4]
if mode == 'pending':
    from tldw_chatbook.Backup_Recovery.control_records import register_pending
    register_pending(bootstrap.default_bootstrap_root(), 'voice-pending', ('voice-profile',),
                     selector.parent / 'control', (selector,))
elif mode.startswith('encrypted') or mode == 'malformed':
    general = '[general]\n' + ('' if mode.endswith('missing') else 'users_name="ordinary"\n')
    encryption = 'encryption=[]\n' if mode == 'malformed' else '[encryption]\nenabled=true\n'
    selector.write_text(encryption + general)
    os.environ['CHATBOOK_SERVED_CHILD'] = '1'
elif mode == 'unreadable':
    selector.chmod(0)
else:
    target = selector.with_name('real.toml')
    selector.rename(target)
    selector.symlink_to(target)
if mode == 'unreadable':
    selector.chmod(0o600)
before = selector.read_bytes()
if mode == 'unreadable':
    selector.chmod(0)
observed = []
launcher.minimal_recovery = lambda reason: observed.append(reason) or 17
real_unlock = launcher.startup_unlock
calls = []
def unlock():
    calls.append(1)
    return real_unlock()
launcher.startup_unlock = unlock
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name not in {'tldw_chatbook.config', 'tldw_chatbook.app'}, name
    assert not name.startswith(('tldw_chatbook.Chat', 'tldw_chatbook.UI', 'tldw_chatbook.TTS')), name
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
argv, path = sys.argv, sys.path[:]
assert entry.main(['--expect-head', head]) == 17
assert calls == [1]
expected = {'pending': 'recovery_pending', 'malformed': 'configuration_invalid'}
if mode.startswith('encrypted'):
    expected[mode] = launcher.UNLOCK_SERVED_REASON
assert observed == [expected.get(mode, 'configuration_unavailable')]
if mode == 'unreadable':
    selector.chmod(0o600)
assert selector.read_bytes() == before
assert sys.argv is argv and sys.path == path
assert 'tldw_chatbook.config' not in sys.modules
print('ORDINARY_RECOVERY_PASSED')
"""


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink/private recovery posture")
@pytest.mark.parametrize(
    "mode",
    ["pending", "unsafe", "encrypted", "encrypted-missing", "malformed", "unreadable"],
)
def test_source_entry_preserves_real_recovery_refusal_before_voice(tmp_path, mode):
    dev_entry = _entry()
    for name in ("home", "config", "data"):
        (tmp_path / name).mkdir(mode=0o700)
    selector = tmp_path / "config" / "config.toml"
    head = _head(dev_entry)
    selector.write_text(
        f"[general]\nusers_name = {json.dumps(f'speculative_voice_dev_{head[:12]}')}\n"
    )
    selector.chmod(0o600)
    environment = dict(os.environ)
    environment.update(
        HOME=str(tmp_path / "home"),
        USERPROFILE=str(tmp_path / "home"),
        XDG_CONFIG_HOME=str(tmp_path / "config"),
        XDG_DATA_HOME=str(tmp_path / "data"),
        TLDW_CONFIG_PATH=str(selector),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _COLD_RECOVERY,
            str(dev_entry.REPO_ROOT),
            mode,
            head,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORDINARY_RECOVERY_PASSED" in result.stdout
