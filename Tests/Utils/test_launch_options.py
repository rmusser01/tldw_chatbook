"""Launch flags for a second machine: ``--config PATH`` and ``--no-splash``.

TASK-34100.16 [coverage-10] / E6 step 1. ``tldw-cli --help`` listed only
``--serve --host --port --web-title --debug --focus``: there was no way to
point one launch at a copied config.toml except the ``TLDW_CONFIG_PATH``
environment variable, which ``--help`` never named, and no way to skip the
splash for one launch. These tests pin:

* the shared parser accepts ``--config PATH`` and ``--no-splash`` and ends
  its help with an epilog naming ``TLDW_CONFIG_PATH`` and the User Guide's
  "Setting up another machine" section;
* ``--config`` is exported as ``TLDW_CONFIG_PATH`` (the flag wins over an
  inherited value) before anything reads the profile, a folder is refused,
  and the helper imports neither config nor Textual;
* ``--no-splash`` reaches the app instance and the splash is not composed,
  without any config write;
* the real ``tldw-cli`` entry runs against the ``--config`` file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# No module-level ``tldw_chatbook.app`` import: whether that import passes the
# config-participant fence at collection depends on which suites were collected
# first (Tests/conftest.py, raw_source_selection_changed). The parser tests use
# the pure helper both entry points share; the real entry points are covered
# end to end in subprocesses below; the compose tests import the app inside a
# ``bootstrap_profile`` test body.

_REPO = Path(__file__).resolve().parents[2]


# --- the parser and its epilog ---------------------------------------------


def test_shared_parser_accepts_config_and_no_splash(tmp_path):
    from tldw_chatbook.Utils.launch_options import build_launch_parser

    target = tmp_path / "copied.toml"
    args = build_launch_parser().parse_args(["--config", str(target), "--no-splash"])
    assert args.config == str(target)
    assert args.no_splash is True
    plain = build_launch_parser().parse_args([])
    assert plain.config is None
    assert plain.no_splash is False


def test_help_ends_with_an_epilog_naming_the_env_var_and_the_guide_section():
    from tldw_chatbook.Utils.launch_options import build_launch_parser

    text = build_launch_parser().format_help()
    assert "--config PATH" in text
    assert "--no-splash" in text
    options_end = text.index("--no-splash")
    epilog = text[options_end:]
    assert "TLDW_CONFIG_PATH" in epilog
    assert "Setting up another machine" in epilog
    assert "Docs/User_Guide/First_Run_Setup.md" in epilog
    # The epilog is the end of --help, and its line breaks survive.
    last_lines = [line for line in text.splitlines() if line.strip()][-3:]
    assert any("Setting up another machine" in line for line in last_lines), text
    # A copied config can carry keys; the help says so where the route is.
    assert "plain text" in epilog


def test_help_points_an_installed_user_at_the_published_guide_section():
    """Review round 1: the wheel ships no Docs/, so a repo path alone dangles.

    The epilog ends with the guide's public URL, anchored at the section, and
    that section really exists in the guide (a renamed heading fails here).
    """
    from tldw_chatbook.Utils.launch_options import (
        SECOND_MACHINE_GUIDE,
        SECOND_MACHINE_SECTION,
        SECOND_MACHINE_URL,
        build_launch_parser,
    )

    anchor = SECOND_MACHINE_SECTION.lower().replace(" ", "-")
    assert SECOND_MACHINE_URL.startswith("https://")
    assert SECOND_MACHINE_URL.endswith(f"/{SECOND_MACHINE_GUIDE}#{anchor}")
    guide = (_REPO / SECOND_MACHINE_GUIDE).read_text(encoding="utf-8")
    assert f"\n## {SECOND_MACHINE_SECTION}\n" in guide
    text = build_launch_parser().format_help()
    assert [line for line in text.splitlines() if line.strip()][-1].strip() == (
        SECOND_MACHINE_URL
    )


def test_help_says_where_the_per_launch_flags_do_not_reach():
    """Review round 1: the --serve browser session is started without them.

    ``--config`` still reaches it through the exported TLDW_CONFIG_PATH, but
    ``--no-splash`` does not, and a missing folder is refused, so --help says
    both instead of promising more.
    """
    from tldw_chatbook.Utils.launch_options import build_launch_parser

    text = " ".join(build_launch_parser().format_help().split())
    assert "not a --serve browser session" in text
    assert "its folder must already exist" in text


# --- --config -> TLDW_CONFIG_PATH -------------------------------------------


def test_config_flag_is_exported_as_an_absolute_config_path(tmp_path, monkeypatch):
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    monkeypatch.chdir(tmp_path)
    environ: dict[str, str] = {}
    chosen = adopt_config_flag(["--config", "copied.toml", "--focus"], environ)
    expected = str(tmp_path / "copied.toml")
    assert chosen == expected
    assert environ == {"TLDW_CONFIG_PATH": expected}


def test_config_flag_wins_over_an_inherited_environment_value(tmp_path):
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    environ = {"TLDW_CONFIG_PATH": str(tmp_path / "from-env.toml")}
    adopt_config_flag([f"--config={tmp_path / 'from-flag.toml'}"], environ)
    assert environ["TLDW_CONFIG_PATH"] == str(tmp_path / "from-flag.toml")


def test_no_config_flag_leaves_the_environment_alone(tmp_path):
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    environ = {"TLDW_CONFIG_PATH": str(tmp_path / "from-env.toml")}
    assert adopt_config_flag(["--no-splash", "--help"], environ) is None
    assert environ == {"TLDW_CONFIG_PATH": str(tmp_path / "from-env.toml")}


def test_config_flag_refuses_a_folder(tmp_path, capsys):
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    environ: dict[str, str] = {}
    with pytest.raises(SystemExit) as stop:
        adopt_config_flag(["--config", str(tmp_path)], environ)
    assert stop.value.code == 2
    assert "is a folder" in capsys.readouterr().err
    assert environ == {}


def test_config_flag_refuses_an_empty_path(capsys):
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    with pytest.raises(SystemExit) as stop:
        adopt_config_flag(["--config", "  "], {})
    assert stop.value.code == 2
    assert "--config" in capsys.readouterr().err


def test_config_flag_refuses_a_file_whose_folder_does_not_exist(tmp_path, capsys):
    """Review round 1: a typo in the folder used to land on a recovery screen.

    Nothing can create a config there (the launch ended on "Recovery
    required: configuration_unavailable"), so it is a usage error naming the
    missing folder, before anything is read or written.
    """
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    missing = tmp_path / "not-yet" / "sub"
    environ: dict[str, str] = {}
    with pytest.raises(SystemExit) as stop:
        adopt_config_flag(["--config", str(missing / "config.toml")], environ)
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert f"{missing} does not exist" in err, err
    assert environ == {}
    assert not missing.exists()


def test_config_flag_accepts_a_missing_file_in_an_existing_folder(tmp_path):
    """The paired control: a new file in an existing folder is a first launch."""
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    environ: dict[str, str] = {}
    target = tmp_path / "new-config.toml"
    assert adopt_config_flag(["--config", str(target)], environ) == str(target)
    assert environ == {"TLDW_CONFIG_PATH": str(target)}
    assert not target.exists()


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that apply to this user",
)
def test_config_flag_under_an_unreadable_folder_is_a_usage_error(tmp_path, capsys):
    """Review round 1: ``Path.is_dir()`` raises PermissionError on Python 3.12.

    argparse turns only ArgumentTypeError/TypeError/ValueError from a
    ``type=`` function into a usage error, so the PermissionError escaped as
    a raw traceback before the startup fence.
    """
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    environ: dict[str, str] = {}
    try:
        with pytest.raises(SystemExit) as stop:
            adopt_config_flag(["--config", str(locked / "config.toml")], environ)
    finally:
        locked.chmod(0o700)
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot read {locked / 'config.toml'}" in err, err
    assert "Traceback" not in err
    assert environ == {}


_POSIX_PERMISSIONS = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="needs POSIX permissions that apply to this user",
)


@_POSIX_PERMISSIONS
def test_config_flag_refuses_a_config_file_it_cannot_read(tmp_path, capsys):
    """Review round 2: ``os.stat`` succeeds on a mode-000 file, so it passed.

    The launch then ended on the Backup & Restore recovery screen ("Recovery
    required: configuration_unavailable") with no explanation -- the outcome
    round 1 removed for a missing folder.
    """
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    target = tmp_path / "config.toml"
    target.write_text("[general]\n")
    target.chmod(0)
    environ: dict[str, str] = {}
    try:
        with pytest.raises(SystemExit) as stop:
            adopt_config_flag(["--config", str(target)], environ)
    finally:
        target.chmod(0o600)
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot read {target}" in err, err
    assert environ == {}


@_POSIX_PERMISSIONS
def test_config_flag_refuses_a_new_file_in_a_folder_it_cannot_write(tmp_path, capsys):
    """Review round 2: nothing can create the config there.

    The launch exited through "Chatbook cannot start: its private storage
    location…" (advice about group- or world-writable folders, which does not
    fit) followed by a PrivatePathError traceback on ``python -m``.
    """
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    folder = tmp_path / "read-only"
    folder.mkdir()
    folder.chmod(0o500)
    environ: dict[str, str] = {}
    try:
        with pytest.raises(SystemExit) as stop:
            adopt_config_flag(["--config", str(folder / "config.toml")], environ)
    finally:
        folder.chmod(0o700)
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert f"cannot create {folder / 'config.toml'}" in err, err
    assert "Traceback" not in err
    assert environ == {}
    assert not (folder / "config.toml").exists()


@_POSIX_PERMISSIONS
def test_config_flag_accepts_a_read_only_config_the_startup_fence_admits(tmp_path):
    """The paired control: an existing file needs to be readable, not writable.

    A read-only config.toml, and a readable one in a read-only folder, both
    pass the startup fence today (probed with ``python -m … --config X
    --help``), so ``--config`` must not refuse them.
    """
    from tldw_chatbook.Utils.launch_options import adopt_config_flag

    read_only = tmp_path / "read-only.toml"
    read_only.write_text("[general]\n")
    read_only.chmod(0o400)
    folder = tmp_path / "read-only-folder"
    folder.mkdir()
    inside = folder / "config.toml"
    inside.write_text("[general]\n")
    folder.chmod(0o500)
    try:
        for target in (read_only, inside):
            environ: dict[str, str] = {}
            assert adopt_config_flag(["--config", str(target)], environ) == str(target)
            assert environ == {"TLDW_CONFIG_PATH": str(target)}
    finally:
        folder.chmod(0o700)
        read_only.chmod(0o600)


def test_launch_options_import_neither_config_nor_textual():
    """It runs before the ADR-126 fence, so it must stay import-light."""
    probe = (
        "import sys\n"
        "import tldw_chatbook.Utils.launch_options as m\n"
        "m.build_launch_parser().format_help()\n"
        "bad = [n for n in ('tldw_chatbook.config', 'textual', 'loguru', 'tldw_chatbook.app') if n in sys.modules]\n"
        "print('HEAVY=' + ','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "HEAVY=\n" in result.stdout, result.stdout


def test_cli_refuses_config_together_with_recovery_profile_selectors(
    tmp_path, monkeypatch, capsys
):
    """A recovery profile selects its own config; the two never mix."""
    import types

    calls = []
    launcher = types.ModuleType("tldw_chatbook.Backup_Recovery.launcher")
    launcher.startup_unlock = lambda: calls.append("unlock")
    launcher.recovery_main = lambda argv: 0
    monkeypatch.setitem(sys.modules, "tldw_chatbook.Backup_Recovery.launcher", launcher)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tldw-cli",
            "--config",
            str(tmp_path / "copied.toml"),
            "--recovery-profile",
            "p",
            "--recovery-control-root",
            str(tmp_path / "control"),
        ],
    )
    before = os.environ.get("TLDW_CONFIG_PATH")
    from tldw_chatbook import cli

    with pytest.raises(SystemExit) as stop:
        cli.main_cli_runner()
    assert stop.value.code == 2
    assert "--config" in capsys.readouterr().err
    assert calls == []
    assert os.environ.get("TLDW_CONFIG_PATH") == before


def test_cli_adopts_config_before_the_startup_unlock(tmp_path, monkeypatch):
    """The one unlock (TASK-34100.4) must already see the selected file."""
    import types

    seen = []
    launcher = types.ModuleType("tldw_chatbook.Backup_Recovery.launcher")
    launcher.startup_unlock = lambda: seen.append(os.environ.get("TLDW_CONFIG_PATH")) or 7
    launcher.recovery_main = lambda argv: 0
    monkeypatch.setitem(sys.modules, "tldw_chatbook.Backup_Recovery.launcher", launcher)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "from-env.toml"))
    target = tmp_path / "copied.toml"
    monkeypatch.setattr(sys, "argv", ["tldw-cli", "--config", str(target)])
    from tldw_chatbook import cli

    assert cli.main_cli_runner() == 7
    assert seen == [str(target)]


# --- --no-splash ------------------------------------------------------------


def test_launch_options_reach_the_app_instance():
    from tldw_chatbook.Utils.launch_options import apply_launch_options

    app = SimpleNamespace()
    apply_launch_options(app, SimpleNamespace(focus=True, no_splash=True))
    assert app._cli_focus_override is True
    assert app._cli_no_splash is True
    apply_launch_options(app, SimpleNamespace(focus=False, no_splash=False))
    assert app._cli_focus_override is False
    assert app._cli_no_splash is False


_CONFIG_WRITERS = (
    "save_setting_to_cli_config",
    "save_settings_to_cli_config",
    "replace_cli_config_serialized",
)


def _forbid_config_writes(monkeypatch) -> list[str]:
    """Make every config writer a recorded failure, wherever it is bound.

    Review round 1 (AC#3, "neither flag writes to config"): the flagged
    launches below must not reach a writer. Returns the calls seen, so a
    writer whose exception is swallowed by the code under test still fails.
    """
    from Tests.app_module_patches import set_app_global
    from tldw_chatbook import config

    calls: list[str] = []

    def writer(name):
        def refuse(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"a launch flag reached {name}{args!r}")

        return refuse

    for name in _CONFIG_WRITERS:
        monkeypatch.setattr(config, name, writer(name))
        if name == "save_setting_to_cli_config":
            set_app_global(monkeypatch, name, writer(name))
    return calls


def _config_bytes() -> bytes | None:
    from tldw_chatbook.config import get_cli_config_path

    path = Path(get_cli_config_path())
    return path.read_bytes() if path.exists() else None


def _compose_first(monkeypatch, **attrs):
    from Tests.app_module_patches import set_app_global
    from tldw_chatbook.app import TldwCli

    set_app_global(
        monkeypatch,
        "get_cli_setting",
        lambda section, key=None, default=None: True
        if (section, key) == ("splash_screen", "enabled")
        else default,
    )
    sentinel = object()
    stub = SimpleNamespace(
        _compose_main_ui=lambda: iter([sentinel]),
        splash_screen_active=False,
        **attrs,
    )
    first = next(iter(TldwCli.compose(stub)))
    return first, sentinel, stub


# These import the app and compose the splash, which reads config through the
# real getters, so they keep the collection-time profile (Tests/conftest.py).
@pytest.mark.bootstrap_profile
def test_no_splash_composes_the_main_ui_even_when_config_enables_the_splash(
    monkeypatch,
):
    writes = _forbid_config_writes(monkeypatch)
    before = _config_bytes()
    first, sentinel, stub = _compose_first(monkeypatch, _cli_no_splash=True)
    assert first is sentinel
    assert stub.splash_screen_active is False
    # Skipping the splash is per launch: nothing is persisted.
    assert writes == []
    assert _config_bytes() == before


@pytest.mark.bootstrap_profile
def test_without_the_flag_the_configured_splash_still_plays(monkeypatch):
    """The paired control for the test above (it passes on pre-fix code too).

    With no flag the configured splash still composes, so the test above
    fails for the flag's reason, not because the splash never composes.
    """
    from tldw_chatbook.Widgets.splash_screen import SplashScreen

    first, _sentinel, stub = _compose_first(monkeypatch)
    assert isinstance(first, SplashScreen)
    assert stub.splash_screen_active is True


# --- the flags reach the app through both real runners ----------------------


class _RecordingApp:
    """Stands in for TldwCli in a runner: records what reached it, never runs."""

    instances: list[_RecordingApp] = []

    def __init__(self) -> None:
        _RecordingApp.instances.append(self)
        self.seen: tuple[object, object] | None = None

    def run(self) -> None:
        self.seen = (
            getattr(self, "_cli_no_splash", None),
            getattr(self, "_cli_focus_override", None),
        )


def _quiet_runner(monkeypatch, app_entry) -> None:
    """Stub every process-global side effect a runner has before ``run()``.

    Signal handlers, the exit watchdog, logging sinks, the CSS build, the
    terminal image probe, metrics servers and the config ensure/migrate all
    touch the test process or the profile; none of them is under test here.
    """
    from tldw_chatbook import config
    from tldw_chatbook.Tools import workspace_file_roots
    from tldw_chatbook.Utils import db_upgrade_notice, terminal_utils

    noop = lambda *args, **kwargs: None  # noqa: E731 -- a stub
    for name in (
        "install_termination_handlers",
        "arm_exit_watchdog",
        "initialize_early_logging",
        "load_cli_config_and_ensure_existence",
        "init_metrics_server",
        "init_otel_metrics",
    ):
        monkeypatch.setattr(app_entry, name, noop)
    monkeypatch.setattr(app_entry, "_is_source_tree", lambda root: False)
    monkeypatch.setattr(app_entry, "supports_emoji", lambda: False)
    monkeypatch.setattr(app_entry, "TldwCli", _RecordingApp)
    monkeypatch.setattr(config, "migrate_config_file_if_needed", noop)
    monkeypatch.setattr(workspace_file_roots, "set_launch_cwd", noop)
    monkeypatch.setattr(terminal_utils, "warm_up_image_protocol", noop)
    monkeypatch.setattr(db_upgrade_notice, "print_db_upgrade_notice_if_pending", noop)
    monkeypatch.setenv("TORCHAUDIO_LOG_LEVEL", "ERROR")


@pytest.fixture
def _restore_logger_levels():
    """main_cli_runner quiets some third-party loggers; put them back after."""
    import logging

    manager = logging.root.manager
    before = {
        name: logger.level
        for name, logger in manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    yield
    for name, logger in list(manager.loggerDict.items()):
        if isinstance(logger, logging.Logger):
            logger.setLevel(before.get(name, logging.NOTSET))


@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("runner", ["main_cli_runner", "_run_module_main"])
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--no-splash"], (True, False)),
        (["--no-splash", "--focus"], (True, True)),
        ([], (False, False)),  # the control: no flag, no skip
    ],
)
@pytest.mark.usefixtures("_restore_logger_levels")
def test_launch_flags_reach_the_app_through_both_real_runners(
    monkeypatch, runner, argv, expected
):
    """Review round 1: the wiring itself, not the helper in isolation.

    ``tldw-cli`` ends in ``app_entry.main_cli_runner`` and ``python -m
    tldw_chatbook.app`` in ``app_entry._run_module_main``. If either stopped
    handing the parsed flags to the instance, ``--no-splash`` would silently
    stop working while the helper's own test stayed green.
    """
    from tldw_chatbook import app_entry

    _quiet_runner(monkeypatch, app_entry)
    writes = _forbid_config_writes(monkeypatch)
    before = _config_bytes()
    _RecordingApp.instances = []
    monkeypatch.setattr(sys, "argv", ["tldw-cli", *argv])
    getattr(app_entry, runner)()
    assert [app.seen for app in _RecordingApp.instances] == [expected]
    assert writes == []
    assert _config_bytes() == before


_ADMISSION_PROBE = r"""
import os, runpy, sys
from tldw_chatbook.Backup_Recovery import storage_admission
from tldw_chatbook.Backup_Recovery.profile_paths import effective_config_path

class Admitted(BaseException):
    pass

def admit_startup():
    print('ADMITTED=' + str(effective_config_path()), flush=True)
    raise Admitted

storage_admission.admit_startup = admit_startup
sys.argv = ['tldw_chatbook.app', *sys.argv[1:]]
try:
    runpy.run_module('tldw_chatbook.app', run_name='__main__', alter_sys=True)
except Admitted:
    pass
"""


@pytest.mark.timeout(120)
def test_module_entry_admits_the_config_flag_file_not_the_inherited_one(tmp_path):
    """Review round 1: ``--config`` must be adopted BEFORE the ADR-126 fence.

    The unlock and config-load checks cannot see admission (it writes no
    config file), so this observes the profile ``admit_startup()`` is asked
    to admit when ``python -m tldw_chatbook.app --config X`` runs while
    TLDW_CONFIG_PATH names another file.
    """
    root = tmp_path.resolve()
    env = _isolated_env(root)
    target = root / "elsewhere" / "copied-config.toml"
    target.parent.mkdir(mode=0o700)
    result = subprocess.run(
        [sys.executable, "-c", _ADMISSION_PROBE, "--config", str(target)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert f"ADMITTED={target}\n" in result.stdout, result.stdout
    # The control: without the flag, the inherited selector is admitted.
    control = subprocess.run(
        [sys.executable, "-c", _ADMISSION_PROBE],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=100,
        check=False,
    )
    assert f"ADMITTED={root / 'config' / 'decoy.toml'}\n" in control.stdout, (
        control.stdout + control.stderr[-2000:]
    )


# --- the real tldw-cli entry -------------------------------------------------


def _isolated_env(root: Path) -> dict[str, str]:
    for name in ("home", "config", "data"):
        (root / name).mkdir(mode=0o700, exist_ok=True)
    env = os.environ.copy()
    env.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_DATA_HOME=str(root / "data"),
        # The flag must win over this inherited selector.
        TLDW_CONFIG_PATH=str(root / "config" / "decoy.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    env.pop("TLDW_VERBOSE_STARTUP", None)
    return env


_COPIED_CONFIG = """[general]
users_name = "copied"

[splash_screen]
enabled = true

[first_run]
setup_started = false
"""


_ENTRY_POINTS = {
    # The packaged `tldw-cli` console script.
    "tldw-cli": [
        "-c",
        "import sys; from tldw_chatbook.cli import main_cli_runner; "
        "sys.exit(main_cli_runner())",
    ],
    "python -m tldw_chatbook.app": ["-m", "tldw_chatbook.app"],
}


@pytest.mark.timeout(240)
@pytest.mark.parametrize("entry", sorted(_ENTRY_POINTS))
def test_real_entry_points_run_that_launch_against_the_config_flag(tmp_path, entry):
    root = tmp_path.resolve()
    env = _isolated_env(root)
    target = root / "elsewhere" / "copied-config.toml"
    target.parent.mkdir(mode=0o700)
    target.write_text(_COPIED_CONFIG)
    target.chmod(0o600)
    result = subprocess.run(
        [
            sys.executable,
            *_ENTRY_POINTS[entry],
            "--config",
            str(target),
            "--no-splash",
            "--help",
        ],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=200,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert "unrecognized arguments" not in result.stderr
    # --help ends with the second-machine epilog on both entry points.
    tail = [line for line in result.stdout.splitlines() if line.strip()][-3:]
    assert any("Setting up another machine" in line for line in tail), result.stdout
    assert "TLDW_CONFIG_PATH" in result.stdout
    # Every launch ensures its effective config exists, so neither the
    # inherited TLDW_CONFIG_PATH nor the default profile being absent means
    # the launch ran against the flagged file.
    assert not (root / "config" / "decoy.toml").exists()
    assert not (root / "home" / ".config" / "tldw_cli" / "config.toml").exists()
    # Neither flag wrote to it: the splash setting and [first_run] are as
    # copied (no silent setup_completed).
    assert target.read_text() == _COPIED_CONFIG
