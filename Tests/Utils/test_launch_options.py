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
    first, sentinel, stub = _compose_first(monkeypatch, _cli_no_splash=True)
    assert first is sentinel
    assert stub.splash_screen_active is False


@pytest.mark.bootstrap_profile
def test_without_the_flag_the_configured_splash_still_plays(monkeypatch):
    from tldw_chatbook.Widgets.splash_screen import SplashScreen

    first, _sentinel, stub = _compose_first(monkeypatch)
    assert isinstance(first, SplashScreen)
    assert stub.splash_screen_active is True


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
