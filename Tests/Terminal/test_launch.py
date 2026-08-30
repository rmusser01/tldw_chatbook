"""Focused contracts for persistent-terminal launch policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tldw_chatbook.Terminal import launch as launch_module
from tldw_chatbook.Terminal.launch import (
    ResolvedLaunch,
    ShellChoice,
    build_terminal_environment,
    discover_shell_choices,
    normalize_session_name,
    resolve_shell_choice,
    resolve_start_directory,
    session_name_key,
)


def _lookup(paths: dict[str, str]):
    return paths.get


def _all_paths_are_files(_path: Path) -> bool:
    return True


def test_session_name_is_trimmed_and_nfc_normalized() -> None:
    assert (
        normalize_session_name("  Cafe\N{COMBINING ACUTE ACCENT}  ")
        == "Caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    )


@pytest.mark.parametrize("name", ["", "   ", "x" * 65])
def test_session_name_refuses_blank_or_more_than_64_characters(name: str) -> None:
    with pytest.raises(ValueError, match="1 to 64"):
        normalize_session_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "bad\nname",
        "bad\x00name",
        "bad\N{RIGHT-TO-LEFT OVERRIDE}name",
        "bad\ud800name",
        "[bold]name[/bold]",
    ],
)
def test_session_name_refuses_controls_and_rich_markup(name: str) -> None:
    with pytest.raises(ValueError):
        normalize_session_name(name)


def test_session_name_key_detects_unicode_casefold_collisions() -> None:
    assert session_name_key("Straße") == session_name_key("STRASSE")


def test_session_name_refuses_casefold_duplicate_live_name() -> None:
    with pytest.raises(ValueError, match="unique"):
        normalize_session_name("STRASSE", existing_names=("Straße",))


def test_posix_default_prefers_valid_account_shell_even_outside_picker_families() -> (
    None
):
    choices = discover_shell_choices(
        platform_name="posix",
        account_shell="/opt/fish/bin/fish",
        executable_lookup=_lookup(
            {"bash": "/bin/bash", "zsh": "/bin/zsh", "sh": "/bin/sh"}
        ),
        executable_is_file=_all_paths_are_files,
    )

    default = resolve_shell_choice("default", choices)
    assert default.family == "account"
    assert default.executable == Path("/opt/fish/bin/fish")
    assert default.argv == ("-fish",)
    assert {choice.key for choice in choices} == {"default", "bash", "zsh"}


def test_posix_default_reads_current_account_shell_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        launch_module,
        "_read_posix_account_shell",
        lambda: "/opt/fish/bin/fish",
        raising=False,
    )

    choices = discover_shell_choices(
        platform_name="posix",
        executable_lookup=_lookup({"bash": "/bin/bash", "sh": "/bin/sh"}),
        executable_is_file=_all_paths_are_files,
    )

    assert resolve_shell_choice("default", choices).executable == Path(
        "/opt/fish/bin/fish"
    )


def test_posix_default_falls_back_to_bash_then_sh() -> None:
    bash_choices = discover_shell_choices(
        platform_name="posix",
        account_shell="relative/fish",
        executable_lookup=_lookup({"bash": "/bin/bash", "sh": "/bin/sh"}),
        executable_is_file=_all_paths_are_files,
    )
    sh_choices = discover_shell_choices(
        platform_name="posix",
        account_shell="relative/unavailable",
        executable_lookup=_lookup({"sh": "/bin/sh"}),
        executable_is_file=_all_paths_are_files,
    )

    assert resolve_shell_choice("default", bash_choices).executable == Path("/bin/bash")
    assert resolve_shell_choice("default", sh_choices).executable == Path("/bin/sh")


def test_posix_named_shell_argv_is_login_interactive_and_code_owned() -> None:
    choices = discover_shell_choices(
        platform_name="posix",
        account_shell="/bin/bash",
        executable_lookup=_lookup({"bash": "/bin/bash", "zsh": "/bin/zsh"}),
        executable_is_file=_all_paths_are_files,
    )

    assert resolve_shell_choice("bash", choices).argv == (
        "/bin/bash",
        "--login",
        "-i",
    )
    assert resolve_shell_choice("zsh", choices).argv == ("/bin/zsh", "-l", "-i")


def test_windows_default_prefers_pwsh_then_powershell_then_cmd() -> None:
    pwsh = r"C:\Program Files\PowerShell\7\pwsh.exe"
    powershell = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
    cmd = r"C:\Windows\System32\cmd.exe"

    first = discover_shell_choices(
        platform_name="nt",
        executable_lookup=_lookup(
            {"pwsh.exe": pwsh, "powershell.exe": powershell, "cmd.exe": cmd}
        ),
        executable_is_file=_all_paths_are_files,
    )
    second = discover_shell_choices(
        platform_name="nt",
        executable_lookup=_lookup({"powershell.exe": powershell, "cmd.exe": cmd}),
        executable_is_file=_all_paths_are_files,
    )
    third = discover_shell_choices(
        platform_name="nt",
        executable_lookup=_lookup({"cmd.exe": cmd}),
        executable_is_file=_all_paths_are_files,
    )

    assert resolve_shell_choice("default", first).executable == Path(pwsh)
    assert resolve_shell_choice("default", second).executable == Path(powershell)
    assert resolve_shell_choice("default", third).executable == Path(cmd)


def test_windows_argv_keeps_profiles_and_interactive_input_enabled() -> None:
    pwsh = r"C:\Program Files\PowerShell\7\pwsh.exe"
    cmd = r"C:\Windows\System32\cmd.exe"
    choices = discover_shell_choices(
        platform_name="nt",
        executable_lookup=_lookup({"pwsh.exe": pwsh, "cmd.exe": cmd}),
        executable_is_file=_all_paths_are_files,
    )

    assert resolve_shell_choice("pwsh", choices).argv == (pwsh, "-NoLogo")
    assert resolve_shell_choice("cmd", choices).argv == (cmd, "/Q")
    flattened = "\n".join(argument for choice in choices for argument in choice.argv)
    for forbidden in ("-NoProfile", "-NonInteractive", "/C", "fixture-command"):
        assert forbidden not in flattened


def test_shell_picker_has_no_arbitrary_executable_entry() -> None:
    choices = discover_shell_choices(
        platform_name="posix",
        account_shell="/opt/custom/bin/fish",
        executable_lookup=_lookup({"bash": "/bin/bash", "zsh": "/bin/zsh"}),
        executable_is_file=_all_paths_are_files,
    )

    with pytest.raises(ValueError, match="shell choice"):
        resolve_shell_choice("/tmp/model-supplied", choices)
    assert all(choice.key != "/opt/custom/bin/fish" for choice in choices)


def test_shell_discovery_fails_closed_when_no_default_exists() -> None:
    with pytest.raises(FileNotFoundError, match="shell"):
        discover_shell_choices(
            platform_name="posix",
            account_shell="relative/unavailable",
            executable_lookup=_lookup({}),
            executable_is_file=_all_paths_are_files,
        )


def test_shell_values_are_frozen_slotted_and_deeply_immutable() -> None:
    choice = ShellChoice(
        key="bash",
        label="Bash",
        family="bash",
        executable=Path("/bin/bash"),
        argv=("/bin/bash", "--login", "-i"),
    )
    launch = ResolvedLaunch(
        name="Terminal 1",
        shell=choice,
        start_directory=Path("/tmp"),
        environment=(("TERM", "linux"),),
    )

    with pytest.raises(FrozenInstanceError):
        choice.key = "zsh"  # type: ignore[misc]
    with pytest.raises(TypeError):
        launch.environment[0] = ("TERM", "xterm")  # type: ignore[index]
    assert not hasattr(choice, "__dict__")
    assert not hasattr(launch, "__dict__")


def test_requested_start_directory_precedes_selected_root_and_real_home(
    tmp_path: Path,
) -> None:
    requested = tmp_path / "requested"
    selected = tmp_path / "selected"
    home = tmp_path / "home"
    requested.mkdir()
    selected.mkdir()
    home.mkdir()

    assert (
        resolve_start_directory(
            selected,
            requested_directory=requested,
            account_home=home,
        )
        == requested
    )


def test_selected_ready_local_root_precedes_real_home(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    home = tmp_path / "home"
    selected.mkdir()
    home.mkdir()

    assert resolve_start_directory(selected, account_home=home) == selected


def test_missing_selection_falls_back_to_real_home(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    assert resolve_start_directory(None, account_home=home) == home


@pytest.mark.parametrize("which", ["requested", "selected", "home"])
def test_start_directory_revalidates_final_absolute_existing_directory(
    tmp_path: Path,
    which: str,
) -> None:
    existing_home = tmp_path / "home"
    existing_home.mkdir()
    selected = tmp_path if which != "selected" else tmp_path / "missing-selected"
    requested = None if which != "requested" else tmp_path / "missing-requested"
    home = existing_home if which != "home" else tmp_path / "missing-home"

    with pytest.raises(ValueError, match="absolute existing directory"):
        resolve_start_directory(
            selected if which == "selected" else None,
            requested_directory=requested,
            account_home=home,
        )


def test_posix_environment_uses_account_reader_and_scrubs_ambient_secrets(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    temp = tmp_path / "tmp"
    for path in (bin_dir, home, temp):
        path.mkdir()
    ambient = {
        "PATH": f"relative{os.pathsep}{bin_dir}{os.pathsep}{bin_dir}",
        "HOME": "/ambient/home",
        "USER": "ambient-user",
        "LOGNAME": "ambient-user",
        "SHELL": "/ambient/shell",
        "TMPDIR": "/ambient/tmp",
        "LANG": "en_US.UTF-8",
        "LC_CTYPE": "UTF-8",
        "LC_PRIVATE_TOKEN": "must-not-cross-boundary",
        "COLORTERM": "truecolor",
        "LINES": "999",
        "COLUMNS": "999",
        "OPENAI_API_KEY": "secret",
        "HTTPS_PROXY": "credential",
        "OTEL_EXPORTER_OTLP_HEADERS": "trace-secret",
        "PYTHONPATH": "/inject",
        "PYTHONHOME": "/inject-home",
        "SSH_AUTH_SOCK": "/agent.sock",
        "UNRELATED": "drop-me",
    }

    environment = build_terminal_environment(
        platform_name="posix",
        ambient=ambient,
        account_reader=lambda: {
            "HOME": str(home),
            "USER": "account-user",
            "LOGNAME": "account-user",
            "SHELL": "/bin/zsh",
        },
        system_reader=lambda: {"TMPDIR": str(temp)},
    )

    assert environment == {
        "PATH": str(bin_dir),
        "HOME": str(home),
        "USER": "account-user",
        "LOGNAME": "account-user",
        "SHELL": "/bin/zsh",
        "TMPDIR": str(temp),
        "LANG": "en_US.UTF-8",
        "LC_CTYPE": "UTF-8",
        "TERM": "linux",
    }


def test_windows_environment_uses_account_and_system_readers_not_ambient() -> None:
    ambient = {
        "Path": r"C:\Tools;relative;C:\Windows\System32",
        "USERPROFILE": r"C:\ambient-user",
        "SYSTEMROOT": r"C:\ambient-windows",
        "PSModulePath": r"C:\ambient-modules",
        "OPENAI_API_KEY": "secret",
        "HTTP_PROXY": "credential",
        "PYTHONPATH": r"C:\inject",
        "TEMP": r"C:\ambient-temp",
        "LANG": "en-US",
    }
    existing = {r"C:\Tools", r"C:\Windows\System32"}

    environment = build_terminal_environment(
        platform_name="nt",
        ambient=ambient,
        account_reader=lambda: {
            "USERPROFILE": r"C:\Users\account",
            "HOMEDRIVE": "C:",
            "HOMEPATH": r"\Users\account",
            "USERNAME": "account",
        },
        system_reader=lambda: {
            "APPDATA": r"C:\Users\account\AppData\Roaming",
            "LOCALAPPDATA": r"C:\Users\account\AppData\Local",
            "PROGRAMDATA": r"C:\ProgramData",
            "PROGRAMFILES": r"C:\Program Files",
            "SYSTEMROOT": r"C:\Windows",
            "WINDIR": r"C:\Windows",
            "COMSPEC": r"C:\Windows\System32\cmd.exe",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
            "TEMP": r"C:\Users\account\AppData\Local\Temp",
            "TMP": r"C:\Users\account\AppData\Local\Temp",
        },
        path_is_directory=lambda path: path in existing,
    )

    assert environment == {
        "PATH": r"C:\Tools;C:\Windows\System32",
        "USERPROFILE": r"C:\Users\account",
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\account",
        "USERNAME": "account",
        "APPDATA": r"C:\Users\account\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\account\AppData\Local",
        "PROGRAMDATA": r"C:\ProgramData",
        "PROGRAMFILES": r"C:\Program Files",
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "TEMP": r"C:\Users\account\AppData\Local\Temp",
        "TMP": r"C:\Users\account\AppData\Local\Temp",
        "LANG": "en-US",
        "TERM": "linux",
    }
    for forbidden in (
        "PSMODULEPATH",
        "COLORTERM",
        "LINES",
        "COLUMNS",
        "OPENAI_API_KEY",
        "HTTP_PROXY",
        "PYTHONPATH",
    ):
        assert forbidden not in environment


def test_default_windows_readers_use_platform_environment_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform_values = {
        "USERPROFILE": r"C:\Users\platform",
        "HOMEDRIVE": "C:",
        "HOMEPATH": r"\Users\platform",
        "USERNAME": "platform-user",
        "APPDATA": r"C:\Users\platform\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\platform\AppData\Local",
        "PROGRAMDATA": r"C:\ProgramData",
        "PROGRAMFILES": r"C:\Program Files",
        "PROGRAMFILES(X86)": r"C:\Program Files (x86)",
        "PROGRAMW6432": r"C:\Program Files",
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "TEMP": r"C:\Users\platform\AppData\Local\Temp",
        "TMP": r"C:\Users\platform\AppData\Local\Temp",
        "UNRELATED": "must-not-cross-boundary",
    }
    monkeypatch.setenv("USERPROFILE", r"C:\Users\ambient")
    monkeypatch.setenv("SYSTEMROOT", r"C:\ambient-windows")
    monkeypatch.setattr(
        launch_module,
        "_read_windows_environment_block",
        lambda: platform_values,
        raising=False,
    )

    assert launch_module._read_account_values("nt") == {
        key: platform_values[key]
        for key in ("USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USERNAME")
    }
    assert launch_module._read_system_values("nt") == {
        key: value
        for key, value in platform_values.items()
        if key not in {"USERPROFILE", "HOMEDRIVE", "HOMEPATH", "USERNAME", "UNRELATED"}
    }


def test_windows_environment_block_parser_normalizes_keys_and_ignores_drive_state() -> (
    None
):
    import ctypes

    block = ctypes.create_unicode_buffer(
        "Path=C:\\Tools\x00=C:=C:\\work\x00UserProfile=C:\\Users\\account\x00\x00"
    )

    assert launch_module._parse_windows_environment_block(block) == {
        "PATH": r"C:\Tools",
        "USERPROFILE": r"C:\Users\account",
    }


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows profile APIs")
def test_native_windows_environment_reader_ignores_process_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = launch_module._read_windows_environment_block()
    monkeypatch.setenv("USERPROFILE", r"C:\ambient-override")
    monkeypatch.setenv("SYSTEMROOT", r"C:\ambient-system-override")

    observed = launch_module._read_windows_environment_block()

    assert observed["USERPROFILE"] == expected["USERPROFILE"]
    assert observed["SYSTEMROOT"] == expected["SYSTEMROOT"]


@pytest.mark.parametrize(
    "missing",
    (
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
    ),
)
def test_windows_environment_requires_qualified_platform_values(missing: str) -> None:
    system = {
        "APPDATA": r"C:\Users\account\AppData\Roaming",
        "LOCALAPPDATA": r"C:\Users\account\AppData\Local",
        "PROGRAMDATA": r"C:\ProgramData",
        "PROGRAMFILES": r"C:\Program Files",
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        "TEMP": r"C:\Users\account\AppData\Local\Temp",
        "TMP": r"C:\Users\account\AppData\Local\Temp",
    }
    system.pop(missing)

    with pytest.raises(ValueError, match=missing.replace("(", r"\(")):
        build_terminal_environment(
            platform_name="nt",
            ambient={"PATH": r"C:\Tools"},
            account_reader=lambda: {
                "USERPROFILE": r"C:\Users\account",
                "HOMEDRIVE": "C:",
                "HOMEPATH": r"\Users\account",
                "USERNAME": "account",
            },
            system_reader=lambda: system,
            path_is_directory=lambda path: path == r"C:\Tools",
        )


def test_environment_rejects_path_without_an_existing_absolute_directory() -> None:
    with pytest.raises(ValueError, match="PATH"):
        build_terminal_environment(
            platform_name="nt",
            ambient={"PATH": r"relative;C:\missing"},
            account_reader=lambda: {
                "USERPROFILE": r"C:\Users\account",
                "HOMEDRIVE": "C:",
                "HOMEPATH": r"\Users\account",
                "USERNAME": "account",
            },
            system_reader=lambda: {},
            path_is_directory=lambda _path: False,
            fallback_path="",
        )


@pytest.mark.parametrize(
    ("shell_name", "profile_name"),
    [("bash", ".bash_profile"), ("zsh", ".zprofile")],
)
def test_posix_login_shell_runs_fresh_profile_from_scrubbed_environment(
    tmp_path: Path,
    shell_name: str,
    profile_name: str,
) -> None:
    executable = shutil.which(shell_name)
    if executable is None:
        pytest.skip(f"{shell_name} is unavailable")
    home = tmp_path / f"{shell_name}-home"
    start = tmp_path / f"{shell_name}-start"
    temp = tmp_path / f"{shell_name}-tmp"
    for path in (home, start, temp):
        path.mkdir()
    (home / profile_name).write_text(
        'if [ -z "${OPENAI_API_KEY+x}" ]; then '
        "export TLDW_INITIAL_SECRET_ABSENT=1; fi\n"
        "export OPENAI_API_KEY=profile-restored\n",
        encoding="utf-8",
    )
    choices = discover_shell_choices(
        platform_name="posix",
        account_shell=executable,
        executable_lookup=shutil.which,
    )
    environment = build_terminal_environment(
        platform_name="posix",
        ambient={
            "PATH": os.environ.get("PATH", os.defpath),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "OPENAI_API_KEY": "ambient-secret",
        },
        account_reader=lambda: {
            "HOME": str(home),
            "USER": "fixture-user",
            "LOGNAME": "fixture-user",
            "SHELL": executable,
        },
        system_reader=lambda: {"TMPDIR": str(temp)},
    )
    shell = resolve_shell_choice("default", choices)
    completed = subprocess.run(
        shell.argv,
        cwd=resolve_start_directory(start, account_home=home),
        env=environment,
        input=(
            'printf "__TLDW_LAUNCH__%s|%s|%s\\n" "$PWD" '
            '"${TLDW_INITIAL_SECRET_ABSENT-0}" "${OPENAI_API_KEY-unset}"\n'
            "exit\n"
        ),
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0
    assert (
        f"__TLDW_LAUNCH__{start}|1|profile-restored"
        in completed.stdout + completed.stderr
    )
