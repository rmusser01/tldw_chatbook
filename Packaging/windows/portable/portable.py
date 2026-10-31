"""Experimental, process-local portable profile launcher (stdlib only)."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


class PortableDataMovedError(ValueError):
    """A used profile needs supported recovery relocation, not re-enrollment."""


def check_directory(path: Path) -> None:
    """Refuse links/junctions in a portable state directory's ancestry."""
    for entry in reversed((path, *path.parents)):
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Portable state cannot use links or junctions")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("Portable state path is not a directory")


def portable_environment(root: Path, inherited: dict[str, str]) -> dict[str, str]:
    """Select OS essentials, then derive state paths from this bundle."""
    allowed = {
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "OS",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "NUMBER_OF_PROCESSORS",
        "USERNAME",
        "USERDOMAIN",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
    }
    env = {
        key.upper(): value for key, value in inherited.items() if key.upper() in allowed
    }
    data = root / "Data"
    portable_home = data / "home"
    directories = {
        "HOME": portable_home,
        "USERPROFILE": portable_home,
        "APPDATA": portable_home / "AppData" / "Roaming",
        "LOCALAPPDATA": portable_home / "AppData" / "Local",
        "XDG_CONFIG_HOME": portable_home / ".config",
        "XDG_DATA_HOME": portable_home / ".local" / "share",
        "XDG_CACHE_HOME": data / "cache",
        "TEMP": data / "tmp",
        "TMP": data / "tmp",
        "TMPDIR": data / "tmp",
        "HF_HOME": data / "cache" / "huggingface",
        "TORCH_HOME": data / "cache" / "torch",
        "NLTK_DATA": data / "cache" / "nltk",
        "CHATBOOK_WEZTERM_HOME": portable_home,
        "CHATBOOK_WEZTERM_DATA_DIR": data / "wezterm" / "data",
        "CHATBOOK_WEZTERM_RUNTIME_DIR": data / "wezterm" / "runtime",
    }
    env.update({key: str(value) for key, value in directories.items()})
    env["TLDW_CONFIG_PATH"] = str(
        portable_home / ".config" / "tldw_cli" / "config.toml"
    )
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["DO_NOT_TRACK"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    system_root = Path(env.get("SYSTEMROOT", r"C:\Windows"))
    env["PATH"] = os.pathsep.join(
        map(
            str,
            (
                root / "runtime",
                root / "runtime" / "Scripts",
                system_root / "System32",
                system_root,
            ),
        )
    )
    env["HOMEDRIVE"], env["HOMEPATH"] = os.path.splitdrive(str(portable_home))
    return env


def prepare_data(root: Path, env: dict[str, str]) -> None:
    """Create only bundle-owned directories, with no host-profile fallback."""
    check_directory(root / "Data")
    for key in (
        "HOME",
        "APPDATA",
        "LOCALAPPDATA",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "TEMP",
        "HF_HOME",
        "TORCH_HOME",
        "NLTK_DATA",
        "CHATBOOK_WEZTERM_DATA_DIR",
        "CHATBOOK_WEZTERM_RUNTIME_DIR",
    ):
        directory = Path(env[key])
        check_directory(directory)
        directory.mkdir(parents=True, exist_ok=True)
        check_directory(directory)


def check_location(root: Path) -> None:
    """Detect moved profiles before opening their recovery or database state."""
    marker = root / "Data" / "portable-location.json"
    if marker.is_symlink() or (
        marker.exists() and getattr(marker.lstat(), "st_file_attributes", 0) & 0x400
    ):
        raise ValueError("Portable location marker cannot be a link or junction")
    try:
        with marker.open("x", encoding="utf-8") as stream:
            json.dump({"version": 1, "root": str(root)}, stream)
    except FileExistsError:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved.get("version") != 1 or not isinstance(saved.get("root"), str):
            raise ValueError("Invalid portable location marker")
        if os.path.normcase(saved["root"]) != os.path.normcase(str(root)):
            raise PortableDataMovedError(
                "This used portable profile has moved. Chatbook's recovery "
                "authority is tied to its original location and file identities. "
                "A destination-side relocation review is required. The copied "
                "profile has not been opened or reset."
            )


def app_arguments(root: Path, app_args: list[str]) -> list[str]:
    """Route moved profiles to review without changing the retained marker."""
    try:
        check_location(root)
    except PortableDataMovedError:
        return ["--portable-relocate", *app_args]
    return app_args


def launch_command(root: Path, app_args: list[str]) -> list[str]:
    """Use exact bundled executables and a dedicated terminal instance."""
    return [
        str(root / "terminal" / "wezterm-gui.exe"),
        "--config-file",
        str(root / "wezterm.lua"),
        "start",
        "--always-new-process",
        "--cwd",
        str(root / "Data"),
        "--",
        str(root / "runtime" / "python.exe"),
        "-B",
        str(root / "app_entry.py"),
        *app_args,
    ]


def main() -> int:
    """Launch the spike or run a console-only installed-runtime diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true")
    args, app_args = parser.parse_known_args()
    if any(
        value.split("=", 1)[0]
        in {
            "--config",
            "--recovery-profile",
            "--recovery-control-root",
            "--recovery-launch-attempt",
            "recovery",
        }
        for value in app_args
    ):
        parser.error("Profile overrides are unavailable in the portable spike")
    root = Path(__file__).absolute().parent
    env = portable_environment(root, dict(os.environ))
    prepare_data(root, env)
    app_args = app_arguments(root, app_args)
    if args.diagnose:
        command = [
            str(root / "runtime" / "python.exe"),
            "-B",
            str(root / "app_entry.py"),
            "--portable-probe",
            *app_args,
        ]
        return subprocess.run(
            command, cwd=root / "Data", env=env, check=False
        ).returncode
    startup = None
    if os.name == "nt":
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 1  # The hidden pythonw launcher must show its GUI child.
    with open(os.devnull, "w") as sink:
        process = subprocess.Popen(
            launch_command(root, app_args),
            cwd=root / "Data",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=sink,
            startupinfo=startup,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return process.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        message = (
            str(exc)
            if isinstance(exc, PortableDataMovedError)
            else (
                "Chatbook Portable could not start. Extract the complete ZIP to a "
                "writable local NTFS folder. Its Data folder must not be a link or "
                f"junction. No installed profile was selected. Error: {type(exc).__name__}"
            )
        )
        if sys.stderr is not None:
            print(message, file=sys.stderr)
        else:
            ctypes.windll.user32.MessageBoxW(None, message, "Chatbook Portable", 0x10)
        raise SystemExit(1)
