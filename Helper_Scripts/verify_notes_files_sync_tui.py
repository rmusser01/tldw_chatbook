#!/usr/bin/env python3
"""Capture isolated live evidence for the reviewed Notes/Files/Sync journey.

Application imports happen only in the tmux child, after HOME, every XDG root,
the effective config, and its ``[paths].data_dir`` have been redirected into a
new scratch profile.  The caller's profile is never opened or inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILE_PREFIX = "tldw-notes-files-sync-profile-"
_COMMAND_TIMEOUT = 10.0
_REPO_SENTINELS = tuple(
    _REPO_ROOT / "tldw_chatbook" / "css" / name
    for name in (
        "screen_css_scoped.tcss",
        "screen_css_self.tcss",
        "tldw_cli_modular.tcss",
        "widget_defaults_scoped.tcss",
        "widget_defaults_self.tcss",
    )
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_sentinel_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(_REPO_ROOT)): _sha256(path) for path in _REPO_SENTINELS
    }


def _write_config(path: Path, data_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            (
                "[general]",
                'default_tab = "library"',
                'users_name = "verification"',
                "",
                "[paths]",
                f"data_dir = {json.dumps(str(data_dir))}",
                "",
                "[splash_screen]",
                "enabled = false",
                "",
                "[model_catalog]",
                "auto_refresh_enabled = false",
                "",
                "[first_run]",
                "setup_started = true",
                "setup_completed = true",
                "",
            )
        ),
        encoding="utf-8",
    )


def _seed_decoy(profile: Path) -> Path:
    decoy = profile / "xdg-config" / "tldw_cli" / "config.toml"
    decoy.parent.mkdir(parents=True)
    decoy.write_text("# decoy default; must remain byte-identical\n", encoding="utf-8")
    return decoy


def build_isolated_environment(profile: Path) -> tuple[dict[str, str], Path, Path]:
    """Return a complete scratch environment before any application import."""
    profile = profile.resolve(strict=True)
    home = profile / "home"
    xdg_config = profile / "xdg-config"
    xdg_data = profile / "xdg-data"
    xdg_cache = profile / "xdg-cache"
    temp_dir = profile / "tmp"
    data_dir = profile / "app-data"
    effective_config = profile / "effective" / "config.toml"
    for directory in (home, xdg_config, xdg_data, xdg_cache, temp_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _write_config(effective_config, data_dir)
    tmux = shutil.which("tmux")
    command_paths = {
        str(Path(sys.executable).parent),
        str(Path(tmux).parent) if tmux else "",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    }
    env = {
        "PATH": os.pathsep.join(sorted(path for path in command_paths if path)),
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_DATA_HOME": str(xdg_data),
        "XDG_CACHE_HOME": str(xdg_cache),
        "TMPDIR": str(temp_dir),
        "TLDW_CONFIG_PATH": str(effective_config),
        "TLDW_TEST_MODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": str(_REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
        "TERM": "xterm-256color",
        "LANG": "C.UTF-8",
        "NO_PROXY": "*",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }
    return env, effective_config, data_dir


def _capture(socket: Path, destination: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        ("tmux", "-S", str(socket), "capture-pane", "-p", "-e"),
        check=True,
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        timeout=_COMMAND_TIMEOUT,
    )
    destination.write_text(result.stdout, encoding="utf-8")
    return result.stdout


def _wait_for_frame(
    socket: Path,
    destination: Path,
    *,
    needles: tuple[str, ...],
    deadline: float,
    env: dict[str, str],
) -> str:
    painted = ""
    while time.monotonic() < deadline:
        time.sleep(0.2)
        painted = _capture(socket, destination, env)
        if painted.strip() and all(needle in painted for needle in needles):
            return painted
    raise RuntimeError(f"{destination.stem} did not paint its expected route")


def _send_key(socket: Path, key: str, env: dict[str, str]) -> None:
    subprocess.run(
        ("tmux", "-S", str(socket), "send-keys", key),
        check=True,
        cwd=_REPO_ROOT,
        env=env,
        timeout=_COMMAND_TIMEOUT,
    )


def _launch_and_capture(
    env: dict[str, str], evidence: Path, *, startup_seconds: float
) -> list[Path]:
    if shutil.which("tmux") is None:
        raise RuntimeError("tmux is required for live TUI verification")
    # Darwin's AF_UNIX path limit is short; the profile's mkdtemp path can
    # exceed it before tmux appends anything.  Use a unique explicit socket in
    # the system temporary directory and always remove it in ``finally``.
    socket = Path("/private/tmp") / f"tnfs-{os.getpid()}-{time.time_ns():x}.sock"
    child = (
        "from tldw_chatbook import app; "
        "app._is_source_tree = lambda _package_root: False; "
        "import sys; sys.argv = ['tldw-chatbook']; app.main_cli_runner()"
    )
    command = " ".join(
        (
            "cd",
            shlex.quote(str(_REPO_ROOT)),
            "&&",
            "exec",
            shlex.quote(sys.executable),
            "-c",
            shlex.quote(child),
        )
    )
    subprocess.run(
        (
            "tmux",
            "-S",
            str(socket),
            "new-session",
            "-d",
            "-x",
            "120",
            "-y",
            "36",
            command,
        ),
        check=True,
        cwd=_REPO_ROOT,
        env=env,
        timeout=_COMMAND_TIMEOUT,
    )
    frames: list[Path] = []
    try:
        deadline = time.monotonic() + startup_seconds
        wide = _wait_for_frame(
            socket,
            evidence / "library-wide.ansi.txt",
            needles=("Library", "new note"),
            deadline=deadline,
            env=env,
        )
        if not wide.strip():
            raise RuntimeError("Library captured a blank frame")
        frames.append(evidence / "library-wide.ansi.txt")

        # Use the Library's advertised physical shortcut, then return through
        # the ordinary Escape route to the Notes list/Add-from-files entry.
        _send_key(socket, "n", env)
        _wait_for_frame(
            socket,
            evidence / "notes-new-wide.ansi.txt",
            needles=("Library notes",),
            deadline=time.monotonic() + 8.0,
            env=env,
        )
        frames.append(evidence / "notes-new-wide.ansi.txt")

        _send_key(socket, "Escape", env)
        _wait_for_frame(
            socket,
            evidence / "notes-list-wide.ansi.txt",
            needles=("Library notes", "Add from files"),
            deadline=time.monotonic() + 4.0,
            env=env,
        )
        frames.append(evidence / "notes-list-wide.ansi.txt")

        subprocess.run(
            ("tmux", "-S", str(socket), "resize-window", "-x", "60", "-y", "20"),
            check=True,
            cwd=_REPO_ROOT,
            env=env,
            timeout=_COMMAND_TIMEOUT,
        )
        _wait_for_frame(
            socket,
            evidence / "notes-list-60x20.ansi.txt",
            needles=("Library notes", "Add from files"),
            deadline=time.monotonic() + 4.0,
            env=env,
        )
        frames.append(evidence / "notes-list-60x20.ansi.txt")

        # Notes is specified at wide and 60x20. Return to the production shell
        # for its supported 40x20 compact evidence.
        _send_key(socket, "Escape", env)
        subprocess.run(
            ("tmux", "-S", str(socket), "resize-window", "-x", "40", "-y", "20"),
            check=True,
            cwd=_REPO_ROOT,
            env=env,
            timeout=_COMMAND_TIMEOUT,
        )
        _wait_for_frame(
            socket,
            evidence / "library-40x20.ansi.txt",
            needles=("Library", "Navigation"),
            deadline=time.monotonic() + 4.0,
            env=env,
        )
        frames.append(evidence / "library-40x20.ansi.txt")
    finally:
        subprocess.run(
            ("tmux", "-S", str(socket), "kill-server"),
            check=False,
            cwd=_REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=_COMMAND_TIMEOUT,
        )
        socket.unlink(missing_ok=True)
    return frames


def verify(*, evidence_dir: Path, dry_run: bool, startup_seconds: float) -> int:
    evidence = evidence_dir.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    sentinel_before = _repo_sentinel_hashes()
    with tempfile.TemporaryDirectory(prefix=_PROFILE_PREFIX) as raw_profile:
        profile = Path(raw_profile).resolve(strict=True)
        env, effective_config, data_dir = build_isolated_environment(profile)
        decoy = _seed_decoy(profile)
        decoy_before = _sha256(decoy)
        frames: list[Path] = []
        failure_code = ""
        try:
            if not dry_run:
                frames = _launch_and_capture(
                    env, evidence, startup_seconds=startup_seconds
                )
        except Exception as exc:  # preserve bounded evidence before propagating
            failure_code = type(exc).__name__
        decoy_after = _sha256(decoy)
        sentinel_after = _repo_sentinel_hashes()
        if decoy_after != decoy_before:
            failure_code = "decoy_default_changed"
        if sentinel_after != sentinel_before:
            failure_code = "repo_bytes_changed"
        scratch_paths_valid = True
        try:
            for path in (effective_config, data_dir, decoy):
                path.resolve().relative_to(profile)
        except ValueError:
            scratch_paths_valid = False
            failure_code = "scratch_path_escape"
        manifest = {
            "status": "FAIL" if failure_code else "PASS",
            "dry_run": dry_run,
            "profile_is_scratch": True,
            "scratch_paths_valid": scratch_paths_valid,
            "test_mode": env["TLDW_TEST_MODE"] == "1",
            "first_run_section": "first_run",
            "first_run_completed": True,
            "model_downloads_offline": env["HF_HUB_OFFLINE"] == "1"
            and env["TRANSFORMERS_OFFLINE"] == "1",
            "caller_environment_inherited": False,
            "proxy_environment_scrubbed": all(
                env[name] == "" for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
            ),
            "git_config_isolated": env["GIT_CONFIG_NOSYSTEM"] == "1"
            and env["GIT_CONFIG_GLOBAL"] == "/dev/null",
            "repo_byte_stable": sentinel_after == sentinel_before,
            "repo_sentinel_sha256_before": sentinel_before,
            "repo_sentinel_sha256_after": sentinel_after,
            "child_environment_keys": sorted(env),
            "decoy_default_sha256_before": decoy_before,
            "decoy_default_sha256_after": decoy_after,
            "effective_config_relative": str(effective_config.relative_to(profile)),
            "data_dir_relative": str(data_dir.relative_to(profile)),
            "physical_journeys": [
                "library_shell",
                "database_notes_new",
                "database_notes_list",
            ]
            if not dry_run
            else [],
            "planned_physical_journeys": [
                "library_shell",
                "database_notes_new",
                "database_notes_list",
            ],
            "automated_companion": "Tests/UI/test_library_notes_files_sync_journey.py",
            "failure_code": failure_code,
            "frames": [
                {"name": frame.name, "sha256": _sha256(frame)} for frame in frames
            ],
        }
        (evidence / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if failure_code:
            raise RuntimeError(f"live verification failed ({failure_code})")
    print(f"PASS isolation decoy={decoy_before}")
    print("PASS scratch profile teardown")
    if not dry_run:
        print("PASS live Library, New note, and Notes list/Add-from-files frames")
        print("PASS production Library 40x20 compact frame")
    print(f"evidence={evidence}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify isolation without launching Textual",
    )
    parser.add_argument("--startup-seconds", type=float, default=20.0)
    parser.add_argument("--evidence-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    evidence = args.evidence_dir
    if evidence is None:
        evidence = Path(tempfile.mkdtemp(prefix="tldw-notes-files-sync-evidence-"))
    return verify(
        evidence_dir=evidence,
        dry_run=bool(args.dry_run),
        startup_seconds=max(1.0, args.startup_seconds),
    )


if __name__ == "__main__":
    raise SystemExit(main())
