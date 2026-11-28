"""One fresh matched Console run; the public fixture marker enters only the owned app process."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

E = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
ROOTS = {}


def load(name, filename, expected):
    path = E / filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_root(role):
    root, head = ROOTS[role]
    for command, wanted in (
        (["rev-parse", "HEAD"], head),
        (["status", "--porcelain"], ""),
    ):
        assert (
            subprocess.check_output(
                ["git", "-C", str(root), *command], text=True
            ).strip()
            == wanted
        )
    return root, head


def app(profile, role, *, check_only=False):
    """Verify effective source/settings before loading a credential or the app."""
    root, head = checked_root(role)
    assert profile.is_relative_to(Path("/private/tmp/pr3023-live-current-r2"))
    assert not (profile / "app-start.json").exists(), "No reconnect/replay of a used profile"
    assert Path.cwd() == root
    import tldw_profile_core

    import tldw_chatbook
    from tldw_chatbook.config import get_chachanotes_db_path, get_cli_setting

    assert Path(tldw_chatbook.__file__).is_relative_to(root)
    assert Path(tldw_profile_core.__file__).is_relative_to(
        root / "packages/tldw_profile_core/src"
    )
    assert get_cli_setting("chat_defaults", "provider") == "deepseek"
    assert get_cli_setting("chat_defaults", "model") == "deepseek-flash"
    assert get_cli_setting("api_settings", "deepseek")["api_retries"] == 0
    assert get_cli_setting("console", "agent_runtime") is True
    assert get_cli_setting("console", "native_tool_calls") is True
    assert get_cli_setting("console", "local_tools_enabled") is False
    assert get_chachanotes_db_path().is_relative_to(profile)
    receipt = {
        "pid": os.getpid(),
        "head": head,
        "role": role,
        "package": str(tldw_chatbook.__file__),
        "profile_core": str(tldw_profile_core.__file__),
        "db_path": str(get_chachanotes_db_path()),
        "api_retries": 0,
        "model": "deepseek-flash",
        "native_tool_calls": True,
    }
    if check_only:
        print(json.dumps(receipt))
        return
    with (profile / "app-start.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    os.environ["DEEPSEEK_API_KEY"] = "ci-loopback-fixture"
    sys.argv = [
        "tldw_chatbook.app",
        "--config",
        str(profile / "config.toml"),
        "--no-splash",
    ]
    try:
        runpy.run_module("tldw_chatbook.app", run_name="__main__")
    finally:
        (profile / "app-returned.json").write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "scope": "Original app entry returned; physical process exit is checked separately",
                }
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=ROOTS)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--app", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="App source/settings preflight; no key or app launch",
    )
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.app:
        assert args.profile and not args.prepare and args.port is None
        app(args.profile.resolve(strict=True), args.role, check_only=args.check)
        return
    assert not args.check
    root, head = checked_root(args.role)
    profiles = Path("/private/tmp/pr3023-live-current-r2")
    profiles.mkdir(mode=0o700, exist_ok=True)
    if args.prepare:
        assert args.profile is None and args.port is None
        helper = load(
            "matched_profiles",
            "pr3023-matched-profile-preparation-r1.py",
            PROFILE_HELPER_SHA,
        )
        private, config = helper.prepare_profile(profiles, args.role)
        print(
            json.dumps(
                {
                    "role": args.role,
                    "head": head,
                    "profile": str(private),
                    "settings": helper.comparison_settings(config),
                }
            )
        )
        return
    assert args.profile and args.port and 1024 < args.port < 65536
    private = args.profile.resolve(strict=True)
    assert private.parent == profiles and private.name.startswith(
        "pr3023-" + args.role + "-"
    )
    assert not (private / "app-start.json").exists(), (
        "No reconnect/replay of a used profile"
    )
    native = load(
        "historical_native",
        "fixture_environment.py",
        "19da2f2d27b5ae6e9933a6bb587d8564af596e75ccca318e85ce63dced3a4975",
    )
    env = native.child_environment(root, private, private / "config.toml")
    observer = E / "pr3023-current-live-http-r2"
    env["PYTHONPATH"] = str(observer) + os.pathsep + env["PYTHONPATH"]
    env["PR3023_HTTP_LOG"] = str(private / "http.jsonl")
    command = shlex.join(
        [
            str(PYTHON),
            "-u",
            str(Path(__file__).resolve()),
            "--app",
            "--role",
            args.role,
            "--profile",
            str(private),
        ]
    )
    code = "import sys; from textual_serve.server import Server; Server(command=sys.argv[1], host='127.0.0.1', port=int(sys.argv[2]), title='PR3023 matched Console').serve()"
    os.chdir(root)
    os.execve(
        str(PYTHON), [str(PYTHON), "-u", "-c", code, command, str(args.port)], env
    )


PROFILE_HELPER_SHA = "860a4465bd70981bfb090e8a0b6b7bc0bcb0007e517b5673545c191d1a8c6fd0"

if __name__ == "__main__":
    main()
