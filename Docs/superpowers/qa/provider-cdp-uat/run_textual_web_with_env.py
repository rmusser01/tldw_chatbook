"""Launch Textual-web with isolated QA config and redacted provider env.

This helper is intentionally small and process-oriented: it prepares a clean
HOME/XDG profile, copies only usable dotenv values into the launched process,
and then keeps ``tldw-serve`` running for CDP automation from another shell.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence


HELPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(HELPER_DIR) not in sys.path:
    sys.path.insert(0, str(HELPER_DIR))

from provider_inventory import KNOWN_PROVIDER_ENV_KEYS, load_env_values, mask_secret, should_use_key_value


DEFAULT_PORT = 8000
DEFAULT_ENV_FILE = HELPER_DIR / ".env"
DEFAULT_QA_ROOT = Path("/private/tmp/tldw-chatbook-provider-cdp-uat")
INHERITED_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TMPDIR",
        "PYTHONIOENCODING",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)
PROVIDER_KEY_ENV_VARS = frozenset(
    {
        "CUSTOM_OPENAI_API_KEY",
        "CUSTOM_OPENAI_API_KEY_2",
        "MISTRALAI_API_KEY",
        "QWEN_API_KEY",
        *(env_name for names in KNOWN_PROVIDER_ENV_KEYS.values() for env_name in names),
    }
)
ISOLATED_CONFIG = """[general]
default_tab = "chat"

[splash_screen]
enabled = false

[console]
collapse_large_pastes = true
paste_collapse_threshold = 50
"""


def build_launch_environment(
    *,
    worktree: Path,
    qa_root: Path,
    env_values: Mapping[str, str],
    port: int = DEFAULT_PORT,
    base_environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment used for the isolated Textual-web process.

    Args:
        worktree: Repository checkout to expose on ``PYTHONPATH``.
        qa_root: Root directory for isolated HOME/XDG state.
        env_values: Dotenv values loaded by ``provider_inventory.load_env_values``.
        port: Textual-web port to expose to the app and server process.
        base_environ: Optional inherited environment for tests.

    Returns:
        Environment mapping safe to pass to ``subprocess.run``.
    """

    inherited = os.environ if base_environ is None else base_environ
    env = {
        key: value
        for key, value in inherited.items()
        if key in INHERITED_ENV_ALLOWLIST and key not in PROVIDER_KEY_ENV_VARS
    }
    for key, value in env_values.items():
        if should_use_key_value(value):
            env[key] = value.strip()
        else:
            env.pop(key, None)

    env["PYTHONPATH"] = str(worktree)
    env["HOME"] = str(qa_root / "home")
    env["XDG_CONFIG_HOME"] = str(qa_root / "config")
    env["XDG_DATA_HOME"] = str(qa_root / "data")
    env["TLDW_CONFIG_PATH"] = str(config_paths(qa_root)[0])
    env["TLDW_TEXTUAL_WEB_PORT"] = str(port)
    return env


def config_paths(qa_root: Path) -> tuple[Path, Path]:
    """Return HOME-default and XDG config paths used by Chatbook."""

    home_config = qa_root / "home" / ".config" / "tldw_cli" / "config.toml"
    xdg_config = qa_root / "config" / "tldw_cli" / "config.toml"
    return home_config, xdg_config


def write_isolated_configs(qa_root: Path) -> tuple[Path, Path]:
    """Write the minimal isolated Chatbook config to both config locations."""

    paths = config_paths(qa_root)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ISOLATED_CONFIG, encoding="utf-8")
    (qa_root / "data").mkdir(parents=True, exist_ok=True)
    return paths


def load_optional_env_values(env_file: Path | None) -> dict[str, str]:
    """Load dotenv values when the file exists."""

    if env_file is None or not env_file.exists():
        return {}
    return load_env_values(env_file)


def masked_key_source_summary(env_values: Mapping[str, str]) -> list[str]:
    """Return a masked summary of usable dotenv keys without raw values."""

    summary: list[str] = []
    for key, value in sorted(env_values.items()):
        if not should_use_key_value(value):
            continue
        summary.append(f"env_file:{key}={mask_secret(value.strip())}")
    return summary


def launch_command(worktree: Path, port: int) -> list[str]:
    """Return the ``tldw-serve`` command for the given worktree and port."""

    return [
        str(worktree / ".venv" / "bin" / "tldw-serve"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def normalize_qa_root(qa_root: Path) -> Path:
    """Return an absolute QA root path without requiring it to exist."""

    return qa_root.expanduser().resolve(strict=False)


def validate_launch_inputs(worktree: Path, port: int) -> list[str]:
    """Return validation errors that must stop launch before side effects."""

    errors: list[str] = []
    if not 1 <= int(port) <= 65535:
        errors.append(f"port must be between 1 and 65535: {port}")

    if not worktree.exists():
        errors.append(f"worktree does not exist: {worktree}")
    elif not worktree.is_dir():
        errors.append(f"worktree is not a directory: {worktree}")
    else:
        serve_path = Path(launch_command(worktree, port)[0])
        if not serve_path.exists():
            errors.append(f"tldw-serve not found: {serve_path}")
    return errors


def print_launch_summary(
    *,
    worktree: Path,
    qa_root: Path,
    env_file: Path | None,
    config_files: Sequence[Path],
    command: Sequence[str],
    port: int,
    env_values: Mapping[str, str],
) -> None:
    """Print launch details without exposing raw dotenv values."""

    print(f"worktree: {worktree}", flush=True)
    print(f"qa_root: {qa_root}", flush=True)
    print(f"env_file: {env_file if env_file is not None else 'none'}", flush=True)
    for path in config_files:
        print(f"config: {path}", flush=True)
    print(f"port: {port}", flush=True)
    print(f"command: {shlex.join(command)}", flush=True)
    key_summary = masked_key_source_summary(env_values)
    print(f"key_sources: {', '.join(key_summary) if key_summary else 'none'}", flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worktree",
        type=Path,
        default=REPO_ROOT,
        help="Repository checkout containing .venv/bin/tldw-serve",
    )
    parser.add_argument(
        "--qa-root",
        type=Path,
        default=DEFAULT_QA_ROOT,
        help="Isolated HOME/XDG root for the Textual-web run",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Dotenv file containing provider keys; defaults to adjacent .env",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Textual-web port",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    worktree = args.worktree.expanduser().resolve(strict=False)
    qa_root = normalize_qa_root(args.qa_root)
    env_file = args.env_file.expanduser().resolve(strict=False) if args.env_file is not None else None
    validation_errors = validate_launch_inputs(worktree, args.port)
    if validation_errors:
        for error in validation_errors:
            print(f"error: {error}", file=sys.stderr, flush=True)
        return 2

    env_values = load_optional_env_values(env_file)
    config_files = write_isolated_configs(qa_root)
    launch_env = build_launch_environment(
        worktree=worktree,
        qa_root=qa_root,
        env_values=env_values,
        port=args.port,
    )
    command = launch_command(worktree, args.port)

    print_launch_summary(
        worktree=worktree,
        qa_root=qa_root,
        env_file=env_file,
        config_files=config_files,
        command=command,
        port=args.port,
        env_values=env_values,
    )
    return subprocess.run(command, cwd=worktree, env=launch_env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
