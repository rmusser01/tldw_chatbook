"""Run the checkout's offline SourceReader commands in a private child."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The parent parses arguments and selects the private profile before the
# child imports application code. Only fixed offline commands reach here.
_OFFLINE_CHILD = """
import json
import sys
from pathlib import Path

from tldw_chatbook.Backup_Recovery.launcher import startup_unlock

admission = startup_unlock()
if admission is not None:
    raise SystemExit(admission)

from tldw_chatbook.Evals.source_reader.experiment import prepare, report

args = json.loads(sys.argv[1])
try:
    if args['command'] == 'prepare':
        prepare(Path(args['output']), include_followup=args['include_followup'])
    else:
        report(
            Path(args['prepared']), Path(args['output']),
            run_dir=Path(args['run_dir']) if args['run_dir'] else None,
            grades_path=Path(args['grades']) if args['grades'] else None,
        )
except Exception:
    print('Experiment refused: invalid fixture, output or private operation.', file=sys.stderr)
    raise SystemExit(2) from None
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Library fixture experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--include-followup", action="store_true")
    report = commands.add_parser("report")
    for name in ("prepared", "output", "run-dir", "grades"):
        report.add_argument(
            f"--{name}", type=Path, required=name in {"prepared", "output"}
        )

    run = commands.add_parser("run", help="Provider dispatch is unavailable")
    run.add_argument("--provider", choices=("openai", "deepseek"), default="openai")
    for name in ("prepared", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    for name in ("main-model", "worker-model", "base-url", "api-key-env"):
        run.add_argument(f"--{name}", required=True)
    for name in ("request-cap", "token-cap", "context-limit"):
        run.add_argument(f"--{name}", type=int, required=True)
    run.add_argument("--spend-cap-usd", type=float, required=True)
    run.add_argument("--input-limit", type=int, default=24000)
    run.add_argument("--output-limit", type=int, default=2000)
    run.add_argument(
        "--split", choices=("development", "heldout", "followup"), required=True
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--retrieval-model", type=Path)
    return parser


def _launch_offline(args: argparse.Namespace) -> int:
    root = Path(tempfile.mkdtemp(prefix="source-reader-")).resolve()
    home, config, data, state, temp = (
        root / name for name in ("home", "config", "data", "state", "tmp")
    )
    for path in (home, config, data, state, temp):
        path.mkdir(mode=0o700)
    selected = config / "config.toml"
    selected.write_text(
        f"[paths]\ndata_dir = {json.dumps(str(data))}\n", encoding="utf-8"
    )
    selected.chmod(0o600)
    env = {
        name: os.environ[name]
        for name in ("PATH", "LANG", "LC_ALL")
        if name in os.environ
    }
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(config),
        XDG_DATA_HOME=str(data),
        XDG_STATE_HOME=str(state),
        XDG_CACHE_HOME=str(root / "cache"),
        TLDW_CONFIG_PATH=str(selected),
        TMPDIR=str(temp),
        TEMP=str(temp),
        TMP=str(temp),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    args_json = json.dumps(
        {
            name: str(value.absolute()) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        }
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", _OFFLINE_CHILD, args_json],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    # The offline child has no RAG model or provider transport. Its normal
    # exit is enough to remove this CLI's scratch, without a native claim.
    if completed.returncode == 0:
        shutil.rmtree(root)
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    """Parse checkout arguments and run only offline fixture commands."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        parser.error("provider dispatch is unavailable for this checkout fixture")
    if args.output.exists() or args.output.is_symlink():
        parser.error("the output directory already exists")
    try:
        return _launch_offline(args)
    except KeyboardInterrupt:
        return 130
    except OSError:
        print("Unable to complete the private fixture command.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
