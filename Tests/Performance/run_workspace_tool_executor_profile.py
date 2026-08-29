"""Profile direct workspace operations against the one-shot pinned executor."""

from __future__ import annotations

import argparse
import json
import math
import ntpath
import os
import platform as platform_module
import shutil
import site
import statistics
import subprocess
import sys
import tempfile
import time
import venv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

OPERATIONS = ("stat", "read", "write", "list", "git_status", "git_diff")
_ISOLATED_RUNTIME_MARKER = "TLDW_TASK19637_PROFILE_RUNTIME"
_CHILD_FAILURE_DIAGNOSTIC = "isolated profile child failed"
_MODE = Literal["direct", "one_shot"]
_Clock = Callable[[], float]
_SampleRunner = Callable[[Path, str, _MODE, int], None]


def nearest_rank_p95(values: Sequence[float]) -> float:
    """Return the one-based nearest-rank 95th percentile."""
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(float(value) for value in values)
    rank = math.ceil(0.95 * len(ordered))
    return ordered[rank - 1]


def _summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize finite milliseconds without imposing a performance gate."""
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("profile timings must be finite")
    return {
        "median": round(float(statistics.median(values)), 6),
        "p95": round(float(nearest_rank_p95(values)), 6),
    }


def _default_metadata() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return {
        "head_commit": completed.stdout.strip(),
        "platform": platform_module.platform(),
        "python": platform_module.python_version(),
    }


def build_profile(
    workspace: Path,
    *,
    samples: int,
    clock: _Clock = time.perf_counter,
    sample_runner: _SampleRunner | None = None,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure paired direct/one-shot calls and return content-free summaries."""
    if not isinstance(samples, int) or isinstance(samples, bool) or samples <= 0:
        raise ValueError("samples must be a positive integer")
    runner = sample_runner or _run_sample
    identity = dict(metadata or _default_metadata())
    if set(identity) != {"head_commit", "platform", "python"}:
        raise ValueError("profile metadata has unexpected keys")

    operations: dict[str, dict[str, dict[str, float]]] = {}
    for operation in OPERATIONS:
        direct_values: list[float] = []
        one_shot_values: list[float] = []
        overhead_values: list[float] = []
        for sample_index in range(samples):
            direct_started = clock()
            runner(workspace, operation, "direct", sample_index)
            direct_ms = (clock() - direct_started) * 1000.0

            one_shot_started = clock()
            runner(workspace, operation, "one_shot", sample_index)
            one_shot_ms = (clock() - one_shot_started) * 1000.0

            if not math.isfinite(direct_ms) or not math.isfinite(one_shot_ms):
                raise ValueError("profile timings must be finite")
            direct_values.append(direct_ms)
            one_shot_values.append(one_shot_ms)
            overhead_values.append(one_shot_ms - direct_ms)
        operations[operation] = {
            "direct_ms": _summary(direct_values),
            "one_shot_ms": _summary(one_shot_values),
            "startup_overhead_ms": _summary(overhead_values),
        }

    return {
        "schema_version": 1,
        "head_commit": identity["head_commit"],
        "platform": identity["platform"],
        "python": identity["python"],
        "samples": samples,
        "operations": operations,
    }


def write_profile(report: Mapping[str, Any], output: Path) -> None:
    """Write strict finite JSON to the requested evidence path."""
    encoded = json.dumps(report, indent=2, sort_keys=False, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")


def _run_git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        timeout=30,
    )


def _prepare_workspace(workspace: Path) -> None:
    workspace.mkdir()
    (workspace / "small.txt").write_text("small profile payload\n", encoding="utf-8")
    (workspace / "profile-write.txt").write_text("initial\n", encoding="utf-8")
    (workspace / "tracked.txt").write_text("base\n", encoding="utf-8")
    (workspace / "listing-entry.txt").write_text("listed\n", encoding="utf-8")
    _run_git(workspace, "init")
    _run_git(workspace, "config", "user.email", "profile@example.invalid")
    _run_git(workspace, "config", "user.name", "Profile Runner")
    _run_git(workspace, "config", "commit.gpgsign", "false")
    _run_git(workspace, "add", ".")
    _run_git(workspace, "commit", "-m", "profile baseline")
    (workspace / "tracked.txt").write_text("base\nprofile diff\n", encoding="utf-8")


def _run_sample(
    workspace: Path,
    operation: str,
    mode: _MODE,
    sample_index: int,
) -> None:
    if mode == "direct":
        _run_direct(workspace, operation, sample_index)
        return
    _run_one_shot(workspace, operation, sample_index)


def _run_direct(workspace: Path, operation: str, sample_index: int) -> None:
    from tldw_chatbook.Tools.git_tool_impls import git_diff, git_status
    from tldw_chatbook.Tools.local_tool_impls import (
        list_directory,
        read_file,
        stat_path,
        write_file,
    )

    if operation == "stat":
        stat_path("small.txt", workspace_root=workspace)
    elif operation == "read":
        read_file("small.txt", workspace_root=workspace)
    elif operation == "write":
        write_file(
            "profile-write.txt",
            f"direct sample {sample_index}\n",
            workspace_root=workspace,
        )
    elif operation == "list":
        list_directory(".", workspace_root=workspace)
    elif operation == "git_status":
        git_status(workspace)
    elif operation == "git_diff":
        git_diff(workspace)
    else:
        raise ValueError("unknown profile operation")


def _run_one_shot(workspace: Path, operation: str, sample_index: int) -> None:
    from tldw_chatbook.Tools.workspace_tool_executor import WorkspaceToolExecutor

    executor = WorkspaceToolExecutor(workspace)
    if operation == "stat":
        executor.execute("stat_path", {"path": "small.txt"}, intent="read")
    elif operation == "read":
        executor.execute("fs_read", {"path": "small.txt"}, intent="read")
    elif operation == "write":
        executor.execute(
            "fs_write",
            {
                "path": "profile-write.txt",
                "content": f"one-shot sample {sample_index}\n",
            },
            intent="write",
        )
    elif operation == "list":
        executor.execute("fs_list", {"path": "."}, intent="read")
    elif operation == "git_status":
        executor.execute("git_status", {}, intent="read")
    elif operation == "git_diff":
        executor.execute("git_diff", {}, intent="read")
    else:
        raise ValueError("unknown profile operation")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _isolated_runtime_python(runtime_root: Path) -> Path:
    """Create a temporary interpreter whose ``-I`` imports this checkout."""
    environment_root = runtime_root / "venv"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment_root)
    scripts = environment_root / ("Scripts" if os.name == "nt" else "bin")
    runtime_python = scripts / ("python.exe" if os.name == "nt" else "python")
    site_query = subprocess.run(
        [
            str(runtime_python),
            "-I",
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    isolated_site_packages = Path(site_query.stdout.strip())
    repository = Path(__file__).resolve().parents[2]
    dependency_paths = [
        Path(value).resolve() for value in site.getsitepackages() if Path(value).is_dir()
    ]
    (isolated_site_packages / "task19637-profile.pth").write_text(
        "\n".join(str(path) for path in (repository, *dependency_paths)) + "\n",
        encoding="utf-8",
    )
    return runtime_python


def _isolated_environment(
    runtime_root: Path,
    *,
    platform_name: str | None = None,
    source_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the minimal child environment around one temporary profile."""
    source = os.environ if source_environment is None else source_environment
    platform_name = os.name if platform_name is None else platform_name
    home = runtime_root / "home"
    config = runtime_root / "config"
    data = runtime_root / "data"
    home.mkdir()
    config.mkdir()
    data.mkdir()
    home_text = str(home)
    environment = {
        "PATH": source.get("PATH", os.defpath),
        "HOME": home_text,
        "XDG_CONFIG_HOME": str(config),
        "XDG_DATA_HOME": str(data),
        "TLDW_CONFIG_PATH": str(config / "config.toml"),
        _ISOLATED_RUNTIME_MARKER: "1",
    }
    if platform_name == "nt":
        home_drive, home_path = ntpath.splitdrive(home_text)
        environment.update(
            {
                "USERPROFILE": home_text,
                "HOMEDRIVE": home_drive,
                "HOMEPATH": home_path,
            }
        )
    for name in ("LANG", "LC_ALL", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"):
        if value := source.get(name):
            environment[name] = value
    return environment


def _run_isolated(args: argparse.Namespace) -> int:
    """Run the profiler under an isolated profile before application import."""
    with tempfile.TemporaryDirectory(prefix="tldw-workspace-profile-runtime-") as raw:
        runtime_root = Path(raw)
        runtime_python = _isolated_runtime_python(runtime_root)
        environment = _isolated_environment(runtime_root)
        try:
            completed = subprocess.run(
                [
                    str(runtime_python),
                    "-I",
                    str(Path(__file__).resolve()),
                    "--samples",
                    str(args.samples),
                    "--output",
                    str(args.output.resolve()),
                ],
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                timeout=1800,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.SubprocessError):
            print(_CHILD_FAILURE_DIAGNOSTIC, file=sys.stderr)
            return 1
        if completed.returncode != 0:
            print(_CHILD_FAILURE_DIAGNOSTIC, file=sys.stderr)
        return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.samples <= 0:
        _parser().error("--samples must be a positive integer")
    if shutil.which("git") is None:
        raise RuntimeError("git is required for this profile")
    if os.environ.get(_ISOLATED_RUNTIME_MARKER) != "1":
        return _run_isolated(args)
    with tempfile.TemporaryDirectory(prefix="tldw-workspace-tool-profile-") as raw:
        workspace = Path(raw) / "workspace"
        _prepare_workspace(workspace)
        report = build_profile(workspace, samples=args.samples)
    write_profile(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
