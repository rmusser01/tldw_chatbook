"""Deterministic contracts for the one-shot workspace executor profiler."""

from __future__ import annotations

import json
import math
import ntpath
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from Tests.Performance import run_workspace_tool_executor_profile as profile


def test_nearest_rank_p95_uses_one_based_ceiling() -> None:
    assert profile.nearest_rank_p95(list(range(1, 31))) == 29
    assert profile.nearest_rank_p95([7.0]) == 7.0


def test_profile_uses_exact_operations_and_fake_clock_sample_injection(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, int]] = []
    ticks = iter(float(value) / 1000.0 for value in range(0, 240, 2))

    def fake_sample(
        _workspace: Path, operation: str, mode: str, sample_index: int
    ) -> None:
        calls.append((operation, mode, sample_index))

    report = profile.build_profile(
        tmp_path,
        samples=2,
        clock=lambda: next(ticks),
        sample_runner=fake_sample,
        metadata={
            "head_commit": "a" * 40,
            "platform": "test-platform",
            "python": "3.12.test",
        },
    )

    assert tuple(report["operations"]) == (
        "stat",
        "read",
        "write",
        "list",
        "git_status",
        "git_diff",
    )
    assert calls == [
        (operation, mode, sample_index)
        for operation in profile.OPERATIONS
        for sample_index in range(2)
        for mode in ("direct", "one_shot")
    ]
    for metrics in report["operations"].values():
        assert metrics == {
            "direct_ms": {"median": 2.0, "p95": 2.0},
            "one_shot_ms": {"median": 2.0, "p95": 2.0},
            "startup_overhead_ms": {"median": 0.0, "p95": 0.0},
        }


def test_profile_json_is_finite_content_free_metadata_without_timing_gate(
    tmp_path: Path,
) -> None:
    private_marker = "private-root-and-content-marker"

    def fake_sample(
        _workspace: Path, _operation: str, _mode: str, _sample_index: int
    ) -> None:
        return None

    ticks = iter(float(value) for value in range(1000))
    report = profile.build_profile(
        tmp_path / private_marker,
        samples=1,
        clock=lambda: next(ticks),
        sample_runner=fake_sample,
        metadata={
            "head_commit": "b" * 40,
            "platform": "bounded-platform",
            "python": "3.12.0",
        },
    )
    output = tmp_path / "profile.json"
    profile.write_profile(report, output)
    raw = output.read_text(encoding="utf-8")
    decoded = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))

    assert set(decoded) == {
        "schema_version",
        "head_commit",
        "platform",
        "python",
        "samples",
        "operations",
    }
    assert private_marker not in raw
    assert "path" not in decoded
    assert "content" not in decoded
    assert "threshold" not in decoded
    assert "passed" not in decoded
    assert "qualification" not in decoded
    for metrics in decoded["operations"].values():
        assert set(metrics) == {"direct_ms", "one_shot_ms", "startup_overhead_ms"}
        for summary in metrics.values():
            assert set(summary) == {"median", "p95"}
            assert all(math.isfinite(value) for value in summary.values())


def test_invalid_sample_count_is_refused() -> None:
    for value in (0, -1, True):
        try:
            profile.build_profile(
                Path("."),
                samples=value,
                sample_runner=lambda *_args: None,
            )
        except ValueError as error:
            assert str(error) == "samples must be a positive integer"
        else:
            raise AssertionError(f"accepted invalid sample count: {value!r}")


def test_cli_parser_refuses_nonpositive_samples_at_input_boundary(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        profile._parser(validate=True).parse_args(
            ["--samples", "0", "--output", str(tmp_path / "profile.json")]
        )


def test_cli_parser_refuses_oversized_samples_at_input_boundary(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        profile._parser(validate=True).parse_args(
            ["--samples", "9" * 4000, "--output", str(tmp_path / "profile.json")]
        )


def test_cli_parser_returns_normalized_output_path(tmp_path: Path) -> None:
    requested = tmp_path / "nested" / ".." / "profile.json"

    args = profile._parser(validate=True).parse_args(
        ["--samples", "1", "--output", str(requested)]
    )

    assert args.samples == 1
    assert args.output == requested.resolve()


def test_outer_cli_parse_does_not_import_application_modules() -> None:
    repository = Path(__file__).resolve().parents[2]
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from Tests.Performance import run_workspace_tool_executor_profile as p; "
                "before=set(sys.modules); "
                "p._parser(validate=False).parse_args(['--samples','1','--output','x']); "
                "print(sorted(name for name in set(sys.modules)-before "
                "if name.startswith('tldw_chatbook')))"
            ),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert probe.stdout.strip() == "[]"


def test_outer_cli_anchors_relative_output_to_caller_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)

    args = profile._parser(validate=False).parse_args(
        ["--samples", "1", "--output", "profile.json"]
    )

    assert args.output == caller / "profile.json"


def test_windows_isolated_environment_uses_only_temporary_home(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A Windows child must never fall back to the developer profile."""
    builder = getattr(profile, "_isolated_environment", None)
    assert builder is not None, "profile runner has no isolated environment builder"
    source_environment = {
        "PATH": "isolated-path",
        "LANG": "C.UTF-8",
        "SYSTEMROOT": r"C:\Windows",
        "WINDIR": r"C:\Windows",
        "TEMP": r"C:\Temp",
        "TMP": r"C:\Temp",
        "HOME": "/real/developer/home",
        "USERPROFILE": r"C:\Users\real-developer",
        "API_SECRET": "must-not-cross-profile-boundary",
    }

    environment = builder(
        tmp_path,
        platform_name="nt",
        source_environment=source_environment,
    )
    isolated_home = str(tmp_path / "home")

    assert environment["HOME"] == isolated_home
    assert environment["USERPROFILE"] == isolated_home
    assert environment["HOMEDRIVE"] + environment["HOMEPATH"] == isolated_home
    assert environment["TEMP"] == str(tmp_path / "temp")
    assert environment["TMP"] == str(tmp_path / "temp")
    assert set(environment) == {
        "PATH",
        "LANG",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "TLDW_CONFIG_PATH",
        profile._ISOLATED_RUNTIME_MARKER,
    }
    assert "real-developer" not in repr(environment)
    assert "must-not-cross-profile-boundary" not in repr(environment)
    assert (tmp_path / "home").is_dir()
    assert (tmp_path / "config").is_dir()
    assert (tmp_path / "data").is_dir()

    with monkeypatch.context() as clean_environment:
        for name in tuple(os.environ):
            clean_environment.delenv(name, raising=False)
        for name, value in environment.items():
            clean_environment.setenv(name, value)
        assert ntpath.expanduser("~") == isolated_home


def test_cli_suppresses_isolated_child_paths_and_secrets_on_failure(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Captured child diagnostics must not become public profile evidence."""
    private_marker = str(tmp_path / "private-runtime-profile")
    secret_marker = "synthetic-profile-secret-19637"

    monkeypatch.delenv(profile._ISOLATED_RUNTIME_MARKER, raising=False)
    monkeypatch.setattr(profile.shutil, "which", lambda _name: "git")
    monkeypatch.setattr(
        profile,
        "_isolated_runtime_python",
        lambda runtime_root: runtime_root / "python",
    )

    def failed_child(command, **kwargs):
        if not kwargs.get("capture_output"):
            print(f"child stdout leaked {private_marker}")
            print(f"child stderr leaked {secret_marker}", file=sys.stderr)
        return subprocess.CompletedProcess(
            command,
            23,
            stdout=f"child stdout leaked {private_marker}",
            stderr=f"child stderr leaked {secret_marker}",
        )

    monkeypatch.setattr(profile.subprocess, "run", failed_child)
    output = tmp_path / "profile.json"

    return_code = profile.main(["--samples", "1", "--output", str(output)])
    captured = capsys.readouterr()
    serialized = output.read_text(encoding="utf-8") if output.exists() else ""

    assert return_code == 23
    assert captured.out == ""
    assert captured.err == "isolated profile child failed\n"
    for public_evidence in (captured.out, captured.err, serialized):
        assert private_marker not in public_evidence
        assert secret_marker not in public_evidence


def test_child_temp_workspace_stays_inside_disposable_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Host TEMP/TMP must not outlive the isolated runtime boundary."""
    hostile_temp = tmp_path / "real-developer-profile" / "Temp"
    hostile_tmp = tmp_path / "host-secret-tmp"
    hostile_temp.mkdir(parents=True)
    hostile_tmp.mkdir()
    runtime_roots: list[Path] = []
    child_paths: list[Path] = []

    monkeypatch.delenv(profile._ISOLATED_RUNTIME_MARKER, raising=False)
    monkeypatch.setenv("TEMP", str(hostile_temp))
    monkeypatch.setenv("TMP", str(hostile_tmp))
    monkeypatch.setattr(profile.shutil, "which", lambda _name: "git")

    def fake_bootstrap(runtime_root: Path) -> Path:
        runtime_roots.append(runtime_root)
        return runtime_root / "python"

    def successful_child(command, **kwargs):
        environment = kwargs["env"]
        child_temp = Path(
            tempfile.mkdtemp(prefix="child-profile-", dir=environment["TEMP"])
        )
        child_workspace = child_temp / "workspace"
        child_workspace.mkdir()
        child_paths.extend((child_temp, child_workspace))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(profile, "_isolated_runtime_python", fake_bootstrap)
    monkeypatch.setattr(profile.subprocess, "run", successful_child)

    result = profile.main(
        ["--samples", "1", "--output", str(tmp_path / "profile.json")]
    )

    assert result == 0
    assert len(runtime_roots) == 1
    runtime_root = runtime_roots[0]
    assert child_paths
    assert all(runtime_root in path.parents for path in child_paths)
    assert not runtime_root.exists()
    assert all(not path.exists() for path in child_paths)
    assert list(hostile_temp.iterdir()) == []
    assert list(hostile_tmp.iterdir()) == []


def test_bootstrap_failure_emits_only_fixed_content_free_diagnostic(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """A bootstrap exception must not expose its command or captured output."""
    private_marker = str(tmp_path / "private-bootstrap-runtime")
    checkout_marker = "/private/checkout/task-19637"
    secret_marker = "synthetic-bootstrap-secret-19637"

    monkeypatch.delenv(profile._ISOLATED_RUNTIME_MARKER, raising=False)
    monkeypatch.setattr(profile.shutil, "which", lambda _name: "git")

    def failed_bootstrap(runtime_root: Path) -> Path:
        raise subprocess.CalledProcessError(
            17,
            [str(runtime_root / private_marker), checkout_marker],
            output=f"bootstrap stdout {private_marker}",
            stderr=f"bootstrap stderr {secret_marker}",
        )

    monkeypatch.setattr(profile, "_isolated_runtime_python", failed_bootstrap)
    output = tmp_path / "profile.json"
    escaped_error: BaseException | None = None
    return_code: int | None = None

    try:
        return_code = profile.main(["--samples", "1", "--output", str(output)])
    except BaseException as error:  # noqa: BLE001 - regression guards CLI boundary
        escaped_error = error
    captured = capsys.readouterr()
    serialized = output.read_text(encoding="utf-8") if output.exists() else ""

    assert escaped_error is None, (
        f"bootstrap exception escaped as {type(escaped_error).__name__}"
    )
    assert return_code == 1
    assert captured.out == ""
    assert captured.err == "isolated profile child failed\n"
    for public_evidence in (captured.out, captured.err, serialized):
        assert private_marker not in public_evidence
        assert checkout_marker not in public_evidence
        assert secret_marker not in public_evidence
