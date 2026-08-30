from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RATCHET = REPO_ROOT / "scripts" / "terminal_qualification" / "format_ratchet.py"


def _load_ratchet_module():
    spec = importlib.util.spec_from_file_location("task22512_format_ratchet", RATCHET)
    assert spec is not None and spec.loader is not None
    qualification_root = str(RATCHET.parent)
    sys.path.insert(0, qualification_root)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(qualification_root)


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RATCHET), *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit_base(repo: Path, source: str) -> str:
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "qualification@example.invalid")
    _git(repo, "config", "user.name", "Qualification Test")
    (repo / "pyproject.toml").write_text(
        '[tool.ruff]\nline-length = 88\ntarget-version = "py311"\n',
        encoding="utf-8",
    )
    (repo / "sample.py").write_text(source, encoding="utf-8")
    _git(repo, "add", "pyproject.toml", "sample.py")
    _git(repo, "commit", "-qm", "base")
    return _git(repo, "rev-parse", "HEAD")


def _snapshot(repo: Path, baseline: Path) -> subprocess.CompletedProcess[str]:
    return _run(
        "snapshot",
        "--base",
        "HEAD",
        "--output",
        str(baseline),
        "--path",
        "sample.py",
        cwd=repo,
    )


def _base_with_inherited_debt() -> str:
    lines = [
        'DEBT = {"alpha": 1, "bravo": 2, "charlie": 3, "delta": 4, "echo": 5, "foxtrot": 6, "golf": 7, "hotel": 8}',
    ]
    lines.extend(f"marker_{number:02d} = {number}" for number in range(1, 25))
    return "\n".join(lines) + "\n"


def test_inherited_formatter_debt_outside_changed_lines_passes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    base_sha = _commit_base(repo, _base_with_inherited_debt())
    baseline = repo / "format-baseline.json"

    snapshot = _snapshot(repo, baseline)

    assert snapshot.returncode == 0, snapshot.stderr
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    assert payload["base_sha"] == base_sha
    assert payload["paths"] == ["sample.py"]
    assert payload["baseline_red_paths"] == ["sample.py"]

    source = (repo / "sample.py").read_text(encoding="utf-8")
    (repo / "sample.py").write_text(
        source.replace("marker_24 = 24", "marker_24 = 240"),
        encoding="utf-8",
    )

    verify = _run("verify", "--baseline", str(baseline), cwd=repo)

    assert verify.returncode == 0, verify.stderr


def test_new_formatter_hunk_on_changed_line_fails(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = (
        "\n".join(f"marker_{number:02d} = {number}" for number in range(1, 25)) + "\n"
    )
    _commit_base(repo, source)
    baseline = repo / "format-baseline.json"
    snapshot = _snapshot(repo, baseline)
    assert snapshot.returncode == 0, snapshot.stderr

    changed = (
        (repo / "sample.py")
        .read_text(encoding="utf-8")
        .replace(
            "marker_24 = 24",
            'marker_24 = {"alpha": 1, "bravo": 2, "charlie": 3, "delta": 4, "echo": 5, "foxtrot": 6, "golf": 7, "hotel": 8}',
        )
    )
    (repo / "sample.py").write_text(changed, encoding="utf-8")

    verify = _run("verify", "--baseline", str(baseline), cwd=repo)

    assert verify.returncode != 0
    assert "formatter-required hunk overlaps changed lines" in verify.stderr


def test_snapshot_refuses_to_replace_existing_baseline_without_flag(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, "answer = 42\n")
    baseline = repo / "format-baseline.json"
    baseline.write_text('{"sentinel": true}\n', encoding="utf-8")

    result = _snapshot(repo, baseline)

    assert result.returncode != 0
    assert json.loads(baseline.read_text(encoding="utf-8")) == {"sentinel": True}


def test_snapshot_rejects_output_outside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, "answer = 42\n")
    outside = tmp_path / "outside-format-baseline.json"

    result = _snapshot(repo, outside)

    assert result.returncode != 0
    assert "repository" in result.stderr
    assert outside.exists() is False


def test_snapshot_rejects_option_like_git_revision(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, "answer = 42\n")
    baseline = repo / "format-baseline.json"

    result = _run(
        "snapshot",
        "--base=-dangerous-option",
        "--output",
        str(baseline),
        "--path",
        "sample.py",
        cwd=repo,
    )

    assert result.returncode != 0
    assert "base revision is invalid" in result.stderr
    assert baseline.exists() is False


def test_verify_rejects_baseline_outside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, "answer = 42\n")
    baseline = repo / "format-baseline.json"
    snapshot = _snapshot(repo, baseline)
    assert snapshot.returncode == 0, snapshot.stderr
    outside = tmp_path / "outside-format-baseline.json"
    outside.write_bytes(baseline.read_bytes())

    result = _run("verify", "--baseline", str(outside), cwd=repo)

    assert result.returncode != 0
    assert "repository" in result.stderr


def test_verify_accepts_explicit_head_ref_from_plan_command(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, "answer = 42\n")
    baseline = repo / "format-baseline.json"
    snapshot = _snapshot(repo, baseline)
    assert snapshot.returncode == 0, snapshot.stderr

    result = _run(
        "verify",
        "--head",
        "HEAD",
        "--baseline",
        str(baseline),
        cwd=repo,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_sha256", "0" * 64),
        ("normalized_diff_sha256", "1" * 64),
        ("formatter_required", False),
        ("debt_units", 0),
        ("source_hunks", []),
    ),
)
def test_verify_rejects_tampered_immutable_base_facts(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    repo = tmp_path / "repo"
    source = _base_with_inherited_debt()
    _commit_base(repo, source)
    baseline = repo / "format-baseline.json"
    snapshot = _snapshot(repo, baseline)
    assert snapshot.returncode == 0, snapshot.stderr
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["files"]["sample.py"][field] = replacement
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    verify = _run("verify", "--baseline", str(baseline), cwd=repo)

    assert verify.returncode != 0
    assert "immutable base" in verify.stderr


def test_verify_rejects_recorded_ruff_version_drift(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, "value = 1\n")
    baseline = repo / "format-baseline.json"
    snapshot = _snapshot(repo, baseline)
    assert snapshot.returncode == 0, snapshot.stderr
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["ruff_version"] = "ruff 0.0.0"
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    verify = _run("verify", "--baseline", str(baseline), cwd=repo)

    assert verify.returncode != 0
    assert "Ruff version drift" in verify.stderr


def test_verify_rejects_tampered_baseline_red_paths(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, _base_with_inherited_debt())
    baseline = repo / "format-baseline.json"
    snapshot = _snapshot(repo, baseline)
    assert snapshot.returncode == 0, snapshot.stderr
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["baseline_red_paths"] = []
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    verify = _run("verify", "--baseline", str(baseline), cwd=repo)

    assert verify.returncode != 0
    assert "immutable base" in verify.stderr


def test_verify_rejects_boolean_schema_version_at_typed_boundary(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    _commit_base(repo, "value = 1\n")
    baseline = repo / "format-baseline.json"
    snapshot = _snapshot(repo, baseline)
    assert snapshot.returncode == 0, snapshot.stderr
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["schema_version"] = True
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    verify = _run("verify", "--baseline", str(baseline), cwd=repo)

    assert verify.returncode != 0
    assert "baseline validation failed" in verify.stderr


def test_public_ratchet_api_has_google_style_contracts() -> None:
    module = _load_ratchet_module()

    for name in ("snapshot", "verify"):
        docstring = getattr(module, name).__doc__ or ""
        assert "Args:" in docstring, name
        assert "Returns:" in docstring, name
        assert "Raises:" in docstring, name


def test_formatter_runner_rejects_bounded_output_overflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_ratchet_module()
    monkeypatch.setattr(
        module,
        "run_bounded",
        lambda *args, **kwargs: type(
            "OverflowResult",
            (),
            {"timed_out": False, "overflowed": True},
        )(),
    )

    with pytest.raises(module.RatchetError, match="output limit"):
        module._run(
            ("formatter", "--check"),
            cwd=tmp_path,
            operation="formatter-overflow",
            timeout_seconds=1.0,
        )
