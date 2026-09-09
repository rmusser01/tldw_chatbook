"""Behavioral checks for the required Mermaid derived-artifact guard."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_canvas_mermaid_assets.py"
EXPECTED_OUTPUTS = {
    "canvas_runtime_worker_v2.js",
    "canvas_renderer_v2.js",
    "mermaid-subset.json",
    "MERMAID_THIRD_PARTY_LICENSES.txt",
    "mermaid-runtime-manifest.json",
    "profile-catalog.json",
}


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_canvas_mermaid_assets", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    scripts = str(SCRIPT_PATH.parent)
    sys.path.insert(0, scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
    return module


def _download(name: str, data: bytes) -> dict[str, dict[str, object]]:
    return {
        name: {
            "url": f"https://example.invalid/{name}",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    }


def _write_manifest(path: Path, downloads: dict[str, dict[str, object]]) -> None:
    path.write_text(json.dumps({"downloads": downloads}), encoding="utf-8")


def test_acquisition_copies_only_declared_hash_and_size_pinned_inputs(tmp_path):
    """Accepting extras or bad bytes would broaden or weaken build provenance."""
    checker = _load_checker()
    source = tmp_path / "source"
    isolated = tmp_path / "isolated"
    source.mkdir()
    payload = b"declared input"
    (source / "declared.bin").write_bytes(payload)
    (source / "undeclared.bin").write_bytes(b"must not be copied")

    checker.acquire_declared_inputs(
        _download("declared.bin", payload), isolated, input_dir=source
    )

    assert [path.name for path in isolated.iterdir()] == ["declared.bin"]
    assert (isolated / "declared.bin").read_bytes() == payload


@pytest.mark.parametrize("mutation", ["size", "hash", "missing"])
def test_acquisition_refuses_missing_or_mismatched_inputs(tmp_path, mutation):
    """A missing, truncated, or hash-mismatched input must stop the rebuild."""
    checker = _load_checker()
    source = tmp_path / "source"
    source.mkdir()
    payload = b"declared input"
    downloads = _download("declared.bin", payload)
    if mutation == "size":
        downloads["declared.bin"]["bytes"] = len(payload) + 1
    elif mutation == "hash":
        downloads["declared.bin"]["sha256"] = "0" * 64
    else:
        (source / "other.bin").write_bytes(payload)
    if mutation != "missing":
        (source / "declared.bin").write_bytes(payload)

    with pytest.raises(checker.MermaidAssetCheckError):
        checker.acquire_declared_inputs(
            downloads, tmp_path / "isolated", input_dir=source
        )


@pytest.mark.parametrize("mutation", ["missing", "drift", "extra"])
def test_output_comparison_refuses_incomplete_or_drifted_generated_sets(
    tmp_path, mutation
):
    """Every build output must exist exactly once and match committed bytes."""
    checker = _load_checker()
    rebuilt = tmp_path / "rebuilt"
    committed = tmp_path / "committed"
    rebuilt.mkdir()
    committed.mkdir()
    generated = {}
    for name in EXPECTED_OUTPUTS:
        data = f"bytes:{name}".encode()
        (rebuilt / name).write_bytes(data)
        (committed / name).write_bytes(data)
        generated[name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    if mutation == "missing":
        (rebuilt / "profile-catalog.json").unlink()
    elif mutation == "drift":
        (committed / "profile-catalog.json").write_bytes(b"stale")
    else:
        (rebuilt / "unexpected.txt").write_bytes(b"unexpected")

    with pytest.raises(checker.MermaidAssetCheckError):
        checker.compare_generated_outputs(rebuilt, committed, generated)


def test_check_runs_existing_builder_in_isolation_and_accepts_exact_outputs(
    tmp_path, monkeypatch
):
    """Skipping the build could let stale packaged bytes pass unchanged."""
    checker = _load_checker()
    source = tmp_path / "source"
    source.mkdir()
    payload = b"declared input"
    (source / "declared.bin").write_bytes(payload)
    manifest = tmp_path / "inputs.json"
    _write_manifest(manifest, _download("declared.bin", payload))
    committed = tmp_path / "committed"
    committed.mkdir()

    def controlled_build(input_dir: Path, output_dir: Path):
        assert input_dir != source
        assert [path.name for path in input_dir.iterdir()] == ["declared.bin"]
        assert (input_dir / "declared.bin").read_bytes() == payload
        result = {}
        for name in EXPECTED_OUTPUTS:
            data = f"rebuilt:{name}".encode()
            (output_dir / name).parent.mkdir(parents=True, exist_ok=True)
            (output_dir / name).write_bytes(data)
            (committed / name).write_bytes(data)
            result[name] = {
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        return result

    monkeypatch.setattr(checker, "build", controlled_build)

    result = checker.check_assets(
        input_dir=source, manifest_path=manifest, committed_dir=committed
    )

    assert set(result) == EXPECTED_OUTPUTS


def test_main_uses_explicit_offline_input_environment(tmp_path, monkeypatch, capsys):
    """Ignoring the local cache would unexpectedly fetch during preflight."""
    checker = _load_checker()
    monkeypatch.setenv("TLDW_CANVAS_MERMAID_INPUT_DIR", str(tmp_path))

    def require_offline_cache(*, input_dir):
        if input_dir != tmp_path:
            raise AssertionError("explicit offline input cache was ignored")
        return {name: {} for name in EXPECTED_OUTPUTS}

    monkeypatch.setattr(checker, "check_assets", require_offline_cache)

    assert checker.main([]) == 0
    assert "6 outputs" in capsys.readouterr().out


def test_main_fails_when_explicit_offline_input_environment_is_missing(
    tmp_path, monkeypatch, capsys
):
    """A configured but absent local cache must fail instead of skipping."""
    checker = _load_checker()
    monkeypatch.setenv("TLDW_CANVAS_MERMAID_INPUT_DIR", str(tmp_path / "missing-cache"))

    assert checker.main([]) == 1
    assert "declared input is missing" in capsys.readouterr().err
