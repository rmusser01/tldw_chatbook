"""Inventory and unsupported live requests never import or call a transport."""

import json
import runpy
import sys
from pathlib import Path

import pytest

PROBE = (
    Path(__file__).resolve().parents[2]
    / "Docs/superpowers/qa/portrait-compositing/2026-10-24/live_adapter_probe.py"
)


@pytest.fixture(autouse=True)
def _disable_model_catalog_refresh():
    """Pure inventory never imports the App or starts model/network discovery."""


@pytest.mark.parametrize(
    "backend,execute,status",
    [
        ("swarmui", False, "unavailable"),
        ("swarmui", True, "unavailable"),
        ("gemini", False, "available_not_run"),
    ],
)
def test_probe_inventory_does_not_send_requests_or_disclose_credentials(
    tmp_path, capsys, backend, execute, status
):
    config = tmp_path / "config.toml"
    config.write_text(
        f'[image_generation]\ndefault_backend="{backend}"\nenabled_backends=["{backend}"]\n[image_generation.{backend}]\napi_key="private-fixture-secret"\n'
    )
    output = tmp_path / "evidence"
    args = ["--config", str(config), "--output", str(output)]
    if execute:
        args.append("--accept-five-image-requests")
    assert runpy.run_path(str(PROBE))["main"](args) == 0
    report = (output / "availability.json").read_text()
    assert json.loads(report)["status"] == status
    assert json.loads(report)["image_requests"] == 0
    assert "private-fixture-secret" not in report + capsys.readouterr().out


@pytest.mark.parametrize("fail", [False, True])
def test_opted_in_probe_preserves_evidence_without_claiming_visual_qualification(
    tmp_path, monkeypatch, capsys, fail
):
    from Tests.Backup_Recovery import test_home_citation_retirement as runner

    calls = []

    def run(root, route, outcome, *, script, timeout):
        calls.append(script)
        assert "images=QualificationImages(live=True)" in script
        assert script.count("set_mode(NetworkMode.ALLOW_ALL)") == 1
        assert "private-fixture-secret" not in script
        settings = root / "image-settings.toml"
        assert "private-fixture-secret" in settings.read_text()
        if sys.platform != "win32":
            assert settings.stat().st_mode & 0o777 == 0o600
        assert timeout == 900
        artifacts = root / "home/portrait-artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "requests.json").write_text(json.dumps([{}] * (1 if fail else 5)))
        (artifacts / "pair-mouth.png").write_bytes(b"fixture-evidence")
        if fail:
            raise RuntimeError("private-fixture-secret in discarded transport log")
        (root / "home/portrait-qualification.json").write_text(
            json.dumps({"real_image_adapter": True, "image_calls": 5})
        )

    monkeypatch.setattr(runner, "_run", run)
    config = tmp_path / "config.toml"
    config.write_text(
        '[image_generation]\ndefault_backend="gemini"\nenabled_backends=["gemini"]\ngemini_image_api_key="private-fixture-secret"\n'
    )
    output = tmp_path / "evidence"
    assert (
        runpy.run_path(str(PROBE))["main"](
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--accept-five-image-requests",
            ]
        )
        == 0
    )
    report = (output / "availability.json").read_text()
    result = json.loads(report)
    assert result["status"] == (
        "failed_or_uncertain" if fail else "executed_review_required"
    )
    assert result["image_requests"] == (1 if fail else 5)
    assert not result["live_qualified"] and len(calls) == 1
    assert (output / "artifacts/pair-mouth.png").read_bytes() == b"fixture-evidence"
    assert "private-fixture-secret" not in report + capsys.readouterr().out


def test_opted_in_probe_refuses_existing_evidence_before_transport(
    tmp_path, monkeypatch
):
    from Tests.Backup_Recovery import test_home_citation_retirement as runner

    def unexpected(*args, **kwargs):
        pytest.fail("Existing evidence must be rejected before any request")

    monkeypatch.setattr(runner, "_run", unexpected)
    config = tmp_path / "config.toml"
    config.write_text(
        '[image_generation]\ndefault_backend="gemini"\nenabled_backends=["gemini"]\n'
    )
    output = tmp_path / "evidence"
    (output / "artifacts").mkdir(parents=True)
    marker = output / "artifacts/kept.txt"
    marker.write_text("prior evidence")
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(PROBE))["main"](
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--accept-five-image-requests",
            ]
        )
    assert error.value.code == 2 and marker.read_text() == "prior evidence"


@pytest.mark.parametrize("interruption_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("complete_manifest", [True, False])
def test_interrupted_probe_keeps_request_uncertainty_and_cleans_private_files(
    tmp_path, monkeypatch, capsys, interruption_type, complete_manifest
):
    from Tests.Backup_Recovery import test_home_citation_retirement as runner

    interruption = interruption_type("interrupted transport")
    private_roots = []
    manifest = '[{"index":0}]' if complete_manifest else '[{"index":'

    def interrupt(root, *args, **kwargs):
        private_roots.append(root)
        assert (root / "image-settings.toml").is_file()
        artifacts = root / "home/portrait-artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "requests.json").write_text(manifest)
        (artifacts / "response-0.png").write_bytes(b"returned-artwork")
        (root / "restart-output.log").write_text("private-fixture-secret")
        raise interruption

    monkeypatch.setattr(runner, "_run", interrupt)
    config = tmp_path / "config.toml"
    config.write_text(
        '[image_generation]\ndefault_backend="gemini"\nenabled_backends=["gemini"]\ngemini_image_api_key="private-fixture-secret"\n'
    )
    output = tmp_path / "evidence"
    with pytest.raises(interruption_type) as caught:
        runpy.run_path(str(PROBE))["main"](
            [
                "--config",
                str(config),
                "--output",
                str(output),
                "--accept-five-image-requests",
            ]
        )
    assert caught.value is interruption
    report = (output / "availability.json").read_text()
    result = json.loads(report)
    assert result["status"] == "failed_or_uncertain" and not result["live_qualified"]
    assert result["image_requests"] == (1 if complete_manifest else None)
    assert (output / "artifacts/requests.json").read_text() == manifest
    assert (output / "artifacts/response-0.png").read_bytes() == b"returned-artwork"
    assert not (output / "restart-output.log").exists()
    assert len(private_roots) == 1 and not private_roots[0].exists()
    captured = capsys.readouterr()
    assert "private-fixture-secret" not in report + captured.out + captured.err
