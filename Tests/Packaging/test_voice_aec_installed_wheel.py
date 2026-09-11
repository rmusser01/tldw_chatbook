"""Installed-wheel qualification entry point used by cibuildwheel."""

from __future__ import annotations

import hashlib
import ctypes
from importlib import import_module
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Packaging.qualify_installed_voice_aec import (  # noqa: E402
    InstalledVoiceAecError,
    qualify_installed_voice_aec,
)


MANIFEST_PATH = ROOT / "Tests" / "Audio" / "fixtures" / "voice_aec" / "manifest.json"


def verify_native_callback(module) -> None:
    """Exercise the installed callback ABI and original AEC without a device."""
    assert getattr(module, "DUPLEX_ABI_VERSION", None) == 1
    assert callable(getattr(module, "NativeDuplexBridge", None))
    assert callable(getattr(module, "AecProcessor", None))
    bridge = module.NativeDuplexBridge(generation=3)
    bridge.set_render_admission(True)
    pcm = b"\x01\x00" * 480
    assert bridge.queue_render(pcm, 0, 0)

    class Times(ctypes.Structure):
        _fields_ = [(name, ctypes.c_double) for name in ("adc", "current", "dac")]

    callback_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(Times),
        ctypes.c_ulong,
        ctypes.c_void_p,
    )
    incoming = ctypes.create_string_buffer(pcm)
    outgoing = ctypes.create_string_buffer(960)
    callback = callback_type(bridge.callback_address)
    assert (
        callback(
            incoming,
            outgoing,
            480,
            ctypes.byref(Times(10, 10.01, 10.02)),
            0,
            bridge.userdata_address,
        )
        == 0
    )
    record = bridge.pop_capture()
    assert record["pcm16"] == record["output_pcm16"] == outgoing.raw == pcm
    assert (
        record["generation"],
        record["submission_id"],
        record["reference_sequence"],
    ) == (3, 0, 0)
    processor = module.AecProcessor(sample_rate=48000, channels=1)
    processor.analyze_render(record["output_pcm16"], delay_ms=20)
    assert len(processor.process_capture(record["pcm16"], delay_ms=20)) == 960
    bridge.deactivate()


def test_explicit_native_wheel_exercises_callback_and_original_aec():
    from Tests.Audio.fakes.native_duplex_helpers import load_native

    verify_native_callback(load_native())


@pytest.mark.parametrize("abi", [None, 0, 2])
def test_wheel_entry_rejects_missing_or_old_native_callback_before_corpus(
    monkeypatch, abi
):
    monkeypatch.setattr(
        sys.modules[__name__],
        "import_module",
        lambda _: SimpleNamespace(DUPLEX_ABI_VERSION=abi, AecProcessor=object),
    )
    with pytest.raises(AssertionError):
        main()


def _passing_report() -> dict[str, object]:
    return {
        "passed": True,
        "effective_mode": "full-duplex",
        "median_erle_db": 25.0,
        "p10_erle_db": 12.0,
        "false_barge_events": 0,
        "false_barge_render_minutes": 30.0,
        "double_talk_recall": 0.97,
        "unqualified_case_ids": [],
    }


def test_probe_hashes_native_extension_and_returns_content_free_evidence(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "_native.test.so"
    extension.write_bytes(b"native-extension")
    module = SimpleNamespace(AecProcessor=object)

    report = qualify_installed_voice_aec(
        MANIFEST_PATH,
        module_loader=lambda _name: module,
        version_reader=lambda _name: "0.1.8.0",
        extension_resolver=lambda: extension,
        corpus_evaluator=lambda _manifest, _factory: _passing_report(),
    )

    assert report == {
        "schema_version": 1,
        "companion_version": "0.1.8.0",
        "extension_sha256": hashlib.sha256(b"native-extension").hexdigest(),
        **_passing_report(),
    }
    assert not ({"transcript", "response", "pcm", "device_name"} & report.keys())


@pytest.mark.parametrize(
    ("report_update", "message"),
    [
        ({"passed": False}, "did not pass"),
        ({"unqualified_case_ids": ["delay-step"]}, "unqualified"),
        ({"effective_mode": "half-duplex"}, "full duplex"),
    ],
)
def test_probe_rejects_any_nonqualifying_corpus_result(
    tmp_path: Path,
    report_update: dict[str, object],
    message: str,
) -> None:
    extension = tmp_path / "_native.test.so"
    extension.write_bytes(b"native-extension")
    report = _passing_report()
    report.update(report_update)

    with pytest.raises(InstalledVoiceAecError, match=message):
        qualify_installed_voice_aec(
            MANIFEST_PATH,
            module_loader=lambda _name: SimpleNamespace(AecProcessor=object),
            version_reader=lambda _name: "0.1.8.0",
            extension_resolver=lambda: extension,
            corpus_evaluator=lambda _manifest, _factory: report,
        )


def test_probe_rejects_missing_or_non_native_extension(tmp_path: Path) -> None:
    missing = tmp_path / "_native.test.so"

    with pytest.raises(InstalledVoiceAecError, match="native extension"):
        qualify_installed_voice_aec(
            MANIFEST_PATH,
            module_loader=lambda _name: SimpleNamespace(AecProcessor=object),
            version_reader=lambda _name: "0.1.8.0",
            extension_resolver=lambda: missing,
            corpus_evaluator=lambda _manifest, _factory: _passing_report(),
        )


def test_wheel_workflow_runs_the_installed_corpus_probe_for_every_wheel() -> None:
    workflow = (ROOT / ".github" / "workflows" / "voice-aec-wheels.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "python {project}/../../Tests/Packaging/test_voice_aec_installed_wheel.py"
        in workflow
    )
    for required_trigger in (
        '"Packaging/voice_aec_corpus.py"',
        '"Packaging/qualify_installed_voice_aec.py"',
        '"Tests/Audio/fixtures/voice_aec/manifest.json"',
        '"Tests/Packaging/test_voice_aec_installed_wheel.py"',
    ):
        assert required_trigger in workflow


def main() -> int:
    """Qualify the wheel installed by cibuildwheel and print safe JSON."""

    verify_native_callback(import_module("tldw_voice_aec"))
    report = qualify_installed_voice_aec(MANIFEST_PATH)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in wheel isolation
    raise SystemExit(main())
