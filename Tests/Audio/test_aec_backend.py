"""Tests for the optional native AEC loader."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tldw_chatbook.Audio import aec_backend


def test_import_does_not_load_native_companion() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import tldw_chatbook.Audio.aec_backend; "
                "assert 'tldw_voice_aec' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stderr


def test_factory_imports_native_companion_only_when_called(monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    class NativeProcessor:
        def __init__(self, *, sample_rate: int, channels: int) -> None:
            calls.append((sample_rate, channels))

    imported: list[str] = []

    def fake_import(name: str) -> object:
        imported.append(name)
        return SimpleNamespace(AecProcessor=NativeProcessor)

    monkeypatch.setattr(aec_backend, "import_module", fake_import)

    processor = aec_backend.create_aec_processor()

    assert processor is not None
    assert imported == ["tldw_voice_aec"]
    assert calls == [(48_000, 1)]


def test_missing_native_companion_selects_unavailable_without_raising(
    monkeypatch,
) -> None:
    def missing(_name: str) -> object:
        raise ModuleNotFoundError("tldw_voice_aec")

    monkeypatch.setattr(aec_backend, "import_module", missing)

    assert aec_backend.create_aec_processor() is None


def test_native_initialization_failure_selects_unavailable(monkeypatch) -> None:
    class BrokenProcessor:
        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("device DSP initialization failed")

    monkeypatch.setattr(
        aec_backend,
        "import_module",
        lambda _name: SimpleNamespace(AecProcessor=BrokenProcessor),
    )

    assert aec_backend.create_aec_processor() is None


def test_real_installed_wheel_converges_through_application_health_gate(
    tmp_path: Path,
) -> None:
    """Exercise the qualified native wheel, not an in-process metrics fake."""

    wheel_value = os.environ.get("TLDW_VOICE_AEC_WHEEL")
    if not wheel_value:
        pytest.skip("set TLDW_VOICE_AEC_WHEEL for the installed-wheel probe")
    wheel = Path(wheel_value).resolve()
    assert wheel.is_file(), wheel
    install_root = tmp_path / "installed"
    install = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--target",
            str(install_root),
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import asyncio
                from collections import deque
                import importlib.util
                import os
                import random
                import struct

                from tldw_chatbook.Audio.duplex_contracts import (
                    AecDelayEvidence,
                    AecHealth,
                    AudioFrame,
                    DuplexMode,
                )
                from tldw_chatbook.Audio.voice_preprocessor import VoicePreprocessor

                install_root = os.environ["INSTALL_ROOT"]
                spec = importlib.util.find_spec("tldw_voice_aec._native")
                assert spec is not None and spec.origin.startswith(install_root), spec

                async def run():
                    processor = VoicePreprocessor.from_native(healthy_streak=3)
                    assert processor.mode is DuplexMode.HALF_DUPLEX
                    native = processor._aec
                    assert native is not None
                    rng = random.Random(0xAEC3)
                    delayed = deque([bytes(960)] * 4)
                    for sequence in range(1_200):
                        base_ns = 1_000_000_000 + sequence * 10_000_000
                        render = struct.pack(
                            "<480h",
                            *(rng.randint(-12_000, 12_000) for _ in range(480)),
                        )
                        capture = delayed.popleft()
                        delayed.append(render)
                        timing = AecDelayEvidence(
                            observed_ns=base_ns,
                            capture_adc_ns=base_ns - 20_000_000,
                            render_dac_ns=base_ns + 20_000_000,
                            delay_ms=40,
                            capture_occupancy_frames=0,
                            render_occupancy_frames=0,
                            occupancy_bounded=True,
                            timing_discontinuity=False,
                            clock_drift=False,
                            status_flags=(),
                        )
                        capture_frame = AudioFrame(
                            sequence=sequence,
                            started_ns=base_ns,
                            ended_ns=base_ns + 10_000_000,
                            pcm16=capture,
                            delay_evidence=timing,
                            render_reference_sequence=sequence,
                        )
                        render_frame = AudioFrame(
                            sequence=sequence,
                            started_ns=base_ns,
                            ended_ns=base_ns + 10_000_000,
                            pcm16=render,
                            delay_evidence=timing,
                        )
                        await processor.process_capture(
                            capture_frame,
                            render_frames=(render_frame,),
                            assistant_rendering=True,
                        )
                        if processor.health is AecHealth.HEALTHY:
                            break
                    assert processor.health is AecHealth.HEALTHY
                    assert processor.mode is DuplexMode.FULL_DUPLEX
                    metrics = native.metrics()
                    assert metrics["delay_estimate_available"] == 1.0
                    assert metrics["delay_estimate_refined"] == 1.0
                    assert metrics["delay_age_blocks"] <= 25.0
                    assert metrics["clock_drift"] == 0.0
                    assert metrics["erle_db"] >= 5.0

                asyncio.run(run())
                """
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "INSTALL_ROOT": str(install_root),
            "PYTHONPATH": os.pathsep.join([str(install_root), str(Path.cwd())]),
        },
    )

    assert probe.returncode == 0, probe.stderr
