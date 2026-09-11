"""Live physical voice runner tests using the production transport seam."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

import pytest

from Packaging import physical_voice_runner
from Packaging.physical_voice_runner import (
    LiveTrialConfig,
    PhysicalVoiceRunnerError,
    PhysicalVoiceTrialRunner,
    RouteIdentity,
    load_physical_reference,
    prompt_operator,
    read_default_route,
)
from Packaging.voice_aec_corpus import (
    iter_voice_aec_case_frames,
    load_voice_aec_corpus,
)
from Packaging.voice_physical_reports import (
    AutomatedPrerequisiteEvidence,
    build_physical_report,
)
from Tests.Audio.fakes.fake_duplex_backend import FakeDuplexBackend, ManualClock
from tldw_chatbook.Audio.acoustic_isolation import AcousticIsolationMonitor
from tldw_chatbook.Audio.duplex_contracts import (
    AcousticDemotionReason,
    AcousticIsolationObservation,
    AcousticSafetyPath,
    AcousticSafetySnapshot,
    AecHealth,
    DuplexMode,
    NearEndDisposition,
)
from tldw_chatbook.Audio.duplex_transport import (
    FRAME_BYTES,
    FRAME_DURATION_NS,
    DuplexAudioTransport,
)


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "Packaging" / "assets" / "voice_physical_reference.wav"
MANIFEST = ROOT / "Packaging" / "assets" / "voice_physical_reference.json"
EXPECTED_ASSET_SHA256 = (
    "d3f1822e269f009d029be6973e89d1f679bd299d9ad6fdec43d80923882cb9c5"
)
SILENCE = bytes(FRAME_BYTES)
_NEAR_END_PREFIX = b"\x39\x30\xc7\xcf\xa0\x5b\x60\xa4"


def _near_end_pcm(frame_index: int) -> bytes:
    state = 0xA5A5A5A5 ^ frame_index
    samples = bytearray(_NEAR_END_PREFIX)
    while len(samples) < FRAME_BYTES:
        state = (1_664_525 * state + 1_013_904_223) & 0xFFFFFFFF
        magnitude = 512 + (state & 0x0FFF)
        value = magnitude if state & 0x10000 else -magnitude
        samples.extend(value.to_bytes(2, "little", signed=True))
        samples.extend((-value).to_bytes(2, "little", signed=True))
    return bytes(samples[:FRAME_BYTES])


def test_reference_asset_is_hash_verified_redistributable_pcm16() -> None:
    frames = load_physical_reference(ASSET, MANIFEST)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert set(manifest) == {
        "schema_version",
        "audio_format",
        "source",
        "license_spdx",
        "redistribution_grant",
        "sha256",
    }
    assert set(manifest["audio_format"]) == {
        "sample_rate_hz",
        "channels",
        "sample_width_bytes",
        "sample_format",
        "frame_duration_ms",
        "frame_count",
    }
    assert set(manifest["source"]) == {
        "kind",
        "generator",
        "source_manifest",
        "source_case_id",
        "source_recipe_sha256",
        "contains_captured_audio",
        "contains_user_audio",
    }
    assert manifest["license_spdx"] == "CC0-1.0"
    assert manifest["redistribution_grant"] == (
        "The deterministic generated audio is dedicated under CC0-1.0 and may be "
        "redistributed in this repository and its distributions without restriction."
    )
    assert (
        physical_voice_runner._EXPECTED_ASSET_SHA256  # noqa: SLF001
        == EXPECTED_ASSET_SHA256
    )
    assert manifest["sha256"] == EXPECTED_ASSET_SHA256
    assert hashlib.sha256(ASSET.read_bytes()).hexdigest() == EXPECTED_ASSET_SHA256
    assert manifest["audio_format"] == {
        "sample_rate_hz": 48_000,
        "channels": 1,
        "sample_width_bytes": 2,
        "sample_format": "pcm_s16le",
        "frame_duration_ms": 10,
        "frame_count": 600,
    }
    assert len(frames) == 600
    assert all(len(frame) == FRAME_BYTES for frame in frames)
    assert ASSET.is_file() and not ASSET.is_symlink()
    assert MANIFEST.is_file() and not MANIFEST.is_symlink()
    with wave.open(str(ASSET), "rb") as wav_file:
        assert (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
            wav_file.getnframes(),
            wav_file.getcomptype(),
        ) == (1, 2, 48_000, 288_000, "NONE")


def test_reference_asset_is_exactly_reproducible_from_declared_recipe() -> None:
    reference_manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    corpus = load_voice_aec_corpus(
        ROOT / "Tests/Audio/fixtures/voice_aec/manifest.json"
    )
    case_id, selection = reference_manifest["source"]["source_case_id"].split(":")
    case = next(item for item in corpus["cases"] if item["id"] == case_id)
    assert selection == "first-600-render-frames"
    assert case["recipe_sha256"] == reference_manifest["source"]["source_recipe_sha256"]
    encoded = io.BytesIO()
    with wave.open(encoded, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(48_000)
        for frame in iter_voice_aec_case_frames(case, limit=600):
            wav_file.writeframesraw(frame.render_pcm16)

    generated = encoded.getvalue()
    assert generated == ASSET.read_bytes()
    assert hashlib.sha256(generated).hexdigest() == EXPECTED_ASSET_SHA256


@pytest.mark.parametrize(
    "mutation",
    [
        "hash",
        "hash-bool",
        "hash-int",
        "format",
        "schema-bool",
        "channels-bool",
        "captured-int",
        "format-container",
        "extra",
        "missing",
        "duplicate-top",
        "duplicate-format",
        "duplicate-source",
        "asset-and-hash",
    ],
)
def test_reference_tampering_is_rejected_before_transport_open(
    tmp_path: Path,
    mutation: str,
) -> None:
    copied_asset = tmp_path / "reference.wav"
    copied_manifest = tmp_path / "reference.json"
    copied_asset.write_bytes(ASSET.read_bytes())
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    value = json.loads(manifest_text)
    if mutation == "hash":
        value["sha256"] = "0" * 64
    elif mutation == "hash-bool":
        value["sha256"] = True
    elif mutation == "hash-int":
        value["sha256"] = 0
    elif mutation == "format":
        value["audio_format"]["sample_rate_hz"] = 44_100
    elif mutation == "schema-bool":
        value["schema_version"] = True
    elif mutation == "channels-bool":
        value["audio_format"]["channels"] = True
    elif mutation == "captured-int":
        value["source"]["contains_captured_audio"] = 0
    elif mutation == "format-container":
        value["audio_format"] = []
    elif mutation == "asset-and-hash":
        encoded = bytearray(copied_asset.read_bytes())
        encoded[-1] ^= 1
        copied_asset.write_bytes(encoded)
        value["sha256"] = hashlib.sha256(encoded).hexdigest()
    elif mutation == "extra":
        value["unexpected"] = "field"
    elif mutation == "missing":
        del value["license_spdx"]
    elif mutation == "duplicate-top":
        manifest_text = manifest_text.replace(
            '  "schema_version": 1,',
            '  "schema_version": 1,\n  "schema_version": 1,',
            1,
        )
    elif mutation == "duplicate-format":
        manifest_text = manifest_text.replace(
            '    "channels": 1,',
            '    "channels": 1,\n    "channels": 1,',
            1,
        )
    else:
        manifest_text = manifest_text.replace(
            '    "contains_captured_audio": false,',
            (
                '    "contains_captured_audio": false,\n'
                '    "contains_captured_audio": false,'
            ),
            1,
        )
    copied_manifest.write_text(
        manifest_text if mutation.startswith("duplicate-") else json.dumps(value),
        encoding="utf-8",
    )
    backend = FakeDuplexBackend()

    runner = PhysicalVoiceTrialRunner(
        device_class="usb",
        config=_short_config(),
        transport_factory=lambda: DuplexAudioTransport(backend=backend),
        route_reader=lambda: _target_route(),
        prompt=_noop_prompt,
        reference_path=copied_asset,
        manifest_path=copied_manifest,
    )

    with pytest.raises(PhysicalVoiceRunnerError, match="reference"):
        asyncio.run(runner.run())
    assert backend.open_count == 0


@pytest.mark.parametrize("symlinked", ["asset", "manifest"])
def test_reference_loader_rejects_symlinks(
    tmp_path: Path,
    symlinked: str,
) -> None:
    copied_asset = tmp_path / "reference.wav"
    copied_manifest = tmp_path / "reference.json"
    if symlinked == "asset":
        copied_asset.symlink_to(ASSET)
        copied_manifest.write_bytes(MANIFEST.read_bytes())
    else:
        copied_asset.write_bytes(ASSET.read_bytes())
        copied_manifest.symlink_to(MANIFEST)

    with pytest.raises(PhysicalVoiceRunnerError, match="reference"):
        load_physical_reference(copied_asset, copied_manifest)


def test_reference_loader_rejects_path_swap_after_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_asset = tmp_path / "reference.wav"
    copied_manifest = tmp_path / "reference.json"
    copied_asset.write_bytes(ASSET.read_bytes())
    copied_manifest.write_bytes(MANIFEST.read_bytes())
    replacement = tmp_path / "replacement.wav"
    replacement.write_bytes(ASSET.read_bytes())
    original_open = os.open
    swapped = False

    def swap_after_open(path: object, flags: int, *args: object) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, *args)  # type: ignore[arg-type]
        if Path(path) == copied_asset and not swapped:  # type: ignore[arg-type]
            os.replace(replacement, copied_asset)
            swapped = True
        return descriptor

    monkeypatch.setattr(physical_voice_runner.os, "open", swap_after_open)

    with pytest.raises(PhysicalVoiceRunnerError, match="reference"):
        load_physical_reference(copied_asset, copied_manifest)
    assert swapped is True


def test_reference_loader_uses_explicitly_bounded_descriptor_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_sizes: list[int] = []
    original_read = os.read

    def bounded_read(descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(physical_voice_runner.os, "read", bounded_read)

    frames = load_physical_reference(ASSET, MANIFEST)

    assert len(frames) == 600
    assert read_sizes
    assert max(read_sizes) <= len(ASSET.read_bytes()) + 1


def test_oversized_sparse_manifest_is_rejected_before_any_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized_manifest = tmp_path / "oversized.json"
    with oversized_manifest.open("wb") as stream:
        stream.truncate(16 * 1024 + 1)

    def forbid_any_read(_descriptor: int, _size: int) -> bytes:
        raise AssertionError("oversized files must be rejected before reading")

    monkeypatch.setattr(physical_voice_runner.os, "read", forbid_any_read)

    with pytest.raises(PhysicalVoiceRunnerError, match="reference"):
        load_physical_reference(ASSET, oversized_manifest)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_reference_loader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "reference.wav"
    os.mkfifo(fifo)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from Packaging.physical_voice_runner import ("
                "PhysicalVoiceRunnerError, load_physical_reference); "
                "import sys; "
                "asset, manifest = map(Path, sys.argv[1:]); "
                "\ntry: load_physical_reference(asset, manifest)"
                "\nexcept PhysicalVoiceRunnerError: raise SystemExit(0)"
                "\nraise SystemExit(1)"
            ),
            str(fifo),
            str(MANIFEST),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert probe.returncode == 0, probe.stderr


def test_transport_exposes_only_bounded_live_diagnostics() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())

    assert transport.is_open is False
    assert transport.control_overflows == 0

    asyncio.run(transport.start())
    assert transport.is_open is True
    asyncio.run(transport.close())
    assert transport.is_open is False


def test_runner_reserves_shared_render_reference_capacity_atomically() -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=ManualClock(),
        capture_capacity=64,
        render_capacity=64,
    )

    async def no_wait(_seconds: float) -> None:
        return None

    runner = PhysicalVoiceTrialRunner(device_class="usb", wait=no_wait)
    runner._reference_frames = (SILENCE,)
    runner._transport = transport

    async def fill_and_tick() -> tuple[object, object]:
        await transport.start()
        for _ in range(63):
            assert transport.queue_render(SILENCE) is not None
            backend.stream.emit_capture(SILENCE)
            assert transport.pop_capture() is not None
        assert transport.queue_render(SILENCE) is not None
        before = transport.buffer_occupancy
        await runner._tick(rendering=True, measure=False)
        after = transport.buffer_occupancy
        await transport.close()
        runner._release_buffers()
        return before, after

    before, after = asyncio.run(fill_and_tick())

    assert (before.render_reference_frames, before.render_frames) == (63, 1)
    assert after.render_frames == 1
    assert transport.render_overflows == 0
    assert transport.render_reference_overflows == 0


def test_processing_tick_does_not_chase_concurrent_capture_producers() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()

    class RefillingTransport(DuplexAudioTransport):
        def __init__(self) -> None:
            super().__init__(backend=backend, clock=clock)
            self.refills = 0

        def pop_capture(self) -> object:
            frame = super().pop_capture()
            if frame is not None and self.refills < 5:
                backend.stream.emit_capture(SILENCE)
                self.refills += 1
            return frame

    class CountingPreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.HALF_DUPLEX,
            demotion_reason=None,
        )

        def __init__(self) -> None:
            self.capture_count = 0

        async def process_capture(self, *_args: object, **_kwargs: object) -> None:
            self.capture_count += 1

    transport = RefillingTransport()
    preprocessor = CountingPreprocessor()
    runner = PhysicalVoiceTrialRunner(device_class="usb", clock=clock)
    runner._transport = transport
    runner._preprocessor = preprocessor  # type: ignore[assignment]

    async def process_one_entry_budget() -> int:
        await transport.start()
        backend.stream.emit_capture(SILENCE)
        await runner._process_available(rendering=False, measure=False)
        remaining = transport.capture_count
        await transport.close()
        return remaining

    remaining = asyncio.run(process_one_entry_budget())

    assert preprocessor.capture_count == 1
    assert remaining == 1


def test_processing_tick_does_not_chase_concurrent_reference_producers() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()

    class RefillingTransport(DuplexAudioTransport):
        def __init__(self) -> None:
            super().__init__(backend=backend, clock=clock, render_capacity=8)
            self.refills = 0

        def pop_render_reference(self) -> object:
            frame = super().pop_render_reference()
            if frame is not None and self.refills < 5:
                assert self.queue_render(SILENCE) is not None
                backend.stream.emit_capture(SILENCE)
                self.refills += 1
            return frame

    transport = RefillingTransport()
    runner = PhysicalVoiceTrialRunner(device_class="usb", clock=clock)
    runner._transport = transport

    class PassivePreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.HALF_DUPLEX,
            admission_open=False,
            demotion_reason=None,
        )

        async def process_capture(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("new captures are outside the entry snapshot budget")

    runner._preprocessor = PassivePreprocessor()  # type: ignore[assignment]

    async def process_one_entry_budget() -> int:
        await transport.start()
        assert transport.queue_render(SILENCE) is not None
        backend.stream.emit_capture(SILENCE)
        assert transport.pop_capture() is not None
        await runner._process_available(rendering=False, measure=False)
        remaining = transport.buffer_occupancy.render_reference_frames
        await transport.close()
        return remaining

    remaining = asyncio.run(process_one_entry_budget())

    assert len(runner._pending_references) == 1
    assert remaining == 1


def test_pending_reference_overflow_is_bounded_and_returns_fail_evidence() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(
        backend=backend,
        clock=clock,
        capture_capacity=8,
        render_capacity=2,
    )

    class PassivePreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.HALF_DUPLEX,
            admission_open=False,
            demotion_reason=None,
        )
        health = AecHealth.DEGRADED

        async def process_capture(self, *_args: object, **_kwargs: object) -> None:
            return None

    runner = PhysicalVoiceTrialRunner(device_class="usb", clock=clock)
    runner._transport = transport
    runner._preprocessor = PassivePreprocessor()  # type: ignore[assignment]

    async def overflow_pending_references() -> None:
        await transport.start()
        for _ in range(3):
            assert transport.queue_render(SILENCE) is not None
            backend.stream.emit_capture(SILENCE)
            assert transport.pop_capture() is not None
            await runner._process_available(rendering=False, measure=False)
        await transport.close()

    asyncio.run(overflow_pending_references())
    observations = runner._build_observations(
        target_route=_target_route(),
        device_identifier="d" * 64,
        operator_checklist={
            "no_glitch_burst": False,
            "no_runaway_feedback": False,
            "route_recovered": False,
            "speech_remained_intelligible": False,
        },
    )

    assert len(runner._pending_references) <= 2
    assert observations["reference_overflows"] == 1
    baseline = json.loads(
        (
            ROOT / "Tests/Packaging/fixtures/speculative_voice_physical/"
            "safe_half_duplex.json"
        ).read_text(encoding="utf-8")
    )
    baseline["reference_overflows"] = observations["reference_overflows"]
    report = build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="bluetooth",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=baseline,
        salt=b"r" * 16,
    )
    assert report["passed"] is False


@pytest.mark.parametrize(
    ("phase", "rendering", "counter_name"),
    [
        ("render-only", True, "_rendered_false_barges"),
        ("silence", False, "_silence_false_barges"),
    ],
)
def test_admission_events_count_distinct_onsets_not_replayed_frames(
    phase: str,
    rendering: bool,
    counter_name: str,
) -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    runner = PhysicalVoiceTrialRunner(device_class="usb")
    admissions = deque([5, 1, 0, 1])

    class ScriptedPreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.AEC,
            demotion_reason=None,
        )

        async def process_capture(
            self,
            capture: object,
            *,
            render_frames: tuple[object, ...],
            assistant_rendering: bool,
        ) -> None:
            del render_frames, assistant_rendering
            for _ in range(admissions.popleft()):
                runner._on_admitted_frame(capture)  # type: ignore[arg-type]

    runner._transport = transport
    runner._preprocessor = ScriptedPreprocessor()  # type: ignore[assignment]
    runner._phase = phase

    async def process_script() -> None:
        await transport.start()
        for _ in range(4):
            backend.stream.emit_capture(SILENCE)
        await runner._process_available(rendering=rendering, measure=True)
        await transport.close()

    asyncio.run(process_script())

    assert getattr(runner, counter_name) == 2
    if phase == "render-only":
        assert runner._render_only_vad_events == 2


def test_admission_onset_state_resets_between_measurement_phases() -> None:
    runner = PhysicalVoiceTrialRunner(device_class="usb")
    runner._enter_phase("render-only")
    runner._capture_admitted = True
    runner._record_admission_onset()

    runner._enter_phase("silence")
    runner._capture_admitted = True
    runner._record_admission_onset()

    assert runner._rendered_false_barges == 1
    assert runner._render_only_vad_events == 1
    assert runner._silence_false_barges == 1


def test_default_route_reader_checks_both_48khz_mono_legs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class InputOutputPair:
        """Match sounddevice's indexable pair (not a tuple/list, no len)."""

        def __getitem__(self, index: int) -> int:
            if index == 0:
                return 7
            if index == 1:
                return 8
            raise IndexError(index)

    class FakeSounddevice:
        default = SimpleNamespace(device=InputOutputPair())

        @staticmethod
        def query_devices(index: int, kind: str) -> dict[str, object]:
            calls.append((f"query-{kind}", {"device": index}))
            return {
                "name": f"PRIVATE {kind}",
                f"max_{kind}_channels": 2,
            }

        @staticmethod
        def check_input_settings(**kwargs: object) -> None:
            calls.append(("check-input", kwargs))

        @staticmethod
        def check_output_settings(**kwargs: object) -> None:
            calls.append(("check-output", kwargs))

    imports: list[str] = []

    def fake_import(name: str) -> object:
        imports.append(name)
        return FakeSounddevice

    monkeypatch.setattr("Packaging.physical_voice_runner.import_module", fake_import)

    route = read_default_route()

    assert imports == ["sounddevice"]
    assert route.identity_key == ("PRIVATE input", "PRIVATE output")
    assert route.input_name == "PRIVATE input"
    assert route.output_name == "PRIVATE output"
    assert calls == [
        ("query-input", {"device": 7}),
        ("query-output", {"device": 8}),
        (
            "check-input",
            {"device": 7, "samplerate": 48_000, "channels": 1, "dtype": "int16"},
        ),
        (
            "check-output",
            {"device": 8, "samplerate": 48_000, "channels": 1, "dtype": "int16"},
        ),
    ]


def test_default_route_refresh_reinitializes_portaudio_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeSounddevice:
        default = SimpleNamespace(device=(1, 2))

        @classmethod
        def _terminate(cls) -> None:
            events.append("terminate")

        @classmethod
        def _initialize(cls) -> None:
            events.append("initialize")
            cls.default.device = (3, 4)

        @staticmethod
        def query_devices(index: int, kind: str) -> dict[str, object]:
            events.append(f"query-{kind}-{index}")
            return {"name": f"PRIVATE {kind}", f"max_{kind}_channels": 2}

        @staticmethod
        def check_input_settings(**_kwargs: object) -> None:
            events.append("check-input")

        @staticmethod
        def check_output_settings(**_kwargs: object) -> None:
            events.append("check-output")

    monkeypatch.setattr(
        "Packaging.physical_voice_runner.import_module",
        lambda _name: FakeSounddevice,
    )

    route = physical_voice_runner.refresh_default_route()

    assert route.identity_key == ("PRIVATE input", "PRIVATE output")
    assert events == [
        "terminate",
        "initialize",
        "query-input-3",
        "query-output-4",
        "check-input",
        "check-output",
    ]


@pytest.mark.parametrize(
    ("defaults", "message"),
    [
        ((7,), "API shape"),
        ((7, 8, 9), "API shape"),
        ((True, 8), "integer indices"),
        ((7.0, 8), "integer indices"),
        ((-1, -1), "unavailable"),
    ],
)
def test_default_route_reader_distinguishes_shape_from_unavailable_defaults(
    monkeypatch: pytest.MonkeyPatch,
    defaults: object,
    message: str,
) -> None:
    fake_sounddevice = SimpleNamespace(default=SimpleNamespace(device=defaults))
    monkeypatch.setattr(
        "Packaging.physical_voice_runner.import_module",
        lambda _name: fake_sounddevice,
    )

    with pytest.raises(PhysicalVoiceRunnerError, match=message):
        read_default_route()


@pytest.mark.skipif(os.name != "posix", reason="POSIX add_reader contract")
def test_operator_prompt_cancellation_removes_stdin_reader_and_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()

    class TtyPipe:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return read_fd

        def readline(self) -> str:
            return os.read(read_fd, 4_096).decode()

    monkeypatch.setattr(sys, "stdin", TtyPipe())

    async def forbidden_thread(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("operator prompts must not create worker threads")

    monkeypatch.setattr(
        "Packaging.physical_voice_runner.asyncio.to_thread",
        forbidden_thread,
    )

    async def cancel_prompt() -> tuple[bool, bool]:
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(prompt_operator("cancel me"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task.done(), loop.remove_reader(read_fd)

    try:
        done, reader_was_still_registered = asyncio.run(cancel_prompt())
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert done is True
    assert reader_was_still_registered is False


@pytest.mark.skipif(os.name != "posix", reason="POSIX add_reader contract")
def test_operator_prompt_acknowledgement_removes_stdin_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_fd, write_fd = os.pipe()

    class TtyPipe:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return read_fd

        def readline(self) -> str:
            return os.read(read_fd, 4_096).decode()

    monkeypatch.setattr(sys, "stdin", TtyPipe())
    os.write(write_fd, b"\n")

    async def acknowledge() -> bool:
        loop = asyncio.get_running_loop()
        await prompt_operator("continue")
        return loop.remove_reader(read_fd)

    try:
        reader_was_still_registered = asyncio.run(acknowledge())
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert reader_was_still_registered is False


def test_windows_operator_prompt_polls_console_and_is_cancellable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = deque([False, True])
    console = SimpleNamespace(
        kbhit=lambda: checks.popleft(),
        getwch=lambda: "\r",
    )
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "Packaging.physical_voice_runner.os",
        SimpleNamespace(name="nt"),
    )
    monkeypatch.setattr(
        "Packaging.physical_voice_runner.import_module",
        lambda name: console if name == "msvcrt" else None,
    )

    asyncio.run(prompt_operator("windows acknowledgement"))
    assert not checks

    console.kbhit = lambda: False

    async def cancel_prompt() -> bool:
        task = asyncio.create_task(prompt_operator("windows cancellation"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return task.done()

    assert asyncio.run(cancel_prompt()) is True


def test_operator_prompt_rejects_noninteractive_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    with pytest.raises(PhysicalVoiceRunnerError, match="interactive stdin"):
        asyncio.run(prompt_operator("cannot confirm"))


class _UnavailableAec:
    health = AecHealth.WARMING

    def analyze_render(self, _pcm16: bytes, *, delay_ms: int) -> None:
        assert 0 <= delay_ms <= 1_000

    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes:
        assert 0 <= delay_ms <= 1_000
        return pcm16

    def metrics(self) -> dict[str, float]:
        return {
            "erle_db": 0.0,
            "delay_ms": 20.0,
            "delay_estimate_available": 0.0,
            "delay_estimate_refined": 0.0,
            "delay_age_blocks": 0.0,
            "clock_drift": 0.0,
        }

    def reset(self) -> None:
        return None


class _HealthyAec(_UnavailableAec):
    health = AecHealth.HEALTHY

    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes:
        del delay_ms
        return pcm16 if pcm16.startswith(_NEAR_END_PREFIX) else SILENCE

    def metrics(self) -> dict[str, float]:
        return {
            "erle_db": 30.0,
            "delay_ms": 20.0,
            "delay_estimate_available": 1.0,
            "delay_estimate_refined": 1.0,
            "delay_age_blocks": 0.0,
            "clock_drift": 0.0,
        }


class _MuteHealthyAec(_HealthyAec):
    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes:
        del pcm16, delay_ms
        return SILENCE


class _AlwaysOpenIsolation(AcousticIsolationMonitor):
    def observe(
        self,
        *,
        capture: object,
        near_end_speech: bool,
        **_kwargs: object,
    ) -> AcousticIsolationObservation:
        return AcousticIsolationObservation(
            AcousticSafetySnapshot(
                AcousticSafetyPath.ACOUSTIC_ISOLATION,
                DuplexMode.FULL_DUPLEX,
                capture.clock_generation,  # type: ignore[attr-defined]
                True,
                None,
            ),
            NearEndDisposition.ADMIT if near_end_speech else None,
        )


class _RouteStampedAec(_HealthyAec):
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def metrics(self) -> dict[str, float]:
        return {**super().metrics(), "erle_db": 20.0 + self.reset_count}


class _PromptFailureThenHealthyAec(_HealthyAec):
    def __init__(self) -> None:
        self.fail_metrics = False
        self.failure_observed = False
        self.failure_reads = 0
        self.recovery_observed = False

    def metrics(self) -> dict[str, float]:
        if self.fail_metrics:
            self.failure_observed = True
            self.failure_reads += 1
            return {
                **super().metrics(),
                "delay_estimate_available": 0.0,
                "delay_estimate_refined": 1.0,
            }
        if self.failure_observed:
            self.recovery_observed = True
        return super().metrics()


class _PromptErleFailureThenHealthyAec(_HealthyAec):
    def __init__(self) -> None:
        self.fail_metrics = False
        self.failure_reads = 0
        self.recovery_observed = False

    def metrics(self) -> dict[str, float]:
        if self.fail_metrics:
            self.failure_reads += 1
            return {**super().metrics(), "erle_db": 101.0}
        if self.failure_reads:
            self.recovery_observed = True
        return super().metrics()


def _target_route() -> RouteIdentity:
    return RouteIdentity(
        input_index=1,
        output_index=2,
        input_name="PRIVATE USB microphone",
        output_name="PRIVATE USB headset",
        input_channels=1,
        output_channels=2,
        sample_rate_hz=48_000,
    )


def _alternate_route() -> RouteIdentity:
    return replace(
        _target_route(),
        input_index=3,
        output_index=4,
        input_name="PRIVATE alternate input",
        output_name="PRIVATE alternate output",
    )


def test_route_identity_survives_portaudio_index_renumbering() -> None:
    target = _target_route()
    renumbered_target = replace(target, input_index=41, output_index=42)
    reused_indices_for_alternate = replace(
        _alternate_route(),
        input_index=target.input_index,
        output_index=target.output_index,
    )

    assert renumbered_target.identity_key == target.identity_key
    assert reused_indices_for_alternate.identity_key != target.identity_key


def _short_config() -> LiveTrialConfig:
    return LiveTrialConfig(
        soak_seconds=0.24,
        route_round_trips=3,
        interruption_trials=1,
        double_talk_trials=1,
    )


async def _noop_prompt(_message: str) -> None:
    return None


class _LiveHarness:
    def __init__(self, *, aec: object) -> None:
        self.clock = ManualClock(now_ns=5_000_000_000)
        self.backend = FakeDuplexBackend(scripted_capture=self.capture_from_render)
        self.aec = aec
        self.route = _target_route()
        self.prompts: list[str] = []
        self.notices: list[str] = []
        self.speaking = False
        self.force_echo = False
        self.pump_count = 0
        self.wait_count = 0
        self.stall_waits = 0

    def transport(self) -> DuplexAudioTransport:
        return DuplexAudioTransport(
            backend=self.backend,
            clock=self.clock,
            capture_capacity=8,
            render_capacity=2,
        )

    def capture_from_render(self, delayed_render: bytes, index: int) -> bytes:
        if self.speaking:
            return _near_end_pcm(index)
        if self.force_echo:
            return delayed_render
        return SILENCE

    async def wait(self, _seconds: float) -> None:
        self.clock.advance(FRAME_DURATION_NS)
        self.wait_count += 1
        if (
            self.wait_count > self.stall_waits
            and self.backend.streams
            and self.backend.stream.started
        ):
            self.backend.stream.emit_scripted_capture(delay_frames=2)
            self.pump_count += 1
        await asyncio.sleep(0)

    async def prompt(self, message: str) -> None:
        self.prompts.append(message)
        lowered = message.casefold()
        if "alternate route" in lowered:
            self.route = _alternate_route()
        elif "target route" in lowered:
            self.route = _target_route()
        self.speaking = "speak" in lowered

    def notify(self, message: str) -> None:
        self.notices.append(message)
        lowered = message.casefold()
        if "stop speaking" in lowered:
            self.speaking = False
        elif "speak now" in lowered:
            self.speaking = True


def _runner(
    harness: _LiveHarness,
    *,
    config: LiveTrialConfig | None = None,
) -> PhysicalVoiceTrialRunner:
    return PhysicalVoiceTrialRunner(
        device_class="usb",
        config=config or _short_config(),
        transport_factory=harness.transport,
        route_reader=lambda: harness.route,
        prompt=harness.prompt,
        notify=harness.notify,
        aec_factory=lambda: harness.aec,
        isolation_factory=lambda: AcousticIsolationMonitor(
            window_frames=1,
            required_windows=1,
        ),
        clock=harness.clock,
        wait=harness.wait,
        identity_salt_factory=lambda _size: b"q" * 16,
    )


def _safe_half_report_with_runner_interruption(
    observations: dict[str, object],
) -> dict[str, object]:
    baseline = json.loads(
        (
            ROOT / "Tests/Packaging/fixtures/speculative_voice_physical/"
            "safe_half_duplex.json"
        ).read_text(encoding="utf-8")
    )
    for key in (
        "manual_interruption_available",
        "stop_latency_samples_ms",
        "unsuppressed_interruption_observed",
    ):
        baseline[key] = observations[key]
    return build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="bluetooth",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=baseline,
        salt=b"r" * 16,
    )


def test_runner_uses_backpressure_and_returns_content_free_measured_observations() -> (
    None
):
    harness = _LiveHarness(aec=_UnavailableAec())
    harness.stall_waits = 4

    observations = asyncio.run(_runner(harness).run())

    assert harness.backend.open_count == 7
    assert harness.backend.close_count == 7
    assert [call["device_pair"] for call in harness.backend.open_calls] == [
        (1, 2),
        (3, 4),
        (1, 2),
        (3, 4),
        (1, 2),
        (3, 4),
        (1, 2),
    ]
    assert harness.pump_count > 0
    assert observations["render_overflows"] == 0
    assert observations["reference_overflows"] == 0
    assert observations["control_overflows"] == 0
    assert observations["post_fence_callbacks"] == 0
    assert observations["safety_path"] == "acoustic-isolation"
    assert observations["correlation_samples"]
    assert observations["leakage_db_samples"]
    assert observations["double_talk_trials"] == 1
    assert len(observations["stop_latency_samples_ms"]) == 1
    assert len(observations["route_generation_trials"]) == 3
    for trial in observations["route_generation_trials"]:
        assert trial["alternate_generation"] == trial["start_generation"] + 1
        assert trial["return_generation"] == trial["alternate_generation"] + 1
        assert trial["alternate_admission_closed"] is True
        assert trial["return_admission_closed"] is True
        assert trial["returned_to_target"] is True
    assert (
        observations["device_identifier"]
        == hashlib.sha256(
            b"q" * 16 + b"1\x002\x00PRIVATE USB microphone\x00PRIVATE USB headset"
        ).hexdigest()
    )
    encoded = repr(observations)
    assert "PRIVATE" not in encoded
    assert "pcm" not in encoded.casefold()
    assert not any(_contains_bytes(value) for value in observations.values())
    expected_keys = set(
        json.loads(
            (
                ROOT / "Tests/Packaging/fixtures/speculative_voice_physical/"
                "safe_isolated_full_duplex.json"
            ).read_text(encoding="utf-8")
        )
    )
    assert set(observations) == expected_keys
    report = build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=observations,
        salt=b"r" * 16,
    )
    assert report["passed"] is False


def test_runner_binds_validated_route_if_defaults_change_before_stream_open() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    runner = _runner(harness)
    original_transport_factory = runner.transport_factory

    def change_default_before_open() -> DuplexAudioTransport:
        harness.route = _alternate_route()
        return original_transport_factory()

    runner.transport_factory = change_default_before_open

    asyncio.run(runner.run())

    assert harness.backend.open_calls[0]["device_pair"] == (1, 2)


def test_runner_reads_changed_routes_only_after_prior_stream_is_closed() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    runner = _runner(harness)
    close_counts: list[int] = []

    def read_route() -> RouteIdentity:
        close_counts.append(harness.backend.close_count)
        return harness.route

    runner.route_reader = read_route
    runner.route_refresh_reader = read_route

    asyncio.run(runner.run())

    assert close_counts == [0, 1, 2, 3, 4, 5, 6]


def test_runner_reports_an_injected_post_teardown_callback_as_valid_fail() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    runner = _runner(harness)
    injected = False

    async def wait_with_one_stale_callback(seconds: float) -> None:
        nonlocal injected
        if (
            not injected
            and seconds == 0
            and harness.backend.streams
            and harness.backend.stream.stopped
        ):
            harness.backend.stream.callback(
                SILENCE,
                SimpleNamespace(),
                SimpleNamespace(),
                False,
            )
            injected = True
        await harness.wait(seconds)

    runner.wait = wait_with_one_stale_callback

    observations = asyncio.run(runner.run())

    assert injected is True
    assert observations["post_fence_callbacks"] == 1
    report = build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=observations,
        salt=b"r" * 16,
    )
    assert report["passed"] is False


def test_runner_reports_native_teardown_failure_as_valid_fail_evidence() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    harness.backend.close_error = RuntimeError("injected native close failure")

    observations = asyncio.run(_runner(harness).run())

    assert observations["device_handle_leaks"] == harness.backend.open_count == 7
    assert all(stream.stop_count == 1 for stream in harness.backend.streams)
    assert all(stream.close_count == 1 for stream in harness.backend.streams)
    report = build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=observations,
        salt=b"r" * 16,
    )
    assert report["passed"] is False


def test_runner_measures_healthy_aec_path_from_processed_frames() -> None:
    harness = _LiveHarness(aec=_HealthyAec())
    harness.force_echo = True

    observations = asyncio.run(_runner(harness).run())

    assert observations["safety_path"] == "aec"
    assert observations["aec_health_path"] == "healthy"
    assert observations["aec_processor_operational"] is True
    assert observations["erle_samples_db"]
    assert observations["rendered_false_barge_events"] == 0
    assert observations["double_talk_detected"] == 1
    assert observations["stop_latency_samples_ms"] == [70.0]
    assert observations["manual_interruption_available"] is True
    assert observations["unsuppressed_interruption_observed"] is False


def test_runner_aborts_before_repeated_trials_when_speech_calibration_is_silent() -> (
    None
):
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(harness)

    async def remain_silent(message: str) -> None:
        harness.prompts.append(message)
        harness.speaking = False

    runner.prompt = remain_silent

    with pytest.raises(PhysicalVoiceRunnerError, match="speech calibration"):
        asyncio.run(runner.run())

    assert not any(message.startswith("Double-talk trial") for message in harness.prompts)
    assert not any(message.startswith("Interruption trial") for message in harness.prompts)
    assert not any(message.startswith("Route trial") for message in harness.prompts)
    assert harness.backend.open_count == harness.backend.close_count == 1


def test_runner_aborts_before_repeated_trials_when_overlap_calibration_fails() -> (
    None
):
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(harness)

    async def stop_speaking_when_overlap_starts(seconds: float) -> None:
        if runner._phase == "double-talk":  # noqa: SLF001
            harness.speaking = False
        await harness.wait(seconds)

    runner.wait = stop_speaking_when_overlap_starts

    with pytest.raises(PhysicalVoiceRunnerError, match="double-talk calibration"):
        asyncio.run(runner.run())

    assert not any(message.startswith("Double-talk trial") for message in harness.prompts)
    assert not any(message.startswith("Interruption trial") for message in harness.prompts)
    assert not any(message.startswith("Route trial") for message in harness.prompts)
    assert harness.backend.open_count == harness.backend.close_count == 1


def test_runner_collects_double_talk_windows_from_one_continuous_speech_prompt() -> (
    None
):
    harness = _LiveHarness(aec=_HealthyAec())
    observations = asyncio.run(
        _runner(
            harness,
            config=replace(_short_config(), double_talk_trials=3),
        ).run()
    )

    interruption_index = next(
        index
        for index, message in enumerate(harness.prompts)
        if message.startswith("Interruption trial")
    )
    assert harness.prompts[:interruption_index] == [harness.prompts[0]]
    assert "calibration" in harness.prompts[0].casefold()
    assert observations["double_talk_trials"] == 3
    assert observations["double_talk_detected"] == 3


def test_runner_reports_double_talk_calibration_and_window_results_immediately() -> (
    None
):
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(
        harness,
        config=replace(_short_config(), double_talk_trials=2),
    )
    notices: list[str] = []

    def record_notice(message: str) -> None:
        notices.append(message)
        harness.notify(message)

    runner.notify = record_notice  # type: ignore[attr-defined]

    asyncio.run(runner.run())

    assert len(notices) >= 4
    assert "speech calibration" in notices[0].casefold()
    assert "passed" in notices[0].casefold()
    assert "double-talk calibration" in notices[1].casefold()
    assert "passed" in notices[1].casefold()
    assert "window 1/2" in notices[2].casefold()
    assert "detected" in notices[2].casefold()
    assert "window 2/2" in notices[3].casefold()
    assert "detected" in notices[3].casefold()


def test_runner_keeps_playback_continuous_across_double_talk_windows() -> None:
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(
        harness,
        config=replace(_short_config(), double_talk_trials=3),
    )
    abort_phases: list[str] = []

    class CountingTransport(DuplexAudioTransport):
        def __init__(self) -> None:
            super().__init__(
                backend=harness.backend,
                clock=harness.clock,
                capture_capacity=8,
                render_capacity=2,
            )

        async def abort_output(self) -> None:
            abort_phases.append(runner._phase)  # noqa: SLF001
            await super().abort_output()

    runner.transport_factory = CountingTransport

    asyncio.run(runner.run())

    assert abort_phases.count("double-talk") == 1


def test_spoken_interruption_allows_a_realistic_human_response_window() -> None:
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(harness)
    interruption_ticks = 0

    async def prepare_delayed_interruption(message: str) -> None:
        await harness.prompt(message)
        if message.startswith("Interruption trial"):
            harness.speaking = False

    async def delayed_speech(seconds: float) -> None:
        nonlocal interruption_ticks
        if runner._phase == "interruption":  # noqa: SLF001
            interruption_ticks += 1
            harness.speaking = interruption_ticks >= 30
        await harness.wait(seconds)

    runner.prompt = prepare_delayed_interruption
    runner.wait = delayed_speech

    observations = asyncio.run(runner.run())

    assert interruption_ticks >= 30
    assert observations["stop_latency_samples_ms"] != [10_000.0]
    assert observations["unsuppressed_interruption_observed"] is False


def test_full_duplex_interruption_trials_use_one_prompt_and_report_each_result() -> (
    None
):
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(
        harness,
        config=replace(_short_config(), interruption_trials=3),
    )
    notices: list[str] = []

    def record_notice(message: str) -> None:
        notices.append(message)
        harness.notify(message)

    runner.notify = record_notice

    observations = asyncio.run(runner.run())

    interruption_prompts = [
        message
        for message in harness.prompts
        if message.startswith("Interruption trial")
    ]
    assert len(interruption_prompts) == 1
    assert not any(message == "Pause now for the next trial." for message in harness.prompts)
    interruption_notices = [
        message for message in notices if message.startswith("Interruption sample")
    ]
    assert len(interruption_notices) == 3
    assert all("passed" in message.casefold() for message in interruption_notices)
    assert len(observations["stop_latency_samples_ms"]) == 3


def test_runner_requires_confirmed_quiet_between_automated_interruption_cues() -> (
    None
):
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(
        harness,
        config=replace(_short_config(), interruption_trials=2),
    )
    recovery_runs: list[int] = []
    last_phase = ""

    async def track_recovery(seconds: float) -> None:
        nonlocal last_phase
        phase = runner._phase  # noqa: SLF001
        if phase == "operator-recovery":
            if last_phase != phase:
                recovery_runs.append(0)
            recovery_runs[-1] += 1
        last_phase = phase
        await harness.wait(seconds)

    runner.wait = track_recovery

    asyncio.run(runner.run())

    assert len(recovery_runs) == 3
    assert all(ticks >= 5 for ticks in recovery_runs)


def test_quiet_recovery_rejects_unverified_capture_processing() -> None:
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(harness)
    process_available = runner._process_available  # noqa: SLF001

    async def invalidate_recovery_capture(**kwargs: object) -> None:
        await process_available(**kwargs)  # type: ignore[arg-type]
        if runner._phase == "operator-recovery":  # noqa: SLF001
            runner._capture_processing_ok = False  # type: ignore[attr-defined]  # noqa: SLF001

    runner._process_available = invalidate_recovery_capture  # type: ignore[method-assign]  # noqa: SLF001

    with pytest.raises(PhysicalVoiceRunnerError, match="verify capture processing"):
        asyncio.run(runner.run())

    assert not any(
        message.startswith("Interruption trial") for message in harness.prompts
    )
    assert harness.backend.open_count == harness.backend.close_count == 1


def test_twenty_trial_flow_is_calibrated_once_and_automatically_paced() -> None:
    harness = _LiveHarness(aec=_HealthyAec())
    config = replace(
        _short_config(),
        route_round_trips=1,
        double_talk_trials=20,
        interruption_trials=20,
    )
    runner = _runner(harness, config=config)
    runner.isolation_factory = _AlwaysOpenIsolation

    observations = asyncio.run(runner.run())

    calibration_prompts = [
        message
        for message in harness.prompts
        if message.startswith("Double-talk calibration")
    ]
    interruption_prompts = [
        message
        for message in harness.prompts
        if message.startswith("Interruption trial")
    ]
    window_notices = [
        message
        for message in harness.notices
        if message.startswith("Double-talk window")
    ]
    sample_notices = [
        message
        for message in harness.notices
        if message.startswith("Interruption sample")
    ]
    quiet_notices = [
        message
        for message in harness.notices
        if message == "Microphone silence confirmed."
    ]

    assert len(calibration_prompts) == 1
    assert len(interruption_prompts) == 1
    assert len(window_notices) == 20
    assert all("detected" in message.casefold() for message in window_notices)
    assert len(sample_notices) == 20
    failed_samples = [
        message for message in sample_notices if "passed" not in message.casefold()
    ]
    assert not failed_samples, failed_samples
    assert len(quiet_notices) == 21
    assert observations["double_talk_trials"] == 20
    assert observations["double_talk_detected"] == 20
    assert len(observations["stop_latency_samples_ms"]) == 20


def test_automated_interruption_flow_stops_after_first_failed_sample() -> None:
    harness = _LiveHarness(aec=_HealthyAec())
    runner = _runner(
        harness,
        config=replace(_short_config(), interruption_trials=20),
    )
    runner.isolation_factory = _AlwaysOpenIsolation

    class FirstInterruptionAbortFails(DuplexAudioTransport):
        def __init__(self) -> None:
            super().__init__(
                backend=harness.backend,
                clock=harness.clock,
                capture_capacity=8,
                render_capacity=2,
            )
            self.abort_calls = 0

        async def abort_output(self) -> None:
            self.abort_calls += 1
            if self.abort_calls == 3:
                raise RuntimeError("injected first interruption abort failure")
            await super().abort_output()

    runner.transport_factory = FirstInterruptionAbortFails

    observations = asyncio.run(runner.run())

    sample_notices = [
        message
        for message in harness.notices
        if message.startswith("Interruption sample")
    ]
    assert sample_notices == ["Interruption sample 1/20: failed (10000.0 ms)."]
    assert "Qualification stopped" in harness.notices[-1]
    assert observations["stop_latency_samples_ms"] == [10_000.0]
    assert observations["operator_checklist"]["interruption_completed"] is False
    assert not any(message.startswith("Route trial") for message in harness.prompts)
    assert harness.backend.open_count == harness.backend.close_count == 1
    report = build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=observations,
        salt=b"r" * 16,
    )
    assert report["passed"] is False
    assert report["trials"]["interruption"]["trials"] == 1


def test_prompt_time_native_failure_is_latched_after_healthy_recovery() -> None:
    aec = _PromptFailureThenHealthyAec()
    harness = _LiveHarness(aec=aec)
    harness.force_echo = True
    runner = _runner(harness)

    async def fail_one_prompt_capture(message: str) -> None:
        await harness.prompt(message)
        if not message.startswith("Double-talk calibration"):
            return
        aec.fail_metrics = True
        while aec.failure_reads < 2:
            await asyncio.sleep(0)
        aec.fail_metrics = False

    runner.prompt = fail_one_prompt_capture

    observations = asyncio.run(runner.run())

    assert aec.failure_observed is True
    assert aec.recovery_observed is True
    assert observations["aec_processor_operational"] is False
    assert observations["safety_path"] == "half-duplex"
    assert observations["playback_speech_admission_closed"] is False
    assert (
        observations["demotion_reason_counts"][
            AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE.value
        ]
        == 1
    )


@pytest.mark.parametrize(
    ("safety", "processor_operational", "expected_admission_closed"),
    [
        (
            AcousticSafetySnapshot(
                AcousticSafetyPath.WARMING,
                DuplexMode.HALF_DUPLEX,
                0,
                False,
                None,
            ),
            True,
            True,
        ),
        (
            AcousticSafetySnapshot(
                AcousticSafetyPath.AEC,
                DuplexMode.FULL_DUPLEX,
                0,
                True,
                None,
            ),
            False,
            False,
        ),
    ],
)
def test_observation_admission_uses_actual_safety_snapshot(
    safety: AcousticSafetySnapshot,
    processor_operational: bool,
    expected_admission_closed: bool,
) -> None:
    class SnapshotPreprocessor:
        health = AecHealth.WARMING

        def __init__(self, snapshot: AcousticSafetySnapshot) -> None:
            self.safety = snapshot

    runner = PhysicalVoiceTrialRunner(device_class="usb")
    runner._transport = DuplexAudioTransport(
        backend=FakeDuplexBackend(),
        clock=ManualClock(),
    )
    runner._preprocessor = SnapshotPreprocessor(safety)  # type: ignore[assignment]
    runner._aec = _HealthyAec()
    runner._processor_operational = processor_operational

    observations = runner._build_observations(
        target_route=_target_route(),
        device_identifier="d" * 64,
        operator_checklist={},
    )

    assert observations["safety_path"] == AcousticSafetyPath.HALF_DUPLEX.value
    assert observations["playback_speech_admission_closed"] is expected_admission_closed


def test_prompt_time_out_of_range_erle_failure_is_latched_after_recovery() -> None:
    aec = _PromptErleFailureThenHealthyAec()
    harness = _LiveHarness(aec=aec)
    harness.force_echo = True
    runner = _runner(harness)

    async def fail_one_prompt_capture(message: str) -> None:
        await harness.prompt(message)
        if not message.startswith("Double-talk calibration"):
            return
        aec.fail_metrics = True
        while aec.failure_reads < 2:
            await asyncio.sleep(0)
        aec.fail_metrics = False

    runner.prompt = fail_one_prompt_capture

    observations = asyncio.run(runner.run())

    assert aec.recovery_observed is True
    assert observations["aec_processor_operational"] is False
    assert (
        observations["demotion_reason_counts"][
            AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE.value
        ]
        == 1
    )
    report = build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=observations,
        salt=b"r" * 16,
    )
    assert report["passed"] is False


@pytest.mark.parametrize(("dsp_ok", "vad_ok"), [(False, True), (True, False)])
def test_failed_processing_acknowledgement_latches_run_failure(
    dsp_ok: bool,
    vad_ok: bool,
) -> None:
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    runner = PhysicalVoiceTrialRunner(device_class="usb")
    runner._transport = transport

    async def acknowledge_failure() -> None:
        await transport.start()
        backend.stream.emit_capture(SILENCE)
        capture = transport.pop_capture()
        assert capture is not None
        runner._acknowledge_processed(
            capture.sequence,
            capture.clock_generation,
            dsp_ok,
            vad_ok,
        )
        await transport.close()

    asyncio.run(acknowledge_failure())

    assert runner._processor_operational is False


@pytest.mark.parametrize(
    "invalid_metrics",
    [
        {
            **_HealthyAec().metrics(),
            "delay_estimate_available": 0.0,
            "delay_estimate_refined": 1.0,
        },
        {**_HealthyAec().metrics(), "erle_db": 101.0},
        {**_HealthyAec().metrics(), "erle_db": True},
        {**_HealthyAec().metrics(), "delay_ms": float("inf")},
        {
            key: value
            for key, value in _HealthyAec().metrics().items()
            if key != "clock_drift"
        },
    ],
)
def test_invalid_final_aec_metrics_return_valid_fail_observations(
    invalid_metrics: dict[str, float],
) -> None:
    class InvalidFinalAec(_HealthyAec):
        def metrics(self) -> dict[str, float]:
            return invalid_metrics

    class PassivePreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.HALF_DUPLEX,
            admission_open=False,
            demotion_reason=AcousticDemotionReason.NATIVE_PROCESSOR_FAILURE,
        )
        health = AecHealth.DEGRADED

    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    runner = PhysicalVoiceTrialRunner(device_class="usb")
    runner._transport = transport
    runner._preprocessor = PassivePreprocessor()  # type: ignore[assignment]
    runner._aec = InvalidFinalAec()
    runner._erle_samples = [30.0]

    observations = runner._build_observations(
        target_route=_target_route(),
        device_identifier="d" * 64,
        operator_checklist={
            "device_switch_completed": False,
            "double_talk_completed": False,
            "interruption_completed": False,
            "rendered_speech_completed": False,
            "silence_completed": False,
            "soak_completed": False,
        },
    )

    assert observations["aec_processor_operational"] is False
    assert observations["aec_delay_estimate_available"] is False
    assert observations["aec_delay_estimate_refined"] is False
    assert observations["erle_samples_db"] == []
    report = build_physical_report(
        evidence_kind="physical",
        source_tree_digest="d" * 64,
        platform_key="macos-arm64",
        device_class="usb",
        app_version="0.1.8.0",
        prerequisites=AutomatedPrerequisiteEvidence(
            source_tree_digest="d" * 64,
            companion_version="0.1.8.0",
            upstream_commit="1" * 40,
            report_hashes={
                "automated_report_sha256": "2" * 64,
                "corpus_section_sha256": "3" * 64,
                "latency_section_sha256": "4" * 64,
                "duplex_soak_report_sha256": "5" * 64,
                "cancellation_soak_report_sha256": "6" * 64,
            },
        ),
        observations=observations,
        salt=b"r" * 16,
    )
    assert report["passed"] is False


def test_safe_half_duplex_measures_manual_stop_without_acoustic_failure() -> None:
    harness = _LiveHarness(aec=None)
    runner = _runner(harness)

    async def high_output_latency_wait(_seconds: float) -> None:
        harness.clock.advance(FRAME_DURATION_NS)
        if harness.backend.streams and harness.backend.stream.started:
            frame_index = harness.backend.stream.frame_index
            current_time = (frame_index + 1) * 0.01
            harness.backend.stream.emit_capture(
                _near_end_pcm(frame_index) if harness.speaking else SILENCE,
                input_adc_time=current_time - 0.01,
                current_time=current_time,
                output_dac_time=current_time + 0.21,
            )
        await asyncio.sleep(0)

    runner.wait = high_output_latency_wait

    observations = asyncio.run(runner.run())

    assert observations["safety_path"] == "half-duplex"
    assert observations["double_talk_detected"] == 0
    assert observations["manual_interruption_available"] is True
    assert observations["unsuppressed_interruption_observed"] is False
    assert observations["stop_latency_samples_ms"][0] > 150.0
    assert any("manual stop" in message.casefold() for message in harness.prompts)
    assert any("stop acknowledged" in message.casefold() for message in harness.notices)
    assert not any("press Enter now" in message for message in harness.notices)
    assert _safe_half_report_with_runner_interruption(observations)["passed"] is True


def test_manual_interruption_fallback_stays_latched_if_path_reopens() -> None:
    harness = _LiveHarness(aec=None)
    runner = _runner(harness)

    async def open_isolation_during_prompt(message: str) -> None:
        await harness.prompt(message)
        if message.startswith("Interruption trial"):
            await runner._process_available(  # noqa: SLF001
                rendering=True,
                measure=False,
            )
            runner._require_preprocessor()._publish_safety(  # noqa: SLF001
                AcousticSafetyPath.ACOUSTIC_ISOLATION
            )

    runner.prompt = open_isolation_during_prompt

    observations = asyncio.run(runner.run())

    interruption_prompts = [
        message
        for message in harness.prompts
        if message.startswith("Interruption trial")
    ]
    assert len(interruption_prompts) == 1
    assert "manual stop" in interruption_prompts[0]
    assert observations["manual_interruption_available"] is True
    assert observations["stop_latency_samples_ms"][0] < 10_000.0
    assert observations["unsuppressed_interruption_observed"] is False
    assert observations["operator_checklist"]["interruption_completed"] is True
    assert harness.backend.close_count == harness.backend.open_count == 7


def test_interruption_prompt_retries_once_then_latches_manual_protocol() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    runner = _runner(harness)

    async def flap_protocol_during_prompt(message: str) -> None:
        await harness.prompt(message)
        if not message.startswith("Interruption trial"):
            return
        await runner._process_available(  # noqa: SLF001
            rendering=True,
            measure=False,
        )
        next_path = (
            AcousticSafetyPath.ACOUSTIC_ISOLATION
            if "manual stop" in message
            else AcousticSafetyPath.HALF_DUPLEX
        )
        runner._require_preprocessor()._publish_safety(next_path)  # noqa: SLF001

    runner.prompt = flap_protocol_during_prompt

    observations = asyncio.run(runner.run())

    interruption_prompts = [
        message
        for message in harness.prompts
        if message.startswith("Interruption trial")
    ]
    assert len(interruption_prompts) == 2
    assert ["manual stop" in message for message in interruption_prompts] == [
        False,
        True,
    ]
    assert observations["manual_interruption_available"] is True
    assert observations["operator_checklist"]["interruption_completed"] is True
    assert harness.backend.close_count == harness.backend.open_count == 7
    assert runner.retained_frame_count == 0
    assert runner._prompt_task is None
    assert runner._transport is None
    assert runner._preprocessor is None


def test_full_duplex_acoustic_failure_keeps_manual_proof_bounded() -> None:
    harness = _LiveHarness(aec=_MuteHealthyAec())
    config = replace(_short_config(), interruption_trials=20)

    with pytest.raises(PhysicalVoiceRunnerError, match="speech calibration"):
        asyncio.run(_runner(harness, config=config).run())

    assert not any(
        message.startswith("Interruption trial") for message in harness.prompts
    )
    assert not any(message.startswith("Route trial") for message in harness.prompts)
    assert harness.backend.open_count == harness.backend.close_count == 1


def test_interruption_collection_uses_acknowledged_acoustic_protocol() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    runner = PhysicalVoiceTrialRunner(
        device_class="usb",
        config=_short_config(),
        clock=clock,
    )

    class PassiveHalfDuplexPreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.HALF_DUPLEX,
            demotion_reason=None,
        )

        async def process_capture(self, *_args: object, **_kwargs: object) -> None:
            return None

    async def callback(_seconds: float) -> None:
        clock.advance(FRAME_DURATION_NS)
        backend.stream.emit_capture(SILENCE)
        await asyncio.sleep(0)

    runner._transport = transport
    runner._preprocessor = PassiveHalfDuplexPreprocessor()  # type: ignore[assignment]
    runner._reference_frames = (SILENCE,)
    runner.wait = callback

    async def collect() -> None:
        await transport.start()
        await runner._collect_interruption_sample(manual_stop=False)
        await transport.close()

    asyncio.run(collect())

    assert runner._unsuppressed_interruption_observed is True
    assert runner._stop_latency_samples_ms == [10_000.0]


def test_abort_failure_returns_valid_failing_manual_stop_evidence() -> None:
    harness = _LiveHarness(aec=None)

    class AbortFailingTransport(DuplexAudioTransport):
        def __init__(self) -> None:
            super().__init__(
                backend=harness.backend,
                clock=harness.clock,
                capture_capacity=8,
                render_capacity=2,
            )
            self.abort_calls = 0

        async def abort_output(self) -> None:
            self.abort_calls += 1
            if self.abort_calls == 3:
                raise RuntimeError("injected interruption abort failure")
            await super().abort_output()

    runner = _runner(harness)
    runner.transport_factory = AbortFailingTransport

    observations = asyncio.run(runner.run())

    assert observations["manual_interruption_available"] is False
    assert observations["unsuppressed_interruption_observed"] is False
    assert observations["stop_latency_samples_ms"] == [10_000.0]
    assert _safe_half_report_with_runner_interruption(observations)["passed"] is False


def test_manual_stop_uses_first_callback_boundary_after_abort() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    runner = PhysicalVoiceTrialRunner(device_class="usb", clock=clock)

    class PassivePreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.AEC,
            demotion_reason=None,
        )

        async def process_capture(self, *_args: object, **_kwargs: object) -> None:
            return None

    async def two_callbacks(_seconds: float) -> None:
        clock.advance(FRAME_DURATION_NS)
        backend.stream.emit_capture(SILENCE)
        backend.stream.emit_capture(SILENCE)

    runner._transport = transport
    runner._preprocessor = PassivePreprocessor()  # type: ignore[assignment]
    runner.wait = two_callbacks

    async def measure() -> float | None:
        await transport.start()
        assert transport.queue_render(SILENCE) is not None
        backend.stream.emit_capture(SILENCE)
        await runner._process_available(rendering=True, measure=False)
        assert runner._last_processed_capture is not None
        last_reference = runner._last_processed_capture.render_reference_sequence
        result = await runner._abort_and_measure_silence_boundary(
            trigger_ns=clock(),
            last_reference_sequence=last_reference,
        )
        await transport.close()
        return result

    assert asyncio.run(measure()) == 20.0


def test_manual_stop_advances_baseline_across_queued_preabort_callback() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    runner = PhysicalVoiceTrialRunner(device_class="usb", clock=clock)

    class PassivePreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.AEC,
            demotion_reason=None,
        )

        async def process_capture(self, *_args: object, **_kwargs: object) -> None:
            return None

    emitted_postabort = False

    async def postabort_callback(_seconds: float) -> None:
        nonlocal emitted_postabort
        clock.advance(FRAME_DURATION_NS)
        if not emitted_postabort:
            backend.stream.emit_capture(SILENCE)
            emitted_postabort = True
        await asyncio.sleep(0)

    runner._transport = transport
    runner._preprocessor = PassivePreprocessor()  # type: ignore[assignment]
    runner.wait = postabort_callback

    async def measure() -> float | None:
        await transport.start()
        assert transport.queue_render(SILENCE) is not None
        backend.stream.emit_capture(SILENCE)
        await runner._process_available(rendering=True, measure=False)
        assert runner._last_processed_capture is not None
        assert runner._last_processed_capture.render_reference_sequence == 0
        last_reference = runner._last_processed_capture.render_reference_sequence

        assert transport.queue_render(SILENCE) is not None
        backend.stream.emit_capture(SILENCE)
        clock.advance(FRAME_DURATION_NS)

        result = await runner._abort_and_measure_silence_boundary(
            trigger_ns=clock(),
            last_reference_sequence=last_reference,
        )
        await transport.close()
        return result

    assert asyncio.run(measure()) == 20.0


def test_manual_stop_rejects_discontinuous_callback_timing() -> None:
    clock = ManualClock()
    backend = FakeDuplexBackend()
    transport = DuplexAudioTransport(backend=backend, clock=clock)
    runner = PhysicalVoiceTrialRunner(device_class="usb", clock=clock)

    class PassivePreprocessor:
        safety = SimpleNamespace(
            path=AcousticSafetyPath.HALF_DUPLEX,
            demotion_reason=None,
        )

        async def process_capture(self, *_args: object, **_kwargs: object) -> None:
            return None

    async def discontinuous_callback(_seconds: float) -> None:
        clock.advance(FRAME_DURATION_NS)
        backend.stream.emit_capture(SILENCE, discontinuity=True)

    runner._transport = transport
    runner._preprocessor = PassivePreprocessor()  # type: ignore[assignment]
    runner.wait = discontinuous_callback

    async def measure() -> float | None:
        await transport.start()
        assert transport.queue_render(SILENCE) is not None
        backend.stream.emit_capture(SILENCE)
        await runner._process_available(rendering=True, measure=False)
        assert runner._last_processed_capture is not None
        last_reference = runner._last_processed_capture.render_reference_sequence
        result = await runner._abort_and_measure_silence_boundary(
            trigger_ns=clock(),
            last_reference_sequence=last_reference,
        )
        await transport.close()
        return result

    assert asyncio.run(measure()) is None
    assert runner._manual_interruption_available is False


def test_route_fences_discard_prior_generation_acoustic_samples() -> None:
    harness = _LiveHarness(aec=_RouteStampedAec())

    observations = asyncio.run(_runner(harness).run())

    assert observations["erle_samples_db"] == [26.0]


def test_prompt_waits_are_serviced_and_excluded_from_active_soak() -> None:
    harness = _LiveHarness(aec=None)

    async def slow_prompt(message: str) -> None:
        for _ in range(100):
            if harness.backend.streams and harness.backend.stream.started:
                harness.backend.stream.emit_scripted_capture(delay_frames=2)
            harness.clock.advance(FRAME_DURATION_NS)
            await asyncio.sleep(0)
        await harness.prompt(message)

    runner = _runner(harness)
    runner.prompt = slow_prompt

    observations = asyncio.run(runner.run())

    assert observations["capture_overflows"] == 0
    assert observations["soak_minutes"] < 0.02
    assert observations["silence_minutes"] < 0.02
    assert observations["unbounded_task_growth"] is False


def test_route_acknowledgement_without_changed_identity_fails() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    harness.speaking = True
    harness.prompt = _noop_prompt  # type: ignore[method-assign]

    with pytest.raises(PhysicalVoiceRunnerError, match="route identity did not change"):
        asyncio.run(_runner(harness).run())
    assert harness.backend.open_count == 1
    assert harness.backend.close_count == 1


def test_acoustic_threshold_failure_returns_valid_fail_observations() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    harness.force_echo = True

    observations = asyncio.run(_runner(harness).run())

    assert observations["safety_path"] == "half-duplex"
    assert observations["aec_processor_operational"] is True
    assert observations["demotion_reason_counts"]


@pytest.mark.parametrize("failure", ["permission", "open", "format"])
def test_device_admission_failures_raise_without_observations(failure: str) -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    if failure == "permission":
        harness.route = replace(_target_route(), input_channels=0)
    elif failure == "open":
        harness.backend.start_failures = 1
    else:
        harness.route = replace(_target_route(), sample_rate_hz=44_100)

    with pytest.raises((PhysicalVoiceRunnerError, RuntimeError, ValueError)):
        asyncio.run(_runner(harness).run())
    assert harness.backend.close_count == harness.backend.open_count


def test_cancellation_closes_stream_and_releases_runner_buffers() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())
    blocker = asyncio.Event()

    async def wait_forever(_seconds: float) -> None:
        await blocker.wait()

    runner = _runner(harness, config=replace(_short_config(), soak_seconds=10.0))
    runner.wait = wait_forever

    async def cancel() -> None:
        task = asyncio.create_task(runner.run())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())

    assert harness.backend.close_count == harness.backend.open_count == 1
    assert runner.retained_frame_count == 0
    assert runner._transport is None
    assert runner._preprocessor is None
    assert runner._monitor is None
    assert runner._aec is None
    assert runner._mutable_scratch == bytearray()


def test_cancellation_during_prompt_releases_the_tracked_prompt_task() -> None:
    harness = _LiveHarness(aec=_UnavailableAec())

    async def cancel_during_prompt() -> PhysicalVoiceTrialRunner:
        entered = asyncio.Event()
        blocker = asyncio.Event()

        async def blocking_prompt(_message: str) -> None:
            entered.set()
            await blocker.wait()

        runner = _runner(harness)
        runner.prompt = blocking_prompt
        task = asyncio.create_task(runner.run())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return runner

    runner = asyncio.run(cancel_during_prompt())

    assert harness.backend.close_count == harness.backend.open_count == 1
    assert runner._prompt_task is None
    assert runner.retained_frame_count == 0


def test_fake_backend_scripts_capture_from_prior_render_callback_output() -> None:
    history: deque[bytes] = deque()
    backend = FakeDuplexBackend(
        scripted_capture=lambda rendered, _index: rendered,
    )
    transport = DuplexAudioTransport(backend=backend, clock=ManualClock())
    asyncio.run(transport.start())
    first = b"\x01\x00" * (FRAME_BYTES // 2)
    second = b"\x02\x00" * (FRAME_BYTES // 2)
    transport.queue_render(first)
    transport.queue_render(second)

    history.append(backend.stream.emit_scripted_capture(delay_frames=1))
    history.append(backend.stream.emit_scripted_capture(delay_frames=1))
    backend.stream.emit_scripted_capture(delay_frames=1)

    captures = [
        transport.pop_capture(),
        transport.pop_capture(),
        transport.pop_capture(),
    ]
    assert [frame.pcm16 for frame in captures if frame is not None] == [
        SILENCE,
        first,
        second,
    ]
    assert list(history) == [first, second]


def _contains_bytes(value: object) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, dict):
        return any(_contains_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_bytes(item) for item in value)
    return False
