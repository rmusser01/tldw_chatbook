# Low-Latency Speculative Duplex Voice Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start provider-agnostic hands-free replies from a fresh rolling transcript after 700 ms of silence by default, cancel and replace provisional replies whenever the same user turn continues, and make speaker-safe acoustic barge-in the default through a qualified cross-platform AEC path.

**Architecture:** Add an import-safe duplex audio/DSP boundary, a rolling transcript protocol, and a serialized speculative-turn coordinator beside the existing legacy hands-free controller. Provisional provider and TTS work stays attempt-local; effectful turns cross a two-second barrier into the ordinary accepted-turn path, and only a winning no-tool attempt may use the existing ADR-094/ADR-097 promotion services. The new pipeline remains behind a hard-disabled qualification gate until native wheels, persistence/trace prerequisites, and the physical-device matrix pass.

**Tech Stack:** Python 3.11+, asyncio, Textual 8.x, sounddevice/PortAudio, WebRTC Audio Processing Module AEC3, pybind11, scikit-build-core, CMake, pytest, Hypothesis, GitHub Actions/cibuildwheel.

---

## Governing records

- Design: `Docs/superpowers/specs/2026-08-28-low-latency-speculative-duplex-voice-pipeline-design.md`
- Decision: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`
- Related boundaries: ADR-023, ADR-039, ADR-094, and ADR-097.

ADR required: yes

ADR path: `backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md`

Reason: native audio/runtime packaging, full-duplex clock ownership, AEC failure policy, provider cancellation, persistence/capture promotion, and tool-effect boundaries are cross-module architectural decisions.

## Preconditions and rollout order

The implementation may begin at Task 1, but the speculative pipeline must remain hard-off in production until all of these conditions are true:

1. ADR-094 terminal-receipt work, currently represented by `TASK-22514`, exposes one transaction that promotes the winning user/assistant pair, creates the terminal receipt, and records the exact unseen mark. Task 9 adapts to that service; it must not recreate the transaction.
2. ADR-097 exposes reservation/settlement and temporary-chat trace behavior. Task 9 uses the trace service for winning-only `provisional_voice_promoted` capture; it must not add a parallel trace store.
3. macOS, Windows, and Linux native wheels and the AEC corpus pass in CI.
4. The latency, deterministic integration, soak, and physical-device qualification in Task 12 pass. Bluetooth may qualify only through explicit safe half duplex.

Until then, `speculative_voice_qualified()` returns `False` in every installed build and `ConsoleHandsFreeController` continues to select `HandsFreeController`. There is no environment-variable, command-line, or saved-setting bypass. Tests exercise the new path by injecting a fake qualification reader into the composition root; local manual development requires building a real capability manifest with the Task 12C generator from qualified local evidence.

Do not create Backlog tasks from this plan unless the user explicitly requests it. When implementation is authorized, create atomic Backlog tasks in this dependency order and link each task to the design and ADR before changing code.

### Dirty-worktree protocol for every task

This plan was written in a shared dirty worktree. Before each task, run `git status --short -- <owned paths>` and `git diff -- <owned tracked paths>`. If an owned tracked file already has changes, preserve them, determine their owner, and either coordinate the overlap or use patch-level staging; never overwrite, revert, or silently absorb them.

Every commit step below uses this safe pattern:

```bash
git status --short -- <task paths>
git diff -- <tracked task paths>
git add --intent-to-add <new task paths>
git add -p -- <all task paths>
git diff --cached --check
git diff --cached -- <all task paths>
git commit -m "<task message>"
```

Select only hunks created for the current task. If patch staging cannot separate an overlap safely, stop and ask the user; do not use whole-file `git add` on that path. The abbreviated commit blocks below name the files and message, but this protocol remains mandatory.

## Target file structure

| Path | Responsibility |
| --- | --- |
| `tldw_chatbook/Chat/console_voice_settings.py` | Bounded new-pipeline settings, presets, qualification gate, and compatibility ownership. |
| `tldw_chatbook/Audio/duplex_contracts.py` | Dependency-free frame, watermark, health, and transport protocols. |
| `tldw_chatbook/Audio/aec_backend.py` | Lazy native AEC loader and narrow processor protocol. |
| `tldw_chatbook/Audio/duplex_transport.py` | One app-owned input/output session, timestamped rings, device lifecycle. |
| `tldw_chatbook/Audio/voice_preprocessor.py` | Resampling, AEC, health hysteresis, post-AEC VAD, and admission. |
| `tldw_chatbook/Audio/rolling_transcript.py` | Common live/rolling-window STT revision and reconciliation contract. |
| `native/voice_aec/` | Reproducible pybind11/CMake WebRTC AEC3 companion package. |
| `Packaging/generate_voice_qualification_manifest.py` | Validates evidence and emits the packaged, platform-keyed rollout authority. |
| `tldw_chatbook/Audio/voice_qualification_manifest.json` | Generated, immutable capability evidence consumed fail-closed at runtime. |
| `tldw_chatbook/Chat/voice_phrase_sequencer.py` | Markdown-safe, sequential, cancellable phrase synthesis/playback. |
| `tldw_chatbook/Chat/console_voice_attempts.py` | Attempt-local generation, output, usage/capture, cancellation, and cleanup. |
| `tldw_chatbook/Chat/console_voice_supervisor.py` | App-lifetime orphan set and voice-dispatch quarantine. |
| `tldw_chatbook/Chat/console_speculative_voice.py` | Serialized logical-turn coordinator and effect barrier state machine. |
| `tldw_chatbook/Chat/console_voice_promotion.py` | Narrow adapters to ADR-094 terminalization and ADR-097 trace promotion. |
| `tldw_chatbook/UI/Console_Modules/hands_free.py` | Legacy/new selection and view-scoped lifecycle wiring. |
| `tldw_chatbook/Widgets/Console/console_voice_preview.py` | Ephemeral user/assistant projection outside the durable message store. |
| `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py` | Canonical settings UI for response eagerness, STT mode, and AEC troubleshooting. |
| `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_panel_types.py` | Separate pipeline settings draft. |
| `tldw_chatbook/Audio/voice_metrics.py` | Content-free latency, mode, restart, AEC, and discarded-usage measurements. |

Keep `Chat/console_hands_free.py`, `Chat/reply_sentence_sequencer.py`, `dictation.acoustic_barge_in`, and `dictation.handsfree_send_delay_seconds` unchanged for the legacy/Realtime compatibility path. Do not add settings to `UI/Tools_Settings_Window.py` or `Widgets/enhanced_settings_sidebar.py`.

## Task 1: Add bounded settings and import-safe contracts

**Files:**

- Create: `tldw_chatbook/Chat/console_voice_settings.py`
- Create: `tldw_chatbook/Audio/duplex_contracts.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_panel_types.py`
- Test: `Tests/Chat/test_console_voice_settings.py`
- Test: `Tests/Audio/test_duplex_contracts.py`

- [ ] **Step 1: Write the failing settings and contract tests.**

```python
from tldw_chatbook.Chat.console_voice_settings import (
    RESPONSE_EAGERNESS_MAX_MS,
    RESPONSE_EAGERNESS_MIN_MS,
    response_eagerness_ms,
    speculative_voice_qualified,
)


def test_response_eagerness_defaults_and_invalid_values_fall_back() -> None:
    assert response_eagerness_ms({}) == 700
    warnings: list[str] = []
    assert response_eagerness_ms(
        {"dictation": {"response_eagerness_ms": 1}}, warn=warnings.append
    ) == 700
    assert warnings == ["Invalid speculative voice response eagerness; using 700 ms."]
    assert response_eagerness_ms(
        {"dictation": {"response_eagerness_ms": 9999}}, warn=warnings.append
    ) == 700


def test_qualification_is_hard_off_even_when_an_environment_variable_is_set(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_DEV_SPECULATIVE_VOICE", "1")
    assert speculative_voice_qualified() is False
```

```python
from tldw_chatbook.Audio.duplex_contracts import AudioFrame, AecHealth


def test_audio_frame_requires_positive_ordered_duration() -> None:
    frame = AudioFrame(sequence=1, started_ns=10, ended_ns=20, pcm16=b"\x00\x00")
    assert frame.duration_ns == 10
    assert AecHealth.WARMING.value == "warming"
```

- [ ] **Step 2: Run the tests and verify RED.**

Run: `pytest -q Tests/Chat/test_console_voice_settings.py Tests/Audio/test_duplex_contracts.py`

Expected: collection fails because the two new modules do not exist.

- [ ] **Step 3: Implement the smallest pure settings and contract surface.**

```python
# tldw_chatbook/Chat/console_voice_settings.py
from __future__ import annotations

import logging
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

RESPONSE_EAGERNESS_MIN_MS = 500
RESPONSE_EAGERNESS_MAX_MS = 3000
RESPONSE_EAGERNESS_DEFAULT_MS = 700
RESPONSE_EAGERNESS_PRESETS = {"fast": 700, "balanced": 1200, "deliberate": 2000}
_LOG = logging.getLogger(__name__)


def response_eagerness_ms(
    config: Mapping[str, Any],
    *,
    warn: Callable[[str], None] = _LOG.warning,
) -> int:
    raw = config.get("dictation", {}).get("response_eagerness_ms", RESPONSE_EAGERNESS_DEFAULT_MS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = -1
    if not RESPONSE_EAGERNESS_MIN_MS <= value <= RESPONSE_EAGERNESS_MAX_MS:
        warn("Invalid speculative voice response eagerness; using 700 ms.")
        return RESPONSE_EAGERNESS_DEFAULT_MS
    return value


def pipeline_aec_enabled(config: Mapping[str, Any]) -> bool:
    return config.get("dictation", {}).get("pipeline_aec_enabled", True) is not False


def speculative_voice_qualified() -> bool:
    # Task 12C replaces this with a validated packaged-manifest lookup.
    return False
```

```python
# tldw_chatbook/Audio/duplex_contracts.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class AecHealth(str, Enum):
    WARMING = "warming"
    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class AudioFrame:
    sequence: int
    started_ns: int
    ended_ns: int
    pcm16: bytes

    def __post_init__(self) -> None:
        if self.sequence < 0 or self.ended_ns <= self.started_ns:
            raise ValueError("audio frames require an ordered sequence and positive duration")

    @property
    def duration_ns(self) -> int:
        return self.ended_ns - self.started_ns


@dataclass(frozen=True, slots=True)
class DrainReceipt:
    capture_watermark_ns: int
    capture_sequence: int
    dsp_sequence: int
    vad_sequence: int


class DuplexAudioPort(Protocol):
    async def abort_playback(self) -> None: ...
    async def drain_capture_through(self, render_boundary_ns: int) -> DrainReceipt: ...
```

Add `_PipelineVoiceSettingsDraft` separately from `_RealtimeSettingsDraft`; do not reuse the Realtime acoustic-barge-in field.

- [ ] **Step 4: Run the targeted tests and import-safety probe.**

Run: `pytest -q Tests/Chat/test_console_voice_settings.py Tests/Audio/test_duplex_contracts.py && python -c "import tldw_chatbook.Audio.duplex_contracts; import tldw_chatbook.Chat.console_voice_settings"`

Expected: tests pass and imports do not load `sounddevice`, PyAudio, or the native extension.

- [ ] **Step 5: Commit only Task 1 files.**

Apply the dirty-worktree protocol to the five Task 1 paths, then commit with `git commit -m "feat: define speculative voice contracts"`.

## Task 2A: Vendor WebRTC AEC3 reproducibly and implement the native binding

**Files:**

- Create: `native/voice_aec/pyproject.toml`
- Create: `native/voice_aec/CMakeLists.txt`
- Create: `native/voice_aec/src/bindings.cpp`
- Create: `native/voice_aec/tldw_voice_aec/__init__.py`
- Create: `native/voice_aec/tools/vendor_webrtc_aec.py`
- Create: `native/voice_aec/vendor/webrtc/UPSTREAM.json`
- Create: `native/voice_aec/vendor/webrtc/FILES.sha256`
- Create: `native/voice_aec/vendor/webrtc/PATCHES.md`
- Create: `native/voice_aec/vendor/webrtc/LICENSE`
- Create: `native/voice_aec/vendor/webrtc/PATENTS`
- Create: `native/voice_aec/THIRD_PARTY_NOTICES.md`
- Create: `native/voice_aec/tests/test_binding.py`
- Test: `native/voice_aec/tests/test_vendored_source.py`

- [ ] **Step 1: Write RED tests for source provenance, import, frame validation, processing, and metrics.**

The Python binding contract is deliberately narrow:

```python
from tldw_voice_aec import AecProcessor


def test_processor_accepts_one_ten_ms_48khz_mono_frame() -> None:
    processor = AecProcessor(sample_rate=48_000, channels=1)
    silence = bytes(480 * 2)
    processor.analyze_render(silence, delay_ms=20)
    cleaned = processor.process_capture(silence, delay_ms=20)
    assert isinstance(cleaned, bytes)
    assert len(cleaned) == len(silence)
    assert {"erle_db", "delay_confidence"} <= processor.metrics().keys()
```

The provenance test pins the [official WebRTC source](https://webrtc.googlesource.com/src/+/109e23c9cec3a44e67c08774874a409741b1e58a/modules/audio_processing/aec3/) at commit `109e23c9cec3a44e67c08774874a409741b1e58a`. The vendoring recipe copies only the dependency closure rooted at `modules/audio_processing`, with allowlisted roots `api/audio`, `api/array_view.h`, `common_audio`, `modules/audio_processing`, `rtc_base`, `system_wrappers`, and required `third_party/abseil-cpp` files. It rejects any copied path outside that allowlist.

- [ ] **Step 2: Run RED without building the extension.**

Run: `pytest -q native/voice_aec/tests/test_binding.py native/voice_aec/tests/test_vendored_source.py`

Expected: import/metadata assertions fail because the package and optional dependency do not exist.

- [ ] **Step 3: Implement and run the deterministic vendoring recipe.**

`vendor_webrtc_aec.py` clones/fetches only the pinned commit into a temporary directory, verifies `HEAD` equals the pinned 40-character commit, derives the compile dependency closure from the checked-in allowlist, copies it without `.git`, and emits:

- `UPSTREAM.json` with repository URL, commit, commit-tree object ID, import timestamp, roots, compiler defines, license path, patent-notice path, and notice-generation version;
- `FILES.sha256` with every vendored relative path and SHA-256 in sorted order; and
- `PATCHES.md` with an ordered patch series. The initial series is explicitly `none`; later updates must add checked-in patch files and their hashes rather than editing vendored source silently.

Run: `python native/voice_aec/tools/vendor_webrtc_aec.py --source /tmp/webrtc-src --verify-clean`

Expected: the script refuses the wrong revision, produces the same manifest on a second run, and `test_vendored_source.py` detects any edited, missing, extra, unlicensed, or patent-notice-uncovered file. `THIRD_PARTY_NOTICES.md` carries the applicable WebRTC BSD license, WebRTC `PATENTS` notice, and notices for every vendored third-party dependency. The checked-in source and manifests make wheel builds network-free. The task is incomplete until all generated hashes and notices are concrete—no placeholder hashes or revision aliases such as `main` are allowed.

- [ ] **Step 4: Implement the binding.**

Expose only:

```python
class AecProcessor:
    def __init__(self, *, sample_rate: int = 48_000, channels: int = 1) -> None: ...
    def analyze_render(self, pcm16: bytes, *, delay_ms: int) -> None: ...
    def process_capture(self, pcm16: bytes, *, delay_ms: int) -> bytes: ...
    def reset(self) -> None: ...
    def metrics(self) -> dict[str, float]: ...
```

The C++ layer must validate exact ten-millisecond mono PCM16 frame sizes, release the GIL during APM work, own all WebRTC objects, and surface Python exceptions rather than terminating. Do not expose WebRTC types to application modules.

- [ ] **Step 5: Build and test a local wheel in an isolated environment.**

Run: `python -m build native/voice_aec --wheel`

Expected: one local platform wheel builds without network access.

Run: `pytest -q native/voice_aec/tests/test_binding.py native/voice_aec/tests/test_vendored_source.py`

Expected: all tests pass.

- [ ] **Step 6: Commit the source and binding with patch-level staging.**

Apply the dirty-worktree protocol to `native/voice_aec`, then commit with `git commit -m "feat: add pinned WebRTC AEC3 binding"`.

## Task 2B: Build, version, distribute, and install the companion wheels

**Files:**

- Create: `Packaging/check_voice_aec_wheel.py`
- Create: `Packaging/check_voice_aec_version_sync.py`
- Create: `Packaging/verify_voice_aec_attestations.py`
- Create: `.github/workflows/voice-aec-wheels.yml`
- Create: `.github/workflows/release-voice-aec.yml`
- Create: `Docs/Development/TTS/voice-aec-release.md`
- Modify: `native/voice_aec/pyproject.toml`
- Modify: `pyproject.toml`
- Test: `Tests/Packaging/test_voice_aec_distribution.py`

- [ ] **Step 1: Write RED distribution and version-lock tests.**

Assert package name `tldw-voice-aec`, companion version exactly equals the application version (`0.1.8.0` at plan time), `speech_recording` pins `tldw-voice-aec==<same-version>`, wheel metadata embeds the upstream commit/tree, `THIRD_PARTY_NOTICES.md`, WebRTC license, and WebRTC patent notice, repaired wheels contain no unexpected shared libraries, and importing `tldw_chatbook` does not import the extension.

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Packaging/test_voice_aec_distribution.py && python Packaging/check_voice_aec_version_sync.py`

Expected: dependency, workflow, and version assertions fail.

- [ ] **Step 3: Add the pull-request wheel matrix.**

`.github/workflows/voice-aec-wheels.yml` builds CPython 3.11-3.13 wheels on macOS x86_64/arm64, Windows x86_64, and Linux x86_64/aarch64; runs `auditwheel`, `delvewheel`, or `delocate`; installs each repaired wheel into a clean environment outside the source tree; runs the binding/corpus smoke tests through cibuildwheel's `test-command`; verifies license and patent notices inside every wheel; generates an SBOM; and uploads one immutable artifact containing wheels, sdist, hashes, SBOM, notices, source-tree digest, and a keyless Sigstore/GitHub OIDC build-provenance bundle for every distribution file. This workflow never publishes.

- [ ] **Step 4: Add an approval-gated release workflow and release order.**

`.github/workflows/release-voice-aec.yml` triggers only from a protected `voice-aec-v<app-version>` tag plus an explicit successful matrix workflow-run ID. It downloads the already-tested immutable artifact from that run; verifies repository, workflow identity, source-tree digest, SHA-256 list, Sigstore/GitHub OIDC provenance bundles, SBOM, license inventory, and patent notices; and refuses locally rebuilt or extra files. In the protected `pypi-release` environment it uses PyPI Trusted Publishing to publish those exact wheel/sdist bytes with attestations enabled. It then installs every published wheel by exact version and hash before marking the release successful. Release never rebuilds a wheel, so nondeterministic ZIP timestamps or repair-tool output cannot invalidate the qualified hashes.

`Docs/Development/TTS/voice-aec-release.md` defines the order: bump app and companion versions together; vendor/update and legal-review source/license/patent notices; pass wheel/AEC qualification; verify keyless build provenance against the repository and exact workflow identity; publish the immutable qualified companion artifacts; verify `pip install tldw-voice-aec==<version>` on all target platforms; only then publish an app release whose `speech_recording` extra pins that exact version. Signing policy requires GitHub OIDC keyless provenance for every wheel/sdist, retention of the verification bundle and SHA list, protected-environment approval, and fail-closed verification before publish. If signing, notice verification, companion publish, or a wheel is missing, the app release is blocked and the legacy pipeline remains selected.

- [ ] **Step 5: Run local distribution checks.**

Run: `python -m build native/voice_aec --wheel && python Packaging/check_voice_aec_wheel.py native/voice_aec/dist/*.whl && python Packaging/check_voice_aec_version_sync.py && pytest -q Tests/Packaging/test_voice_aec_distribution.py`

Expected: all metadata, version, license, patent-notice, symbol, dependency, and offline-install checks pass. Attestation verification is exercised against a checked-in synthetic test bundle locally and against the real OIDC bundle in CI.

- [ ] **Step 6: Commit distribution separately.**

Apply the dirty-worktree protocol to the Task 2B files, then commit with `git commit -m "build: distribute cross-platform voice AEC wheels"`.

## Task 3: Build the app-owned duplex transport and fail-closed preprocessor

**Files:**

- Create: `tldw_chatbook/Audio/aec_backend.py`
- Create: `tldw_chatbook/Audio/duplex_transport.py`
- Create: `tldw_chatbook/Audio/voice_preprocessor.py`
- Modify: `tldw_chatbook/Audio/duplex_contracts.py`
- Create: `Tests/Audio/fakes/fake_duplex_backend.py`
- Test: `Tests/Audio/test_aec_backend.py`
- Test: `Tests/Audio/test_duplex_transport.py`
- Test: `Tests/Audio/test_voice_preprocessor.py`

- [ ] **Step 1: Write deterministic RED tests with an injected monotonic clock.**

Cover strictly increasing capture sequences, a shared render/capture clock, ten-millisecond normalization, bounded overflow behavior, render-before-capture AEC ordering, admission closed while warming/degraded, immediate close on health failure, device reset, content-free `DeviceRouteChanged` emission before the old clock can publish again, and `drain_capture_through(R)` receipts.

```python
async def test_degraded_aec_closes_speech_admission_before_vad() -> None:
    aec = FakeAec(health=AecHealth.DEGRADED)
    observed = []
    preprocessor = VoicePreprocessor(aec=aec, on_admitted_frame=observed.append)
    await preprocessor.process_capture(frame(sequence=1), assistant_rendering=True)
    assert observed == []
    assert preprocessor.mode is DuplexMode.HALF_DUPLEX
```

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Audio/test_aec_backend.py Tests/Audio/test_duplex_transport.py Tests/Audio/test_voice_preprocessor.py`

Expected: collection fails for missing implementation modules.

- [ ] **Step 3: Implement the lazy backend and bounded callback bridge.**

`aec_backend.py` imports `tldw_voice_aec` only inside `create_aec_processor()`. `duplex_transport.py` opens one `sounddevice.RawStream` where possible, stamps input frames before DSP, and confines callbacks to bounded copies/ring operations. Device callbacks never invoke STT, UI, logging, provider cancellation, or Python text reconciliation. Route loss/change atomically closes render admission, invalidates the old clock generation, and enqueues `DeviceRouteChanged(old_clock_generation, route_kind)` on the control queue; it never includes a raw device name.

Use a 48 kHz mono processing domain and explicit resamplers at boundaries. Feed render frames to `analyze_render` in scheduled order, then process capture with the same delay estimator. File-producing TTS must enter via decoded PCM; no hands-free code may launch an external player.

- [ ] **Step 4: Implement health hysteresis and the terminal capture drain.**

Health states are `warming`, `healthy`, and `degraded`. The gate opens only after a bounded healthy streak and closes on the first confidence/discontinuity failure. `drain_capture_through(R)` waits until the input watermark is later than `R`, then verifies capture→AEC→VAD acknowledgements through the greatest earlier sequence. A 500 ms caller deadline, sequence gap, or device reset returns failure rather than guessing.

- [ ] **Step 5: Run targeted tests plus installed/absent dependency probes.**

Run: `pytest -q Tests/Audio/test_aec_backend.py Tests/Audio/test_duplex_transport.py Tests/Audio/test_voice_preprocessor.py`

Run: `python -c "import sys; import tldw_chatbook.Audio.duplex_transport; assert 'sounddevice' not in sys.modules and 'tldw_voice_aec' not in sys.modules"`

Expected: all tests and import-safety assertion pass. Also run the tests once in an environment with the native companion installed and once with it absent; absence must select honest half duplex without hanging.

- [ ] **Step 6: Commit the transport boundary.**

Apply the dirty-worktree protocol to the Task 3 files, then commit with `git commit -m "feat: add fail-closed duplex voice transport"`.

## Task 4: Add native-live and rolling-window transcription behind one revision protocol

**Files:**

- Create: `tldw_chatbook/Audio/rolling_transcript.py`
- Modify: `tldw_chatbook/Audio/dictation_service_lazy.py`
- Test: `Tests/Audio/test_rolling_transcript.py`
- Test: `Tests/Audio/test_dictation_lazy_transcription.py`
- Test: `Tests/Audio/test_dictation_speech_resumed.py`

- [ ] **Step 1: Write RED protocol and reconciliation tests.**

```python
@dataclass(frozen=True, slots=True)
class TranscriptTiming:
    capture_duration_ms: int = 0
    provider_latency_ms: int = 0
    processed_duration_ms: int = 0
    duplicated_duration_ms: int = 0
    usage_units: int = 0


@dataclass(frozen=True, slots=True)
class TranscriptRevision:
    turn_id: str
    revision_id: int
    stable_text: str
    revisable_text: str
    covered_through_ns: int
    mode: Literal["live", "rolling-window"]
    is_final: bool = False
    failure_code: Literal["backend_failed", "fallback_failed"] | None = None
    timing: TranscriptTiming = TranscriptTiming()
```

`TranscriptTiming` contains only capture duration, provider latency, processed duration, duplicated duration, and token/usage counts—never audio or text. Test monotonically increasing revisions, stable-prefix immutability, tail replacement, punctuation/case/whitespace as non-material, word edits as material, coverage tolerance, downstream acknowledgement through admitted-audio sequence, typed failure state, and content-free timing/usage metadata.

For rolling fallback, use a controllably slow fake provider. Assert one batch job at a time, replacement of one obsolete queued window by the newest desired window, overlap reconciliation, no unbounded backlog, and content-free duplicated-duration accounting.

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Audio/test_rolling_transcript.py Tests/Audio/test_dictation_lazy_transcription.py Tests/Audio/test_dictation_speech_resumed.py`

Expected: new protocol tests fail; existing dictation tests remain green.

- [ ] **Step 3: Implement `TranscriptEngine` and adapters.**

Native streaming adapters translate provider events directly. Rolling fallback maintains bounded admitted-audio rings, schedules overlapping windows during speech, and never retains a superseded pending request. Reconcile normalized timestamped tokens while preserving display punctuation from the newest hypothesis.

Add only the smallest clean-frame/revision hook to `LazyLiveDictationService`; keep ordinary Mic dictation and its 60-second behavior unchanged. A backend failure may start exactly one configured fallback adapter attempt. If that attempt fails, publish a terminal failure revision, retain the latest transcript as an editable draft, and suspend automatic dispatch until fresh admitted speech or explicit manual retry. It must not remain indefinitely in `transcribing`.

- [ ] **Step 4: Add freshness and seal behavior.**

`fresh_for(speech_end_ns)` is true only when the revision covers the detected tail within declared frame tolerance. `seal_through(admitted_sequence)` waits until all revisions derived from downstream-acknowledged audio are incorporated. Slow STT keeps status `transcribing`; it never dispatches a truncated snapshot.

- [ ] **Step 5: Run targeted tests.**

Run: `pytest -q Tests/Audio/test_rolling_transcript.py Tests/Audio/test_dictation_lazy_transcription.py Tests/Audio/test_dictation_speech_resumed.py`

Expected: all pass, including slow-fake starvation, one-in-flight, one-fallback-only, editable-draft, and suspended-dispatch assertions.

- [ ] **Step 6: Commit rolling transcription.**

Apply the dirty-worktree protocol to the Task 4 files, then commit with `git commit -m "feat: add rolling transcript revisions"`.

## Task 5: Add sequential cancellable phrase speech through the duplex PCM path

**Files:**

- Create: `tldw_chatbook/Chat/voice_phrase_sequencer.py`
- Modify: `tldw_chatbook/TTS/request_admission.py`
- Modify: `tldw_chatbook/TTS/pcm_stream.py`
- Test: `Tests/Chat/test_voice_phrase_sequencer.py`
- Test: `Tests/TTS/test_tts_request_admission.py`
- Test: `Tests/TTS/test_pcm_stream_plan.py`

- [ ] **Step 1: Write RED phrase and cancellation tests.**

Test punctuation-first emission; resolved-Markdown word/time fallback; no speech inside incomplete code fences, inline code, links, or list prefixes; exactly one synthesis in flight; FIFO PCM writes; attempt-epoch rejection; cancellation closing lazy byte streams; and WAV/PCM decode to the app transport.

```python
async def test_barge_in_aborts_active_audio_and_discards_queued_phrases() -> None:
    sink = FakeDuplexAudioPort(block_on_write=True)
    sequencer = PhraseSpeechSequencer(epoch=3, synthesizer=FakeTts(), sink=sink)
    await sequencer.feed("First phrase. Second phrase.")
    await sequencer.cancel(epoch=3)
    assert sink.abort_count == 1
    assert sequencer.pending_phrase_count == 0
```

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Chat/test_voice_phrase_sequencer.py Tests/TTS/test_tts_request_admission.py Tests/TTS/test_pcm_stream_plan.py`

Expected: new tests fail for the missing sequencer/PCM admission seam.

- [ ] **Step 3: Implement the sequencer without changing legacy semantics.**

Create a new sequencer; do not retrofit `reply_sentence_sequencer.py`. It accepts `(attempt_epoch, delta)`, maintains Markdown state, selects the earliest safe phrase, and allows one bounded fallback only after both a word threshold and a time threshold are reached. One worker drains phrases sequentially.

- [ ] **Step 4: Add a hands-free PCM synthesis seam.**

Extend `TTSRequestAdmissionCoordinator` with an explicit response-format override preferring raw PCM and falling back to WAV only when the adapter/catalog declares it. Drain and close `TTSAudioResponse.byte_stream` in the owner task. Decode WAV to normalized PCM before the duplex transport. If the selected adapter offers neither PCM nor WAV, text generation may finish but speech reports a recoverable failure; never invoke an OS media player.

- [ ] **Step 5: Run targeted tests.**

Run: `pytest -q Tests/Chat/test_voice_phrase_sequencer.py Tests/TTS/test_tts_request_admission.py Tests/TTS/test_pcm_stream_plan.py`

Expected: all pass; the lazy-stream test proves bytes are actually consumed and closed.

- [ ] **Step 6: Commit phrase speech.**

Apply the dirty-worktree protocol to the Task 5 files, then commit with `git commit -m "feat: stream cancellable voice phrases"`.

## Task 6: Fence provider attempts and enforce app-lifetime orphan quarantine

**Files:**

- Create: `tldw_chatbook/Chat/console_voice_attempts.py`
- Create: `tldw_chatbook/Chat/console_voice_supervisor.py`
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py`
- Modify: `tldw_chatbook/Chat/console_runtime.py`
- Test: `Tests/Chat/test_console_voice_attempts.py`
- Test: `Tests/Chat/test_console_voice_supervisor.py`

- [ ] **Step 1: Write RED attempt-fencing and cleanup-race tests.**

Cover immutable request snapshots, no ordinary store sink, inert tool-schema forwarding, complete tool-request capture without execution/approval, typed ordinary provider-failure events with no response-body/exception leakage, late-delta/usage/capture rejection after epoch advance, cooperative cancellation, two-second conservative transition, five-second force-close, 500 ms detach, at most two obsolete cleanups, and no content-bearing orphan retention.

Test two simultaneous cleanup tasks independently becoming orphans. Assert the app-lifetime set contains both and all hands-free dispatch is rejected across hands-free exit/re-entry, provider change, and session reconstruction until every member exits. Typed provider dispatch must remain allowed.

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Chat/test_console_voice_attempts.py Tests/Chat/test_console_voice_supervisor.py`

Expected: modules are missing.

- [ ] **Step 3: Implement attempt-local generation and atomic epoch fencing.**

`VoiceAttempt.start()` calls `ConsoleProviderGateway.stream_chat(..., tools=prepared.tools, signals=ConsoleProviderStreamSignals(exchange_capture_enabled=...))` directly, never through `AgentService` or an execution loop. Tool schemas are inert provider input. The attempt buffers the first complete native tool-request event and freezes further provisional speech, but it has no tool executor, approval hook, citation sink, or durable owner. It also owns response text, usage, sanitized in-memory exchange capture, TTS handles, and its force-closable provider transport. None of these objects can write conversation state.

The first cancellation action is synchronous epoch invalidation. All callbacks check both the attempt-local fence and the coordinator's current epoch before publishing. A normal gateway exception is reduced to `ProviderAttemptFailed(attempt_epoch, error_class)` and handed to the coordinator; the attempt never logs/retains exception text that might contain request content and never converts it into cancellation-orphan handling unless its transport also fails to close.

- [ ] **Step 4: Implement one app-lifetime supervisor.**

Construct `VoiceDispatchSupervisor` from the app/runtime composition root, not the view. It owns a set of opaque cleanup handles capped by the two-obsolete-attempt limit. A nonempty set rejects every hands-free provider dispatch before provider/session resolution. Exiting hands-free, changing provider, rebuilding a session, or ending the logical turn never clears it. Automatic recovery occurs only when every orphan exits; app restart is the explicit last resort.

- [ ] **Step 5: Run race tests repeatedly.**

Run: `pytest -q Tests/Chat/test_console_voice_attempts.py Tests/Chat/test_console_voice_supervisor.py`

Run: `pytest -q Tests/Chat/test_console_voice_attempts.py Tests/Chat/test_console_voice_supervisor.py --count=20`

Expected: all runs pass. If `pytest-repeat` is not installed, use the repository's existing repeat mechanism or run the command in a short shell loop; do not add a runtime dependency.

- [ ] **Step 6: Commit attempt lifecycle and supervisor.**

Apply the dirty-worktree protocol to the Task 6 files, then commit with `git commit -m "feat: fence speculative voice attempts"`.

## Task 7: Implement the serialized logical-turn coordinator

**Files:**

- Create: `tldw_chatbook/Chat/console_speculative_voice.py`
- Test: `Tests/Chat/test_console_speculative_voice.py`
- Test: `Tests/Chat/test_console_speculative_voice_properties.py`

- [ ] **Step 1: Write a pure model-based RED suite.**

Use an injected scheduler/clock and a serialized event mailbox. Cover:

- first admitted frame creates one logical turn;
- every admitted frame resets eagerness;
- dispatch waits for nonempty, fresh tail coverage;
- default 700 ms and bounded configured values;
- speech during generation or playback fences the attempt, aborts audio, extends the same turn, and redispatches from accumulated text;
- material STT changes restart, while case/punctuation/whitespace changes do not;
- the first material correction fences immediately, starts a 120 ms correction-coalescing window, and a burst emits exactly one replacement from the latest snapshot; repeated corrections may extend the debounce but a 250 ms hard cap prevents starvation;
- the first three attempts in a rolling ten seconds use configured eagerness, later attempts use 1.5 seconds;
- two obsolete cleanups pause dispatch and two-second cleanup enters `serialized_conservative`;
- conservative mode requires two seconds of quiet and zero obsolete attempts;
- speech after the terminal audible boundary starts a new turn;
- explicit Stop, Esc, mic toggle, hands-free exit, navigation, and teardown discard provisional work and preserve no history;
- TTS failure can commit completed text only if no newer event exists.
- a device-route change during generation or playback fences output immediately, retains the latest transcript, aborts the old transport, rebuilds one clock/AEC domain with admission closed, and regenerates only after the new route is healthy (or explicitly half duplex) plus a fresh silence interval;
- an ordinary LLM/provider terminal failure retains the logical user transcript as an editable draft, emits no assistant/promotion, reopens listening for the same logical turn, and allows additional speech or explicit manual retry.

Use Hypothesis state-machine tests to generate event permutations and assert: at most one current epoch, no old epoch emits output, no promotion with pending pre-boundary speech, and cancelled content never reaches a persistence fake.

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Chat/test_console_speculative_voice.py Tests/Chat/test_console_speculative_voice_properties.py`

Expected: collection fails because the coordinator is absent.

- [ ] **Step 3: Implement the mailbox and explicit state transitions.**

```python
class SpeculativeTurnCoordinator:
    async def submit(self, event: VoiceEvent) -> None:
        await self._mailbox.put(event)

    async def _run(self) -> None:
        while True:
            event = await self._mailbox.get()
            await self._reduce(event)
```

Timers enqueue events; they never mutate state directly. The reducer owns turn ID, attempt epoch, transcript snapshot, speech/render boundaries, audio clock generation, obsolete cleanup count, restart history, correction-coalescing start/deadline, and terminal status. Spoken-command classification precedes LLM dispatch. The first material correction synchronously advances the epoch and aborts output; subsequent material revisions inside the 120 ms window only replace the pending snapshot. Dispatch occurs once at the quiet/coalescing deadline, with a 250 ms maximum from the first correction.

`DeviceRouteChanged` synchronously advances the epoch, aborts phrase/TTS/provider output, invalidates pending render/capture seals, retains the latest transcript/draft, and enters `rebuilding_audio`. The coordinator accepts no speculative dispatch until the transport publishes a new clock generation and AEC reports healthy or explicit half duplex; it then restarts the response-eagerness timer and regenerates from the retained exact snapshot. A rebuild failure leaves an editable draft and suspends speculation.

`ProviderAttemptFailed` is distinct from cancellation cleanup failure. It fences the failed epoch, records only a content-free error class, retains the logical transcript as an editable draft, enters `listening_after_provider_failure`, and creates no assistant row, terminal receipt, capture promotion, or automatic retry. Fresh admitted speech extends the same logical turn and may dispatch after silence; manual Retry may reuse the exact draft; explicit exit discards it.

- [ ] **Step 4: Implement the terminal render/capture/STT seal.**

After final playback boundary `R`, request `drain_capture_through(R)`. Only after the input watermark is later than `R` and AEC/VAD acknowledges every sequence at or before it may the coordinator call `seal_through(admitted_sequence)`. The combined operation has a 500 ms deadline. Timeout, a sequence gap, device reset, or failed acknowledgement preserves an editable draft, suspends speculation, and requires a healthy duplex rebuild.

In intentional half duplex, no playback-period capture is eligible; render completion closes the gated interval, and only manual interruption can extend that same turn during playback.

- [ ] **Step 5: Run deterministic and property tests.**

Run: `pytest -q Tests/Chat/test_console_speculative_voice.py Tests/Chat/test_console_speculative_voice_properties.py`

Expected: all tests pass under the fake clock without wall-clock sleeps.

- [ ] **Step 6: Commit the coordinator.**

Apply the dirty-worktree protocol to the Task 7 files, then commit with `git commit -m "feat: coordinate speculative voice turns"`.

## Task 8: Route effectful contexts through the ordinary accepted-turn pipeline

**Files:**

- Create: `tldw_chatbook/Chat/console_voice_eligibility.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_speculative_voice.py`
- Test: `Tests/Chat/test_console_voice_eligibility.py`
- Test: `Tests/Chat/test_console_voice_effect_barrier.py`

- [ ] **Step 1: Write RED eligibility and handoff tests.**

Plain provider turns with a frozen system/conversation context are eligible, including turns whose prepared provider request contains tool schemas. Tool schemas are forwarded to the attempt-local gateway so the model may produce a tool request, but no executor or approval hook exists. Any turn requiring automatic Library/RAG retrieval, citation creation, or another pre-dispatch authority is ineligible for provisional execution; it waits for two seconds of stable silence and exact transcript freshness, then calls the ordinary accepted controller once.

Assert that a complete first tool request freezes provisional speech, starts/continues the two-second stable-silence barrier, cannot execute or create an approval, and is discarded before ordinary re-dispatch. Speech before barrier expiry fences the attempt normally. Assert that navigation after handoff behaves according to ADR-094, while navigation before it cancels the view-scoped provisional turn.

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Chat/test_console_voice_eligibility.py Tests/Chat/test_console_voice_effect_barrier.py`

Expected: tests fail for the missing classifier/handoff seam.

- [ ] **Step 3: Add one public ordinary-turn handoff seam.**

Add a narrow `submit_accepted_voice_turn(exact_user_text, frozen_session_context)` method to `ConsoleChatController`. It must enter the same preparation, tool/RAG, approvals, persistence, provider gateway, and navigation-surviving runtime path as typed Send. Do not duplicate accepted-turn preparation or call private controller internals from the voice coordinator.

- [ ] **Step 4: Implement the two-second effect barrier.**

The coordinator classifies only pre-dispatch-effect contexts before provisional dispatch. Those contexts show `waiting for stable turn`, never invoke the speculative generation gateway, and hand off exactly once after 2,000 ms of stable silence plus fresh transcript coverage.

For tool-capable speculative attempts, the first complete provider tool request closes phrase input, freezes provisional speech, and waits until total stable silence since the latest admitted speech reaches 2,000 ms. The coordinator then fences/discards the attempt and its capture envelope and hands the exact immutable transcript to the ordinary accepted pipeline with its normal tool schemas, Capture On reservation, approvals, persistence, and runtime custody. The speculative tool request itself is never reused or executed.

- [ ] **Step 5: Run the targeted effect-boundary tests.**

Run: `pytest -q Tests/Chat/test_console_voice_eligibility.py Tests/Chat/test_console_voice_effect_barrier.py Tests/Chat/test_console_chat_controller.py`

Expected: voice barrier and existing controller tests pass.

- [ ] **Step 6: Commit the effect barrier.**

Apply the dirty-worktree protocol to the Task 8 files, then commit with `git commit -m "feat: gate effectful voice turns"`.

## Task 9: Promote only the winning no-tool attempt through ADR-094 and ADR-097

**Blocked until:** the ADR-094 terminalization transaction and ADR-097 trace promotion service named in Preconditions are merged and available on this branch.

**Files:**

- Create: `tldw_chatbook/Chat/console_voice_promotion.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `tldw_chatbook/Chat/console_exchange_capture.py`
- Modify: `tldw_chatbook/Chat/console_speculative_voice.py`
- Test: `Tests/Chat/test_console_voice_promotion.py`
- Test: `Tests/Chat/test_console_voice_capture.py`

- [ ] **Step 1: Inspect and name the landed service interfaces before writing tests.**

Replace generic names in this task with the actual ADR-094 and ADR-097 public methods. If either service is absent, stop this task and leave the rollout gate off; do not implement a substitute.

- [ ] **Step 2: Write RED atomic-promotion and capture tests.**

Test that one immutable winning snapshot atomically creates the user message, assistant message, terminal receipt, and exact unseen mark. Cancelled attempts, failed seals, and stale epochs create none. Retry/idempotency keys prevent duplicate promotion.

With Capture On, promote only the winner's sanitized in-memory provider exchange after terminalization and label it `provisional_voice_promoted` with explicit post-dispatch provenance. Capture failure is best-effort and does not roll back the accepted conversation pair. Temporary chats never create durable capture, and saving one later does not synthesize a historical trace.

- [ ] **Step 3: Run RED.**

Run: `pytest -q Tests/Chat/test_console_voice_promotion.py Tests/Chat/test_console_voice_capture.py`

Expected: adapter tests fail until the narrow promotion integration exists.

- [ ] **Step 4: Implement adapters only.**

`VoiceWinningPromotion` receives the exact winning prompt, completed assistant text, terminal boundary, attempt identity, content-free usage, and optional sanitized capture envelope. It delegates transaction ownership to ADR-094 and trace ownership to ADR-097. It never writes SQL directly and never calls ordinary streaming sinks.

- [ ] **Step 5: Run persistence, receipt, and capture tests.**

Run: `pytest -q Tests/Chat/test_console_voice_promotion.py Tests/Chat/test_console_voice_capture.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_exchange_capture.py`

Expected: all pass, including exact unseen-mark acknowledgement and temporary-chat cases.

- [ ] **Step 6: Commit the promotion adapter.**

Apply the dirty-worktree protocol to the Task 9 files, then commit with `git commit -m "feat: promote winning speculative voice turns"`.

## Task 10: Wire view-scoped hands-free lifecycle and ephemeral preview rows

**Files:**

- Create: `tldw_chatbook/Widgets/Console/console_voice_preview.py`
- Modify: `tldw_chatbook/UI/Console_Modules/hands_free.py`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/Widgets/Console/console_speech_controls.py`
- Test: `Tests/UI/test_console_speculative_voice_wiring.py`
- Test: `Tests/UI/test_console_voice_preview.py`
- Test: `Tests/UI/test_console_voice_accessibility.py`
- Test: `Tests/UI/test_console_hands_free_wiring.py`

- [ ] **Step 1: Write RED UI/lifecycle tests before mounting hardware.**

Patch audio/native factories before creating the Textual app. Assert:

- the qualification gate selects legacy `HandsFreeController` when false;
- the development-qualified path constructs one view-scoped duplex engine/coordinator;
- rolling user text and current assistant output appear as visually provisional rows that are not present in `ConsoleStore` or `ConsoleTranscript.set_messages()` input;
- replacing an attempt removes only the old assistant preview;
- winning promotion removes previews after durable rows arrive;
- statuses distinguish listening, transcribing, responding, speaking, response update, AEC warming, half duplex, and cleanup quarantine;
- Stop/Esc, mic toggle, hands-free exit, and screen navigation cancel previews and audio;
- the app-lifetime orphan supervisor survives view teardown.
- the separate Realtime engine, manual per-message **Speak** action, and accessibility announcement throttling retain their current construction and behavior.

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/UI/test_console_speculative_voice_wiring.py Tests/UI/test_console_voice_preview.py Tests/UI/test_console_hands_free_wiring.py`

Expected: missing preview/wiring tests fail; legacy hands-free tests stay green.

- [ ] **Step 3: Mount a separate ephemeral preview widget.**

`ConsoleVoicePreview` accepts a pure projection:

```python
@dataclass(frozen=True, slots=True)
class VoicePreviewProjection:
    turn_id: str
    attempt_epoch: int
    user_text: str
    assistant_text: str
    status: str
```

Mount it adjacent to the transcript's message region and style it as provisional. Do not fabricate `ConsoleChatMessage` IDs, call `set_messages`, or insert preview rows into durable transcript grouping/action logic.

- [ ] **Step 4: Branch the existing controller wiring.**

In `ConsoleHandsFreeController`, select the new pipeline only when the internal qualification gate is true. Reuse frozen provider/session selection but keep audio/coordinator resources view-scoped per ADR-094. Preserve the existing Realtime engine and legacy hands-free flow unchanged. Manual interruption must bypass acoustic admission and synchronously fence the active epoch before awaiting cleanup.

- [ ] **Step 5: Run UI and legacy regressions.**

Run: `pytest -q Tests/UI/test_console_speculative_voice_wiring.py Tests/UI/test_console_voice_preview.py Tests/UI/test_console_voice_accessibility.py Tests/UI/test_console_hands_free_wiring.py Tests/Chat/test_console_hands_free.py Tests/Chat/test_console_voice_input.py Tests/UI/test_console_speech_controls.py Tests/TTS/test_console_speech_snapshot_admission.py`

Expected: all pass without touching real microphone, speaker, or native libraries; manual message speech remains snapshot-gated; and rapidly changing provisional text is announced at a bounded cadence rather than on every STT token.

- [ ] **Step 6: Commit Console wiring.**

Apply the dirty-worktree protocol to the Task 10 files, then commit with `git commit -m "feat: wire speculative Console voice previews"`.

## Task 11: Add canonical settings, privacy-safe metrics, and honest documentation

**Files:**

- Create: `tldw_chatbook/Audio/voice_metrics.py`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`
- Modify: `tldw_chatbook/Chat/console_voice_settings.py`
- Modify: `tldw_chatbook/config.py`
- Modify: `Docs/User_Guide/console/attachments-images-voice.md`
- Modify: `Docs/User_Guide/console.md`
- Create: `Docs/Development/TTS/speculative-duplex-voice.md`
- Test: `Tests/Audio/test_voice_metrics.py`
- Test: `Tests/UI/test_settings_speculative_voice_panel.py`
- Test: `Tests/Chat/test_console_settings_defaults.py`

- [ ] **Step 1: Write RED settings-ownership and content-privacy tests.**

Test response-eagerness presets Fast 700/Balanced 1200/Deliberate 2000, valid numeric bounds 500-3000, invalid-value content-free warning plus fallback to 700, `pipeline_aec_enabled=true` by default, troubleshooting-only false→half-duplex behavior, and no config write merely from opening Settings.

Assert `dictation.acoustic_barge_in` remains consumed only by Realtime compatibility code, and `dictation.handsfree_send_delay_seconds` remains legacy-only. No startup migration rewrites either key or infers eagerness from the old delay.

Metrics tests must reject transcript text, response text, or PCM fields and retain only durations, counts, modes, health states, provider result class, and split winning/discarded usage.

- [ ] **Step 2: Run RED.**

Run: `pytest -q Tests/Audio/test_voice_metrics.py Tests/UI/test_settings_speculative_voice_panel.py Tests/Chat/test_console_settings_defaults.py`

Expected: new settings and metrics assertions fail.

- [ ] **Step 3: Add one canonical “Pipeline conversation” settings block.**

Modify only the F9 Settings Speech & TTS panel. Show native-live versus rolling-window STT mode, overlapping-remote-audio cost disclosure, speculative-call cost disclosure, response eagerness, and the troubleshooting-only AEC disable. Explain that unhealthy/unavailable AEC automatically becomes half duplex. Do not show the development qualification gate.

- [ ] **Step 4: Implement content-free metrics and diagnostics.**

Record EOS→dispatch, barge→audible-stop, replacement dispatch, first audio, AEC state/ERLE aggregate, underruns, restarts, conservative-mode entry, duplicated STT audio duration, and winning/discarded usage. Logs may include turn/attempt opaque IDs but no transcript, response, audio, provider request body, or sanitized capture body.

- [ ] **Step 5: Update user and developer docs.**

Document same-turn interruption semantics, the 700 ms default and safe range, live/fallback STT modes, remote overlap cost, AEC warming/degraded half duplex, manual interruption, cancelled-attempt privacy, temporary-chat capture behavior, and the unchanged Realtime engine. Add a bounded compatibility note: old send-delay and acoustic-barge-in keys remain for legacy/Realtime and are not migrated.

- [ ] **Step 6: Run targeted settings/docs tests.**

Run: `pytest -q Tests/Audio/test_voice_metrics.py Tests/UI/test_settings_speculative_voice_panel.py Tests/Chat/test_console_settings_defaults.py Tests/UI/test_settings_speech_tts_panel.py Tests/Chat/test_console_voice_input.py`

Expected: all pass; opening Settings leaves config bytes unchanged; the bounded compatibility note is present; and legacy delay plus Realtime acoustic behavior remains unchanged.

- [ ] **Step 7: Commit settings, metrics, and documentation.**

Apply the dirty-worktree protocol to the Task 11 files, then commit with `git commit -m "feat: expose qualified speculative voice settings"`.

## Task 12A: Automate DSP, latency, lifecycle, and durable-owner qualification

**Files:**

- Create: `Tests/Audio/fixtures/voice_aec/manifest.json`
- Create: `Tests/Audio/test_voice_aec_corpus.py`
- Create: `Tests/integration/test_speculative_voice_pipeline.py`
- Create: `Tests/Performance/test_speculative_voice_latency.py`
- Create: `Tests/Packaging/test_voice_aec_installed_wheel.py`
- Create: `Tests/Chat/test_console_voice_ephemerality.py`
- Create: `Scripts/qualify_speculative_voice.py`
- Create: `Packaging/compute_voice_source_digest.py`
- Create: `Packaging/speculative_voice_source_paths.txt`
- Create: `Docs/Development/TTS/speculative-voice-durable-owner-inventory.md`
- Create: `Artifacts/voice_qualification/automated/.gitkeep`
- Modify: `.github/workflows/voice-aec-wheels.yml`

- [ ] **Step 1: Build a licensed, content-safe DSP corpus and RED qualification tests.**

Use synthetic speech/noise plus redistributable fixtures with source/license/hash in the manifest; never use captured user audio. Cases include stationary/nonlinear echo, delay steps, clock drift, double-talk, underrun/overrun, device reset, and Bluetooth-like latency. Assert:

- median ERLE at least 20 dB and p10 at least 10 dB on the reference corpus;
- false acoustic barge-in no more than one per 30 minutes of rendered speech;
- double-talk recall at least 95%;
- any unqualified case closes admission and selects half duplex.

- [ ] **Step 2: Add deterministic end-to-end and latency harnesses.**

The integration path must consume the real lazy TTS byte stream, deliver rendered PCM through the duplex transport, feed synthetic echo plus user speech into capture, revise the rolling transcript, cancel a blocked provider/TTS attempt, and promote exactly one winner through test doubles for the landed ADR-094/097 services.

Latency tests with warm deterministic fakes assert:

- post-AEC EOS→LLM dispatch ≤850 ms p95;
- speech detection→audible stop ≤150 ms p95;
- post-AEC EOS of the added speech→replacement handoff ≤850 ms p95;
- end of speech→first assistant audio ≤1.5 s median and ≤2.5 s p95.

- [ ] **Step 3: Inventory and probe every default durable owner with sentinel content.**

`speculative-voice-durable-owner-inventory.md` names the concrete owner, backing path/table, writer seam, and test probe for each of these surfaces: Console message/session DB, terminal receipts and unseen marks, ADR-097 traces/capture, provider usage rows, tool/approval state, citations, notifications, replay/trajectory state, chatbook/export serialization, temporary-chat memory and save-later behavior, persistent file logs, stdlib/loguru handlers, in-app log buffers/share-log output, and exception/crash diagnostics.

`test_console_voice_ephemerality.py` runs cancelled attempts with unique user and assistant poison sentinels through the outermost Console voice wiring in an isolated app-data directory. It then inspects every inventory owner—not only mocks—including all SQLite database files/tables, trace/usage repositories, notification and approval repositories, replay/export serializers, temporary-chat stores, file logs, in-memory log views, and captured exception representations. The poison strings must be absent everywhere after cancellation; content-free attempt counts/usage may remain. A control write must prove each probe can detect its sentinel, so an empty or disconnected probe cannot pass.

- [ ] **Step 4: Define the non-self-referential code-under-test identity.**

`speculative_voice_source_paths.txt` explicitly lists every runtime, native source/build, dependency metadata, qualification harness/schema, packaging generator, and test file whose bytes affect the feature. `compute_voice_source_digest.py` sorts those paths and hashes the canonical sequence `relative-path NUL file-sha256 LF`. It fails if a listed path is missing, untracked, or dirty.

Generated evidence and authority are deliberately outside the digest: `Artifacts/voice_qualification/**`, `Docs/Development/TTS/speculative-voice-qualification.md`, `tldw_chatbook/Audio/voice_qualification_manifest.json`, `tldw_chatbook/Audio/voice_build_identity.json`, SBOMs, attestations, signatures, caches, and `.git`. Tests prove changing included code changes the digest, while adding an evidence report or regenerating the manifest does not. Reports use `source_tree_digest`; git revision is optional provenance only and never an acceptance key.

- [ ] **Step 5: Run deterministic qualification locally as a non-authoritative harness check.**

Run: `.venv/bin/python -m pytest -q Tests/Audio/test_voice_aec_corpus.py Tests/integration/test_speculative_voice_pipeline.py Tests/Performance/test_speculative_voice_latency.py Tests/Packaging/test_voice_aec_installed_wheel.py Tests/Chat/test_console_voice_ephemerality.py`

Expected: all automated gates pass after confirming that `.venv/bin/python` has the intended `webrtcvad`, `sounddevice`, and companion wheel. `Scripts/qualify_speculative_voice.py --scenario automated --source-tree-digest $(.venv/bin/python Packaging/compute_voice_source_digest.py) --output /tmp/speculative-voice-automated.json` writes schema version, source-tree digest, optional git provenance, interpreter, companion version/upstream commit/wheel SHA-256, corpus thresholds, latency distributions using the correct clock boundaries, lifecycle results, and durable-owner inventory/report hashes. It contains no transcript, response, audio, raw device name, or credential. This `/tmp` report only validates the harness; Task 12C regenerates authoritative evidence after all qualification/runtime code is committed.

- [ ] **Step 6: Run bounded native and lifecycle harness checks.**

Run: `.venv/bin/python Scripts/qualify_speculative_voice.py --scenario duplex-soak --minutes 30 --output Artifacts/voice_qualification/automated/<platform>-duplex-soak.json`

Run: `.venv/bin/python Scripts/qualify_speculative_voice.py --scenario cancellation-soak --minutes 30 --output Artifacts/voice_qualification/automated/<platform>-cancellation-soak.json`

Expected: no unbounded task/process growth, no audio ring growth, no orphan-set size above two, no post-fence callback, no device-handle leak, and no content-bearing diagnostic output. Include at least one real native-process fixture because injected mocks cannot prove process reaping.

- [ ] **Step 7: Commit automated qualification separately.**

Apply the dirty-worktree protocol to the Task 12A files, then commit with `git commit -m "test: qualify speculative voice automation"`.

## Task 12B: Build the three-platform physical-device qualification harness

**Files:**

- Create: `Scripts/qualify_physical_voice.py`
- Create: `Packaging/voice_physical_report.schema.json`
- Create: `Artifacts/voice_qualification/physical/.gitkeep`
- Create: `Docs/Development/TTS/speculative-voice-qualification.md`
- Test: `Tests/Packaging/test_voice_physical_reports.py`

- [ ] **Step 1: Write RED schema and safety tests.**

The report schema requires source-tree digest, platform/architecture, app and companion versions, upstream commit, generated corpus/latency/soak report hashes, hashed device identity, transport/sample-rate information, AEC health path, ERLE distribution, false-barge rate, double-talk recall, stop latency, degradation behavior, operator checklist, and pass/fail. It rejects raw device names, transcript/response text, PCM, and `passed=true` when any automated prerequisite hash is missing, uses a different source-tree digest, or a full-duplex safety threshold fails.

- [ ] **Step 2: Implement the guided physical harness.**

`qualify_physical_voice.py` guides repeatable rendered-speech, double-talk, interruption, silence, device-switch, and 30-minute soak trials. It hashes device identifiers with a report-local salt, never records microphone or response content, and emits a schema-valid JSON report plus a human-readable summary. A failed or unavailable AEC path is passing only when playback-period speech admission is observably closed and manual interruption remains available.

- [ ] **Step 3: Validate the harness with synthetic pass/fail fixtures.**

Run the harness in `--fixture` mode against a checked-in synthetic safe-full-duplex case, safe-half-duplex case, and unsafe-unsuppressed case. Only the first two validate as passing. Do not record final physical evidence yet; Task 12C runs the real matrix after all source-digested qualification/runtime code is committed.

The final command shape is: `<absolute-python> Scripts/qualify_physical_voice.py --source-tree-digest <digest> --platform <platform-arch> --device-class <builtin|usb|bluetooth> --automated-report <path> --output Artifacts/voice_qualification/physical/<platform-arch>-<device-class>.json`.

Bluetooth passes only if healthy full duplex meets the thresholds or explicit safe half duplex closes playback-period admission; unsuppressed interruption is a failure.

- [ ] **Step 4: Validate and commit the harness.**

Run: `pytest -q Tests/Packaging/test_voice_physical_reports.py`

Expected: schema and synthetic harness tests pass with no privacy-forbidden fields. Apply the dirty-worktree protocol and commit with `git commit -m "test: add speculative voice device qualification harness"`.

## Task 12C: Generate the rollout capability manifest and enable only qualified platforms

**Files:**

- Create: `Packaging/voice_qualification_manifest.schema.json`
- Create: `Packaging/generate_voice_qualification_manifest.py`
- Create: `Packaging/speculative_voice_python_paths.txt`
- Create: `tldw_chatbook/Audio/voice_qualification_manifest.json`
- Create: `tldw_chatbook/Audio/voice_build_identity.json`
- Modify: `tldw_chatbook/Chat/console_voice_settings.py`
- Modify: `pyproject.toml`
- Modify: `.github/workflows/release-voice-aec.yml`
- Test: `Tests/Packaging/test_voice_qualification_manifest.py`
- Test: `Tests/Packaging/test_speculative_voice_lint_scope.py`
- Test: `Tests/Chat/test_console_voice_settings.py`

- [ ] **Step 1: Write RED manifest-generation and fail-closed runtime tests.**

Schema version 1 requires exact app/AEC identity and the platform keys `macos-arm64`, `macos-x86_64`, `windows-x86_64`, `linux-x86_64`, and `linux-aarch64`. Hash fields use the JSON Schema constraint `{"type": "string", "pattern": "^[0-9a-f]{64}$"}`. Its logical shape is:

```json
{
  "schema_version": 1,
  "pipeline_version": 1,
  "app_version": "0.1.8.0",
  "source_tree_digest": "64-character lowercase SHA-256 constrained by schema",
  "aec_package": {
    "name": "tldw-voice-aec",
    "version": "0.1.8.0",
    "upstream_commit": "109e23c9cec3a44e67c08774874a409741b1e58a"
  },
  "platforms": {
    "macos-arm64": {
      "qualified": true,
      "wheel_sha256": "64-character lowercase SHA-256 constrained by schema",
      "extension_sha256": "64-character lowercase SHA-256 constrained by schema",
      "automated_report_sha256": "64-character lowercase SHA-256 constrained by schema",
      "physical_report_sha256": "64-character lowercase SHA-256 constrained by schema"
    }
  }
}
```

The prose strings above illustrate field meaning; the checked-in manifest must contain concrete schema-valid hashes. Tests prove the generator refuses a missing target platform, device class, report hash, failed threshold, version/commit mismatch, mixed source-tree digest, source-tree digest unequal to the computed included-source digest, unknown field, or privacy-forbidden evidence. Runtime selection returns false on missing/malformed manifest, manifest/build-identity digest mismatch, unknown platform, `qualified=false`, companion import/version/commit mismatch, or installed native-extension file hash mismatch. Environment variables, config values, and CLI arguments cannot override the result.

- [ ] **Step 2: Implement the deterministic evidence generator.**

`generate_voice_qualification_manifest.py` consumes an explicit `--source-tree-digest`, exact repaired-wheel paths, Task 12A automated/soak reports, and all Task 12B physical reports. It first recomputes the included-source digest and requires every report to use it. It validates report schemas/hashes, requires built-in/USB/Bluetooth evidence for every declared platform key, records the wheel SHA-256, extracts and hashes the platform extension binary for runtime verification, and emits sorted canonical JSON plus `voice_build_identity.json`. It does not accept a manual `--qualified` flag.

The final generation command in Step 6 is: `.venv/bin/python Packaging/generate_voice_qualification_manifest.py --source-tree-digest "$voice_source_digest" --app-version 0.1.8.0 --aec-upstream 109e23c9cec3a44e67c08774874a409741b1e58a --wheelhouse <qualified-wheelhouse> --automated-dir Artifacts/voice_qualification/automated --physical-dir Artifacts/voice_qualification/physical --output tldw_chatbook/Audio/voice_qualification_manifest.json --build-identity-output tldw_chatbook/Audio/voice_build_identity.json`.

Expected: generation succeeds only when all evidence is present and passing; the output is byte-identical on a second run.

- [ ] **Step 3: Implement fail-closed runtime lookup and package the manifest.**

`speculative_voice_qualified()` maps the current OS/architecture to one manifest key, requires the packaged build-identity digest to equal the manifest digest, validates schema/app/pipeline/companion version and upstream commit, hashes the installed native-extension file and compares it with `extension_sha256`, and returns that platform's `qualified` value. Any exception returns false with a content-free warning. `pyproject.toml` includes both JSON files as package data. `pipeline_aec_enabled=false` remains a local half-duplex troubleshooting switch; it does not bypass qualification.

- [ ] **Step 4: Commit all qualification/runtime code before producing final evidence.**

Commit the schema, source-digest tool/list, evidence generators, physical harness, runtime reader, tests, workflow logic, and a checked-in hard-off manifest/build identity with no qualified platforms. No product or qualification code may change after this commit without invalidating and rerunning every Task 12A/12B report.

- [ ] **Step 5: Compute one clean source-tree digest and generate final evidence.**

Require a clean status for every path in `speculative_voice_source_paths.txt`, then run: `voice_source_digest=$(.venv/bin/python Packaging/compute_voice_source_digest.py)`.

Using exactly that digest and source commit, rerun Task 12A automated, duplex-soak, cancellation-soak, and durable-owner probes for every platform/architecture. Then, on macOS, Windows, and Linux, run Task 12B for built-in, USB, and Bluetooth device classes with `--source-tree-digest "$voice_source_digest"`. Bluetooth passes only if healthy full duplex meets thresholds or safe half duplex closes playback-period admission. Store final schema-valid reports under `Artifacts/voice_qualification/{automated,physical}` and record their hashes in `Docs/Development/TTS/speculative-voice-qualification.md`.

- [ ] **Step 6: Generate the manifest and prove the evidence chain is stable.**

Run the Step 2 generation command with `--source-tree-digest "$voice_source_digest"`. Re-run `compute_voice_source_digest.py` before and after staging the generated evidence, qualification docs, manifest, and build identity; both outputs must equal `$voice_source_digest`. This is possible because only generated evidence/authority paths are excluded—the code that produced and consumes them is included and was committed before measurement.

- [ ] **Step 7: Make the release workflow regenerate and compare authority.**

The approval-gated release workflow downloads the exact qualified wheel/evidence artifacts, recomputes the included-source digest, requires it to match every report and the packaged build identity, regenerates the manifest, fails unless it matches the reviewed checked-in file byte-for-byte, installs the built app wheel in a clean environment, and runs `test_voice_qualification_manifest.py` against every supported platform key before publishing the app. Git commit IDs remain provenance only. A missing/unqualified platform keeps legacy hands-free on that platform.

- [ ] **Step 8: Run final targeted verification and complete static checks.**

Run: `pytest -q Tests/Audio/test_duplex_contracts.py Tests/Audio/test_aec_backend.py Tests/Audio/test_duplex_transport.py Tests/Audio/test_voice_preprocessor.py Tests/Audio/test_rolling_transcript.py Tests/Audio/test_voice_aec_corpus.py Tests/Chat/test_voice_phrase_sequencer.py Tests/Chat/test_console_voice_attempts.py Tests/Chat/test_console_voice_supervisor.py Tests/Chat/test_console_speculative_voice.py Tests/Chat/test_console_speculative_voice_properties.py Tests/Chat/test_console_voice_eligibility.py Tests/Chat/test_console_voice_effect_barrier.py Tests/Chat/test_console_voice_promotion.py Tests/Chat/test_console_voice_capture.py Tests/Chat/test_console_voice_ephemerality.py Tests/UI/test_console_speculative_voice_wiring.py Tests/UI/test_console_voice_preview.py Tests/UI/test_console_voice_accessibility.py Tests/UI/test_settings_speculative_voice_panel.py Tests/integration/test_speculative_voice_pipeline.py Tests/Performance/test_speculative_voice_latency.py Tests/Packaging/test_voice_aec_distribution.py Tests/Packaging/test_voice_aec_installed_wheel.py Tests/Packaging/test_voice_physical_reports.py Tests/Packaging/test_voice_qualification_manifest.py Tests/Packaging/test_speculative_voice_lint_scope.py`

`Packaging/speculative_voice_python_paths.txt` lists every Python file created or modified by Tasks 1-12, including Audio, Chat, TTS, provider gateway, runtime, controller, persistence/capture, UI/Widgets, config, native wrapper, packaging scripts, and tests. Run: `xargs ruff check < Packaging/speculative_voice_python_paths.txt`

Run: `xargs ruff format --check < Packaging/speculative_voice_python_paths.txt`

Run: `clang-format --dry-run --Werror native/voice_aec/src/bindings.cpp`

Run: `git diff --check`

Expected: targeted tests, Python/C++ formatting, lint, and whitespace checks pass. The lint-scope test compares the implementation plan's Python file inventory with `speculative_voice_python_paths.txt` so no modified Python path is omitted. Per repository policy, ask the user before running the full `pytest` suite; do not treat targeted success as full-suite evidence.

- [ ] **Step 9: Commit rollout evidence and authority separately.**

Apply the dirty-worktree protocol to generated Task 12 evidence, qualification documentation, manifest, and build identity only; inspect hashes explicitly and prove the source-tree digest remains unchanged. Then commit with `git commit -m "feat: enable qualified speculative voice platforms"`.

## Implementation completion checklist

- [ ] Every implementation Backlog task links the design and ADR, has checked acceptance criteria, implementation notes, targeted verification evidence, and the correct status.
- [ ] ADR-094 and ADR-097 prerequisite interfaces were reused, not reimplemented.
- [ ] Cancelled attempts have no durable content, effects, citations, approvals, replay, or content-bearing diagnostics.
- [ ] Every hands-free TTS path renders through app-owned PCM with a timestamped AEC reference.
- [ ] The orphan set is app-lifetime, bounded at two, and globally quarantines hands-free dispatch until empty.
- [ ] The 500 ms render-boundary capture/VAD/STT seal fails closed.
- [ ] Native-live and rolling-window STT both pass freshness/revision tests.
- [ ] Settings ownership and no-write migration behavior are verified.
- [ ] All three desktop platforms pass packaging and physical-device qualification, or the rollout gate remains off on the unqualified platform.
- [ ] User approval was obtained before any full-suite test run.
