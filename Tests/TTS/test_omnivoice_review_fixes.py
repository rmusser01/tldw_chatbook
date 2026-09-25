"""Regressions for the PR #2825 review + UAT findings.

Each test pins a defect the fake-session suite could not see: the real
``tokenizers`` return type, the ct03 graph's declared input ranks, the
upstream loudness/reference contract, the sequence-scaled timeout budget,
task-cancel propagation, and the Settings save bindings.
"""

from __future__ import annotations

import asyncio
import threading
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tldw_chatbook.TTS.backends.omnivoice as omnivoice_module
from tldw_chatbook.TTS.adapter_types import TTSOperationError
from tldw_chatbook.TTS.backends.omnivoice import (
    _OrtBatchRunner,
    _OrtWaveRunner,
    _postprocess,
    _resolve_language,
)
from tldw_chatbook.TTS.omnivoice_prompt import build_prompt_inputs

from Tests.TTS.test_omnivoice_backend import _make_backend, _make_tree


# --- tokenizer: the real library returns an Encoding, not a list ---------------


def test_build_prompt_inputs_accepts_real_tokenizers_encoding() -> None:
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, "hello": 1, "world": 2}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    assert not isinstance(tok.encode("hello", add_special_tokens=False), list)

    prompt = build_prompt_inputs(tok, text="hello world", target_len=4)

    assert prompt.input_ids.shape[0] == 8
    assert prompt.input_ids.dtype == np.int64
    assert (prompt.input_ids[:, -4:] == 1024).all()


# --- ORT adapters honour the declared input ranks -------------------------------


class _Input(SimpleNamespace):
    pass


class _RecordingSession:
    def __init__(self, inputs, output):
        self._inputs = inputs
        self._output = output
        self.feeds: list[dict] = []

    def get_inputs(self):
        return self._inputs

    def run(self, _names, feeds):
        self.feeds.append(feeds)
        return [self._output]


def _ct03_lm_session() -> _RecordingSession:
    # Exactly the ct03 omnivoice_lm_int8_hq signature (inspected on the model).
    return _RecordingSession(
        [
            _Input(name="input_ids", type="tensor(int64)", shape=["batch", 8, "sequence"]),
            _Input(name="audio_mask", type="tensor(bool)", shape=["batch", "sequence"]),
            _Input(
                name="attention_mask",
                type="tensor(bool)",
                shape=["batch", 1, "sequence", "sequence"],
            ),
            _Input(name="position_ids", type="tensor(int64)", shape=["batch", "sequence"]),
        ],
        np.zeros((2, 8, 5, 1025), dtype=np.float32),
    )


def test_batch_runner_feeds_ct03_ranks_and_caches_static_inputs() -> None:
    session = _ct03_lm_session()
    runner = _OrtBatchRunner(session)
    ids = np.zeros((2, 8, 5), dtype=np.int64)
    audio = np.zeros((2, 8, 5), dtype=np.bool_)
    audio[:, :, 3:] = True
    attention = np.ones((2, 1, 5, 5), dtype=np.bool_)

    runner.run(ids, audio, attention)
    runner.run(ids, audio, attention)

    first, second = session.feeds
    assert first["input_ids"].shape == (2, 8, 5)
    assert first["audio_mask"].shape == (2, 5)
    assert first["audio_mask"][:, 3:].all() and not first["audio_mask"][:, :3].any()
    assert first["position_ids"].shape == (2, 5)
    assert first["position_ids"][1].tolist() == [0, 1, 2, 3, 4]
    assert first["attention_mask"].shape == (2, 1, 5, 5)
    # per-request constants are built once, not per diffusion step
    assert second["position_ids"] is first["position_ids"]
    assert second["attention_mask"] is first["attention_mask"]


def test_wave_runner_feeds_batch_channel_samples() -> None:
    session = _RecordingSession(
        [_Input(name="audio", type="tensor(float)", shape=["batch", 1, "num_samples"])],
        np.zeros((1, 8, 3), dtype=np.int64),
    )
    _OrtWaveRunner(session).run(np.zeros(960, dtype=np.float32))
    assert session.feeds[0]["audio"].shape == (1, 1, 960)


# --- language -----------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "auto", "AUTO", "none", " None "])
def test_language_agnostic_values_resolve_to_none(value) -> None:
    assert _resolve_language(value) is None


def test_language_ids_pass_through() -> None:
    assert _resolve_language("es") == "es"


async def test_auto_language_prompts_none_and_request_language_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, tokenizer, _ = _make_backend(
        root, monkeypatch, extra_config={"OMNIVOICE_LANGUAGE": "auto"}
    )
    [c async for c in backend.generate_speech_stream(text="hello", voice="")]
    assert any("<|lang_start|>None<|lang_end|>" in t for t in tokenizer.encodes)

    request = SimpleNamespace(
        input="hola",
        voice="",
        response_format="wav",
        speed=1.0,
        extra_params={"language": "es"},
    )
    tokenizer.encodes.clear()
    [c async for c in backend.generate_speech_stream(request)]
    assert any("<|lang_start|>es<|lang_end|>" in t for t in tokenizer.encodes)


# --- cloning contract -----------------------------------------------------------


async def test_reference_audio_without_transcript_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    with pytest.raises(TTSOperationError) as err:
        async for _ in backend.generate_speech_stream(
            text="hello", reference_audio=str(tmp_path / "ref.wav")
        ):
            pass
    assert err.value.code == "request_invalid"
    assert "reference_text" in str(err.value)


def test_postprocess_follows_upstream_loudness_contract() -> None:
    tone = (0.3 * np.sin(np.linspace(0, 200 * np.pi, 24_000))).astype(np.float32)
    peak = lambda a: float(np.max(np.abs(a)))  # noqa: E731

    # no reference: peak-normalize to 0.5
    assert peak(_postprocess(tone.copy(), None)) == pytest.approx(0.5, rel=1e-3)
    # loud reference (>= 0.1 RMS): cloned loudness left alone
    assert peak(_postprocess(tone.copy(), 0.2)) == pytest.approx(0.3, rel=1e-3)
    # quiet reference: scaled back down by ref_rms / 0.1
    assert peak(_postprocess(tone.copy(), 0.05)) == pytest.approx(0.15, rel=1e-3)


def test_encode_reference_boosts_quiet_clip_and_trims_to_whole_frames(
    tmp_path: Path,
) -> None:
    samples = 24_000 + 123  # not a multiple of the 320-sample hop
    quiet = (0.02 * np.sin(np.linspace(0, 400 * np.pi, samples))).astype(np.float32)
    ref = tmp_path / "quiet.wav"
    with wave.open(str(ref), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24_000)
        w.writeframes((quiet * 32767).astype("<i2").tobytes())

    seen: list[np.ndarray] = []

    class _Encoder:
        def run(self, waveform):
            seen.append(waveform)
            return [np.zeros((1, 8, waveform.size // 320), dtype=np.int64)]

    backend = omnivoice_module.OmniVoiceOnnxTTSBackend({})
    backend._encoder = _Encoder()
    codes = backend._encode_reference(str(ref))

    fed = seen[0]
    assert fed.size % 320 == 0 and fed.size == samples - samples % 320
    assert float(np.sqrt(np.mean(fed**2))) == pytest.approx(0.1, rel=0.02)
    assert backend._last_reference_rms == pytest.approx(0.02 / np.sqrt(2), rel=0.02)
    assert codes.shape == (8, fed.size // 320)


# --- timeout budget scales with the whole sequence ----------------------------


async def test_long_reference_short_line_does_not_time_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """20 s of reference codes + a ~1 s line: each step runs the LM over the
    whole sequence, so 20 s of sampling is normal. The old target-only budget
    (~8 s) timed this out; UAT measured 32.5 s wall for this shape."""
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(
        root, monkeypatch, extra_config={"OMNIVOICE_TIMEOUT_FACTOR": 1.0}
    )
    ref_codes = np.zeros((8, 1500), dtype=np.int64)  # 20 s at 75 fps
    monkeypatch.setattr(
        backend, "_encode_reference", lambda path, max_duration=None: ref_codes
    )
    clock = {"now": 1000.0}
    monkeypatch.setattr(omnivoice_module.time, "monotonic", lambda: clock["now"])

    def slow_sampling(lm, prompt_ids, target_len, *, cancel_check=None, **_):
        clock["now"] += 20.0
        assert not cancel_check()
        return np.zeros((8, target_len), dtype=np.int64)

    monkeypatch.setattr(omnivoice_module, "run_diffusion_sampling", slow_sampling)
    # A transcript whose length matches ~20 s of speech, so the duration
    # estimator sizes "Yes." realistically (~1 s) — as in the UAT run.
    transcript = (
        "Once upon a time, in a small village near the mountains, there lived "
        "an old clockmaker who repaired every clock in town. Each morning he "
        "opened his shop at seven, brewed a pot of strong tea, and listened "
        "carefully to the ticking of a hundred tiny machines. Children would "
        "gather at the window to watch him work, and he would wave at them."
    )
    chunks = [
        c
        async for c in backend.generate_speech_stream(
            text="Yes.", reference_audio="ref.wav", reference_text=transcript
        )
    ]
    assert len(chunks) == 1


# --- cancelling the awaiting task stops the sampler before the lock frees -----


async def test_task_cancel_stops_sampler_thread_before_releasing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "m"
    _make_tree(root)
    backend, _, _ = _make_backend(root, monkeypatch)
    started = threading.Event()
    stopped = threading.Event()

    def blocking_sampling(lm, prompt_ids, target_len, *, cancel_check=None, **_):
        started.set()
        while not cancel_check():
            threading.Event().wait(0.01)
        stopped.set()
        raise omnivoice_module.OmniVoiceSamplingCancelled("stop")

    monkeypatch.setattr(omnivoice_module, "run_diffusion_sampling", blocking_sampling)

    async def consume():
        async for _ in backend.generate_speech_stream(text="hello", voice=""):
            pass

    task = asyncio.create_task(consume())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()  # the thread observed the cancel ...
    assert not backend._generation_lock.locked()  # ... before the lock freed
    assert not backend._cancel.is_set()  # a task cancel never poisons the backend


# --- Settings save bindings -----------------------------------------------------


def test_every_speech_settings_key_has_a_persistence_binding() -> None:
    """A key without a binding is silently dropped by _persist_settings —
    OmniVoice's Save wrote nothing until its keys were bound."""
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
        _TTS_SETTING_BINDINGS,
    )
    from tldw_chatbook.UI.Screens.settings_speech_tts import (
        BUILT_IN_TTS_PROVIDER_ORDER,
        _PROVIDER_NON_SECRET_DEFAULTS,
        _provider_event_settings,
    )

    for provider_id in BUILT_IN_TTS_PROVIDER_ORDER:
        defaults = _PROVIDER_NON_SECRET_DEFAULTS.get(provider_id)
        if not defaults or provider_id == "audio_cpp":
            continue
        values = {key: value for key, value in defaults.items()}
        values.setdefault("random_seed", "")
        try:
            emitted = _provider_event_settings(provider_id, values)
        except KeyError:
            continue  # provider needs fields beyond the non-secret defaults
        unbound = sorted(set(emitted) - set(_TTS_SETTING_BINDINGS))
        assert not unbound, f"{provider_id}: {unbound}"


def test_omnivoice_settings_bind_to_omnivoice_section() -> None:
    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
        _TTS_SETTING_BINDINGS,
    )

    binding = _TTS_SETTING_BINDINGS["OMNIVOICE_VOICE_SAMPLES_DIR"]
    assert binding.destinations == (("OmniVoiceSettings", "voice_samples_dir"),)
    assert binding.provider_id == "omnivoice"


def test_blank_model_root_saves_and_emits_bound_keys() -> None:
    """Blank model_root means "use the managed artifact"; a fresh install's
    blank value must not block saving the other OmniVoice fields (UAT: the
    Save reported "Model root: This field is required")."""
    from copy import deepcopy

    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
        _TTS_SETTING_BINDINGS,
    )
    from tldw_chatbook.UI.Screens.settings_speech_tts import (
        build_global_speech_tts_save_proposal,
        load_global_speech_tts_state,
    )

    original = load_global_speech_tts_state({}, environment={})
    assert original.providers["omnivoice"]["model_root"] == ""
    draft = deepcopy(original)
    draft.providers["omnivoice"]["num_steps"] = 16

    proposal = build_global_speech_tts_save_proposal(
        original, draft, configure_provider="omnivoice"
    )

    assert proposal.settings["OMNIVOICE_NUM_STEPS"] == 16
    assert proposal.settings["OMNIVOICE_MODEL_ROOT"] == ""
    assert set(proposal.settings) <= set(_TTS_SETTING_BINDINGS)


# --- request admission accepts the Speech Lab knobs ---------------------------


def test_admission_accepts_lab_defaults_for_omnivoice_options() -> None:
    """UAT: every Lab generation failed with invalid_selection/provider_options
    because unlisted numeric options were range-checked against [0, 1]."""
    from tldw_chatbook.TTS.effective_settings import _validated_options

    options = _validated_options(
        "omnivoice",
        {"num_steps": 32, "guidance_scale": 2.0, "max_reference_duration": 30.0},
        None,
    )
    assert dict(options) == {
        "num_steps": 32,
        "guidance_scale": 2.0,
        "max_reference_duration": 30.0,
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"num_steps": 0},
        {"num_steps": 200},
        {"num_steps": 8.5},
        {"guidance_scale": 11.0},
        {"max_reference_duration": 0.5},
    ],
)
def test_admission_rejects_out_of_range_omnivoice_options(bad) -> None:
    from tldw_chatbook.TTS.effective_settings import (
        TTSEffectiveResolutionError,
        _validated_options,
    )

    with pytest.raises(TTSEffectiveResolutionError):
        _validated_options("omnivoice", bad, None)


# --- Qodo review (PR #2825): paths + optional deps -----------------------------


def test_configured_root_with_nul_is_model_invalid() -> None:
    from tldw_chatbook.TTS.backends.omnivoice import resolve_model_root

    with pytest.raises(TTSOperationError) as err:
        resolve_model_root({"OMNIVOICE_MODEL_ROOT": "bad\x00root"}, managed=None)
    assert err.value.code == "model_invalid"


@pytest.mark.parametrize("reference", ["bad\x00ref.wav", "/definitely/missing/ref.wav"])
def test_reference_path_is_validated_and_not_echoed(reference: str) -> None:
    backend = omnivoice_module.OmniVoiceOnnxTTSBackend({})
    with pytest.raises(TTSOperationError) as err:
        backend._encode_reference(reference)
    assert err.value.code == "request_invalid"
    assert "missing" not in str(err.value)  # no raw path in the user message


async def test_dependency_check_goes_through_optional_deps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[tuple[str, str]] = []

    def fake_check(module: str, feature: str | None = None) -> bool:
        checked.append((module, feature))
        return module != "tokenizers"

    monkeypatch.setattr(omnivoice_module, "check_dependency", fake_check)
    backend = omnivoice_module.OmniVoiceOnnxTTSBackend({})
    with pytest.raises(TTSOperationError) as err:
        await backend.initialize()
    assert err.value.code == "dependency_missing"
    assert checked == [
        ("onnxruntime", "omnivoice_tts"),
        ("tokenizers", "omnivoice_tts"),
    ]


def test_legacy_playground_path_routes_omnivoice_to_its_backend() -> None:
    """The Voice Cloning test button uses the legacy playground path, which
    derived an internal id of "default" for omnivoice (no such backend)."""
    from uuid import uuid4

    from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
        STTSEventHandler,
    )
    from tldw_chatbook.TTS import STTSPlaygroundRequest

    request = STTSPlaygroundRequest(
        operation_id=str(uuid4()),
        provider_id="omnivoice",
        model_id="default",
        text="hello",
        voice_id="profile:narrator",
        response_format="wav",
        speed=1.0,
        options={"source": "voice_cloning_test"},
    )
    assert (
        STTSEventHandler._legacy_internal_model_id(request, dict(request.options))
        == "local_omnivoice_default"
    )
