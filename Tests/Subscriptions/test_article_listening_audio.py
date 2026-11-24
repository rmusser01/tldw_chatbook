"""Article generation exercises actual TTS admission and source-bound publication."""

import asyncio
from dataclasses import asdict, replace
from hashlib import sha256
from importlib import import_module, util
from io import BytesIO
import json
import wave

import pytest

from Tests.Subscriptions.test_article_audio_assets import (
    listening,  # noqa: F401
    native_effects,
    publish,
)  # noqa: F401
from Tests.Subscriptions.test_article_listening_store import records  # noqa: F401
from Tests.TTS.test_tts_request_admission import (
    _CapturingAdapter,
    _native_service,
    _snapshot,
)
from tldw_chatbook.Subscriptions.article_listening_models import FrozenVoice
from tldw_chatbook.TTS.profile_service import LoadedTTSProfile
from Tests.TTS.test_default_profile_request_resolver import (
    _profile,
    _FakeDefaultProfileService,
)


def api():
    name = "tldw_chatbook.Subscriptions.article_listening_audio"
    assert util.find_spec(name), "missing bounded article synthesis owner"
    return import_module(name)


def mono(frames=800, rate=8000):
    dst = BytesIO()
    with wave.open(dst, "wb") as wav:
        wav.setparams((1, 2, rate, frames, "NONE", ""))
        wav.writeframes(b"\x01\x01" * frames)
    return dst.getvalue()


class Transport(_CapturingAdapter):
    def admitted_outbound_endpoint(self):
        return "https://api.openai.com/v1/audio/speech"


def service(adapter=None, voice="alloy"):
    adapter = adapter or Transport("openai")
    adapter.chunks = (mono(),)
    selection = _snapshot(
        provider_id="openai", model_id="tts-1", voice_mode="exact", voice_id=voice
    )
    return _native_service(adapter, selection)[0], adapter


async def ready():
    return None


@pytest.mark.asyncio
async def test_frozen_saved_uuid_survives_restart_and_revision_change_pauses():
    mod = api()
    svc, _ = service(voice="nova")
    profile = _profile(provider_id="openai", model_id="tts-1", voice_id="alloy")
    profiles = _FakeDefaultProfileService(result=LoadedTTSProfile(2, profile))
    try:
        frozen = await mod.freeze_listening_voice(
            tts_service=svc,
            profile_service=profiles,
            default_profile_id=str(profile.profile_id),
        )
        assert frozen.voice == "alloy"
        assert [
            (dep.kind, dep.identifier, dep.revision) for dep in frozen.dependencies
        ] == [("profile", str(profile.profile_id), str(profile.revision))]
        reopened = FrozenVoice.from_json(json.dumps(asdict(frozen)))
        later = _FakeDefaultProfileService(result=LoadedTTSProfile(24, profile))
        args = await mod.resolve_listening_voice(
            reopened, tts_service=svc, profile_service=later
        )
        assert args["default_profile"].profile_id == profile.profile_id
        replaced = _FakeDefaultProfileService(
            result=LoadedTTSProfile(24, replace(profile, revision=profile.revision + 1))
        )
        with pytest.raises(ValueError, match="voice_changed"):
            await mod.resolve_listening_voice(
                reopened, tts_service=svc, profile_service=replaced
            )
    finally:
        await svc.close()
        await svc.wait_closed()


def test_nullable_native_voice_and_provider_specific_boolean_roundtrip():
    assert (
        FrozenVoice.from_json(
            json.dumps(asdict(FrozenVoice("audio_cpp", "model", None)))
        ).voice
        is None
    )
    knob = (("use_speaker_boost", False),)
    assert FrozenVoice("elevenlabs", "model", "voice", options=knob).options == knob
    for provider, value in [("openai", True), ("elevenlabs", 1)]:
        with pytest.raises(ValueError):
            FrozenVoice(
                provider, "model", "voice", options=(("use_speaker_boost", value),)
            )


@native_effects
@pytest.mark.asyncio
async def test_verified_audio_spends_no_tts_calls(listening):  # noqa: F811
    store, session, root = listening
    entry = session.queue[0]
    audio = publish(store, session, root, entry)
    owner = api().ArticleListeningAudio(store, root, tts_service=None, readiness=ready)
    result = await owner.prepare(
        entry.entry_id, expected_revision=store.load_session().revision
    )
    assert result.wav.sha256 == audio.wav.sha256


@native_effects
@pytest.mark.asyncio
async def test_all_saved_words_reach_real_admission_and_single_tail(listening):  # noqa: F811
    store, session, root = listening
    from Tests.Subscriptions.test_article_listening_store import prepare

    body = "Distinctive phrase. " * 250 + "Final sentence."
    with store.db.transaction() as c:
        c.execute("UPDATE subscription_items SET content=? WHERE id=1", (body,))
    candidate = prepare(store, session.authority, session.voice)
    current = store.replace_session(candidate, expected_revision=session.revision)
    svc, adapter = service(voice="nova")
    try:
        owner = api().ArticleListeningAudio(
            store, root, tts_service=svc, readiness=ready
        )
        audio = await owner.prepare(
            current.foreground.entry_id, expected_revision=current.revision
        )
        requests = [req.options["_legacy_openai_request"] for req in adapter.requests]
        assert len(requests) > 1 and all(req.voice == "alloy" for req in requests)
        assert " ".join(" ".join(req.input for req in requests).split()) == " ".join(
            body.split()
        )
        assert adapter.response_close_calls == len(requests)
        assert sha256(audio.wav.path.read_bytes()).hexdigest() == audio.wav.sha256
        with wave.open(str(audio.wav.path), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
                48000,
                2,
                2,
            )
            assert wav.getnframes() == audio.wav.content_frames + 12000
            wav.setpos(audio.wav.content_frames - 1)
            assert wav.readframes(12001) == b"\x01\x01" * 2 + bytes(48000)
    finally:
        await svc.close()
        await svc.wait_closed()


@native_effects
@pytest.mark.asyncio
@pytest.mark.parametrize("retirement", ["cancel", "source", "epoch"])
async def test_late_response_cannot_attach(listening, retirement):  # noqa: F811
    store, session, root = listening
    mod = api()
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldAdapter(Transport):
        async def synthesize(self, request, progress_sink=None):
            entered.set()
            await release.wait()
            return await super().synthesize(request, progress_sink)

    svc, adapter = service(HeldAdapter("openai"))
    owner = mod.ArticleListeningAudio(store, root, tts_service=svc, readiness=ready)
    entry = session.queue[0]

    async def start():
        return await owner.prepare(entry.entry_id, expected_revision=session.revision)

    task = asyncio.create_task(start())
    try:
        async with asyncio.timeout(5):
            await entered.wait()
        if retirement == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(ValueError, match="busy"):
                await start()
        elif retirement == "source":
            store.db.delete_article(entry.item_id)
        else:
            original = store._current
            store._current = lambda: replace(original(), policy_epoch=99)
        release.set()
        if retirement != "cancel":
            with pytest.raises(ValueError):
                await task
        await owner.settle()
        assert adapter.response_close_calls == 1
        assert not any(r["status"] == "ready" for r in store.audio_inventory())
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await svc.close()
        await svc.wait_closed()


@pytest.mark.asyncio
async def test_accumulator_rejects_expansion_and_final_copy_before_publication():
    mod = api()

    async def pieces(wav):
        yield wav

    assembly = mod.ArticleWavAssembly(limit=110000)
    with pytest.raises(ValueError, match="audio_limit"):
        await assembly.append(pieces(mono(1000, 1)), audio_format="wav")
    assembly = mod.ArticleWavAssembly(limit=60000)
    await assembly.append(pieces(mono(20)), audio_format="wav")
    with pytest.raises(ValueError, match="audio_limit"):
        assembly.finish()


@native_effects
@pytest.mark.asyncio
async def test_cancellation_inside_attachment_rolls_back_and_retires(
    listening,  # noqa: F811
    monkeypatch,
):
    import threading

    store, session, root = listening
    entered, release = threading.Event(), threading.Event()
    original = store._attach

    def hold_after_mutating(*args):
        original(*args)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(store, "_attach", hold_after_mutating)
    svc, _ = service()
    owner = api().ArticleListeningAudio(store, root, tts_service=svc, readiness=ready)
    task = asyncio.create_task(
        owner.prepare(session.queue[0].entry_id, expected_revision=session.revision)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        owner.revoke()
        release.set()
        with pytest.raises(ValueError, match="retired"):
            await task
        assert store.load_session().queue[0].artifact is None
        assert not any(r["status"] == "ready" for r in store.audio_inventory())
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await svc.close()
        await svc.wait_closed()


@native_effects
@pytest.mark.asyncio
async def test_failed_physical_cleanup_keeps_admission_closed(listening):  # noqa: F811
    store, session, root = listening

    class UncertainAdapter(Transport):
        async def synthesize(self, request, progress_sink=None):
            response = await super().synthesize(request, progress_sink)

            async def fail_cleanup():
                raise OSError("private detail")

            response.add_cleanup(fail_cleanup)
            return response

    svc, adapter = service(UncertainAdapter("openai"))
    owner = api().ArticleListeningAudio(store, root, tts_service=svc, readiness=ready)

    async def start():
        return await owner.prepare(
            session.queue[0].entry_id, expected_revision=session.revision
        )

    try:
        with pytest.raises(ValueError, match="retirement_unconfirmed"):
            await start()
        with pytest.raises(ValueError, match="retirement_unconfirmed"):
            await start()
        assert len(adapter.requests) == 1
        assert store.load_session().queue[0].artifact is None
    finally:
        await svc.close()
        await svc.wait_closed()


@native_effects
@pytest.mark.asyncio
async def test_retired_before_adapter_submission_spends_zero_requests(listening):  # noqa: F811
    store, session, root = listening
    entered, release = asyncio.Event(), asyncio.Event()

    class HeldReadiness(Transport):
        async def ensure_ready(self):
            entered.set()
            await release.wait()
            await super().ensure_ready()

    svc, adapter = service(HeldReadiness("openai"))
    owner = api().ArticleListeningAudio(store, root, tts_service=svc, readiness=ready)
    task = asyncio.create_task(
        owner.prepare(session.queue[0].entry_id, expected_revision=session.revision)
    )
    try:
        async with asyncio.timeout(5):
            await entered.wait()
        owner.revoke()
        release.set()
        with pytest.raises(ValueError):
            await task
        assert adapter.requests == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await svc.close()
        await svc.wait_closed()


@native_effects
@pytest.mark.asyncio
async def test_missing_artifact_requires_explicit_regeneration(listening):  # noqa: F811
    store, session, root = listening
    entry = session.queue[0]
    old = publish(store, session, root, entry)
    old.wav.path.unlink()
    svc, adapter = service()
    owner = api().ArticleListeningAudio(store, root, tts_service=svc, readiness=ready)
    try:
        with pytest.raises(ValueError, match="article_audio_missing"):
            await owner.prepare(
                entry.entry_id, expected_revision=store.load_session().revision
            )
        current = store.load_session()
        assert current.queue[0].artifact == old.generation
        assert not adapter.requests
        fresh = await owner.prepare(
            entry.entry_id, expected_revision=current.revision, regenerate=True
        )
        assert fresh.generation != old.generation
        assert store.load_session().queue[0].bookmark.frame == 0
    finally:
        await svc.close()
        await svc.wait_closed()


@pytest.mark.asyncio
async def test_clone_dependency_digests_and_admission_keep_native_reference(tmp_path):
    from Tests.TTS.test_tts_request_admission import (
        _CloneCapturingAdapter,
        _clone_reference,
        generation_module,
        TTSCloneReferenceMaterializer,
        TTSCloneRecipeRequirement,
    )

    mod = api()
    adapter = _CloneCapturingAdapter()
    svc, _ = _native_service(adapter, _snapshot(model_id="clone-model"))
    recipe = TTSCloneRecipeRequirement("pocket_tts", 1, "clone-model")
    ref = _clone_reference(recipe)
    profile = _profile(
        provider_id="audio_cpp",
        model_id=recipe.model_id,
        voice_id=None,
        reference=ref.summary,
    )
    profiles = _FakeDefaultProfileService(result=LoadedTTSProfile(5, profile))
    profiles.reference_result = ref
    svc._clone_materializer = TTSCloneReferenceMaterializer(tmp_path / "materializer")

    async def facts(requirement):
        return generation_module.AudioCppGuidedDependencySnapshot(
            state="exact",
            provider_configuration_revision=1,
            saved_generation=1,
            applied_generation=1,
            pending_configuration=False,
            saved_requirement=requirement,
            applied_requirement=requirement,
        )

    svc.audio_cpp_guided_dependency_snapshot = facts
    try:
        frozen = await mod.freeze_listening_voice(
            tts_service=svc,
            default_profile_id=str(profile.profile_id),
            profile_service=profiles,
        )
        serialized = json.dumps(asdict(frozen))
        assert ref.reference_text not in serialized and ref.sha256 in serialized
        assert frozen.voice is None and len(frozen.dependencies) == 2
        args = await mod.resolve_listening_voice(
            frozen, tts_service=svc, profile_service=profiles
        )
        from tldw_chatbook.TTS.profile_reference_materialization import _POSIX_SUPPORTED

        if _POSIX_SUPPORTED:
            response, _ = await svc.synthesize_effective(text="synthetic", **args)
            await response.aclose()
            assert len(adapter.clone_requests) == 1
        else:
            from tldw_chatbook.TTS.adapter_types import TTSOperationError

            with pytest.raises(TTSOperationError) as refused:
                await svc.synthesize_effective(text="synthetic", **args)
            assert refused.value.code == "request_invalid"
            assert not adapter.requests and not adapter.clone_requests
        for changed in [
            replace(ref, reference_text="different synthetic transcript"),
            _clone_reference(replace(recipe, recipe_revision=2)),
        ]:
            profiles.reference_result = changed
            profiles.result = LoadedTTSProfile(
                5, replace(profile, reference=changed.summary)
            )
            with pytest.raises(ValueError, match="voice_changed"):
                await mod.resolve_listening_voice(
                    frozen, tts_service=svc, profile_service=profiles
                )
    finally:
        await svc.close()
        await svc.wait_closed()


@pytest.mark.asyncio
async def test_reject_loose_local_reference_and_unbound_native_default():
    mod = api()
    svc, _ = service()
    try:
        for frozen in [
            FrozenVoice("higgs", "higgs-audio-v2", "clip.wav"),
            FrozenVoice("omnivoice", "omnivoice-int8hq", "clip.wav"),
            FrozenVoice("audio_cpp", "opaque", None),
        ]:
            with pytest.raises(ValueError, match="voice_unbound"):
                await mod.resolve_listening_voice(frozen, tts_service=svc)
    finally:
        await svc.close()
        await svc.wait_closed()


@pytest.mark.asyncio
async def test_budget_stops_before_requesting_next_piece():
    assembly = api().ArticleWavAssembly(limit=100000)

    async def stream():
        yield bytes(40000)
        pytest.fail("drained past aggregate refusal")

    with pytest.raises(ValueError, match="audio_limit"):
        await assembly.append(stream(), audio_format="wav")


@pytest.mark.asyncio
async def test_elevenlabs_backend_controls_are_frozen_as_typed_options():
    from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
    from tldw_chatbook.TTS.adapter_types import TTSProviderSpec, TTSProviderDescriptor
    from tldw_chatbook.TTS.TTS_Generation import TTSService

    provider, model, voice = (
        "elevenlabs",
        "eleven_multilingual_v2",
        "21m00Tcm4TlvDq8ikWAM",
    )
    prefs = _snapshot(
        provider_id=provider, model_id=model, voice_mode="exact", voice_id=voice
    )

    def build(style):
        adapter = _CapturingAdapter(provider)
        descriptor = TTSProviderDescriptor(
            provider_id=provider, display_name="Test", native=False
        )
        config = {
            "app_config": {
                "app_tts": {
                    "ELEVENLABS_USE_SPEAKER_BOOST": False,
                    "ELEVENLABS_VOICE_STABILITY": 0.3,
                    "ELEVENLABS_SIMILARITY_BOOST": 0.8,
                    "ELEVENLABS_STYLE": 0.1,
                },
                "elevenlabs_" + model: {"ELEVENLABS_STYLE": style},
            }
        }
        registry = TTSAdapterRegistry(
            specs=(
                TTSProviderSpec(
                    descriptor=descriptor,
                    factory=lambda _: adapter,
                    initial_config=config,
                    exclusive_reconfigure=False,
                ),
            ),
            aliases={},
        )
        return TTSService(registry, preferences_snapshot=prefs), adapter

    original, _ = build(0.7)
    current, transport = build(0.1)
    try:
        frozen = await api().freeze_listening_voice(tts_service=original)
        assert dict(frozen.options) == {
            "use_speaker_boost": False,
            "stability": 0.3,
            "style": 0.7,
            "similarity_boost": 0.8,
        }
        args = await api().resolve_listening_voice(frozen, tts_service=current)
        response, _ = await current.synthesize_effective(text="synthetic", **args)
        await response.aclose()
        request = transport.requests[0].options["_legacy_openai_request"]
        assert request.extra_params == dict(frozen.options)
    finally:
        await original.wait_closed()
        await current.wait_closed()


@pytest.mark.asyncio
async def test_elevenlabs_error_never_renders_or_chains_private_provider_data(
    monkeypatch,
):
    import httpx
    from loguru import logger
    from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
    from tldw_chatbook.TTS.backends import elevenlabs

    monkeypatch.setattr(
        elevenlabs, "get_cli_setting", lambda section, key, default=None: default
    )
    marker = "synthetic-sensitive-example"
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                400, json={"detail": {"message": marker}}, request=request
            )
        )
    )
    monkeypatch.setattr(
        "tldw_chatbook.TTS.base_backends.build_httpx_async_client",
        lambda **kwargs: client,
    )
    backend = elevenlabs.ElevenLabsTTSBackend(config={"ELEVENLABS_API_KEY": marker})
    messages = []
    sink = logger.add(messages.append, diagnose=True, backtrace=True)
    try:
        request = OpenAISpeechRequest(
            model="eleven_multilingual_v2",
            voice="21m00Tcm4TlvDq8ikWAM",
            input=marker,
            response_format="pcm",
        )
        with pytest.raises(ValueError) as refused:
            _ = [piece async for piece in backend.generate_speech_stream(request)]
        assert marker not in "".join(messages)
        assert marker not in str(refused.value)
        assert refused.value.__cause__ is None and refused.value.__context__ is None
        assert "400" in str(refused.value)
    finally:
        logger.remove(sink)
        await backend.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("onnx", [True, False])
async def test_kokoro_native_generation_error_excludes_request_locals(onnx):
    from loguru import logger
    from tldw_chatbook.TTS.audio_schemas import OpenAISpeechRequest
    from tldw_chatbook.TTS.adapter_types import TTSOperationError
    from tldw_chatbook.TTS.backends.kokoro import KokoroTTSBackend

    marker = "synthetic-sensitive-kokoro"
    from types import SimpleNamespace

    def failed(voice):
        raise RuntimeError(marker)

    native = SimpleNamespace(kokoro_model_pt=object(), _parse_voice_config=failed)
    generate = (
        KokoroTTSBackend._generate_onnx_stream
        if onnx
        else KokoroTTSBackend._generate_pytorch_stream
    )
    messages = []
    sink = logger.add(messages.append, diagnose=True, backtrace=True)
    try:
        request = OpenAISpeechRequest(
            model="kokoro", voice="af_bella", input=marker, response_format="pcm"
        )
        with pytest.raises(TTSOperationError) as refused:
            _ = [piece async for piece in generate(native, request)]
        assert marker not in "".join(messages)
        assert refused.value.__cause__ is None and refused.value.__context__ is None
    finally:
        logger.remove(sink)


@native_effects
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["platform", "conversion", "chunks"])
async def test_preflight_refusal_precedes_every_submission(
    listening,  # noqa: F811
    monkeypatch,
    failure,
):
    store, session, root = listening
    if failure == "chunks":
        from Tests.Subscriptions.test_article_listening_store import prepare

        with store.db.transaction() as connection:
            connection.execute(
                "UPDATE subscription_items SET content=? WHERE id=1",
                ("z" * (128 * 1800 + 1),),
            )
        candidate = prepare(store, session.authority, session.voice)
        session = store.replace_session(candidate, expected_revision=session.revision)
        entry = session.foreground
    else:
        entry = session.queue[0]

    async def readiness():
        return "unsupported_platform" if failure == "platform" else None

    svc, transport = service()
    try:
        if failure == "conversion":
            monkeypatch.setattr(api().audio_stitch, "PYDUB_AVAILABLE", False)
        owner = api().ArticleListeningAudio(
            store, root, tts_service=svc, readiness=readiness
        )
        with pytest.raises(
            ValueError,
            match={
                "platform": "unsupported_platform",
                "conversion": "conversion_unavailable",
                "chunks": "chunk_limit",
            }[failure],
        ):
            await owner.prepare(entry.entry_id, expected_revision=session.revision)
        assert not transport.requests
        assert store.load_session().revision == session.revision
    finally:
        await svc.wait_closed()


@pytest.mark.asyncio
async def test_pocket_native_route_cannot_freeze_a_model_it_ignores():
    from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
    from tldw_chatbook.TTS.adapter_types import TTSProviderSpec, TTSProviderDescriptor
    from tldw_chatbook.TTS.TTS_Generation import TTSService

    adapter = Transport("openai")
    registry = TTSAdapterRegistry(
        specs=(
            TTSProviderSpec(
                descriptor=TTSProviderDescriptor(
                    provider_id="openai", display_name="Pocket", native=False
                ),
                factory=lambda _: adapter,
                initial_config={
                    "app_config": {
                        "app_tts": {"OPENAI_BASE_URL": "http://localhost:8000/tts"}
                    }
                },
                exclusive_reconfigure=False,
            ),
        ),
        aliases={},
    )
    svc = TTSService(
        registry,
        preferences_snapshot=_snapshot(
            provider_id="openai",
            model_id="tts-1",
            voice_mode="exact",
            voice_id="alloy",
        ),
    )
    try:
        with pytest.raises(ValueError, match="voice_unbound"):
            await api().freeze_listening_voice(tts_service=svc)
        assert not adapter.requests
    finally:
        await svc.wait_closed()


@native_effects
@pytest.mark.asyncio
async def test_pocket_route_detected_at_dispatch_never_receives_article(listening):  # noqa: F811
    store, session, root = listening
    svc, adapter = service()
    adapter.admitted_outbound_endpoint = lambda: "http://localhost:8000/tts"
    owner = api().ArticleListeningAudio(store, root, tts_service=svc, readiness=ready)
    try:
        with pytest.raises(ValueError):
            await owner.prepare(
                session.queue[0].entry_id, expected_revision=session.revision
            )
        assert not adapter.requests and store.load_session().queue[0].artifact is None
    finally:
        await svc.wait_closed()
