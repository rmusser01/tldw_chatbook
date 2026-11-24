"""Source-bound, on-demand preparation of qualified article WAVs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import struct
from threading import Lock
from typing import TYPE_CHECKING
import wave

from tldw_chatbook.Subscriptions.article_audio_assets import (
    ArticleAudioAssets,
    ArticleAudioError,
)
from tldw_chatbook.Subscriptions.article_listening_models import (
    ArticleAudio,
    FrozenVoice,
    GeneratedArticleAudio,
    ListeningConflict,
    VoiceDependency,
)
from tldw_chatbook.Subscriptions.article_speech import (
    ArticleSpeechError,
    article_chunks,
)
from tldw_chatbook.TTS import audio_stitch
from tldw_chatbook.TTS.audio_limits import MAX_BUFFERED_AUDIO_BYTES
from tldw_chatbook.TTS.controlled_wav_profile import (
    PADDING_FRAMES,
    RATE,
    controlled_wav_readiness,
)
from tldw_chatbook.TTS.default_profile_request_resolver import resolve_default_profile
from tldw_chatbook.TTS.effective_settings import (
    TTSDefaultProfileSelection,
    TTSSelectionOverrides,
)
from tldw_chatbook.TTS.legacy_catalogs import ELEVENLABS_MODELS
from tldw_chatbook.TTS.legacy_bridge import legacy_provider_outbound_endpoint
from tldw_chatbook.TTS.openai_compatible_config import (
    normalize_openai_compatible_endpoint,
)
from tldw_chatbook.TTS.pocket_tts_native import is_pocket_tts_native_url
from tldw_chatbook.TTS.request_admission import TTSProviderStartAuthorization

if TYPE_CHECKING:
    from tldw_chatbook.Subscriptions.article_listening_store import (
        ArticleListeningStore,
    )
    from tldw_chatbook.TTS.profile_service import TTSProfileService
    from tldw_chatbook.TTS.TTS_Generation import TTSService


def _digest(value) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _exact(provider, model, voice, fmt, speed, options):
    return TTSSelectionOverrides(
        provider_id=provider,
        model_mode="exact",
        model_id=model,
        voice_mode="server_default" if voice is None else "exact",
        voice_id=voice,
        response_format=fmt,
        speed=speed,
        provider_options=options,
    )


async def _bound_profile(identifier, service):
    try:
        resolution = await resolve_default_profile(
            text="Listening voice",
            default_profile_id=identifier,
            profile_service=service,
        )
    except Exception:
        pass
    else:
        request = resolution.request
        if request is not None and str(resolution.profile_id) == identifier:
            ref = resolution.reference
            selection = _exact(
                request.provider_id,
                request.model_id,
                request.voice,
                request.response_format,
                request.speed,
                request.options,
            )
            profile = TTSDefaultProfileSelection(
                selection=selection,
                repository_generation=resolution.repository_generation,
                profile_id=resolution.profile_id,
                profile_revision=resolution.profile_revision,
                reference=ref,
            )
            revision = str(profile.profile_revision)
            dependencies = []
            ref_binding = None
            if ref is not None:
                recipe = ref.recipe_requirement
                if recipe is None:
                    raise ArticleSpeechError("listening_voice_unbound")
                dependencies.append(
                    VoiceDependency(
                        "reference", str(ref.summary.reference_id), revision, ref.sha256
                    )
                )
                ref_binding = (
                    str(ref.summary.reference_id),
                    ref.sha256,
                    sha256(ref.reference_text.encode()).hexdigest(),
                    asdict(recipe),
                )
            projection = (
                "listening-profile-v1",
                identifier,
                revision,
                request.provider_id,
                request.model_id,
                request.voice,
                request.response_format,
                request.speed,
                dict(request.options),
                ref_binding,
            )
            dependencies.insert(
                0, VoiceDependency("profile", identifier, revision, _digest(projection))
            )
            return profile, tuple(dependencies)
    raise ArticleSpeechError("listening_voice_changed")


def _official_voice(model, voice):
    # These are the identities the official backend passes through verbatim.
    return model in ("tts-1", "tts-1-hd") and voice in (
        "alloy",
        "echo",
        "fable",
        "onyx",
        "nova",
        "shimmer",
    )


async def _voice_projection(
    service, provider, model, voice, options, *, reference, freeze
):
    """Limit durable values to exact identities the selected backend can reproduce."""
    if provider == "audio_cpp":
        if options or (voice is None and reference is None):
            raise ArticleSpeechError("listening_voice_unbound")
    elif provider == "openai":
        config = await service.registry.provider_configuration_snapshot(provider)
        endpoint = legacy_provider_outbound_endpoint(
            provider, config.applied_config.get("app_config", {})
        )
        official = normalize_openai_compatible_endpoint(endpoint).official
        if (
            options
            or is_pocket_tts_native_url(endpoint)
            or official
            and not _official_voice(model, voice)
        ):
            raise ArticleSpeechError("listening_voice_unbound")
    elif provider == "elevenlabs":
        from tldw_chatbook.TTS.backends.elevenlabs import (
            get_elevenlabs_voice_id,
            voice_settings,
        )

        if not voice or model not in ELEVENLABS_MODELS:
            raise ArticleSpeechError("listening_voice_unbound")
        voice = get_elevenlabs_voice_id(voice)
        if len(voice) < 15:
            raise ArticleSpeechError("listening_voice_unbound")
        if freeze:
            config = await service.registry.provider_configuration_snapshot(provider)
            backend = config.applied_config["app_config"]
            effective = {}
            for section in ("global_tts_settings", "app_tts", "elevenlabs_" + model):
                effective.update(backend.get(section, {}))
            options = {**voice_settings(effective), **options}
        if set(options) != {
            "stability",
            "similarity_boost",
            "style",
            "use_speaker_boost",
        }:
            raise ArticleSpeechError("listening_voice_unbound")
    else:
        # These adapters can silently select a mutable backend default or loose
        # file/profile. They do not expose a reconstructible identity for it.
        raise ArticleSpeechError("listening_voice_unbound")
    return voice, tuple(sorted(options.items()))


async def freeze_listening_voice(
    *,
    tts_service: TTSService,
    default_profile_id: str | None = None,
    profile_service: TTSProfileService | None = None,
) -> FrozenVoice:
    """Capture the app-wide selection without persisting live provider identity or secrets."""
    try:
        profile, deps = (
            await _bound_profile(default_profile_id, profile_service)
            if default_profile_id
            else (None, ())
        )
        result = await tts_service.resolve_effective_voice(default_profile=profile)
        _require_reconstructible(result.provider_id, result.speed)
        voice, options = await _voice_projection(
            tts_service,
            result.provider_id,
            result.model_id,
            result.voice_id,
            result.provider_options,
            reference=profile.reference if profile else None,
            freeze=True,
        )
        return FrozenVoice(
            result.provider_id,
            result.model_id,
            voice,
            "pcm" if result.provider_id in {"elevenlabs", "kokoro"} else "wav",
            result.speed,
            deps,
            options,
        )
    except ArticleSpeechError:
        raise
    except Exception:
        pass
    raise ArticleSpeechError("listening_voice_unavailable")


async def resolve_listening_voice(
    frozen: FrozenVoice,
    *,
    tts_service: TTSService,
    profile_service: TTSProfileService | None = None,
) -> dict:
    """Reacquire the saved profile's current authorized reference and admission layers."""
    try:
        _require_reconstructible(frozen.provider, frozen.speed)
        dependencies = frozen.dependencies
        profile = None
        if dependencies:
            if dependencies[0].kind != "profile":
                raise ArticleSpeechError("listening_voice_changed")
            profile, live = await _bound_profile(
                dependencies[0].identifier, profile_service
            )
            if dependencies != live:
                raise ArticleSpeechError("listening_voice_changed")
        if frozen.format != (
            "pcm" if frozen.provider in {"elevenlabs", "kokoro"} else "wav"
        ):
            raise ArticleSpeechError("listening_voice_unbound")
        voice, options = await _voice_projection(
            tts_service,
            frozen.provider,
            frozen.model,
            frozen.voice,
            dict(frozen.options),
            reference=profile.reference if profile else None,
            freeze=False,
        )
        if (voice, options) != (frozen.voice, frozen.options):
            raise ArticleSpeechError("listening_voice_changed")
        if profile:
            if profile.reference is not None:
                await tts_service.resolve_effective_voice(default_profile=profile)
            selected = profile.selection
            original_voice = selected.voice_id
            if selected.provider_id == "elevenlabs":
                from tldw_chatbook.TTS.backends.elevenlabs import (
                    get_elevenlabs_voice_id,
                )

                original_voice = get_elevenlabs_voice_id(original_voice)
            if (
                selected.provider_id,
                selected.model_id,
                original_voice,
                selected.speed,
            ) != (frozen.provider, frozen.model, frozen.voice, frozen.speed):
                raise ArticleSpeechError("listening_voice_changed")
            return {
                "default_profile": profile,
                "explicit": TTSSelectionOverrides(
                    voice_mode=selected.voice_mode,
                    voice_id=voice,
                    response_format=frozen.format,
                    provider_options=dict(options),
                ),
            }
        return {
            "explicit": _exact(
                frozen.provider,
                frozen.model,
                voice,
                frozen.format,
                frozen.speed,
                dict(options),
            )
        }
    except ArticleSpeechError:
        raise
    except Exception:
        pass
    raise ArticleSpeechError("listening_voice_unavailable")


class ArticleWavAssembly:
    """Accumulate complete normalized PCM chunks within one aggregate byte budget."""

    def __init__(self, *, limit: int = MAX_BUFFERED_AUDIO_BYTES) -> None:
        if type(limit) is not int or not 0 < limit <= MAX_BUFFERED_AUDIO_BYTES:
            raise ValueError("article_audio_limit")
        self.limit = limit
        self.parts: list[bytes] = []
        self.size = 0

    def _budget(self, additional: int) -> None:
        if self.size + additional > self.limit:
            raise ArticleSpeechError("article_audio_limit")

    async def append(
        self,
        stream: AsyncIterator[bytes],
        *,
        audio_format: str,
        check: Callable[[], None] | None = None,
    ) -> None:
        """Charge incoming and growing buffer storage before copying each piece."""
        with BytesIO() as encoded:
            async for piece in stream:
                if check is not None:
                    check()
                if type(piece) is not bytes:
                    raise ArticleSpeechError("invalid_article_audio")
                if piece:
                    # Includes old and new allocation when the buffer grows.
                    self._budget(3 * (encoded.tell() + len(piece)))
                    encoded.write(piece)
                del piece
            normalized = await asyncio.to_thread(self._normalize, encoded, audio_format)
        self.parts.append(normalized)
        self.size += len(normalized)

    def _normalize(self, encoded: BytesIO, fmt: str) -> bytes:
        nbytes = encoded.tell()
        if not nbytes:
            raise ArticleSpeechError("invalid_article_audio")
        encoded.seek(0)
        if fmt == "wav":
            with wave.open(encoded, "rb") as reader:
                channels, width, rate, frames, compression, _ = reader.getparams()
                if compression != "NONE":
                    raise ArticleSpeechError("invalid_article_audio")
                self._conversion_budget(nbytes, frames, width, rate, channels)
                raw = reader.readframes(frames + 1)
        elif fmt == "pcm":
            channels, width, rate, frames = 1, 2, 24000, nbytes // 2
            self._conversion_budget(nbytes, frames, width, rate, channels)
            raw = encoded.read()
        else:
            raise ArticleSpeechError("invalid_article_audio")
        if len(raw) != frames * channels * width:
            raise ArticleSpeechError("invalid_article_audio")
        if width == 1:
            from pydub.utils import audioop

            raw = audioop.bias(raw, 1, -128)
        segment = audio_stitch.AudioSegment(
            data=raw, sample_width=width, channels=channels, frame_rate=rate
        )
        normalized = segment.set_sample_width(2).set_frame_rate(RATE).set_channels(2)
        return normalized.raw_data

    def _conversion_budget(self, encoded, frames, width, rate, channels):
        if (
            width not in (1, 2, 3, 4)
            or channels not in (1, 2)
            or frames < 1
            or rate < 1
        ):
            raise ArticleSpeechError("invalid_article_audio")
        if frames * channels * width > encoded:
            raise ArticleSpeechError("invalid_article_audio")
        out_frames = (frames * RATE + rate - 1) // rate
        # Width conversion (including pydub 24->32), ratecv's transient output
        # and stereo expansion can each require a new native and Python buffer.
        self._budget(2 * encoded + 4 * (frames * channels * 4 + out_frames * 4))

    def finish(self) -> GeneratedArticleAudio:
        """Join one WAV with the qualified tail, reserving both copies first."""
        if not self.size:
            raise ArticleSpeechError("invalid_article_audio")
        padding = 4 * PADDING_FRAMES
        size = self.size + padding
        self._budget(44 + size + padding + 44)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            2,
            RATE,
            RATE * 4,
            4,
            16,
            b"data",
            size,
        )
        wav = b"".join((header, *self.parts, bytes(padding)))
        content_frames = self.size // 4
        self.parts.clear()
        self.size = 0
        return GeneratedArticleAudio(wav, sha256(wav).hexdigest(), content_frames)


class _ArticleDecision:
    """Order revocation against this entry's final database attachment decision."""

    def __init__(self):
        self._lock = Lock()
        self._revoked = False
        self._committing = False

    def revoke(self) -> bool:
        with self._lock:
            if self._committing:
                return False
            self._revoked = True
            return True

    def check(self) -> None:
        with self._lock:
            if self._revoked:
                raise ArticleSpeechError("article_audio_retired")

    def commit(self) -> None:
        # The exact claim is validated inside the store's IMMEDIATE transaction.
        # Once selected, its actual commit success or failure is the caller's outcome.
        with self._lock:
            if self._revoked:
                raise ArticleSpeechError("article_audio_retired")
            self._committing = True


def _require_reconstructible(provider: str, speed: float) -> None:
    if provider == "kokoro":
        raise ArticleSpeechError("article_kokoro_engine_unbound")
    if provider == "elevenlabs" and speed != 1:
        raise ArticleSpeechError("article_speed_unavailable")


class ArticleListeningAudio:
    """Prepare one source-owned artifact at a time for this profile; never starts output."""

    def __init__(
        self,
        store: ArticleListeningStore,
        profile_data_root: Path,
        *,
        tts_service: TTSService | None,
        profile_service: TTSProfileService | None = None,
        readiness: Callable[[], Awaitable[str | None]] = controlled_wav_readiness,
    ) -> None:
        self.store = store
        self.root = profile_data_root
        self.tts = tts_service
        self.profiles = profile_service
        self.readiness = readiness
        self._task = None
        self._decision = None
        self._unretired = None

    def revoke(self) -> None:
        """Revoke the active observer's permission immediately; workers retain custody."""
        if self._decision is not None:
            self._decision.revoke()

    async def settle(self) -> None:
        """Wait for the current response, workers, and reservation cleanup to retire."""
        task = self._task
        if task is not None:
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))
        if self._unretired is not None:
            raise ArticleSpeechError("article_retirement_unconfirmed")

    async def prepare(
        self, entry_id: str, *, expected_revision: int, regenerate: bool = False
    ) -> ArticleAudio:
        """Verify retained audio first, otherwise synthesize the exact saved entry."""
        if self._unretired is not None:
            raise ArticleSpeechError("article_retirement_unconfirmed")
        if self._task is not None and not self._task.done():
            raise ArticleSpeechError("article_audio_busy")
        decision = _ArticleDecision()
        self._decision = decision
        task = asyncio.create_task(
            self._prepare(entry_id, expected_revision, regenerate, decision)
        )
        self._task = task
        # An abandoned observer must never leave an unobserved exception.
        task.add_done_callback(self._settled)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled() or decision.revoke():
                    raise
                # Attachment already won. Deliver its real database result even
                # if the observer was cancelled while SQLite was committing.

    def _settled(self, task):
        if not task.cancelled():
            task.exception()
        if self._task is task:
            self._task = None

    async def _prepare(self, entry_id, revision, regenerate, decision):
        store = self.store
        authority = store._authority()

        def check():
            decision.check()
            store._authority(authority)

        def current():
            check()
            return store.current_entry(
                entry_id, expected_revision=revision, authority=authority
            )

        async def authorize(provider, endpoint):
            check()
            await asyncio.to_thread(store.verify_pending_audio, claim)
            check()
            if provider != entry.voice.provider:
                return False
            if provider == "openai":
                if is_pocket_tts_native_url(endpoint):
                    return False
                if normalize_openai_compatible_endpoint(endpoint).official:
                    return _official_voice(entry.voice.model, entry.voice.voice)
            return True

        def authorize_commit():
            check()
            if store.db.listening_image != image:
                raise ListeningConflict("listening_database_changed")
            decision.commit()

        # This guard runs again inside the IMMEDIATE attachment transaction,
        # after validating the exact minted claim and before commit.
        assets = ArticleAudioAssets(
            store, profile_data_root=self.root, authorize_commit=authorize_commit
        )
        claim = None
        try:
            entry, image = await asyncio.to_thread(current)
            if type(regenerate) is not bool:
                raise ArticleSpeechError("invalid_regeneration_intent")
            missing_attachment = False
            if entry.artifact is not None:
                try:
                    artifact = await asyncio.to_thread(
                        assets.read_verified, entry.artifact
                    )
                except ArticleAudioError as error:
                    if str(error) not in {
                        "article_audio_missing",
                        "article_audio_corrupt",
                    }:
                        raise
                    missing_attachment = True
                    if not regenerate:
                        raise
                    if entry.availability == "ready":
                        # Marking a referenced ready generation missing advances
                        # its session once. Accept only that exact transition.
                        revision += 1
                        updated, current_image = await asyncio.to_thread(current)
                        if current_image != image or updated != replace(
                            entry, availability="missing"
                        ):
                            raise ListeningConflict("listening_entry_changed") from None
                        entry = updated
                else:
                    if (entry, image) != await asyncio.to_thread(current):
                        raise ListeningConflict("listening_entry_changed")
                    decision.commit()
                    return artifact
            artifact = await asyncio.to_thread(
                assets.reuse,
                entry_id,
                expected_revision=revision,
                replace_missing=missing_attachment,
            )
            if artifact is not None:
                return artifact
            _require_reconstructible(entry.voice.provider, entry.voice.speed)
            if entry.voice.provider == "audio_cpp":
                # This adapter buffers the entire WAV before the caller can
                # inspect it; its independent producer cap exceeds this budget.
                raise ArticleSpeechError("article_native_budget_unavailable")
            chunks = await asyncio.to_thread(article_chunks, entry.text)
            if not audio_stitch.PYDUB_AVAILABLE:
                raise ArticleSpeechError("article_conversion_unavailable")
            if reason := await self.readiness():
                raise ArticleSpeechError(reason)
            check()
            claim = await asyncio.to_thread(
                store.reserve_audio,
                entry_id,
                expected_revision=revision,
                regenerate=regenerate,
            )
            if claim.image != image:
                raise ListeningConflict("listening_database_changed")
            assembly = ArticleWavAssembly()
            for chunk in chunks:
                args = await resolve_listening_voice(
                    entry.voice, tts_service=self.tts, profile_service=self.profiles
                )
                if (entry, image) != await asyncio.to_thread(current):
                    raise ListeningConflict("listening_entry_changed")
                check()
                response, selected = await self.tts.synthesize_effective(
                    text=chunk,
                    admission_authorizer=TTSProviderStartAuthorization(authorize),
                    **args,
                )
                try:
                    check()
                    voice = entry.voice
                    if (
                        selected.provider_id,
                        selected.model_id,
                        selected.voice_id,
                        selected.speed,
                        selected.response_format,
                        dict(selected.provider_options),
                    ) != (
                        voice.provider,
                        voice.model,
                        voice.voice,
                        voice.speed,
                        voice.format,
                        dict(voice.options),
                    ) or (
                        response.provider_id,
                        response.model_id,
                        response.audio_format,
                    ) != (voice.provider, voice.model, voice.format):
                        raise ArticleSpeechError("listening_voice_changed")
                    if voice.format == "pcm" and (
                        response.sample_rate != 24000
                        or response.metadata.get("channels") != 1
                        or response.metadata.get("sample_encoding") != "pcm_s16le"
                    ):
                        raise ArticleSpeechError("invalid_article_audio")
                    await assembly.append(
                        response.byte_stream,
                        audio_format=response.audio_format,
                        check=check,
                    )
                finally:
                    try:
                        await response.aclose()
                    except BaseException:
                        self._unretired = response
                    if self._unretired is not None:
                        raise ArticleSpeechError(
                            "article_retirement_unconfirmed"
                        ) from None
            candidate = await asyncio.to_thread(assembly.finish)
            await resolve_listening_voice(
                entry.voice, tts_service=self.tts, profile_service=self.profiles
            )
            check()
            producer = ArticleAudioAssets(
                store,
                profile_data_root=self.root,
                reservation=claim,
                authorize_commit=authorize_commit,
            )
            artifact = await asyncio.to_thread(
                producer.publish,
                candidate,
                source_item_id=entry.item_id,
                generation=claim.generation,
            )
            claim = None
            return artifact
        except (ArticleSpeechError, ListeningConflict, ArticleAudioError):
            raise
        except Exception:
            pass
        finally:
            if claim is not None:
                try:
                    await asyncio.to_thread(store.cancel_audio, claim)
                except ListeningConflict:
                    pass
                except Exception:
                    self._unretired = self._unretired or claim
                    raise ArticleSpeechError("article_retirement_unconfirmed") from None
        raise ArticleSpeechError("article_audio_unavailable")
