"""Qualify on-device voices and link their existing Local TTS history."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from itertools import islice, product

from tldw_chatbook.tldw_api.audio_schemas import OpenAISpeechRequest

from .studio_generation import StudioGeneratedOutput, spoken_script
from .studio_models import STUDIO_SPEECH_FORMATS, StudioSpeechChoice


class LocalStudioAudio:
    def __init__(self, scope) -> None:
        self.scope = scope

    @property
    def persistent_history(self) -> bool:
        return bool(
            self.scope and getattr(self.scope.local_service, "history_store_path", None)
        )

    async def choices(self) -> tuple[StudioSpeechChoice, ...]:
        if not self.persistent_history:
            return ()
        from tldw_chatbook.TTS import get_tts_service

        try:
            service = await get_tts_service()
            observation = service.latest_native_capability_observation("audio_cpp")
            if observation is None:
                return ()
            snapshot = observation.snapshot
            if snapshot.configuration_revision != service.configuration_revision(
                "audio_cpp"
            ):
                return ()
            catalog = snapshot.catalog
        except (OSError, RuntimeError, ValueError):
            return ()
        if (
            catalog is None
            or catalog.provider_id != "audio_cpp"
            or catalog.health.state != "available"
            or not catalog.health.fresh
            or catalog.approximate
        ):
            return ()

        def trusted_voices(model):
            voices = snapshot.voice_results.get(model.model_id)
            return voices.voices if voices and voices.state == "complete" else ()

        entries = (
            StudioSpeechChoice(
                catalog.provider_id, model.model_id, voice, format, model.supports_speed
            )
            for model in catalog.models
            if "tts" in model.speech_capabilities
            for voice, format in product(trusted_voices(model), model.formats)
            if format in STUDIO_SPEECH_FORMATS and format != "ogg"
        )
        return tuple(dict.fromkeys(islice(entries, 150)))

    async def detail(self, owner_id: str, *, include_content: bool = False):
        if not self.persistent_history:
            return None
        try:
            number = int(owner_id)
            if number < 1 or str(number) != owner_id:
                raise ValueError
            row = await self.scope.get_tts_history_entry(
                mode="local",
                history_id=number,
                include_content=include_content,
            )
        except (KeyError, ValueError):
            return None
        if row.get("id") != number or row.get("size_bytes", 0) < 1:
            return None
        return {**row, "version": 1, "format": row["response_format"]}

    async def prepare(self, request, generated, cancel_event) -> StudioGeneratedOutput:
        choice = request.generation.speech
        if choice is None or choice not in await self.choices():
            raise ValueError("The selected on-device voice or model is unavailable")
        if cancel_event.is_set():
            return StudioGeneratedOutput("cancelled")
        payload = OpenAISpeechRequest(
            backend=choice.provider,
            model=choice.model,
            voice=choice.voice,
            input=spoken_script(generated.text, limit=choice.max_characters),
            response_format=choice.format,
            speed=request.generation.speech_speed,
            stream=False,
        )
        native = asyncio.create_task(
            self.scope.create_audio_speech(mode="local", request_data=payload)
        )
        stop = asyncio.create_task(cancel_event.wait())
        try:
            await asyncio.wait((native, stop), return_when=asyncio.FIRST_COMPLETED)
            if not native.done():
                return StudioGeneratedOutput("cancelled")
            receipt = native.result()
        finally:
            native.cancel()
            stop.cancel()
            await asyncio.gather(native, stop, return_exceptions=True)
        history_id = receipt.get("history_id")
        if type(history_id) is not int or history_id < 1:
            raise ValueError("Local speech returned no saved history ID")
        saved = await self.detail(str(history_id))
        if not saved or (
            saved["provider"],
            saved["model"],
            saved["voice"],
            saved["format"],
        ) != (
            choice.provider,
            choice.model,
            choice.voice,
            choice.format,
        ):
            raise ValueError("Local speech did not save the selected voice and model")
        return replace(generated, native_owner=(str(history_id), 1))
