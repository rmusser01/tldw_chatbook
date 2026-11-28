"""Use native Server TTS jobs as the owner for Studio audio summaries."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import replace
from itertools import islice, product

from pydantic import ValidationError

from tldw_chatbook.tldw_api.audio_schemas import OpenAISpeechRequest
from tldw_chatbook.tldw_api.exceptions import (
    APIConnectionError,
    APIResponseError,
    TLDWAPIError,
)

from .studio_generation import NativeSubmissionUnconfirmed, StudioGeneratedOutput, spoken_script
from .studio_models import STUDIO_SPEECH_FORMATS, StudioSpeechChoice
from .studio_voice_preview import synthesize_voice_preview

_STOPPED = StudioGeneratedOutput("cancelled")
_UNCONFIRMED = (
    "Studio could not confirm whether Server Speech queued the audio summary. "
    "Check Server Jobs before starting a new generation."
)


def speech_artifact_path(output_id: int) -> str:
    return f"/api/v1/outputs/{output_id}/download"


def native_speech_ids(owner_id: str) -> tuple[int, int]:
    if not isinstance(owner_id, str) or not re.fullmatch(
        r"[1-9][0-9]{0,18}:[1-9][0-9]{0,18}", owner_id
    ):
        raise ValueError(
            "The workspace link has an invalid native speech job or artifact ID"
        )
    job_id, output_id = map(int, owner_id.split(":"))
    return job_id, output_id


class ServerStudioAudio:
    def __init__(self, scope, *, jobs=None) -> None:
        self.scope = scope
        self.jobs = {} if jobs is None else jobs

    async def choices(self) -> tuple[StudioSpeechChoice, ...]:
        try:
            health, catalog = await asyncio.gather(
                self.scope.get_tts_health(mode="server"),
                self.scope.list_tts_providers(mode="server"),
            )
        except (TLDWAPIError, ValidationError, ValueError):
            return ()
        if (
            health.get("status") != "healthy"
            or catalog.get("supports_explicit_backend") is not True
        ):
            return ()
        result = tuple(
            dict.fromkeys(islice(self._catalog_choices(catalog["providers"]), 150))
        )
        if not result:
            return ()
        try:
            probe = await self.scope.list_audio_speech_job_artifacts(
                mode="server", job_id=0
            )
        except (TLDWAPIError, ValidationError, ValueError):
            return ()
        return result if probe.get("job_id") == 0 else ()

    @staticmethod
    def _catalog_choices(providers):
        for backend, provider in providers.items():
            if not backend.startswith("gateway:") or not isinstance(provider, Mapping):
                continue
            discovery = provider.get("discovery") or {}
            mode = discovery.get("status")
            if discovery.get("stale") or mode not in {"fresh", "disabled"}:
                continue
            for model, capability in (provider.get("model_capabilities") or {}).items():
                if mode == "fresh" and model != provider.get("default_model"):
                    continue
                voices = capability.get("voices") or ()
                formats = [
                    fmt
                    for fmt in capability.get("formats") or ()
                    if fmt in STUDIO_SPEECH_FORMATS
                ]
                for voice, audio_format in product(voices, formats):
                    try:
                        yield StudioSpeechChoice(
                            backend,
                            model,
                            voice,
                            audio_format,
                            capability.get("supports_speed") is True,
                            min(5000, capability.get("max_input_characters", 5000)),
                        )
                    except ValueError:
                        continue

    def _record_job(self, key, job_id):
        self.jobs[key] = job_id
        for stale in tuple(self.jobs)[:-32]:
            del self.jobs[stale]

    @staticmethod
    def _artifact(row):
        output_id = row.get("output_id")
        return (
            row.get("type") == "tts_audio"
            and type(output_id) is int
            and output_id > 0
            and row.get("format") in STUDIO_SPEECH_FORMATS
            and isinstance(row.get("metadata"), Mapping)
            and row.get("download_url") == speech_artifact_path(output_id)
        )

    async def detail(self, ref, owner_id: str):
        job_id, output_id = native_speech_ids(owner_id)
        try:
            result = await self.scope.list_audio_speech_job_artifacts(
                mode="server", job_id=job_id
            )
        except APIResponseError as exc:
            if exc.status_code == 404:
                return None
            raise
        if result.get("job_id") != job_id:
            raise ValueError(
                "The Server returned speech artifacts from a different job"
            )
        return next(
            (
                row
                for row in result["artifacts"]
                if row.get("output_id") == output_id and self._artifact(row)
            ),
            None,
        )

    async def preview(self, choice: StudioSpeechChoice, speed: float) -> bytes:
        if choice not in await self.choices():
            raise ValueError("The selected Server voice or speech model is unavailable")
        return await synthesize_voice_preview(self.scope, "server", choice, speed)

    async def prepare(self, request, generated, cancel_event) -> StudioGeneratedOutput:
        selection = request.generation.speech
        if selection is None:
            raise ValueError("Choose an available Server speech route")
        key = (request.workspace, request.operation_id)
        if key not in self.jobs:
            if cancel_event.is_set():
                return _STOPPED
            if selection not in await self.choices():
                raise ValueError(
                    "The selected Server voice or speech model is no longer available"
                )
            payload = OpenAISpeechRequest(
                model=selection.model,
                input=spoken_script(generated.text, limit=selection.max_characters),
                voice=selection.voice,
                response_format=selection.format,
                backend=selection.provider,
                allow_fallback=False,
                normalization_options=None,
                stream=False,
                **(
                    {"speed": request.generation.speech_speed}
                    if selection.supports_speed
                    else {}
                ),
            )
            try:
                admission = await self.scope.create_audio_speech_job(
                    mode="server", request_data=payload
                )
            except (APIConnectionError, ValidationError) as exc:
                self._record_job(key, None)
                raise NativeSubmissionUnconfirmed(_UNCONFIRMED) from exc
            except APIResponseError as exc:
                if exc.status_code >= 500:
                    self._record_job(key, None)
                    raise NativeSubmissionUnconfirmed(_UNCONFIRMED) from exc
                raise
            except asyncio.CancelledError:
                self._record_job(key, None)
                raise
            self._record_job(key, admission.get("job_id"))
        job_id = self.jobs[key]
        if type(job_id) is not int or job_id < 1:
            raise NativeSubmissionUnconfirmed(_UNCONFIRMED)

        subject = request.workspace.principal_id
        expected_user = (
            subject.removeprefix("jwt-sub:")
            if re.fullmatch(r"jwt-sub:[1-9][0-9]{0,18}", subject)
            else None
        )
        for _ in range(300):
            if cancel_event.is_set():
                return _STOPPED
            job = await self.scope.get_audio_job(mode="server", job_id=job_id)
            if (
                job.get("id") != job_id
                or job.get("job_type") != "tts_longform"
                or (expected_user and job.get("owner_user_id") != expected_user)
            ):
                raise ValueError("The Server returned a different speech job")
            status = job.get("status")
            if status == "completed":
                result = await self.scope.list_audio_speech_job_artifacts(
                    mode="server", job_id=job_id
                )
                if result.get("job_id") != job_id:
                    raise ValueError(
                        "The Server returned speech artifacts from a different job"
                    )
                artifacts = [
                    row
                    for row in result["artifacts"]
                    if self._artifact(row)
                    and row["format"] == selection.format
                    and (
                        row["metadata"].get("provider"),
                        row["metadata"].get("model"),
                        row["metadata"].get("voice"),
                    )
                    == (selection.provider, selection.model, selection.voice)
                ]
                if len(artifacts) != 1:
                    raise ValueError(
                        "The Server did not publish one audio artifact for the selected speech route"
                    )
                return replace(
                    generated, native_owner=(f"{job_id}:{artifacts[0]['output_id']}", 1)
                )
            if status in {"failed", "cancelled", "quarantined", "dead_letter"}:
                raise ValueError(
                    "The Server speech job ended before publishing audio. Check Server Jobs."
                )
            try:
                await asyncio.wait_for(cancel_event.wait(), timeout=1)
            except TimeoutError:
                continue
        raise ValueError("The Server speech job is still running. Retry to reconnect.")
