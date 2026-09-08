"""Compatibility facade adapting app TTS responses to the Audio sequencer."""

from __future__ import annotations

import asyncio
from typing import Any

from tldw_chatbook.Audio.voice_phrase_sequencer import (
    PhraseSpeechSequencer as _AudioPhraseSpeechSequencer,
)
from tldw_chatbook.Audio.voice_process_types import (
    NormalizedPcmError,
    NormalizedPcmStream,
)
from tldw_chatbook.TTS.pcm_stream import PcmStreamError, iter_normalized_pcm_frames
from tldw_chatbook.Utils.persistent_diagnostics import persist_event


def _persist_phrase_event(event: str, fields: dict[str, object]) -> None:
    persist_event("speculative_voice", event, **fields)


class _ResponseFrames:
    def __init__(self, response: Any, exhausted: asyncio.Event) -> None:
        self._response = response
        self._exhausted = exhausted
        self._iterator = self._iterate()

    def __aiter__(self):
        return self._iterator

    async def _iterate(self):
        try:
            channels = self._response.metadata.get("channels", 1)
            if type(channels) is not int:
                raise NormalizedPcmError("invalid_audio_stream")
            try:
                async for frame in iter_normalized_pcm_frames(
                    audio_format=self._response.audio_format,
                    sample_rate=self._response.sample_rate,
                    channels=channels,
                    byte_stream=self._response.byte_stream,
                ):
                    yield frame
            except PcmStreamError as error:
                raise NormalizedPcmError(error.code) from None
        finally:
            self._exhausted.set()

    async def aclose(self) -> None:
        try:
            await self._iterator.aclose()
        finally:
            self._exhausted.set()


class _NormalizedSynthesizer:
    def __init__(self, synthesizer: Any) -> None:
        self._synthesizer = synthesizer

    async def synthesize_hands_free(self, *, text: str) -> NormalizedPcmStream:
        response = await self._synthesizer.synthesize_hands_free(text=text)
        exhausted = asyncio.Event()

        async def cleanup() -> None:
            await exhausted.wait()
            await response.aclose()

        return NormalizedPcmStream(_ResponseFrames(response, exhausted), cleanup())


class PhraseSpeechSequencer(_AudioPhraseSpeechSequencer):
    """Preserve the parent API while the Audio owner consumes normalized PCM."""

    def __init__(
        self, *, synthesizer: Any, diagnostic_sink=None, **kwargs: Any
    ) -> None:
        if diagnostic_sink is None:
            diagnostic_sink = _persist_phrase_event
        super().__init__(
            synthesizer=_NormalizedSynthesizer(synthesizer),
            diagnostic_sink=diagnostic_sink,
            **kwargs,
        )


__all__ = ["PhraseSpeechSequencer"]
