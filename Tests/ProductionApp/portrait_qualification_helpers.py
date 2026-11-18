"""External image/TTS/device boundaries for installed portrait journeys.

Product routing, request validation, compositing, speech ownership, PCM analysis and
avatar clocks remain real. The automatic stream supplies timed device callbacks;
it never publishes a speech snapshot or calls an avatar tick.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from array import array
from types import SimpleNamespace

from Tests.Audio.test_streaming_sink import FakeStream


class AutomaticStream(FakeStream):
    """Drive the existing native stream seam from actual submitted PCM bytes."""

    def __init__(self, *, samplerate, channels, blocksize, callback, finished_callback):
        super().__init__(callback, samplerate, channels, blocksize, finished_callback)
        self.rate = samplerate
        self._halt = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        super().start()
        self._thread.start()

    def _run(self):
        while not self._halt.is_set() and not self.finished and not self.aborted:
            now = time.monotonic()
            self.tick(
                time_info=SimpleNamespace(
                    currentTime=now, outputBufferDacTime=now + 0.02
                )
            )
            self._halt.wait(self.blocksize / self.rate)

    def abort(self):
        super().abort()
        self._halt.set()

    def close(self):
        super().close()
        self._halt.set()


class QualificationAudio:
    """Fake only synthesis transport and device callbacks, retaining the real sink."""

    def __init__(self):
        self.streams = []
        self.synthesized = []
        self.responses = []
        self.formats = []
        self._permit = asyncio.Event()

    def install(self):
        from tldw_chatbook.Audio.streaming_sink import StreamingPcmSink
        from tldw_chatbook.Event_Handlers.TTS_Events import tts_events
        from tldw_chatbook.TTS.backends.openai import OpenAITTSBackend

        def factory(**kwargs):
            stream = AutomaticStream(**kwargs)
            self.streams.append(stream)
            return stream

        def sink(**kwargs):
            return StreamingPcmSink(stream_factory=factory, **kwargs)

        async def synthesize(_backend, request):
            self.formats.append((request.response_format, _backend.is_custom_endpoint))
            await self._permit.wait()
            import wave
            from io import BytesIO

            frequency = 220 + len(self.synthesized) * 20
            samples = array("h")
            for amplitude in (0, 12000, 0, 1800, 12000, 0):
                samples.extend(
                    int(amplitude * math.sin(2 * math.pi * frequency * index / 24000))
                    for index in range(9600)
                )
            data = samples.tobytes()
            self.synthesized.append((request.input, data))
            if request.response_format == "wav":
                with BytesIO() as output:
                    with wave.open(output, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(24000)
                        wav.writeframes(data)
                    data = output.getvalue()
            else:
                assert request.response_format == "pcm"
            for start in range(0, len(data), 4800):
                yield data[start : start + 4800]
                await asyncio.sleep(0.005)

        validate = tts_events.TTSEventHandler._validate_speech_response

        def observed(handler, response, *args):
            self.responses.append(
                (
                    response.audio_format,
                    response.sample_rate,
                    tts_events.sink_available(),
                )
            )
            return validate(handler, response, *args)

        tts_events.TTSEventHandler._validate_speech_response = observed
        OpenAITTSBackend.generate_speech_stream = synthesize
        tts_events.StreamingPcmSink = sink
        # The supplied stream factory is the available external device in this harness.
        tts_events.sink_available = lambda: True

    def allow_synthesis(self):
        """Release simulated transport only after the real preview is visible."""
        self._permit.set()

    def hold_synthesis(self):
        self._permit.clear()

    async def settle(self):
        for stream in self.streams:
            await asyncio.to_thread(stream._thread.join, 3)
            assert not stream._thread.is_alive() and stream.closed
        assert self.synthesized
        for _, data in self.synthesized:
            assert any(data in b"".join(stream.out) for stream in self.streams)


TTS_CONFIG = """
[app_tts]
default_provider="openai"
default_model_mode="exact"
default_model="tts-1"
default_voice_mode="exact"
default_voice="alloy"
default_format="wav"
OPENAI_BASE_URL="https://api.openai.com/v1/audio/speech"
OPENAI_AUTH_MODE="api_key"
OPENAI_API_KEY="fake-pcm-transport"
[api_settings.openai]
api_key="fake-pcm-transport"
"""


class QualificationImages:
    """Record actual validated requests, faking only external image transport in CI."""

    def __init__(self, *, live=False):
        self.live = live
        self.calls = []
        self.results = []
        self.fail = set()
        self._delegate = None

    def install(self):
        from dataclasses import replace

        from tldw_chatbook.Image_Generation import adapter_registry, config
        from tldw_chatbook.Image_Generation.capabilities import (
            resolve_backend_reference_image_capability,
        )

        cfg = config.get_image_generation_config()
        if self.live:
            if (
                cfg.default_backend not in cfg.enabled_backends
                or not resolve_backend_reference_image_capability(
                    cfg.default_backend, config=cfg
                ).supported
            ):
                raise ValueError(
                    "Live qualification needs a configured enabled reference-image backend; no request sent."
                )
            registry = adapter_registry.get_registry()
            self._delegate = registry.get_adapter(cfg.default_backend)
            if self._delegate is None:
                raise ValueError(
                    "Configured live adapter unavailable; no request sent."
                )
        else:
            cfg = replace(
                cfg,
                default_backend="gemini",
                enabled_backends=["gemini"],
                gemini_image_default_model="installed-fixture",
                gemini_image_api_key="fake-external-boundary",
            )
            config._config_cache = cfg
            registry = adapter_registry.ImageAdapterRegistry()
            adapter_registry._registry = registry
        registry._adapters[cfg.default_backend] = self
        return cfg

    def generate(self, request):
        import hashlib
        from io import BytesIO

        from PIL import Image

        from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult

        index = len(self.calls)
        self.calls.append(request)
        if self.live:
            import json
            from pathlib import Path

            artifacts = Path.home() / "portrait-artifacts"
            artifacts.mkdir(exist_ok=True)
            (artifacts / "requests.json").write_text(
                json.dumps(
                    [
                        {
                            "index": i,
                            "backend": call.backend,
                            "model": call.model,
                            "width": call.width,
                            "height": call.height,
                            "reference_sha256": call.reference_image.file_id,
                        }
                        for i, call in enumerate(self.calls)
                    ],
                    indent=2,
                )
            )
        if index in self.fail:
            assert not self.live
            raise TimeoutError("deterministic uncertain image response")
        if self._delegate:
            result = self._delegate.generate(request)
        else:
            color = (
                (index * 23 + 30) % 255,
                (index * 37 + 60) % 255,
                (index * 53 + 90) % 255,
            )
            with (
                Image.new("RGB", (request.width, request.height), color) as image,
                BytesIO() as output,
            ):
                image.save(output, format="PNG")
                data = output.getvalue()
            result = ImageGenResult(data, "image/png", len(data))
        if self.live:
            from pathlib import Path

            artifacts = Path.home() / "portrait-artifacts"
            artifacts.mkdir(exist_ok=True)
            suffix = {
                "image/png": "png",
                "image/jpeg": "jpg",
                "image/webp": "webp",
            }.get(result.content_type, "bin")
            (artifacts / f"response-{index}.{suffix}").write_bytes(result.content)
        self.results.append(
            {
                "index": index,
                "sha256": hashlib.sha256(result.content).hexdigest(),
                "bytes": len(result.content),
            }
        )
        return result
