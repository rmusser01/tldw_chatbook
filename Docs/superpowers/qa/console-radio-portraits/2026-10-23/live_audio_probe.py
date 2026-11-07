"""Opt-in real-device probe; copy to Tests/Audio/_radio_live_probe.py to run."""

import asyncio
import json
import math
import struct
import time
import wave
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from textual.app import App

from tldw_chatbook.Audio.speech_presentation import (
    SpeechPresentation,
    SpeechPresentationOwner,
)
from tldw_chatbook.Audio.streaming_sink import StreamingPcmSink, pump
from tldw_chatbook.Character_Chat.portrait_performance import (
    FRAME_ROLES,
    PerformanceSource,
    PortraitPerformanceRecord,
    normalize_performance_frames,
)
from tldw_chatbook.Widgets.Console.character_expression_avatar import (
    CharacterExpressionAvatar,
)

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.real_audio_device]
OUTPUT = Path("Docs/superpowers/qa/console-radio-portraits/2026-10-23")
RATE = 24000


def png(color):
    with Image.new("RGB", (32, 32), color) as image, BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["pcm", "wav", "external"])
async def test_real_local_device_portrait_timing(monkeypatch, route):
    import sounddevice as sd

    native = sd.OutputStream
    callbacks, paints, events = [], [], []
    streams = []
    submitted_blocks = []
    observer_offset = [None]
    original_publish = StreamingPcmSink._publish_presentation

    def observed_publish(current_sink):
        original_publish(current_sink)
        if (
            current_sink._observer is not None
            and current_sink._observer.offset is not None
        ):
            observer_offset[0] = current_sink._observer.offset

    monkeypatch.setattr(StreamingPcmSink, "_publish_presentation", observed_publish)

    def observed_stream(**options):
        callback = options["callback"]

        def observed(outdata, frames, timing, status):
            now = time.monotonic()
            try:
                return callback(outdata, frames, timing, status)
            finally:
                if len(callbacks) < 1000:
                    submitted_blocks.append(outdata.tobytes())
                    callbacks.append(
                        {
                            "at": now,
                            "current": timing.currentTime,
                            "dac": timing.outputBufferDacTime,
                            "frames": frames,
                            "peak": int(abs(outdata.astype("int32")).max()),
                            "status": str(status),
                        }
                    )

        finished = options["finished_callback"]

        def observed_finish():
            events.append({"at": time.monotonic(), "type": "NativeFinished"})
            finished()

        options["finished_callback"] = observed_finish
        options["callback"] = observed
        stream = native(**options)
        streams.append(stream)
        return stream

    monkeypatch.setattr(sd, "OutputStream", observed_stream)
    speech = SpeechPresentation(
        SpeechPresentationOwner("live-calibration", object(), "probe", 1, 7, None, None)
    )
    frames = normalize_performance_frames(
        {
            role: png(("red", "green", "blue")[i % 3])
            for i, role in enumerate(FRAME_ROLES)
        }
    )
    record = PortraitPerformanceRecord(
        PerformanceSource(7, "neutral", "0" * 64, "portrait", ()), 1, True, frames
    )

    class Probe(App):
        def compose(self):
            self.avatar = CharacterExpressionAvatar(
                frames[0].png,
                box=(20, 12),
                animate=True,
                is_current=lambda: True,
                monochrome=False,
                mode="pixels",
                id="portrait",
                performance=record,
                read_speech=speech.read,
            )
            yield self.avatar

        def record_terminal_flush(self):
            if (
                getattr(self, "avatar", None) is None
                or self.avatar.current_image is None
            ):
                return
            image = self.avatar.current_image
            color = image.getpixel((image.width // 2, image.height // 2))[:3]
            paints.append({"at": time.monotonic(), "color": list(color)})

    # Two audible periods separated by a nonzero, sub-gate background pause.
    values = []
    for seconds, amplitude in (
        (0.2, 0),
        (0.6, 4096),
        (0.35, 16),
        (0.6, 4096),
        (0.25, 0),
    ):
        values.extend(
            round(amplitude * math.sin(2 * math.pi * 440 * i / RATE))
            for i in range(round(seconds * RATE))
        )
    pcm = struct.pack("<" + str(len(values)) + "h", *values)
    with wave.open(str(OUTPUT / "calibration.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    sink = StreamingPcmSink(
        presentation=speech,
        on_event=lambda event: events.append(
            {"at": time.monotonic(), "type": type(event).__name__}
        ),
    )
    app = Probe()
    try:
        async with app.run_test(headless=False, size=(45, 18)):
            async with asyncio.timeout(10):
                while app.avatar.current_image is None:
                    await asyncio.sleep(0.02)
            original_flush = app._driver.flush

            def observed_flush():
                original_flush()
                app.record_terminal_flush()

            monkeypatch.setattr(app._driver, "flush", observed_flush)
            if route == "pcm":
                sink.open(RATE, 1)
                assert sink.state == "open", sink.fail_reason

                async def chunks():
                    for start in range(0, len(pcm), 4800):
                        yield pcm[start : start + 4800]
                        await asyncio.sleep(0.04)

                result = await pump(sink, chunks())
                assert result.outcome == "drained", result
                await asyncio.to_thread(sink.settle)
            elif route == "wav":
                from Tests.TTS.test_sentence_chunked_speech import _Handler, _service
                from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
                    TTSCompleteEvent,
                )

                service, requests = _service(
                    request_body=(OUTPUT / "calibration.wav").read_bytes()
                )
                handler = _Handler(service)
                try:
                    await handler._generate_tts(
                        "Known waveform artifact.",
                        "live-wav",
                        None,
                        presentation=speech,
                        on_finished=handler.finished.append,
                    )
                    complete = [
                        event
                        for event in handler.messages
                        if isinstance(event, TTSCompleteEvent)
                    ]
                    assert (
                        len(complete) == 1
                        and complete[0].error is None
                        and complete[0].audio_file is None
                    )
                    assert handler.finished == [True] and len(requests) == 1
                    assert not handler._audio_files
                    events.append(
                        {"at": time.monotonic(), "type": "WavHandlerReturned"}
                    )
                finally:
                    await service.close()
                    await service.wait_closed()
                    await handler.cleanup_tts_resources()
            else:
                from Tests.TTS.test_sentence_chunked_speech import _Handler
                from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
                    TTSPlaybackEvent,
                    TTSPlaybackLifecycle,
                )

                handler = _Handler(None)
                clip = handler._create_tts_artifact("wav")
                clip.write_bytes((OUTPUT / "calibration.wav").read_bytes())
                lifecycle = TTSPlaybackLifecycle(
                    message_id="external",
                    request_id=1,
                    validator=lambda: True,
                    callback=lambda state: events.append(
                        {"at": time.monotonic(), "type": state}
                    ),
                    presentation=speech,
                )
                handler._audio_files["external"] = clip
                handler._audio_file_owners["external"] = lifecycle
                try:
                    await handler.handle_tts_playback(
                        TTSPlaybackEvent(
                            "play", "external", playback_lifecycle=lifecycle
                        )
                    )
                    task = handler._active_file_playback_task
                    assert task is not None
                    async with asyncio.timeout(10):
                        while not task.done():
                            observed = speech.read()
                            if observed is not None:
                                assert (
                                    not observed.timing_valid
                                    and observed.mouth == "rest"
                                )
                            await asyncio.sleep(0.02)
                        await task
                    assert lifecycle.state == "stopped"
                    assert any(event["type"] == "playing" for event in events)
                finally:
                    await handler.handle_tts_playback(
                        TTSPlaybackEvent(
                            "stop", "external", playback_lifecycle=lifecycle
                        )
                    )
                    await handler.cleanup_tts_resources()
            latency = streams[-1].latency if streams else None
            await asyncio.sleep(0.15)
    finally:
        sink.stop()
        sink.settle()
        if submitted_blocks:
            with wave.open(
                str(OUTPUT / f"{route}-device-submitted.wav"), "wb"
            ) as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(RATE)
                output.writeframes(b"".join(submitted_blocks))
        (OUTPUT / f"native-audio-{route}-observations.json").write_text(
            json.dumps(
                {
                    "sample_rate": RATE,
                    "output_device": sd.query_devices(kind="output")["name"],
                    "route": route,
                    "driver_latency_seconds": locals().get("latency"),
                    "observer_clock_offset": observer_offset[0],
                    "callbacks": callbacks,
                    "terminal_writes": paints,
                    "events": events,
                    "physical_loopback": False,
                },
                indent=2,
            )
        )
    if route == "external":
        assert not callbacks and all(row["color"] == [255, 0, 0] for row in paints)
    else:
        assert pcm in b"".join(submitted_blocks), (
            "Device submission lost part of the waveform"
        )
        assert any(row["peak"] > 1000 for row in callbacks)
        assert any(row["color"] == [0, 0, 255] for row in paints), (
            "No automatically painted loud pose"
        )
