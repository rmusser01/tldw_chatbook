"""The speech-model warm-up and download, run inside a live Textual app.

TASK-33621.21. With no model cached, Dictate sat on "Preparing speech model…"
forever. The first Hugging Face download built tqdm's write lock, which on
macOS starts CPython's resource tracker with ``sys.stderr.fileno()``; inside
Textual that is ``-1``, so the spawn raised ``ValueError: bad value(s) in
fds_to_keep`` and tqdm's class lock stayed held. parakeet-mlx hid the error
behind a local-path fallback ("[Errno 2] ... config.json"), the controller
called that "may be slow" and opened the microphone, and the streaming load
then blocked on the orphaned lock.

Every test here runs the warm-up through the real controller while a Textual
app holds stderr (asserted: ``fileno() == -1``), against a local stand-in for
the model host. Nothing in the download path is stubbed: the network is a
loopback HTTP server, ``huggingface_hub`` and tqdm are real, and the first
test loads through the real parakeet-mlx loader. Only the microphone is fake.

Two pieces of process-global state would otherwise let these pass by
accident, so each test starts from a first launch: tqdm's lock not yet built,
and a resource tracker that has not been started.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from textual.app import App

from Tests.Chat import model_download_support as _support
from tldw_chatbook.Chat import console_voice_input as cvi

#: Real controller and transcription stack read config: keep the collection
#: profile (the per-test redirect fails config admission), and allow the
#: loopback model host.
pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.loopback_network]

pytest.importorskip("huggingface_hub")

tqdm_std = _support.tqdm_std
#: Shared fixtures: a first launch (tqdm's lock unbuilt, no resource tracker)
#: and a loopback model host.
first_download_in_process = _support.first_download_in_process
model_host = _support.model_host

REPO_ID = "tldw-test/speech-model"
#: Long enough for a real local load, short enough that a hang fails the test.
ATTEMPT_TIMEOUT_SECONDS = 30.0


class _Microphone:
    """The dictation service's capture half. Its transcriber is the real one."""

    def __init__(self, transcriber) -> None:
        self._transcriber = transcriber
        self.start_calls = 0
        self.releases = 0

    @property
    def transcription_service(self):
        return self._transcriber

    def start_dictation(self, **_callbacks) -> bool:
        self.start_calls += 1
        return True

    def stop_dictation(self):
        return None

    def abandon(self) -> None:
        self.releases += 1


class _HubFetchingTranscriber:
    """A provider loader reduced to its download: one real `hf_hub_download`."""

    def __init__(self) -> None:
        self.downloaded: Path | None = None
        self.error: BaseException | None = None

    def transcribe_buffer(self, audio_data, sample_rate, **_kwargs):
        from huggingface_hub import hf_hub_download

        try:
            self.downloaded = Path(hf_hub_download(REPO_ID, "config.json"))
        except BaseException as exc:
            self.error = exc
            raise
        return {"text": ""}


class _Host(App):
    """Any running Textual app: it is the stderr capture that matters."""


def _controller(monkeypatch, provider: str, service: _Microphone):
    monkeypatch.setattr(cvi, "probe", lambda: cvi.Availability(ok=True))
    monkeypatch.setattr(
        cvi,
        "resolve",
        lambda: cvi.EffectiveConfig(
            provider=provider,
            model=None,
            language="en",
            configured_provider=provider,
            was_overridden=False,
        ),
    )
    monkeypatch.setattr(cvi, "warm_before_capture_enabled", lambda: True)
    monkeypatch.setattr(cvi, "model_prepare_stall_seconds", lambda: 20.0, raising=False)
    events: list = []
    controller = cvi.ConsoleVoiceInputController(
        emit=events.append,
        spawn=lambda thunk: thunk(),
        service_factory=lambda **_kwargs: service,
    )
    return controller, events


async def _attempt(controller) -> float:
    """Press Dictate once on a worker thread; return how long it took.

    A plain daemon thread, not `asyncio.to_thread`: a load that hangs must
    fail this test, not hang the event loop's executor shutdown.
    """
    started = time.monotonic()
    worker = threading.Thread(target=controller.start, daemon=True)
    worker.start()
    while worker.is_alive() and time.monotonic() - started < ATTEMPT_TIMEOUT_SECONDS:
        await asyncio.sleep(0.05)
    elapsed = time.monotonic() - started
    assert not worker.is_alive(), (
        f"the warm-up was still blocked after {elapsed:.0f} s -- "
        "'Preparing speech model…' would never end"
    )
    return elapsed


def _tqdm_lock_is_free() -> bool:
    """True when another thread can take tqdm's class lock right now."""
    lock = tqdm_std.TqdmDefaultWriteLock.th_lock
    if not lock.acquire(timeout=2):
        return False
    lock.release()
    return True


def _single_failure(events) -> str:
    failures = [e for e in events if isinstance(e, cvi.VoiceFailed)]
    assert len(failures) == 1, events
    return f"{failures[0].reason} {failures[0].remedy}"


@pytest.mark.skipif(
    sys.platform != "darwin"
    or importlib.util.find_spec("parakeet_mlx") is None
    or importlib.util.find_spec("mlx") is None,
    reason="the real parakeet-mlx loader needs Apple Silicon and parakeet-mlx",
)
async def test_parakeet_mlx_warm_up_inside_the_app_fails_cleanly_and_twice_alike(
    monkeypatch, model_host, first_download_in_process
):
    """The live defect, end to end, through the real parakeet-mlx loader.

    The host serves a download that completes but is not a usable model, so
    the load has to fail after the fetch. It must fail as one plain message,
    without opening the microphone, and the next press must behave the same.
    """
    from tldw_chatbook.Local_Ingestion.transcription_service import (
        TranscriptionService,
    )

    model_host.files = {"config.json": b"{}", "model.safetensors": b"\0" * 4096}
    async with _Host().run_test():
        assert sys.stderr.fileno() == -1, "Textual's stderr capture is not active"
        texts = []
        for _press in range(2):
            microphone = _Microphone(TranscriptionService())
            controller, events = _controller(monkeypatch, "parakeet-mlx", microphone)

            await _attempt(controller)

            assert controller.state == cvi.STATE_IDLE
            assert microphone.start_calls == 0, "capture opened after a failed load"
            texts.append(_single_failure(events))
            assert _tqdm_lock_is_free(), "the download left tqdm's lock held"

    assert texts[0] == texts[1]
    text = texts[0]
    assert "speech model" in text and "try again" in text.lower()
    assert "Errno" not in text and "/" not in text and "may be slow" not in text
    # The fetch really ran in the app: both files came from the model host.
    fetched = [line for line in model_host.requests if line.startswith("GET ")]
    assert any(line.endswith("/config.json") for line in fetched), fetched
    assert any(line.endswith("/model.safetensors") for line in fetched), fetched


async def test_a_hub_download_in_the_warm_up_completes_inside_the_app(
    monkeypatch, model_host, first_download_in_process
):
    """The download half on its own, wherever huggingface_hub is installed.

    A real `hf_hub_download` from the warm-up, under Textual's stderr, must
    deliver the file; then the capture starts. Before the fix it died on
    `fds_to_keep` inside tqdm.
    """
    body = json.dumps({"model": "tiny"}).encode()
    model_host.files = {"config.json": body}
    transcriber = _HubFetchingTranscriber()
    microphone = _Microphone(transcriber)
    controller, events = _controller(monkeypatch, "faster-whisper", microphone)

    async with _Host().run_test():
        assert sys.stderr.fileno() == -1, "Textual's stderr capture is not active"
        await _attempt(controller)
        assert transcriber.error is None, (
            f"the in-app download failed: {transcriber.error!r}"
        )
        assert transcriber.downloaded is not None
        assert transcriber.downloaded.read_bytes() == body
        assert [e for e in events if isinstance(e, cvi.VoiceFailed)] == []
        assert controller.state == cvi.STATE_LISTENING
        assert microphone.start_calls == 1
        assert _tqdm_lock_is_free()
