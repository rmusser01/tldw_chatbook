"""Kokoro asset downloads must be bounded, off-loop, and atomic (TASK-19560).

Three defects sat in one code path:

* `requests.get(url, stream=True)` carried **no timeout**, so a half-open
  connection hung forever;
* the call ran inline inside `async def`, so a several-hundred-megabyte
  transfer blocked the event loop -- the whole TUI -- for its duration;
* it streamed straight to the final path, so an interrupted download left a
  truncated file that the next run's `os.path.exists()` check accepted as a
  complete model.

These tests exercise the real helper against a fake `requests`, so they assert
the behaviour rather than the shape of the code.
"""

from __future__ import annotations

import asyncio
import os
import threading

import pytest

from tldw_chatbook.TTS.backends import kokoro as kokoro_module


class _FakeResponse:
    def __init__(self, chunks, *, headers=None, raise_on_chunk=None):
        self._chunks = chunks
        self.headers = headers or {}
        self._raise_on_chunk = raise_on_chunk
        self.raised_for_status = False

    def raise_for_status(self):
        self.raised_for_status = True

    def iter_content(self, chunk_size=8192):
        for index, chunk in enumerate(self._chunks):
            if self._raise_on_chunk is not None and index == self._raise_on_chunk:
                raise OSError("connection reset mid-transfer")
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_requests(monkeypatch):
    """Replace `requests.get`, recording the kwargs it was called with."""
    calls: list[dict] = []
    response_holder: dict = {}

    def fake_get(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return response_holder["response"]

    monkeypatch.setattr(kokoro_module.requests, "get", fake_get)
    return calls, response_holder


def test_download_sends_connect_and_read_timeouts(tmp_path, fake_requests):
    """A download with no timeout hangs forever on a half-open connection.

    Args:
        tmp_path: pytest temporary directory.
        fake_requests: (calls, holder) from the fake `requests.get` fixture.
    """
    calls, holder = fake_requests
    holder["response"] = _FakeResponse([b"abc"])

    dest = tmp_path / "model.bin"
    kokoro_module._kokoro_stream_download(
        "https://example.invalid/model", str(dest), label="test"
    )

    assert calls, "requests.get was never called"
    timeout = calls[0].get("timeout")
    assert timeout is not None, "the download carried no timeout at all"
    connect, read = timeout
    assert connect > 0 and read > 0, timeout
    assert dest.read_bytes() == b"abc"


def test_interrupted_download_leaves_no_file_the_next_run_would_trust(
    tmp_path, fake_requests
):
    """The load-bearing one: a truncated file must not survive.

    `_download_model_if_needed` gates purely on `os.path.exists`, so a partial
    file left behind is indistinguishable from a finished model and the app
    silently loads a corrupt one.
    """
    calls, holder = fake_requests
    holder["response"] = _FakeResponse([b"aaaa", b"bbbb"], raise_on_chunk=1)

    dest = tmp_path / "model.bin"
    with pytest.raises(OSError):
        kokoro_module._kokoro_stream_download(
            "https://example.invalid/model", str(dest), label="test"
        )

    assert not dest.exists(), "a partial download was left at the final path"
    # Scoped to this download's own artefacts: the shared tmp_path may hold
    # unrelated fixture directories.
    strays = [
        item.name
        for item in tmp_path.iterdir()
        if item.name.startswith(dest.name)
    ]
    assert strays == [], f"partial artefacts left behind: {strays}"


def test_download_is_atomic_final_path_appears_only_when_complete(
    tmp_path, fake_requests
):
    calls, holder = fake_requests
    dest = tmp_path / "model.bin"
    seen_during: list[bool] = []

    class _WatchingResponse(_FakeResponse):
        def iter_content(self, chunk_size=8192):
            for chunk in self._chunks:
                # The final path must not exist yet while bytes are in flight.
                seen_during.append(dest.exists())
                yield chunk

    holder["response"] = _WatchingResponse([b"aa", b"bb", b"cc"])
    kokoro_module._kokoro_stream_download(
        "https://example.invalid/model", str(dest), label="test"
    )

    assert seen_during and not any(seen_during), (
        "the final path existed while the download was still streaming"
    )
    assert dest.read_bytes() == b"aabbcc"


def test_hasher_sees_every_byte(tmp_path, fake_requests):
    """The checksum must cover the whole body, not just the first chunk.

    Args:
        tmp_path: pytest temporary directory.
        fake_requests: (calls, holder) from the fake `requests.get` fixture.
    """
    import hashlib

    calls, holder = fake_requests
    holder["response"] = _FakeResponse([b"aa", b"bb"])
    hasher = hashlib.sha256()

    kokoro_module._kokoro_stream_download(
        "https://example.invalid/m", str(tmp_path / "m.bin"),
        label="test", hasher=hasher,
    )
    assert hasher.hexdigest() == hashlib.sha256(b"aabb").hexdigest()


@pytest.mark.asyncio
async def test_model_download_does_not_run_on_the_event_loop(
    tmp_path, fake_requests, monkeypatch
):
    """A multi-hundred-MB transfer must not execute on the loop thread."""
    calls, holder = fake_requests
    threads: list[int] = []

    class _ThreadRecordingResponse(_FakeResponse):
        def iter_content(self, chunk_size=8192):
            threads.append(threading.get_ident())
            yield b"data"

    holder["response"] = _ThreadRecordingResponse([b"data"])

    backend = kokoro_module.KokoroTTSBackend.__new__(
        kokoro_module.KokoroTTSBackend
    )
    backend.model_path = str(tmp_path / "kokoro.pth")
    monkeypatch.setattr(backend, "_load_pytorch_model", lambda: None, raising=False)

    loop_thread = threading.get_ident()
    await backend._download_model_if_needed()

    assert threads, "the download never streamed; the test proved nothing"
    assert loop_thread not in threads, (
        "the Kokoro model download ran on the event loop thread; the TUI "
        "would freeze for the whole transfer"
    )
