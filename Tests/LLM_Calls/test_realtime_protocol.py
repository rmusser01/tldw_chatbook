"""Provider-neutral realtime session protocol (V4 task 1). See
`.superpowers/sdd/2026-08-04-realtime-voice-engine/task-1-brief.md`.

Covers the dataclasses/Protocol in `tldw_chatbook.LLM_Calls.realtime.
protocol` -- structural typing only, no network, no provider-specific
transport. `RealtimeSession` is a `typing.Protocol`; `get_protocol_members`
is 3.13+ only, so conformance is asserted via `isinstance` against a runtime-
checkable protocol using a minimal conforming stub, rather than introspecting
protocol members directly.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.LLM_Calls.realtime.protocol import (
    RealtimeCallbacks,
    RealtimeSession,
    RealtimeSessionConfig,
)


# ---------------------------------------------------------------------------
# RealtimeSessionConfig
# ---------------------------------------------------------------------------


def test_session_config_requires_api_key_and_model():
    config = RealtimeSessionConfig(api_key="sk-test", model="gpt-realtime")
    assert config.api_key == "sk-test"
    assert config.model == "gpt-realtime"


def test_session_config_defaults():
    config = RealtimeSessionConfig(api_key="sk-test", model="gpt-realtime")
    assert config.voice is None
    assert config.input_sample_rate == 24000
    assert config.output_sample_rate == 24000
    assert config.instructions is None


def test_session_config_is_frozen():
    config = RealtimeSessionConfig(api_key="sk-test", model="gpt-realtime")
    with pytest.raises(AttributeError):
        config.api_key = "sk-other"  # type: ignore[misc]


def test_session_config_accepts_all_fields():
    config = RealtimeSessionConfig(
        api_key="sk-test",
        model="gpt-realtime",
        voice="marin",
        input_sample_rate=16000,
        output_sample_rate=16000,
        instructions="Be concise.",
    )
    assert config.voice == "marin"
    assert config.input_sample_rate == 16000
    assert config.output_sample_rate == 16000
    assert config.instructions == "Be concise."


# ---------------------------------------------------------------------------
# RealtimeCallbacks
# ---------------------------------------------------------------------------


def test_callbacks_all_none_by_default():
    callbacks = RealtimeCallbacks()
    assert callbacks.on_ready is None
    assert callbacks.on_audio_delta is None
    assert callbacks.on_reply_started is None
    assert callbacks.on_first_audio is None
    assert callbacks.on_reply_done is None
    assert callbacks.on_turn_committed is None
    assert callbacks.on_input_transcript is None
    assert callbacks.on_output_transcript_delta is None
    assert callbacks.on_speech_started is None
    assert callbacks.on_usage is None
    assert callbacks.on_error is None
    assert callbacks.on_closed is None


def test_callbacks_is_mutable_and_fields_are_settable():
    # Dataclass without frozen=True -- unlike RealtimeSessionConfig, callers
    # build this incrementally (`cb = RealtimeCallbacks(); cb.on_ready =
    # ...`), so it must not be frozen.
    callbacks = RealtimeCallbacks()
    seen = []
    callbacks.on_ready = lambda: seen.append("ready")
    callbacks.on_closed = lambda reason: seen.append(reason)
    callbacks.on_ready()
    callbacks.on_closed("idle-timeout")
    assert seen == ["ready", "idle-timeout"]


def test_callbacks_accepts_all_fields_at_construction():
    calls = []
    callbacks = RealtimeCallbacks(
        on_ready=lambda: calls.append("ready"),
        on_audio_delta=lambda b: calls.append(("audio", b)),
        on_reply_started=lambda item_id: calls.append(("started", item_id)),
        on_first_audio=lambda: calls.append("first_audio"),
        on_reply_done=lambda: calls.append("done"),
        on_turn_committed=lambda: calls.append("committed"),
        on_input_transcript=lambda t: calls.append(("input", t)),
        on_output_transcript_delta=lambda t: calls.append(("output", t)),
        on_speech_started=lambda: calls.append("speech"),
        on_usage=lambda u: calls.append(("usage", u)),
        on_error=lambda e: calls.append(("error", e)),
        on_closed=lambda reason: calls.append(("closed", reason)),
    )
    callbacks.on_ready()
    callbacks.on_audio_delta(b"pcm")
    callbacks.on_reply_started("item-1")
    callbacks.on_first_audio()
    callbacks.on_reply_done()
    callbacks.on_turn_committed()
    callbacks.on_input_transcript("hello")
    callbacks.on_output_transcript_delta("hi")
    callbacks.on_speech_started()
    callbacks.on_usage({"tokens": 1})
    callbacks.on_error(RuntimeError("boom"))
    callbacks.on_closed("done")
    assert calls == [
        "ready",
        ("audio", b"pcm"),
        ("started", "item-1"),
        "first_audio",
        "done",
        "committed",
        ("input", "hello"),
        ("output", "hi"),
        "speech",
        ("usage", {"tokens": 1}),
        ("error", calls[10][1]),
        ("closed", "done"),
    ]


# ---------------------------------------------------------------------------
# RealtimeSession protocol
# ---------------------------------------------------------------------------


def test_realtime_session_is_a_runtime_checkable_protocol():
    assert getattr(RealtimeSession, "_is_protocol", False) is True
    assert getattr(RealtimeSession, "_is_runtime_protocol", False) is True


def test_minimal_conforming_stub_satisfies_the_protocol():
    class _StubSession:
        async def connect(self) -> None:
            ...

        def append_audio(self, frames: bytes) -> None:
            ...

        def send_seed(self, items, instructions):
            ...

        def send_text_item(self, text: str, *, request_response: bool) -> None:
            ...

        def cancel_response(self, played_ms: int) -> None:
            ...

        async def close(self) -> None:
            ...

    assert isinstance(_StubSession(), RealtimeSession)


def test_object_missing_methods_does_not_satisfy_the_protocol():
    class _Incomplete:
        async def connect(self) -> None:
            ...

    assert not isinstance(_Incomplete(), RealtimeSession)


# ---------------------------------------------------------------------------
# Import-lightness of the `realtime` package (brief Step 4's timing probe,
# pinned as a regression test rather than only a manual `python -c` check).
# Runs in a fresh subprocess -- importing anything else in this test module
# (or an earlier-collected test) may have already pulled `websockets` into
# `sys.modules`, which would make an in-process check meaningless.
# ---------------------------------------------------------------------------


def test_realtime_package_import_does_not_pull_in_websockets():
    """`realtime/__init__.py` must not import `websockets` (or otherwise be
    heavy) at package-import time -- only `transport`/session modules import
    it, and only when actually constructing a session.

    The brief's own manual probe (`import tldw_chatbook.LLM_Calls.realtime`
    timed from a cold interpreter) inevitably also pays for importing the
    `tldw_chatbook` and `tldw_chatbook.LLM_Calls` parent packages, which this
    task does not touch and which already cost tens of ms on their own. So
    the meaningful, task-scoped assertion here is the *incremental* cost of
    importing `.realtime` specifically, measured against that same parent
    baseline in a second cold interpreter -- plus the `websockets`-absence
    check, which is exact regardless of parent-package weight.
    """
    import subprocess
    import sys

    baseline_probe = (
        "import time\n"
        "t0 = time.monotonic()\n"
        "import tldw_chatbook.LLM_Calls\n"
        "print(time.monotonic() - t0)\n"
    )
    realtime_probe = (
        "import sys, time\n"
        "t0 = time.monotonic()\n"
        "import tldw_chatbook.LLM_Calls.realtime\n"
        "elapsed = time.monotonic() - t0\n"
        "assert 'websockets' not in sys.modules, "
        "'websockets imported at realtime package import time'\n"
        "print(elapsed)\n"
    )

    baseline_result = subprocess.run(
        [sys.executable, "-c", baseline_probe],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert baseline_result.returncode == 0, baseline_result.stdout + baseline_result.stderr
    baseline_seconds = float(baseline_result.stdout.strip().splitlines()[-1])

    realtime_result = subprocess.run(
        [sys.executable, "-c", realtime_probe],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert realtime_result.returncode == 0, realtime_result.stdout + realtime_result.stderr
    realtime_seconds = float(realtime_result.stdout.strip().splitlines()[-1])

    incremental_seconds = realtime_seconds - baseline_seconds
    assert incremental_seconds < 0.2, (
        f"realtime package added {incremental_seconds:.3f}s over the "
        f"tldw_chatbook.LLM_Calls baseline ({baseline_seconds:.3f}s) -- "
        "check for a heavy/websockets import at package level"
    )
