"""TASK-34100.8: the Voice step's pure decisions.

What the step prefills from the raw ``[app_tts]`` table, when Next may write
at all (an untouched step writes nothing), what a save carries, and how a
failed sample is explained. The mounted behaviour is pinned in
``test_first_run_voice_step_sf3.py``.
"""

from __future__ import annotations

import asyncio
import io
import socket
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs

import pytest

from tldw_chatbook.UI.Wizards import first_run_voice_prefill as prefill
from tldw_chatbook.UI.Wizards import first_run_voice_status as status
from tldw_chatbook.UI.Wizards import first_run_voice_step_state as vs

pytestmark = pytest.mark.allow_network

_OFFICIAL = "https://api.openai.com/v1/audio/speech"


def _draft(**changes: object) -> vs.VoiceSetupDraft:
    values: dict[str, object] = {
        "endpoint": vs.POCKET_TTS_ENDPOINT,
        "authentication_mode": "none",
        "model_id": vs.POCKET_TTS_MODEL,
        "voice_id": vs.POCKET_TTS_VOICE,
        "response_format": "wav",
        "speed": 1.0,
        "sample_text": vs.DEFAULT_SAMPLE_TEXT,
        "use_as_default": False,
    }
    values.update(changes)
    return vs.VoiceSetupDraft(**values)  # type: ignore[arg-type]


# -- PocketTTS preset ------------------------------------------------------


def test_pocket_tts_preset_targets_the_servers_native_route() -> None:
    """voice-speech-03 root: the official server has no OpenAI route."""
    assert vs.POCKET_TTS_ENDPOINT == "http://127.0.0.1:8000/tts"
    pocket = vs.apply_voice_preset(
        _draft(endpoint=_OFFICIAL, authentication_mode="api_key"),
        vs.VOICE_PRESET_POCKET_TTS,
    )
    assert (pocket.endpoint, pocket.authentication_mode) == (
        "http://127.0.0.1:8000/tts",
        "none",
    )
    assert (pocket.model_id, pocket.voice_id, pocket.response_format) == (
        "pocket-tts",
        "alba",
        "wav",
    )


# -- prefill from the raw [app_tts] table ----------------------------------


@pytest.mark.parametrize("raw", (None, {}, {"OPENAI_ORG_ID": "org"}))
def test_nothing_saved_prefills_no_voice(raw) -> None:
    assert prefill.saved_voice_from_config(raw) is None


def test_saved_official_openai_voice_prefills_the_openai_preset() -> None:
    saved = prefill.saved_voice_from_config(
        {
            "OPENAI_BASE_URL": _OFFICIAL,
            "OPENAI_AUTH_MODE": "api_key",
            "default_provider": "openai",
            "default_model": "tts-1-hd",
            "default_voice": "shimmer",
            "default_format": "mp3",
            "default_speed": 1.0,
        }
    )

    assert saved is not None
    assert saved.preset == vs.VOICE_PRESET_OFFICIAL_OPENAI
    assert saved.draft.endpoint == _OFFICIAL
    assert saved.draft.authentication_mode == "api_key"
    assert (saved.draft.model_id, saved.draft.voice_id) == ("tts-1-hd", "shimmer")
    assert saved.draft.use_as_default is True
    assert prefill.current_voice_copy(saved) == (
        "Current voice: OpenAI · tts-1-hd · shimmer — unchanged unless you edit it."
    )


def test_a_default_provider_without_an_endpoint_is_the_official_openai_voice() -> None:
    saved = prefill.saved_voice_from_config(
        {"default_provider": "openai", "default_voice": "nova"}
    )

    assert saved is not None
    assert saved.preset == vs.VOICE_PRESET_OFFICIAL_OPENAI
    assert saved.draft.voice_id == "nova"
    assert saved.draft.model_id == "tts-1-hd"  # the runtime's own fallback


def test_saved_custom_endpoint_prefills_custom_with_its_axes() -> None:
    saved = prefill.saved_voice_from_config(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:8880/v1/audio/speech",
            "OPENAI_AUTH_MODE": "none",
            "default_provider": "openai",
            "default_model": "kokoro",
            "default_voice": "af_bella",
            "default_format": "flac",
            "default_speed": 1.25,
        }
    )

    assert saved is not None
    assert saved.preset == vs.VOICE_PRESET_CUSTOM
    assert saved.draft.endpoint == "http://127.0.0.1:8880/v1/audio/speech"
    assert (saved.draft.model_id, saved.draft.voice_id) == ("kokoro", "af_bella")
    assert (saved.draft.response_format, saved.draft.speed) == ("flac", 1.25)


def test_saved_pocket_tts_endpoint_prefills_pocket_tts() -> None:
    saved = prefill.saved_voice_from_config(
        {"OPENAI_BASE_URL": vs.POCKET_TTS_ENDPOINT, "OPENAI_AUTH_MODE": "none"}
    )

    assert saved is not None
    assert saved.preset == vs.VOICE_PRESET_POCKET_TTS
    assert saved.draft.use_as_default is False


def test_saved_omnivoice_default_prefills_omnivoice() -> None:
    saved = prefill.saved_voice_from_config(
        {"default_provider": "omnivoice", "default_speed": 1.5}
    )

    assert saved is not None
    assert saved.preset == vs.VOICE_PRESET_OMNIVOICE
    assert saved.draft.speed == 1.5
    assert saved.draft.use_as_default is True


def test_another_providers_default_is_named_and_left_alone() -> None:
    saved = prefill.saved_voice_from_config({"default_provider": "kokoro"})

    assert saved is not None
    assert saved.preset == vs.VOICE_PRESET_NONE
    assert "kokoro" in prefill.current_voice_copy(saved)


def test_the_raw_table_is_read_from_the_comprehensive_raw_config() -> None:
    """The loaded view back-fills default_provider = openai; never trust it."""
    app_config = {
        "APP_TTS_CONFIG": {"default_provider": "openai"},
        "app_tts": {"default_provider": "openai"},
        "COMPREHENSIVE_CONFIG_RAW": {"general": {}},
    }

    assert prefill.raw_app_tts(app_config) == {}
    assert prefill.saved_voice_from_config(prefill.raw_app_tts(app_config)) is None


# -- the delta gate ------------------------------------------------------------


def test_an_untouched_draft_is_not_persisted() -> None:
    saved = prefill.saved_voice_from_config(
        {
            "OPENAI_BASE_URL": _OFFICIAL,
            "OPENAI_AUTH_MODE": "api_key",
            "default_provider": "openai",
            "default_model": "tts-1-hd",
            "default_voice": "shimmer",
            "default_format": "mp3",
        }
    )
    assert saved is not None

    assert not prefill.should_persist_voice_config(
        saved.draft, saved.draft, acted_this_run=False
    )
    # The sample text never persists, so editing it is not an edit.
    assert not prefill.should_persist_voice_config(
        vs.replace_draft(saved.draft, sample_text="Other words."),
        saved.draft,
        acted_this_run=False,
    )


def test_an_edit_a_test_or_a_tick_is_persisted() -> None:
    baseline = _draft()

    assert prefill.should_persist_voice_config(
        _draft(voice_id="marius"), baseline, acted_this_run=False
    )
    assert prefill.should_persist_voice_config(baseline, baseline, acted_this_run=True)
    assert prefill.should_persist_voice_config(baseline, None, acted_this_run=False)


# -- what a save carries -------------------------------------------------------


def test_unticked_save_writes_no_default_selection() -> None:
    event = vs.build_voice_setup_save_event(_draft(), include_voice_axes=False)

    assert event.settings == {
        "OPENAI_BASE_URL": vs.POCKET_TTS_ENDPOINT,
        "OPENAI_AUTH_MODE": "none",
    }
    assert event.preferences is None
    assert event.persist_default_preferences is False
    assert event.commit_defaults_after_handoff is False


def test_unticked_save_on_the_reply_voice_slot_keeps_the_presets_axes() -> None:
    """A PocketTTS URL is never paired with tts-1-hd / shimmer / mp3."""
    event = vs.build_voice_setup_save_event(_draft(), include_voice_axes=True)

    assert event.settings == {
        "OPENAI_BASE_URL": vs.POCKET_TTS_ENDPOINT,
        "OPENAI_AUTH_MODE": "none",
        "default_model": "pocket-tts",
        "default_voice": "alba",
        "default_format": "wav",
        "default_speed": 1.0,
    }
    assert "default_provider" not in event.settings
    assert event.preferences is None
    assert event.persist_default_preferences is False


def test_ticked_save_makes_the_drafts_own_axes_the_default() -> None:
    event = vs.build_voice_setup_save_event(_draft(use_as_default=True))

    assert event.persist_default_preferences is True
    assert event.commit_defaults_after_handoff is True
    assert event.preferences is not None
    assert (
        event.preferences.provider_id,
        event.preferences.model_id,
        event.preferences.voice_id,
        event.preferences.response_format,
    ) == ("openai", "pocket-tts", "alba", "wav")


def test_a_staged_key_travels_on_the_same_setting_settings_uses() -> None:
    event = vs.build_voice_setup_save_event(
        _draft(endpoint=_OFFICIAL, authentication_mode="api_key", use_as_default=True),
        credential="sk-test-staged",
    )

    assert event.settings["openai_api_key"] == "sk-test-staged"


def test_a_blank_sample_does_not_block_a_save() -> None:
    event = vs.build_voice_setup_save_event(_draft(sample_text="   "))

    assert event.settings["OPENAI_BASE_URL"] == vs.POCKET_TTS_ENDPOINT


# -- labels --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "label"),
    (
        (
            {
                "default_provider": "openai",
                "OPENAI_BASE_URL": _OFFICIAL,
                "default_model": "tts-1-hd",
                "default_voice": "shimmer",
            },
            "OpenAI · tts-1-hd · shimmer",
        ),
        (
            {
                "default_provider": "openai",
                "OPENAI_BASE_URL": vs.POCKET_TTS_ENDPOINT,
                "default_model": "pocket-tts",
                "default_voice": "alba",
            },
            "PocketTTS · pocket-tts · alba",
        ),
        ({"default_provider": "omnivoice"}, "OmniVoice"),
    ),
)
def test_voice_label_names_service_model_and_voice(raw, label) -> None:
    saved = prefill.saved_voice_from_config(raw)
    assert saved is not None
    assert prefill.voice_label(saved) == label


# -- the sample request and its failures -------------------------------------


def _streamed_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24_000)
        audio.writeframes(b"\x01\x00" * 2400)
    body = bytearray(output.getvalue())
    body[4:8] = (0x77359424).to_bytes(4, "little")
    body[40:44] = (0x77359400).to_bytes(4, "little")
    return bytes(body)


class _Server:
    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self.server = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, *_exc) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


def _handler(status_code: int, body: bytes, content_type: str, seen: list):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            seen.append((self.path, self.headers.get("Content-Type", ""), raw))
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    return Handler


@pytest.mark.asyncio
async def test_pocket_tts_sample_sends_the_native_form_and_plays() -> None:
    seen: list = []
    with _Server(_handler(200, _streamed_wav(), "audio/wav", seen)) as origin:
        result = await vs.run_voice_sample(_draft(endpoint=f"{origin}/tts"))

    assert result.playable is True
    [(path, content_type, raw)] = seen
    assert path == "/tts"
    assert content_type.startswith("application/x-www-form-urlencoded")
    assert {k: v[0] for k, v in parse_qs(raw.decode()).items()} == {
        "text": vs.DEFAULT_SAMPLE_TEXT,
        "voice_url": "alba",
    }


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.asyncio
async def test_a_closed_local_port_is_classified_as_not_running() -> None:
    port = _closed_port()

    with pytest.raises(vs.VoiceSampleError) as caught:
        await vs.run_voice_sample(_draft(endpoint=f"http://127.0.0.1:{port}/tts"))

    assert caught.value.kind == "not_running"
    copy = status.voice_test_failure_copy(
        caught.value, preset=vs.VOICE_PRESET_POCKET_TTS
    )
    assert copy.startswith("Test failed — ")
    assert f"isn't running at 127.0.0.1:{port}" in copy
    assert "PocketTTS" in copy


@pytest.mark.parametrize(
    ("status_code", "body", "content_type", "kind", "words"),
    (
        (401, b"{}", "application/json", "key_rejected", "rejected the API key"),
        (403, b"{}", "application/json", "key_rejected", "rejected the API key"),
        (404, b"{}", "application/json", "no_endpoint", "no speech endpoint"),
        (200, b"<html>hi</html>", "text/html", "not_audio", "not with audio"),
        (500, b"{}", "application/json", "http_status", "HTTP 500"),
    ),
)
@pytest.mark.asyncio
async def test_server_answers_are_classified(
    status_code, body, content_type, kind, words
) -> None:
    seen: list = []
    with _Server(_handler(status_code, body, content_type, seen)) as origin:
        with pytest.raises(vs.VoiceSampleError) as caught:
            await vs.run_voice_sample(
                _draft(endpoint=f"{origin}/v1/audio/speech", model_id="m")
            )

    assert caught.value.kind == kind
    copy = status.voice_test_failure_copy(caught.value, preset=vs.VOICE_PRESET_CUSTOM)
    assert copy.startswith("Test failed — ")
    assert words in copy


@pytest.mark.asyncio
async def test_a_silent_server_is_classified_as_a_timeout() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        with pytest.raises(vs.VoiceSampleError) as caught:
            await vs.run_voice_sample(
                _draft(endpoint=f"http://127.0.0.1:{port}/tts"), timeout_seconds=0.3
            )
    finally:
        listener.close()

    assert caught.value.kind == "timeout"
    assert "didn't answer within" in status.voice_test_failure_copy(
        caught.value, preset=vs.VOICE_PRESET_POCKET_TTS
    )


# -- the reachability probe ------------------------------------------------------


def test_probe_is_one_short_connect() -> None:
    assert status.PROBE_TIMEOUT_SECONDS < 1.0
    port = _closed_port()
    assert status.probe_endpoint_reachable(f"http://127.0.0.1:{port}/tts") is False

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        url = f"http://127.0.0.1:{listener.getsockname()[1]}/tts"
        assert status.probe_endpoint_reachable(url) is True
    finally:
        listener.close()


def test_service_status_lines_say_whether_a_service_will_work() -> None:
    assert status.service_status_copy(
        vs.VOICE_PRESET_POCKET_TTS, endpoint=vs.POCKET_TTS_ENDPOINT, reachable=False
    ) == (
        "PocketTTS — not running at 127.0.0.1:8000. It is a separate local "
        "server: start it with pocket-tts serve, or pick another service."
    )
    assert "running at 127.0.0.1:8000" in status.service_status_copy(
        vs.VOICE_PRESET_POCKET_TTS, endpoint=vs.POCKET_TTS_ENDPOINT, reachable=True
    )
    assert "key found" in status.service_status_copy(
        vs.VOICE_PRESET_OFFICIAL_OPENAI, endpoint=_OFFICIAL, key_found=True
    )
    assert "needs an OpenAI API key" in status.service_status_copy(
        vs.VOICE_PRESET_OFFICIAL_OPENAI, endpoint=_OFFICIAL, key_found=False
    )
    assert status.service_status_copy(vs.VOICE_PRESET_NONE, endpoint="") == (
        "Nothing is saved. Set up a voice any time in Settings ▸ Speech & TTS."
    )


def test_probe_runs_off_the_event_loop() -> None:
    """The probe is a blocking connect; callers must run it in a worker."""
    assert not asyncio.iscoroutinefunction(status.probe_endpoint_reachable)
