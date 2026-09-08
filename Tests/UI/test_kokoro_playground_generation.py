"""Kokoro language controls must survive the real Studio admission path."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Select, Switch, TextArea

from Tests.TTS.adapter_fakes import FakeAdapter
from Tests.UI.speech_playground_fixtures import _resolved, _wait_until
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import TTSProviderDescriptor, TTSProviderSpec
from tldw_chatbook.TTS.legacy_catalogs import legacy_catalog
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.studio_preferences import StudioTTSPreferencesSnapshot
from tldw_chatbook.TTS.TTS_Generation import TTSService
from tldw_chatbook.UI.Speech.speech_playground_pane import SpeechPlaygroundPane
from tldw_chatbook.UI.stts_playground_catalog import (
    LOADING_SELECT_VALUE,
    UNAVAILABLE_SELECT_VALUE,
)


class _RecordingAdapter(FakeAdapter):
    """Replace only audio execution; keep the provider's real static catalog."""

    def __init__(self, provider_id):
        super().__init__(provider_id, chunks=(b"test-audio",))
        self.requests = []

    async def get_catalog(self, refresh=False):
        return legacy_catalog(self.provider_id)

    async def synthesize(self, request, progress_sink=None):
        self.requests.append(request)
        return await super().synthesize(request, progress_sink)


class _Host(App):
    def __init__(self, pane):
        super().__init__()
        self.pane = pane
        self.requests = []
        self.notices = []

    def compose(self) -> ComposeResult:
        yield self.pane

    def post_message(self, message):
        if isinstance(message, STTSPlaygroundGenerateEvent):
            self.requests.append(message.request)
            return True
        return super().post_message(message)

    def notify(self, message, *, severity="information", **kwargs):
        self.notices.append((str(message), severity))


@pytest.fixture
def lab_factory(monkeypatch):
    @asynccontextmanager
    async def mount(initial_provider="kokoro", studio=None):
        studio = studio or StudioTTSPreferencesSnapshot()
        preferences = TTSPreferencesSnapshot(
            provider_id=initial_provider,
            model_mode="exact",
            model_id="kokoro" if initial_provider == "kokoro" else "tts-1",
            voice_mode="exact",
            voice_id="af_heart" if initial_provider == "kokoro" else "alloy",
            response_format="wav",
            speed=1.0,
        )
        adapters = {
            provider: _RecordingAdapter(provider) for provider in ("kokoro", "openai")
        }
        registry = TTSAdapterRegistry(
            specs=tuple(
                TTSProviderSpec(
                    descriptor=TTSProviderDescriptor(
                        provider_id=provider, display_name=provider, native=False
                    ),
                    factory=lambda config, adapter=adapter: adapter,
                    initial_config={},
                    exclusive_reconfigure=True,
                )
                for provider, adapter in adapters.items()
            ),
            aliases={},
        )
        service = TTSService(
            registry,
            preferences_snapshot=preferences,
            studio_preferences_loader=lambda: studio,
        )
        monkeypatch.setattr(
            SpeechPlaygroundPane,
            "_tts_service_factory",
            lambda self: _resolved(service),
        )
        monkeypatch.setattr(
            SpeechPlaygroundPane, "_check_higgs_installation", lambda self: None
        )
        pane = SpeechPlaygroundPane(
            provider=initial_provider,
            studio_preferences=studio,
            global_preferences=preferences,
        )
        app = _Host(pane)
        handler = STTSEventHandler(app)
        handler._stts_service = service
        app._stts_handler = handler
        try:
            async with app.run_test(size=(150, 65)) as pilot:
                await _wait_until(
                    pilot,
                    lambda: (
                        pane.query_one("#tts-provider-select", Select).value
                        == initial_provider
                        and not pane.query_one("#tts-generate-btn", Button).disabled
                    ),
                )
                yield SimpleNamespace(
                    app=app,
                    pane=pane,
                    pilot=pilot,
                    service=service,
                    handler=handler,
                    adapters=adapters,
                )
        finally:
            await handler.cleanup_tts_resources()
            await service.close()
            await service.wait_closed()

    return mount


async def _switch(lab, provider):
    lab.pane.query_one("#tts-provider-select", Select).value = provider
    await _wait_until(
        lab.pilot,
        lambda: (
            lab.pane.provider == provider
            and not lab.pane.query_one("#tts-generate-btn", Button).disabled
        ),
    )
    await lab.pilot.pause()


async def _generate(lab):
    lab.pane.query_one("#tts-text-input", TextArea).text = "A short test reply."
    await lab.pilot.pause()
    count = len(lab.app.requests)
    lab.pane.action_generate_tts()
    assert len(lab.app.requests) == count + 1, lab.app.notices
    await lab.handler.handle_playground_generate(
        STTSPlaygroundGenerateEvent(lab.app.requests[-1])
    )
    await lab.pilot.pause()
    artifact = lab.handler._current_playground_artifact
    assert artifact is not None, lab.app.notices
    assert artifact.path.read_bytes() == b"test-audio"
    assert not lab.pane.query_one("#tts-generate-btn", Button).disabled
    return lab.adapters["kokoro"].requests[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_provider", ["kokoro", "openai"])
@pytest.mark.parametrize("use_onnx", [True, False])
async def test_automatic_language_generates_repeatedly_through_admission(
    lab_factory, initial_provider, use_onnx
):
    async with lab_factory(initial_provider) as lab:
        if initial_provider != "kokoro":
            await _switch(lab, "kokoro")
        language = lab.pane.query_one("#tts-language-select", Select)
        assert language.value not in {LOADING_SELECT_VALUE, UNAVAILABLE_SELECT_VALUE}
        lab.pane.query_one("#tts-kokoro-use-onnx", Switch).value = use_onnx
        for _ in range(2):
            request = await _generate(lab)
            assert "language" not in (
                request.options["_legacy_openai_request"].extra_params or {}
            )
            assert request.options["_legacy_internal_model_id"] == (
                "local_kokoro_default_onnx"
                if use_onnx
                else "local_kokoro_default_pytorch"
            )
        assert len(lab.adapters["kokoro"].requests) == 2
        if initial_provider == "kokoro":
            response = await lab.service.synthesize_default(text="An automatic reply.")
            try:
                assert (
                    b"".join([chunk async for chunk in response.byte_stream])
                    == b"test-audio"
                )
            finally:
                await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_provider", ["kokoro", "openai"])
async def test_fresh_kokoro_defaults_to_onnx_and_allows_explicit_pytorch(
    lab_factory, initial_provider
):
    async with lab_factory(initial_provider) as lab:
        if initial_provider != "kokoro":
            await _switch(lab, "kokoro")
        engine = lab.pane.query_one("#tts-kokoro-use-onnx", Switch)
        assert engine.value is True
        request = await _generate(lab)
        assert (
            request.options["_legacy_internal_model_id"] == "local_kokoro_default_onnx"
        )

        engine.value = False
        request = await _generate(lab)
        assert (
            request.options["_legacy_internal_model_id"]
            == "local_kokoro_default_pytorch"
        )


@pytest.mark.asyncio
async def test_explicit_language_code_survives_provider_round_trip(lab_factory):
    async with lab_factory() as lab:
        language = lab.pane.query_one("#tts-language-select", Select)
        assert "fr" in language._legal_values
        language.value = "fr"
        await lab.pilot.pause()
        request = await _generate(lab)
        assert (
            request.options["_legacy_openai_request"].extra_params["language"] == "fr"
        )
        await _switch(lab, "openai")
        await _switch(lab, "kokoro")
        assert lab.pane.query_one("#tts-language-select", Select).value == "fr"
        request = await _generate(lab)
        assert (
            request.options["_legacy_openai_request"].extra_params["language"] == "fr"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "placeholder", [LOADING_SELECT_VALUE, UNAVAILABLE_SELECT_VALUE]
)
async def test_transient_language_placeholder_is_omitted(lab_factory, placeholder):
    async with lab_factory() as lab:
        language = lab.pane.query_one("#tts-language-select", Select)
        language.set_options([("Transient language state", placeholder)])
        language.value = placeholder
        request = await _generate(lab)
        assert "language" not in request.options["_legacy_openai_request"].extra_params
