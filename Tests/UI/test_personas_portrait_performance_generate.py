"""Real request validation and mounted draft workflow, fake external adapter only."""
# ruff: noqa: F811 -- pytest fixture re-exports are test parameters

from dataclasses import replace

import pytest

from Tests.UI.test_personas_portrait_performance import (
    expr_db,  # noqa: F401 -- fixture dependency
    personas_editor_with_saved_character,  # noqa: F401 -- fixture dependency
    png,
    saved_source,
)
from tldw_chatbook.Character_Chat.portrait_performance import FRAME_ROLES
from tldw_chatbook.Image_Generation.adapter_registry import ImageAdapterRegistry
from tldw_chatbook.Image_Generation.adapters.base import ImageGenResult
from tldw_chatbook.Widgets.Persona_Widgets.portrait_performance_editor import (
    CharacterPerformanceRequested,
)

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def image_engine(monkeypatch):
    import tldw_chatbook.Image_Generation.adapter_registry as registry_module
    import tldw_chatbook.Image_Generation.config as config_module

    cfg = replace(
        config_module.get_image_generation_config(),
        default_backend="gemini",
        enabled_backends=["gemini"],
        gemini_image_default_model="test-reference-model",
        gemini_image_api_key="test-key-not-sent",
    )
    monkeypatch.setattr(config_module, "_config_cache", cfg)
    registry = ImageAdapterRegistry()
    calls = []

    class Adapter:
        def generate(self, request):
            calls.append(request)
            data = png((len(calls) * 20, 0, 0))
            return ImageGenResult(data, "image/png", len(data))

    registry._adapters["gemini"] = Adapter()
    monkeypatch.setattr(registry_module, "_registry", registry)
    return cfg, calls, registry


@pytest.mark.asyncio
async def test_six_explicit_requests_then_one_slot_retry_without_activation(
    personas_editor_with_saved_character, image_engine, monkeypatch
):
    _, screen, db, character_id, _editor = saved_source(
        personas_editor_with_saved_character
    )
    cfg, calls, _registry = image_engine
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    disclosures = []

    async def confirm(dialog):
        disclosures.append(dialog.message)
        return True

    monkeypatch.setattr(screen.app, "push_screen_wait", confirm)
    assert await controller.handle(CharacterPerformanceRequested("generate"))
    assert len(calls) == 6
    assert controller.draft.staged_roles == FRAME_ROLES
    assert "6 image-edit requests" in disclosures[0]
    assert "test-reference-model" in disclosures[0]
    assert all(
        call.backend == "gemini" and call.model == cfg.gemini_image_default_model
        for call in calls
    )
    assert all(call.reference_image is calls[0].reference_image for call in calls)
    assert (
        db.get_character_portrait_performance(
            character_id=character_id, expression_key="neutral"
        )
        is None
    )
    before = controller.draft.frames()
    assert await controller.handle(
        CharacterPerformanceRequested("regenerate", role="small")
    )
    assert len(calls) == 7
    assert all(
        frame.png == before[index].png
        for index, frame in enumerate(controller.draft.frames())
        if frame.role != "small"
    )


@pytest.mark.asyncio
async def test_partial_failures_keep_successes_and_only_retry_the_selected_slot(
    personas_editor_with_saved_character, image_engine, monkeypatch
):
    _, screen, _db, _id, _editor = saved_source(personas_editor_with_saved_character)
    _cfg, calls, registry = image_engine

    async def confirm(_dialog):
        return True

    monkeypatch.setattr(screen.app, "push_screen_wait", confirm)
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    adapter = registry._adapters["gemini"]
    original = adapter.generate

    def failing(request):
        if len(calls) == 1:
            calls.append(request)
            raise TimeoutError("uncertain paid response")
        if len(calls) == 2:
            calls.append(request)
            return ImageGenResult(b"invalid", "image/png", 7)
        return original(request)

    monkeypatch.setattr(adapter, "generate", failing)
    assert await controller.handle(CharacterPerformanceRequested("generate"))
    assert len(calls) == 6
    assert controller.draft.staged_roles == (
        "rest",
        "blink_rest",
        "blink_small",
        "blink_open",
    )
    saved = {
        role: controller.draft.frame(role) for role in controller.draft.staged_roles
    }
    monkeypatch.setattr(adapter, "generate", original)
    assert await controller.handle(
        CharacterPerformanceRequested("regenerate", role="small")
    )
    assert len(calls) == 7
    assert all(controller.draft.frame(role) == frame for role, frame in saved.items())


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["cancel", "source", "profile"])
async def test_uncancellable_late_generation_stops_remaining_requests(
    personas_editor_with_saved_character, image_engine, monkeypatch, tmp_path, change
):
    import asyncio
    from threading import Event

    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    _, screen, db, character_id, _editor = saved_source(
        personas_editor_with_saved_character
    )
    _cfg, calls, registry = image_engine

    async def confirm(_dialog):
        return True

    monkeypatch.setattr(screen.app, "push_screen_wait", confirm)
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    entered, release = Event(), Event()
    adapter = registry._adapters["gemini"]
    original = adapter.generate

    def held(request):
        result = original(request)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(adapter, "generate", held)
    task = asyncio.create_task(
        controller.handle(CharacterPerformanceRequested("generate"))
    )
    other = None
    try:
        assert await asyncio.to_thread(entered.wait, 3)
        assert not await controller.handle(CharacterPerformanceRequested("generate"))
        if change == "cancel":
            await controller.handle(CharacterPerformanceRequested("cancel"))
            assert calls[0].cancel_event.is_set()
        elif change == "source":
            record = db.get_character_card_by_id(character_id)
            assert db.update_character_card(
                character_id, {"image": png("blue")}, record["version"]
            )
        else:
            other = CharactersRAGDB(tmp_path / "replacement.db", "other")
            screen.app_instance.chachanotes_db = other
            controller._check_context()
        release.set()
        assert not await task
        assert len(calls) == 1
        assert controller.draft is None
        assert (
            db.get_character_portrait_performance(
                character_id=character_id, expression_key="neutral"
            )
            is None
        )
    finally:
        release.set()
        await task
        screen.app_instance.chachanotes_db = db
        if other is not None:
            other.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "refusal", ["disabled", "incompatible", "unconfigured", "destination"]
)
async def test_generation_refusals_submit_nothing(
    personas_editor_with_saved_character, image_engine, monkeypatch, refusal
):
    import tldw_chatbook.Image_Generation.config as config_module

    _, screen, _db, _id, _editor = saved_source(personas_editor_with_saved_character)
    cfg, calls, _registry = image_engine
    if refusal == "disabled":
        monkeypatch.setattr(
            config_module, "_config_cache", replace(cfg, enabled_backends=[])
        )
    elif refusal == "incompatible":
        monkeypatch.setattr(
            config_module,
            "_config_cache",
            replace(
                cfg,
                default_backend="stable_diffusion_cpp",
                enabled_backends=["stable_diffusion_cpp"],
            ),
        )

    if refusal == "unconfigured":
        monkeypatch.setattr(
            config_module, "_config_cache", replace(cfg, gemini_image_api_key=None)
        )
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    async def refuse(_dialog):
        return False

    monkeypatch.setattr(screen.app, "push_screen_wait", refuse)
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    assert not await controller.handle(CharacterPerformanceRequested("generate"))
    assert calls == []
    assert controller.draft is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["missing", "refused", "changed"])
async def test_missing_tts_or_refused_destination_preserves_full_draft(
    personas_editor_with_saved_character, monkeypatch, refusal
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from Tests.Chat.test_portrait_performance_state import performance_record

    _, screen, db, character_id, _editor = saved_source(
        personas_editor_with_saved_character
    )
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    for frame in performance_record().frames:
        controller.draft.stage(frame)
    handler = SimpleNamespace(
        resolve_console_speech_destination=AsyncMock(
            return_value=SimpleNamespace(
                fingerprint="sha256:" + "a" * 64,
                provider_label="Test TTS",
                sanitized_destination="http://localhost:8888",
                charges_may_apply=False,
            )
        ),
        speak_guarded_utterance=AsyncMock(),
    )
    monkeypatch.setattr(
        screen.app_instance,
        "_ensure_tts_handler",
        AsyncMock(return_value=handler if refusal != "missing" else None),
        raising=False,
    )

    if refusal == "changed":
        first = handler.resolve_console_speech_destination.return_value
        changed = SimpleNamespace(
            **{**vars(first), "fingerprint": "sha256:" + "b" * 64}
        )
        handler.resolve_console_speech_destination.side_effect = [first, changed]

    async def refuse(_dialog):
        return refusal == "changed"

    monkeypatch.setattr(screen.app, "push_screen_wait", refuse)
    before = controller.draft.frames()
    assert not await controller.handle(CharacterPerformanceRequested("sample"))
    assert controller.draft.frames() == before
    handler.speak_guarded_utterance.assert_not_awaited()
    assert await controller.handle(CharacterPerformanceRequested("apply"))
    assert db.get_character_portrait_performance(
        character_id=character_id, expression_key="neutral"
    ).active


@pytest.mark.asyncio
async def test_guarded_sample_uses_selected_character_and_disposable_observer(
    personas_editor_with_saved_character, monkeypatch
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from Tests.Chat.test_portrait_performance_state import performance_record
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

    _, screen, db, character_id, _editor = saved_source(
        personas_editor_with_saved_character
    )
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    for frame in performance_record().frames:
        controller.draft.stage(frame)
    handler = TTSEventHandler()
    destination = SimpleNamespace(
        fingerprint="sha256:" + "a" * 64,
        provider_label="Test TTS",
        sanitized_destination="http://localhost:8888",
        charges_may_apply=False,
    )
    monkeypatch.setattr(
        screen.app_instance,
        "_ensure_tts_handler",
        AsyncMock(return_value=handler),
        raising=False,
    )
    monkeypatch.setattr(
        handler,
        "resolve_console_speech_destination",
        AsyncMock(return_value=destination),
    )
    identity = AsyncMock(return_value=object())
    monkeypatch.setattr(handler, "_resolve_speech_request_identity", identity)
    monkeypatch.setattr(
        handler,
        "_prepare_tts_text",
        AsyncMock(return_value="Hello. This is a portrait preview. Notice the pauses."),
    )
    captured = []

    async def admit(**kwargs):
        captured.append(kwargs)
        kwargs["playback_lifecycle"].report("playing")
        kwargs["playback_lifecycle"].report_terminal("stopped")

    monkeypatch.setattr(handler, "_admit_tts_generation", admit)
    cleanup = AsyncMock()
    monkeypatch.setattr(handler, "handle_tts_playback", cleanup)

    async def confirm(_dialog):
        return True

    monkeypatch.setattr(screen.app, "push_screen_wait", confirm)
    assert await controller.handle(CharacterPerformanceRequested("sample"))
    args = identity.await_args.kwargs
    assert args["assistant_kind"] == "character"
    assert args["character_ref"].character_id == str(character_id)
    assert args["character_ref"].authority_id == db.get_local_authority_id()
    assert args["text"] == "Hello. This is a portrait preview. Notice the pauses."
    speech = captured[0]["presentation"]
    assert speech.owner.character_id == character_id
    assert speech.owner.message_id is None
    assert speech.read().state == "stopped"
    assert (
        cleanup.await_args.args[0].playback_lifecycle
        is captured[0]["playback_lifecycle"]
    )
    assert (
        db.get_character_portrait_performance(
            character_id=character_id, expression_key="neutral"
        )
        is None
    )


@pytest.mark.asyncio
async def test_cancel_sample_waits_for_exact_owner_cleanup(
    personas_editor_with_saved_character, monkeypatch
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from Tests.Chat.test_portrait_performance_state import performance_record
    from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import TTSEventHandler

    _, screen, _db, _id, _editor = saved_source(personas_editor_with_saved_character)
    controller = screen._portrait_performance()
    assert await controller.handle(CharacterPerformanceRequested("prepare"))
    for frame in performance_record().frames:
        controller.draft.stage(frame)
    handler = TTSEventHandler()
    destination = SimpleNamespace(
        fingerprint="sha256:" + "a" * 64,
        provider_label="Test TTS",
        sanitized_destination="http://localhost:8888",
        charges_may_apply=False,
    )
    monkeypatch.setattr(
        screen.app_instance,
        "_ensure_tts_handler",
        AsyncMock(return_value=handler),
        raising=False,
    )
    monkeypatch.setattr(
        handler,
        "resolve_console_speech_destination",
        AsyncMock(return_value=destination),
    )
    monkeypatch.setattr(
        handler, "_resolve_speech_request_identity", AsyncMock(return_value=object())
    )
    monkeypatch.setattr(handler, "_prepare_tts_text", AsyncMock(return_value="sample"))
    entered, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    captured = []

    async def admit(**kwargs):
        captured.append(kwargs)
        kwargs["playback_lifecycle"].report("playing")
        entered.set()
        await asyncio.Event().wait()

    async def cleanup(event):
        assert event.playback_lifecycle is captured[0]["playback_lifecycle"]
        stopping.set()
        await release.wait()
        event.playback_lifecycle.report_terminal("stopped")

    monkeypatch.setattr(handler, "_admit_tts_generation", admit)
    monkeypatch.setattr(handler, "handle_tts_playback", cleanup)

    async def confirm(_dialog):
        return True

    monkeypatch.setattr(screen.app, "push_screen_wait", confirm)
    sample = asyncio.create_task(
        controller.handle(CharacterPerformanceRequested("sample"))
    )
    cancel = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        cancel = asyncio.create_task(
            controller.handle(CharacterPerformanceRequested("cancel"))
        )
        await asyncio.wait_for(stopping.wait(), 3)
        assert not cancel.done()
        assert controller.draft is None
        release.set()
        assert not await cancel
        with pytest.raises(asyncio.CancelledError):
            await sample
        assert captured[0]["presentation"].read().state == "stopped"
    finally:
        release.set()
        await asyncio.gather(
            sample, *([cancel] if cancel else []), return_exceptions=True
        )
