"""Focused Console lifecycle tests for the ComfyUI H3 image-edit command."""

from __future__ import annotations

import asyncio
import builtins
from dataclasses import fields
from io import BytesIO
from types import SimpleNamespace
import threading

import pytest
from PIL import Image as PILImage

import tldw_chatbook.UI.Console_Modules.video as console_video_module
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_models import GenerationVariantMeta
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_command_grammar import CommandParse
from tldw_chatbook.Chat.console_generate_image import BatchResult
from tldw_chatbook.Chat.console_image_edit_operations import (
    ImageEditCompletion,
    ImageEditFailureNotice,
    ImageEditOperationRegistry,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Image_Generation.exceptions import ImageGenerationCancelled
from tldw_chatbook.UI.Console_Modules import image as image_module

from Tests.Chat.test_console_generation_actions import (
    _FakeComposer,
    _bare_generation_screen,
)


def _png_bytes(size: tuple[int, int] = (11, 7)) -> bytes:
    buffer = BytesIO()
    PILImage.new("RGB", size, (20, 40, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


def _pending(
    *,
    attachment_id: str = "source-attachment",
    data: bytes | None = None,
    file_type: str = "image",
) -> PendingAttachment:
    content = _png_bytes() if data is None else data
    return PendingAttachment(
        file_path="/private/sentinel/source-name.png",
        display_name="sentinel-source-name.png",
        file_type=file_type,
        insert_mode="attachment",
        data=content,
        mime_type="image/png" if file_type == "image" else "text/plain",
        original_size=len(content),
        processed_size=len(content),
        attachment_id=attachment_id,
    )


def _cfg(**overrides) -> SimpleNamespace:
    values = {
        "default_backend": "comfyui",
        "default_batch": 8,
        "max_variants_per_message": 8,
        "comfyui_image_default_seed": -1,
        "comfyui_image_default_steps": 17,
        "comfyui_image_default_sampler": "res_multistep",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _screen_with_h3_store(
    store: ConsoleChatStore,
    *,
    composer_text: str = "/generate-image :comfyui preserve  internal   spacing",
):
    screen = _bare_generation_screen(store)
    screen._session._default_console_session_settings = lambda: ConsoleSessionSettings(
        provider="openai"
    )
    composer = _FakeComposer(composer_text)
    screen._console_composer_or_none = lambda: composer
    screen.app_instance.console_image_edit_operations = ImageEditOperationRegistry()
    screen.app_instance._console_pending_attachment_stash = {}
    screen.app_instance._console_h3_image_edit_screen = screen
    screen._request_console_control_bar_sync = lambda: None
    return screen, composer


class _PrivacyLogger:
    def __init__(self) -> None:
        self.bindings: list[dict[str, str]] = []
        self.messages: list[str] = []

    def bind(self, **kwargs):
        self.bindings.append(kwargs)
        return self

    def error(self, message: str) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_operation_registry_is_app_owned_generation_checked_and_byte_free():
    registry = ImageEditOperationRegistry()
    release_first = asyncio.Event()
    first_event = threading.Event()

    async def first_runner(_generation: str) -> None:
        await release_first.wait()

    first = registry.start(
        session_id="session-a",
        attachment_id="attachment-a",
        captured_draft="draft-a",
        cancel_event=first_event,
        runner=first_runner,
    )
    assert first is not None
    assert first.cancel_event is first_event
    assert registry.active("session-a") == first
    assert (
        registry.start(
            session_id="session-a",
            attachment_id="duplicate",
            captured_draft="duplicate",
            cancel_event=threading.Event(),
            runner=first_runner,
        )
        is None
    )
    assert registry.request_cancel("session-a") == first
    assert first_event.is_set()

    release_first.set()
    await first.task
    assert registry.active("session-a") is None

    release_second = asyncio.Event()

    async def second_runner(_generation: str) -> None:
        await release_second.wait()

    second = registry.start(
        session_id="session-a",
        attachment_id="attachment-b",
        captured_draft="draft-b",
        cancel_event=threading.Event(),
        runner=second_runner,
    )
    assert second is not None
    assert not registry.remove_active("session-a", first.generation)
    assert registry.active("session-a") == second
    stale_completion = ImageEditCompletion(
        session_id="session-a",
        generation=first.generation,
        message_id="stale-persisted-message",
        attachment_id="attachment-a",
        captured_draft="draft-a",
    )
    assert not registry.publish_completion(stale_completion)
    assert registry.completion("session-a") is None

    completion = ImageEditCompletion(
        session_id="session-a",
        generation=second.generation,
        message_id="persisted-message",
        attachment_id="attachment-b",
        captured_draft="draft-b",
    )
    assert {field.name for field in fields(completion)} == {
        "session_id",
        "generation",
        "message_id",
        "attachment_id",
        "captured_draft",
    }
    with pytest.raises(TypeError):
        ImageEditCompletion(  # type: ignore[call-arg]
            session_id="session-a",
            generation=second.generation,
            message_id="persisted-message",
            attachment_id="attachment-b",
            captured_draft="draft-b",
            source_bytes=b"forbidden",
        )
    assert registry.publish_completion(completion)
    assert registry.completion("session-a") == completion
    assert not registry.ack_completion("session-a", first.generation)
    assert registry.ack_completion("session-a", second.generation)

    failure_notice = ImageEditFailureNotice(
        session_id="session-a",
        generation=second.generation,
        message_id="persisted-system-message",
    )
    assert {field.name for field in fields(failure_notice)} == {
        "session_id",
        "generation",
        "message_id",
    }
    with pytest.raises(TypeError):
        ImageEditFailureNotice(  # type: ignore[call-arg]
            session_id="session-a",
            generation=second.generation,
            message_id="persisted-system-message",
            source_path="/private/forbidden.png",
        )
    assert registry.publish_failure_notice(failure_notice)
    assert registry.failure_notice("session-a") == failure_notice
    assert not registry.ack_failure_notice("session-a", first.generation)
    assert registry.ack_failure_notice("session-a", second.generation)

    assert registry.publish_completion(completion)
    assert registry.publish_failure_notice(failure_notice)
    registry.drop_session("session-a")
    assert second.cancel_event.is_set()
    assert registry.active("session-a") is None
    assert registry.completion("session-a") is None
    assert registry.failure_notice("session-a") is None
    release_second.set()
    await second.task


@pytest.mark.asyncio
async def test_unacknowledged_completion_blocks_duplicate_after_active_task_settles():
    registry = ImageEditOperationRegistry()

    async def runner(generation: str) -> None:
        assert registry.publish_completion(
            ImageEditCompletion(
                session_id="session-a",
                generation=generation,
                message_id="persisted-message",
                attachment_id="attachment-a",
                captured_draft="draft-a",
            )
        )

    operation = registry.start(
        session_id="session-a",
        attachment_id="attachment-a",
        captured_draft="draft-a",
        cancel_event=threading.Event(),
        runner=runner,
    )
    assert operation is not None
    await operation.task
    assert registry.active("session-a") is None
    assert registry.completion("session-a") is not None
    assert (
        registry.start(
            session_id="session-a",
            attachment_id="attachment-b",
            captured_draft="draft-b",
            cancel_event=threading.Event(),
            runner=runner,
        )
        is None
    )


@pytest.mark.asyncio
async def test_h3_command_uses_raw_instruction_one_memory_image_and_count_one(
    monkeypatch,
):
    store = ConsoleChatStore()
    screen, composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    screen._console_visible_draft_session_id = session.id
    pending = _pending()
    assert store.add_pending_attachment(session.id, pending)

    generic_calls: list[object] = []

    def _generic_must_not_run(*args, **kwargs):
        generic_calls.append((args, kwargs))
        raise AssertionError("generic image preparation must not run for H3")

    monkeypatch.setattr(
        image_module, "prepare_generation_request", _generic_must_not_run
    )
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )

    started = threading.Event()
    release = threading.Event()
    batch_calls: list[dict] = []

    def _batch(**kwargs):
        batch_calls.append(kwargs)
        started.set()
        assert release.wait(2)
        meta = GenerationVariantMeta(
            prompt=kwargs["prompt"],
            negative_prompt="",
            backend="comfyui",
            model=None,
            seed=23,
            style=None,
            params={"operation": "edit", "format": "png"},
        )
        return BatchResult(successes=[(_png_bytes(), "image/png", meta)], errors=[])

    monkeypatch.setattr(image_module, "run_generation_batch", _batch)

    persistence_order: list[str] = []
    original_append = store.append_generation_message

    def _durable_append(session_id: str, **kwargs):
        message = original_append(session_id, **kwargs)
        message.persisted_message_id = "persisted-h3-message"
        persistence_order.append("persisted")
        return message

    monkeypatch.setattr(store, "append_generation_message", _durable_append)
    registry = screen.app_instance.console_image_edit_operations
    original_publish = registry.publish_completion

    def _ordered_publish(completion):
        persistence_order.append("completion")
        return original_publish(completion)

    monkeypatch.setattr(registry, "publish_completion", _ordered_publish)
    original_consume = store.consume_pending_attachment

    def _ordered_consume(*args, **kwargs):
        persistence_order.append("consume")
        return original_consume(*args, **kwargs)

    monkeypatch.setattr(store, "consume_pending_attachment", _ordered_consume)
    monkeypatch.setattr(
        console_video_module.ConsoleVideoController,
        "_ensure_console_video_store",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("H3 image edit must not access the Video store")
        ),
    )
    # `0b8e9e408 refactor: extract Console video controller` moved video
    # generation out of `chat_screen` into `UI/Console_Modules/video.py`,
    # which binds `run_video_generation` at ITS module scope. This guard
    # kept patching the old, now-absent `chat_screen` attribute, so
    # `monkeypatch.setattr` raised AttributeError instead of arming the
    # assertion -- the guard had stopped guarding anything.
    monkeypatch.setattr(
        console_video_module,
        "run_video_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("H3 image edit must not call Video generation")
        ),
    )
    real_import = builtins.__import__

    def _guard_video_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "Video_Generation" in name:
            raise AssertionError("H3 image edit must not import Video Generation")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guard_video_import)

    parse = CommandParse(
        kind="command",
        name="generate-image",
        args=":comfyui preserve  internal   spacing",
    )
    task = asyncio.create_task(screen._image._console_command_generate_image(parse))
    assert await asyncio.to_thread(started.wait, 2)
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    assert composer.draft_text().endswith("preserve  internal   spacing")
    release.set()
    await task
    await operation.task

    assert generic_calls == []
    assert len(batch_calls) == 1
    call = batch_calls[0]
    assert call["backend"] == "comfyui"
    assert call["prompt"] == "preserve  internal   spacing"
    assert call["negative_prompt"] is None
    assert call["style_name"] is None
    assert call["count"] == 1
    assert call["seed"] == -1
    assert call["steps"] == 17
    assert call["build"].keywords["sampler"] == "res_multistep"
    assert call["width"] is None and call["height"] is None
    reference = call["reference_image"]
    assert reference.file_id == pending.attachment_id
    assert reference.filename is None
    assert reference.content is pending.data
    assert reference.temp_path is None
    assert reference.mime_type == "image/png"
    assert (reference.width, reference.height) == (11, 7)
    assert call["cancel_event"] is not None
    assert store.pending_attachments(session.id) == []
    assert composer.draft_text() == ""
    messages = store.messages_for_session(session.id)
    assert len(messages) == 1
    assert messages[0].persisted_message_id == "persisted-h3-message"
    assert persistence_order == ["persisted", "completion", "consume"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("args", "pendings"),
    [
        (":comfyui", [_pending()]),
        (":comfyui @anime change it", [_pending()]),
        (":comfyui change it", []),
        (":comfyui change it", [_pending(), _pending(attachment_id="other")]),
        (":comfyui change it", [_pending(file_type="text")]),
        (":comfyui change it", [_pending(data=b"")]),
    ],
)
async def test_h3_refusals_happen_before_generic_preparation_or_generation(
    monkeypatch, args, pendings
):
    store = ConsoleChatStore()
    screen, composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    for pending in pendings:
        assert store.add_pending_attachment(session.id, pending)
    system_copy: list[str] = []

    async def _append(copy: str, *, session_id: str | None = None) -> None:
        system_copy.append(copy)

    screen._message._append_native_console_system_message = _append
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    monkeypatch.setattr(
        image_module,
        "prepare_generation_request",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("generic prep called")),
    )
    monkeypatch.setattr(
        image_module,
        "run_generation_batch",
        lambda **_k: (_ for _ in ()).throw(AssertionError("generation called")),
    )

    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=args)
    )

    assert len(system_copy) == 1
    assert "sentinel" not in system_copy[0]
    assert store.pending_attachments(session.id) == pendings
    assert composer.draft_text().endswith("preserve  internal   spacing")


@pytest.mark.asyncio
async def test_h3_source_header_read_runs_off_loop_while_pump_remains_responsive(
    monkeypatch,
):
    store = ConsoleChatStore()
    screen, _composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    store.add_pending_attachment(session.id, _pending())
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    header_started = threading.Event()
    release_header = threading.Event()
    original_header_read = screen._image._h3_reference_from_snapshot

    def _barrier_header_read(snapshot):
        header_started.set()
        assert threading.current_thread() is not threading.main_thread()
        assert release_header.wait(2)
        return original_header_read(snapshot)

    monkeypatch.setattr(
        screen._image, "_h3_reference_from_snapshot", _barrier_header_read
    )
    monkeypatch.setattr(
        image_module,
        "run_generation_batch",
        lambda **_kwargs: (_ for _ in ()).throw(ImageGenerationCancelled()),
    )

    command = asyncio.create_task(
        screen._image._console_command_generate_image(
            CommandParse(
                kind="command", name="generate-image", args=":comfyui change it"
            )
        )
    )
    assert await asyncio.to_thread(header_started.wait, 2)
    responsive = asyncio.Event()
    asyncio.get_running_loop().call_soon(responsive.set)
    await asyncio.wait_for(responsive.wait(), timeout=0.2)
    release_header.set()
    await command
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    await operation.task


@pytest.mark.asyncio
async def test_h3_oversize_source_is_rejected_before_decode_or_dispatch(monkeypatch):
    from tldw_chatbook.Image_Generation.request_validation import (
        IMAGE_GEN_REFERENCE_MAX_BYTES,
    )

    store = ConsoleChatStore()
    screen, _composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    store.add_pending_attachment(
        session.id,
        _pending(data=b"x" * (IMAGE_GEN_REFERENCE_MAX_BYTES + 1)),
    )
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    monkeypatch.setattr(
        screen._image,
        "_h3_reference_from_snapshot",
        lambda *_args: (_ for _ in ()).throw(AssertionError("decode called")),
    )
    monkeypatch.setattr(
        image_module,
        "run_generation_batch",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("dispatch called")),
    )
    copy: list[str] = []

    async def _append(message: str, *, session_id: str | None = None) -> None:
        copy.append(message)

    screen._message._append_native_console_system_message = _append
    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=":comfyui change it")
    )

    assert copy == [
        "ComfyUI image edits require one valid in-memory PNG, JPEG, or WebP image."
    ]
    assert screen.app_instance.console_image_edit_operations.active(session.id) is None


@pytest.mark.asyncio
async def test_h3_canonical_validation_performs_the_only_full_source_decode(
    monkeypatch,
):
    from tldw_chatbook.Image_Generation.request_validation import (
        validate_image_generation_request,
    )
    from PIL.PngImagePlugin import PngImageFile

    store = ConsoleChatStore()
    screen, _composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    store.add_pending_attachment(session.id, _pending())
    config = _cfg(max_width=1024, max_height=1024, max_pixels=1024 * 1024)
    monkeypatch.setattr(image_module, "get_image_generation_config", lambda: config)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    real_load = PngImageFile.load
    load_calls = 0

    def _count_load(image, *args, **kwargs):
        nonlocal load_calls
        load_calls += 1
        return real_load(image, *args, **kwargs)

    monkeypatch.setattr(PngImageFile, "load", _count_load)
    canonical_calls = 0

    def _canonical_batch(**kwargs):
        nonlocal canonical_calls
        canonical_calls += 1
        issues = validate_image_generation_request(
            {
                "backend": kwargs["backend"],
                "prompt": kwargs["prompt"],
                "width": kwargs["width"],
                "height": kwargs["height"],
                "steps": kwargs["steps"],
                "cfg_scale": kwargs["cfg_scale"],
                "reference_image": kwargs["reference_image"],
            },
            config=config,
        )
        assert issues == []
        return BatchResult(successes=[], errors=["canonical-only"])

    monkeypatch.setattr(image_module, "run_generation_batch", _canonical_batch)
    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=":comfyui change it")
    )
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    await operation.task

    assert canonical_calls == 1
    assert load_calls == 1


@pytest.mark.asyncio
async def test_h3_warning_band_is_rejected_by_canonical_ceiling_before_full_decode(
    monkeypatch,
):
    import math

    from tldw_chatbook.Image_Generation import request_validation

    store = ConsoleChatStore()
    screen, _composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    store.add_pending_attachment(session.id, _pending())
    side = math.isqrt(request_validation.PILLOW_DECOMPRESSION_WARNING_MAX_PIXELS) + 1
    config = _cfg(
        max_width=side + 1,
        max_height=side + 1,
        max_pixels=(side + 1) * (side + 1),
    )
    monkeypatch.setattr(image_module, "get_image_generation_config", lambda: config)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    open_calls = 0
    load_calls = 0

    class _WarningBandImage:
        format = "PNG"
        mode = "RGB"
        size = (side, side)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def load(self):
            nonlocal load_calls
            load_calls += 1
            raise AssertionError("warning-band image was fully decoded")

    def _open_warning_band(*_args, **_kwargs):
        nonlocal open_calls
        open_calls += 1
        return _WarningBandImage()

    monkeypatch.setattr(PILImage, "open", _open_warning_band)
    canonical_issues = []

    def _canonical_batch(**kwargs):
        canonical_issues.extend(
            request_validation.validate_image_generation_request(
                {
                    "backend": kwargs["backend"],
                    "prompt": kwargs["prompt"],
                    "width": kwargs["width"],
                    "height": kwargs["height"],
                    "steps": kwargs["steps"],
                    "cfg_scale": kwargs["cfg_scale"],
                    "reference_image": kwargs["reference_image"],
                },
                config=config,
            )
        )
        return BatchResult(successes=[], errors=["canonical-refusal"])

    monkeypatch.setattr(image_module, "run_generation_batch", _canonical_batch)
    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=":comfyui change it")
    )
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    await operation.task

    assert open_calls == 2
    assert load_calls == 0
    assert any("safe decode limits" in issue.message for issue in canonical_issues)


@pytest.mark.asyncio
async def test_stop_before_adapter_success_is_expected_and_retains_source(monkeypatch):
    store = ConsoleChatStore()
    screen, composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    pending = _pending()
    store.add_pending_attachment(session.id, pending)
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    observed: list[threading.Event] = []
    started = threading.Event()

    def _cancelled_batch(**kwargs):
        event = kwargs["cancel_event"]
        observed.append(event)
        started.set()
        assert event.wait(2)
        raise ImageGenerationCancelled()

    monkeypatch.setattr(image_module, "run_generation_batch", _cancelled_batch)
    task = asyncio.create_task(
        screen._image._console_command_generate_image(
            CommandParse(
                kind="command", name="generate-image", args=":comfyui change it"
            )
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    active = screen.app_instance.console_image_edit_operations.active(session.id)
    assert active is not None
    assert active.cancel_event is observed[0]
    screen.app_instance.console_image_edit_operations.request_cancel(session.id)
    await task
    await active.task

    assert store.messages_for_session(session.id) == []
    assert store.pending_attachments(session.id) == [pending]
    assert composer.draft_text().endswith("preserve  internal   spacing")


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ("success", "cancel"))
async def test_app_owned_task_cancellation_drains_linearized_runner_before_reraise(
    monkeypatch, winner
):
    store = ConsoleChatStore()
    screen, _composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    pending = _pending()
    store.add_pending_attachment(session.id, pending)
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    boundary_reached = threading.Event()
    thread_started = threading.Event()
    release_result = threading.Event()
    thread_settled = threading.Event()

    def _linearized_batch(**kwargs):
        try:
            thread_started.set()
            if winner == "cancel":
                assert kwargs["cancel_event"].wait(2)
            boundary_reached.set()
            assert release_result.wait(2)
            if winner == "cancel":
                raise ImageGenerationCancelled()
            return BatchResult(
                successes=[
                    (
                        _png_bytes(),
                        "image/png",
                        GenerationVariantMeta(
                            prompt=kwargs["prompt"],
                            negative_prompt="",
                            backend="comfyui",
                            model=None,
                            seed=4,
                            style=None,
                            params={"operation": "edit"},
                        ),
                    )
                ],
                errors=[],
            )
        finally:
            thread_settled.set()

    monkeypatch.setattr(image_module, "run_generation_batch", _linearized_batch)
    append_count = 0
    original_append = store.append_generation_message

    def _durable_append(session_id: str, **kwargs):
        nonlocal append_count
        append_count += 1
        message = original_append(session_id, **kwargs)
        message.persisted_message_id = "persisted-after-owned-task-cancel"
        return message

    monkeypatch.setattr(store, "append_generation_message", _durable_append)
    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=":comfyui change it")
    )
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    assert await asyncio.to_thread(thread_started.wait, 2)
    if winner == "success":
        assert await asyncio.to_thread(boundary_reached.wait, 2)

    operation.task.cancel()
    try:
        assert await asyncio.to_thread(operation.cancel_event.wait, 2)
        assert await asyncio.to_thread(boundary_reached.wait, 2)
        assert not operation.task.done()
    finally:
        operation.cancel_event.set()
        release_result.set()
        assert await asyncio.to_thread(thread_settled.wait, 2)
    with pytest.raises(asyncio.CancelledError):
        await operation.task
    assert thread_settled.is_set()
    assert screen.app_instance.console_image_edit_operations.active(session.id) is None

    if winner == "success":
        assert append_count == 1
        assert len(store.messages_for_session(session.id)) == 1
        assert store.pending_attachments(session.id) == []
    else:
        assert append_count == 0
        assert store.messages_for_session(session.id) == []
        assert store.pending_attachments(session.id) == [pending]


@pytest.mark.asyncio
async def test_terminal_generation_never_syncs_stale_origin_screen(monkeypatch):
    store = ConsoleChatStore()
    screen, _composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    store.add_pending_attachment(session.id, _pending())
    screen.app_instance._console_h3_image_edit_screen = screen
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    started = threading.Event()
    release_result = threading.Event()

    def _success_batch(**kwargs):
        started.set()
        assert release_result.wait(2)
        return BatchResult(
            successes=[
                (
                    _png_bytes(),
                    "image/png",
                    GenerationVariantMeta(
                        prompt=kwargs["prompt"],
                        negative_prompt="",
                        backend="comfyui",
                        model=None,
                        seed=4,
                        style=None,
                        params={"operation": "edit"},
                    ),
                )
            ],
            errors=[],
        )

    monkeypatch.setattr(image_module, "run_generation_batch", _success_batch)
    original_append = store.append_generation_message

    def _durable_append(session_id: str, **kwargs):
        message = original_append(session_id, **kwargs)
        message.persisted_message_id = "persisted-after-unmount"
        return message

    monkeypatch.setattr(store, "append_generation_message", _durable_append)
    caller = asyncio.create_task(
        screen._image._console_command_generate_image(
            CommandParse(
                kind="command", name="generate-image", args=":comfyui change it"
            )
        )
    )
    assert await asyncio.to_thread(started.wait, 2)
    active = screen.app_instance.console_image_edit_operations.active(session.id)
    assert active is not None
    screen._console_h3_terminal_generations = {active.generation}
    screen.app_instance._console_h3_image_edit_screen = None
    release_result.set()
    await caller
    await active.task

    screen._sync_native_console_chat_ui.assert_not_awaited()


@pytest.mark.asyncio
async def test_persistence_failure_retains_source_and_emits_sanitized_copy(monkeypatch):
    store = ConsoleChatStore()
    screen, composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    pending = _pending()
    store.add_pending_attachment(session.id, pending)
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )

    def _success_batch(**kwargs):
        return BatchResult(
            successes=[
                (
                    _png_bytes(),
                    "image/png",
                    GenerationVariantMeta(
                        prompt=kwargs["prompt"],
                        negative_prompt="",
                        backend="comfyui",
                        model=None,
                        seed=4,
                        style=None,
                        params={"operation": "edit"},
                    ),
                )
            ],
            errors=[],
        )

    monkeypatch.setattr(image_module, "run_generation_batch", _success_batch)
    monkeypatch.setattr(
        store,
        "append_generation_message",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("sentinel body /private/sentinel/source-name.png")
        ),
    )
    privacy_logger = _PrivacyLogger()
    monkeypatch.setattr(image_module, "logger", privacy_logger)
    system_copy: list[str] = []

    async def _append(copy: str, *, session_id: str | None = None) -> None:
        system_copy.append(copy)

    screen._message._append_native_console_system_message = _append

    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=":comfyui change it")
    )
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    await operation.task

    messages = store.messages_for_session(session.id)
    assert len(messages) == 1
    assert messages[0].role.value == "system"
    assert messages[0].content == (
        "The edited image could not be saved locally. The source remains staged."
    )
    assert store.pending_attachments(session.id) == [pending]
    assert composer.draft_text().endswith("preserve  internal   spacing")
    assert system_copy == []
    assert privacy_logger.bindings == [
        {
            "component": "image_edit",
            "phase": "persistence",
            "error_type": "RuntimeError",
        }
    ]
    assert "sentinel" not in repr(
        (system_copy, privacy_logger.bindings, privacy_logger.messages)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("persistence_failure_timing", "expected_attempts"),
    (("before_append", [True, False]), ("after_append", [True])),
)
async def test_failure_guidance_persistence_error_falls_back_without_masking_primary(
    monkeypatch,
    persistence_failure_timing,
    expected_attempts,
):
    store = ConsoleChatStore()
    screen, composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    pending = _pending()
    store.add_pending_attachment(session.id, pending)
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )
    monkeypatch.setattr(
        image_module,
        "run_generation_batch",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("primary sentinel response /private/source.png")
        ),
    )
    privacy_logger = _PrivacyLogger()
    monkeypatch.setattr(image_module, "logger", privacy_logger)
    real_append = store.append_message
    persist_attempts: list[bool] = []

    def _append_with_durable_failure(*args, persist=False, **kwargs):
        persist_attempts.append(persist)
        if persist:
            if persistence_failure_timing == "after_append":
                real_append(*args, persist=False, **kwargs)
            raise RuntimeError("secondary sentinel credential body")
        return real_append(*args, persist=persist, **kwargs)

    monkeypatch.setattr(store, "append_message", _append_with_durable_failure)

    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=":comfyui change it")
    )
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    await operation.task

    assert persist_attempts == expected_attempts
    messages = store.messages_for_session(session.id)
    assert len(messages) == 1
    assert messages[0].role.value == "system"
    assert messages[0].content == (
        "The image-edit operation did not complete. Please try again."
    )
    assert store.pending_attachments(session.id) == [pending]
    assert composer.draft_text().endswith("preserve  internal   spacing")
    assert (
        screen.app_instance.console_image_edit_operations.failure_notice(session.id)
        is None
    )
    assert privacy_logger.bindings == [
        {
            "component": "image_edit",
            "phase": "history_polling",
            "error_type": "RuntimeError",
        },
        {
            "component": "image_edit",
            "phase": "failure_guidance_persistence",
            "error_type": "RuntimeError",
        },
    ]
    assert "sentinel" not in repr(
        (messages, privacy_logger.bindings, privacy_logger.messages)
    )


@pytest.mark.asyncio
async def test_postcommit_consume_exception_keeps_success_and_logs_only_type(
    monkeypatch,
):
    store = ConsoleChatStore()
    screen, composer = _screen_with_h3_store(store)
    session = store.ensure_session(settings=ConsoleSessionSettings(provider="openai"))
    pending = _pending()
    store.add_pending_attachment(session.id, pending)
    monkeypatch.setattr(image_module, "get_image_generation_config", _cfg)
    monkeypatch.setattr(
        image_module,
        "list_image_models_for_catalog",
        lambda: [{"name": "comfyui", "is_configured": True}],
    )

    def _success_batch(**kwargs):
        return BatchResult(
            successes=[
                (
                    _png_bytes(),
                    "image/png",
                    GenerationVariantMeta(
                        prompt=kwargs["prompt"],
                        negative_prompt="",
                        backend="comfyui",
                        model=None,
                        seed=4,
                        style=None,
                        params={"operation": "edit"},
                    ),
                )
            ],
            errors=[],
        )

    monkeypatch.setattr(image_module, "run_generation_batch", _success_batch)
    original_append = store.append_generation_message

    def _durable_append(session_id: str, **kwargs):
        message = original_append(session_id, **kwargs)
        message.persisted_message_id = "persisted-before-consume-error"
        return message

    monkeypatch.setattr(store, "append_generation_message", _durable_append)
    monkeypatch.setattr(
        store,
        "consume_pending_attachment",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("sentinel descriptor and response body")
        ),
    )
    privacy_logger = _PrivacyLogger()
    monkeypatch.setattr(image_module, "logger", privacy_logger)
    system_copy: list[str] = []

    async def _append(copy: str, *, session_id: str | None = None) -> None:
        system_copy.append(copy)

    screen._message._append_native_console_system_message = _append

    await screen._image._console_command_generate_image(
        CommandParse(kind="command", name="generate-image", args=":comfyui change it")
    )
    operation = screen.app_instance.console_image_edit_operations.active(session.id)
    assert operation is not None
    await operation.task

    messages = store.messages_for_session(session.id)
    assert len(messages) == 1
    assert messages[0].persisted_message_id == "persisted-before-consume-error"
    assert store.pending_attachments(session.id) == [pending]
    assert composer.draft_text().endswith("preserve  internal   spacing")
    assert system_copy == []
    completion = screen.app_instance.console_image_edit_operations.completion(
        session.id
    )
    assert completion is not None
    assert privacy_logger.bindings == [
        {
            "component": "image_edit",
            "phase": "persistence",
            "error_type": "RuntimeError",
        }
    ]
    assert "sentinel" not in repr(
        (system_copy, privacy_logger.bindings, privacy_logger.messages)
    )
