"""Plain-fake contracts for Console auto-speak ownership in HandsFree."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.TTS.profile_types import CharacterRef
from tldw_chatbook.UI.Console_Modules.hands_free import ConsoleHandsFreeController


def _controller(**attributes: Any) -> ConsoleHandsFreeController:
    """Build only the dependency surface exercised by these unit tests."""
    controller = ConsoleHandsFreeController.__new__(ConsoleHandsFreeController)
    for name, value in attributes.items():
        setattr(controller, name, value)
    return controller


def test_auto_speak_dependencies_are_required_named_keyword_only_arguments() -> None:
    signature = inspect.signature(ConsoleHandsFreeController)

    for name in (
        "request_auto_speak_enabled",
        "request_auto_speak_resume",
        "request_auto_speak_retry",
        "sync_auto_speak_controls",
    ):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [None, object()])
async def test_destination_resolution_returns_none_for_unavailable_factory(
    factory: object,
) -> None:
    app = SimpleNamespace()
    if factory is not None:
        app._ensure_tts_handler = factory
    controller = _controller(app_instance=app)

    assert await controller._resolve_console_auto_speak_destination(None, None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("resolver", [None, object()])
async def test_destination_resolution_returns_none_for_unavailable_resolver(
    resolver: object,
) -> None:
    handler = SimpleNamespace()
    if resolver is not None:
        handler.resolve_console_speech_destination = resolver

    async def ensure_handler() -> object:
        return handler

    controller = _controller(
        app_instance=SimpleNamespace(_ensure_tts_handler=ensure_handler)
    )

    assert await controller._resolve_console_auto_speak_destination(None, None) is None


@pytest.mark.asyncio
async def test_destination_resolution_awaits_factory_then_forwards_exact_arguments() -> (
    None
):
    calls: list[tuple[object, ...]] = []
    expected_destination = object()
    character_ref = CharacterRef(
        source="local",
        authority_id="authority-2",
        character_id="character-7",
    )

    async def resolver(
        assistant_kind: str | None,
        received_ref: CharacterRef | None,
    ) -> object:
        calls.append(("resolver", assistant_kind, received_ref))
        return expected_destination

    async def ensure_handler() -> object:
        calls.append(("factory",))
        return SimpleNamespace(resolve_console_speech_destination=resolver)

    controller = _controller(
        app_instance=SimpleNamespace(_ensure_tts_handler=ensure_handler)
    )

    result = await controller._resolve_console_auto_speak_destination(
        "character", character_ref
    )

    assert result is expected_destination
    assert calls == [
        ("factory",),
        ("resolver", "character", character_ref),
    ]
    assert calls[1][2] is character_ref


@pytest.mark.asyncio
async def test_destination_resolution_propagates_factory_exception() -> None:
    expected = RuntimeError("factory unavailable")

    async def ensure_handler() -> object:
        raise expected

    controller = _controller(
        app_instance=SimpleNamespace(_ensure_tts_handler=ensure_handler)
    )

    with pytest.raises(RuntimeError) as raised:
        await controller._resolve_console_auto_speak_destination(None, None)

    assert raised.value is expected


@pytest.mark.asyncio
async def test_destination_resolution_contains_resolver_exception() -> None:
    async def resolver(*_args: object) -> object:
        raise RuntimeError("resolver rejected request")

    async def ensure_handler() -> object:
        return SimpleNamespace(resolve_console_speech_destination=resolver)

    controller = _controller(
        app_instance=SimpleNamespace(_ensure_tts_handler=ensure_handler)
    )

    assert await controller._resolve_console_auto_speak_destination(None, None) is None


@pytest.mark.asyncio
async def test_destination_resolution_propagates_cancellation() -> None:
    cancelled = asyncio.CancelledError()

    async def resolver(*_args: object) -> object:
        raise cancelled

    async def ensure_handler() -> object:
        return SimpleNamespace(resolve_console_speech_destination=resolver)

    controller = _controller(
        app_instance=SimpleNamespace(_ensure_tts_handler=ensure_handler)
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await controller._resolve_console_auto_speak_destination(None, None)

    assert raised.value is cancelled


def test_control_projection_forwards_values_with_false_retry_default() -> None:
    calls: list[tuple[bool, bool, bool]] = []
    controller = _controller(
        _sync_auto_speak_controls_fn=lambda *args: calls.append(args)
    )

    controller._sync_console_auto_speak_controls(True, False)
    controller._sync_console_auto_speak_controls(False, True, True)

    assert calls == [(True, False, False), (False, True, True)]


@pytest.mark.parametrize(
    ("method_name", "callback_name", "arguments", "expected"),
    [
        (
            "on_console_auto_speak_changed",
            "_request_auto_speak_enabled_fn",
            (True,),
            (True,),
        ),
        (
            "on_console_auto_speak_resume_requested",
            "_request_auto_speak_resume_fn",
            (),
            (),
        ),
        (
            "on_console_auto_speak_retry_requested",
            "_request_auto_speak_retry_fn",
            (),
            (),
        ),
    ],
)
def test_auto_speak_commands_call_the_named_dependency_once(
    method_name: str,
    callback_name: str,
    arguments: tuple[object, ...],
    expected: tuple[object, ...],
) -> None:
    calls: list[tuple[object, ...]] = []
    controller = _controller(**{callback_name: lambda *args: calls.append(args)})

    getattr(controller, method_name)(*arguments)

    assert calls == [expected]


@pytest.mark.parametrize(
    ("method_name", "callback_name", "arguments"),
    [
        (
            "on_console_auto_speak_changed",
            "_request_auto_speak_enabled_fn",
            (False,),
        ),
        (
            "on_console_auto_speak_resume_requested",
            "_request_auto_speak_resume_fn",
            (),
        ),
        (
            "on_console_auto_speak_retry_requested",
            "_request_auto_speak_retry_fn",
            (),
        ),
    ],
)
def test_auto_speak_commands_propagate_dependency_exceptions(
    method_name: str,
    callback_name: str,
    arguments: tuple[object, ...],
) -> None:
    expected = RuntimeError(method_name)

    def fail(*_args: object) -> None:
        raise expected

    controller = _controller(**{callback_name: fail})

    with pytest.raises(RuntimeError) as raised:
        getattr(controller, method_name)(*arguments)

    assert raised.value is expected
