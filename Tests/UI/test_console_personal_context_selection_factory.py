"""Next Send selection metadata stays attached to one captured preview."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleContextSnapshot
from tldw_chatbook.Personal_Context.context_service import (
    ProfileContextRequest,
    ProfileContextSelectionExplanation,
    ProfileContextSelectionRow,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Widgets.Console.console_next_send_selection import (
    ConsoleNextSendSelectionResult,
)


class _Screen:
    def __init__(self) -> None:
        self.draft = "question"

    def query_one(self, *_args):
        return SimpleNamespace(draft_text=lambda: self.draft)

    def _estimate_tokens(self, _payload):
        return 2


class _Store:
    def __init__(self) -> None:
        self.active_session_id = "session-1"
        self.session = SimpleNamespace(id="session-1", workspace_id="workspace-1")
        self.workspace_context = SimpleNamespace(
            active_workspace_id="workspace-1", allowed_sources=[]
        )
        self.epoch = 1

    def sessions(self):
        return [self.session]

    def pending_attachments(self, _session_id):
        return []

    def conversation_context_epoch(self, _session_id):
        return self.epoch


class _Controller:
    def __init__(self) -> None:
        self.store = _Store()
        self.run_state = SimpleNamespace(status="idle")
        self.revision = 1
        self.configuration = SimpleNamespace(
            provider_selection=SimpleNamespace(provider="openai", model="test")
        )
        self.snapshot = ConsoleContextSnapshot(
            current_messages=[], next_send_payload={"messages": []}
        )
        self.explanation = ProfileContextSelectionExplanation(
            state="available",
            rows=(ProfileContextSelectionRow("eligible-only", "selected", 3),),
        )
        self.request = ProfileContextRequest(
            current_user_text="question",
            available_input_tokens=100,
            model="test",
            provider="openai",
            active_workspace_id="workspace-1",
        )
        self.builder = object()
        self.valid = AsyncMock(return_value=True)
        self.release: asyncio.Event | None = None

    def _lifecycle_revision_for(self, _session_id):
        return self.revision

    def resolve_turn_execution_context(self, _session_id):
        return self.configuration

    async def personal_context_selection_current(
        self, builder, request, explanation, *, provider_selection
    ):
        return await self.valid(builder, request, explanation, provider_selection)

    async def build_context_snapshot(self, **kwargs):
        kwargs["profile_selection_sink"](self.builder, self.request, self.explanation)
        if self.release is not None:
            await self.release.wait()
        return self.snapshot


@pytest.mark.asyncio
async def test_screen_factory_returns_inspector_only_selection_and_validates_owner():
    screen = _Screen()
    controller = _Controller()
    factory, *_ = ChatScreen._console_inspector_next_send_factories(
        screen, controller, "session-1"
    )

    result = await factory()

    assert isinstance(result, ConsoleNextSendSelectionResult)
    assert result.snapshot is controller.snapshot
    assert result.explanation is controller.explanation
    assert "eligible-only" not in repr(result)
    assert await result.is_current()
    controller.valid.assert_awaited_once_with(
        controller.builder,
        controller.request,
        controller.explanation,
        controller.configuration.provider_selection,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        lambda screen, controller: setattr(screen, "draft", "edited"),
        lambda screen, controller: setattr(
            controller.store, "active_session_id", "other"
        ),
        lambda screen, controller: setattr(
            controller.store.workspace_context, "active_workspace_id", "other"
        ),
        lambda screen, controller: setattr(controller.store, "epoch", 2),
        lambda screen, controller: setattr(controller, "revision", 2),
        lambda screen, controller: setattr(
            controller, "configuration", SimpleNamespace(provider_selection="other")
        ),
    ],
)
async def test_screen_factory_drops_selection_after_owner_change(change):
    screen = _Screen()
    controller = _Controller()
    factory, *_ = ChatScreen._console_inspector_next_send_factories(
        screen, controller, "session-1"
    )
    result = await factory()
    change(screen, controller)

    assert not await result.is_current()
    controller.valid.assert_not_awaited()


@pytest.mark.asyncio
async def test_screen_factory_rechecks_inputs_after_async_validation():
    screen = _Screen()
    controller = _Controller()
    started = asyncio.Event()
    release = asyncio.Event()

    async def validate(*_args):
        started.set()
        await release.wait()
        return True

    controller.valid.side_effect = validate
    factory, *_ = ChatScreen._console_inspector_next_send_factories(
        screen, controller, "session-1"
    )
    result = await factory()
    pending = asyncio.create_task(result.is_current())
    await started.wait()
    screen.draft = "edited during validation"
    release.set()

    assert not await pending


@pytest.mark.asyncio
async def test_screen_factory_drops_late_selection_before_return():
    screen = _Screen()
    controller = _Controller()
    controller.release = asyncio.Event()
    factory, *_ = ChatScreen._console_inspector_next_send_factories(
        screen, controller, "session-1"
    )
    pending = asyncio.create_task(factory())
    await asyncio.sleep(0)
    screen.draft = "edited"
    controller.release.set()

    result = await pending

    assert isinstance(result, ConsoleNextSendSelectionResult)
    assert result.explanation is None
    controller.valid.assert_not_awaited()


@pytest.mark.asyncio
async def test_screen_factory_fails_closed_when_owner_read_fails():
    screen = _Screen()
    controller = _Controller()
    factory, *_ = ChatScreen._console_inspector_next_send_factories(
        screen, controller, "session-1"
    )
    result = await factory()
    controller.store.pending_attachments = Mock(side_effect=RuntimeError("gone"))

    assert not await result.is_current()
    controller.valid.assert_not_awaited()


@pytest.mark.asyncio
async def test_screen_factory_does_not_choose_among_multiple_diagnostics():
    screen = _Screen()
    controller = _Controller()
    original = controller.build_context_snapshot

    async def duplicated(**kwargs):
        snapshot = await original(**kwargs)
        kwargs["profile_selection_sink"](
            controller.builder, controller.request, controller.explanation
        )
        return snapshot

    controller.build_context_snapshot = duplicated
    factory, *_ = ChatScreen._console_inspector_next_send_factories(
        screen, controller, "session-1"
    )

    result = await factory()

    assert result is controller.snapshot
    controller.valid.assert_not_awaited()
