"""Rollback correlation uses ephemeral tokens, never navigation payload text."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    LocalCharacterConversationTarget,
    ResolvedLocalCharacterKey,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_conversation_activation import (
    CharacterConversationActivationRequest,
    ConsoleActivationCommit,
    ConsoleConversationActivationCoordinator,
)
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController


@pytest.fixture
def rollback_records():
    records = []
    sink = logger.add(lambda message: records.append(message.record), level="ERROR")
    try:
        yield records
    finally:
        logger.remove(sink)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", ("coordinator", "workspace"))
async def test_activation_rollback_logs_have_safe_attempt_context(
    owner, rollback_records
):
    secret = "private-transcript-key-profile-path"
    target = LocalCharacterConversationTarget(
        ResolvedLocalCharacterKey(secret, 7), secret
    )
    request = CharacterConversationActivationRequest(target, secret, 1)
    token = SimpleNamespace(title=secret, path=secret)
    fail = AsyncMock(side_effect=RuntimeError("synthetic rollback failure"))
    if owner == "coordinator":
        coordinator = ConsoleConversationActivationCoordinator(
            capture_state=lambda: {"draft": secret},
            revalidate=lambda _request: None,
            open_target=lambda _request: ConsoleActivationCommit(False, token),
            rollback_opened_target=fail,
            restore_state=fail,
            exact_target_visible=lambda _target: False,
        )
        await coordinator.activate(request)
    else:
        controller = ConsoleWorkspaceController.__new__(ConsoleWorkspaceController)
        store = ConsoleChatStore()
        store.create_session(title=secret)
        controller._chat_store_accessor = lambda: store
        controller._open_character_conversation_activation = AsyncMock(
            return_value=ConsoleActivationCommit(False, token)
        )
        controller._rollback_character_conversation_activation = fail
        controller._restore_character_conversation_prior_session = fail
        await controller.activate_character_conversation_after_commit(request)
    assert len(rollback_records) == 2
    operation_ids = set()
    stages = set()
    for record in rollback_records:
        extra = record["extra"]
        assert type(extra.get("operation_id")) is int
        assert extra["operation_id"] > 0
        assert extra.get("target_type") == "local_character_conversation"
        operation_ids.add(extra["operation_id"])
        stages.add(extra.get("stage"))
        assert secret not in repr(extra) + record["message"]
    assert len(operation_ids) == 1
    assert stages == {"remove_owned_runtime", "restore_prior_runtime"}


@pytest.mark.asyncio
async def test_promotion_fallback_logs_only_ephemeral_screen_context(rollback_records):
    secret = "private-title-profile-path"
    caller = SimpleNamespace(
        _result_callbacks=[object()], is_running=False, name=secret
    )
    candidate = SimpleNamespace(
        _result_callbacks=[object()], is_running=False, name=secret
    )
    app = SimpleNamespace(
        _screen_stack=[object(), caller, candidate],
        _remove_promoted_screen_caller=AsyncMock(
            side_effect=RuntimeError("synthetic remove")
        ),
        _get_screen=lambda _caller: (_ for _ in ()).throw(
            RuntimeError("synthetic restore")
        ),
    )
    caller._pop_result_callback = lambda: caller._result_callbacks.pop()
    with pytest.raises(RuntimeError, match="synthetic remove"):
        await TldwCli._transfer_pushed_console_to_content(app, candidate, caller)
    assert len(rollback_records) == 2
    assert {record["extra"].get("stage") for record in rollback_records} == {
        "restore_caller",
        "cleanup_candidate",
    }
    for record in rollback_records:
        extra = record["extra"]
        assert extra.get("candidate_token") == id(candidate)
        assert extra.get("caller_token") == id(caller)
        assert secret not in repr(extra) + record["message"]
