"""Task 16: one closed-state provider-history and visible-copy policy."""

from __future__ import annotations

from dataclasses import fields
from itertools import product

import pytest

from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    process_db_messages_to_ui_history,
)
from tldw_chatbook.Chat import assistant_generation_state as generation_state
from tldw_chatbook.Chat.assistant_generation_state import (
    AssistantGenerationState,
    render_exported_assistant_content,
)
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationCall,
    ContinuationRound,
    ProviderContinuationCheckpoint,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    provider_continuation_recovery_state,
)


_STATES = (
    None,
    "accepted",
    "dispatch_started",
    "continuation_active",
    "complete",
    "stopped",
    "failed",
    "discarded",
)


def _allows(state: str | None, content: str, valid: bool) -> bool:
    if valid:
        return False
    return bool(content) and state in {None, "complete", "stopped"}


_MATRIX = tuple(product(_STATES, ("", "visible"), (False, True)))


def _active_continuation() -> ProviderContinuationCheckpoint:
    return ProviderContinuationCheckpoint(
        schema_version=1,
        checkpoint_revision=1,
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
        state="active",
        rounds=(
            ContinuationRound(
                assistant_content="",
                reasoning_blocks=("private",),
                calls=(
                    ContinuationCall(
                        call_id="call-1",
                        name="calculator",
                        arguments='{"expression":"2+2"}',
                        state="pending",
                    ),
                ),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("state", "content", "valid_continuation"),
    _MATRIX,
    ids=lambda value: "none" if value is None else str(value) or "empty",
)
def test_cartesian_history_predicate_and_console_provider_builder_agree(
    state: str | None,
    content: str,
    valid_continuation: bool,
) -> None:
    """Kills any divergent local state check in the real Console builder."""
    expected = _allows(state, content, valid_continuation)
    assert (
        generation_state.assistant_state_allows_provider_history(
            state=state,
            has_valid_continuation=valid_continuation,
            content=content,
        )
        is expected
    )

    controller = ConsoleChatController(
        store=ConsoleChatStore(), provider_gateway=object()
    )
    assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        provider_continuation=(_active_continuation() if valid_continuation else None),
    )
    assistant.assistant_generation_state = state
    messages = [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="question"),
        assistant,
    ]

    payloads = controller._provider_message_payloads(messages, skip_failed=False)

    assert [item["role"] for item in payloads].count("assistant") == int(expected)


def test_active_continuation_is_sidecar_only_and_never_an_ordinary_blank_item() -> None:
    """Kills continuation_active admission through the ordinary message list."""
    store = ConsoleChatStore()
    session = store.create_session(title="sidecar", ephemeral=True)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    owner = store.append_message(
        session.id,
        role=ConsoleMessageRole.ASSISTANT,
        content="",
    )
    live_owner = store._nodes_by_session[session.id][owner.id]
    live_owner.provider_continuation = _active_continuation()
    live_owner.assistant_generation_state = "continuation_active"
    controller = ConsoleChatController(store=store, provider_gateway=object())

    payloads = controller._provider_message_payloads(
        store.messages_for_session(session.id), skip_failed=False
    )
    sidecar = controller._provider_continuation_sidecar_for_session(session.id)

    assert payloads == [{"role": "user", "content": "question"}]
    assert len(sidecar) == 1
    assert sidecar[0].owner_message_id == owner.id
    assert sidecar[0].checkpoint == _active_continuation()


def test_console_message_owns_the_portable_generation_state_field() -> None:
    """Kills dropping state while rebuilding active-path Console messages."""
    assert "assistant_generation_state" in {
        field.name for field in fields(ConsoleChatMessage)
    }


def test_legacy_continuation_surface_stays_visible_while_actions_are_disabled() -> None:
    """Kills Resume/Discard enablement before a normalized committed handle."""
    owner = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        provider_continuation=_active_continuation(),
        provider_continuation_message_version=7,
    )
    owner.provider_continuation_actions_enabled = False

    state = provider_continuation_recovery_state(owner, replay_available=True)

    assert state is not None
    assert state.actions_enabled is False


@pytest.mark.parametrize(
    ("state", "content", "expected"),
    [
        (None, "legacy", [("question", "legacy")]),
        ("accepted", "", [("question", None)]),
        ("accepted", "hidden", [("question", None)]),
        ("dispatch_started", "hidden", [("question", None)]),
        ("continuation_active", "hidden", [("question", None)]),
        ("complete", "", [("question", None)]),
        ("complete", "answer", [("question", "answer")]),
        ("stopped", "partial", [("question", "partial")]),
        ("failed", "hidden", [("question", None)]),
        ("discarded", "hidden", [("question", None)]),
    ],
)
def test_legacy_character_history_uses_the_shared_closed_state_policy(
    state: str | None,
    content: str,
    expected: list[tuple[str | None, str | None]],
) -> None:
    """Kills the legacy Character Chat helper's unconditional assistant append."""
    messages = [
        {"sender": "User", "role": "user", "content": "question"},
        {
            "sender": "Character",
            "role": "assistant",
            "content": content,
            "assistant_generation_state": state,
        },
    ]

    assert (
        process_db_messages_to_ui_history(
            messages,
            "Character",
            "User",
        )
        == expected
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            AssistantGenerationState.ACCEPTED,
            "Response accepted on another device; waiting for dispatch.",
        ),
        (
            AssistantGenerationState.DISPATCH_STARTED,
            "Response delivery status is unknown on the source device.",
        ),
        (AssistantGenerationState.COMPLETE, "No response was generated."),
        (AssistantGenerationState.FAILED, "Response failed."),
        (AssistantGenerationState.DISCARDED, "Response discarded."),
    ],
)
def test_empty_closed_state_exports_bounded_literal_copy(
    state: AssistantGenerationState,
    expected: str,
) -> None:
    """Kills blank text/Markdown/document projection for closed owners."""
    assert (
        render_exported_assistant_content(role="assistant", content="", state=state)
        == expected
    )


def test_stopped_partial_content_is_preserved_but_empty_stopped_stays_empty() -> None:
    """Pins established stopped partial history/export without inventing text."""
    assert (
        render_exported_assistant_content(
            role="assistant", content="partial", state="stopped"
        )
        == "partial"
    )
    assert (
        render_exported_assistant_content(role="assistant", content="", state="stopped")
        == ""
    )
