import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleProviderSelection,
    ConsoleRunState,
    ConsoleRunStatus,
    ConsoleStagedSource,
    ConsoleVariantSet,
    ConsoleWorkspaceContext,
)


def test_run_state_blocks_with_visible_recovery_copy():
    state = ConsoleRunState.blocked("Provider blocked: select a model")

    assert state.status is ConsoleRunStatus.BLOCKED
    assert state.visible_copy == "Provider blocked: select a model"
    assert state.is_send_allowed is True
    assert state.is_stop_allowed is False


def test_run_state_retrying_is_visible_and_not_sendable():
    state = ConsoleRunState.retrying("Retrying failed response")

    assert state.status is ConsoleRunStatus.RETRYING
    assert state.visible_copy == "Retrying failed response"
    assert state.is_send_allowed is False
    assert state.is_stop_allowed is False


def test_variant_set_selects_current_variant_for_continue():
    variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
        selected_index=1,
    )

    assert variants.current.content == "second"
    assert variants.can_go_previous is True
    assert variants.can_go_next is False


def test_variant_set_rejects_empty_contents():
    with pytest.raises(ValueError, match="at least one variant"):
        ConsoleVariantSet.from_contents(turn_id="turn-1", contents=[])


@pytest.mark.parametrize("selected_index", [-1, 2])
def test_variant_set_rejects_out_of_range_selected_index(selected_index):
    with pytest.raises(ValueError, match="selected_index"):
        ConsoleVariantSet.from_contents(
            turn_id="turn-1",
            contents=["first", "second"],
            selected_index=selected_index,
        )


def test_workspace_context_blocks_cross_workspace_sources():
    context = ConsoleWorkspaceContext(
        active_workspace_id="workspace-a",
        staged_sources=(
            ConsoleStagedSource(
                source_id="note-1",
                label="Other workspace note",
                source_type="note",
                workspace_id="workspace-b",
            ),
        ),
    )

    assert context.has_policy_blocks is True
    assert context.allowed_sources == []
    assert "Other workspace note" in context.recovery_copy
    assert "workspace-a" in context.recovery_copy


def test_provider_selection_carries_workspace_context():
    context = ConsoleWorkspaceContext(active_workspace_id="workspace-a")
    selection = ConsoleProviderSelection(
        provider="llama_cpp",
        base_url="http://127.0.0.1:9099",
        explicit_model="local-model",
        workspace_context=context,
    )

    assert selection.provider == "llama_cpp"
    assert selection.workspace_context.active_workspace_id == "workspace-a"


def test_chat_message_defaults_to_complete_status():
    message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello")

    assert message.role is ConsoleMessageRole.USER
    assert message.content == "hello"
    assert message.status == "complete"
    assert message.id
