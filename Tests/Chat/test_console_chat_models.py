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


def test_console_provider_selection_carries_sampling_settings() -> None:
    selection = ConsoleProviderSelection(
        provider="llama_cpp",
        explicit_model="m",
        temperature=0.3,
        top_p=0.8,
        min_p=0.05,
        top_k=40,
        max_tokens=512,
        streaming=False,
    )

    assert selection.temperature == 0.3
    assert selection.streaming is False


def test_console_provider_selection_system_prompt_defaults_to_none_and_can_be_set() -> (
    None
):
    default_selection = ConsoleProviderSelection(provider="llama_cpp")
    assert default_selection.system_prompt is None

    selection = ConsoleProviderSelection(
        provider="llama_cpp",
        explicit_model="m",
        system_prompt="Session system prompt.",
    )
    assert selection.system_prompt == "Session system prompt."


def test_chat_message_defaults_to_complete_status():
    message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello")

    assert message.role is ConsoleMessageRole.USER
    assert message.content == "hello"
    assert message.status == "complete"
    assert message.id


from tldw_chatbook.Chat.console_chat_models import (  # noqa: E402
    CONSOLE_AUTO_TITLE_MAX_LENGTH,
    derive_console_session_title,
    is_default_console_session_title,
)


def test_is_default_console_session_title_matches_chat_number_pattern():
    assert is_default_console_session_title("Chat 1")
    assert is_default_console_session_title("  Chat 42  ")
    assert not is_default_console_session_title("API refactor plan")
    assert not is_default_console_session_title("Chat")
    assert not is_default_console_session_title("Chat one")
    assert not is_default_console_session_title("")


def test_derive_console_session_title_collapses_whitespace():
    assert (
        derive_console_session_title("  fix   the\nlogin  bug ") == "fix the login bug"
    )


def test_derive_console_session_title_truncates_long_drafts():
    draft = "please review the workspace registry service for thread safety"
    title = derive_console_session_title(draft)
    assert len(title) <= CONSOLE_AUTO_TITLE_MAX_LENGTH
    assert title.endswith("...")
    assert title == "please review the workspace..."


def test_derive_console_session_title_empty_draft_returns_empty():
    assert derive_console_session_title("   \n  ") == ""


def test_derive_console_session_title_handles_max_length_below_ellipsis_width():
    assert derive_console_session_title("hello world", max_length=2) == "he"
    assert derive_console_session_title("hello world", max_length=0) == ""


def test_console_chat_message_has_parent_message_id_default_none():
    msg = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi")
    assert msg.parent_message_id is None
    assert (msg.sibling_index, msg.sibling_count) == (0, 1)

    msg2 = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="yo",
        parent_message_id="p1",
        sibling_index=1,
        sibling_count=3,
    )
    assert msg2.parent_message_id == "p1"
    assert (msg2.sibling_index, msg2.sibling_count) == (1, 3)


def test_pending_round_copy_names_the_kind_that_is_waiting():
    """Qodo #4 (task-32345): one vocabulary for "what is waiting on you".

    Approval wins any mix, because it is the only kind the Inspector's
    pending-approval count can see -- picking anything else there would put
    the chip and the Inspector back in disagreement. A lone question asks
    for an answer; every other kind, present or future, asks for a
    confirmation, so a sixth interrupt kind cannot silently inherit
    approval wording.
    """
    from tldw_chatbook.Chat.console_chat_models import (
        CONSOLE_PENDING_ROUND_DEFAULT_COPY,
        console_pending_round_copy,
    )

    assert console_pending_round_copy({"approval"}) == "Waiting for your approval"
    assert console_pending_round_copy({"question"}) == "Waiting for your answer"
    assert (
        console_pending_round_copy({"question", "approval"})
        == "Waiting for your approval"
    )
    for kind in ("skill_install", "skill_script", "worktree_merge", "invented_later"):
        assert console_pending_round_copy({kind}) == CONSOLE_PENDING_ROUND_DEFAULT_COPY
    assert (
        console_pending_round_copy({"question", "skill_script"})
        == CONSOLE_PENDING_ROUND_DEFAULT_COPY
    )
    # No kinds at all -- the pre-kind behaviour every surface had.
    assert console_pending_round_copy(()) == "Waiting for your approval"


def test_pending_round_copy_survives_a_controller_that_cannot_answer():
    """The three render surfaces reach the controller late-bound and are
    driven by partial doubles, so the kind lookup must degrade, never
    raise -- and must degrade to what they all said before kinds existed."""
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_chat_models import console_pending_round_copy_for

    class _Angry:
        def pending_round_kinds(self, session_id):
            raise RuntimeError("no registry here")

    assert (
        console_pending_round_copy_for(SimpleNamespace(), "s1")
        == "Waiting for your approval"
    )
    assert console_pending_round_copy_for(_Angry(), "s1") == "Waiting for your approval"
    assert (
        console_pending_round_copy_for(
            SimpleNamespace(pending_round_kinds=lambda sid: {"question"}), "s1"
        )
        == "Waiting for your answer"
    )
