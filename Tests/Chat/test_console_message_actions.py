import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariantSet,
)
from tldw_chatbook.Chat.console_message_actions import ConsoleMessageActionService


def test_assistant_message_actions_include_required_order():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]


def test_failed_user_row_offers_no_retry_action():
    """TASK-457(a): retry regenerates a failed ASSISTANT response; a failed USER
    row (the send-blocked optimistic echo) has nothing to regenerate, so it must
    not offer 'retry' — a failed ASSISTANT row still does."""
    service = ConsoleMessageActionService()

    failed_user = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="hello", status="failed"
    )
    assert "retry" not in [
        action.action_id for action in service.available_actions(failed_user)
    ]

    failed_assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", status="failed"
    )
    assert "retry" in [
        action.action_id for action in service.available_actions(failed_assistant)
    ]


def test_streaming_assistant_message_shows_completed_actions_disabled_with_reasons():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
        status="streaming",
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]
    assert all(action.enabled is False for action in actions)
    assert all(action.disabled_reason for action in actions)
    assert all(
        "finish" in action.disabled_reason.lower() or "WIP" in action.disabled_reason
        for action in actions
    )


def test_pending_assistant_message_shows_completed_actions_disabled_with_reasons():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        status="pending",
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]
    assert all(action.enabled is False for action in actions)
    assert all(action.disabled_reason for action in actions)


def test_unavailable_save_destinations_carry_honest_default_reason():
    service = ConsoleMessageActionService(available_save_destinations={"Chatbook"})
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    destinations = service.save_as_destinations(message)

    note = next(
        destination for destination in destinations if destination.label == "Note"
    )
    assert note.available is False
    assert note.reason == "Save as Note is not available in this session."
    assert "WIP" not in note.reason


def test_unavailable_save_destinations_use_provided_specific_reason():
    service = ConsoleMessageActionService(
        available_save_destinations={"Note", "Media", "Prompt"},
        unavailable_save_reasons={
            "Chatbook": "Only assistant responses can be saved as Chatbook artifacts.",
        },
    )
    message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="question")

    destinations = service.save_as_destinations(message)

    chatbook = next(
        destination for destination in destinations if destination.label == "Chatbook"
    )
    assert chatbook.available is False
    assert (
        chatbook.reason
        == "Only assistant responses can be saved as Chatbook artifacts."
    )
    assert [d.label for d in destinations if d.available] == ["Note", "Media", "Prompt"]


def test_available_save_destinations_have_no_reason():
    service = ConsoleMessageActionService(
        available_save_destinations={"Chatbook", "Note", "Media", "Prompt"},
    )
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    destinations = service.save_as_destinations(message)

    assert all(destination.available for destination in destinations)
    assert all(destination.reason == "" for destination in destinations)


def test_action_labels_fit_compact_terminal_width_budget():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    labels = service.plain_action_labels(message)

    assert " ".join(labels) == "Copy Edit Save as... ♻ ---> 👍 👎 🗑"
    assert len(" ".join(labels)) <= 48


def test_variant_action_labels_use_symbolic_navigation():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        id="m1",
        sibling_index=1,
        sibling_count=2,
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "<",
        ">",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]


@pytest.mark.parametrize(
    ("sibling_index", "sibling_count", "previous_enabled", "next_enabled"),
    [
        (0, 3, False, True),
        (1, 3, True, True),
        (2, 3, True, False),
    ],
)
def test_variant_nav_actions_gate_on_sibling_position(
    sibling_index: int,
    sibling_count: int,
    previous_enabled: bool,
    next_enabled: bool,
):
    """TASK-7: `<`/`>` enable state follows the sibling position, not the
    retired ``ConsoleVariantSet`` selection index."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="reply",
        id="m1",
        sibling_index=sibling_index,
        sibling_count=sibling_count,
    )

    actions = {
        action.action_id: action for action in service.available_actions(message)
    }

    assert actions["variant-previous"].enabled is previous_enabled
    assert actions["variant-next"].enabled is next_enabled


def test_variant_nav_actions_absent_for_linear_single_child_message():
    """TASK-7: gate is now ``sibling_count > 1``, not ``variants is not None``
    -- a linear (unforked) message offers no `<`/`>` at all."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="reply", id="m1"
    )

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "variant-previous" not in action_ids
    assert "variant-next" not in action_ids


def test_variant_action_labels_fit_compact_terminal_width_budget():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        id="m1",
        sibling_index=1,
        sibling_count=2,
    )

    labels = service.plain_action_labels(message)

    assert " ".join(labels) == "Copy Edit Save as... < > ♻ ---> 👍 👎 🗑"
    assert len(" ".join(labels)) <= 52


def test_failed_action_labels_include_retry_inside_terminal_width_budget():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="failed",
        status="failed",
    )

    labels = service.plain_action_labels(message)

    assert " ".join(labels) == "Copy Edit Save as... Try ♻ ---> 👍 👎 🗑"
    assert len(" ".join(labels)) <= 52


def test_copy_action_returns_clipboard_text():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("copy", message)

    assert result.status == "completed"
    assert result.clipboard_text == "answer"


def test_delete_action_returns_completed_result():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("delete", message)

    assert result.status == "completed"
    assert result.visible_copy == "Deleted message from transcript."
    assert result.target_message_id == message.id


@pytest.mark.parametrize(
    ("action_id", "expected_feedback"),
    [("feedback-up", "up"), ("feedback-down", "down")],
)
def test_feedback_actions_return_completed_result(
    action_id: str, expected_feedback: str
):
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch(action_id, message)

    assert result.status == "completed"
    assert result.visible_copy == f"Marked message feedback: {expected_feedback}."
    assert result.target_message_id == message.id
    assert result.target_content == expected_feedback


def test_edit_action_requests_modal_with_current_message_content():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("edit", message)

    assert result.status == "edit_requested"
    assert result.visible_copy == "Opened Edit Message."
    assert result.target_message_id == message.id
    assert result.target_content == "answer"


def test_unimplemented_actions_return_wip_reason():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("save-later", message)

    assert result.status == "wip"
    assert "WIP" in result.visible_copy


def test_regression_no_generation_kwargs_matches_text_sibling_gating():
    """Regression guard: callers that don't pass the new generation kwargs
    (every existing call site as of this task) must see byte-identical
    behavior to before -- pinned against a real text-sibling case."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="reply",
        id="m1",
        sibling_index=1,
        sibling_count=3,
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "<",
        ">",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]
    by_id = {action.action_id: action for action in actions}
    assert by_id["variant-previous"].enabled is True
    assert by_id["variant-next"].enabled is True
    assert "keep" not in by_id


def test_generation_variant_nav_hidden_at_count_one():
    """A single-variant generation message offers no `<`/`>`/keep at all."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    action_ids = [
        action.action_id
        for action in service.available_actions(
            message, generation_variant_count=1, generation_browsed_index=0
        )
    ]

    assert "variant-previous" not in action_ids
    assert "variant-next" not in action_ids
    assert "keep" not in action_ids


def test_generation_variant_nav_visible_at_count_two():
    """Two-plus variants show `<`/`>`, gated by the GENERATION browsed index
    -- not by the message's (absent) text-sibling fields."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message, generation_variant_count=2, generation_browsed_index=0
        )
    }

    assert "variant-previous" in actions
    assert "variant-next" in actions
    assert actions["variant-previous"].enabled is False
    assert actions["variant-next"].enabled is True


@pytest.mark.parametrize(
    ("browsed_index", "variant_count", "previous_enabled", "next_enabled"),
    [
        (0, 3, False, True),
        (1, 3, True, True),
        (2, 3, True, False),
    ],
)
def test_generation_variant_nav_boundary_enables(
    browsed_index: int,
    variant_count: int,
    previous_enabled: bool,
    next_enabled: bool,
):
    """Boundary-enable mirrors the text-sibling check exactly, but keyed off
    the generation browsed index/count instead of sibling_index/count."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message,
            generation_variant_count=variant_count,
            generation_browsed_index=browsed_index,
        )
    }

    assert actions["variant-previous"].enabled is previous_enabled
    assert actions["variant-next"].enabled is next_enabled


@pytest.mark.parametrize("browsed_index", [0, 1, 2])
def test_keep_action_only_visible_when_browsed_away_from_canonical(browsed_index: int):
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    action_ids = [
        action.action_id
        for action in service.available_actions(
            message, generation_variant_count=3, generation_browsed_index=browsed_index
        )
    ]

    assert ("keep" in action_ids) is (browsed_index != 0)


def test_generation_message_ignores_text_sibling_fields():
    """A generation message that (hypothetically) also carries stale
    text-sibling fields must still be gated by the generation kwargs --
    generation-variant gating takes precedence (spec §5.1/§7)."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] x",
        sibling_index=0,
        sibling_count=1,  # would hide <> under the old sibling-only gate
    )

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message, generation_variant_count=4, generation_browsed_index=2
        )
    }

    assert "variant-previous" in actions
    assert "variant-next" in actions
    assert actions["variant-previous"].enabled is True
    assert actions["variant-next"].enabled is True
    assert "keep" in actions


def test_generation_regenerate_stays_visible_and_enabled():
    """Regenerate (`♻`) stays visible on a generation message, still gated
    only by assistant-role as today (spec §7)."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message, generation_variant_count=3, generation_browsed_index=1
        )
    }

    assert actions["regenerate"].enabled is True


def test_keep_action_dispatch_returns_completed_result():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="[image] x", id="m1"
    )

    result = service.dispatch("keep", message)

    assert result.status == "completed"
    assert result.target_message_id == "m1"


def test_continue_action_targets_selected_variant_content():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="first", id="m1"
    )
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
        selected_index=1,
    )

    result = service.dispatch("continue", message)

    assert result.status == "continue_requested"
    assert result.target_message_id == "m1"
    assert result.target_content == "second"
