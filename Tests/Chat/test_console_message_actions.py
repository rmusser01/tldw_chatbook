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

    note = next(destination for destination in destinations if destination.label == "Note")
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

    chatbook = next(destination for destination in destinations if destination.label == "Chatbook")
    assert chatbook.available is False
    assert chatbook.reason == "Only assistant responses can be saved as Chatbook artifacts."
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
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
        selected_index=1,
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


def test_variant_action_labels_fit_compact_terminal_width_budget():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
        selected_index=1,
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
def test_feedback_actions_return_completed_result(action_id: str, expected_feedback: str):
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


def test_continue_action_targets_selected_variant_content():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="first", id="m1")
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
        selected_index=1,
    )

    result = service.dispatch("continue", message)

    assert result.status == "continue_requested"
    assert result.target_message_id == "m1"
    assert result.target_content == "second"
