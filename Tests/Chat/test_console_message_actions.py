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
        "Save",
        "Regenerate",
        "Continue",
        "Feedback",
        "Delete",
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
        "Save",
        "Regenerate",
        "Continue",
        "Feedback",
        "Delete",
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
        "Save",
        "Regenerate",
        "Continue",
        "Feedback",
        "Delete",
    ]
    assert all(action.enabled is False for action in actions)
    assert all(action.disabled_reason for action in actions)


def test_unavailable_save_destinations_are_explicit_wip():
    service = ConsoleMessageActionService(available_save_destinations={"Chatbook"})
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    destinations = service.save_as_destinations(message)

    note = next(destination for destination in destinations if destination.label == "Note")
    assert note.available is False
    assert "WIP" in note.reason


def test_copy_action_returns_clipboard_text():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("copy", message)

    assert result.status == "completed"
    assert result.clipboard_text == "answer"


def test_unimplemented_actions_return_wip_reason():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("delete", message)

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
